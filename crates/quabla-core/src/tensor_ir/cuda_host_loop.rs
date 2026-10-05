//! Host-driven CUDA execution of loop regions.
//!
//! The fused loop kernels of `cuda.rs` give each GPU thread one carry lane for
//! the whole loop, so they require a purely elementwise body. Bodies that
//! slice, concatenate, reshape across lanes or reduce (an adaptive ODE step
//! packs its state into one carry and takes norms), loop VJPs whose capture
//! gradients reduce over broadcast axes inside the body, and `while_loop`,
//! whose trip count depends on a predicate, cannot be lowered that way.
//!
//! Such a loop runs here instead. Each of its regions (body, predicate, body
//! JVP or body VJP) is compiled once, in the parent's CUDA context, to the
//! ordinary per-node device program, and the host launches that program once
//! per iteration on device-resident carry buffers. Nothing crosses to the host
//! per iteration except kernel launches, apart from the scalar `While`
//! predicate, which is read back once per iteration exactly like `Cond`.
//! Reverse passes follow the CPU scheme: the carry tape is kept on the device
//! in full for short loops and as `TensorCarryCheckpoints` blocks (one stored
//! carry per block, replayed per block in reverse) otherwise, and each capture
//! gradient is accumulated once per iteration in reverse iteration order, its
//! broadcast axes reduced inside the body VJP. The reductions themselves use
//! the device `Sum` kernels, so float32 results differ from the CPU only by
//! rounding order.
//!
//! The fused kernels stay the fast path; a node falls back to this executor
//! only when its fused lowering fails (`cuda_loop_fused_error`), and grouped
//! loop results (`Scan` targets, loop VJP targets) choose one mode per group.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

use super::super::{
    element_count, DynamicTensor, SymbolicCotangent, TensorCarryCheckpoints, TensorExecutionPlan,
    TensorForiVjpTarget, TensorIr, TensorNodeId, TensorOp, TensorScanTarget, TensorScanVjpTarget,
};
use super::{
    cuda_fori_body_is_lowerable, cuda_fori_jvp_is_lowerable, cuda_fori_jvp_tangent_names,
    cuda_fori_vjp_jvp_is_lowerable, cuda_fori_vjp_plan, cuda_scalar_predicate,
    cuda_scan_body_is_lowerable, cuda_scan_vjp_plans, cuda_value, execute_cuda_device_program,
    recycle_cuda_computed_values, CudaBackend, CudaBufferPool, CudaExecutionPlan,
    CudaExecutionState, CudaProgramRuntime,
};

/// One loop node compiled for host-driven execution.
#[derive(Clone, Debug)]
pub(super) struct CudaHostLoop {
    kind: HostLoopKind,
    /// Device programs in the order `HostLoopKind` documents for its variant.
    regions: Vec<CudaExecutionPlan>,
}

/// Static bounds and capture names of one fixed-bound loop.
#[derive(Clone, Debug)]
struct LoopNames {
    lower: usize,
    upper: usize,
    carry: String,
    index: String,
}

/// Output layout of a reverse region: the carry cotangent outputs (one, or a
/// value/tangent pair for forward-over-reverse) when the body reads its carry,
/// then one accumulated gradient per differentiable capture. Each accumulator
/// is an extra region input added to the capture gradient inside the region,
/// so the running sum stays on the device and fuses with the body VJP.
#[derive(Clone, Debug)]
struct ReverseLayout {
    cotangent: String,
    carry_gradient: bool,
    /// `(capture name, accumulator input name, capture shape)`.
    accumulators: Vec<(String, String, Vec<usize>)>,
}

#[derive(Clone, Debug)]
enum HostLoopKind {
    /// Regions: `[predicate, body]`.
    While { carry: String },
    /// Regions: `[body]`.
    Fori { names: LoopNames },
    /// Regions: `[forward]` returning `(next carry, next tangent)`.
    ForiJvp {
        names: LoopNames,
        tangents: BTreeMap<String, String>,
    },
    /// Regions: `[body, reverse]`.
    ForiVjp {
        names: LoopNames,
        reverse: ReverseLayout,
    },
    /// Regions: `[forward, reverse]`; the reverse region returns the carry
    /// cotangent, its tangent, then the accumulated gradient tangents.
    ForiVjpJvp {
        names: LoopNames,
        forward_tangents: BTreeMap<String, String>,
        reverse_tangents: BTreeMap<String, String>,
        reverse: ReverseLayout,
    },
    /// Regions: `[body]` returning `(next carry, output)`.
    Scan { names: LoopNames },
    /// Regions: `[carry body, reverse]`.
    ScanVjp {
        names: LoopNames,
        output_cotangent: String,
        reverse: ReverseLayout,
    },
}

/// Group key of a loop node whose sibling nodes share one execution.
fn loop_group_key(op: &TensorOp) -> Option<(u8, usize)> {
    match op {
        TensorOp::Scan { group, .. } => Some((0, *group)),
        TensorOp::ScanVjp { group, .. } => Some((1, *group)),
        TensorOp::ForiVjp { group, .. } => Some((2, *group)),
        TensorOp::ForiVjpJvp { group, .. } => Some((3, *group)),
        TensorOp::ScanVjpJvp { group, .. } => Some((4, *group)),
        _ => None,
    }
}

/// Nodes produced by the same loop execution as `node_id`, in node order.
pub(super) fn cuda_loop_group_members(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
) -> Vec<TensorNodeId> {
    let Some(key) = loop_group_key(&plan.nodes[node_id].op) else {
        return vec![node_id];
    };
    plan.nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| loop_group_key(&node.op) == Some(key))
        .map(|(member, _)| member)
        .collect()
}

/// Why the fused per-lane kernel cannot lower this loop node (or its group);
/// `None` for fused-lowerable loops, non-loop nodes, and `ScanVjpJvp`, which
/// has no host-driven fallback.
pub(super) fn cuda_loop_fused_error(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
) -> Option<String> {
    let members = cuda_loop_group_members(plan, node_id);
    let error = |member: TensorNodeId| -> Option<String> {
        match &plan.nodes[member].op {
            TensorOp::While { .. } => Some("while_loop has a data-dependent trip count".into()),
            TensorOp::Fori { loop_plan, .. } => cuda_fori_body_is_lowerable(loop_plan).err(),
            TensorOp::ForiJvp { loop_plan, .. } => cuda_fori_jvp_is_lowerable(loop_plan).err(),
            TensorOp::ForiVjp {
                loop_plan, target, ..
            } => cuda_fori_vjp_plan(loop_plan, target).err(),
            TensorOp::ForiVjpJvp { plan, .. } => cuda_fori_vjp_jvp_is_lowerable(plan).err(),
            TensorOp::Scan { scan_plan, .. } => cuda_scan_body_is_lowerable(scan_plan).err(),
            TensorOp::ScanVjp {
                scan_plan, target, ..
            } => cuda_scan_vjp_plans(scan_plan, target).err(),
            _ => None,
        }
    };
    members.into_iter().find_map(error)
}

/// The set of nodes that run host-driven, every member of a host-driven group
/// included; their fused kernels are not emitted.
pub(super) fn cuda_host_driven_nodes(plan: &TensorExecutionPlan) -> BTreeSet<TensorNodeId> {
    let mut nodes = BTreeSet::new();
    for node_id in 0..plan.nodes.len() {
        if !nodes.contains(&node_id) && cuda_loop_fused_error(plan, node_id).is_some() {
            nodes.extend(cuda_loop_group_members(plan, node_id));
        }
    }
    nodes
}

/// A capture name that collides with none of `taken`.
fn fresh_name(base: &str, taken: &BTreeSet<String>) -> String {
    let mut name = base.to_string();
    while taken.contains(&name) {
        name.push('_');
    }
    name
}

fn loop_names(lower: usize, upper: usize, carry: &str, index: &str) -> LoopNames {
    LoopNames {
        lower,
        upper,
        carry: carry.to_string(),
        index: index.to_string(),
    }
}

/// Adds `accumulator + gradient` outputs for every capture with a gradient.
fn push_accumulators(
    graph: &mut TensorIr,
    gradients: &BTreeMap<String, TensorNodeId>,
    captures: &BTreeMap<String, Vec<usize>>,
    taken: &mut BTreeSet<String>,
    outputs: &mut Vec<TensorNodeId>,
) -> Result<Vec<(String, String, Vec<usize>)>, String> {
    let mut accumulators = Vec::new();
    for (index, (name, shape)) in captures.iter().enumerate() {
        let Some(&gradient) = gradients.get(name) else {
            continue;
        };
        let accumulator_name =
            fresh_name(&format!("__quabla_cuda_loop_accumulator_{index}"), taken);
        taken.insert(accumulator_name.clone());
        let dtype = graph.node_dtype(gradient)?;
        let accumulator = graph.input_typed(accumulator_name.clone(), shape.clone(), dtype)?;
        outputs.push(graph.add(accumulator, gradient)?);
        accumulators.push((name.clone(), accumulator_name, shape.clone()));
    }
    Ok(accumulators)
}

/// Builds the region plans of a host-driven loop node; also the validation
/// that the fallback can lower.
fn host_loop_ir(op: &TensorOp) -> Result<(HostLoopKind, Vec<TensorExecutionPlan>), String> {
    match op {
        TensorOp::While { loop_plan, .. } => Ok((
            HostLoopKind::While {
                carry: loop_plan.carry_name().to_string(),
            },
            vec![
                loop_plan.predicate_plan().clone(),
                loop_plan.body_plan().clone(),
            ],
        )),
        TensorOp::Fori { loop_plan, .. } => Ok((
            HostLoopKind::Fori {
                names: loop_names(
                    loop_plan.lower,
                    loop_plan.upper,
                    &loop_plan.carry_name,
                    &loop_plan.index_name,
                ),
            },
            vec![loop_plan.body.plan.clone()],
        )),
        TensorOp::ForiJvp { loop_plan, .. } => {
            let tangents = cuda_fori_jvp_tangent_names(loop_plan);
            let body = &loop_plan.body.plan;
            let forward = body
                .as_ir()
                .symbolic_jvp_with_tangent_inputs(body.output_node_id, &tangents)?;
            let (forward, _) = forward
                .graph
                .compile_cpu_many(&[forward.value, forward.tangent])?;
            Ok((
                HostLoopKind::ForiJvp {
                    names: loop_names(
                        loop_plan.lower,
                        loop_plan.upper,
                        &loop_plan.carry_name,
                        &loop_plan.index_name,
                    ),
                    tangents,
                },
                vec![forward],
            ))
        }
        TensorOp::ForiVjp { loop_plan, .. } => {
            let body = &loop_plan.body.plan;
            let mut taken = loop_plan.body.captures().keys().cloned().collect();
            let cotangent = fresh_name("__quabla_cuda_loop_cotangent", &taken);
            taken.insert(cotangent.clone());
            let vjp = body.as_ir().symbolic_vjp(body.output_node_id, &cotangent)?;
            let mut graph = vjp.graph;
            let mut outputs = Vec::new();
            let carry_gradient = vjp.gradients.get(&loop_plan.carry_name).copied();
            outputs.extend(carry_gradient);
            let accumulators = push_accumulators(
                &mut graph,
                &vjp.gradients,
                loop_plan.external_captures(),
                &mut taken,
                &mut outputs,
            )?;
            let (reverse, _) = graph.compile_cpu_many(&outputs)?;
            Ok((
                HostLoopKind::ForiVjp {
                    names: loop_names(
                        loop_plan.lower,
                        loop_plan.upper,
                        &loop_plan.carry_name,
                        &loop_plan.index_name,
                    ),
                    reverse: ReverseLayout {
                        cotangent,
                        carry_gradient: carry_gradient.is_some(),
                        accumulators,
                    },
                },
                vec![body.clone(), reverse],
            ))
        }
        TensorOp::ForiVjpJvp { plan, .. } => {
            let loop_plan = &plan.loop_plan;
            let body = &loop_plan.body.plan;
            let mut forward_tangents = plan.tangent_names.clone();
            forward_tangents.remove(&plan.cotangent_name);
            let forward = body
                .as_ir()
                .symbolic_jvp_with_tangent_inputs(body.output_node_id, &forward_tangents)?;
            let (forward, _) = forward
                .graph
                .compile_cpu_many(&[forward.value, forward.tangent])?;
            let vjp = body
                .as_ir()
                .symbolic_vjp(body.output_node_id, &plan.cotangent_name)?;
            let carry_gradient = vjp.gradients.get(&loop_plan.carry_name).copied();
            let capture_gradients = loop_plan
                .external_captures()
                .keys()
                .filter_map(|name| vjp.gradients.get(name).map(|id| (name.clone(), *id)))
                .collect::<Vec<_>>();
            let differentiated = carry_gradient
                .iter()
                .copied()
                .chain(capture_gradients.iter().map(|(_, id)| *id))
                .collect::<Vec<_>>();
            let jvp = vjp
                .graph
                .symbolic_jvp_many_with_tangent_inputs(&differentiated, &plan.tangent_names)?;
            let mut graph = jvp.graph;
            let mut outputs = Vec::new();
            if carry_gradient.is_some() {
                outputs.push(jvp.values[0]);
                outputs.push(jvp.tangents[0]);
            }
            let offset = usize::from(carry_gradient.is_some());
            let tangent_gradients = capture_gradients
                .iter()
                .enumerate()
                .map(|(index, (name, _))| (name.clone(), jvp.tangents[offset + index]))
                .collect::<BTreeMap<_, _>>();
            let mut taken = graph_input_names(&graph);
            let accumulators = push_accumulators(
                &mut graph,
                &tangent_gradients,
                loop_plan.external_captures(),
                &mut taken,
                &mut outputs,
            )?;
            let (reverse, _) = graph.compile_cpu_many(&outputs)?;
            Ok((
                HostLoopKind::ForiVjpJvp {
                    names: loop_names(
                        loop_plan.lower,
                        loop_plan.upper,
                        &loop_plan.carry_name,
                        &loop_plan.index_name,
                    ),
                    forward_tangents,
                    reverse_tangents: plan.tangent_names.clone(),
                    reverse: ReverseLayout {
                        cotangent: plan.cotangent_name.clone(),
                        carry_gradient: carry_gradient.is_some(),
                        accumulators,
                    },
                },
                vec![forward, reverse],
            ))
        }
        TensorOp::Scan { scan_plan, .. } => Ok((
            HostLoopKind::Scan {
                names: loop_names(
                    scan_plan.lower,
                    scan_plan.upper,
                    &scan_plan.carry_name,
                    &scan_plan.index_name,
                ),
            },
            vec![scan_plan.body.plan.clone()],
        )),
        TensorOp::ScanVjp { scan_plan, .. } => {
            let body = &scan_plan.body.plan;
            let [carry_output, step_output] = body.output_node_ids() else {
                return Err("scan body must return a carry and an output".to_string());
            };
            let ir = body.as_ir();
            let carry_body = ir.compile_cpu(*carry_output)?;
            let mut taken = scan_plan.body.captures().keys().cloned().collect();
            let cotangent = fresh_name("__quabla_cuda_loop_cotangent", &taken);
            taken.insert(cotangent.clone());
            let output_cotangent = fresh_name("__quabla_cuda_loop_output_cotangent", &taken);
            taken.insert(output_cotangent.clone());
            let vjp = ir.symbolic_vjp_many(&[
                (*carry_output, SymbolicCotangent::Input(cotangent.clone())),
                (
                    *step_output,
                    SymbolicCotangent::Input(output_cotangent.clone()),
                ),
            ])?;
            let mut graph = vjp.graph;
            let mut outputs = Vec::new();
            let carry_gradient = vjp.gradients.get(&scan_plan.carry_name).copied();
            outputs.extend(carry_gradient);
            let accumulators = push_accumulators(
                &mut graph,
                &vjp.gradients,
                scan_plan.external_captures(),
                &mut taken,
                &mut outputs,
            )?;
            let (reverse, _) = graph.compile_cpu_many(&outputs)?;
            Ok((
                HostLoopKind::ScanVjp {
                    names: loop_names(
                        scan_plan.lower,
                        scan_plan.upper,
                        &scan_plan.carry_name,
                        &scan_plan.index_name,
                    ),
                    output_cotangent,
                    reverse: ReverseLayout {
                        cotangent,
                        carry_gradient: carry_gradient.is_some(),
                        accumulators,
                    },
                },
                vec![carry_body, reverse],
            ))
        }
        _ => Err(format!(
            "{} has no host-driven CUDA loop lowering",
            super::cuda_op_name(op)
        )),
    }
}

fn graph_input_names(graph: &TensorIr) -> BTreeSet<String> {
    graph
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            TensorOp::Input { name } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

/// Validates that every region of the host-driven fallback lowers to CUDA.
pub(super) fn validate_cuda_host_loop(op: &TensorOp) -> Result<(), (String, String)> {
    let (_, regions) = host_loop_ir(op).map_err(|error| ("loop".to_string(), error))?;
    for region in &regions {
        super::validate_cuda_plan(region)?;
    }
    Ok(())
}

impl CudaHostLoop {
    pub(super) fn compile(
        backend: &CudaBackend,
        context: &Arc<CudaContext>,
        node_id: TensorNodeId,
        op: &TensorOp,
    ) -> Result<Self, String> {
        let (kind, regions) = host_loop_ir(op)?;
        let regions = regions
            .into_iter()
            .map(|region| {
                backend
                    .compile_in_context(region, Some(context))
                    .map_err(|error| {
                        format!("CUDA loop node {node_id} region cannot lower: {error}")
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { kind, regions })
    }

    /// Runs the loop and returns the result buffer of every group member.
    pub(super) fn execute(
        &self,
        plan: &TensorExecutionPlan,
        node_id: TensorNodeId,
        values: &[Option<CudaSlice<f32>>],
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<(TensorNodeId, CudaSlice<f32>)>, String> {
        let mut host = HostRun {
            stream: stream.clone(),
            // Loop working sets are bounded by the carry tape, so a private
            // unbudgeted pool reuses every per-iteration buffer; it is freed
            // when the loop finishes.
            pool: CudaBufferPool {
                budget: usize::MAX,
                ..CudaBufferPool::default()
            },
        };
        let members = cuda_loop_group_members(plan, node_id);
        let op = &plan.nodes[node_id].op;
        let captures = loop_captures(op, values)?;
        match &self.kind {
            HostLoopKind::While { carry } => {
                let TensorOp::While { carry: initial, .. } = op else {
                    return Err(mismatch(node_id));
                };
                let mut state = host.copy(cuda_value(values, *initial)?)?;
                loop {
                    let mut bindings = captures.clone();
                    bindings.insert(carry.clone(), &state);
                    let predicate = host.run_one(&self.regions[0], &bindings)?;
                    // Host synchronization point: one scalar is read back per iteration.
                    let predicate_value = host.stream.clone_dtoh(&predicate).map_err(|error| {
                        format!("failed to read CUDA While node {node_id} predicate: {error:?}")
                    })?;
                    host.pool.recycle(predicate);
                    if !cuda_scalar_predicate(&predicate_value)? {
                        break;
                    }
                    let next = host.run_one(&self.regions[1], &bindings)?;
                    drop(bindings);
                    host.pool.recycle(std::mem::replace(&mut state, next));
                }
                Ok(vec![(node_id, state)])
            }
            HostLoopKind::Fori { names } => {
                let TensorOp::Fori { carry, .. } = op else {
                    return Err(mismatch(node_id));
                };
                let mut index = LoopIndex::new(&mut host, names)?;
                let mut state = host.copy(cuda_value(values, *carry)?)?;
                for offset in 0..names.upper - names.lower {
                    index.set(&host.stream, offset)?;
                    let mut bindings = captures.clone();
                    bindings.insert(names.carry.clone(), &state);
                    bindings.insert(names.index.clone(), &index.current);
                    let next = host.run_one(&self.regions[0], &bindings)?;
                    drop(bindings);
                    host.pool.recycle(std::mem::replace(&mut state, next));
                }
                Ok(vec![(node_id, state)])
            }
            HostLoopKind::ForiJvp { names, tangents } => {
                let TensorOp::ForiJvp {
                    carry,
                    carry_tangent,
                    tangent_captures,
                    ..
                } = op
                else {
                    return Err(mismatch(node_id));
                };
                let tangent_values = named_values(tangent_captures, values)?;
                let mut index = LoopIndex::new(&mut host, names)?;
                let mut state = vec![
                    host.copy(cuda_value(values, *carry)?)?,
                    host.copy(cuda_value(values, *carry_tangent)?)?,
                ];
                for offset in 0..names.upper - names.lower {
                    let next = forward_jvp_step(
                        &mut host,
                        &mut index,
                        &self.regions[0],
                        names,
                        tangents,
                        &captures,
                        &tangent_values,
                        offset,
                        &state,
                    )?;
                    host.recycle_all(std::mem::replace(&mut state, next));
                }
                let tangent = state.pop().ok_or_else(|| mismatch(node_id))?;
                Ok(vec![(node_id, tangent)])
            }
            HostLoopKind::ForiVjp { names, reverse } => {
                let TensorOp::ForiVjp {
                    carry,
                    output_cotangent,
                    ..
                } = op
                else {
                    return Err(mismatch(node_id));
                };
                let mut index = LoopIndex::new(&mut host, names)?;
                let body = &self.regions[0];
                let mut step = |host: &mut HostRun,
                                index: &mut LoopIndex,
                                offset: usize,
                                state: &[CudaSlice<f32>]|
                 -> Result<Vec<CudaSlice<f32>>, String> {
                    index.set(&host.stream, offset)?;
                    let mut bindings = captures.clone();
                    bindings.insert(names.carry.clone(), &state[0]);
                    bindings.insert(names.index.clone(), &index.current);
                    host.run(body, &bindings)
                };
                let initial = vec![host.copy(cuda_value(values, *carry)?)?];
                let mut tape = DeviceTape::record(
                    &mut host,
                    &mut index,
                    names.upper - names.lower,
                    initial,
                    &mut step,
                )?;
                let mut cotangent = vec![host.copy(cuda_value(values, *output_cotangent)?)?];
                let mut accumulators = host.zero_accumulators(reverse)?;
                while let Some((start, states)) =
                    tape.pop_block(&mut host, &mut index, &mut step)?
                {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        index.set(&host.stream, start + local)?;
                        let mut bindings = captures.clone();
                        bindings.insert(names.carry.clone(), &state[0]);
                        bindings.insert(names.index.clone(), &index.current);
                        bindings.insert(reverse.cotangent.clone(), &cotangent[0]);
                        let outputs =
                            host.reverse_step(&self.regions[1], bindings, reverse, &accumulators)?;
                        host.recycle_all(state);
                        (cotangent, accumulators) =
                            host.advance_reverse(outputs, reverse, 1, cotangent, accumulators)?;
                    }
                }
                let mut gradients = host.gradient_results(reverse, accumulators)?;
                gradients.insert(names.carry.clone(), cotangent.swap_remove(0));
                fori_vjp_results(&mut host, plan, &members, &names.carry, gradients)
            }
            HostLoopKind::ForiVjpJvp {
                names,
                forward_tangents,
                reverse_tangents,
                reverse,
            } => {
                let TensorOp::ForiVjpJvp {
                    carry,
                    carry_tangent,
                    output_cotangent,
                    output_cotangent_tangent,
                    tangent_captures,
                    plan: hvp,
                    ..
                } = op
                else {
                    return Err(mismatch(node_id));
                };
                let tangent_values = named_values(tangent_captures, values)?;
                let mut index = LoopIndex::new(&mut host, names)?;
                let forward = &self.regions[0];
                let mut step = |host: &mut HostRun,
                                index: &mut LoopIndex,
                                offset: usize,
                                state: &[CudaSlice<f32>]|
                 -> Result<Vec<CudaSlice<f32>>, String> {
                    forward_jvp_step(
                        host,
                        index,
                        forward,
                        names,
                        forward_tangents,
                        &captures,
                        &tangent_values,
                        offset,
                        state,
                    )
                };
                let initial = vec![
                    host.copy(cuda_value(values, *carry)?)?,
                    host.copy(cuda_value(values, *carry_tangent)?)?,
                ];
                let mut tape = DeviceTape::record(
                    &mut host,
                    &mut index,
                    names.upper - names.lower,
                    initial,
                    &mut step,
                )?;
                let mut cotangent = vec![
                    host.copy(cuda_value(values, *output_cotangent)?)?,
                    host.copy(cuda_value(values, *output_cotangent_tangent)?)?,
                ];
                let cotangent_tangent_name = reverse_tangents
                    .get(&hvp.cotangent_name)
                    .ok_or_else(|| "CUDA Fori VJP JVP has no cotangent tangent".to_string())?;
                let mut accumulators = host.zero_accumulators(reverse)?;
                while let Some((start, states)) =
                    tape.pop_block(&mut host, &mut index, &mut step)?
                {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        index.set(&host.stream, start + local)?;
                        let mut bindings = captures.clone();
                        bindings.insert(names.carry.clone(), &state[0]);
                        bindings.insert(names.index.clone(), &index.current);
                        bindings.insert(reverse.cotangent.clone(), &cotangent[0]);
                        bindings.insert(cotangent_tangent_name.clone(), &cotangent[1]);
                        for (name, tangent_name) in reverse_tangents {
                            if name == &names.carry {
                                bindings.insert(tangent_name.clone(), &state[1]);
                            } else if let Some(tangent) = tangent_values.get(name) {
                                bindings.insert(tangent_name.clone(), tangent);
                            }
                        }
                        let outputs =
                            host.reverse_step(&self.regions[1], bindings, reverse, &accumulators)?;
                        host.recycle_all(state);
                        (cotangent, accumulators) =
                            host.advance_reverse(outputs, reverse, 2, cotangent, accumulators)?;
                    }
                }
                let mut gradients = host.gradient_results(reverse, accumulators)?;
                let carry_tangent = cotangent.pop().ok_or_else(|| mismatch(node_id))?;
                host.recycle_all(cotangent);
                gradients.insert(names.carry.clone(), carry_tangent);
                fori_vjp_results(&mut host, plan, &members, &names.carry, gradients)
            }
            HostLoopKind::Scan { names } => {
                let TensorOp::Scan { carry, .. } = op else {
                    return Err(mismatch(node_id));
                };
                let mut index = LoopIndex::new(&mut host, names)?;
                let steps = names.upper - names.lower;
                let output_count = scan_output_count(op)?;
                let mut stacked = host.take(steps * output_count)?;
                let mut state = host.copy(cuda_value(values, *carry)?)?;
                for offset in 0..steps {
                    index.set(&host.stream, offset)?;
                    let mut bindings = captures.clone();
                    bindings.insert(names.carry.clone(), &state);
                    bindings.insert(names.index.clone(), &index.current);
                    let mut outputs = host.run(&self.regions[0], &bindings)?;
                    drop(bindings);
                    let output = outputs.pop().ok_or_else(|| mismatch(node_id))?;
                    let next = outputs.pop().ok_or_else(|| mismatch(node_id))?;
                    let start = offset * output_count;
                    host.stream
                        .memcpy_dtod(&output, &mut stacked.slice_mut(start..start + output_count))
                        .map_err(|error| {
                            format!("failed to store CUDA Scan node {node_id} output: {error:?}")
                        })?;
                    host.pool.recycle(output);
                    host.pool.recycle(std::mem::replace(&mut state, next));
                }
                let mut state = Some(state);
                let mut stacked = Some(stacked);
                members
                    .iter()
                    .map(|member| {
                        let TensorOp::Scan { target, .. } = &plan.nodes[*member].op else {
                            return Err(mismatch(*member));
                        };
                        let buffer = match target {
                            TensorScanTarget::Carry => state.take(),
                            TensorScanTarget::Outputs => stacked.take(),
                        };
                        buffer
                            .ok_or_else(|| format!("CUDA Scan group repeats target {target:?}"))
                            .map(|buffer| (*member, buffer))
                    })
                    .collect()
            }
            HostLoopKind::ScanVjp {
                names,
                output_cotangent,
                reverse,
            } => {
                let TensorOp::ScanVjp {
                    carry,
                    final_carry_cotangent,
                    output_cotangent: output_cotangents,
                    scan_plan,
                    ..
                } = op
                else {
                    return Err(mismatch(node_id));
                };
                let output_count = element_count(&scan_plan.body.output_shapes()[1])?;
                let output_cotangents = cuda_value(values, *output_cotangents)?;
                let mut index = LoopIndex::new(&mut host, names)?;
                let body = &self.regions[0];
                let mut step = |host: &mut HostRun,
                                index: &mut LoopIndex,
                                offset: usize,
                                state: &[CudaSlice<f32>]|
                 -> Result<Vec<CudaSlice<f32>>, String> {
                    index.set(&host.stream, offset)?;
                    let mut bindings = captures.clone();
                    bindings.insert(names.carry.clone(), &state[0]);
                    bindings.insert(names.index.clone(), &index.current);
                    host.run(body, &bindings)
                };
                let initial = vec![host.copy(cuda_value(values, *carry)?)?];
                let mut tape = DeviceTape::record(
                    &mut host,
                    &mut index,
                    names.upper - names.lower,
                    initial,
                    &mut step,
                )?;
                let mut cotangent = vec![host.copy(cuda_value(values, *final_carry_cotangent)?)?];
                let mut accumulators = host.zero_accumulators(reverse)?;
                let mut step_cotangent = host.take(output_count)?;
                while let Some((start, states)) =
                    tape.pop_block(&mut host, &mut index, &mut step)?
                {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        let offset = start + local;
                        index.set(&host.stream, offset)?;
                        host.stream
                            .memcpy_dtod(
                                &output_cotangents
                                    .slice(offset * output_count..(offset + 1) * output_count),
                                &mut step_cotangent,
                            )
                            .map_err(|error| {
                                format!("failed to bind CUDA Scan VJP output cotangent: {error:?}")
                            })?;
                        let mut bindings = captures.clone();
                        bindings.insert(names.carry.clone(), &state[0]);
                        bindings.insert(names.index.clone(), &index.current);
                        bindings.insert(reverse.cotangent.clone(), &cotangent[0]);
                        bindings.insert(output_cotangent.clone(), &step_cotangent);
                        let outputs =
                            host.reverse_step(&self.regions[1], bindings, reverse, &accumulators)?;
                        host.recycle_all(state);
                        (cotangent, accumulators) =
                            host.advance_reverse(outputs, reverse, 1, cotangent, accumulators)?;
                    }
                }
                let mut gradients = host.gradient_results(reverse, accumulators)?;
                gradients.insert(names.carry.clone(), cotangent.swap_remove(0));
                members
                    .iter()
                    .map(|member| {
                        let TensorOp::ScanVjp { target, .. } = &plan.nodes[*member].op else {
                            return Err(mismatch(*member));
                        };
                        let name = match target {
                            TensorScanVjpTarget::Carry => &names.carry,
                            TensorScanVjpTarget::External(name) => name,
                        };
                        take_gradient(&mut host, &mut gradients, plan, *member, name)
                    })
                    .collect()
            }
        }
    }
}

fn mismatch(node_id: TensorNodeId) -> String {
    format!("CUDA host-driven loop node {node_id} does not match its compiled regions")
}

fn loop_captures<'a>(
    op: &TensorOp,
    values: &'a [Option<CudaSlice<f32>>],
) -> Result<BTreeMap<String, &'a CudaSlice<f32>>, String> {
    let captures = match op {
        TensorOp::While { captures, .. }
        | TensorOp::Fori { captures, .. }
        | TensorOp::ForiJvp { captures, .. }
        | TensorOp::ForiVjp { captures, .. }
        | TensorOp::ForiVjpJvp { captures, .. }
        | TensorOp::Scan { captures, .. }
        | TensorOp::ScanVjp { captures, .. } => captures,
        _ => return Err("CUDA host-driven loop node has no captures".to_string()),
    };
    named_values(captures, values)
}

fn named_values<'a>(
    bindings: &[(String, TensorNodeId)],
    values: &'a [Option<CudaSlice<f32>>],
) -> Result<BTreeMap<String, &'a CudaSlice<f32>>, String> {
    bindings
        .iter()
        .map(|(name, node)| cuda_value(values, *node).map(|value| (name.clone(), value)))
        .collect()
}

fn scan_output_count(op: &TensorOp) -> Result<usize, String> {
    let TensorOp::Scan { scan_plan, .. } = op else {
        return Err("CUDA host-driven Scan node is not a Scan".to_string());
    };
    element_count(&scan_plan.body.output_shapes()[1])
}

/// Moves one named gradient to a group member, or zeros when the body does
/// not depend on that capture.
fn take_gradient(
    host: &mut HostRun,
    gradients: &mut BTreeMap<String, CudaSlice<f32>>,
    plan: &TensorExecutionPlan,
    member: TensorNodeId,
    name: &str,
) -> Result<(TensorNodeId, CudaSlice<f32>), String> {
    match gradients.remove(name) {
        Some(gradient) => Ok((member, gradient)),
        None => {
            let count = element_count(&plan.nodes[member].shape)?;
            host.zeros(count).map(|zeros| (member, zeros))
        }
    }
}

fn fori_vjp_results(
    host: &mut HostRun,
    plan: &TensorExecutionPlan,
    members: &[TensorNodeId],
    carry: &str,
    mut gradients: BTreeMap<String, CudaSlice<f32>>,
) -> Result<Vec<(TensorNodeId, CudaSlice<f32>)>, String> {
    members
        .iter()
        .map(|member| {
            let target = match &plan.nodes[*member].op {
                TensorOp::ForiVjp { target, .. } | TensorOp::ForiVjpJvp { target, .. } => target,
                _ => return Err(mismatch(*member)),
            };
            let name = match target {
                TensorForiVjpTarget::Carry => carry,
                TensorForiVjpTarget::External(name) => name,
            };
            take_gradient(host, &mut gradients, plan, *member, name)
        })
        .collect()
}

/// One forward-mode body step over a `(carry, tangent)` state.
#[allow(clippy::too_many_arguments)]
fn forward_jvp_step(
    host: &mut HostRun,
    index: &mut LoopIndex,
    region: &CudaExecutionPlan,
    names: &LoopNames,
    tangents: &BTreeMap<String, String>,
    captures: &BTreeMap<String, &CudaSlice<f32>>,
    tangent_values: &BTreeMap<String, &CudaSlice<f32>>,
    offset: usize,
    state: &[CudaSlice<f32>],
) -> Result<Vec<CudaSlice<f32>>, String> {
    index.set(&host.stream, offset)?;
    let mut bindings = captures.clone();
    bindings.insert(names.carry.clone(), &state[0]);
    bindings.insert(names.index.clone(), &index.current);
    for (name, tangent_name) in tangents {
        if name == &names.carry {
            bindings.insert(tangent_name.clone(), &state[1]);
        } else {
            let tangent = tangent_values
                .get(name)
                .ok_or_else(|| format!("CUDA Fori JVP lacks tangent capture {name:?}"))?;
            bindings.insert(tangent_name.clone(), tangent);
        }
    }
    host.run(region, &bindings)
}

/// Device stream plus the loop's private buffer pool.
struct HostRun {
    stream: Arc<CudaStream>,
    pool: CudaBufferPool,
}

impl HostRun {
    fn take(&mut self, count: usize) -> Result<CudaSlice<f32>, String> {
        super::take_cuda_buffer(&self.stream, &mut self.pool, count, usize::MAX)
    }

    fn copy(&mut self, source: &CudaSlice<f32>) -> Result<CudaSlice<f32>, String> {
        let mut buffer = self.take(source.len())?;
        self.stream
            .memcpy_dtod(source, &mut buffer)
            .map_err(|error| format!("failed to copy a CUDA loop buffer: {error:?}"))?;
        Ok(buffer)
    }

    fn zeros(&mut self, count: usize) -> Result<CudaSlice<f32>, String> {
        let mut buffer = self.take(count)?;
        self.stream
            .memset_zeros(&mut buffer)
            .map_err(|error| format!("failed to clear a CUDA loop buffer: {error:?}"))?;
        Ok(buffer)
    }

    fn recycle_all(&mut self, buffers: Vec<CudaSlice<f32>>) {
        for buffer in buffers {
            self.pool.recycle(buffer);
        }
    }

    /// Executes one region with `bindings` as its captures and copies each
    /// output into a buffer of this loop's pool. The region keeps its own value
    /// table between iterations, so steady-state iterations allocate nothing.
    fn run(
        &mut self,
        region: &CudaExecutionPlan,
        bindings: &BTreeMap<String, &CudaSlice<f32>>,
    ) -> Result<Vec<CudaSlice<f32>>, String> {
        let mut state = region
            .state
            .lock()
            .map_err(|_| "CUDA loop region state lock is poisoned".to_string())?;
        let CudaExecutionState {
            values,
            free_buffers,
            ..
        } = &mut *state;
        if values.len() != region.plan.nodes.len() {
            *values = std::iter::repeat_with(|| None)
                .take(region.plan.nodes.len())
                .collect();
        }
        recycle_cuda_computed_values(&region.plan, values, free_buffers);
        execute_cuda_device_program(
            &region.plan,
            &BTreeMap::new(),
            CudaProgramRuntime {
                stream: &self.stream,
                module: &region.module,
                blas: region.blas.as_ref(),
                solver: region.solver.as_ref(),
                cond_branches: &region.cond_branches,
                host_loops: &region.host_loops,
                region_captures: bindings,
                constant_uploads: &region.constant_uploads,
            },
            values,
            free_buffers,
            &BTreeSet::new(),
            false,
        )?;
        region
            .plan
            .output_node_ids()
            .iter()
            .map(|output| {
                let source = cuda_value(values, *output)?;
                let mut buffer = self.take(source.len())?;
                self.stream
                    .memcpy_dtod(source, &mut buffer)
                    .map_err(|error| {
                        format!("failed to copy a CUDA loop region output: {error:?}")
                    })?;
                Ok(buffer)
            })
            .collect()
    }

    fn run_one(
        &mut self,
        region: &CudaExecutionPlan,
        bindings: &BTreeMap<String, &CudaSlice<f32>>,
    ) -> Result<CudaSlice<f32>, String> {
        let mut outputs = self.run(region, bindings)?;
        let output = outputs
            .pop()
            .ok_or_else(|| "CUDA loop region produced no output".to_string())?;
        self.recycle_all(outputs);
        Ok(output)
    }

    fn zero_accumulators(&mut self, layout: &ReverseLayout) -> Result<Vec<CudaSlice<f32>>, String> {
        layout
            .accumulators
            .iter()
            .map(|(_, _, shape)| element_count(shape).and_then(|count| self.zeros(count)))
            .collect()
    }

    /// Binds the accumulators and runs one reverse iteration.
    fn reverse_step<'a>(
        &mut self,
        region: &CudaExecutionPlan,
        mut bindings: BTreeMap<String, &'a CudaSlice<f32>>,
        layout: &ReverseLayout,
        accumulators: &'a [CudaSlice<f32>],
    ) -> Result<Vec<CudaSlice<f32>>, String> {
        for ((_, name, _), accumulator) in layout.accumulators.iter().zip(accumulators) {
            bindings.insert(name.clone(), accumulator);
        }
        self.run(region, &bindings)
    }

    /// Splits reverse outputs into the next carry cotangent state (`width`
    /// buffers, or zeros when the body ignores its carry) and accumulators.
    fn advance_reverse(
        &mut self,
        mut outputs: Vec<CudaSlice<f32>>,
        layout: &ReverseLayout,
        width: usize,
        cotangent: Vec<CudaSlice<f32>>,
        accumulators: Vec<CudaSlice<f32>>,
    ) -> Result<(DeviceState, DeviceState), String> {
        self.recycle_all(accumulators);
        let next_accumulators = outputs.split_off(if layout.carry_gradient { width } else { 0 });
        let next_cotangent = if layout.carry_gradient {
            self.recycle_all(cotangent);
            outputs
        } else {
            let counts = cotangent.iter().map(CudaSlice::len).collect::<Vec<_>>();
            self.recycle_all(cotangent);
            counts
                .into_iter()
                .map(|count| self.zeros(count))
                .collect::<Result<_, _>>()?
        };
        Ok((next_cotangent, next_accumulators))
    }

    fn gradient_results(
        &mut self,
        layout: &ReverseLayout,
        accumulators: Vec<CudaSlice<f32>>,
    ) -> Result<BTreeMap<String, CudaSlice<f32>>, String> {
        Ok(layout
            .accumulators
            .iter()
            .map(|(name, _, _)| name.clone())
            .zip(accumulators)
            .collect())
    }
}

/// Device copy of the loop indices; `current` is bound as the scalar index
/// capture and refreshed by a device-to-device copy per iteration.
struct LoopIndex {
    indices: CudaSlice<f32>,
    current: CudaSlice<f32>,
}

impl LoopIndex {
    fn new(host: &mut HostRun, names: &LoopNames) -> Result<Self, String> {
        // Loop indices are exact in float32 up to 2^24, like the fused kernels'
        // `(float)step`.
        let indices = (names.lower..names.upper)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let indices = if indices.is_empty() {
            host.zeros(1)?
        } else {
            host.stream
                .clone_htod(&indices)
                .map_err(|error| format!("failed to upload CUDA loop indices: {error:?}"))?
        };
        Ok(Self {
            indices,
            current: host.zeros(1)?,
        })
    }

    fn set(&mut self, stream: &Arc<CudaStream>, offset: usize) -> Result<(), String> {
        stream
            .memcpy_dtod(&self.indices.slice(offset..offset + 1), &mut self.current)
            .map_err(|error| format!("failed to set the CUDA loop index: {error:?}"))
    }
}

/// The device buffers of one loop state: a carry, or a carry and its tangent,
/// or the carry-cotangent pair of a reverse pass.
type DeviceState = Vec<CudaSlice<f32>>;

type StepFn<'a> = dyn FnMut(
        &mut HostRun,
        &mut LoopIndex,
        usize,
        &[CudaSlice<f32>],
    ) -> Result<Vec<CudaSlice<f32>>, String>
    + 'a;

/// Device carry tape for reverse passes, using the CPU checkpoint scheme:
/// `TensorCarryCheckpoints::block_size` decides between keeping every state
/// and keeping one state per block, replayed per block in reverse.
struct DeviceTape {
    /// Full tape (`block == None`): the state before every step. Otherwise
    /// the state at the start of every block.
    states: Vec<Vec<CudaSlice<f32>>>,
    block: Option<usize>,
    steps: usize,
}

impl DeviceTape {
    fn record(
        host: &mut HostRun,
        index: &mut LoopIndex,
        steps: usize,
        initial: Vec<CudaSlice<f32>>,
        step: &mut StepFn<'_>,
    ) -> Result<Self, String> {
        let block = TensorCarryCheckpoints::<DynamicTensor>::block_size(steps);
        let mut states = vec![initial];
        // The latest state when it is not itself a stored checkpoint.
        let mut current: Option<Vec<CudaSlice<f32>>> = None;
        // The final state is not needed by the reverse pass.
        for offset in 0..steps.saturating_sub(1) {
            let source = match &current {
                Some(state) => state.as_slice(),
                None => states
                    .last()
                    .expect("the tape starts with the initial state"),
            };
            let next = step(host, index, offset, source)?;
            if let Some(stale) = current.take() {
                host.recycle_all(stale);
            }
            match block {
                Some(block) if !(offset + 1).is_multiple_of(block) => current = Some(next),
                _ => states.push(next),
            }
        }
        if let Some(stale) = current {
            host.recycle_all(stale);
        }
        Ok(Self {
            states,
            block,
            steps,
        })
    }

    fn pop_block(
        &mut self,
        host: &mut HostRun,
        index: &mut LoopIndex,
        step: &mut StepFn<'_>,
    ) -> Result<Option<(usize, Vec<DeviceState>)>, String> {
        let Some(block) = self.block else {
            if self.states.is_empty() || self.steps == 0 {
                return Ok(None);
            }
            return Ok(Some((0, std::mem::take(&mut self.states))));
        };
        let Some(first) = self.states.pop() else {
            return Ok(None);
        };
        let start = self.states.len() * block;
        let length = block.min(self.steps - start);
        let mut states = vec![first];
        for offset in start..start + length - 1 {
            let next = step(host, index, offset, states.last().expect("non-empty"))?;
            states.push(next);
        }
        Ok(Some((start, states)))
    }
}
