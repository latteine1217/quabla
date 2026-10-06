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
//! Per-iteration cost. A region run copies its per-iteration inputs (carry,
//! loop index, cotangents) into the region's input buffers, launches the
//! region program, and copies its outputs into buffers of the loop's pool.
//! Loop-invariant captures are copied once per loop execution. Every launch
//! and copy costs a driver call and a device scheduling slot, which dominate
//! small bodies (tens of microseconds each under WDDM). So after two eager
//! runs a region whose program only launches NVRTC kernels and device copies
//! is recorded once as a CUDA graph (`RegionGraph`), and each further run
//! retargets the graph's input and output copies to that run's buffers and
//! launches the whole iteration as one graph. The graph replays the eager
//! launch sequence exactly, so results are bit-identical to eager runs.
//!
//! The fused kernels stay the fast path; a node falls back to this executor
//! only when its fused lowering fails (`cuda_loop_fused_error`), and grouped
//! loop results (`Scan` targets, loop VJP targets) choose one mode per group.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use cudarc::driver::{
    result, sys, CudaContext, CudaGraph, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr,
};

use super::super::{
    element_count, DynamicTensor, RegionKind, RegionNode, RegionView, SymbolicCotangent,
    TensorCarryCheckpoints, TensorExecutionPlan, TensorForiVjpTarget, TensorIr, TensorNodeId,
    TensorOp, TensorScanTarget, TensorScanVjpTarget,
};
use super::{
    cuda_fori_jvp_tangent_names, cuda_fused_loop_lowering, cuda_scalar_predicate, cuda_value,
    execute_cuda_device_program, recycle_cuda_computed_values, CudaBackend, CudaBufferPool,
    CudaExecutionPlan, CudaExecutionState, CudaProgramRuntime, CudaReal,
};

/// One loop node compiled for host-driven execution.
#[derive(Clone, Debug)]
pub(super) struct CudaHostLoop<T: CudaReal> {
    kind: HostLoopKind,
    /// Device programs in the order `HostLoopKind` documents for its variant.
    regions: Vec<CudaExecutionPlan<T>>,
    /// Whether each region may be recorded as a CUDA graph (`cuda_graph_capturable`).
    capturable: Vec<bool>,
    /// The non-blocking stream that records region graphs; it never executes
    /// work. `None` when no region is capturable.
    capture_stream: Option<Arc<CudaStream>>,
}

/// Eager runs of a region before it is recorded as a graph: the first binds the
/// loop invariants and fills the region's value table and buffer pool, the
/// second confirms the steady state the recording needs (no allocation). Loops
/// of one or two iterations never pay for a recording.
const CUDA_GRAPH_EAGER_RUNS: usize = 2;

/// Whether every launch of `region` can be recorded into a CUDA graph and
/// replayed: its program may only launch NVRTC kernels and device copies or
/// memsets on the given stream, and cuBLAS matrix products (the region's own
/// handle is pointed at the recording stream). Excluded are nodes that read
/// back to the host (`Cond`, `While`), call cuSOLVER, run nested loops
/// (host-driven or fused, which allocate tapes), or are Cholesky kernels.
/// Every elementwise math kind (`UnaryMath`, `BinaryMath`) is admitted:
/// each runs as one NVRTC kernel of the per-node program.
fn cuda_graph_capturable<T: CudaReal>(region: &CudaExecutionPlan<T>) -> bool {
    region.host_loops.is_empty()
        && region.cond_branches.is_empty()
        && region.matmul_bias_tanh.is_none()
        && !region.fused_elementwise
        && region.plan.nodes.iter().all(|node| {
            matches!(
                node.op,
                TensorOp::Input { .. }
                    | TensorOp::Constant { .. }
                    | TensorOp::ScalarConstant { .. }
                    | TensorOp::Add { .. }
                    | TensorOp::Sub { .. }
                    | TensorOp::Div { .. }
                    | TensorOp::Mul { .. }
                    | TensorOp::Greater { .. }
                    | TensorOp::Compare { .. }
                    | TensorOp::Where { .. }
                    | TensorOp::Sqrt { .. }
                    | TensorOp::SqrtDerivative { .. }
                    | TensorOp::Powi { .. }
                    | TensorOp::UnaryMath { .. }
                    | TensorOp::BinaryMath { .. }
                    | TensorOp::CumSum { .. }
                    | TensorOp::Matmul { .. }
                    | TensorOp::Sum { .. }
                    | TensorOp::Mean { .. }
                    | TensorOp::SumAxis { .. }
                    | TensorOp::MeanAxis { .. }
                    | TensorOp::ExtremumAxis { .. }
                    | TensorOp::Transpose { .. }
                    | TensorOp::Concat { .. }
                    | TensorOp::Slice { .. }
                    | TensorOp::PadSlice { .. }
                    | TensorOp::Gather { .. }
                    | TensorOp::ScatterAdd { .. }
                    | TensorOp::Broadcast { .. }
                    | TensorOp::Reshape { .. }
                    | TensorOp::Cast { .. }
                    | TensorOp::StopGradient { .. }
                    | TensorOp::Custom { .. }
            )
        })
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
    /// Regions: `[forward, reverse]` as for `ForiVjpJvp`; the reverse region
    /// also reads the step's output cotangent and its tangent.
    ScanVjpJvp {
        names: LoopNames,
        forward_tangents: BTreeMap<String, String>,
        reverse_tangents: BTreeMap<String, String>,
        reverse: ReverseLayout,
    },
}

/// Group key of a loop node whose sibling nodes share one execution: group
/// ids are compared within one op kind.
fn loop_group_key(op: &TensorOp) -> Option<(&'static str, usize)> {
    let region = RegionView::of(op)?;
    Some((region.name(), region.group()?))
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
/// `None` for fused-lowerable loops and non-loop nodes.
pub(super) fn cuda_loop_fused_error(
    plan: &TensorExecutionPlan,
    node_id: TensorNodeId,
) -> Option<String> {
    cuda_loop_group_members(plan, node_id)
        .into_iter()
        .find_map(|member| cuda_fused_loop_lowering(&plan.nodes[member].op)?.err())
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

/// The reverse region of a forward-over-reverse loop from the body VJP `graph`
/// and its `gradients`: the carry cotangent and its tangent (when the body
/// reads its carry), then one accumulated tangent per capture gradient.
/// Returns the region, whether it has the carry outputs, and the accumulator
/// layout.
#[allow(clippy::type_complexity)]
fn forward_over_reverse_region(
    graph: &TensorIr,
    gradients: &BTreeMap<String, TensorNodeId>,
    carry: &str,
    captures: &BTreeMap<String, Vec<usize>>,
    tangent_names: &BTreeMap<String, String>,
) -> Result<(TensorExecutionPlan, bool, Vec<(String, String, Vec<usize>)>), String> {
    let carry_gradient = gradients.get(carry).copied();
    let capture_gradients = captures
        .keys()
        .filter_map(|name| gradients.get(name).map(|id| (name.clone(), *id)))
        .collect::<Vec<_>>();
    let differentiated = carry_gradient
        .iter()
        .copied()
        .chain(capture_gradients.iter().map(|(_, id)| *id))
        .collect::<Vec<_>>();
    let jvp = graph.symbolic_jvp_many_with_tangent_inputs(&differentiated, tangent_names)?;
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
        captures,
        &mut taken,
        &mut outputs,
    )?;
    let (reverse, _) = graph.compile_cpu_many(&outputs)?;
    Ok((reverse, carry_gradient.is_some(), accumulators))
}

/// Builds the region plans of a host-driven loop node; also the validation
/// that the fallback can lower.
fn host_loop_ir(op: &TensorOp) -> Result<(HostLoopKind, Vec<TensorExecutionPlan>), String> {
    match op {
        TensorOp::Region(RegionNode {
            kind: RegionKind::While { loop_plan, .. },
            ..
        }) => Ok((
            HostLoopKind::While {
                carry: loop_plan.carry_name().to_string(),
            },
            vec![
                loop_plan.predicate_plan().clone(),
                loop_plan.body_plan().clone(),
            ],
        )),
        TensorOp::Region(RegionNode {
            kind: RegionKind::Fori { loop_plan, .. },
            ..
        }) => Ok((
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
        TensorOp::Region(RegionNode {
            kind: RegionKind::ForiJvp { loop_plan, .. },
            ..
        }) => {
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
            let (reverse, carry_gradient, accumulators) = forward_over_reverse_region(
                &vjp.graph,
                &vjp.gradients,
                &loop_plan.carry_name,
                loop_plan.external_captures(),
                &plan.tangent_names,
            )?;
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
                        carry_gradient,
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
        TensorOp::ScanVjpJvp { plan, .. } => {
            let scan_plan = &plan.scan_plan;
            let body = &scan_plan.body.plan;
            let [carry_output, step_output] = body.output_node_ids() else {
                return Err("scan body must return a carry and an output".to_string());
            };
            let ir = body.as_ir();
            // The forward pass differentiates the carry along the body captures' tangents; the
            // two cotangent inputs exist only in the reverse region.
            let mut forward_tangents = plan.tangent_names.clone();
            forward_tangents.remove(&plan.carry_cotangent_name);
            forward_tangents.remove(&plan.output_cotangent_name);
            let forward = ir.symbolic_jvp_with_tangent_inputs(*carry_output, &forward_tangents)?;
            let (forward, _) = forward
                .graph
                .compile_cpu_many(&[forward.value, forward.tangent])?;
            // One VJP of both body outputs, so each gradient is the sum of the carry and the
            // step-output contributions, as in `TensorScanVjpJvpExecutionPlan::jvp`.
            let vjp = ir.symbolic_vjp_many(&[
                (
                    *carry_output,
                    SymbolicCotangent::Input(plan.carry_cotangent_name.clone()),
                ),
                (
                    *step_output,
                    SymbolicCotangent::Input(plan.output_cotangent_name.clone()),
                ),
            ])?;
            let (reverse, carry_gradient, accumulators) = forward_over_reverse_region(
                &vjp.graph,
                &vjp.gradients,
                &scan_plan.carry_name,
                scan_plan.external_captures(),
                &plan.tangent_names,
            )?;
            Ok((
                HostLoopKind::ScanVjpJvp {
                    names: loop_names(
                        scan_plan.lower,
                        scan_plan.upper,
                        &scan_plan.carry_name,
                        &scan_plan.index_name,
                    ),
                    forward_tangents,
                    reverse_tangents: plan.tangent_names.clone(),
                    reverse: ReverseLayout {
                        cotangent: plan.carry_cotangent_name.clone(),
                        carry_gradient,
                        accumulators,
                    },
                },
                vec![forward, reverse],
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

impl<T: CudaReal> CudaHostLoop<T> {
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
        for region in &regions {
            // A region runs once per iteration and starts each run by returning all of its
            // computed buffers, outputs included, to its pool. The plan's frontier budget
            // excludes outputs, so it would free some buffers every run and allocate them again
            // (and a graph recording, which must not allocate, would fail). Without a budget the
            // pool keeps one run's working set: the frontier plus the outputs.
            region
                .state
                .lock()
                .map_err(|_| "CUDA loop region state lock is poisoned".to_string())?
                .free_buffers
                .budget = usize::MAX;
        }
        let capturable = regions
            .iter()
            .map(cuda_graph_capturable)
            .collect::<Vec<_>>();
        // Plan contexts run without cudarc event tracking (see `compile_in_context`), so this
        // stream does not make the context synchronize buffers across streams.
        let capture_stream = if capturable.contains(&true) {
            Some(context.new_stream().map_err(|error| {
                format!("failed to create the CUDA loop node {node_id} capture stream: {error:?}")
            })?)
        } else {
            None
        };
        Ok(Self {
            kind,
            regions,
            capturable,
            capture_stream,
        })
    }

    /// Runs the loop and returns the result buffer of every group member.
    pub(super) fn execute(
        &self,
        plan: &TensorExecutionPlan,
        node_id: TensorNodeId,
        values: &[Option<CudaSlice<T>>],
        stream: &Arc<CudaStream>,
    ) -> Result<Vec<(TensorNodeId, CudaSlice<T>)>, String> {
        let mut host = HostRun::new(self, stream)?;
        let members = cuda_loop_group_members(plan, node_id);
        let op = &plan.nodes[node_id].op;
        let captures = loop_captures(op, values)?;
        match &self.kind {
            HostLoopKind::While { carry } => {
                let TensorOp::Region(RegionNode {
                    kind: RegionKind::While { carry: initial, .. },
                    ..
                }) = op
                else {
                    return Err(mismatch(node_id));
                };
                let mut state = host.copy(cuda_value(values, *initial)?)?;
                loop {
                    let dynamic = [(carry.as_str(), state.as_view())];
                    let predicate = host.run_one(0, &[&captures], &dynamic)?;
                    // Host synchronization point: one scalar is read back per iteration.
                    let predicate_value = host.stream.clone_dtoh(&predicate).map_err(|error| {
                        format!("failed to read CUDA While node {node_id} predicate: {error:?}")
                    })?;
                    host.pool.recycle(predicate);
                    if !cuda_scalar_predicate(&predicate_value)? {
                        break;
                    }
                    let next = host.run_one(1, &[&captures], &dynamic)?;
                    host.pool.recycle(std::mem::replace(&mut state, next));
                }
                Ok(vec![(node_id, state)])
            }
            HostLoopKind::Fori { names } => {
                let TensorOp::Region(RegionNode {
                    kind: RegionKind::Fori { carry, .. },
                    ..
                }) = op
                else {
                    return Err(mismatch(node_id));
                };
                let index = LoopIndex::new(&mut host, names)?;
                let mut state = host.copy(cuda_value(values, *carry)?)?;
                for offset in 0..names.upper - names.lower {
                    let dynamic = [
                        (names.carry.as_str(), state.as_view()),
                        (names.index.as_str(), index.at(offset)),
                    ];
                    let next = host.run_one(0, &[&captures], &dynamic)?;
                    host.pool.recycle(std::mem::replace(&mut state, next));
                }
                Ok(vec![(node_id, state)])
            }
            HostLoopKind::ForiJvp { names, tangents } => {
                let TensorOp::Region(RegionNode {
                    kind:
                        RegionKind::ForiJvp {
                            carry,
                            carry_tangent,
                            ..
                        },
                    tangent_captures,
                    ..
                }) = op
                else {
                    return Err(mismatch(node_id));
                };
                let tangent_values = named_values(tangent_captures, values)?;
                let tangent_bindings = forward_tangent_bindings(names, tangents, &tangent_values)?;
                let index = LoopIndex::new(&mut host, names)?;
                let mut state = vec![
                    host.copy(cuda_value(values, *carry)?)?,
                    host.copy(cuda_value(values, *carry_tangent)?)?,
                ];
                for offset in 0..names.upper - names.lower {
                    let next = forward_jvp_step(
                        &mut host,
                        names,
                        tangents,
                        &[&captures, &tangent_bindings],
                        &index,
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
                let index = LoopIndex::new(&mut host, names)?;
                let mut step = |host: &mut HostRun<T>,
                                offset: usize,
                                state: &[CudaSlice<T>]|
                 -> Result<Vec<CudaSlice<T>>, String> {
                    host.run(
                        0,
                        &[&captures],
                        &[
                            (names.carry.as_str(), state[0].as_view()),
                            (names.index.as_str(), index.at(offset)),
                        ],
                    )
                };
                let initial = vec![host.copy(cuda_value(values, *carry)?)?];
                let mut tape =
                    DeviceTape::record(&mut host, names.upper - names.lower, initial, &mut step)?;
                let mut cotangent = vec![host.copy(cuda_value(values, *output_cotangent)?)?];
                let mut accumulators = host.zero_accumulators(reverse)?;
                while let Some((start, states)) = tape.pop_block(&mut host, &mut step)? {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        let dynamic = vec![
                            (names.carry.as_str(), state[0].as_view()),
                            (names.index.as_str(), index.at(start + local)),
                            (reverse.cotangent.as_str(), cotangent[0].as_view()),
                        ];
                        let outputs =
                            host.reverse_step(1, &[&captures], dynamic, reverse, &accumulators)?;
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
                let forward_bindings =
                    forward_tangent_bindings(names, forward_tangents, &tangent_values)?;
                let index = LoopIndex::new(&mut host, names)?;
                let mut step = |host: &mut HostRun<T>,
                                offset: usize,
                                state: &[CudaSlice<T>]|
                 -> Result<Vec<CudaSlice<T>>, String> {
                    forward_jvp_step(
                        host,
                        names,
                        forward_tangents,
                        &[&captures, &forward_bindings],
                        &index,
                        offset,
                        state,
                    )
                };
                let initial = vec![
                    host.copy(cuda_value(values, *carry)?)?,
                    host.copy(cuda_value(values, *carry_tangent)?)?,
                ];
                let mut tape =
                    DeviceTape::record(&mut host, names.upper - names.lower, initial, &mut step)?;
                let mut cotangent = vec![
                    host.copy(cuda_value(values, *output_cotangent)?)?,
                    host.copy(cuda_value(values, *output_cotangent_tangent)?)?,
                ];
                let cotangent_tangent_name = reverse_tangents
                    .get(&hvp.cotangent_name)
                    .ok_or_else(|| "CUDA Fori VJP JVP has no cotangent tangent".to_string())?;
                let (reverse_bindings, carry_tangent_name) =
                    reverse_tangent_bindings(&names.carry, reverse_tangents, &tangent_values);
                let mut accumulators = host.zero_accumulators(reverse)?;
                while let Some((start, states)) = tape.pop_block(&mut host, &mut step)? {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        let mut dynamic = vec![
                            (names.carry.as_str(), state[0].as_view()),
                            (names.index.as_str(), index.at(start + local)),
                            (reverse.cotangent.as_str(), cotangent[0].as_view()),
                            (cotangent_tangent_name.as_str(), cotangent[1].as_view()),
                        ];
                        if let Some(name) = carry_tangent_name {
                            dynamic.push((name, state[1].as_view()));
                        }
                        let outputs = host.reverse_step(
                            1,
                            &[&captures, &reverse_bindings],
                            dynamic,
                            reverse,
                            &accumulators,
                        )?;
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
                let index = LoopIndex::new(&mut host, names)?;
                let steps = names.upper - names.lower;
                let output_count = scan_output_count(op)?;
                let initial = cuda_value(values, *carry)?;
                let mut stacked = host.take(steps * output_count)?;
                let mut state = host.copy(initial)?;
                for offset in 0..steps {
                    // The step output is written straight into its row of the stack.
                    let mut next = host.take(initial.len())?;
                    let start = offset * output_count;
                    host.run_into(
                        0,
                        &[&captures],
                        &[
                            (names.carry.as_str(), state.as_view()),
                            (names.index.as_str(), index.at(offset)),
                        ],
                        &mut [
                            next.as_view_mut(),
                            stacked.slice_mut(start..start + output_count),
                        ],
                    )?;
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
                let index = LoopIndex::new(&mut host, names)?;
                let mut step = |host: &mut HostRun<T>,
                                offset: usize,
                                state: &[CudaSlice<T>]|
                 -> Result<Vec<CudaSlice<T>>, String> {
                    host.run(
                        0,
                        &[&captures],
                        &[
                            (names.carry.as_str(), state[0].as_view()),
                            (names.index.as_str(), index.at(offset)),
                        ],
                    )
                };
                let initial = vec![host.copy(cuda_value(values, *carry)?)?];
                let mut tape =
                    DeviceTape::record(&mut host, names.upper - names.lower, initial, &mut step)?;
                let mut cotangent = vec![host.copy(cuda_value(values, *final_carry_cotangent)?)?];
                let mut accumulators = host.zero_accumulators(reverse)?;
                while let Some((start, states)) = tape.pop_block(&mut host, &mut step)? {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        let offset = start + local;
                        let dynamic = vec![
                            (names.carry.as_str(), state[0].as_view()),
                            (names.index.as_str(), index.at(offset)),
                            (reverse.cotangent.as_str(), cotangent[0].as_view()),
                            (
                                output_cotangent.as_str(),
                                output_cotangents
                                    .slice(offset * output_count..(offset + 1) * output_count),
                            ),
                        ];
                        let outputs =
                            host.reverse_step(1, &[&captures], dynamic, reverse, &accumulators)?;
                        host.recycle_all(state);
                        (cotangent, accumulators) =
                            host.advance_reverse(outputs, reverse, 1, cotangent, accumulators)?;
                    }
                }
                let mut gradients = host.gradient_results(reverse, accumulators)?;
                gradients.insert(names.carry.clone(), cotangent.swap_remove(0));
                scan_vjp_results(&mut host, plan, &members, &names.carry, gradients)
            }
            HostLoopKind::ScanVjpJvp {
                names,
                forward_tangents,
                reverse_tangents,
                reverse,
            } => {
                let TensorOp::ScanVjpJvp {
                    carry,
                    carry_tangent,
                    final_carry_cotangent,
                    final_carry_cotangent_tangent,
                    output_cotangent: output_cotangents,
                    output_cotangent_tangent: output_cotangent_tangents,
                    plan: hvp,
                    tangent_captures,
                    ..
                } = op
                else {
                    return Err(mismatch(node_id));
                };
                let output_count = element_count(&hvp.scan_plan.body.output_shapes()[1])?;
                let output_cotangents = cuda_value(values, *output_cotangents)?;
                let output_cotangent_tangents = cuda_value(values, *output_cotangent_tangents)?;
                let tangent_values = named_values(tangent_captures, values)?;
                let forward_bindings =
                    forward_tangent_bindings(names, forward_tangents, &tangent_values)?;
                let index = LoopIndex::new(&mut host, names)?;
                let mut step = |host: &mut HostRun<T>,
                                offset: usize,
                                state: &[CudaSlice<T>]|
                 -> Result<Vec<CudaSlice<T>>, String> {
                    forward_jvp_step(
                        host,
                        names,
                        forward_tangents,
                        &[&captures, &forward_bindings],
                        &index,
                        offset,
                        state,
                    )
                };
                let initial = vec![
                    host.copy(cuda_value(values, *carry)?)?,
                    host.copy(cuda_value(values, *carry_tangent)?)?,
                ];
                let mut tape =
                    DeviceTape::record(&mut host, names.upper - names.lower, initial, &mut step)?;
                let mut cotangent = vec![
                    host.copy(cuda_value(values, *final_carry_cotangent)?)?,
                    host.copy(cuda_value(values, *final_carry_cotangent_tangent)?)?,
                ];
                let tangent_name = |name: &String| {
                    reverse_tangents.get(name).ok_or_else(|| {
                        format!("CUDA Scan VJP JVP has no tangent input for {name:?}")
                    })
                };
                let cotangent_tangent_name = tangent_name(&hvp.carry_cotangent_name)?;
                let output_cotangent_tangent_name = tangent_name(&hvp.output_cotangent_name)?;
                let (reverse_bindings, carry_tangent_name) =
                    reverse_tangent_bindings(&names.carry, reverse_tangents, &tangent_values);
                let mut accumulators = host.zero_accumulators(reverse)?;
                while let Some((start, states)) = tape.pop_block(&mut host, &mut step)? {
                    for (local, state) in states.into_iter().enumerate().rev() {
                        let offset = start + local;
                        let rows = offset * output_count..(offset + 1) * output_count;
                        let mut dynamic = vec![
                            (names.carry.as_str(), state[0].as_view()),
                            (names.index.as_str(), index.at(offset)),
                            (reverse.cotangent.as_str(), cotangent[0].as_view()),
                            (cotangent_tangent_name.as_str(), cotangent[1].as_view()),
                            (
                                hvp.output_cotangent_name.as_str(),
                                output_cotangents.slice(rows.clone()),
                            ),
                            (
                                output_cotangent_tangent_name.as_str(),
                                output_cotangent_tangents.slice(rows),
                            ),
                        ];
                        if let Some(name) = carry_tangent_name {
                            dynamic.push((name, state[1].as_view()));
                        }
                        let outputs = host.reverse_step(
                            1,
                            &[&captures, &reverse_bindings],
                            dynamic,
                            reverse,
                            &accumulators,
                        )?;
                        host.recycle_all(state);
                        (cotangent, accumulators) =
                            host.advance_reverse(outputs, reverse, 2, cotangent, accumulators)?;
                    }
                }
                let mut gradients = host.gradient_results(reverse, accumulators)?;
                let carry_tangent = cotangent.pop().ok_or_else(|| mismatch(node_id))?;
                host.recycle_all(cotangent);
                gradients.insert(names.carry.clone(), carry_tangent);
                scan_vjp_results(&mut host, plan, &members, &names.carry, gradients)
            }
        }
    }
}

fn mismatch(node_id: TensorNodeId) -> String {
    format!("CUDA host-driven loop node {node_id} does not match its compiled regions")
}

fn loop_captures<'a, T: CudaReal>(
    op: &TensorOp,
    values: &'a [Option<CudaSlice<T>>],
) -> Result<BTreeMap<String, &'a CudaSlice<T>>, String> {
    let captures = match RegionView::of(op) {
        Some(region) if region.is_loop() => region.captures(),
        _ => return Err("CUDA host-driven loop node has no captures".to_string()),
    };
    named_values(captures, values)
}

fn named_values<'a, T: CudaReal>(
    bindings: &[(String, TensorNodeId)],
    values: &'a [Option<CudaSlice<T>>],
) -> Result<BTreeMap<String, &'a CudaSlice<T>>, String> {
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
fn take_gradient<T: CudaReal>(
    host: &mut HostRun<T>,
    gradients: &mut BTreeMap<String, CudaSlice<T>>,
    plan: &TensorExecutionPlan,
    member: TensorNodeId,
    name: &str,
) -> Result<(TensorNodeId, CudaSlice<T>), String> {
    match gradients.remove(name) {
        Some(gradient) => Ok((member, gradient)),
        None => {
            let count = element_count(&plan.nodes[member].shape)?;
            host.zeros(count).map(|zeros| (member, zeros))
        }
    }
}

fn fori_vjp_results<T: CudaReal>(
    host: &mut HostRun<T>,
    plan: &TensorExecutionPlan,
    members: &[TensorNodeId],
    carry: &str,
    mut gradients: BTreeMap<String, CudaSlice<T>>,
) -> Result<Vec<(TensorNodeId, CudaSlice<T>)>, String> {
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

fn scan_vjp_results<T: CudaReal>(
    host: &mut HostRun<T>,
    plan: &TensorExecutionPlan,
    members: &[TensorNodeId],
    carry: &str,
    mut gradients: BTreeMap<String, CudaSlice<T>>,
) -> Result<Vec<(TensorNodeId, CudaSlice<T>)>, String> {
    members
        .iter()
        .map(|member| {
            let target = match &plan.nodes[*member].op {
                TensorOp::ScanVjp { target, .. } | TensorOp::ScanVjpJvp { target, .. } => target,
                _ => return Err(mismatch(*member)),
            };
            let name = match target {
                TensorScanVjpTarget::Carry => carry,
                TensorScanVjpTarget::External(name) => name,
            };
            take_gradient(host, &mut gradients, plan, *member, name)
        })
        .collect()
}

/// The loop-invariant tangent inputs of a forward-mode body region, keyed by
/// the region's tangent input names; the carry tangent is per-iteration state.
fn forward_tangent_bindings<'a, T: CudaReal>(
    names: &LoopNames,
    tangents: &BTreeMap<String, String>,
    tangent_values: &BTreeMap<String, &'a CudaSlice<T>>,
) -> Result<BTreeMap<String, &'a CudaSlice<T>>, String> {
    tangents
        .iter()
        .filter(|(name, _)| *name != &names.carry)
        .map(|(name, tangent_name)| {
            tangent_values
                .get(name)
                .map(|tangent| (tangent_name.clone(), *tangent))
                .ok_or_else(|| format!("CUDA loop JVP lacks tangent capture {name:?}"))
        })
        .collect()
}

/// The loop-invariant tangent inputs of a forward-over-reverse region, and the
/// region input name of the carry tangent when the region reads it. Tangent
/// names without a tangent capture (the cotangent's own tangent) are bound by
/// the caller.
fn reverse_tangent_bindings<'a, 'n, T: CudaReal>(
    carry: &str,
    tangents: &'n BTreeMap<String, String>,
    tangent_values: &BTreeMap<String, &'a CudaSlice<T>>,
) -> (BTreeMap<String, &'a CudaSlice<T>>, Option<&'n str>) {
    let mut bindings = BTreeMap::new();
    let mut carry_tangent = None;
    for (name, tangent_name) in tangents {
        if name == carry {
            carry_tangent = Some(tangent_name.as_str());
        } else if let Some(tangent) = tangent_values.get(name) {
            bindings.insert(tangent_name.clone(), *tangent);
        }
    }
    (bindings, carry_tangent)
}

/// One forward-mode body step (region 0) over a `(carry, tangent)` state.
#[allow(clippy::too_many_arguments)]
fn forward_jvp_step<T: CudaReal>(
    host: &mut HostRun<T>,
    names: &LoopNames,
    tangents: &BTreeMap<String, String>,
    invariant: &[&BTreeMap<String, &CudaSlice<T>>],
    index: &LoopIndex<T>,
    offset: usize,
    state: &[CudaSlice<T>],
) -> Result<Vec<CudaSlice<T>>, String> {
    let mut dynamic = vec![
        (names.carry.as_str(), state[0].as_view()),
        (names.index.as_str(), index.at(offset)),
    ];
    if let Some(tangent_name) = tangents.get(&names.carry) {
        dynamic.push((tangent_name.as_str(), state[1].as_view()));
    }
    host.run(0, invariant, &dynamic)
}

/// Per-iteration inputs of a region run, by region input name.
type Bindings<'n, 'v, T> = [(&'n str, CudaView<'v, T>)];

/// One input buffer of a region program.
#[derive(Clone, Debug)]
struct RegionInput {
    name: String,
    node: TensorNodeId,
    count: usize,
    /// Bound once per loop execution from a capture rather than per run.
    invariant: bool,
}

/// What one loop execution knows about one of its regions.
struct RegionRun {
    inputs: Vec<RegionInput>,
    /// Names of `inputs`; they stay resident in the region's value table.
    input_names: BTreeSet<String>,
    runs: usize,
    graph: GraphState,
}

enum GraphState {
    /// Eager runs only: the region is not capturable, or recording failed.
    Eager,
    /// Capturable; recorded after `CUDA_GRAPH_EAGER_RUNS` eager runs.
    Pending,
    Ready(RegionGraph),
}

impl RegionRun {
    fn new<T: CudaReal>(region: &CudaExecutionPlan<T>, capturable: bool) -> Result<Self, String> {
        let inputs = region
            .plan
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(node, value)| match &value.op {
                TensorOp::Input { name } => {
                    Some(element_count(&value.shape).map(|count| RegionInput {
                        name: name.clone(),
                        node,
                        count,
                        invariant: false,
                    }))
                }
                _ => None,
            })
            .collect::<Result<Vec<_>, String>>()?;
        let input_names = inputs.iter().map(|input| input.name.clone()).collect();
        #[cfg(test)]
        let capturable = capturable && !CUDA_LOOP_GRAPHS_DISABLED.with(std::cell::Cell::get);
        Ok(Self {
            inputs,
            input_names,
            runs: 0,
            graph: if capturable {
                GraphState::Pending
            } else {
                GraphState::Eager
            },
        })
    }
}

#[cfg(test)]
thread_local! {
    /// Forces eager region runs, so tests can compare graph replays against them.
    pub(super) static CUDA_LOOP_GRAPHS_DISABLED: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
    /// Counts region graph launches, so tests can check that replays happened.
    pub(super) static CUDA_LOOP_GRAPH_LAUNCHES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

fn find_binding<'b, 'v, T>(
    dynamic: &'b Bindings<'_, 'v, T>,
    name: &str,
) -> Option<&'b CudaView<'v, T>> {
    dynamic
        .iter()
        .find(|(candidate, _)| *candidate == name)
        .map(|(_, view)| view)
}

fn find_invariant<'a, T>(
    invariant: &[&BTreeMap<String, &'a CudaSlice<T>>],
    name: &str,
) -> Option<&'a CudaSlice<T>> {
    invariant
        .iter()
        .find_map(|bindings| bindings.get(name).copied())
}

/// Copies `source` into the value-table buffer of region input `input`.
fn bind_input<T: CudaReal, S: DevicePtr<T>>(
    stream: &Arc<CudaStream>,
    values: &mut [Option<CudaSlice<T>>],
    free_buffers: &mut CudaBufferPool<T>,
    input: &RegionInput,
    source: &S,
) -> Result<(), String> {
    if source.len() != input.count {
        return Err(format!(
            "CUDA loop region input {:?} has {} elements, expected {}",
            input.name,
            source.len(),
            input.count
        ));
    }
    let slot = values
        .get_mut(input.node)
        .ok_or_else(|| format!("CUDA loop region input {:?} has no buffer slot", input.name))?;
    if slot.is_none() {
        *slot = Some(super::take_cuda_buffer(
            stream,
            free_buffers,
            input.count,
            input.node,
        )?);
    }
    stream
        .memcpy_dtod(source, slot.as_mut().expect("allocated above"))
        .map_err(|error| {
            format!(
                "failed to bind CUDA loop region input {:?}: {error:?}",
                input.name
            )
        })
}

/// Writes the per-iteration inputs, and with `invariant` the loop invariants,
/// of one region run. The first run of a loop execution passes `invariant`
/// and decides which inputs are invariant; later runs bind only the others.
fn bind_region_inputs<T: CudaReal>(
    stream: &Arc<CudaStream>,
    run: &mut RegionRun,
    values: &mut [Option<CudaSlice<T>>],
    free_buffers: &mut CudaBufferPool<T>,
    invariant: Option<&[&BTreeMap<String, &CudaSlice<T>>]>,
    dynamic: &Bindings<'_, '_, T>,
) -> Result<(), String> {
    for input in &mut run.inputs {
        if let Some(view) = find_binding(dynamic, &input.name) {
            if invariant.is_some() {
                input.invariant = false;
            } else if input.invariant {
                return Err(format!(
                    "CUDA loop region input {:?} changed from loop-invariant to per-iteration",
                    input.name
                ));
            }
            bind_input(stream, values, free_buffers, input, view)?;
        } else if let Some(invariant) = invariant {
            let capture = find_invariant(invariant, &input.name)
                .ok_or_else(|| format!("CUDA loop region input {:?} is not bound", input.name))?;
            input.invariant = true;
            bind_input(stream, values, free_buffers, input, capture)?;
        } else if !input.invariant {
            return Err(format!(
                "CUDA loop region input {:?} is not bound",
                input.name
            ));
        }
    }
    Ok(())
}

/// Runs a region program whose inputs `bind_region_inputs` has written.
fn execute_region_program<T: CudaReal>(
    region: &CudaExecutionPlan<T>,
    stream: &Arc<CudaStream>,
    values: &mut Vec<Option<CudaSlice<T>>>,
    free_buffers: &mut CudaBufferPool<T>,
    input_names: &BTreeSet<String>,
) -> Result<(), String> {
    // The input names are passed as retained so that input buffers outlive their last
    // reader: invariant inputs are written only by a loop execution's first run.
    execute_cuda_device_program(
        &region.plan,
        &BTreeMap::new(),
        CudaProgramRuntime {
            stream,
            module: &region.module,
            blas: region.blas.as_ref(),
            solver: region.solver.as_ref(),
            cond_branches: &region.cond_branches,
            host_loops: &region.host_loops,
            region_captures: &BTreeMap::new(),
            prebound_inputs: true,
            constant_uploads: &region.constant_uploads,
        },
        values,
        free_buffers,
        input_names,
        false,
    )
    .map(|_| ())
}

fn copy_region_outputs<T: CudaReal>(
    stream: &Arc<CudaStream>,
    region: &CudaExecutionPlan<T>,
    values: &[Option<CudaSlice<T>>],
    outputs: &mut [CudaViewMut<'_, T>],
) -> Result<(), String> {
    let output_ids = region.plan.output_node_ids();
    if output_ids.len() != outputs.len() {
        return Err(format!(
            "CUDA loop region has {} outputs, the loop expected {}",
            output_ids.len(),
            outputs.len()
        ));
    }
    for (output, target) in output_ids.iter().zip(outputs.iter_mut()) {
        let source = cuda_value(values, *output)?;
        if source.len() != target.len() {
            return Err(format!(
                "CUDA loop region output node {output} has {} elements, expected {}",
                source.len(),
                target.len()
            ));
        }
        stream
            .memcpy_dtod(source, target)
            .map_err(|error| format!("failed to copy a CUDA loop region output: {error:?}"))?;
    }
    Ok(())
}

/// One region run with ordinary launches.
fn run_eager<T: CudaReal>(
    stream: &Arc<CudaStream>,
    region: &CudaExecutionPlan<T>,
    state: &mut CudaExecutionState<T>,
    run: &mut RegionRun,
    invariant: &[&BTreeMap<String, &CudaSlice<T>>],
    dynamic: &Bindings<'_, '_, T>,
    outputs: &mut [CudaViewMut<'_, T>],
) -> Result<(), String> {
    let CudaExecutionState {
        values,
        free_buffers,
        ..
    } = state;
    if values.len() != region.plan.nodes.len() {
        *values = std::iter::repeat_with(|| None)
            .take(region.plan.nodes.len())
            .collect();
    }
    recycle_cuda_computed_values(&region.plan, values, free_buffers);
    let first = (run.runs == 1).then_some(invariant);
    bind_region_inputs(stream, run, values, free_buffers, first, dynamic)?;
    execute_region_program(region, stream, values, free_buffers, &run.input_names)?;
    copy_region_outputs(stream, region, values, outputs)
}

/// Device address of a buffer, for graph copy parameters.
fn device_address<T, S: DevicePtr<T>>(buffer: &S, stream: &CudaStream) -> sys::CUdeviceptr {
    // Plan buffers carry no events (event tracking is disabled on plan contexts), so the
    // returned guard records nothing.
    buffer.device_ptr(stream).0
}

/// A recorded device-to-device copy whose addresses change between replays.
struct GraphCopy {
    node: sys::CUgraphNode,
    params: sys::CUDA_MEMCPY3D,
    count: usize,
}

/// One recorded region run: the copies of the per-iteration inputs into the
/// region's input buffers, the region program, and the copies of its outputs
/// into the run's result buffers.
///
/// The program's own buffers (value table and pool) stay fixed while the loop
/// executes: replays only retarget the input sources and output destinations.
/// The graph is launched on the default stream, ordered after the eager work
/// that produced the inputs and before the work that reads the outputs.
struct RegionGraph {
    graph: CudaGraph,
    bindings: Vec<(String, GraphCopy)>,
    outputs: Vec<GraphCopy>,
}

impl GraphCopy {
    /// Points the copy at new addresses, updating the executable graph only
    /// when one changed.
    fn retarget(
        &mut self,
        exec: sys::CUgraphExec,
        context: sys::CUcontext,
        source: Option<sys::CUdeviceptr>,
        destination: Option<sys::CUdeviceptr>,
    ) -> Result<(), String> {
        let mut params = self.params;
        if let Some(source) = source {
            params.srcDevice = source;
            params.srcXInBytes = 0;
        }
        if let Some(destination) = destination {
            params.dstDevice = destination;
            params.dstXInBytes = 0;
        }
        if params == self.params {
            return Ok(());
        }
        // SAFETY: `exec` is the live executable instantiated from the graph that owns `node`,
        // a 1D device-to-device copy node of `count` elements; only its addresses change, to
        // buffers of the same context with at least `count` elements (checked by the callers).
        unsafe { sys::cuGraphExecMemcpyNodeSetParams(exec, self.node, &params, context) }
            .result()
            .map_err(|error| format!("failed to retarget a CUDA loop graph copy: {error:?}"))?;
        self.params = params;
        Ok(())
    }
}

/// The device-to-device copy nodes of a recorded graph with their parameters.
fn graph_copies(graph: &CudaGraph) -> Result<Vec<(sys::CUgraphNode, sys::CUDA_MEMCPY3D)>, String> {
    let handle = graph.cu_graph();
    let error =
        |error: result::DriverError| format!("failed to inspect a CUDA loop graph: {error:?}");
    let mut count = 0usize;
    // SAFETY: `handle` is the live graph owned by `graph`; a null array queries the count.
    unsafe { sys::cuGraphGetNodes(handle, std::ptr::null_mut(), &mut count) }
        .result()
        .map_err(error)?;
    let mut nodes = vec![std::ptr::null_mut(); count];
    // SAFETY: `nodes` has room for the `count` node handles the driver writes.
    unsafe { sys::cuGraphGetNodes(handle, nodes.as_mut_ptr(), &mut count) }
        .result()
        .map_err(error)?;
    nodes.truncate(count);
    let mut copies = Vec::new();
    for node in nodes {
        let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        // SAFETY: `node` belongs to the live graph `handle`.
        unsafe { sys::cuGraphNodeGetType(node, &mut kind) }
            .result()
            .map_err(error)?;
        if kind != sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY {
            continue;
        }
        let mut params = std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::uninit();
        // SAFETY: `node` is a memcpy node of the live graph and `params` is writable.
        unsafe { sys::cuGraphMemcpyNodeGetParams(node, params.as_mut_ptr()) }
            .result()
            .map_err(error)?;
        // SAFETY: on success the driver has written the node's complete parameter struct,
        // with valid memory-type values.
        copies.push((node, unsafe { params.assume_init() }));
    }
    Ok(copies)
}

/// Finds the recorded copy from `source` to `destination`.
fn take_graph_copy(
    copies: &mut Vec<(sys::CUgraphNode, sys::CUDA_MEMCPY3D)>,
    source: sys::CUdeviceptr,
    destination: sys::CUdeviceptr,
    count: usize,
) -> Result<GraphCopy, String> {
    let position = copies
        .iter()
        .position(|(_, params)| {
            params.srcDevice + params.srcXInBytes as sys::CUdeviceptr == source
                && params.dstDevice + params.dstXInBytes as sys::CUdeviceptr == destination
        })
        .ok_or_else(|| "a CUDA loop graph lacks a recorded binding copy".to_string())?;
    let (node, params) = copies.swap_remove(position);
    Ok(GraphCopy {
        node,
        params,
        count,
    })
}

/// Points the region's cuBLAS handle, if any, at `stream`.
fn set_blas_stream<T: CudaReal>(
    region: &CudaExecutionPlan<T>,
    stream: &Arc<CudaStream>,
) -> Result<(), String> {
    let Some(blas) = &region.blas else {
        return Ok(());
    };
    let mut blas = blas
        .lock()
        .map_err(|_| "CUDA loop region cuBLAS lock is poisoned".to_string())?;
    // SAFETY: the handle is owned by this region's plan and only used by its program, which runs
    // under the region state lock held by the caller. While it points at the capture stream,
    // work is only recorded, never executed, so no two streams run its kernels concurrently.
    unsafe { blas.set_stream(stream.clone()) }
        .map_err(|error| format!("failed to set the CUDA loop region cuBLAS stream: {error:?}"))
}

/// Records one region run on `capture` (which executes nothing) and returns
/// the graph; the caller then launches it for this run.
///
/// The recording repeats an eager run without the invariant bindings: it
/// returns the computed buffers to the region pool and takes them again in the
/// same order, so in the steady state after the eager runs nothing is
/// allocated. The pool refuses to allocate or free while `capturing` is set,
/// which turns any deviation into an error and the region back to eager runs.
fn capture_region<T: CudaReal>(
    stream: &Arc<CudaStream>,
    capture: &Arc<CudaStream>,
    region: &CudaExecutionPlan<T>,
    state: &mut CudaExecutionState<T>,
    run: &mut RegionRun,
    dynamic: &Bindings<'_, '_, T>,
    outputs: &mut [CudaViewMut<'_, T>],
) -> Result<RegionGraph, String> {
    let CudaExecutionState {
        values,
        free_buffers,
        ..
    } = state;
    // cuBLAS calls are recorded only when issued on the capture stream. The handle belongs to
    // this region alone and is restored before anything executes on it again.
    set_blas_stream(region, capture)?;
    if let Err(error) =
        capture.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
    {
        set_blas_stream(region, stream)?;
        return Err(format!(
            "failed to start recording a CUDA loop region: {error:?}"
        ));
    }
    free_buffers.capturing = true;
    recycle_cuda_computed_values(&region.plan, values, free_buffers);
    let recorded = bind_region_inputs(capture, run, values, free_buffers, None, dynamic)
        .and_then(|()| {
            execute_region_program(region, capture, values, free_buffers, &run.input_names)
        })
        .and_then(|()| copy_region_outputs(capture, region, values, outputs));
    // Always leave capture mode, also after a failed recording.
    let ended = capture.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    free_buffers.capturing = false;
    set_blas_stream(region, stream)?;
    recorded?;
    let graph = ended
        .map_err(|error| format!("failed to record a CUDA loop region: {error:?}"))?
        .ok_or_else(|| "a CUDA loop region recorded an empty graph".to_string())?;
    let mut copies = graph_copies(&graph)?;
    let mut bindings = Vec::new();
    for input in run.inputs.iter().filter(|input| !input.invariant) {
        let source = find_binding(dynamic, &input.name)
            .ok_or_else(|| format!("CUDA loop region input {:?} is not bound", input.name))?;
        let destination = cuda_value(values, input.node)?;
        let copy = take_graph_copy(
            &mut copies,
            device_address(source, capture),
            device_address(destination, capture),
            input.count,
        )?;
        bindings.push((input.name.clone(), copy));
    }
    let mut output_copies = Vec::new();
    for (output, target) in region.plan.output_node_ids().iter().zip(outputs.iter()) {
        let source = cuda_value(values, *output)?;
        output_copies.push(take_graph_copy(
            &mut copies,
            device_address(source, capture),
            device_address(target, capture),
            target.len(),
        )?);
    }
    Ok(RegionGraph {
        graph,
        bindings,
        outputs: output_copies,
    })
}

impl RegionGraph {
    /// One region run: retargets the copies to this run's buffers and
    /// launches the graph on `stream`.
    fn replay<T: CudaReal>(
        &mut self,
        stream: &Arc<CudaStream>,
        dynamic: &Bindings<'_, '_, T>,
        outputs: &mut [CudaViewMut<'_, T>],
    ) -> Result<(), String> {
        let exec = self.graph.cu_graph_exec();
        let context = stream.context().cu_ctx();
        for (name, copy) in &mut self.bindings {
            let source = find_binding(dynamic, name)
                .ok_or_else(|| format!("CUDA loop region input {name:?} is not bound"))?;
            if source.len() != copy.count {
                return Err(format!(
                    "CUDA loop region input {name:?} has {} elements, expected {}",
                    source.len(),
                    copy.count
                ));
            }
            copy.retarget(exec, context, Some(device_address(source, stream)), None)?;
        }
        if outputs.len() != self.outputs.len() {
            return Err("CUDA loop region graph output count changed".to_string());
        }
        for (copy, target) in self.outputs.iter_mut().zip(outputs.iter()) {
            if target.len() != copy.count {
                return Err(format!(
                    "CUDA loop region output has {} elements, expected {}",
                    target.len(),
                    copy.count
                ));
            }
            copy.retarget(exec, context, None, Some(device_address(target, stream)))?;
        }
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| format!("failed to bind the CUDA context: {error:?}"))?;
        // SAFETY: `exec` is the live executable of `self.graph`. It reads the binding sources
        // and the region's value-table buffers and writes those and the output targets; all
        // stay allocated until the work queued on `stream` before their release has run, since
        // every release in this loop is ordered on `stream` after this launch.
        unsafe { result::graph::launch(exec, stream.cu_stream()) }
            .map_err(|error| format!("failed to launch a CUDA loop region graph: {error:?}"))?;
        #[cfg(test)]
        CUDA_LOOP_GRAPH_LAUNCHES.with(|launches| launches.set(launches.get() + 1));
        Ok(())
    }
}

/// Default stream plus the loop's private buffer pool and per-region state.
struct HostRun<'a, T: CudaReal> {
    stream: Arc<CudaStream>,
    pool: CudaBufferPool<T>,
    regions: &'a [CudaExecutionPlan<T>],
    capture_stream: Option<&'a Arc<CudaStream>>,
    runs: Vec<RegionRun>,
}

impl<'a, T: CudaReal> HostRun<'a, T> {
    fn new(host_loop: &'a CudaHostLoop<T>, stream: &Arc<CudaStream>) -> Result<Self, String> {
        let runs = host_loop
            .regions
            .iter()
            .zip(&host_loop.capturable)
            .map(|(region, capturable)| {
                RegionRun::new(region, *capturable && host_loop.capture_stream.is_some())
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            stream: stream.clone(),
            // Loop working sets are bounded by the carry tape, so a private
            // unbudgeted pool reuses every per-iteration buffer; it is freed
            // when the loop finishes.
            pool: CudaBufferPool {
                budget: usize::MAX,
                ..CudaBufferPool::default()
            },
            regions: &host_loop.regions,
            capture_stream: host_loop.capture_stream.as_ref(),
            runs,
        })
    }

    fn take(&mut self, count: usize) -> Result<CudaSlice<T>, String> {
        super::take_cuda_buffer(&self.stream, &mut self.pool, count, usize::MAX)
    }

    fn copy(&mut self, source: &CudaSlice<T>) -> Result<CudaSlice<T>, String> {
        let mut buffer = self.take(source.len())?;
        self.stream
            .memcpy_dtod(source, &mut buffer)
            .map_err(|error| format!("failed to copy a CUDA loop buffer: {error:?}"))?;
        Ok(buffer)
    }

    fn zeros(&mut self, count: usize) -> Result<CudaSlice<T>, String> {
        let mut buffer = self.take(count)?;
        self.stream
            .memset_zeros(&mut buffer)
            .map_err(|error| format!("failed to clear a CUDA loop buffer: {error:?}"))?;
        Ok(buffer)
    }

    fn recycle_all(&mut self, buffers: Vec<CudaSlice<T>>) {
        for buffer in buffers {
            self.pool.recycle(buffer);
        }
    }

    /// Runs region `region` once with `dynamic` as its per-iteration inputs
    /// and returns its outputs in buffers of this loop's pool. `invariant` is
    /// read by the first run of the loop execution only.
    fn run(
        &mut self,
        region: usize,
        invariant: &[&BTreeMap<String, &CudaSlice<T>>],
        dynamic: &Bindings<'_, '_, T>,
    ) -> Result<Vec<CudaSlice<T>>, String> {
        let regions = self.regions;
        let plan = &regions[region].plan;
        let mut outputs = plan
            .output_node_ids()
            .iter()
            .map(|output| {
                element_count(&plan.nodes[*output].shape).and_then(|count| self.take(count))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut targets = outputs
            .iter_mut()
            .map(CudaSlice::as_view_mut)
            .collect::<Vec<_>>();
        self.run_into(region, invariant, dynamic, &mut targets)?;
        drop(targets);
        Ok(outputs)
    }

    /// `run` writing the region outputs into `outputs`.
    fn run_into(
        &mut self,
        region: usize,
        invariant: &[&BTreeMap<String, &CudaSlice<T>>],
        dynamic: &Bindings<'_, '_, T>,
        outputs: &mut [CudaViewMut<'_, T>],
    ) -> Result<(), String> {
        let regions = self.regions;
        let plan = &regions[region];
        let run = &mut self.runs[region];
        let mut state = plan
            .state
            .lock()
            .map_err(|_| "CUDA loop region state lock is poisoned".to_string())?;
        run.runs += 1;
        if matches!(run.graph, GraphState::Pending) && run.runs > CUDA_GRAPH_EAGER_RUNS {
            let capture = self
                .capture_stream
                .ok_or_else(|| "CUDA loop region has no capture stream".to_string())?;
            // A failed recording leaves the region to eager runs, which are always valid;
            // the recording executed nothing, so this run is then repeated eagerly.
            let recorded = capture_region(
                &self.stream,
                capture,
                plan,
                &mut state,
                run,
                dynamic,
                outputs,
            );
            run.graph = match recorded {
                Ok(graph) => GraphState::Ready(graph),
                Err(_) => GraphState::Eager,
            };
        }
        match &mut run.graph {
            GraphState::Ready(graph) => graph.replay(&self.stream, dynamic, outputs),
            GraphState::Eager | GraphState::Pending => run_eager(
                &self.stream,
                plan,
                &mut state,
                run,
                invariant,
                dynamic,
                outputs,
            ),
        }
    }

    fn run_one(
        &mut self,
        region: usize,
        invariant: &[&BTreeMap<String, &CudaSlice<T>>],
        dynamic: &Bindings<'_, '_, T>,
    ) -> Result<CudaSlice<T>, String> {
        let mut outputs = self.run(region, invariant, dynamic)?;
        let output = outputs
            .pop()
            .ok_or_else(|| "CUDA loop region produced no output".to_string())?;
        self.recycle_all(outputs);
        Ok(output)
    }

    fn zero_accumulators(&mut self, layout: &ReverseLayout) -> Result<Vec<CudaSlice<T>>, String> {
        layout
            .accumulators
            .iter()
            .map(|(_, _, shape)| element_count(shape).and_then(|count| self.zeros(count)))
            .collect()
    }

    /// Binds the accumulators and runs one reverse iteration of region `region`.
    fn reverse_step<'n, 'v>(
        &mut self,
        region: usize,
        invariant: &[&BTreeMap<String, &CudaSlice<T>>],
        mut dynamic: Vec<(&'n str, CudaView<'v, T>)>,
        layout: &'n ReverseLayout,
        accumulators: &'v [CudaSlice<T>],
    ) -> Result<Vec<CudaSlice<T>>, String> {
        for ((_, name, _), accumulator) in layout.accumulators.iter().zip(accumulators) {
            dynamic.push((name.as_str(), accumulator.as_view()));
        }
        self.run(region, invariant, &dynamic)
    }

    /// Splits reverse outputs into the next carry cotangent state (`width`
    /// buffers, or zeros when the body ignores its carry) and accumulators.
    fn advance_reverse(
        &mut self,
        mut outputs: Vec<CudaSlice<T>>,
        layout: &ReverseLayout,
        width: usize,
        cotangent: Vec<CudaSlice<T>>,
        accumulators: Vec<CudaSlice<T>>,
    ) -> Result<(DeviceState<T>, DeviceState<T>), String> {
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
        accumulators: Vec<CudaSlice<T>>,
    ) -> Result<BTreeMap<String, CudaSlice<T>>, String> {
        Ok(layout
            .accumulators
            .iter()
            .map(|(name, _, _)| name.clone())
            .zip(accumulators)
            .collect())
    }
}

/// Device copy of the loop indices; a run binds a one-element view of it as
/// the scalar index input.
struct LoopIndex<T: CudaReal> {
    indices: CudaSlice<T>,
}

impl<T: CudaReal> LoopIndex<T> {
    fn new(host: &mut HostRun<T>, names: &LoopNames) -> Result<Self, String> {
        // Loop indices are exact in float32 up to 2^24, like the fused kernels'
        // `(float)step`.
        let indices = (names.lower..names.upper)
            .map(|index| T::from_f64(index as f64))
            .collect::<Vec<_>>();
        let indices = if indices.is_empty() {
            host.zeros(1)?
        } else {
            host.stream
                .clone_htod(&indices)
                .map_err(|error| format!("failed to upload CUDA loop indices: {error:?}"))?
        };
        Ok(Self { indices })
    }

    fn at(&self, offset: usize) -> CudaView<'_, T> {
        self.indices.slice(offset..offset + 1)
    }
}

/// The device buffers of one loop state: a carry, or a carry and its tangent,
/// or the carry-cotangent pair of a reverse pass.
type DeviceState<T> = Vec<CudaSlice<T>>;

/// The first step index of one replayed tape block and the state before each of its steps.
type ReplayBlock<T> = (usize, Vec<DeviceState<T>>);

type StepFn<'a, 'r, T> = dyn FnMut(&mut HostRun<'r, T>, usize, &[CudaSlice<T>]) -> Result<Vec<CudaSlice<T>>, String>
    + 'a;

/// Device carry tape for reverse passes, using the CPU checkpoint scheme:
/// `TensorCarryCheckpoints::block_size` decides between keeping every state
/// and keeping one state per block, replayed per block in reverse.
struct DeviceTape<T: CudaReal> {
    /// Full tape (`block == None`): the state before every step. Otherwise
    /// the state at the start of every block.
    states: Vec<Vec<CudaSlice<T>>>,
    block: Option<usize>,
    steps: usize,
}

impl<T: CudaReal> DeviceTape<T> {
    fn record<'r>(
        host: &mut HostRun<'r, T>,
        steps: usize,
        initial: Vec<CudaSlice<T>>,
        step: &mut StepFn<'_, 'r, T>,
    ) -> Result<Self, String> {
        let block = TensorCarryCheckpoints::<DynamicTensor>::block_size(steps);
        let mut states = vec![initial];
        // The latest state when it is not itself a stored checkpoint.
        let mut current: Option<Vec<CudaSlice<T>>> = None;
        // The final state is not needed by the reverse pass.
        for offset in 0..steps.saturating_sub(1) {
            let source = match &current {
                Some(state) => state.as_slice(),
                None => states
                    .last()
                    .expect("the tape starts with the initial state"),
            };
            let next = step(host, offset, source)?;
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

    fn pop_block<'r>(
        &mut self,
        host: &mut HostRun<'r, T>,
        step: &mut StepFn<'_, 'r, T>,
    ) -> Result<Option<ReplayBlock<T>>, String> {
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
            let next = step(host, offset, states.last().expect("non-empty"))?;
            states.push(next);
        }
        Ok(Some((start, states)))
    }
}
