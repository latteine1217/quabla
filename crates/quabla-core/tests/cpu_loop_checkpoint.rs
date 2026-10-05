use quabla_core::tensor_ir::{
    DynamicTensor, TensorDType, TensorExecutionPlan, TensorForiExecutionPlan,
    TensorForiVjpJvpExecutionPlan, TensorIr, TensorScanExecutionPlan,
    TensorScanVjpJvpExecutionPlan,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Every allocation delegates the unchanged layout and pointer to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies a valid layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: The caller supplies the original pointer and layout.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

type Gradients = (
    DynamicTensor,
    DynamicTensor,
    BTreeMap<String, DynamicTensor>,
);

fn body(dtype: TensorDType, n: usize) -> Result<TensorExecutionPlan, String> {
    let mut graph = TensorIr::new();
    let carry = graph.input_typed("c", vec![n], dtype)?;
    let index = graph.input_typed("i", vec![], dtype)?;
    let weight = graph.input_typed("w", vec![n], dtype)?;
    let weighted = graph.mul(carry, weight)?;
    let next = graph.tanh(weighted)?;
    let scale = graph.scalar_constant(0.0001);
    let index = graph.mul(index, scale)?;
    let next = graph.add(next, index)?;
    graph.compile_cpu(next)
}

fn reference(
    plan: &TensorForiExecutionPlan,
    body: &TensorExecutionPlan,
    lower: usize,
    steps: usize,
    initial: DynamicTensor,
    external: &BTreeMap<String, DynamicTensor>,
    seed: DynamicTensor,
) -> Result<Gradients, String> {
    let (output, tape) = plan.evaluate_with_tape(initial, external)?;
    assert_eq!(tape.len(), steps + 1);
    let mut cotangent = seed;
    let mut gradient = DynamicTensor::filled(external["w"].shape().to_vec(), 0.0)?;
    for offset in (0..steps).rev() {
        let mut inputs = external.clone();
        inputs.insert("c".into(), tape.carry_at(offset)?.clone());
        inputs.insert(
            "i".into(),
            DynamicTensor::new(vec![], vec![(lower + offset) as f64])?,
        );
        let (_, gradients) = body.value_and_vjp(&inputs, cotangent)?;
        cotangent = gradients["c"].clone();
        gradient = DynamicTensor::new(
            gradient.shape().to_vec(),
            gradient
                .data()
                .iter()
                .zip(gradients["w"].data().iter())
                .map(|(left, right)| left + right)
                .collect(),
        )?;
    }
    Ok((output, cotangent, BTreeMap::from([("w".into(), gradient)])))
}

fn bits(value: &DynamicTensor) -> Vec<u64> {
    value.data().iter().map(|value| value.to_bits()).collect()
}

#[test]
fn checkpoint_vjp_matches_full_tape_bits_across_block_boundaries() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for steps in [0, 1, 4, 8, 17, 64, 101] {
            let lower = 3;
            let body = body(dtype, 4)?;
            let plan = TensorForiExecutionPlan::new(lower, lower + steps, body.clone(), "c", "i")?;
            let initial = DynamicTensor::with_dtype(vec![4], vec![-0.0, 0.1, 0.5, -1.2], dtype)?;
            let external = BTreeMap::from([(
                "w".into(),
                DynamicTensor::with_dtype(vec![4], vec![0.9, 1.01, 0.7, -0.2], dtype)?,
            )]);
            let seed = DynamicTensor::filled(vec![4], 1.0)?;
            let expected = reference(
                &plan,
                &body,
                lower,
                steps,
                initial.clone(),
                &external,
                seed.clone(),
            )?;
            let actual = plan.value_and_vjp(initial, &external, seed)?;
            assert_eq!(bits(&actual.0), bits(&expected.0));
            assert_eq!(bits(&actual.1), bits(&expected.1));
            assert_eq!(bits(&actual.2["w"]), bits(&expected.2["w"]));
        }
    }
    Ok(())
}

// Solves `c * y = 1` for a one-element carry: a zero carry is a checked
// runtime error (log and division follow IEEE and no longer fail).
fn singular_solve(graph: &mut TensorIr, carry: usize) -> Result<usize, String> {
    let matrix = graph.reshape(carry, vec![1, 1])?;
    let rhs = graph.constant(DynamicTensor::filled(vec![1, 1], 1.0)?, false);
    let solution = graph.solve(matrix, rhs)?;
    graph.reshape(solution, vec![1])
}

#[test]
fn checkpoint_preserves_first_forward_failure() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let carry = graph.input("c", vec![1])?;
    let output = singular_solve(&mut graph, carry)?;
    let body = graph.compile_cpu(output)?;
    let plan = TensorForiExecutionPlan::new(0, 101, body, "c", "i")?;
    let initial = DynamicTensor::filled(vec![1], 0.0)?;
    let expected = plan
        .evaluate_with_tape(initial.clone(), &BTreeMap::new())
        .unwrap_err();
    assert_eq!(
        plan.value_and_vjp(
            initial,
            &BTreeMap::new(),
            DynamicTensor::filled(vec![1], 1.0)?
        )
        .unwrap_err(),
        expected
    );
    Ok(())
}

#[test]
fn scan_checkpoint_matches_public_full_tape_and_preserves_outputs() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for steps in [1, 4, 8, 17, 64, 101] {
            let mut graph = TensorIr::new();
            let carry = graph.input_typed("c", vec![4], dtype)?;
            let weight = graph.input_typed("w", vec![4], dtype)?;
            let product = graph.mul(carry, weight)?;
            let next = graph.tanh(product)?;
            let output = graph.sin(next)?;
            let body = graph.compile_cpu_many(&[next, output])?.0;
            let scan = TensorScanExecutionPlan::new(3, 3 + steps, body.clone(), "c", "i")?;
            let initial = DynamicTensor::with_dtype(vec![4], vec![-0.0, 0.1, 0.5, -1.2], dtype)?;
            let external = BTreeMap::from([(
                "w".into(),
                DynamicTensor::with_dtype(vec![4], vec![0.9, 1.01, 0.7, -0.2], dtype)?,
            )]);
            let seed = DynamicTensor::filled(vec![4], 1.0)?;
            let output_seed = DynamicTensor::filled(vec![steps, 4], 0.2)?;
            let (value, outputs, tape) = scan.evaluate_with_tape(initial.clone(), &external)?;
            assert_eq!(tape.len(), steps + 1);
            let mut cotangent = seed.clone();
            let mut gradient = DynamicTensor::filled(vec![4], 0.0)?;
            for offset in (0..steps).rev() {
                let mut inputs = external.clone();
                inputs.insert("c".into(), tape.carry_at(offset)?.clone());
                inputs.insert(
                    "i".into(),
                    DynamicTensor::filled(vec![], (3 + offset) as f64)?,
                );
                let (_, gradients) = body.value_and_vjp_many(
                    &inputs,
                    vec![cotangent, DynamicTensor::filled(vec![4], 0.2)?],
                )?;
                cotangent = gradients["c"].clone();
                gradient = DynamicTensor::new(
                    vec![4],
                    gradient
                        .data()
                        .iter()
                        .zip(gradients["w"].data().iter())
                        .map(|(a, b)| a + b)
                        .collect(),
                )?;
            }
            let actual = scan.value_and_vjp(initial, &external, seed, output_seed)?;
            for (actual, expected) in [
                (&actual.0, &value),
                (&actual.1, &outputs),
                (&actual.2, &cotangent),
                (&actual.3["w"], &gradient),
            ] {
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(actual.dtype(), expected.dtype());
                assert_eq!(bits(actual), bits(expected));
            }
        }
    }
    Ok(())
}

#[test]
fn scan_checkpoint_keeps_checked_output_errors_before_cotangent_validation() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let carry = graph.input("c", vec![1])?;
    let output = singular_solve(&mut graph, carry)?;
    let body = graph.compile_cpu_many(&[carry, output])?.0;
    let plan = TensorScanExecutionPlan::new(0, 17, body, "c", "i")?;
    let initial = DynamicTensor::filled(vec![1], 0.0)?;
    let expected = plan
        .evaluate_with_tape(initial.clone(), &BTreeMap::new())
        .unwrap_err();
    let actual = plan
        .value_and_vjp(
            initial,
            &BTreeMap::new(),
            DynamicTensor::filled(vec![], 1.0)?,
            DynamicTensor::filled(vec![], 1.0)?,
        )
        .unwrap_err();
    assert_eq!(actual, expected);
    Ok(())
}

fn sum_f64(a: &DynamicTensor, b: &DynamicTensor) -> Result<DynamicTensor, String> {
    DynamicTensor::new(
        a.shape().to_vec(),
        a.data()
            .iter()
            .zip(b.data().iter())
            .map(|(a, b)| a + b)
            .collect(),
    )
}

#[test]
fn higher_order_checkpoints_match_single_step_reverse_recurrence() -> Result<(), String> {
    for scan_case in [false, true] {
        for dtype in [TensorDType::F64, TensorDType::F32] {
            for steps in [0, 1, 8, 17, 64] {
                if scan_case && steps == 0 {
                    continue;
                }
                let mut graph = TensorIr::new();
                let c = graph.input_typed("c", vec![4], dtype)?;
                let w = graph.input_typed("w", vec![4], dtype)?;
                let product = graph.mul(c, w)?;
                let next = graph.tanh(product)?;
                let out = graph.sin(next)?;
                let carry_body = graph.compile_cpu(next)?;
                let body = graph.compile_cpu_many(&[next, out])?.0;
                let external = BTreeMap::from([(
                    "w".into(),
                    DynamicTensor::with_dtype(vec![4], vec![0.9, 1.01, 0.7, -0.2], dtype)?,
                )]);
                let directions =
                    BTreeMap::from([("w".into(), DynamicTensor::filled(vec![4], 0.03)?)]);
                let initial =
                    DynamicTensor::with_dtype(vec![4], vec![-0.0, 0.1, 0.5, -1.2], dtype)?;
                let initial_tangent = DynamicTensor::filled(vec![4], 0.02)?;
                let mut carry = initial.clone();
                let mut tangent = initial_tangent.clone();
                let mut states = Vec::new();
                for _ in 0..steps {
                    states.push((carry.clone(), tangent.clone()));
                    let mut inputs = external.clone();
                    inputs.insert("c".into(), carry);
                    let mut tangents = directions.clone();
                    tangents.insert("c".into(), tangent);
                    (carry, tangent) = carry_body.jvp(&inputs, &tangents)?;
                }
                let seed = DynamicTensor::filled(vec![4], 1.0)?;
                let seed_tangent = DynamicTensor::filled(vec![4], 0.01)?;
                let mut cotangent = seed.clone();
                let mut cotangent_tangent = seed_tangent.clone();
                let mut weight_gradient = DynamicTensor::filled(vec![4], 0.0)?;
                for (offset, (carry, tangent)) in states.into_iter().enumerate().rev() {
                    let (primal, directional) = if scan_case {
                        let local = TensorScanExecutionPlan::new(
                            3 + offset,
                            4 + offset,
                            body.clone(),
                            "c",
                            "i",
                        )?;
                        let derivative = TensorScanVjpJvpExecutionPlan::new(local.clone(), "test")?;
                        let output_seed = DynamicTensor::filled(vec![1, 4], 0.2)?;
                        let output_seed_tangent = DynamicTensor::filled(vec![1, 4], 0.04)?;
                        let (_, _, primal, _) = local.value_and_vjp(
                            carry.clone(),
                            &external,
                            cotangent.clone(),
                            output_seed.clone(),
                        )?;
                        let directional = derivative.jvp(
                            carry,
                            tangent,
                            &external,
                            &directions,
                            cotangent,
                            cotangent_tangent,
                            output_seed,
                            output_seed_tangent,
                        )?;
                        (primal, directional)
                    } else {
                        let local = TensorForiExecutionPlan::new(
                            3 + offset,
                            4 + offset,
                            carry_body.clone(),
                            "c",
                            "i",
                        )?;
                        let derivative = TensorForiVjpJvpExecutionPlan::new(local.clone(), "test")?;
                        let (_, primal, _) =
                            local.value_and_vjp(carry.clone(), &external, cotangent.clone())?;
                        let directional = derivative.jvp(
                            carry,
                            tangent,
                            &external,
                            &directions,
                            cotangent,
                            cotangent_tangent,
                        )?;
                        (primal, directional)
                    };
                    cotangent = primal;
                    cotangent_tangent = directional["c"].clone();
                    weight_gradient = sum_f64(&weight_gradient, &directional["w"])?;
                }
                let actual = if scan_case {
                    let local = TensorScanExecutionPlan::new(3, 3 + steps, body, "c", "i")?;
                    TensorScanVjpJvpExecutionPlan::new(local, "test")?.jvp(
                        initial,
                        initial_tangent,
                        &external,
                        &directions,
                        seed,
                        seed_tangent,
                        DynamicTensor::filled(vec![steps, 4], 0.2)?,
                        DynamicTensor::filled(vec![steps, 4], 0.04)?,
                    )?
                } else {
                    let local = TensorForiExecutionPlan::new(3, 3 + steps, carry_body, "c", "i")?;
                    TensorForiVjpJvpExecutionPlan::new(local, "test")?.jvp(
                        initial,
                        initial_tangent,
                        &external,
                        &directions,
                        seed,
                        seed_tangent,
                    )?
                };
                assert_eq!(bits(&actual["c"]), bits(&cotangent_tangent));
                assert_eq!(bits(&actual["w"]), bits(&weight_gradient));
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "isolated complete loop VJP allocation and timing probe"]
fn profile_checkpoint_vjp() -> Result<(), String> {
    for steps in [64, 512, 4096] {
        let n = 512;
        let body = body(TensorDType::F64, n)?;
        let plan = TensorForiExecutionPlan::new(3, 3 + steps, body.clone(), "c", "i")?;
        let initial = DynamicTensor::filled(vec![n], 0.1)?;
        let external = BTreeMap::from([("w".into(), DynamicTensor::filled(vec![n], 0.9)?)]);
        let seed = DynamicTensor::filled(vec![n], 1.0)?;
        let modes = if std::env::var_os("QUABLA_CHECKPOINT_FIRST").is_some() {
            [false, true]
        } else {
            [true, false]
        };
        for full_tape in modes {
            let mut times = Vec::new();
            let mut peaks = Vec::new();
            for _ in 0..7 {
                let baseline = LIVE.load(Ordering::Relaxed);
                PEAK.store(baseline, Ordering::Relaxed);
                let start = std::time::Instant::now();
                let result = if full_tape {
                    reference(
                        &plan,
                        &body,
                        3,
                        steps,
                        initial.clone(),
                        &external,
                        seed.clone(),
                    )?
                } else {
                    plan.value_and_vjp(initial.clone(), &external, seed.clone())?
                };
                std::hint::black_box(result);
                times.push(start.elapsed().as_nanos());
                peaks.push(PEAK.load(Ordering::Relaxed) - baseline);
            }
            times.sort();
            peaks.sort();
            println!("{{\"steps\":{steps},\"full_tape\":{full_tape},\"median_ns\":{},\"peak_heap_bytes\":{}}}", times[3], peaks[3]);
        }
    }
    Ok(())
}
