use quabla_core::tensor_ir::{DynamicTensor, SymbolicCotangent, TensorDType, TensorIr};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: All allocation operations delegate the unchanged pointer/layout to System.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller provides a valid allocation layout.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        pointer
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: The caller supplies the original allocation pointer and layout.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn folded_fanout_aliases_and_retained_outputs_match_node_execution() -> Result<(), String> {
    for dtype in [TensorDType::F32, TensorDType::F64] {
        let mut graph = TensorIr::new();
        let base = graph.constant(
            DynamicTensor::with_dtype(vec![4], vec![0.1, -0.0, 0.7, 1.2], dtype)?,
            false,
        );
        let input = graph.input_typed("x", vec![4], dtype)?;
        let mut chain = base;
        let mut outputs = vec![base];
        for depth in 0..40 {
            chain = graph.sin(chain)?;
            if depth % 9 == 0 {
                outputs.push(chain);
            }
        }
        let repeated = graph.add(chain, chain)?;
        let alias = graph.reshape(repeated, vec![4])?;
        let runtime = graph.add(input, chain)?;
        let duplicate = graph.sin(base)?;
        outputs.extend([chain, repeated, alias, runtime, duplicate, base]);
        let inputs = BTreeMap::from([(
            "x".into(),
            DynamicTensor::with_dtype(vec![4], vec![2.0; 4], dtype)?,
        )]);
        let expected = outputs
            .iter()
            .map(|output| graph.evaluate(*output, &inputs))
            .collect::<Result<Vec<_>, _>>()?;
        let (plan, _) = graph.compile_cpu_many(&outputs)?;
        let actual = plan.evaluate_many(&inputs)?;
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(actual.dtype(), expected.dtype());
            assert_eq!(actual.shape(), expected.shape());
            assert_eq!(
                actual
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .data()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
    Ok(())
}

#[test]
fn folding_preserves_checked_constant_errors() -> Result<(), String> {
    let mut graph = TensorIr::new();
    // A singular solve is a checked runtime error; log and division follow IEEE.
    let singular = graph.constant(DynamicTensor::filled(vec![2, 2], 0.0)?, false);
    let rhs = graph.constant(DynamicTensor::filled(vec![2, 1], 1.0)?, false);
    let invalid = graph.solve(singular, rhs)?;
    let plan = graph.compile_cpu(invalid)?;
    assert_eq!(
        plan.evaluate(&BTreeMap::new()).unwrap_err(),
        graph.evaluate(invalid, &BTreeMap::new()).unwrap_err()
    );
    Ok(())
}

#[test]
fn symbolic_pruning_keeps_unused_inputs_and_multi_vjp_primal_map() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    let unused = graph.input("unused", vec![2])?;
    let square = graph.mul(x, x)?;
    let mut dead = unused;
    for _ in 0..500 {
        dead = graph.sin(dead)?;
    }
    let inputs = BTreeMap::from([
        ("x".into(), DynamicTensor::new(vec![2], vec![2.0, 3.0])?),
        ("unused".into(), DynamicTensor::filled(vec![2], 0.5)?),
    ]);
    let jvp = graph.symbolic_jvp(square, "unused")?;
    assert!(jvp.graph.node_count() < 20);
    assert_eq!(
        jvp.graph.evaluate(jvp.tangent, &inputs)?.data().as_ref(),
        &[0.0, 0.0]
    );
    let vjp = graph.symbolic_vjp(square, "seed")?;
    assert!(vjp.graph.node_count() < 20);
    let mut seeded_inputs = inputs.clone();
    seeded_inputs.insert("seed".into(), DynamicTensor::filled(vec![2], 1.0)?);
    assert_eq!(
        vjp.graph
            .evaluate(vjp.gradients["x"], &seeded_inputs)?
            .data()
            .as_ref(),
        &[4.0, 6.0]
    );
    assert_eq!(
        vjp.graph
            .evaluate(vjp.gradients["unused"], &seeded_inputs)?
            .data()
            .as_ref(),
        &[0.0, 0.0]
    );
    let full = graph.symbolic_vjp_many(&[(square, SymbolicCotangent::Ones)])?;
    assert_eq!(full.primals.len(), graph.node_count());
    assert_eq!(
        full.graph
            .evaluate(full.primals[dead], &inputs)?
            .data()
            .as_ref(),
        graph.evaluate(dead, &inputs)?.data().as_ref()
    );
    Ok(())
}

#[test]
#[ignore = "isolated allocation measurement; run with --ignored --exact --test-threads=1"]
fn constant_chain_compile_peak_follows_live_frontier() -> Result<(), String> {
    let n = 32_768;
    for depth in [8, 128] {
        let mut graph = TensorIr::new();
        let mut output = graph.constant(DynamicTensor::filled(vec![n], 0.5)?, false);
        for _ in 0..depth {
            output = graph.sin(output)?;
        }
        let baseline = LIVE.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        let start = std::time::Instant::now();
        let plan = graph.compile_cpu(output)?;
        let elapsed = start.elapsed();
        let extra_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
        assert_eq!(plan.node_count(), 1);
        assert!(extra_peak < n * 8 * 6 + depth * 4096, "peak {extra_peak}");
        println!(
            "depth={depth} elements={n} compile_extra_peak_bytes={extra_peak} elapsed_us={}",
            elapsed.as_micros()
        );
    }
    Ok(())
}

#[test]
#[ignore = "isolated transform allocation measurement"]
fn dead_symbolic_transform_cost_probe() -> Result<(), String> {
    for depth in [8, 512] {
        let mut graph = TensorIr::new();
        let x = graph.input("x", vec![32_768])?;
        let unused = graph.input("unused", vec![32_768])?;
        let output = graph.mul(x, x)?;
        let mut dead = unused;
        for _ in 0..depth {
            dead = graph.sin(dead)?;
        }
        for mode in ["jvp", "vjp"] {
            let baseline = LIVE.load(Ordering::Relaxed);
            PEAK.store(baseline, Ordering::Relaxed);
            let start = std::time::Instant::now();
            let nodes = if mode == "jvp" {
                graph.symbolic_jvp(output, "x")?.graph.node_count()
            } else {
                graph.symbolic_vjp(output, "seed")?.graph.node_count()
            };
            let elapsed = start.elapsed();
            let extra_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
            println!("mode={mode} dead_depth={depth} transformed_nodes={nodes} extra_peak_bytes={extra_peak} elapsed_us={}", elapsed.as_micros());
        }
    }
    Ok(())
}

#[test]
#[ignore = "isolated metadata allocation and timing measurement"]
fn structural_key_compile_cost_probe() -> Result<(), String> {
    for depth in [2_000, 20_000] {
        let mut graph = TensorIr::new();
        let mut value = graph.input("x", vec![1])?;
        for _ in 0..depth {
            value = graph.sin(value)?;
        }
        let baseline = LIVE.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        let start = std::time::Instant::now();
        let plan = graph.compile_cpu(value)?;
        let elapsed = start.elapsed();
        let extra_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
        assert_eq!(plan.node_count(), depth + 1);
        println!(
            "depth={depth} compile_extra_peak_bytes={extra_peak} elapsed_us={}",
            elapsed.as_micros()
        );
    }
    Ok(())
}

#[test]
fn linear_reverse_chain_and_dead_errors_preserve_vjp_contract() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        let mut graph = TensorIr::new();
        let x = graph.input_typed("x", vec![4], dtype)?;
        let offset = graph.scalar_constant(0.25);
        let mut output = x;
        for _ in 0..128 {
            output = graph.add(output, offset)?;
        }
        let inputs = BTreeMap::from([(
            "x".into(),
            DynamicTensor::with_dtype(vec![4], vec![0.1, -0.0, 0.7, 1.2], dtype)?,
        )]);
        let expected = graph.evaluate(output, &inputs)?;
        let (value, gradients) =
            graph.value_and_vjp(output, &inputs, DynamicTensor::filled(vec![4], 1.0)?)?;
        assert_eq!(
            value.data().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(gradients["x"].data().as_ref(), &[1.0; 4]);
        let singular = graph.constant(DynamicTensor::filled(vec![1, 1], 0.0)?, false);
        let rhs = graph.constant(DynamicTensor::filled(vec![1, 1], 1.0)?, false);
        graph.solve(singular, rhs)?;
        assert!(graph
            .value_and_vjp(output, &inputs, DynamicTensor::filled(vec![4], 1.0)?)
            .is_err());
    }
    Ok(())
}

#[test]
#[ignore = "isolated reverse tape peak allocation measurement"]
fn linear_reverse_peak_probe() -> Result<(), String> {
    let n = 32_768;
    for depth in [8, 128] {
        let mut graph = TensorIr::new();
        let x = graph.input("x", vec![n])?;
        let offset = graph.scalar_constant(0.25);
        let mut output = x;
        for _ in 0..depth {
            output = graph.add(output, offset)?;
        }
        let inputs = BTreeMap::from([("x".into(), DynamicTensor::filled(vec![n], 0.125)?)]);
        let seed = DynamicTensor::filled(vec![n], 1.0)?;
        let baseline = LIVE.load(Ordering::Relaxed);
        PEAK.store(baseline, Ordering::Relaxed);
        let start = std::time::Instant::now();
        let (_, gradients) = graph.value_and_vjp(output, &inputs, seed)?;
        let elapsed = start.elapsed();
        let extra_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
        assert!(gradients["x"].data().iter().all(|v| *v == 1.0));
        println!(
            "depth={depth} reverse_extra_peak_bytes={extra_peak} elapsed_us={}",
            elapsed.as_micros()
        );
    }
    Ok(())
}

#[test]
#[ignore = "isolated complete storage conversion allocation measurement"]
fn narrow_cast_peak_probe() {
    use quabla_core::tensor_ir::HostTensorStorage;
    let n = 1_048_576;
    for (source_dtype, target_dtype) in [
        (TensorDType::F32, TensorDType::Bool),
        (TensorDType::Bool, TensorDType::F32),
    ] {
        let source = HostTensorStorage::from_f64(vec![1.0; n], source_dtype);
        for widened in [true, false] {
            let baseline = LIVE.load(Ordering::Relaxed);
            PEAK.store(baseline, Ordering::Relaxed);
            let start = std::time::Instant::now();
            let result = if widened {
                HostTensorStorage::from_f64(source.to_vec(), target_dtype)
            } else {
                source.clone().into_dtype(target_dtype)
            };
            let elapsed = start.elapsed();
            let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
            assert!(result.iter().all(|value| value == 1.0));
            println!(
                "source={source_dtype:?} target={target_dtype:?} widened={widened} peak_heap_bytes={peak} elapsed_us={}",
                elapsed.as_micros()
            );
        }
    }
}

#[test]
#[ignore = "isolated complete fanout VJP allocation and timing measurement"]
fn gradient_fanout_accumulation_probe() -> Result<(), String> {
    for dtype in [TensorDType::F64, TensorDType::F32] {
        for depth in [8, 128] {
            let n = 32768;
            let mut graph = TensorIr::new();
            let input = graph.input_typed("x", vec![n], dtype)?;
            let mut output = input;
            for _ in 0..depth {
                output = graph.add(output, input)?;
            }
            let inputs = BTreeMap::from([(
                "x".into(),
                DynamicTensor::with_dtype(vec![n], vec![0.125; n], dtype)?,
            )]);
            let seed = DynamicTensor::filled(vec![n], 1.0)?;
            let mut times = Vec::new();
            let mut peaks = Vec::new();
            let mut counts = Vec::new();
            for _ in 0..9 {
                let baseline = LIVE.load(Ordering::Relaxed);
                PEAK.store(baseline, Ordering::Relaxed);
                let before_count = ALLOCATIONS.load(Ordering::Relaxed);
                let start = std::time::Instant::now();
                let result = graph.value_and_vjp(output, &inputs, seed.clone())?;
                let elapsed = start.elapsed().as_nanos();
                let peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
                let count = ALLOCATIONS.load(Ordering::Relaxed) - before_count;
                assert!(result.1["x"]
                    .data()
                    .iter()
                    .all(|v| *v == (depth + 1) as f64));
                assert!(inputs["x"].data().iter().all(|v| *v == 0.125));
                std::hint::black_box(result);
                times.push(elapsed);
                peaks.push(peak);
                counts.push(count);
            }
            times.remove(0);
            peaks.remove(0);
            counts.remove(0);
            times.sort();
            peaks.sort();
            counts.sort();
            println!("{{\"dtype\":\"{dtype}\",\"depth\":{depth},\"median_ns\":{},\"peak_heap_bytes\":{},\"allocations\":{}}}",times[4],peaks[4],counts[4]);
        }
    }
    Ok(())
}
