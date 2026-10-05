use quabla_core::tensor_ir::{DynamicTensor, TensorIr, TensorNodeId};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Allocation and deallocation forward the original pointer/layout to System.
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
        // SAFETY: The caller supplies the original pointer and allocation layout.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[allow(clippy::needless_range_loop)] // Triangular recurrence references earlier rows.
fn reference_factor(
    graph: &mut TensorIr,
    input: TensorNodeId,
    n: usize,
) -> Result<TensorNodeId, String> {
    let mut rows: Vec<Vec<TensorNodeId>> = Vec::with_capacity(n);
    for row in 0..n {
        let mut current = Vec::with_capacity(n);
        for column in 0..n {
            if column > row {
                current.push(graph.scalar_constant(0.0));
                continue;
            }
            let entry = graph.slice_axis(input, 0, row, row + 1)?;
            let entry = graph.slice_axis(entry, 1, column, column + 1)?;
            let mut reduced = graph.reshape(entry, vec![])?;
            for inner in 0..column {
                let other = if row == column {
                    current[inner]
                } else {
                    rows[column][inner]
                };
                let product = graph.mul(current[inner], other)?;
                reduced = graph.sub(reduced, product)?;
            }
            current.push(if row == column {
                graph.sqrt(reduced)?
            } else {
                graph.div(reduced, rows[column][column])?
            });
        }
        rows.push(current);
    }
    let mut output_rows = Vec::with_capacity(n);
    for row in rows {
        let entries = row
            .into_iter()
            .map(|entry| graph.reshape(entry, vec![1]))
            .collect::<Result<Vec<_>, _>>()?;
        let row = graph.concat(entries, 0)?;
        output_rows.push(graph.reshape(row, vec![1, n])?);
    }
    graph.concat(output_rows, 0)
}

fn workload(
    mode: &str,
    kind: &str,
    n: usize,
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<(DynamicTensor, usize), String> {
    let mut graph = TensorIr::new();
    let input = graph.input("matrix", vec![n, n])?;
    let output = if mode == "native" {
        graph.cholesky(input)?
    } else {
        reference_factor(&mut graph, input, n)?
    };
    let (graph, output) = match kind {
        "primal" => (graph, output),
        "jvp" => {
            let derivative = graph.symbolic_jvp_with_tangent_inputs(
                output,
                &BTreeMap::from([("matrix".into(), "direction".into())]),
            )?;
            (derivative.graph, derivative.tangent)
        }
        "vjp" => {
            let derivative = graph.symbolic_vjp(output, "cotangent")?;
            let output = derivative.gradients["matrix"];
            (derivative.graph, output)
        }
        "hvp" => {
            let derivative = graph.symbolic_vjp(output, "cotangent")?;
            let gradient = derivative.gradients["matrix"];
            let second = derivative.graph.symbolic_jvp_with_tangent_inputs(
                gradient,
                &BTreeMap::from([("matrix".into(), "direction".into())]),
            )?;
            (second.graph, second.tangent)
        }
        _ => return Err("unknown Cholesky profile kind".into()),
    };
    let plan = graph.compile_cpu(output)?;
    let nodes = plan.node_count();
    Ok((plan.evaluate(inputs)?, nodes))
}

fn inputs(n: usize) -> Result<BTreeMap<String, DynamicTensor>, String> {
    Ok(BTreeMap::from([
        (
            "matrix".into(),
            DynamicTensor::new(
                vec![n, n],
                (0..n * n)
                    .map(|i| {
                        if i / n == i % n {
                            4.0 + (i / n) as f64
                        } else {
                            0.07
                        }
                    })
                    .collect(),
            )?,
        ),
        (
            "direction".into(),
            DynamicTensor::new(
                vec![n, n],
                (0..n * n).map(|i| (i % 7) as f64 * 0.003 - 0.008).collect(),
            )?,
        ),
        (
            "cotangent".into(),
            DynamicTensor::new(
                vec![n, n],
                (0..n * n).map(|i| 0.03 + (i % 11) as f64 * 0.017).collect(),
            )?,
        ),
    ]))
}

#[test]
fn complete_native_cholesky_profiles_preserve_results() -> Result<(), String> {
    for n in [2, 4, 8] {
        let inputs = inputs(n)?;
        for kind in ["primal", "jvp", "vjp", "hvp"] {
            let (actual, _) = workload("native", kind, n, &inputs)?;
            let (expected, _) = workload("reference", kind, n, &inputs)?;
            for (actual, expected) in actual.data().iter().zip(expected.data().iter()) {
                assert!((actual - expected).abs() <= 1e-12 + 1e-12 * expected.abs());
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "isolated complete construction/compile/execution heap and timing profile"]
fn complete_cholesky_memory_probe() -> Result<(), String> {
    let mode = std::env::var("QUABLA_CHOLESKY_PROFILE_MODE").unwrap_or_else(|_| "native".into());
    assert!(matches!(mode.as_str(), "native" | "reference"));
    for n in [8, 16, 32] {
        let inputs = inputs(n)?;
        for kind in ["primal", "jvp", "vjp", "hvp"] {
            // Warm the allocator and instruction/data caches, including compilation.
            let (warm, _) = workload(&mode, kind, n, &inputs)?;
            let checksum = warm.data().iter().sum::<f64>();
            drop(warm);
            let mut timings = Vec::new();
            let mut peak = 0;
            let mut nodes = 0;
            for _ in 0..3 {
                let baseline = LIVE.load(Ordering::Relaxed);
                PEAK.store(baseline, Ordering::Relaxed);
                let start = Instant::now();
                let (result, count) = workload(&mode, kind, n, &inputs)?;
                let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
                let observed_peak = PEAK.load(Ordering::Relaxed).saturating_sub(baseline);
                assert!(
                    (result.data().iter().sum::<f64>() - checksum).abs()
                        <= 1e-12 + 1e-12 * checksum.abs()
                );
                drop(result);
                timings.push(elapsed);
                peak = peak.max(observed_peak);
                nodes = count;
            }
            timings.sort_by(f64::total_cmp);
            println!("{{\"mode\":\"{mode}\",\"kind\":\"{kind}\",\"n\":{n},\"median_ms\":{},\"peak_heap_bytes\":{peak},\"compiled_nodes\":{nodes},\"checksum\":{checksum}}}", timings[1]);
        }
    }
    Ok(())
}
