use quabla_core::tensor_ir::{DynamicTensor, TensorDType, TensorIr};
use std::collections::BTreeMap;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Every allocation delegates the unchanged pointer and layout to System.
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

fn graph(dtype: TensorDType, fallback: bool, n: usize) -> Result<(TensorIr, usize), String> {
    let mut graph = TensorIr::new();
    let matrix = graph.input_typed("matrix", vec![n, n], dtype)?;
    let rhs = graph.input_typed("rhs", vec![n, 2], dtype)?;
    graph.input_typed("unused", vec![1], dtype)?;
    graph.input_typed("flag", vec![1], TensorDType::Bool)?;
    // A harmless constant selects the general JVP without changing the solve.
    if fallback {
        graph.scalar_constant(0.0);
    }
    let output = graph.solve(matrix, rhs)?;
    Ok((graph, output))
}

fn fixture(
    dtype: TensorDType,
    matrix: Vec<f64>,
    n: usize,
) -> Result<BTreeMap<String, DynamicTensor>, String> {
    Ok(BTreeMap::from([
        (
            "matrix".into(),
            DynamicTensor::with_dtype(vec![n, n], matrix, dtype)?,
        ),
        (
            "rhs".into(),
            DynamicTensor::with_dtype(
                vec![n, 2],
                (0..2 * n).map(|i| i as f64 * 0.17 - 0.8).collect(),
                dtype,
            )?,
        ),
        (
            "unused".into(),
            DynamicTensor::with_dtype(vec![1], vec![1.0], dtype)?,
        ),
        (
            "flag".into(),
            DynamicTensor::with_dtype(vec![1], vec![1.0], TensorDType::Bool)?,
        ),
    ]))
}

fn bits(value: &DynamicTensor) -> Vec<u64> {
    value.data().iter().map(|value| value.to_bits()).collect()
}

#[test]
fn replay_matches_general_jvp_bits_and_validation() -> Result<(), String> {
    for dtype in [TensorDType::F32, TensorDType::F64] {
        for matrix in [
            vec![0.2, 4.1, -1.2, 2.7, 0.1, 1.3, 1.1, -0.9, 3.4],
            vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, -1.0, 2.0, 4.0],
            vec![1.0; 9],
            vec![f64::INFINITY, 0.1, 0.2, 0.3, 2.0, 0.5, 0.6, 0.7, 3.0],
        ] {
            let inputs = fixture(dtype, matrix, 3)?;
            let mut tangents = fixture(dtype, vec![0.1; 9], 3)?;
            tangents.remove("flag");
            let (direct, output) = graph(dtype, false, 3)?;
            let (general, general_output) = graph(dtype, true, 3)?;
            for missing in ["matrix", "rhs", "unused", "flag"] {
                let mut invalid_inputs = inputs.clone();
                invalid_inputs.remove(missing);
                assert_eq!(
                    direct.jvp(output, &invalid_inputs, &tangents).unwrap_err(),
                    general
                        .jvp(general_output, &invalid_inputs, &tangents)
                        .unwrap_err()
                );
            }
            for failure in 0..5 {
                let mut selected = tangents.clone();
                match failure {
                    1 => {
                        selected.remove("unused");
                    }
                    2 => {
                        selected
                            .insert("unused".into(), DynamicTensor::new(vec![2], vec![0.0; 2])?);
                    }
                    3 => {
                        selected.insert("flag".into(), DynamicTensor::new(vec![1], vec![1.0])?);
                    }
                    4 => {
                        selected.insert(
                            "rhs".into(),
                            DynamicTensor::with_dtype(vec![3, 2], vec![f64::INFINITY; 6], dtype)?,
                        );
                    }
                    _ => {}
                }
                match (
                    direct.jvp(output, &inputs, &selected),
                    general.jvp(general_output, &inputs, &selected),
                ) {
                    (Ok((value, tangent)), Ok((expected_value, expected_tangent))) => {
                        assert_eq!(bits(&value), bits(&expected_value));
                        assert_eq!(bits(&tangent), bits(&expected_tangent));
                        assert_eq!(tangent.dtype(), expected_tangent.dtype());
                    }
                    (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                    pair => panic!("different outcomes: {pair:?}"),
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "isolated complete JVP timing probe"]
fn profile_direct_solve_jvp() -> Result<(), String> {
    use std::time::Instant;
    for n in [8, 64, 192] {
        let matrix = (0..n * n)
            .map(|i| {
                if i / n == i % n {
                    n as f64 + 1.0
                } else {
                    (i % 13) as f64 * 0.01 + 0.01
                }
            })
            .collect();
        let inputs = fixture(TensorDType::F64, matrix, n)?;
        let mut tangents = fixture(TensorDType::F64, vec![0.01; n * n], n)?;
        tangents.remove("flag");
        let modes = if std::env::var_os("QUABLA_SOLVE_REPLAY_FIRST").is_some() {
            [false, true]
        } else {
            [true, false]
        };
        for fallback in modes {
            let (graph, output) = graph(TensorDType::F64, fallback, n)?;
            let mut times = Vec::new();
            let mut peaks = Vec::new();
            for _ in 0..15 {
                let baseline = LIVE.load(Ordering::Relaxed);
                PEAK.store(baseline, Ordering::Relaxed);
                let start = Instant::now();
                std::hint::black_box(graph.jvp(output, &inputs, &tangents)?);
                times.push(start.elapsed().as_nanos());
                peaks.push(PEAK.load(Ordering::Relaxed) - baseline);
            }
            times.sort();
            peaks.sort();
            println!(
                "{{\"n\":{n},\"fallback\":{fallback},\"median_ns\":{},\"peak_heap_bytes\":{}}}",
                times[7], peaks[7]
            );
        }
    }
    Ok(())
}
