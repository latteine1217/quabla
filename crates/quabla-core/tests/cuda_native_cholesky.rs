#![cfg(all(feature = "cuda", target_os = "linux"))]

mod support;

use quabla_core::tensor_ir::{CudaBackend, DynamicTensor, TensorIr, TensorNodeId};
use std::collections::BTreeMap;

// Rows are built incrementally, including the diagonal absent from prior rows.
#[allow(clippy::needless_range_loop)]
fn reference(graph: &mut TensorIr, input: TensorNodeId, n: usize) -> Result<TensorNodeId, String> {
    let mut rows: Vec<Vec<TensorNodeId>> = Vec::new();
    for row in 0..n {
        let mut entries = Vec::new();
        for column in 0..n {
            if column > row {
                entries.push(graph.scalar_constant(0.0));
                continue;
            }
            let entry = graph.slice_axis(input, 0, row, row + 1)?;
            let entry = graph.slice_axis(entry, 1, column, column + 1)?;
            let mut reduced = graph.reshape(entry, vec![])?;
            for inner in 0..column {
                let rhs = if row == column {
                    entries[inner]
                } else {
                    rows[column][inner]
                };
                let product = graph.mul(entries[inner], rhs)?;
                reduced = graph.sub(reduced, product)?;
            }
            let value = if row == column {
                graph.sqrt(reduced)?
            } else {
                graph.div(reduced, rows[column][column])?
            };
            entries.push(value);
        }
        rows.push(entries);
    }
    let mut output = Vec::new();
    for row in rows {
        let entries = row
            .into_iter()
            .map(|entry| graph.reshape(entry, vec![1]))
            .collect::<Result<Vec<_>, _>>()?;
        let row = graph.concat(entries, 0)?;
        output.push(graph.reshape(row, vec![1, n])?);
    }
    graph.concat(output, 0)
}

fn assert_close(actual: &[f64], expected: &[f64], bits: bool) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "lane{i}: {a} vs {b}");
        } else if a.is_infinite() || b.is_infinite() {
            assert_eq!(a, b, "lane{i}: nonfinite classification/sign mismatch");
        } else if bits {
            assert_eq!((*a as f32).to_bits(), (*b as f32).to_bits(), "lane{i}");
        } else {
            assert!(
                a == b || (a - b).abs() <= 1e-5 + 1e-5 * b.abs(),
                "lane{i}: {a} vs {b}"
            );
        }
    }
}

#[test]
fn native_cholesky_cuda_matches_expanded_primal_and_ad() -> Result<(), String> {
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    for (batch, n) in [(1, 1), (1, 3), (1, 8), (1, 17), (2, 3)] {
        let shape = if batch == 1 {
            vec![n, n]
        } else {
            vec![batch, n, n]
        };
        let data = (0..batch * n * n)
            .map(|i| {
                let row = (i / n) % n;
                let col = i % n;
                if row == col {
                    n as f64 + 2.0
                } else if row > col {
                    ((row + col) % 5) as f64 * 0.03125
                } else {
                    999.0
                }
            })
            .collect::<Vec<_>>();
        let mut native = TensorIr::new();
        let a = native.input("a", shape.clone())?;
        let y = native.cholesky(a)?;
        let mut expanded = TensorIr::new();
        let a = expanded.input("a", shape.clone())?;
        let ref_y = if batch == 1 {
            reference(&mut expanded, a, n)?
        } else {
            let mut outputs = Vec::new();
            for b in 0..batch {
                let matrix = expanded.slice_axis(a, 0, b, b + 1)?;
                let matrix = expanded.reshape(matrix, vec![n, n])?;
                let matrix = reference(&mut expanded, matrix, n)?;
                outputs.push(expanded.reshape(matrix, vec![1, n, n])?);
            }
            expanded.concat(outputs, 0)?
        };
        let inputs = BTreeMap::from([
            ("a".to_string(), DynamicTensor::new(shape.clone(), data)?),
            (
                "d".to_string(),
                DynamicTensor::new(
                    shape.clone(),
                    (0..batch * n * n)
                        .map(|i| ((i % 7) as f64 - 3.0) * 0.0625)
                        .collect(),
                )?,
            ),
            (
                "seed".to_string(),
                DynamicTensor::new(
                    shape.clone(),
                    (0..batch * n * n)
                        .map(|i| ((i % 5) as f64 - 2.0) * 0.125)
                        .collect(),
                )?,
            ),
        ]);
        let plans = [native.compile_cpu(y)?, expanded.compile_cpu(ref_y)?];
        let results = plans
            .into_iter()
            .map(|p| CudaBackend::new(0).compile(p)?.execute(&inputs))
            .collect::<Result<Vec<_>, String>>()?;
        assert_close(&results[0].data(), &results[1].data(), false);
        for mode in ["jvp", "vjp", "mixed", "hvp"] {
            // Large expanded higher-order baselines dominate NVRTC compile time;
            // first-order tests cover n=17, while n<=8 covers all four AD kinds.
            if n > 8 && matches!(mode, "mixed" | "hvp") {
                continue;
            }
            println!("validating n={n} mode={mode}");
            let mut results = Vec::new();
            for (graph, out) in [(&native, y), (&expanded, ref_y)] {
                let directions = BTreeMap::from([("a".to_string(), "d".to_string())]);
                let (g, output) = match mode {
                    "jvp" => {
                        let t = graph.symbolic_jvp_with_tangent_inputs(out, &directions)?;
                        (t.graph, t.tangent)
                    }
                    "vjp" => {
                        let t = graph.symbolic_vjp(out, "seed")?;
                        let id = t.gradients["a"];
                        (t.graph, id)
                    }
                    "mixed" => {
                        let t = graph.symbolic_jvp_with_tangent_inputs(out, &directions)?;
                        let t = t.graph.symbolic_jvp_with_tangent_inputs(
                            t.tangent,
                            &BTreeMap::from([("a".to_string(), "second".to_string())]),
                        )?;
                        (t.graph, t.tangent)
                    }
                    _ => {
                        let t = graph.symbolic_vjp(out, "seed")?;
                        let t = t
                            .graph
                            .symbolic_jvp_with_tangent_inputs(t.gradients["a"], &directions)?;
                        (t.graph, t.tangent)
                    }
                };
                let mut all_inputs = inputs.clone();
                all_inputs.insert("second".to_string(), inputs["d"].clone());
                results.push(
                    CudaBackend::new(0)
                        .compile(g.compile_cpu(output)?)?
                        .execute(&all_inputs)?,
                );
            }
            assert_close(&results[0].data(), &results[1].data(), false);
        }
    }
    Ok(())
}

#[test]
fn native_cholesky_cuda_preserves_exceptional_primal() -> Result<(), String> {
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    for data in [
        vec![0.0, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0, 2.0],
        vec![f64::INFINITY, 0.0, 1.0, 2.0],
        vec![f64::NAN, 0.0, 1.0, 2.0],
        vec![4.0, 0.0, f64::INFINITY, 2.0],
        vec![1e-30, 0.0, 1e-30, 1.0],
    ] {
        let mut graph = TensorIr::new();
        let a = graph.input("a", vec![2, 2])?;
        let native = graph.cholesky(a)?;
        let expanded = reference(&mut graph, a, 2)?;
        let inputs = BTreeMap::from([("a".to_string(), DynamicTensor::new(vec![2, 2], data)?)]);
        let plans = [graph.compile_cpu(native)?, graph.compile_cpu(expanded)?];
        let results = plans
            .into_iter()
            .map(|p| CudaBackend::new(0).compile(p)?.execute(&inputs))
            .collect::<Result<Vec<_>, String>>()?;
        assert_close(&results[0].data(), &results[1].data(), true);
    }
    Ok(())
}

#[test]
#[ignore = "complete device operation and cold compilation profile"]
fn profile_native_cholesky_cuda() -> Result<(), String> {
    use std::time::Instant;
    if std::env::var_os("QUABLA_CUDA_PROFILE").is_none() {
        return Ok(());
    }
    let mode = std::env::var("QUABLA_CHOLESKY_PROFILE_MODE").unwrap_or_else(|_| "native".into());
    for n in [4, 16] {
        for derivative in ["primal", "jvp", "vjp"] {
            let construction_start = Instant::now();
            let mut graph = TensorIr::new();
            let input = graph.input("a", vec![n, n])?;
            let output = if mode == "native" {
                graph.cholesky(input)?
            } else {
                reference(&mut graph, input, n)?
            };
            let (graph, output) = match derivative {
                "jvp" => {
                    let t = graph.symbolic_jvp_with_tangent_inputs(
                        output,
                        &BTreeMap::from([("a".to_string(), "d".to_string())]),
                    )?;
                    (t.graph, t.tangent)
                }
                "vjp" => {
                    let t = graph.symbolic_vjp(output, "seed")?;
                    let out = t.gradients["a"];
                    (t.graph, out)
                }
                _ => (graph, output),
            };
            let plan = graph.compile_cpu(output)?;
            let nodes = plan.node_count();
            let construction_ms = construction_start.elapsed().as_secs_f64() * 1000.0;
            let compilation_start = Instant::now();
            let compiled = CudaBackend::new(0).compile(plan)?;
            let compilation_ms = compilation_start.elapsed().as_secs_f64() * 1000.0;
            let inputs = BTreeMap::from([
                (
                    "a".to_string(),
                    DynamicTensor::new(
                        vec![n, n],
                        (0..n * n)
                            .map(|i| {
                                if i / n == i % n {
                                    n as f64 + 2.0
                                } else {
                                    0.03125
                                }
                            })
                            .collect(),
                    )?,
                ),
                (
                    "d".to_string(),
                    DynamicTensor::new(vec![n, n], vec![0.125; n * n])?,
                ),
                (
                    "seed".to_string(),
                    DynamicTensor::new(vec![n, n], vec![0.25; n * n])?,
                ),
            ]);
            let mut samples = Vec::new();
            let mut checksum = 0.0;
            for _ in 0..11 {
                let start = Instant::now();
                checksum = compiled.execute(&inputs)?.data().iter().sum::<f64>();
                samples.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            samples.remove(0);
            samples.sort_by(f64::total_cmp);
            let scratch_bytes = if mode == "native" {
                match derivative {
                    "jvp" => n * n * 8,
                    "vjp" => n * n * 8,
                    _ => 0,
                }
            } else {
                0
            };
            println!("{{\"mode\":\"{mode}\",\"n\":{n},\"derivative\":\"{derivative}\",\"nodes\":{nodes},\"construction_ms\":{construction_ms},\"compilation_ms\":{compilation_ms},\"median_ms\":{},\"native_scratch_bytes\":{scratch_bytes},\"checksum\":{checksum}}}",samples[5]);
        }
    }
    Ok(())
}

#[test]
fn native_cholesky_cuda_preserves_exceptional_ad() -> Result<(), String> {
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    for data in [
        vec![0.0, 0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0, 2.0],
        vec![f64::INFINITY, 0.0, 1.0, 2.0],
        vec![f64::NAN, 0.0, 1.0, 2.0],
        vec![4.0, 0.0, f64::INFINITY, 2.0],
        vec![1e-30, 0.0, 1e-30, 1.0],
    ] {
        for mode in ["jvp", "vjp", "mixed", "hvp"] {
            let mut results = Vec::new();
            for native in [true, false] {
                let mut graph = TensorIr::new();
                let a = graph.input("a", vec![2, 2])?;
                let out = if native {
                    graph.cholesky(a)?
                } else {
                    reference(&mut graph, a, 2)?
                };
                let directions = BTreeMap::from([("a".to_string(), "d".to_string())]);
                let (g, output) = match mode {
                    "jvp" => {
                        let t = graph.symbolic_jvp_with_tangent_inputs(out, &directions)?;
                        (t.graph, t.tangent)
                    }
                    "vjp" => {
                        let t = graph.symbolic_vjp(out, "seed")?;
                        let id = t.gradients["a"];
                        (t.graph, id)
                    }
                    "mixed" => {
                        let t = graph.symbolic_jvp_with_tangent_inputs(out, &directions)?;
                        let t = t.graph.symbolic_jvp_with_tangent_inputs(
                            t.tangent,
                            &BTreeMap::from([("a".to_string(), "second".to_string())]),
                        )?;
                        (t.graph, t.tangent)
                    }
                    _ => {
                        let t = graph.symbolic_vjp(out, "seed")?;
                        let t = t
                            .graph
                            .symbolic_jvp_with_tangent_inputs(t.gradients["a"], &directions)?;
                        (t.graph, t.tangent)
                    }
                };
                let inputs = BTreeMap::from([
                    (
                        "a".to_string(),
                        DynamicTensor::new(vec![2, 2], data.clone())?,
                    ),
                    (
                        "d".to_string(),
                        DynamicTensor::new(vec![2, 2], vec![0.125; 4])?,
                    ),
                    (
                        "second".to_string(),
                        DynamicTensor::new(vec![2, 2], vec![0.25; 4])?,
                    ),
                    (
                        "seed".to_string(),
                        DynamicTensor::new(vec![2, 2], vec![0.25; 4])?,
                    ),
                ]);
                results.push(
                    CudaBackend::new(0)
                        .compile(g.compile_cpu(output)?)?
                        .execute(&inputs)?,
                );
            }
            println!("exceptional AD mode={mode} input={data:?}");
            assert_close(&results[0].data(), &results[1].data(), false);
        }
    }
    Ok(())
}

#[test]
fn native_cholesky_cuda_preserves_f32_public_dtype() -> Result<(), String> {
    use quabla_core::tensor_ir::TensorDType;
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    let mut graph = TensorIr::new();
    let input = graph.input_typed("a", vec![3, 3], TensorDType::F32)?;
    let native = graph.cholesky(input)?;
    let expanded = reference(&mut graph, input, 3)?;
    let inputs = BTreeMap::from([(
        "a".to_string(),
        DynamicTensor::with_dtype(
            vec![3, 3],
            vec![4.0, 999.0, 999.0, 0.125, 4.0, 999.0, 0.25, 0.125, 4.0],
            TensorDType::F32,
        )?,
    )]);
    let mut results = Vec::new();
    for output in [native, expanded] {
        let result = CudaBackend::new(0)
            .compile(graph.compile_cpu(output)?)?
            .execute(&inputs)?;
        assert_eq!(result.dtype(), TensorDType::F32);
        assert_eq!(result.shape(), [3, 3]);
        results.push(result);
    }
    assert_close(&results[0].data(), &results[1].data(), false);
    Ok(())
}
