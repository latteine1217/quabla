#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn profile_repeated_solve_workspace() -> Result<(), String> {
    use quabla_core::tensor_ir::{CudaBackend, DynamicTensor, TensorIr};
    use std::collections::BTreeMap;
    use std::time::Instant;
    if std::env::var_os("QUABLA_CUDA_PROFILE").is_none() {
        return Ok(());
    }
    for n in [8, 64, 192] {
        let columns = 4;
        let mut graph = TensorIr::new();
        let matrix = graph.input("matrix", vec![n, n])?;
        let rhs = graph.input("rhs", vec![n, columns])?;
        let solution = graph.solve(matrix, rhs)?;
        let plan = CudaBackend::new(0).compile(graph.compile_cpu(solution)?)?;
        let mut data = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                data[row * n + column] = if row == column { 2.0 } else { 0.01 };
            }
        }
        let inputs = BTreeMap::from([
            ("matrix".into(), DynamicTensor::new(vec![n, n], data)?),
            (
                "rhs".into(),
                DynamicTensor::new(vec![n, columns], vec![1.0; n * columns])?,
            ),
        ]);
        let expected = 1.0 / (2.0 + (n - 1) as f64 * 0.01);
        for _ in 0..3 {
            let output = plan.execute(&inputs)?;
            for value in output.data().iter() {
                assert!((value - expected).abs() < 1e-5);
            }
        }
        let mut times = Vec::new();
        for _ in 0..15 {
            let start = Instant::now();
            for _ in 0..10 {
                plan.execute(&inputs)?;
            }
            times.push(start.elapsed().as_secs_f64() * 1000.0 / 10.0);
        }
        times.sort_by(f64::total_cmp);
        println!("{{\"case\":\"cuda_solve\",\"n\":{n},\"rhs_columns\":{columns},\"median_ms\":{},\"samples_ms\":{:?}}}", times[7], times);
    }
    Ok(())
}
