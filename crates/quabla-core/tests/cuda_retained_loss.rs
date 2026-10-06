mod support;

use quabla_core::tensor_ir::{DynamicTensor, SymbolicCotangent, TensorIr};
use std::collections::BTreeMap;

#[test]
fn primary_loss_compilation_excludes_backward_nodes() -> Result<(), String> {
    let mut graph = TensorIr::new();
    let parameter = graph.input("parameter", vec![4])?;
    let batch = graph.input("batch", vec![4])?;
    let residual = graph.sub(parameter, batch)?;
    let squared = graph.mul(residual, residual)?;
    let loss = graph.sum(squared)?;
    let reverse = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    let gradient = reverse.gradients["parameter"];
    let (shared, _) = reverse
        .graph
        .compile_cpu_many(&[reverse.primals[loss], gradient])?;
    let forward = graph.compile_cpu(loss)?;
    assert!(forward.node_count() < shared.node_count());
    let inputs = BTreeMap::from([
        (
            "parameter".into(),
            DynamicTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0])?,
        ),
        ("batch".into(), DynamicTensor::new(vec![4], vec![0.0; 4])?),
    ]);
    assert_eq!(forward.evaluate(&inputs)?.data().as_ref(), &[30.0]);
    assert_eq!(shared.evaluate(&inputs)?.data().as_ref(), &[30.0]);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn retained_loss_uses_updated_parameters_and_latest_batch_without_adam_mutation(
) -> Result<(), String> {
    use quabla_core::tensor_ir::CudaBackend;
    use std::collections::BTreeSet;
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    let mut graph = TensorIr::new();
    let parameter = graph.input("parameter", vec![4])?;
    let batch = graph.input("batch", vec![4])?;
    let residual = graph.sub(parameter, batch)?;
    let squared = graph.mul(residual, residual)?;
    let loss = graph.sum(squared)?;
    let reverse = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    let gradient = reverse.gradients["parameter"];
    let (shared, outputs) = reverse
        .graph
        .compile_cpu_many(&[reverse.primals[loss], gradient])?;
    let candidate = CudaBackend::new(0).compile(shared.clone())?;
    let reference = CudaBackend::new(0).compile(shared)?;
    let mut inputs = BTreeMap::from([
        (
            "parameter".into(),
            DynamicTensor::new(vec![4], vec![1.0, 2.0, 3.0, 4.0])?,
        ),
        ("batch".into(), DynamicTensor::new(vec![4], vec![0.0; 4])?),
    ]);
    let retained = BTreeSet::from(["parameter".into(), "batch".into()]);
    let parameter_only = BTreeSet::from(["parameter".into()]);
    assert_eq!(
        candidate
            .execute_primary_retaining(&inputs, &retained)?
            .data()
            .as_ref(),
        &[30.0]
    );
    reference.execute_retaining_without_output(&inputs, &retained)?;
    for iteration in 0..4 {
        inputs.insert(
            "batch".into(),
            DynamicTensor::new(vec![4], vec![iteration as f64 / 4.0; 4])?,
        );
        let before = candidate.retained_input_to_host("parameter")?;
        let loss = candidate.execute_primary_retaining(&inputs, &parameter_only)?;
        assert_eq!(
            candidate.retained_input_to_host("parameter")?.data(),
            before.data()
        );
        let expected = reference.execute_retaining(&inputs, &parameter_only)?;
        assert_eq!(loss.data(), expected.data());
        let buffer_count = candidate.device_buffer_count()?;
        candidate.execute_primary_retaining(&inputs, &retained)?;
        assert_eq!(candidate.device_buffer_count()?, buffer_count);
        // With both inputs retained, this step must use the batch last supplied
        // to the forward query. Repeated loss queries must not advance Adam.
        candidate.execute_retaining_without_output(&inputs, &retained)?;
        for plan in [&candidate, &reference] {
            plan.adam_step_input_from_node("parameter", outputs[1], 0.01, 0.9, 0.999, 1e-8)?;
        }
        assert_eq!(
            candidate.retained_input_to_host("parameter")?.data(),
            reference.retained_input_to_host("parameter")?.data()
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn profile_primary_loss_against_shared_forward_backward() -> Result<(), String> {
    use quabla_core::tensor_ir::CudaBackend;
    use std::collections::BTreeSet;
    use std::time::Instant;
    if std::env::var_os("QUABLA_CUDA_PROFILE").is_none() {
        return Ok(());
    }
    let n = 65536;
    let mut graph = TensorIr::new();
    let parameter = graph.input("parameter", vec![n])?;
    let mut output = parameter;
    for _ in 0..12 {
        output = graph.tanh(output)?;
    }
    let loss = graph.sum(output)?;
    let reverse = graph.symbolic_vjp_many(&[(loss, SymbolicCotangent::Ones)])?;
    let (shared, _) = reverse
        .graph
        .compile_cpu_many(&[reverse.primals[loss], reverse.gradients["parameter"]])?;
    let plan = CudaBackend::new(0).compile(shared)?;
    let inputs = BTreeMap::from([(
        "parameter".into(),
        DynamicTensor::new(vec![n], vec![0.25; n])?,
    )]);
    let retained = BTreeSet::from(["parameter".into()]);
    let full = plan.execute_retaining(&inputs, &retained)?;
    let primary = plan.execute_primary_retaining(&inputs, &retained)?;
    assert_eq!(full.data(), primary.data());
    let mut samples = [Vec::new(), Vec::new()];
    for round in 0..15 {
        for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let start = Instant::now();
            for _ in 0..10 {
                if mode == 0 {
                    plan.execute_retaining(&inputs, &retained)?;
                } else {
                    plan.execute_primary_retaining(&inputs, &retained)?;
                }
            }
            samples[mode].push(start.elapsed().as_secs_f64() * 1000.0 / 10.0);
        }
    }
    for (mode, times) in samples.iter_mut().enumerate() {
        times.sort_by(f64::total_cmp);
        println!("{{\"case\":\"cuda_loss\",\"forward_only\":{},\"elements\":{n},\"layers\":12,\"median_ms\":{},\"samples_ms\":{:?}}}", mode == 1, times[7], times);
    }
    Ok(())
}
