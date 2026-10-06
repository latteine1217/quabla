mod support;

use quabla_core::tensor_ir::TensorIr;

#[test]
fn shared_dag_cuda_source_is_linear_in_nodes() -> Result<(), String> {
    for depth in [6, 18, 40] {
        let mut ir = TensorIr::new();
        let mut output = ir.input("x", vec![8])?;
        for _ in 0..depth {
            output = ir.add(output, output)?;
        }
        let plan = ir.compile_cpu(output)?;
        let source = plan.cuda_source()?;
        assert_eq!(
            source.matches("const float quabla_value_").count(),
            plan.node_count()
        );
        assert!(source.len() < 512 * plan.node_count() + 2000);
        assert!(source.contains("output[index] = quabla_value_"));
    }
    Ok(())
}

#[test]
fn where_cuda_source_selects_shared_values_without_recursive_calls() -> Result<(), String> {
    let mut ir = TensorIr::new();
    let x = ir.input("x", vec![8])?;
    let zero = ir.scalar_constant(0.0);
    let condition = ir.greater(x, zero)?;
    let mut selected = x;
    for _ in 0..18 {
        selected = ir.add(selected, selected)?;
    }
    let otherwise = ir.sqrt(x)?;
    let output = ir.where_select(condition, selected, otherwise)?;
    let plan = ir.compile_cpu(output)?;
    let source = plan.cuda_source()?;
    assert_eq!(
        source.matches("const float quabla_value_").count(),
        plan.node_count()
    );
    assert!(source.contains("? quabla_value_"));
    assert!(!source.contains("quabla_eval_"));
    assert!(source.len() < 512 * plan.node_count() + 2000);
    Ok(())
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
#[test]
fn shared_dag_and_where_execute_with_cuda_cpu_parity() -> Result<(), String> {
    use quabla_core::tensor_ir::{CudaBackend, DynamicTensor, TensorBackend};
    use std::collections::BTreeMap;
    if !crate::support::Gate::Cuda.enabled() {
        return Ok(());
    }
    for lazy in [false, true] {
        for reduced in [false, true] {
            let mut ir = TensorIr::new();
            let x = ir.input("x", vec![8])?;
            let mut output = x;
            for _ in 0..18 {
                output = ir.add(output, output)?;
            }
            if lazy {
                let zero = ir.scalar_constant(0.0);
                let condition = ir.greater(x, zero)?;
                let minus_one = ir.scalar_constant(-1.0);
                let negated = ir.mul(x, minus_one)?;
                let negative = ir.sqrt(negated)?;
                output = ir.where_select(condition, output, negative)?;
            }
            if reduced {
                output = ir.sum(output)?;
            }
            let plan = ir.compile_cpu(output)?;
            let inputs = BTreeMap::from([(
                "x".into(),
                DynamicTensor::new(vec![8], vec![-1.0, -0.25, 0.0, 0.125, 0.5, 1.0, 2.0, 3.0])?,
            )]);
            let cpu = ir.evaluate(output, &inputs)?;
            eprintln!("CUDA shared DAG parity: lazy={lazy}, reduced={reduced}");
            let gpu = CudaBackend::new(0).execute(&plan, &inputs)?;
            for (expected, actual) in cpu.data().iter().zip(gpu.data().iter()) {
                assert!(
                    expected.is_nan() && actual.is_nan() || expected == actual,
                    "{expected} != {actual}"
                );
            }
        }
    }
    Ok(())
}
