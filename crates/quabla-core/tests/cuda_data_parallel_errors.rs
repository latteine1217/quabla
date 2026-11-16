use quabla_core::compiler::QuablaCompileError;
use quabla_core::tensor_ir::{CudaBackend, TensorExecutionPlan, TensorIr};

fn plan() -> Result<TensorExecutionPlan, String> {
    let mut graph = TensorIr::new();
    let x = graph.input("x", vec![2])?;
    graph.compile_cpu(x)
}

#[cfg(not(all(feature = "cuda-nccl", target_os = "linux")))]
#[test]
fn unavailable_compilation_preserves_the_legacy_diagnostic() -> Result<(), String> {
    let backend = CudaBackend::new(0);
    let checked = backend.compile_data_parallel_checked(plan()?, vec![0, 1]);
    let Err(QuablaCompileError::Unavailable(message)) = checked else {
        return Err(format!("expected unavailable compilation, got {checked:?}"));
    };
    let legacy = backend.compile_data_parallel(plan()?, vec![0, 1]);
    assert_eq!(legacy.err(), Some(message));
    Ok(())
}

#[cfg(all(feature = "cuda-nccl", target_os = "linux"))]
#[test]
fn invalid_device_selection_is_not_an_availability_failure() -> Result<(), String> {
    // These selections are rejected before any driver or NCCL probe.
    let backend = CudaBackend::new(0);
    for devices in [vec![0], vec![0, 0]] {
        let checked = backend.compile_data_parallel_checked(plan()?, devices.clone());
        let Err(QuablaCompileError::InvalidProgram(message)) = checked else {
            return Err(format!("expected invalid selection, got {checked:?}"));
        };
        let legacy = backend.compile_data_parallel(plan()?, devices);
        assert_eq!(legacy.err(), Some(message));
    }
    Ok(())
}
