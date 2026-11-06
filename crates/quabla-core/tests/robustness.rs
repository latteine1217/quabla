use quabla_core::tensor_ir::{DynamicTensor, TensorIr};

#[test]
fn invalid_slice_error_formatting_cannot_overflow() -> Result<(), String> {
    let value = DynamicTensor::new(vec![2], vec![1.0, 2.0])?;
    for axis in [0, 1, usize::MAX] {
        for start in [0, 2, usize::MAX] {
            assert!(value.slice_axis(axis, start, usize::MAX).is_err());
        }
    }
    Ok(())
}

#[test]
fn inferred_shapes_and_factory_capacity_are_checked() -> Result<(), String> {
    assert!(DynamicTensor::filled(vec![isize::MAX as usize], 0.0).is_err());
    let extent = 1usize << (usize::BITS / 2);
    let mut graph = TensorIr::new();
    let a = graph.input("a", vec![extent, 1])?;
    let b = graph.input("b", vec![1, extent])?;
    assert!(graph.add(a, b).is_err());
    assert!(graph.matmul(a, b).is_err());
    // A rejected builder operation must leave the existing graph usable.
    let output = graph.sum(a)?;
    graph.compile_cpu(output)?;
    Ok(())
}
