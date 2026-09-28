use quabla_core::tensor::Tensor2;

#[test]
fn tensor2_tracks_shape_in_type_and_indexes_values() {
    let tensor = Tensor2::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    assert_eq!(tensor.shape(), [2, 3]);
    assert_eq!(tensor.get(0, 2), 3.0);
    assert_eq!(tensor.get(1, 0), 4.0);
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn tensor2_matmul_returns_compile_time_output_shape() {
    let a = Tensor2::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b = Tensor2::<3, 2>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

    let c: Tensor2<2, 2> = a.matmul(&b);

    assert_eq!(c.shape(), [2, 2]);
    assert_eq!(c.to_array(), [[58.0, 64.0], [139.0, 154.0]]);
}

#[test]
fn tensor2_map_supports_elementwise_transforms_without_losing_shape() {
    let tensor = Tensor2::<2, 2>::from_array([[1.0, -2.0], [3.0, -4.0]]);

    let squared: Tensor2<2, 2> = tensor.map(|value| value * value);

    assert_eq!(squared.to_array(), [[1.0, 4.0], [9.0, 16.0]]);
}
