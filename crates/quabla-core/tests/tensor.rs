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
fn tensor2_matmul_preserves_inner_accumulation_and_empty_shapes() {
    let a = Tensor2::<2, 4>::from_array([[1e16, 1.0, -1e16, 0.25], [-0.0, 2.0, -3.0, 4.0]]);
    let b = Tensor2::<4, 3>::from_array([
        [1.0, -1.0, 0.5],
        [1.0, 2.0, -3.0],
        [1.0, -1.0, 0.5],
        [4.0, -0.0, 2.0],
    ]);
    let actual = a.matmul(&b);
    for row in 0..2 {
        for column in 0..3 {
            let mut expected = 0.0;
            for inner in 0..4 {
                expected += a.get(row, inner) * b.get(inner, column);
            }
            assert_eq!(actual.get(row, column).to_bits(), expected.to_bits());
        }
    }
    let empty = Tensor2::<2, 0>::from_array([[], []]).matmul(&Tensor2::<0, 3>::from_array([]));
    assert!(empty
        .as_slice()
        .iter()
        .all(|value| value.to_bits() == 0.0_f64.to_bits()));
    let no_columns = a.matmul(&Tensor2::<4, 0>::from_array([[], [], [], []]));
    assert!(no_columns.as_slice().is_empty());
}

#[test]
fn tensor2_map_supports_elementwise_transforms_without_losing_shape() {
    let tensor = Tensor2::<2, 2>::from_array([[1.0, -2.0], [3.0, -4.0]]);

    let squared: Tensor2<2, 2> = tensor.map(|value| value * value);

    assert_eq!(squared.to_array(), [[1.0, 4.0], [9.0, 16.0]]);
}
