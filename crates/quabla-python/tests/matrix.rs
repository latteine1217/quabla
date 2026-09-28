use quabla::PyMatrix;

#[test]
fn matrix_tracks_shape_and_row_major_values() -> Result<(), String> {
    let matrix = PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;

    assert_eq!(matrix.dims(), (2, 3));
    assert_eq!(
        matrix.to_rows(),
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
    );

    Ok(())
}

#[test]
fn matrix_matmul_computes_expected_result() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
    let b = PyMatrix::from_rows(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]])?;

    let c = a.try_matmul(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.to_rows(), vec![vec![58.0, 64.0], vec![139.0, 154.0]]);

    Ok(())
}

#[test]
fn matrix_add_computes_elementwise_sum() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = PyMatrix::from_rows(vec![vec![0.5, 1.5], vec![2.5, 3.5]])?;

    let c = a.try_add(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.to_rows(), vec![vec![1.5, 3.5], vec![5.5, 7.5]]);

    Ok(())
}

#[test]
fn matrix_sub_computes_elementwise_difference() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = PyMatrix::from_rows(vec![vec![0.5, 1.5], vec![2.5, 3.5]])?;

    let c = a.try_sub(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.to_rows(), vec![vec![0.5, 0.5], vec![0.5, 0.5]]);

    Ok(())
}

#[test]
fn matrix_mul_computes_elementwise_product() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = PyMatrix::from_rows(vec![vec![0.5, 1.5], vec![2.5, 3.5]])?;

    let c = a.try_mul(&b)?;

    assert_eq!(c.dims(), (2, 2));
    assert_eq!(c.to_rows(), vec![vec![0.5, 3.0], vec![7.5, 14.0]]);

    Ok(())
}

#[test]
fn matrix_broadcasts_row_and_column_for_elementwise_ops() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
    let row = PyMatrix::from_rows(vec![vec![10.0, 20.0, 30.0]])?;
    let column = PyMatrix::from_rows(vec![vec![2.0], vec![3.0]])?;

    assert_eq!(
        a.try_add(&row)?.to_rows(),
        vec![vec![11.0, 22.0, 33.0], vec![14.0, 25.0, 36.0]]
    );
    assert_eq!(
        a.try_mul(&column)?.to_rows(),
        vec![vec![2.0, 4.0, 6.0], vec![12.0, 15.0, 18.0]]
    );

    Ok(())
}

#[test]
fn matrix_gt_and_where_select_with_broadcast_masks() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![-1.0, 0.5, 2.0], vec![3.0, -4.0, 5.0]])?;
    let zero = PyMatrix::filled(1, 1, 0.0);

    let mask = a.try_gt(&zero)?;
    let selected = PyMatrix::try_where(&mask, &a, &zero)?;

    assert_eq!(
        mask.to_rows(),
        vec![vec![0.0, 1.0, 1.0], vec![1.0, 0.0, 1.0]]
    );
    assert_eq!(
        selected.to_rows(),
        vec![vec![0.0, 0.5, 2.0], vec![3.0, 0.0, 5.0]]
    );

    Ok(())
}

#[test]
fn matrix_sum_reduces_to_single_value() -> Result<(), String> {
    let matrix = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;

    let sum = matrix.sum();

    assert_eq!(sum.dims(), (1, 1));
    assert_eq!(sum.to_rows(), vec![vec![10.0]]);

    Ok(())
}

#[test]
fn matrix_axis_reductions_preserve_reduced_rank() -> Result<(), String> {
    let matrix = PyMatrix::from_rows(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;

    assert_eq!(matrix.sum_axis(0)?.to_rows(), vec![vec![5.0, 7.0, 9.0]]);
    assert_eq!(matrix.sum_axis(1)?.to_rows(), vec![vec![6.0], vec![15.0]]);
    assert_eq!(matrix.mean_axis(0)?.to_rows(), vec![vec![2.5, 3.5, 4.5]]);
    assert_eq!(matrix.mean_axis(1)?.to_rows(), vec![vec![2.0], vec![5.0]]);

    Ok(())
}

#[test]
fn matrix_concat_joins_rows_or_columns() -> Result<(), String> {
    let top = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let bottom = PyMatrix::from_rows(vec![vec![5.0, 6.0]])?;
    let left = PyMatrix::from_rows(vec![vec![1.0], vec![2.0]])?;
    let right = PyMatrix::from_rows(vec![vec![3.0, 4.0], vec![5.0, 6.0]])?;

    let vertical = PyMatrix::try_concat(&[top, bottom], 0)?;
    let horizontal = PyMatrix::try_concat(&[left, right], 1)?;

    assert_eq!(
        vertical.to_rows(),
        vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
    );
    assert_eq!(
        horizontal.to_rows(),
        vec![vec![1.0, 3.0, 4.0], vec![2.0, 5.0, 6.0]]
    );

    Ok(())
}

#[test]
fn matrix_rejects_ragged_rows() {
    let err = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0]]).unwrap_err();

    assert!(err.contains("ragged"));
}

#[test]
fn matrix_rejects_incompatible_matmul_shapes() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0]])?;
    let b = PyMatrix::from_rows(vec![vec![3.0, 4.0]])?;

    let err = a
        .try_matmul(&b)
        .err()
        .ok_or_else(|| "expected incompatible matmul to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn matrix_rejects_incompatible_add_shapes() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = PyMatrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0], vec![9.0, 10.0]])?;

    let err = a
        .try_add(&b)
        .err()
        .ok_or_else(|| "expected incompatible add to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn matrix_rejects_incompatible_sub_shapes() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = PyMatrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0], vec![9.0, 10.0]])?;

    let err = a
        .try_sub(&b)
        .err()
        .ok_or_else(|| "expected incompatible sub to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}

#[test]
fn matrix_rejects_incompatible_mul_shapes() -> Result<(), String> {
    let a = PyMatrix::from_rows(vec![vec![1.0, 2.0], vec![3.0, 4.0]])?;
    let b = PyMatrix::from_rows(vec![vec![5.0, 6.0], vec![7.0, 8.0], vec![9.0, 10.0]])?;

    let err = a
        .try_mul(&b)
        .err()
        .ok_or_else(|| "expected incompatible mul to fail".to_string())?;

    assert!(err.contains("incompatible"));

    Ok(())
}
