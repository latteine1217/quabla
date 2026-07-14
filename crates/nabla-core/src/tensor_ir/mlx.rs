use std::collections::BTreeMap;

use mlx_rs::{ops, Array, StreamOrDevice};

use super::{
    sqrt_derivative_coefficient, DynamicTensor, TensorBackend, TensorExecutionPlan, TensorOp,
};

/// Apple MLX backend for the supported rank-N Tensor IR primitives.
///
/// MLX uses unified memory on Apple silicon. This backend keeps intermediate
/// arrays on MLX's GPU stream and only materializes the final value for Nabla's
/// host-facing `DynamicTensor` result.
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxBackend;

impl TensorBackend for MlxBackend {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        let stream = StreamOrDevice::gpu();
        let mut values = Vec::with_capacity(plan.nodes.len());

        for (node_id, node) in plan.nodes.iter().enumerate() {
            let value = match &node.op {
                TensorOp::Input { name } => {
                    let input = inputs
                        .get(name)
                        .ok_or_else(|| format!("missing input {name:?}"))?;
                    if input.shape() != node.shape {
                        return Err(format!(
                            "input {name:?} has shape {:?}, expected {:?}",
                            input.shape(),
                            node.shape
                        ));
                    }
                    let shape = mlx_shape(&node.shape)?;
                    let data = input
                        .data()
                        .iter()
                        .map(|value| *value as f32)
                        .collect::<Vec<_>>();
                    Ok(Array::from_slice(&data, &shape))
                }
                TensorOp::ScalarConstant { value } if value.is_finite() => {
                    Ok(Array::from_f32(*value as f32))
                }
                TensorOp::ScalarConstant { .. } => {
                    return Err("MLX backend does not support non-finite constants".to_string())
                }
                TensorOp::Add { lhs, rhs } => mlx_value(&values, *lhs)?
                    .add_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Sub { lhs, rhs } => mlx_value(&values, *lhs)?
                    .subtract_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Div { lhs, rhs } => mlx_value(&values, *lhs)?
                    .divide_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Mul { lhs, rhs } => mlx_value(&values, *lhs)?
                    .multiply_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Greater { lhs, rhs } => mlx_value(&values, *lhs)?
                    .gt_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => ops::r#where_device(
                    mlx_value(&values, *condition)?,
                    mlx_value(&values, *on_true)?,
                    mlx_value(&values, *on_false)?,
                    &stream,
                )
                .map_err(|error| error.to_string()),
                TensorOp::Tanh { input } => ops::tanh_device(mlx_value(&values, *input)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Exp { input } => mlx_value(&values, *input)?
                    .exp_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Sqrt { input } => mlx_value(&values, *input)?
                    .sqrt_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::SqrtDerivative { input, order } => {
                    let input = mlx_value(&values, *input)?;
                    let zero = Array::from_f32(0.0);
                    let exponent = Array::from_f32(0.5 - *order as f32);
                    let coefficient = Array::from_f32(sqrt_derivative_coefficient(*order) as f32);
                    let power = input
                        .power_device(&exponent, &stream)
                        .map_err(|error| error.to_string())?;
                    let scaled = power
                        .multiply_device(&coefficient, &stream)
                        .map_err(|error| error.to_string())?;
                    let positive = input
                        .gt_device(&zero, &stream)
                        .map_err(|error| error.to_string())?;
                    ops::r#where_device(&positive, &scaled, &zero, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Sin { input } => mlx_value(&values, *input)?
                    .sin_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Cos { input } => mlx_value(&values, *input)?
                    .cos_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Log { input } => mlx_value(&values, *input)?
                    .log_device(&stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Powi { input, exponent } => {
                    let exponent = Array::from_f32(*exponent as f32);
                    mlx_value(&values, *input)?
                        .power_device(&exponent, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Matmul { lhs, rhs } => mlx_value(&values, *lhs)?
                    .matmul_device(mlx_value(&values, *rhs)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Sum { input } => mlx_value(&values, *input)?
                    .sum_device(None, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Mean { input } => mlx_value(&values, *input)?
                    .mean_device(None, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::SumAxis { input, axis } => mlx_value(&values, *input)?
                    .sum_axis_device(
                        i32::try_from(*axis).map_err(|_| "MLX sum axis exceeds i32".to_string())?,
                        None,
                        &stream,
                    )
                    .map_err(|error| error.to_string()),
                TensorOp::MeanAxis { input, axis } => mlx_value(&values, *input)?
                    .mean_axis_device(
                        i32::try_from(*axis)
                            .map_err(|_| "MLX mean axis exceeds i32".to_string())?,
                        None,
                        &stream,
                    )
                    .map_err(|error| error.to_string()),
                TensorOp::Reshape { input } => mlx_value(&values, *input)?
                    .reshape_device(&mlx_shape(&node.shape)?, &stream)
                    .map_err(|error| error.to_string()),
                TensorOp::Transpose { input, axes } => {
                    let axes = axes
                        .iter()
                        .map(|axis| {
                            i32::try_from(*axis)
                                .map_err(|_| "MLX transpose axis exceeds i32".to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    mlx_value(&values, *input)?
                        .transpose_axes_device(&axes, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Concat { inputs, axis } => {
                    let axis = i32::try_from(*axis)
                        .map_err(|_| "MLX concat axis exceeds i32".to_string())?;
                    let arrays = inputs
                        .iter()
                        .map(|input| mlx_value(&values, *input))
                        .collect::<Result<Vec<_>, _>>()?;
                    ops::concatenate_axis_device(&arrays, axis, &stream)
                        .map_err(|error| error.to_string())
                }
                TensorOp::Broadcast { input } => ops::broadcast_to_device(
                    mlx_value(&values, *input)?,
                    &mlx_shape(&node.shape)?,
                    &stream,
                )
                .map_err(|error| error.to_string()),
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => {
                    let start = i32::try_from(*start)
                        .map_err(|_| "MLX slice start exceeds i32".to_string())?;
                    let length = i32::try_from(*length)
                        .map_err(|_| "MLX slice length exceeds i32".to_string())?;
                    let indices = (start
                        ..start
                            .checked_add(length)
                            .ok_or_else(|| "MLX slice index range overflows i32".to_string())?)
                        .collect::<Vec<_>>();
                    let indices = Array::from_slice(&indices, &[length]);
                    mlx_value(&values, *input)?
                        .take_axis_device(
                            &indices,
                            i32::try_from(*axis)
                                .map_err(|_| "MLX slice axis exceeds i32".to_string())?,
                            &stream,
                        )
                        .map_err(|error| error.to_string())
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let rank = node.shape.len();
                    let mut widths = vec![(0_i32, 0_i32); rank];
                    let before = i32::try_from(*start)
                        .map_err(|_| "MLX pad start exceeds i32".to_string())?;
                    let after = node.shape[*axis]
                        .checked_sub(start + plan.nodes[*input].shape[*axis])
                        .ok_or_else(|| "MLX pad extent underflows".to_string())?;
                    widths[*axis] = (
                        before,
                        i32::try_from(after)
                            .map_err(|_| "MLX pad extent exceeds i32".to_string())?,
                    );
                    ops::pad_device(
                        mlx_value(&values, *input)?,
                        widths.as_slice(),
                        None,
                        None,
                        &stream,
                    )
                    .map_err(|error| error.to_string())
                }
            }
            .map_err(|error| {
                format!(
                    "MLX node {node_id} {} failed: {error}",
                    mlx_op_name(&node.op)
                )
            })?;
            values.push(value);
        }

        let output = mlx_value(&values, plan.output_node_id)?;
        output
            .eval()
            .map_err(|error| format!("MLX output evaluation failed: {error}"))?;
        DynamicTensor::new(
            plan.output_shape()?,
            output
                .as_slice::<f32>()
                .iter()
                .copied()
                .map(f64::from)
                .collect(),
        )
    }
}

fn mlx_shape(shape: &[usize]) -> Result<Vec<i32>, String> {
    shape
        .iter()
        .map(|extent| {
            i32::try_from(*extent).map_err(|_| "MLX shape extent exceeds i32".to_string())
        })
        .collect()
}

fn mlx_value(values: &[Array], node_id: usize) -> Result<&Array, String> {
    values
        .get(node_id)
        .ok_or_else(|| format!("MLX operand node {node_id} is missing"))
}

fn mlx_op_name(op: &TensorOp) -> &'static str {
    match op {
        TensorOp::Input { .. } => "input",
        TensorOp::ScalarConstant { .. } => "constant",
        TensorOp::Add { .. } => "add",
        TensorOp::Sub { .. } => "sub",
        TensorOp::Div { .. } => "div",
        TensorOp::Mul { .. } => "mul",
        TensorOp::Greater { .. } => "greater",
        TensorOp::Where { .. } => "where",
        TensorOp::Sum { .. } => "sum",
        TensorOp::SumAxis { .. } => "sum_axis",
        TensorOp::Matmul { .. } => "matmul",
        TensorOp::Tanh { .. } => "tanh",
        TensorOp::Exp { .. } => "exp",
        TensorOp::Sqrt { .. } => "sqrt",
        TensorOp::SqrtDerivative { .. } => "sqrt_derivative",
        TensorOp::Reshape { .. } => "reshape",
        TensorOp::Mean { .. } => "mean",
        TensorOp::MeanAxis { .. } => "mean_axis",
        TensorOp::Sin { .. } => "sin",
        TensorOp::Cos { .. } => "cos",
        TensorOp::Powi { .. } => "powi",
        TensorOp::Transpose { .. } => "transpose",
        TensorOp::Log { .. } => "log",
        TensorOp::Concat { .. } => "concat",
        TensorOp::Slice { .. } => "slice",
        TensorOp::PadSlice { .. } => "pad_slice",
        TensorOp::Broadcast { .. } => "broadcast",
    }
}
