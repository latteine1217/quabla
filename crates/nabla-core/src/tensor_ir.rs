use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
use std::collections::BTreeSet;

#[cfg(all(feature = "cuda", target_os = "linux"))]
mod cuda;

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx;

#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use cuda::{CudaBackend, CudaExecutionPlan};

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use mlx::MlxBackend;

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
#[derive(Clone, Copy, Debug, Default)]
pub struct MlxBackend;

#[cfg(not(all(feature = "mlx", target_os = "macos")))]
impl TensorBackend for MlxBackend {
    fn name(&self) -> &'static str {
        "mlx"
    }

    fn execute(
        &self,
        _plan: &TensorExecutionPlan,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        Err("MLX backend is unavailable: build Nabla on macOS with --features mlx".to_string())
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CudaBackend {
    device_ordinal: usize,
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[derive(Clone, Debug)]
pub struct CudaExecutionPlan {
    plan: TensorExecutionPlan,
    device_ordinal: usize,
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl CudaBackend {
    pub fn new(device_ordinal: usize) -> Self {
        Self { device_ordinal }
    }

    pub fn compile(&self, plan: TensorExecutionPlan) -> Result<CudaExecutionPlan, String> {
        let _ = plan;
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl CudaExecutionPlan {
    pub fn node_count(&self) -> usize {
        self.plan.node_count()
    }

    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    pub fn uses_cublas(&self) -> bool {
        false
    }

    pub fn uses_fused_matmul_bias_tanh(&self) -> bool {
        false
    }

    pub fn device_buffer_count(&self) -> Result<usize, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn synchronize(&self) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn execute(
        &self,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn execute_retaining(
        &self,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _retained_inputs: &BTreeSet<String>,
    ) -> Result<DynamicTensor, String> {
        self.execute(_inputs)
    }

    pub fn sgd_step_input_from_output(
        &self,
        _parameter_name: &str,
        _learning_rate: f32,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn execute_retaining_without_output(
        &self,
        _inputs: &BTreeMap<String, DynamicTensor>,
        _retained_inputs: &BTreeSet<String>,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn adam_step_input_from_output(
        &self,
        _parameter_name: &str,
        _learning_rate: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn adam_step_input_from_node(
        &self,
        _parameter_name: &str,
        _gradient_node_id: usize,
        _learning_rate: f32,
        _beta1: f32,
        _beta2: f32,
        _epsilon: f32,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn retained_input_to_host(&self, _name: &str) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn computed_node_to_host(&self, _node_id: usize) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }

    pub fn sync_retained_input_to(
        &self,
        _source_name: &str,
        _target: &Self,
        _target_name: &str,
    ) -> Result<(), String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
impl TensorBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn execute(
        &self,
        _plan: &TensorExecutionPlan,
        _inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        Err(format!(
            "CUDA backend is unavailable for device {}: build Nabla on Linux with --features cuda",
            self.device_ordinal
        ))
    }
}

pub type TensorNodeId = usize;

#[derive(Clone, Debug, PartialEq)]
pub struct DynamicTensor {
    shape: Vec<usize>,
    data: Vec<f64>,
}

#[derive(Clone, Debug)]
enum TensorOp {
    Input {
        name: String,
    },
    ScalarConstant {
        value: f64,
    },
    Add {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Sub {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Div {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Mul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Greater {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Where {
        condition: TensorNodeId,
        on_true: TensorNodeId,
        on_false: TensorNodeId,
    },
    Sum {
        input: TensorNodeId,
    },
    SumAxis {
        input: TensorNodeId,
        axis: usize,
    },
    Matmul {
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    },
    Tanh {
        input: TensorNodeId,
    },
    Exp {
        input: TensorNodeId,
    },
    Reshape {
        input: TensorNodeId,
    },
    Mean {
        input: TensorNodeId,
    },
    MeanAxis {
        input: TensorNodeId,
        axis: usize,
    },
    Sin {
        input: TensorNodeId,
    },
    Cos {
        input: TensorNodeId,
    },
    Powi {
        input: TensorNodeId,
        exponent: u32,
    },
    Transpose {
        input: TensorNodeId,
        axes: Vec<usize>,
    },
    Log {
        input: TensorNodeId,
    },
    Concat {
        inputs: Vec<TensorNodeId>,
        axis: usize,
    },
    Slice {
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
    },
    PadSlice {
        input: TensorNodeId,
        axis: usize,
        start: usize,
    },
    Broadcast {
        input: TensorNodeId,
    },
}

#[derive(Clone, Debug)]
struct TensorNode {
    op: TensorOp,
    shape: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct TensorIr {
    nodes: Vec<TensorNode>,
}

#[derive(Clone, Debug)]
pub struct SymbolicJvp {
    pub graph: TensorIr,
    pub value: TensorNodeId,
    pub tangent: TensorNodeId,
}

#[derive(Clone, Debug)]
pub struct SymbolicVjp {
    pub graph: TensorIr,
    pub value: TensorNodeId,
    pub cotangent: TensorNodeId,
    pub gradients: BTreeMap<String, TensorNodeId>,
}

#[derive(Clone, Debug)]
pub struct TensorExecutionPlan {
    nodes: Vec<TensorNode>,
    output_node_id: TensorNodeId,
    fused_elementwise_output: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorKernelNode {
    pub id: TensorNodeId,
    pub op: String,
    pub shape: Vec<usize>,
    pub inputs: Vec<TensorNodeId>,
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorKernelProgram {
    pub nodes: Vec<TensorKernelNode>,
    pub output_node_id: TensorNodeId,
}

impl TensorKernelProgram {
    pub fn validate(&self) -> Result<(), String> {
        for (position, node) in self.nodes.iter().enumerate() {
            if node.id != position {
                return Err(format!(
                    "kernel node id {} does not match its position {position}",
                    node.id
                ));
            }
            if node.shape.contains(&0) {
                return Err(format!("kernel node {} has a zero tensor extent", node.id));
            }
            for input in &node.inputs {
                if *input >= position {
                    return Err(format!(
                        "kernel node {} references non-dominating input {input}",
                        node.id
                    ));
                }
            }
            if node.op == "input" && node.name.is_none() {
                return Err(format!("input kernel node {} has no name", node.id));
            }
        }
        if self.output_node_id >= self.nodes.len() {
            return Err(format!(
                "kernel output node {} does not exist",
                self.output_node_id
            ));
        }
        Ok(())
    }
}

pub trait TensorBackend {
    fn name(&self) -> &'static str;

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl TensorBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn execute(
        &self,
        plan: &TensorExecutionPlan,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        plan.execute_cpu(inputs)
    }
}

#[derive(Clone)]
struct MixedTangent {
    value: DynamicTensor,
    first: DynamicTensor,
    second: DynamicTensor,
    mixed: DynamicTensor,
}

impl DynamicTensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Result<Self, String> {
        let expected = element_count(&shape)?;
        if data.len() != expected {
            return Err(format!(
                "tensor data length {} does not match shape {:?} with {expected} elements",
                data.len(),
                shape
            ));
        }

        Ok(Self { shape, data })
    }

    pub fn filled(shape: Vec<usize>, value: f64) -> Result<Self, String> {
        let count = element_count(&shape)?;
        Ok(Self {
            shape,
            data: vec![value; count],
        })
    }

    fn one_hot(shape: Vec<usize>, index: usize) -> Result<Self, String> {
        let count = element_count(&shape)?;
        if index >= count {
            return Err(format!(
                "one-hot index {index} is out of bounds for {count} elements"
            ));
        }
        let mut data = vec![0.0; count];
        data[index] = 1.0;
        Self::new(shape, data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }

    fn add(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs + rhs)
    }

    fn mul(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs * rhs)
    }

    fn sub(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| lhs - rhs)
    }

    fn neg(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| -value).collect(),
        )
    }

    fn div(&self, rhs: &Self) -> Result<Self, String> {
        if rhs.data.contains(&0.0) {
            return Err("division by zero is not supported".to_string());
        }
        self.elementwise(rhs, |lhs, rhs| lhs / rhs)
    }

    fn greater(&self, rhs: &Self) -> Result<Self, String> {
        self.elementwise(rhs, |lhs, rhs| f64::from(lhs > rhs))
    }

    fn where_select(&self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        let shape = broadcast_shape(
            &broadcast_shape(&self.shape, &on_true.shape)?,
            &on_false.shape,
        )?;
        let count = element_count(&shape)?;
        let condition_strides = contiguous_strides(&self.shape);
        let true_strides = contiguous_strides(&on_true.shape);
        let false_strides = contiguous_strides(&on_false.shape);
        let mut data = Vec::with_capacity(count);

        for index in 0..count {
            let condition_index = broadcast_offset(index, &shape, &self.shape, &condition_strides);
            let true_index = broadcast_offset(index, &shape, &on_true.shape, &true_strides);
            let false_index = broadcast_offset(index, &shape, &on_false.shape, &false_strides);
            data.push(if self.data[condition_index] != 0.0 {
                on_true.data[true_index]
            } else {
                on_false.data[false_index]
            });
        }

        Self::new(shape, data)
    }

    fn reciprocal(&self) -> Result<Self, String> {
        if self.data.contains(&0.0) {
            return Err("division by zero is not supported".to_string());
        }
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| 1.0 / value).collect(),
        )
    }

    fn scale(&self, factor: f64) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value * factor).collect(),
        )
    }

    fn sum_all(&self) -> Result<Self, String> {
        Self::new(vec![], vec![self.data.iter().sum()])
    }

    fn reduce_axis(&self, axis: usize, scale: f64) -> Result<Self, String> {
        let output_shape = reduced_shape(&self.shape, axis)?;
        let output_strides = contiguous_strides(&output_shape);
        let mut data = vec![0.0; element_count(&output_shape)?];

        for (source_index, value) in self.data.iter().enumerate() {
            let mut remaining = source_index;
            let mut output_index = 0;
            for source_axis in (0..self.shape.len()).rev() {
                let coordinate = remaining % self.shape[source_axis];
                remaining /= self.shape[source_axis];
                if source_axis != axis {
                    let output_axis = if source_axis < axis {
                        source_axis
                    } else {
                        source_axis - 1
                    };
                    output_index += coordinate * output_strides[output_axis];
                }
            }
            data[output_index] += value * scale;
        }

        Self::new(output_shape, data)
    }

    fn expand_reduced_axis(&self, target_shape: &[usize], axis: usize) -> Result<Self, String> {
        let expected_shape = reduced_shape(target_shape, axis)?;
        if self.shape != expected_shape {
            return Err(format!(
                "cannot expand reduced tensor shape {:?} along axis {axis} to {:?}",
                self.shape, target_shape
            ));
        }

        let source_strides = contiguous_strides(&self.shape);
        let mut data = Vec::with_capacity(element_count(target_shape)?);
        for target_index in 0..element_count(target_shape)? {
            let mut remaining = target_index;
            let mut source_index = 0;
            for target_axis in (0..target_shape.len()).rev() {
                let coordinate = remaining % target_shape[target_axis];
                remaining /= target_shape[target_axis];
                if target_axis != axis {
                    let source_axis = if target_axis < axis {
                        target_axis
                    } else {
                        target_axis - 1
                    };
                    source_index += coordinate * source_strides[source_axis];
                }
            }
            data.push(self.data[source_index]);
        }

        Self::new(target_shape.to_vec(), data)
    }

    fn tanh(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.tanh()).collect(),
        )
    }

    fn exp(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.exp()).collect(),
        )
    }

    fn reshape(&self, shape: Vec<usize>) -> Result<Self, String> {
        if element_count(&shape)? != self.data.len() {
            return Err(format!(
                "cannot reshape tensor with {} elements to shape {:?}",
                self.data.len(),
                shape
            ));
        }
        Self::new(shape, self.data.clone())
    }

    fn concat(inputs: &[&Self], axis: usize) -> Result<Self, String> {
        let input_shapes = inputs
            .iter()
            .map(|input| input.shape.as_slice())
            .collect::<Vec<_>>();
        let shape = concat_shape(&input_shapes, axis)?;
        let mut data = Vec::with_capacity(element_count(&shape)?);
        let outer = element_count(&shape[..axis])?;
        let inner = element_count(&shape[axis + 1..])?;
        for outer_index in 0..outer {
            for input in inputs {
                let start = outer_index * input.shape[axis] * inner;
                let end = start + input.shape[axis] * inner;
                data.extend_from_slice(&input.data[start..end]);
            }
        }
        Self::new(shape, data)
    }

    fn slice_axis(&self, axis: usize, start: usize, length: usize) -> Result<Self, String> {
        if axis >= self.shape.len()
            || start
                .checked_add(length)
                .is_none_or(|end| end > self.shape[axis])
        {
            return Err(format!(
                "invalid slice [{start}..{}) on axis {axis} for shape {:?}",
                start + length,
                self.shape
            ));
        }
        let mut shape = self.shape.clone();
        shape[axis] = length;
        let outer = element_count(&self.shape[..axis])?;
        let inner = element_count(&self.shape[axis + 1..])?;
        let mut data = Vec::with_capacity(element_count(&shape)?);
        for outer_index in 0..outer {
            let source_start = (outer_index * self.shape[axis] + start) * inner;
            let source_end = source_start + length * inner;
            data.extend_from_slice(&self.data[source_start..source_end]);
        }
        Self::new(shape, data)
    }

    fn pad_slice(&self, output_shape: &[usize], axis: usize, start: usize) -> Result<Self, String> {
        if axis >= output_shape.len() || self.shape.len() != output_shape.len() {
            return Err(format!(
                "cannot pad tensor shape {:?} into {:?} along axis {axis}",
                self.shape, output_shape
            ));
        }
        for (input_extent, output_extent) in self.shape.iter().zip(output_shape) {
            if input_extent != output_extent && input_extent != &self.shape[axis] {
                return Err(format!(
                    "cannot pad tensor shape {:?} into {:?} along axis {axis}",
                    self.shape, output_shape
                ));
            }
        }
        if self
            .shape
            .iter()
            .enumerate()
            .any(|(index, extent)| index != axis && *extent != output_shape[index])
            || start
                .checked_add(self.shape[axis])
                .is_none_or(|end| end > output_shape[axis])
        {
            return Err(format!(
                "cannot pad tensor shape {:?} into {:?} along axis {axis}",
                self.shape, output_shape
            ));
        }
        let outer = element_count(&output_shape[..axis])?;
        let inner = element_count(&output_shape[axis + 1..])?;
        let mut data = vec![0.0; element_count(output_shape)?];
        for outer_index in 0..outer {
            let source_start = outer_index * self.shape[axis] * inner;
            let destination_start = (outer_index * output_shape[axis] + start) * inner;
            let width = self.shape[axis] * inner;
            data[destination_start..destination_start + width]
                .copy_from_slice(&self.data[source_start..source_start + width]);
        }
        Self::new(output_shape.to_vec(), data)
    }

    fn mean_all(&self) -> Result<Self, String> {
        self.sum_all()?.scale(1.0 / self.data.len() as f64)
    }

    fn sin(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.sin()).collect(),
        )
    }

    fn cos(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.cos()).collect(),
        )
    }

    fn powi(&self, exponent: u32) -> Result<Self, String> {
        let exponent = i32::try_from(exponent)
            .map_err(|_| "powi exponent must fit in a signed 32-bit integer".to_string())?;
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.powi(exponent)).collect(),
        )
    }

    fn log(&self) -> Result<Self, String> {
        if self.data.iter().any(|value| *value <= 0.0) {
            return Err("log requires strictly positive tensor values".to_string());
        }
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| value.ln()).collect(),
        )
    }

    fn tanh_derivative_from_output(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data.iter().map(|value| 1.0 - value * value).collect(),
        )
    }

    fn tanh_second_derivative_from_output(&self) -> Result<Self, String> {
        Self::new(
            self.shape.clone(),
            self.data
                .iter()
                .map(|value| -2.0 * value * (1.0 - value * value))
                .collect(),
        )
    }

    fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        let shape = matmul_shape(&self.shape, &rhs.shape)?;
        let lhs_rows = self.shape[self.shape.len() - 2];
        let lhs_inner = self.shape[self.shape.len() - 1];
        let rhs_cols = rhs.shape[rhs.shape.len() - 1];
        let lhs_batch_shape = &self.shape[..self.shape.len() - 2];
        let rhs_batch_shape = &rhs.shape[..rhs.shape.len() - 2];
        let batch_shape = &shape[..shape.len() - 2];
        let batch_count = element_count(batch_shape)?;
        let lhs_batch_strides = contiguous_strides(lhs_batch_shape);
        let rhs_batch_strides = contiguous_strides(rhs_batch_shape);
        let mut data = vec![0.0; element_count(&shape)?];

        for batch_index in 0..batch_count {
            let lhs_batch = broadcast_offset(
                batch_index,
                batch_shape,
                lhs_batch_shape,
                &lhs_batch_strides,
            );
            let rhs_batch = broadcast_offset(
                batch_index,
                batch_shape,
                rhs_batch_shape,
                &rhs_batch_strides,
            );
            for row in 0..lhs_rows {
                let lhs_row_start = lhs_batch * lhs_rows * lhs_inner + row * lhs_inner;
                let output_row_start = batch_index * lhs_rows * rhs_cols + row * rhs_cols;
                for inner in 0..lhs_inner {
                    let lhs_value = self.data[lhs_row_start + inner];
                    let rhs_row_start = rhs_batch * lhs_inner * rhs_cols + inner * rhs_cols;
                    for col in 0..rhs_cols {
                        data[output_row_start + col] += lhs_value * rhs.data[rhs_row_start + col];
                    }
                }
            }
        }

        Self::new(shape, data)
    }

    fn transpose_last_two(&self) -> Result<Self, String> {
        if self.shape.len() < 2 {
            return Err(format!(
                "transpose_last_two requires a tensor with at least two dimensions, got {:?}",
                self.shape
            ));
        }

        let rows = self.shape[self.shape.len() - 2];
        let cols = self.shape[self.shape.len() - 1];
        let batch_count = element_count(&self.shape[..self.shape.len() - 2])?;
        let mut shape = self.shape.clone();
        let last_axis = shape.len() - 1;
        shape.swap(last_axis - 1, last_axis);
        let mut data = vec![0.0; self.data.len()];
        for batch in 0..batch_count {
            for row in 0..rows {
                for col in 0..cols {
                    data[batch * cols * rows + col * rows + row] =
                        self.data[batch * rows * cols + row * cols + col];
                }
            }
        }

        Self::new(shape, data)
    }

    fn transpose(&self, axes: &[usize]) -> Result<Self, String> {
        validate_permutation(axes, self.shape.len())?;
        let output_shape = axes
            .iter()
            .map(|axis| self.shape[*axis])
            .collect::<Vec<_>>();
        let input_strides = contiguous_strides(&self.shape);
        let mut data = vec![0.0; self.data.len()];

        for (output_index, output_value) in data.iter_mut().enumerate() {
            let mut remaining = output_index;
            let mut input_index = 0;
            for output_axis in (0..output_shape.len()).rev() {
                let coordinate = remaining % output_shape[output_axis];
                remaining /= output_shape[output_axis];
                input_index += coordinate * input_strides[axes[output_axis]];
            }
            *output_value = self.data[input_index];
        }

        Self::new(output_shape, data)
    }

    /// Returns a materialized permutation of the tensor axes.
    ///
    /// This is intentionally a small public boundary for transform wrappers;
    /// compiled execution plans remain responsible for backend lowering.
    pub fn permute(&self, axes: &[usize]) -> Result<Self, String> {
        self.transpose(axes)
    }

    fn broadcast_to_shape(&self, target_shape: &[usize]) -> Result<Self, String> {
        if broadcast_shape(&self.shape, target_shape)? != target_shape {
            return Err(format!(
                "cannot broadcast tensor shape {:?} to {:?}",
                self.shape, target_shape
            ));
        }

        let count = element_count(target_shape)?;
        let strides = contiguous_strides(&self.shape);
        let data = (0..count)
            .map(|index| self.data[broadcast_offset(index, target_shape, &self.shape, &strides)])
            .collect();
        Self::new(target_shape.to_vec(), data)
    }

    fn elementwise(&self, rhs: &Self, f: impl Fn(f64, f64) -> f64) -> Result<Self, String> {
        let shape = broadcast_shape(&self.shape, &rhs.shape)?;
        let count = element_count(&shape)?;
        let lhs_strides = contiguous_strides(&self.shape);
        let rhs_strides = contiguous_strides(&rhs.shape);
        let mut data = Vec::with_capacity(count);

        for index in 0..count {
            let lhs_index = broadcast_offset(index, &shape, &self.shape, &lhs_strides);
            let rhs_index = broadcast_offset(index, &shape, &rhs.shape, &rhs_strides);
            data.push(f(self.data[lhs_index], rhs.data[rhs_index]));
        }

        Self::new(shape, data)
    }

    fn reduce_to_shape(&self, target_shape: &[usize]) -> Result<Self, String> {
        if target_shape.len() > self.shape.len() {
            return Err(format!(
                "cannot reduce tensor shape {:?} to higher-rank shape {:?}",
                self.shape, target_shape
            ));
        }

        let rank_offset = self.shape.len() - target_shape.len();
        for (source_extent, target_extent) in self.shape[rank_offset..].iter().zip(target_shape) {
            if source_extent != target_extent && *target_extent != 1 {
                return Err(format!(
                    "cannot reduce tensor shape {:?} to {:?}",
                    self.shape, target_shape
                ));
            }
        }

        let target_count = element_count(target_shape)?;
        let target_strides = contiguous_strides(target_shape);
        let mut data = vec![0.0; target_count];

        for (index, value) in self.data.iter().enumerate() {
            let mut remaining = index;
            let mut target_index = 0;
            for axis in (0..self.shape.len()).rev() {
                let coordinate = remaining % self.shape[axis];
                remaining /= self.shape[axis];
                if axis >= rank_offset {
                    let target_axis = axis - rank_offset;
                    if target_shape[target_axis] != 1 {
                        target_index += coordinate * target_strides[target_axis];
                    }
                }
            }
            data[target_index] += value;
        }

        Self::new(target_shape.to_vec(), data)
    }
}

impl TensorIr {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn symbolic_jvp(
        &self,
        output: TensorNodeId,
        differentiated_input: &str,
    ) -> Result<SymbolicJvp, String> {
        self.node(output)?;
        let mut transformed = TensorIr::new();
        let mut pairs = Vec::with_capacity(self.nodes.len());
        let mut found_input = false;

        for node in &self.nodes {
            let pair = match &node.op {
                TensorOp::Input { name } => {
                    let value = transformed.input(name.clone(), node.shape.clone())?;
                    let tangent = if name == differentiated_input {
                        found_input = true;
                        transformed.powi(value, 0)?
                    } else {
                        transformed.sub(value, value)?
                    };
                    (value, tangent)
                }
                TensorOp::ScalarConstant { value } => (
                    transformed.scalar_constant(*value),
                    transformed.scalar_constant(0.0),
                ),
                TensorOp::Add { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    (
                        transformed.add(lhs_value, rhs_value)?,
                        transformed.add(lhs_tangent, rhs_tangent)?,
                    )
                }
                TensorOp::Sub { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    (
                        transformed.sub(lhs_value, rhs_value)?,
                        transformed.sub(lhs_tangent, rhs_tangent)?,
                    )
                }
                TensorOp::Div { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let left_term = transformed.mul(lhs_tangent, rhs_value)?;
                    let right_term = transformed.mul(lhs_value, rhs_tangent)?;
                    let numerator = transformed.sub(left_term, right_term)?;
                    let denominator = transformed.mul(rhs_value, rhs_value)?;
                    (
                        transformed.div(lhs_value, rhs_value)?,
                        transformed.div(numerator, denominator)?,
                    )
                }
                TensorOp::Mul { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let left_term = transformed.mul(lhs_tangent, rhs_value)?;
                    let right_term = transformed.mul(lhs_value, rhs_tangent)?;
                    (
                        transformed.mul(lhs_value, rhs_value)?,
                        transformed.add(left_term, right_term)?,
                    )
                }
                TensorOp::Greater { lhs, rhs } => {
                    let (lhs_value, _) = pairs[*lhs];
                    let (rhs_value, _) = pairs[*rhs];
                    let value = transformed.greater(lhs_value, rhs_value)?;
                    let tangent = transformed.scalar_constant(0.0);
                    (value, tangent)
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let (condition_value, _) = pairs[*condition];
                    let (true_value, true_tangent) = pairs[*on_true];
                    let (false_value, false_tangent) = pairs[*on_false];
                    (
                        transformed.where_select(condition_value, true_value, false_value)?,
                        transformed.where_select(condition_value, true_tangent, false_tangent)?,
                    )
                }
                TensorOp::Tanh { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.tanh(input_value)?;
                    let one = transformed.scalar_constant(1.0);
                    let squared = transformed.mul(value, value)?;
                    let derivative = transformed.sub(one, squared)?;
                    (value, transformed.mul(input_tangent, derivative)?)
                }
                TensorOp::Exp { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.exp(input_value)?;
                    (value, transformed.mul(input_tangent, value)?)
                }
                TensorOp::Sin { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.sin(input_value)?;
                    let derivative = transformed.cos(input_value)?;
                    (value, transformed.mul(input_tangent, derivative)?)
                }
                TensorOp::Cos { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.cos(input_value)?;
                    let derivative = transformed.sin(input_value)?;
                    let tangent = transformed.mul(input_tangent, derivative)?;
                    let zero = transformed.scalar_constant(0.0);
                    (value, transformed.sub(zero, tangent)?)
                }
                TensorOp::Log { input } => {
                    let (input_value, input_tangent) = pairs[*input];
                    (
                        transformed.log(input_value)?,
                        transformed.div(input_tangent, input_value)?,
                    )
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let (lhs_value, lhs_tangent) = pairs[*lhs];
                    let (rhs_value, rhs_tangent) = pairs[*rhs];
                    let left_term = transformed.matmul(lhs_tangent, rhs_value)?;
                    let right_term = transformed.matmul(lhs_value, rhs_tangent)?;
                    (
                        transformed.matmul(lhs_value, rhs_value)?,
                        transformed.add(left_term, right_term)?,
                    )
                }
                TensorOp::Sum { input } => {
                    let (value, tangent) = pairs[*input];
                    (transformed.sum(value)?, transformed.sum(tangent)?)
                }
                TensorOp::Mean { input } => {
                    let (value, tangent) = pairs[*input];
                    (transformed.mean(value)?, transformed.mean(tangent)?)
                }
                TensorOp::SumAxis { input, axis } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.sum_axis(value, *axis as isize)?,
                        transformed.sum_axis(tangent, *axis as isize)?,
                    )
                }
                TensorOp::MeanAxis { input, axis } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.mean_axis(value, *axis as isize)?,
                        transformed.mean_axis(tangent, *axis as isize)?,
                    )
                }
                TensorOp::Reshape { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.reshape(value, node.shape.clone())?,
                        transformed.reshape(tangent, node.shape.clone())?,
                    )
                }
                TensorOp::Transpose { input, axes } => {
                    let (value, tangent) = pairs[*input];
                    let axes: Vec<isize> = axes.iter().map(|axis| *axis as isize).collect();
                    (
                        transformed.transpose(value, Some(axes.clone()))?,
                        transformed.transpose(tangent, Some(axes))?,
                    )
                }
                TensorOp::Powi { input, exponent } => {
                    let (input_value, input_tangent) = pairs[*input];
                    let value = transformed.powi(input_value, *exponent)?;
                    let tangent = if *exponent == 0 {
                        transformed.sub(input_value, input_value)?
                    } else {
                        let coefficient = transformed.scalar_constant(*exponent as f64);
                        let lower_power = transformed.powi(input_value, *exponent - 1)?;
                        let derivative = transformed.mul(coefficient, lower_power)?;
                        transformed.mul(input_tangent, derivative)?
                    };
                    (value, tangent)
                }
                TensorOp::Concat { inputs, axis } => {
                    let values = inputs.iter().map(|input| pairs[*input].0).collect();
                    let tangents = inputs.iter().map(|input| pairs[*input].1).collect();
                    (
                        transformed.concat(values, *axis as isize)?,
                        transformed.concat(tangents, *axis as isize)?,
                    )
                }
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.slice(value, *axis, *start, *length)?,
                        transformed.slice(tangent, *axis, *start, *length)?,
                    )
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.pad_slice(value, node.shape.clone(), *axis, *start)?,
                        transformed.pad_slice(tangent, node.shape.clone(), *axis, *start)?,
                    )
                }
                TensorOp::Broadcast { input } => {
                    let (value, tangent) = pairs[*input];
                    (
                        transformed.broadcast_to(value, node.shape.clone())?,
                        transformed.broadcast_to(tangent, node.shape.clone())?,
                    )
                }
            };
            pairs.push(pair);
        }

        if !found_input {
            return Err(format!("input {differentiated_input:?} does not exist"));
        }
        let (value, tangent) = pairs[output];
        Ok(SymbolicJvp {
            graph: transformed,
            value,
            tangent,
        })
    }

    /// Emits a reverse-mode transformable TensorIr graph.
    ///
    /// The returned graph takes every original input plus `cotangent_name`.
    /// Its `gradients` nodes are the corresponding VJP outputs, so a backend
    /// can compile primal and reverse graphs without returning to the CPU AD
    /// evaluator.
    pub fn symbolic_vjp(
        &self,
        output: TensorNodeId,
        cotangent_name: &str,
    ) -> Result<SymbolicVjp, String> {
        self.node(output)?;
        if self
            .nodes
            .iter()
            .any(|node| matches!(&node.op, TensorOp::Input { name } if name == cotangent_name))
        {
            return Err(format!(
                "cotangent input name {cotangent_name:?} conflicts with an existing input"
            ));
        }

        let mut transformed = TensorIr::new();
        let mut values = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = match &node.op {
                TensorOp::Input { name } => transformed.input(name.clone(), node.shape.clone())?,
                TensorOp::ScalarConstant { value } => transformed.scalar_constant(*value),
                TensorOp::Add { lhs, rhs } => transformed.add(values[*lhs], values[*rhs])?,
                TensorOp::Sub { lhs, rhs } => transformed.sub(values[*lhs], values[*rhs])?,
                TensorOp::Div { lhs, rhs } => transformed.div(values[*lhs], values[*rhs])?,
                TensorOp::Mul { lhs, rhs } => transformed.mul(values[*lhs], values[*rhs])?,
                TensorOp::Greater { lhs, rhs } => {
                    transformed.greater(values[*lhs], values[*rhs])?
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => transformed.where_select(
                    values[*condition],
                    values[*on_true],
                    values[*on_false],
                )?,
                TensorOp::Sum { input } => transformed.sum(values[*input])?,
                TensorOp::SumAxis { input, axis } => {
                    transformed.sum_axis(values[*input], *axis as isize)?
                }
                TensorOp::Matmul { lhs, rhs } => transformed.matmul(values[*lhs], values[*rhs])?,
                TensorOp::Tanh { input } => transformed.tanh(values[*input])?,
                TensorOp::Exp { input } => transformed.exp(values[*input])?,
                TensorOp::Reshape { input } => {
                    transformed.reshape(values[*input], node.shape.clone())?
                }
                TensorOp::Mean { input } => transformed.mean(values[*input])?,
                TensorOp::MeanAxis { input, axis } => {
                    transformed.mean_axis(values[*input], *axis as isize)?
                }
                TensorOp::Sin { input } => transformed.sin(values[*input])?,
                TensorOp::Cos { input } => transformed.cos(values[*input])?,
                TensorOp::Powi { input, exponent } => {
                    transformed.powi(values[*input], *exponent)?
                }
                TensorOp::Transpose { input, axes } => transformed.transpose(
                    values[*input],
                    Some(axes.iter().map(|axis| *axis as isize).collect()),
                )?,
                TensorOp::Log { input } => transformed.log(values[*input])?,
                TensorOp::Concat { inputs, axis } => transformed.concat(
                    inputs.iter().map(|input| values[*input]).collect(),
                    *axis as isize,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => transformed.slice(values[*input], *axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => {
                    transformed.pad_slice(values[*input], node.shape.clone(), *axis, *start)?
                }
                TensorOp::Broadcast { input } => {
                    transformed.broadcast_to(values[*input], node.shape.clone())?
                }
            };
            values.push(value);
        }

        let cotangent = transformed.input(cotangent_name, self.node(output)?.shape.clone())?;
        let mut cotangents = vec![None; self.nodes.len()];
        cotangents[output] = Some(cotangent);

        for node_id in (0..self.nodes.len()).rev() {
            let Some(upstream) = cotangents[node_id] else {
                continue;
            };
            let node = self.node(node_id)?;
            match &node.op {
                TensorOp::Input { .. }
                | TensorOp::ScalarConstant { .. }
                | TensorOp::Greater { .. } => {}
                TensorOp::Add { lhs, rhs } => {
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Sub { lhs, rhs } => {
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let zero = transformed.scalar_constant(0.0);
                    let negated = transformed.sub(zero, upstream)?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        negated,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Div { lhs, rhs } => {
                    let lhs_value = values[*lhs];
                    let rhs_value = values[*rhs];
                    let lhs_contribution = transformed.div(upstream, rhs_value)?;
                    let rhs_squared = transformed.mul(rhs_value, rhs_value)?;
                    let numerator = transformed.mul(upstream, lhs_value)?;
                    let quotient = transformed.div(numerator, rhs_squared)?;
                    let zero = transformed.scalar_constant(0.0);
                    let rhs_contribution = transformed.sub(zero, quotient)?;
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        lhs_contribution,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        rhs_contribution,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_contribution = transformed.mul(upstream, values[*rhs])?;
                    let rhs_contribution = transformed.mul(upstream, values[*lhs])?;
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        lhs_contribution,
                        &node.shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        rhs_contribution,
                        &node.shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let zero = transformed.scalar_constant(0.0);
                    let true_contribution =
                        transformed.where_select(values[*condition], upstream, zero)?;
                    let false_contribution =
                        transformed.where_select(values[*condition], zero, upstream)?;
                    let true_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        true_contribution,
                        &node.shape,
                        &self.node(*on_true)?.shape,
                    )?;
                    let false_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        false_contribution,
                        &node.shape,
                        &self.node(*on_false)?.shape,
                    )?;
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        *on_true,
                        true_contribution,
                    )?;
                    symbolic_accumulate(
                        &mut transformed,
                        &mut cotangents,
                        *on_false,
                        false_contribution,
                    )?;
                }
                TensorOp::Sum { input } => {
                    let contribution =
                        symbolic_broadcast_like(&mut transformed, upstream, values[*input])?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Broadcast { input } => {
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        upstream,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::SumAxis { input, axis } => {
                    let contribution = symbolic_expand_reduced_axis(
                        &mut transformed,
                        upstream,
                        values[*input],
                        *axis,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_transposed = symbolic_transpose_last_two(
                        &mut transformed,
                        values[*lhs],
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_transposed = symbolic_transpose_last_two(
                        &mut transformed,
                        values[*rhs],
                        &self.node(*rhs)?.shape,
                    )?;
                    let lhs_contribution = transformed.matmul(upstream, rhs_transposed)?;
                    let rhs_contribution = transformed.matmul(lhs_transposed, upstream)?;
                    let lhs_shape = transformed.node_shape(lhs_contribution)?;
                    let rhs_shape = transformed.node_shape(rhs_contribution)?;
                    let lhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        lhs_contribution,
                        &lhs_shape,
                        &self.node(*lhs)?.shape,
                    )?;
                    let rhs_contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        rhs_contribution,
                        &rhs_shape,
                        &self.node(*rhs)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *lhs, lhs_contribution)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *rhs, rhs_contribution)?;
                }
                TensorOp::Tanh { input } => {
                    let one = transformed.scalar_constant(1.0);
                    let squared = transformed.mul(values[node_id], values[node_id])?;
                    let derivative = transformed.sub(one, squared)?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Exp { input } => {
                    let contribution = transformed.mul(upstream, values[node_id])?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Reshape { input } => {
                    let contribution =
                        transformed.reshape(upstream, self.node(*input)?.shape.clone())?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Mean { input } => {
                    let contribution =
                        symbolic_broadcast_like(&mut transformed, upstream, values[*input])?;
                    let scale = transformed
                        .scalar_constant(1.0 / element_count(&self.node(*input)?.shape)? as f64);
                    let contribution = transformed.mul(contribution, scale)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::MeanAxis { input, axis } => {
                    let contribution = symbolic_expand_reduced_axis(
                        &mut transformed,
                        upstream,
                        values[*input],
                        *axis,
                    )?;
                    let scale =
                        transformed.scalar_constant(1.0 / self.node(*input)?.shape[*axis] as f64);
                    let contribution = transformed.mul(contribution, scale)?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Sin { input } => {
                    let derivative = transformed.cos(values[*input])?;
                    let contribution = transformed.mul(upstream, derivative)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Cos { input } => {
                    let derivative = transformed.sin(values[*input])?;
                    let product = transformed.mul(upstream, derivative)?;
                    let zero = transformed.scalar_constant(0.0);
                    let contribution = transformed.sub(zero, product)?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent != 0 {
                        let coefficient = transformed.scalar_constant(*exponent as f64);
                        let lower_power = transformed.powi(values[*input], *exponent - 1)?;
                        let derivative = transformed.mul(coefficient, lower_power)?;
                        let contribution = transformed.mul(upstream, derivative)?;
                        let contribution = symbolic_reduce_to_shape(
                            &mut transformed,
                            contribution,
                            &node.shape,
                            &self.node(*input)?.shape,
                        )?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                    }
                }
                TensorOp::Transpose { input, axes } => {
                    let inverse = inverse_permutation(axes)?;
                    let contribution = transformed.transpose(
                        upstream,
                        Some(inverse.into_iter().map(|axis| axis as isize).collect()),
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Log { input } => {
                    let contribution = transformed.div(upstream, values[*input])?;
                    let contribution = symbolic_reduce_to_shape(
                        &mut transformed,
                        contribution,
                        &node.shape,
                        &self.node(*input)?.shape,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::Concat { inputs, axis } => {
                    let mut start = 0;
                    for input in inputs {
                        let input_shape = &self.node(*input)?.shape;
                        let contribution =
                            transformed.slice(upstream, *axis, start, input_shape[*axis])?;
                        symbolic_accumulate(
                            &mut transformed,
                            &mut cotangents,
                            *input,
                            contribution,
                        )?;
                        start += input_shape[*axis];
                    }
                }
                TensorOp::Slice {
                    input, axis, start, ..
                } => {
                    let contribution = transformed.pad_slice(
                        upstream,
                        self.node(*input)?.shape.clone(),
                        *axis,
                        *start,
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
                TensorOp::PadSlice { input, axis, start } => {
                    let contribution = transformed.slice(
                        upstream,
                        *axis,
                        *start,
                        self.node(*input)?.shape[*axis],
                    )?;
                    symbolic_accumulate(&mut transformed, &mut cotangents, *input, contribution)?;
                }
            }
        }

        let mut gradients = BTreeMap::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let TensorOp::Input { name } = &node.op {
                let gradient = cotangents[node_id]
                    .unwrap_or(symbolic_zero_like(&mut transformed, values[node_id])?);
                gradients.insert(name.clone(), gradient);
            }
        }
        Ok(SymbolicVjp {
            graph: transformed,
            value: values[output],
            cotangent,
            gradients,
        })
    }

    pub fn input(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, String> {
        element_count(&shape)?;
        let name = name.into();
        if self.nodes.iter().any(
            |node| matches!(node.op, TensorOp::Input { name: ref existing } if existing == &name),
        ) {
            return Err(format!("input {name:?} already exists"));
        }

        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Input { name },
            shape,
        });
        Ok(id)
    }

    pub fn scalar_constant(&mut self, value: f64) -> TensorNodeId {
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::ScalarConstant { value },
            shape: vec![],
        });
        id
    }

    pub fn add(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary(TensorOp::Add { lhs, rhs }, lhs, rhs)
    }

    pub fn mul(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary(TensorOp::Mul { lhs, rhs }, lhs, rhs)
    }

    pub fn sub(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary(TensorOp::Sub { lhs, rhs }, lhs, rhs)
    }

    pub fn div(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        self.binary(TensorOp::Div { lhs, rhs }, lhs, rhs)
    }

    pub fn greater(
        &mut self,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        self.binary(TensorOp::Greater { lhs, rhs }, lhs, rhs)
    }

    pub fn where_select(
        &mut self,
        condition: TensorNodeId,
        on_true: TensorNodeId,
        on_false: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let condition_shape = self.node(condition)?.shape.clone();
        let true_shape = self.node(on_true)?.shape.clone();
        let false_shape = self.node(on_false)?.shape.clone();
        let shape = broadcast_shape(
            &broadcast_shape(&condition_shape, &true_shape)?,
            &false_shape,
        )?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Where {
                condition,
                on_true,
                on_false,
            },
            shape,
        });
        Ok(id)
    }

    pub fn sum(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.node(input)?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Sum { input },
            shape: vec![],
        });
        Ok(id)
    }

    pub fn sum_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axis = normalize_axis(axis, input_shape.len())?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::SumAxis { input, axis },
            shape: reduced_shape(&input_shape, axis)?,
        });
        Ok(id)
    }

    pub fn matmul(&mut self, lhs: TensorNodeId, rhs: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = matmul_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape)?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Matmul { lhs, rhs },
            shape,
        });
        Ok(id)
    }

    pub fn tanh(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Tanh { input },
            shape,
        });
        Ok(id)
    }

    pub fn exp(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Exp { input },
            shape,
        });
        Ok(id)
    }

    pub fn reshape(
        &mut self,
        input: TensorNodeId,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if element_count(&shape)? != element_count(&input_shape)? {
            return Err(format!(
                "cannot reshape tensor shape {:?} to {:?}",
                input_shape, shape
            ));
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Reshape { input },
            shape,
        });
        Ok(id)
    }

    pub fn mean(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        self.node(input)?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Mean { input },
            shape: vec![],
        });
        Ok(id)
    }

    pub fn mean_axis(&mut self, input: TensorNodeId, axis: isize) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axis = normalize_axis(axis, input_shape.len())?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::MeanAxis { input, axis },
            shape: reduced_shape(&input_shape, axis)?,
        });
        Ok(id)
    }

    pub fn sin(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Sin { input },
            shape,
        });
        Ok(id)
    }

    pub fn cos(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Cos { input },
            shape,
        });
        Ok(id)
    }

    pub fn powi(&mut self, input: TensorNodeId, exponent: u32) -> Result<TensorNodeId, String> {
        if exponent > i32::MAX as u32 {
            return Err("powi exponent must fit in a signed 32-bit integer".to_string());
        }
        let shape = self.node(input)?.shape.clone();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Powi { input, exponent },
            shape,
        });
        Ok(id)
    }

    pub fn transpose(
        &mut self,
        input: TensorNodeId,
        axes: Option<Vec<isize>>,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axes = normalize_permutation(axes, input_shape.len())?;
        let shape = axes.iter().map(|axis| input_shape[*axis]).collect();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Transpose { input, axes },
            shape,
        });
        Ok(id)
    }

    pub fn log(&mut self, input: TensorNodeId) -> Result<TensorNodeId, String> {
        let shape = self.node(input)?.shape.clone();
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Log { input },
            shape,
        });
        Ok(id)
    }

    pub fn concat(
        &mut self,
        inputs: Vec<TensorNodeId>,
        axis: isize,
    ) -> Result<TensorNodeId, String> {
        if inputs.is_empty() {
            return Err("concat requires at least one input".to_string());
        }
        let shapes = inputs
            .iter()
            .map(|input| self.node(*input).map(|node| node.shape.as_slice()))
            .collect::<Result<Vec<_>, _>>()?;
        let axis = normalize_axis(axis, shapes[0].len())?;
        let shape = concat_shape(&shapes, axis)?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Concat { inputs, axis },
            shape,
        });
        Ok(id)
    }

    pub fn broadcast_to(
        &mut self,
        input: TensorNodeId,
        shape: Vec<usize>,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if broadcast_shape(&input_shape, &shape)? != shape {
            return Err(format!(
                "cannot broadcast tensor shape {input_shape:?} to {shape:?}"
            ));
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Broadcast { input },
            shape,
        });
        Ok(id)
    }

    fn slice(
        &mut self,
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if axis >= input_shape.len()
            || start
                .checked_add(length)
                .is_none_or(|end| end > input_shape[axis])
        {
            return Err(format!(
                "invalid slice [{start}..{}) on axis {axis} for shape {input_shape:?}",
                start + length
            ));
        }
        let mut shape = input_shape;
        shape[axis] = length;
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::Slice {
                input,
                axis,
                start,
                length,
            },
            shape,
        });
        Ok(id)
    }

    pub fn slice_axis(
        &mut self,
        input: TensorNodeId,
        axis: isize,
        start: usize,
        stop: usize,
    ) -> Result<TensorNodeId, String> {
        let rank = self.node(input)?.shape.len();
        let axis = normalize_axis(axis, rank)?;
        let length = stop.checked_sub(start).ok_or_else(|| {
            format!("slice stop {stop} must be greater than or equal to start {start}")
        })?;
        self.slice(input, axis, start, length)
    }

    fn pad_slice(
        &mut self,
        input: TensorNodeId,
        output_shape: Vec<usize>,
        axis: usize,
        start: usize,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        if input_shape.len() != output_shape.len()
            || axis >= output_shape.len()
            || input_shape
                .iter()
                .enumerate()
                .any(|(index, extent)| index != axis && *extent != output_shape[index])
            || start
                .checked_add(input_shape[axis])
                .is_none_or(|end| end > output_shape[axis])
        {
            return Err(format!(
                "cannot pad tensor shape {input_shape:?} into {output_shape:?} along axis {axis}"
            ));
        }
        let id = self.nodes.len();
        self.nodes.push(TensorNode {
            op: TensorOp::PadSlice { input, axis, start },
            shape: output_shape,
        });
        Ok(id)
    }

    pub fn evaluate(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        self.node(output)?;
        let values = self.evaluate_all(inputs)?;
        values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))
    }

    pub fn node_shape(&self, id: TensorNodeId) -> Result<Vec<usize>, String> {
        Ok(self.node(id)?.shape.clone())
    }

    pub fn input_node_id(&self, name: &str) -> Result<TensorNodeId, String> {
        self.nodes
            .iter()
            .position(
                |node| matches!(&node.op, TensorOp::Input { name: candidate } if candidate == name),
            )
            .ok_or_else(|| format!("input {name:?} does not exist"))
    }

    pub fn compile_cpu(&self, output: TensorNodeId) -> Result<TensorExecutionPlan, String> {
        let (plan, _) = self.compile_cpu_many(&[output])?;
        Ok(plan)
    }

    /// Freezes the union of several output-reachable subgraphs into one plan.
    ///
    /// The returned node ids correspond to `outputs` in order after DCE and
    /// structural CSE. Backends can execute the shared prefix once and consume
    /// each retained output directly, which is required for coherent
    /// multi-parameter reverse-mode updates.
    pub fn compile_cpu_many(
        &self,
        outputs: &[TensorNodeId],
    ) -> Result<(TensorExecutionPlan, Vec<TensorNodeId>), String> {
        if outputs.is_empty() {
            return Err("execution plan requires at least one output".to_string());
        }
        for output in outputs {
            self.node(*output)?;
        }
        let mut reachable = HashSet::new();
        let mut pending = outputs.to_vec();
        while let Some(node_id) = pending.pop() {
            if !reachable.insert(node_id) {
                continue;
            }
            pending.extend(tensor_op_inputs(&self.node(node_id)?.op));
        }

        let mut remap = HashMap::new();
        let mut cse_nodes = HashMap::new();
        let mut nodes = Vec::with_capacity(reachable.len());
        for (old_id, node) in self.nodes.iter().enumerate() {
            if !reachable.contains(&old_id) {
                continue;
            }
            let op = remap_tensor_op(&node.op, &remap)?;
            if let Some(key) = pure_tensor_op_cse_key(&op, &node.shape) {
                if let Some(existing_id) = cse_nodes.get(&key) {
                    remap.insert(old_id, *existing_id);
                    continue;
                }
                cse_nodes.insert(key, nodes.len());
            }
            remap.insert(old_id, nodes.len());
            nodes.push(TensorNode {
                op,
                shape: node.shape.clone(),
            });
        }
        let output_node_ids = outputs
            .iter()
            .map(|output| {
                remap
                    .get(output)
                    .copied()
                    .ok_or_else(|| format!("output node {output} is not reachable"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let output_node_id = *output_node_ids
            .first()
            .ok_or_else(|| "execution plan requires at least one output".to_string())?;
        let fused_elementwise_output =
            output_node_ids.len() == 1 && is_fusable_elementwise_subgraph(&nodes, output_node_id);
        Ok((
            TensorExecutionPlan {
                nodes,
                output_node_id,
                fused_elementwise_output,
            },
            output_node_ids,
        ))
    }

    pub fn vjp(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        self.value_and_vjp(output, inputs, output_cotangent)
            .map(|(_, gradients)| gradients)
    }

    pub fn value_and_vjp(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        let output_node = self.node(output)?;
        if output_cotangent.shape != output_node.shape {
            return Err(format!(
                "output cotangent shape {:?} does not match output shape {:?}",
                output_cotangent.shape, output_node.shape
            ));
        }

        let values = self.evaluate_all(inputs)?;
        let output_value = values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))?;
        let mut cotangents = vec![None; self.nodes.len()];
        cotangents[output] = Some(output_cotangent);

        for node_id in (0..self.nodes.len()).rev() {
            let cotangent = match cotangents[node_id].clone() {
                Some(value) => value,
                None => continue,
            };
            match &self.nodes[node_id].op {
                TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } => {}
                TensorOp::Add { lhs, rhs } => {
                    let lhs_contribution = cotangent.reduce_to_shape(&self.node(*lhs)?.shape)?;
                    let rhs_contribution = cotangent.reduce_to_shape(&self.node(*rhs)?.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Sub { lhs, rhs } => {
                    let lhs_contribution = cotangent.reduce_to_shape(&self.node(*lhs)?.shape)?;
                    let rhs_contribution =
                        cotangent.neg()?.reduce_to_shape(&self.node(*rhs)?.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Div { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let reciprocal = rhs_value.reciprocal()?;
                    let reciprocal_squared = reciprocal.mul(&reciprocal)?;
                    let lhs_contribution = cotangent
                        .mul(&reciprocal)?
                        .reduce_to_shape(&lhs_value.shape)?;
                    let rhs_contribution = cotangent
                        .mul(lhs_value)?
                        .mul(&reciprocal_squared)?
                        .neg()?
                        .reduce_to_shape(&rhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let lhs_contribution = cotangent
                        .mul(rhs_value)?
                        .reduce_to_shape(&lhs_value.shape)?;
                    let rhs_contribution = cotangent
                        .mul(lhs_value)?
                        .reduce_to_shape(&rhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Greater { .. } => {}
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => {
                    let condition_value = values
                        .get(*condition)
                        .ok_or_else(|| format!("node {condition} has no evaluated value"))?;
                    let zero = DynamicTensor::filled(vec![], 0.0)?;
                    let true_contribution = condition_value
                        .where_select(&cotangent, &zero)?
                        .reduce_to_shape(&self.node(*on_true)?.shape)?;
                    let false_contribution = condition_value
                        .where_select(&zero, &cotangent)?
                        .reduce_to_shape(&self.node(*on_false)?.shape)?;
                    accumulate(&mut cotangents[*on_true], true_contribution)?;
                    accumulate(&mut cotangents[*on_false], false_contribution)?;
                }
                TensorOp::Sum { input } => {
                    let contribution = cotangent.broadcast_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::SumAxis { input, axis } => {
                    let contribution =
                        cotangent.expand_reduced_axis(&self.node(*input)?.shape, *axis)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let lhs_contribution = cotangent
                        .matmul(&rhs_value.transpose_last_two()?)?
                        .reduce_to_shape(&lhs_value.shape)?;
                    let rhs_contribution = lhs_value
                        .transpose_last_two()?
                        .matmul(&cotangent)?
                        .reduce_to_shape(&rhs_value.shape)?;
                    accumulate(&mut cotangents[*lhs], lhs_contribution)?;
                    accumulate(&mut cotangents[*rhs], rhs_contribution)?;
                }
                TensorOp::Tanh { input } => {
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&output_value.tanh_derivative_from_output()?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Exp { input } => {
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(output_value)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Reshape { input } => {
                    let contribution = cotangent.reshape(self.node(*input)?.shape.clone())?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Mean { input } => {
                    let input_shape = self.node(*input)?.shape.clone();
                    let input_count = element_count(&input_shape)?;
                    let contribution = cotangent
                        .broadcast_to_shape(&input_shape)?
                        .scale(1.0 / input_count as f64)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::MeanAxis { input, axis } => {
                    let input_shape = self.node(*input)?.shape.clone();
                    let contribution = cotangent
                        .expand_reduced_axis(&input_shape, *axis)?
                        .scale(1.0 / input_shape[*axis] as f64)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Sin { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.cos()?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Cos { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.sin()?)?
                        .neg()?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent == 0 {
                        continue;
                    }
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.powi(*exponent - 1)?)?
                        .scale(*exponent as f64)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Transpose { input, axes } => {
                    let contribution = cotangent.transpose(&inverse_permutation(axes)?)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Log { input } => {
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    let contribution = cotangent
                        .mul(&input_value.reciprocal()?)?
                        .reduce_to_shape(&self.node(*input)?.shape)?;
                    accumulate(&mut cotangents[*input], contribution)?;
                }
                TensorOp::Concat { inputs, axis } => {
                    let mut start = 0;
                    for input in inputs {
                        let input_shape = &self.node(*input)?.shape;
                        accumulate(
                            &mut cotangents[*input],
                            cotangent.slice_axis(*axis, start, input_shape[*axis])?,
                        )?;
                        start += input_shape[*axis];
                    }
                }
                TensorOp::Slice {
                    input, axis, start, ..
                } => accumulate(
                    &mut cotangents[*input],
                    cotangent.pad_slice(&self.node(*input)?.shape, *axis, *start)?,
                )?,
                TensorOp::PadSlice { input, axis, start } => accumulate(
                    &mut cotangents[*input],
                    cotangent.slice_axis(*axis, *start, self.node(*input)?.shape[*axis])?,
                )?,
                TensorOp::Broadcast { input } => accumulate(
                    &mut cotangents[*input],
                    cotangent.reduce_to_shape(&self.node(*input)?.shape)?,
                )?,
            }
        }

        let mut gradients = BTreeMap::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if let TensorOp::Input { name } = &node.op {
                let gradient = match cotangents[node_id].clone() {
                    Some(value) => value,
                    None => DynamicTensor::filled(node.shape.clone(), 0.0)?,
                };
                gradients.insert(name.clone(), gradient);
            }
        }
        Ok((output_value, gradients))
    }

    pub fn jvp(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.node(output)?;
        let values = self.evaluate_all(inputs)?;
        let mut tangents: Vec<DynamicTensor> = Vec::with_capacity(self.nodes.len());

        for (node_id, node) in self.nodes.iter().enumerate() {
            let tangent = match &node.op {
                TensorOp::Input { name } => {
                    let input_tangent = input_tangents
                        .get(name)
                        .ok_or_else(|| format!("missing input tangent {name:?}"))?;
                    if input_tangent.shape != node.shape {
                        return Err(format!(
                            "input tangent {name:?} has shape {:?}, expected {:?}",
                            input_tangent.shape, node.shape
                        ));
                    }
                    input_tangent.clone()
                }
                TensorOp::ScalarConstant { .. } => DynamicTensor::filled(vec![], 0.0)?,
                TensorOp::Add { lhs, rhs } => tangents
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?
                    .add(
                        tangents
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?,
                    )?,
                TensorOp::Sub { lhs, rhs } => tangents
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?
                    .sub(
                        tangents
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?,
                    )?,
                TensorOp::Div { lhs, rhs } => {
                    let lhs_tangent = tangents
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    let reciprocal = rhs_value.reciprocal()?;
                    lhs_tangent.mul(&reciprocal)?.sub(
                        &lhs_value
                            .mul(rhs_tangent)?
                            .mul(&reciprocal.mul(&reciprocal)?)?,
                    )?
                }
                TensorOp::Mul { lhs, rhs } => {
                    let lhs_tangent = tangents
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    lhs_tangent
                        .mul(rhs_value)?
                        .add(&lhs_value.mul(rhs_tangent)?)?
                }
                TensorOp::Greater { .. } => DynamicTensor::filled(node.shape.clone(), 0.0)?,
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => values
                    .get(*condition)
                    .ok_or_else(|| format!("node {condition} has no evaluated value"))?
                    .where_select(
                        tangents
                            .get(*on_true)
                            .ok_or_else(|| format!("node {on_true} has no evaluated tangent"))?,
                        tangents
                            .get(*on_false)
                            .ok_or_else(|| format!("node {on_false} has no evaluated tangent"))?,
                    )?,
                TensorOp::Sum { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .sum_all()?,
                TensorOp::SumAxis { input, axis } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .reduce_axis(*axis, 1.0)?,
                TensorOp::Matmul { lhs, rhs } => {
                    let lhs_tangent = tangents
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated tangent"))?;
                    let rhs_tangent = tangents
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated tangent"))?;
                    let lhs_value = values
                        .get(*lhs)
                        .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                    let rhs_value = values
                        .get(*rhs)
                        .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                    lhs_tangent
                        .matmul(rhs_value)?
                        .add(&lhs_value.matmul(rhs_tangent)?)?
                }
                TensorOp::Tanh { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    input_tangent.mul(&output_value.tanh_derivative_from_output()?)?
                }
                TensorOp::Exp { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let output_value = values
                        .get(node_id)
                        .ok_or_else(|| format!("node {node_id} has no evaluated value"))?;
                    input_tangent.mul(output_value)?
                }
                TensorOp::Reshape { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .reshape(node.shape.clone())?,
                TensorOp::Mean { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .mean_all()?,
                TensorOp::MeanAxis { input, axis } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .reduce_axis(*axis, 1.0 / self.node(*input)?.shape[*axis] as f64)?,
                TensorOp::Sin { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.cos()?)?
                }
                TensorOp::Cos { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.sin()?)?.neg()?
                }
                TensorOp::Powi { input, exponent } => {
                    if *exponent == 0 {
                        DynamicTensor::filled(node.shape.clone(), 0.0)?
                    } else {
                        let input_tangent = tangents
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                        let input_value = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        input_tangent
                            .mul(&input_value.powi(*exponent - 1)?)?
                            .scale(*exponent as f64)?
                    }
                }
                TensorOp::Transpose { input, axes } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .transpose(axes)?,
                TensorOp::Log { input } => {
                    let input_tangent = tangents
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated tangent"))?;
                    let input_value = values
                        .get(*input)
                        .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                    input_tangent.mul(&input_value.reciprocal()?)?
                }
                TensorOp::Concat { inputs, axis } => DynamicTensor::concat(
                    &inputs
                        .iter()
                        .map(|input| {
                            tangents
                                .get(*input)
                                .ok_or_else(|| format!("node {input} has no evaluated tangent"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    *axis,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .slice_axis(*axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .pad_slice(&node.shape, *axis, *start)?,
                TensorOp::Broadcast { input } => tangents
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated tangent"))?
                    .broadcast_to_shape(&node.shape)?,
            };
            tangents.push(tangent);
        }

        let value = values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))?;
        let tangent = tangents
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no tangent"))?;
        Ok((value, tangent))
    }

    pub fn hessian_scalar(
        &self,
        output: TensorNodeId,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<Vec<f64>>, String> {
        if !self.node(output)?.shape.is_empty() {
            return Err(format!(
                "hessian_scalar requires a scalar output, got shape {:?}",
                self.node(output)?.shape
            ));
        }
        let input_shape = self
            .nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name } if name == input_name => Some(node.shape.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("input {input_name:?} does not exist"))?;
        let input_count = element_count(&input_shape)?;
        let mut hessian = vec![vec![0.0; input_count]; input_count];

        for (row, hessian_row) in hessian.iter_mut().enumerate() {
            for (col, entry) in hessian_row.iter_mut().enumerate() {
                let first_tangents = BTreeMap::from([(
                    input_name.to_string(),
                    DynamicTensor::one_hot(input_shape.clone(), row)?,
                )]);
                let second_tangents = BTreeMap::from([(
                    input_name.to_string(),
                    DynamicTensor::one_hot(input_shape.clone(), col)?,
                )]);
                let result =
                    self.evaluate_mixed(output, inputs, &first_tangents, &second_tangents)?;
                *entry = result.mixed.data[0];
            }
        }

        Ok(hessian)
    }

    pub fn hvp_scalar(
        &self,
        output: TensorNodeId,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangent: DynamicTensor,
    ) -> Result<DynamicTensor, String> {
        if !self.node(output)?.shape.is_empty() {
            return Err(format!(
                "hvp_scalar requires a scalar output, got shape {:?}",
                self.node(output)?.shape
            ));
        }
        let input_shape = self
            .nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name } if name == input_name => Some(node.shape.clone()),
                _ => None,
            })
            .ok_or_else(|| format!("input {input_name:?} does not exist"))?;
        if input_tangent.shape != input_shape {
            return Err(format!(
                "input tangent shape {:?} does not match input {input_name:?} shape {:?}",
                input_tangent.shape, input_shape
            ));
        }

        let input_count = element_count(&input_shape)?;
        let second_tangents = BTreeMap::from([(input_name.to_string(), input_tangent)]);
        let mut data = Vec::with_capacity(input_count);
        for index in 0..input_count {
            let first_tangents = BTreeMap::from([(
                input_name.to_string(),
                DynamicTensor::one_hot(input_shape.clone(), index)?,
            )]);
            let result = self.evaluate_mixed(output, inputs, &first_tangents, &second_tangents)?;
            data.push(result.mixed.data[0]);
        }
        DynamicTensor::new(input_shape, data)
    }

    pub fn lower_text(&self) -> String {
        self.nodes
            .iter()
            .enumerate()
            .map(|(id, node)| match &node.op {
                TensorOp::Input { name } => {
                    format!("%{id} = input[name={name}] : {}", format_shape(&node.shape))
                }
                TensorOp::ScalarConstant { value } => {
                    format!(
                        "%{id} = constant[value={value}] : {}",
                        format_shape(&node.shape)
                    )
                }
                TensorOp::Add { lhs, rhs } => format!(
                    "%{id} = add(%{lhs}, %{rhs}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Sub { lhs, rhs } => format!(
                    "%{id} = sub(%{lhs}, %{rhs}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Div { lhs, rhs } => format!(
                    "%{id} = div(%{lhs}, %{rhs}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Mul { lhs, rhs } => format!(
                    "%{id} = mul(%{lhs}, %{rhs}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Greater { lhs, rhs } => format!(
                    "%{id} = greater(%{lhs}, %{rhs}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => format!(
                    "%{id} = where(%{condition}, %{on_true}, %{on_false}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Sum { input } => {
                    format!("%{id} = sum(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::SumAxis { input, axis } => format!(
                    "%{id} = sum(%{input}, axis={axis}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Matmul { lhs, rhs } => format!(
                    "%{id} = matmul(%{lhs}, %{rhs}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Tanh { input } => {
                    format!("%{id} = tanh(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::Exp { input } => {
                    format!("%{id} = exp(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::Reshape { input } => {
                    format!("%{id} = reshape(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::Mean { input } => {
                    format!("%{id} = mean(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::MeanAxis { input, axis } => format!(
                    "%{id} = mean(%{input}, axis={axis}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Sin { input } => {
                    format!("%{id} = sin(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::Cos { input } => {
                    format!("%{id} = cos(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::Powi { input, exponent } => format!(
                    "%{id} = powi(%{input}, {exponent}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Transpose { input, axes } => format!(
                    "%{id} = transpose(%{input}, axes={axes:?}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Log { input } => {
                    format!("%{id} = log(%{input}) : {}", format_shape(&node.shape))
                }
                TensorOp::Concat { inputs, axis } => format!(
                    "%{id} = concat({}) axis={axis} : {}",
                    inputs
                        .iter()
                        .map(|input| format!("%{input}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    format_shape(&node.shape)
                ),
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => format!(
                    "%{id} = slice(%{input}, axis={axis}, start={start}, length={length}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::PadSlice { input, axis, start } => format!(
                    "%{id} = pad_slice(%{input}, axis={axis}, start={start}) : {}",
                    format_shape(&node.shape)
                ),
                TensorOp::Broadcast { input } => format!(
                    "%{id} = broadcast(%{input}) : {}",
                    format_shape(&node.shape)
                ),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn binary(
        &mut self,
        op: TensorOp,
        lhs: TensorNodeId,
        rhs: TensorNodeId,
    ) -> Result<TensorNodeId, String> {
        let shape = broadcast_shape(&self.node(lhs)?.shape, &self.node(rhs)?.shape)?;
        let id = self.nodes.len();
        self.nodes.push(TensorNode { op, shape });
        Ok(id)
    }

    fn evaluate_all(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        Self::evaluate_tensor_nodes(&self.nodes, inputs)
    }

    fn evaluate_tensor_nodes(
        nodes: &[TensorNode],
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<DynamicTensor>, String> {
        let mut values: Vec<DynamicTensor> = Vec::with_capacity(nodes.len());

        for node in nodes {
            let value = match &node.op {
                TensorOp::Input { name } => {
                    let input = inputs
                        .get(name)
                        .ok_or_else(|| format!("missing input {name:?}"))?;
                    if input.shape != node.shape {
                        return Err(format!(
                            "input {name:?} has shape {:?}, expected {:?}",
                            input.shape, node.shape
                        ));
                    }
                    input.clone()
                }
                TensorOp::ScalarConstant { value } => DynamicTensor::filled(vec![], *value)?,
                TensorOp::Add { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .add(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Sub { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .sub(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Div { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .div(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Mul { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .mul(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Greater { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .greater(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Where {
                    condition,
                    on_true,
                    on_false,
                } => values
                    .get(*condition)
                    .ok_or_else(|| format!("node {condition} has no evaluated value"))?
                    .where_select(
                        values
                            .get(*on_true)
                            .ok_or_else(|| format!("node {on_true} has no evaluated value"))?,
                        values
                            .get(*on_false)
                            .ok_or_else(|| format!("node {on_false} has no evaluated value"))?,
                    )?,
                TensorOp::Sum { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sum_all()?,
                TensorOp::SumAxis { input, axis } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_axis(*axis, 1.0)?,
                TensorOp::Matmul { lhs, rhs } => values
                    .get(*lhs)
                    .ok_or_else(|| format!("node {lhs} has no evaluated value"))?
                    .matmul(
                        values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?,
                    )?,
                TensorOp::Tanh { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .tanh()?,
                TensorOp::Exp { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .exp()?,
                TensorOp::Reshape { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reshape(node.shape.clone())?,
                TensorOp::Mean { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .mean_all()?,
                TensorOp::MeanAxis { input, axis } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .reduce_axis(*axis, 1.0 / nodes[*input].shape[*axis] as f64)?,
                TensorOp::Sin { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .sin()?,
                TensorOp::Cos { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .cos()?,
                TensorOp::Powi { input, exponent } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .powi(*exponent)?,
                TensorOp::Transpose { input, axes } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .transpose(axes)?,
                TensorOp::Log { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .log()?,
                TensorOp::Concat { inputs, axis } => DynamicTensor::concat(
                    &inputs
                        .iter()
                        .map(|input| {
                            values
                                .get(*input)
                                .ok_or_else(|| format!("node {input} has no evaluated value"))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    *axis,
                )?,
                TensorOp::Slice {
                    input,
                    axis,
                    start,
                    length,
                } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .slice_axis(*axis, *start, *length)?,
                TensorOp::PadSlice { input, axis, start } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .pad_slice(&node.shape, *axis, *start)?,
                TensorOp::Broadcast { input } => values
                    .get(*input)
                    .ok_or_else(|| format!("node {input} has no evaluated value"))?
                    .broadcast_to_shape(&node.shape)?,
            };
            values.push(value);
        }
        Ok(values)
    }

    fn evaluate_mixed(
        &self,
        output: TensorNodeId,
        inputs: &BTreeMap<String, DynamicTensor>,
        first_tangents: &BTreeMap<String, DynamicTensor>,
        second_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<MixedTangent, String> {
        self.node(output)?;
        let mut values: Vec<MixedTangent> = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let value =
                match &node.op {
                    TensorOp::Input { name } => {
                        let value = input_value(inputs, name, &node.shape)?;
                        MixedTangent {
                            first: input_tangent_or_zero(first_tangents, name, &node.shape)?,
                            second: input_tangent_or_zero(second_tangents, name, &node.shape)?,
                            mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                            value,
                        }
                    }
                    TensorOp::Log { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let reciprocal = input.value.reciprocal()?;
                        let reciprocal_squared = reciprocal.mul(&reciprocal)?;
                        MixedTangent {
                            value: input.value.log()?,
                            first: input.first.mul(&reciprocal)?,
                            second: input.second.mul(&reciprocal)?,
                            mixed: input
                                .mixed
                                .mul(&reciprocal)?
                                .sub(&input.first.mul(&input.second)?.mul(&reciprocal_squared)?)?,
                        }
                    }
                    TensorOp::Reshape { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.reshape(node.shape.clone())?,
                            first: input.first.reshape(node.shape.clone())?,
                            second: input.second.reshape(node.shape.clone())?,
                            mixed: input.mixed.reshape(node.shape.clone())?,
                        }
                    }
                    TensorOp::Mean { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.mean_all()?,
                            first: input.first.mean_all()?,
                            second: input.second.mean_all()?,
                            mixed: input.mixed.mean_all()?,
                        }
                    }
                    TensorOp::MeanAxis { input, axis } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let scale = 1.0 / input.value.shape[*axis] as f64;
                        MixedTangent {
                            value: input.value.reduce_axis(*axis, scale)?,
                            first: input.first.reduce_axis(*axis, scale)?,
                            second: input.second.reduce_axis(*axis, scale)?,
                            mixed: input.mixed.reduce_axis(*axis, scale)?,
                        }
                    }
                    TensorOp::Sin { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let value = input.value.sin()?;
                        let cosine = input.value.cos()?;
                        MixedTangent {
                            first: input.first.mul(&cosine)?,
                            second: input.second.mul(&cosine)?,
                            mixed: input
                                .mixed
                                .mul(&cosine)?
                                .sub(&input.first.mul(&input.second)?.mul(&value)?)?,
                            value,
                        }
                    }
                    TensorOp::Cos { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let value = input.value.cos()?;
                        let sine = input.value.sin()?;
                        MixedTangent {
                            first: input.first.mul(&sine)?.neg()?,
                            second: input.second.mul(&sine)?.neg()?,
                            mixed: input
                                .mixed
                                .mul(&sine)?
                                .neg()?
                                .sub(&input.first.mul(&input.second)?.mul(&value)?)?,
                            value,
                        }
                    }
                    TensorOp::Powi { input, exponent } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let value = input.value.powi(*exponent)?;
                        if *exponent == 0 {
                            MixedTangent {
                                value,
                                first: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                                second: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                                mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                            }
                        } else {
                            let first_derivative =
                                input.value.powi(*exponent - 1)?.scale(*exponent as f64)?;
                            let second_derivative = if *exponent < 2 {
                                DynamicTensor::filled(node.shape.clone(), 0.0)?
                            } else {
                                input
                                    .value
                                    .powi(*exponent - 2)?
                                    .scale((*exponent as f64) * ((*exponent - 1) as f64))?
                            };
                            MixedTangent {
                                first: input.first.mul(&first_derivative)?,
                                second: input.second.mul(&first_derivative)?,
                                mixed: input.mixed.mul(&first_derivative)?.add(
                                    &input.first.mul(&input.second)?.mul(&second_derivative)?,
                                )?,
                                value,
                            }
                        }
                    }
                    TensorOp::Transpose { input, axes } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.transpose(axes)?,
                            first: input.first.transpose(axes)?,
                            second: input.second.transpose(axes)?,
                            mixed: input.mixed.transpose(axes)?,
                        }
                    }
                    TensorOp::Exp { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let value = input.value.exp()?;
                        MixedTangent {
                            first: input.first.mul(&value)?,
                            second: input.second.mul(&value)?,
                            mixed: input
                                .mixed
                                .add(&input.first.mul(&input.second)?)?
                                .mul(&value)?,
                            value,
                        }
                    }
                    TensorOp::ScalarConstant { value } => MixedTangent {
                        value: DynamicTensor::filled(vec![], *value)?,
                        first: DynamicTensor::filled(vec![], 0.0)?,
                        second: DynamicTensor::filled(vec![], 0.0)?,
                        mixed: DynamicTensor::filled(vec![], 0.0)?,
                    },
                    TensorOp::Add { lhs, rhs } => {
                        let lhs = values
                            .get(*lhs)
                            .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                        let rhs = values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                        MixedTangent {
                            value: lhs.value.add(&rhs.value)?,
                            first: lhs.first.add(&rhs.first)?,
                            second: lhs.second.add(&rhs.second)?,
                            mixed: lhs.mixed.add(&rhs.mixed)?,
                        }
                    }
                    TensorOp::Sub { lhs, rhs } => {
                        let lhs = values
                            .get(*lhs)
                            .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                        let rhs = values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                        MixedTangent {
                            value: lhs.value.sub(&rhs.value)?,
                            first: lhs.first.sub(&rhs.first)?,
                            second: lhs.second.sub(&rhs.second)?,
                            mixed: lhs.mixed.sub(&rhs.mixed)?,
                        }
                    }
                    TensorOp::Div { lhs, rhs } => {
                        let lhs = values
                            .get(*lhs)
                            .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                        let rhs = values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                        let reciprocal = rhs.value.reciprocal()?;
                        let reciprocal_squared = reciprocal.mul(&reciprocal)?;
                        let reciprocal_cubed = reciprocal_squared.mul(&reciprocal)?;
                        MixedTangent {
                            value: lhs.value.mul(&reciprocal)?,
                            first: lhs
                                .first
                                .mul(&reciprocal)?
                                .sub(&lhs.value.mul(&rhs.first)?.mul(&reciprocal_squared)?)?,
                            second: lhs
                                .second
                                .mul(&reciprocal)?
                                .sub(&lhs.value.mul(&rhs.second)?.mul(&reciprocal_squared)?)?,
                            mixed: lhs
                                .mixed
                                .mul(&reciprocal)?
                                .sub(&lhs.first.mul(&rhs.second)?.mul(&reciprocal_squared)?)?
                                .sub(&lhs.second.mul(&rhs.first)?.mul(&reciprocal_squared)?)?
                                .sub(&lhs.value.mul(&rhs.mixed)?.mul(&reciprocal_squared)?)?
                                .add(
                                    &lhs.value
                                        .mul(&rhs.first)?
                                        .mul(&rhs.second)?
                                        .mul(&reciprocal_cubed)?
                                        .scale(2.0)?,
                                )?,
                        }
                    }
                    TensorOp::Mul { lhs, rhs } => {
                        let lhs = values
                            .get(*lhs)
                            .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                        let rhs = values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                        MixedTangent {
                            value: lhs.value.mul(&rhs.value)?,
                            first: lhs
                                .first
                                .mul(&rhs.value)?
                                .add(&lhs.value.mul(&rhs.first)?)?,
                            second: lhs
                                .second
                                .mul(&rhs.value)?
                                .add(&lhs.value.mul(&rhs.second)?)?,
                            mixed: lhs
                                .mixed
                                .mul(&rhs.value)?
                                .add(&lhs.first.mul(&rhs.second)?)?
                                .add(&lhs.second.mul(&rhs.first)?)?
                                .add(&lhs.value.mul(&rhs.mixed)?)?,
                        }
                    }
                    TensorOp::Greater { lhs, rhs } => {
                        let lhs = values
                            .get(*lhs)
                            .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                        let rhs = values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                        MixedTangent {
                            value: lhs.value.greater(&rhs.value)?,
                            first: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                            second: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                            mixed: DynamicTensor::filled(node.shape.clone(), 0.0)?,
                        }
                    }
                    TensorOp::Where {
                        condition,
                        on_true,
                        on_false,
                    } => {
                        let condition = values
                            .get(*condition)
                            .ok_or_else(|| format!("node {condition} has no evaluated value"))?;
                        let on_true = values
                            .get(*on_true)
                            .ok_or_else(|| format!("node {on_true} has no evaluated value"))?;
                        let on_false = values
                            .get(*on_false)
                            .ok_or_else(|| format!("node {on_false} has no evaluated value"))?;
                        MixedTangent {
                            value: condition
                                .value
                                .where_select(&on_true.value, &on_false.value)?,
                            first: condition
                                .value
                                .where_select(&on_true.first, &on_false.first)?,
                            second: condition
                                .value
                                .where_select(&on_true.second, &on_false.second)?,
                            mixed: condition
                                .value
                                .where_select(&on_true.mixed, &on_false.mixed)?,
                        }
                    }
                    TensorOp::Sum { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.sum_all()?,
                            first: input.first.sum_all()?,
                            second: input.second.sum_all()?,
                            mixed: input.mixed.sum_all()?,
                        }
                    }
                    TensorOp::SumAxis { input, axis } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.reduce_axis(*axis, 1.0)?,
                            first: input.first.reduce_axis(*axis, 1.0)?,
                            second: input.second.reduce_axis(*axis, 1.0)?,
                            mixed: input.mixed.reduce_axis(*axis, 1.0)?,
                        }
                    }
                    TensorOp::Matmul { lhs, rhs } => {
                        let lhs = values
                            .get(*lhs)
                            .ok_or_else(|| format!("node {lhs} has no evaluated value"))?;
                        let rhs = values
                            .get(*rhs)
                            .ok_or_else(|| format!("node {rhs} has no evaluated value"))?;
                        MixedTangent {
                            value: lhs.value.matmul(&rhs.value)?,
                            first: lhs
                                .first
                                .matmul(&rhs.value)?
                                .add(&lhs.value.matmul(&rhs.first)?)?,
                            second: lhs
                                .second
                                .matmul(&rhs.value)?
                                .add(&lhs.value.matmul(&rhs.second)?)?,
                            mixed: lhs
                                .mixed
                                .matmul(&rhs.value)?
                                .add(&lhs.first.matmul(&rhs.second)?)?
                                .add(&lhs.second.matmul(&rhs.first)?)?
                                .add(&lhs.value.matmul(&rhs.mixed)?)?,
                        }
                    }
                    TensorOp::Tanh { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        let value = input.value.tanh()?;
                        let derivative = value.tanh_derivative_from_output()?;
                        let second_derivative = value.tanh_second_derivative_from_output()?;
                        MixedTangent {
                            first: input.first.mul(&derivative)?,
                            second: input.second.mul(&derivative)?,
                            mixed: input
                                .mixed
                                .mul(&derivative)?
                                .add(&input.first.mul(&input.second)?.mul(&second_derivative)?)?,
                            value,
                        }
                    }
                    TensorOp::Concat { inputs, axis } => {
                        let parts = |select: fn(&MixedTangent) -> &DynamicTensor| {
                            inputs
                                .iter()
                                .map(|input| {
                                    values.get(*input).map(select).ok_or_else(|| {
                                        format!("node {input} has no evaluated value")
                                    })
                                })
                                .collect::<Result<Vec<_>, String>>()
                        };
                        MixedTangent {
                            value: DynamicTensor::concat(&parts(|value| &value.value)?, *axis)?,
                            first: DynamicTensor::concat(&parts(|value| &value.first)?, *axis)?,
                            second: DynamicTensor::concat(&parts(|value| &value.second)?, *axis)?,
                            mixed: DynamicTensor::concat(&parts(|value| &value.mixed)?, *axis)?,
                        }
                    }
                    TensorOp::Slice {
                        input,
                        axis,
                        start,
                        length,
                    } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.slice_axis(*axis, *start, *length)?,
                            first: input.first.slice_axis(*axis, *start, *length)?,
                            second: input.second.slice_axis(*axis, *start, *length)?,
                            mixed: input.mixed.slice_axis(*axis, *start, *length)?,
                        }
                    }
                    TensorOp::PadSlice { input, axis, start } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.pad_slice(&node.shape, *axis, *start)?,
                            first: input.first.pad_slice(&node.shape, *axis, *start)?,
                            second: input.second.pad_slice(&node.shape, *axis, *start)?,
                            mixed: input.mixed.pad_slice(&node.shape, *axis, *start)?,
                        }
                    }
                    TensorOp::Broadcast { input } => {
                        let input = values
                            .get(*input)
                            .ok_or_else(|| format!("node {input} has no evaluated value"))?;
                        MixedTangent {
                            value: input.value.broadcast_to_shape(&node.shape)?,
                            first: input.first.broadcast_to_shape(&node.shape)?,
                            second: input.second.broadcast_to_shape(&node.shape)?,
                            mixed: input.mixed.broadcast_to_shape(&node.shape)?,
                        }
                    }
                };
            values.push(value);
        }

        values
            .get(output)
            .cloned()
            .ok_or_else(|| format!("output node {output} has no value"))
    }

    fn node(&self, id: TensorNodeId) -> Result<&TensorNode, String> {
        self.nodes
            .get(id)
            .ok_or_else(|| format!("node {id} does not exist"))
    }
}

fn symbolic_accumulate(
    graph: &mut TensorIr,
    cotangents: &mut [Option<TensorNodeId>],
    target: TensorNodeId,
    contribution: TensorNodeId,
) -> Result<(), String> {
    cotangents[target] = Some(match cotangents[target] {
        Some(existing) => graph.add(existing, contribution)?,
        None => contribution,
    });
    Ok(())
}

fn symbolic_zero_like(graph: &mut TensorIr, value: TensorNodeId) -> Result<TensorNodeId, String> {
    graph.sub(value, value)
}

fn symbolic_broadcast_like(
    graph: &mut TensorIr,
    value: TensorNodeId,
    target: TensorNodeId,
) -> Result<TensorNodeId, String> {
    let zero = symbolic_zero_like(graph, target)?;
    let ones = graph.powi(zero, 0)?;
    graph.mul(value, ones)
}

fn symbolic_expand_reduced_axis(
    graph: &mut TensorIr,
    value: TensorNodeId,
    target: TensorNodeId,
    axis: usize,
) -> Result<TensorNodeId, String> {
    let target_shape = graph.node_shape(target)?;
    if axis >= target_shape.len() {
        return Err(format!(
            "cannot expand symbolic reduced axis {axis} for shape {target_shape:?}"
        ));
    }
    let mut expanded_shape = target_shape.clone();
    expanded_shape[axis] = 1;
    let value_shape = graph.node_shape(value)?;
    let expected_shape = reduced_shape(&target_shape, axis)?;
    if value_shape != expected_shape {
        return Err(format!(
            "cannot expand symbolic reduced tensor shape {value_shape:?} along axis {axis} to {target_shape:?}"
        ));
    }
    let expanded = graph.reshape(value, expanded_shape)?;
    graph.broadcast_to(expanded, target_shape)
}

fn symbolic_reduce_to_shape(
    graph: &mut TensorIr,
    value: TensorNodeId,
    source_shape: &[usize],
    target_shape: &[usize],
) -> Result<TensorNodeId, String> {
    if target_shape.len() > source_shape.len() {
        return Err(format!(
            "cannot symbolically reduce tensor shape {source_shape:?} to higher-rank shape {target_shape:?}"
        ));
    }
    let rank_offset = source_shape.len() - target_shape.len();
    for (source_extent, target_extent) in source_shape[rank_offset..].iter().zip(target_shape) {
        if source_extent != target_extent && *target_extent != 1 {
            return Err(format!(
                "cannot symbolically reduce tensor shape {source_shape:?} to {target_shape:?}"
            ));
        }
    }

    let mut reduced = value;
    for axis in (0..source_shape.len()).rev() {
        let must_reduce =
            axis < rank_offset || target_shape[axis - rank_offset] == 1 && source_shape[axis] != 1;
        if must_reduce {
            reduced = graph.sum_axis(reduced, axis as isize)?;
        }
    }
    graph.reshape(reduced, target_shape.to_vec())
}

fn symbolic_transpose_last_two(
    graph: &mut TensorIr,
    value: TensorNodeId,
    shape: &[usize],
) -> Result<TensorNodeId, String> {
    if shape.len() < 2 {
        return Err(format!(
            "symbolic VJP matmul requires rank-two operands, got shape {shape:?}"
        ));
    }
    let mut axes = (0..shape.len()).collect::<Vec<_>>();
    let last = axes.len() - 1;
    axes.swap(last - 1, last);
    graph.transpose(
        value,
        Some(axes.into_iter().map(|axis| axis as isize).collect()),
    )
}

impl TensorExecutionPlan {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn uses_fused_elementwise_kernel(&self) -> bool {
        self.fused_elementwise_output
    }

    pub fn output_shape(&self) -> Result<Vec<usize>, String> {
        self.nodes
            .get(self.output_node_id)
            .map(|node| node.shape.clone())
            .ok_or_else(|| "execution plan output node does not exist".to_string())
    }

    pub fn input_shape(&self, name: &str) -> Result<Vec<usize>, String> {
        self.nodes
            .iter()
            .find_map(|node| match &node.op {
                TensorOp::Input { name: candidate } if candidate == name => {
                    Some(node.shape.clone())
                }
                _ => None,
            })
            .ok_or_else(|| format!("execution plan input {name:?} does not exist"))
    }

    pub fn lower_text(&self) -> String {
        self.as_ir().lower_text()
    }

    pub fn kernel_ir(&self) -> TensorKernelProgram {
        TensorKernelProgram {
            nodes: self
                .nodes
                .iter()
                .enumerate()
                .map(|(id, node)| TensorKernelNode {
                    id,
                    op: tensor_op_name(&node.op).to_string(),
                    shape: node.shape.clone(),
                    inputs: tensor_op_inputs(&node.op),
                    name: match &node.op {
                        TensorOp::Input { name } => Some(name.clone()),
                        _ => None,
                    },
                })
                .collect(),
            output_node_id: self.output_node_id,
        }
    }

    /// Lowers a fused rank-N elementwise plan to one CUDA C kernel.
    ///
    /// The generated kernel deliberately excludes operations whose current CPU
    /// semantics need host-side validation or separate lowering: reductions,
    /// matmul, reshape, transpose, division, and log.
    pub fn cuda_source(&self) -> Result<String, String> {
        if !self.fused_elementwise_output {
            return Err(
                "CUDA lowering currently requires a pure fused elementwise output graph"
                    .to_string(),
            );
        }

        let output = self
            .nodes
            .get(self.output_node_id)
            .ok_or_else(|| "execution plan output node does not exist".to_string())?;
        let input_nodes = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| match &node.op {
                TensorOp::Input { name } => Some((id, name.as_str(), node.shape.as_slice())),
                _ => None,
            })
            .collect::<Vec<_>>();
        let expression = cuda_expression(&self.nodes, self.output_node_id)?;
        let parameters = input_nodes
            .iter()
            .map(|(id, _, _)| format!("const float* input_{id}"))
            .chain([
                "float* output".to_string(),
                "unsigned long long count".to_string(),
            ])
            .collect::<Vec<_>>()
            .join(", ");
        let offsets = input_nodes
            .iter()
            .map(|(id, _, shape)| cuda_broadcast_offset_function(*id, &output.shape, shape))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format!(
            "__device__ __forceinline__ float nabla_powi(float base, unsigned int exponent) {{\n\
    float result = 1.0f;\n\
    while (exponent != 0U) {{\n\
        if ((exponent & 1U) != 0U) result *= base;\n\
        base *= base;\n\
        exponent >>= 1U;\n\
    }}\n\
    return result;\n\
}}\n\
{offsets}\n\
extern \"C\" __global__ void nabla_fused_elementwise({parameters}) {{\n\
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;\n\
    if (index < count) output[index] = {expression};\n\
}}\n"
        ))
    }

    pub fn evaluate(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        CpuBackend.execute(self, inputs)
    }

    fn execute_cpu(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<DynamicTensor, String> {
        if self.fused_elementwise_output {
            return evaluate_fused_elementwise(&self.nodes, self.output_node_id, inputs);
        }
        TensorIr::evaluate_tensor_nodes(&self.nodes, inputs)?
            .get(self.output_node_id)
            .cloned()
            .ok_or_else(|| format!("output node {} has no value", self.output_node_id))
    }

    pub fn vjp(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<BTreeMap<String, DynamicTensor>, String> {
        self.as_ir()
            .vjp(self.output_node_id, inputs, output_cotangent)
    }

    pub fn value_and_vjp(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        output_cotangent: DynamicTensor,
    ) -> Result<(DynamicTensor, BTreeMap<String, DynamicTensor>), String> {
        self.as_ir()
            .value_and_vjp(self.output_node_id, inputs, output_cotangent)
    }

    pub fn jvp(
        &self,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangents: &BTreeMap<String, DynamicTensor>,
    ) -> Result<(DynamicTensor, DynamicTensor), String> {
        self.as_ir()
            .jvp(self.output_node_id, inputs, input_tangents)
    }

    pub fn hessian_scalar(
        &self,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
    ) -> Result<Vec<Vec<f64>>, String> {
        self.as_ir()
            .hessian_scalar(self.output_node_id, input_name, inputs)
    }

    pub fn hvp_scalar(
        &self,
        input_name: &str,
        inputs: &BTreeMap<String, DynamicTensor>,
        input_tangent: DynamicTensor,
    ) -> Result<DynamicTensor, String> {
        self.as_ir()
            .hvp_scalar(self.output_node_id, input_name, inputs, input_tangent)
    }

    fn as_ir(&self) -> TensorIr {
        TensorIr {
            nodes: self.nodes.clone(),
        }
    }
}

fn cuda_expression(nodes: &[TensorNode], node_id: TensorNodeId) -> Result<String, String> {
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("CUDA lowering references missing node {node_id}"))?;
    let child = |child_id| cuda_expression(nodes, child_id);
    match &node.op {
        TensorOp::Input { .. } => Ok(format!("input_{node_id}[nabla_offset_{node_id}(index)]")),
        TensorOp::ScalarConstant { value } if value.is_finite() => Ok(cuda_scalar_literal(*value)),
        TensorOp::ScalarConstant { .. } => Err(
            "CUDA lowering does not support non-finite scalar constants".to_string(),
        ),
        TensorOp::Add { lhs, rhs } => Ok(format!("({} + {})", child(*lhs)?, child(*rhs)?)),
        TensorOp::Sub { lhs, rhs } => Ok(format!("({} - {})", child(*lhs)?, child(*rhs)?)),
        TensorOp::Mul { lhs, rhs } => Ok(format!("({} * {})", child(*lhs)?, child(*rhs)?)),
        TensorOp::Greater { lhs, rhs } => Ok(format!(
            "(({} > {}) ? 1.0f : 0.0f)",
            child(*lhs)?,
            child(*rhs)?
        )),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => Ok(format!(
            "(({} != 0.0f) ? {} : {})",
            child(*condition)?,
            child(*on_true)?,
            child(*on_false)?
        )),
        TensorOp::Tanh { input } => Ok(format!("tanhf({})", child(*input)?)),
        TensorOp::Exp { input } => Ok(format!("expf({})", child(*input)?)),
        TensorOp::Sin { input } => Ok(format!("sinf({})", child(*input)?)),
        TensorOp::Cos { input } => Ok(format!("cosf({})", child(*input)?)),
        TensorOp::Powi { input, exponent } => Ok(format!(
            "nabla_powi({}, {}U)",
            child(*input)?,
            exponent
        )),
        TensorOp::Div { .. } | TensorOp::Log { .. } => Err(
            "CUDA lowering does not yet support div or log because their CPU execution has checked domain semantics"
                .to_string(),
        ),
        TensorOp::Sum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Broadcast { .. } => Err(format!(
            "CUDA lowering does not yet support {}",
            tensor_op_name(&node.op)
        )),
    }
}

fn cuda_broadcast_offset_function(
    node_id: TensorNodeId,
    output_shape: &[usize],
    input_shape: &[usize],
) -> String {
    let input_strides = contiguous_strides(input_shape);
    let rank_offset = output_shape.len() - input_shape.len();
    let terms = (0..output_shape.len())
        .filter_map(|axis| {
            if axis < rank_offset {
                return None;
            }
            let input_axis = axis - rank_offset;
            (input_shape[input_axis] != 1).then(|| {
                let output_stride = output_shape[axis + 1..].iter().product::<usize>();
                format!(
                    "((index / {output_stride}ULL) % {}ULL) * {}ULL",
                    output_shape[axis], input_strides[input_axis]
                )
            })
        })
        .collect::<Vec<_>>();
    let offset = if terms.is_empty() {
        "0ULL".to_string()
    } else {
        terms.join(" + ")
    };
    format!(
        "__device__ __forceinline__ unsigned long long nabla_offset_{node_id}(unsigned long long index) {{ return {offset}; }}"
    )
}

fn cuda_scalar_literal(value: f64) -> String {
    let value = value as f32;
    if value.fract() == 0.0 {
        format!("{value:.1}f")
    } else {
        format!("{value:?}f")
    }
}

fn is_fusable_elementwise_subgraph(nodes: &[TensorNode], node_id: TensorNodeId) -> bool {
    let Some(node) = nodes.get(node_id) else {
        return false;
    };
    match &node.op {
        TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } => true,
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Greater { lhs, rhs } => {
            is_fusable_elementwise_subgraph(nodes, *lhs)
                && is_fusable_elementwise_subgraph(nodes, *rhs)
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => {
            is_fusable_elementwise_subgraph(nodes, *condition)
                && is_fusable_elementwise_subgraph(nodes, *on_true)
                && is_fusable_elementwise_subgraph(nodes, *on_false)
        }
        TensorOp::Tanh { input }
        | TensorOp::Exp { input }
        | TensorOp::Sin { input }
        | TensorOp::Cos { input }
        | TensorOp::Powi { input, .. } => is_fusable_elementwise_subgraph(nodes, *input),
        TensorOp::Sum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Div { .. }
        | TensorOp::Log { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Broadcast { .. } => false,
    }
}

fn evaluate_fused_elementwise(
    nodes: &[TensorNode],
    output_node_id: TensorNodeId,
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<DynamicTensor, String> {
    let output = nodes
        .get(output_node_id)
        .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;
    for node in nodes {
        if let TensorOp::Input { name } = &node.op {
            let input = inputs
                .get(name)
                .ok_or_else(|| format!("missing input {name:?}"))?;
            if input.shape != node.shape {
                return Err(format!(
                    "input {name:?} has shape {:?}, expected {:?}",
                    input.shape, node.shape
                ));
            }
        }
    }

    let count = element_count(&output.shape)?;
    let mut data = Vec::with_capacity(count);
    for index in 0..count {
        data.push(evaluate_fused_element(
            nodes,
            output_node_id,
            index,
            &output.shape,
            inputs,
        )?);
    }
    DynamicTensor::new(output.shape.clone(), data)
}

fn evaluate_fused_element(
    nodes: &[TensorNode],
    node_id: TensorNodeId,
    output_index: usize,
    output_shape: &[usize],
    inputs: &BTreeMap<String, DynamicTensor>,
) -> Result<f64, String> {
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("node {node_id} does not exist"))?;
    let child =
        |child_id| evaluate_fused_element(nodes, child_id, output_index, output_shape, inputs);
    match &node.op {
        TensorOp::Input { name } => {
            let input = inputs
                .get(name)
                .ok_or_else(|| format!("missing input {name:?}"))?;
            let strides = contiguous_strides(&node.shape);
            Ok(input.data[broadcast_offset(output_index, output_shape, &node.shape, &strides)])
        }
        TensorOp::ScalarConstant { value } => Ok(*value),
        TensorOp::Add { lhs, rhs } => Ok(child(*lhs)? + child(*rhs)?),
        TensorOp::Sub { lhs, rhs } => Ok(child(*lhs)? - child(*rhs)?),
        TensorOp::Div { lhs, rhs } => {
            let denominator = child(*rhs)?;
            if denominator == 0.0 {
                return Err("division by zero is not supported".to_string());
            }
            Ok(child(*lhs)? / denominator)
        }
        TensorOp::Mul { lhs, rhs } => Ok(child(*lhs)? * child(*rhs)?),
        TensorOp::Greater { lhs, rhs } => Ok(f64::from(child(*lhs)? > child(*rhs)?)),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => {
            if child(*condition)? != 0.0 {
                child(*on_true)
            } else {
                child(*on_false)
            }
        }
        TensorOp::Tanh { input } => Ok(child(*input)?.tanh()),
        TensorOp::Exp { input } => Ok(child(*input)?.exp()),
        TensorOp::Sin { input } => Ok(child(*input)?.sin()),
        TensorOp::Cos { input } => Ok(child(*input)?.cos()),
        TensorOp::Powi { input, exponent } => {
            let exponent = i32::try_from(*exponent)
                .map_err(|_| "powi exponent must fit in a signed 32-bit integer".to_string())?;
            Ok(child(*input)?.powi(exponent))
        }
        TensorOp::Log { input } => {
            let value = child(*input)?;
            if value <= 0.0 {
                return Err("log requires strictly positive tensor values".to_string());
            }
            Ok(value.ln())
        }
        TensorOp::Sum { .. }
        | TensorOp::SumAxis { .. }
        | TensorOp::Matmul { .. }
        | TensorOp::Reshape { .. }
        | TensorOp::Mean { .. }
        | TensorOp::MeanAxis { .. }
        | TensorOp::Transpose { .. }
        | TensorOp::Concat { .. }
        | TensorOp::Slice { .. }
        | TensorOp::PadSlice { .. }
        | TensorOp::Broadcast { .. } => Err(format!(
            "node {node_id} is not supported by the fused elementwise evaluator"
        )),
    }
}

fn tensor_op_inputs(op: &TensorOp) -> Vec<TensorNodeId> {
    match op {
        TensorOp::Input { .. } | TensorOp::ScalarConstant { .. } => Vec::new(),
        TensorOp::Add { lhs, rhs }
        | TensorOp::Sub { lhs, rhs }
        | TensorOp::Div { lhs, rhs }
        | TensorOp::Mul { lhs, rhs }
        | TensorOp::Greater { lhs, rhs }
        | TensorOp::Matmul { lhs, rhs } => {
            vec![*lhs, *rhs]
        }
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => vec![*condition, *on_true, *on_false],
        TensorOp::Sum { input }
        | TensorOp::SumAxis { input, .. }
        | TensorOp::Tanh { input }
        | TensorOp::Exp { input }
        | TensorOp::Reshape { input }
        | TensorOp::Mean { input }
        | TensorOp::MeanAxis { input, .. }
        | TensorOp::Sin { input }
        | TensorOp::Cos { input }
        | TensorOp::Powi { input, .. }
        | TensorOp::Transpose { input, .. }
        | TensorOp::Log { input }
        | TensorOp::Slice { input, .. }
        | TensorOp::PadSlice { input, .. }
        | TensorOp::Broadcast { input } => {
            vec![*input]
        }
        TensorOp::Concat { inputs, .. } => inputs.clone(),
    }
}

fn tensor_op_name(op: &TensorOp) -> &'static str {
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

fn pure_tensor_op_cse_key(op: &TensorOp, shape: &[usize]) -> Option<String> {
    let key = match op {
        TensorOp::Input { .. } => return None,
        TensorOp::ScalarConstant { value } => format!("constant:{value}:{shape:?}"),
        TensorOp::Add { lhs, rhs } => format!("add:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Sub { lhs, rhs } => format!("sub:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Div { lhs, rhs } => format!("div:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Mul { lhs, rhs } => format!("mul:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Greater { lhs, rhs } => format!("greater:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => format!("where:{condition}:{on_true}:{on_false}:{shape:?}"),
        TensorOp::Sum { input } => format!("sum:{input}:{shape:?}"),
        TensorOp::SumAxis { input, axis } => format!("sum_axis:{input}:{axis}:{shape:?}"),
        TensorOp::Matmul { lhs, rhs } => format!("matmul:{lhs}:{rhs}:{shape:?}"),
        TensorOp::Tanh { input } => format!("tanh:{input}:{shape:?}"),
        TensorOp::Exp { input } => format!("exp:{input}:{shape:?}"),
        TensorOp::Reshape { input } => format!("reshape:{input}:{shape:?}"),
        TensorOp::Mean { input } => format!("mean:{input}:{shape:?}"),
        TensorOp::MeanAxis { input, axis } => format!("mean_axis:{input}:{axis}:{shape:?}"),
        TensorOp::Sin { input } => format!("sin:{input}:{shape:?}"),
        TensorOp::Cos { input } => format!("cos:{input}:{shape:?}"),
        TensorOp::Powi { input, exponent } => format!("powi:{input}:{exponent}:{shape:?}"),
        TensorOp::Transpose { input, axes } => format!("transpose:{input}:{axes:?}:{shape:?}"),
        TensorOp::Log { input } => format!("log:{input}:{shape:?}"),
        TensorOp::Concat { inputs, axis } => format!("concat:{inputs:?}:{axis}:{shape:?}"),
        TensorOp::Slice {
            input,
            axis,
            start,
            length,
        } => {
            format!("slice:{input}:{axis}:{start}:{length}:{shape:?}")
        }
        TensorOp::PadSlice { input, axis, start } => {
            format!("pad_slice:{input}:{axis}:{start}:{shape:?}")
        }
        TensorOp::Broadcast { input } => format!("broadcast:{input}:{shape:?}"),
    };
    Some(key)
}

fn remap_tensor_op(
    op: &TensorOp,
    remap: &HashMap<TensorNodeId, TensorNodeId>,
) -> Result<TensorOp, String> {
    let remap_node = |node_id: TensorNodeId| {
        remap
            .get(&node_id)
            .copied()
            .ok_or_else(|| format!("node {node_id} is missing from execution plan remap"))
    };
    match op {
        TensorOp::Input { name } => Ok(TensorOp::Input { name: name.clone() }),
        TensorOp::ScalarConstant { value } => Ok(TensorOp::ScalarConstant { value: *value }),
        TensorOp::Add { lhs, rhs } => Ok(TensorOp::Add {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Sub { lhs, rhs } => Ok(TensorOp::Sub {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Div { lhs, rhs } => Ok(TensorOp::Div {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Mul { lhs, rhs } => Ok(TensorOp::Mul {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Greater { lhs, rhs } => Ok(TensorOp::Greater {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Where {
            condition,
            on_true,
            on_false,
        } => Ok(TensorOp::Where {
            condition: remap_node(*condition)?,
            on_true: remap_node(*on_true)?,
            on_false: remap_node(*on_false)?,
        }),
        TensorOp::Sum { input } => Ok(TensorOp::Sum {
            input: remap_node(*input)?,
        }),
        TensorOp::SumAxis { input, axis } => Ok(TensorOp::SumAxis {
            input: remap_node(*input)?,
            axis: *axis,
        }),
        TensorOp::Matmul { lhs, rhs } => Ok(TensorOp::Matmul {
            lhs: remap_node(*lhs)?,
            rhs: remap_node(*rhs)?,
        }),
        TensorOp::Tanh { input } => Ok(TensorOp::Tanh {
            input: remap_node(*input)?,
        }),
        TensorOp::Exp { input } => Ok(TensorOp::Exp {
            input: remap_node(*input)?,
        }),
        TensorOp::Reshape { input } => Ok(TensorOp::Reshape {
            input: remap_node(*input)?,
        }),
        TensorOp::Mean { input } => Ok(TensorOp::Mean {
            input: remap_node(*input)?,
        }),
        TensorOp::MeanAxis { input, axis } => Ok(TensorOp::MeanAxis {
            input: remap_node(*input)?,
            axis: *axis,
        }),
        TensorOp::Sin { input } => Ok(TensorOp::Sin {
            input: remap_node(*input)?,
        }),
        TensorOp::Cos { input } => Ok(TensorOp::Cos {
            input: remap_node(*input)?,
        }),
        TensorOp::Powi { input, exponent } => Ok(TensorOp::Powi {
            input: remap_node(*input)?,
            exponent: *exponent,
        }),
        TensorOp::Transpose { input, axes } => Ok(TensorOp::Transpose {
            input: remap_node(*input)?,
            axes: axes.clone(),
        }),
        TensorOp::Log { input } => Ok(TensorOp::Log {
            input: remap_node(*input)?,
        }),
        TensorOp::Concat { inputs, axis } => Ok(TensorOp::Concat {
            inputs: inputs
                .iter()
                .map(|input| remap_node(*input))
                .collect::<Result<Vec<_>, _>>()?,
            axis: *axis,
        }),
        TensorOp::Slice {
            input,
            axis,
            start,
            length,
        } => Ok(TensorOp::Slice {
            input: remap_node(*input)?,
            axis: *axis,
            start: *start,
            length: *length,
        }),
        TensorOp::PadSlice { input, axis, start } => Ok(TensorOp::PadSlice {
            input: remap_node(*input)?,
            axis: *axis,
            start: *start,
        }),
        TensorOp::Broadcast { input } => Ok(TensorOp::Broadcast {
            input: remap_node(*input)?,
        }),
    }
}

fn accumulate(slot: &mut Option<DynamicTensor>, contribution: DynamicTensor) -> Result<(), String> {
    match slot {
        Some(existing) => {
            *existing = existing.add(&contribution)?;
        }
        None => *slot = Some(contribution),
    }
    Ok(())
}

fn input_value(
    inputs: &BTreeMap<String, DynamicTensor>,
    name: &str,
    shape: &[usize],
) -> Result<DynamicTensor, String> {
    let input = inputs
        .get(name)
        .ok_or_else(|| format!("missing input {name:?}"))?;
    if input.shape != shape {
        return Err(format!(
            "input {name:?} has shape {:?}, expected {:?}",
            input.shape, shape
        ));
    }
    Ok(input.clone())
}

fn input_tangent_or_zero(
    tangents: &BTreeMap<String, DynamicTensor>,
    name: &str,
    shape: &[usize],
) -> Result<DynamicTensor, String> {
    match tangents.get(name) {
        Some(tangent) => {
            if tangent.shape != shape {
                return Err(format!(
                    "input tangent {name:?} has shape {:?}, expected {:?}",
                    tangent.shape, shape
                ));
            }
            Ok(tangent.clone())
        }
        None => DynamicTensor::filled(shape.to_vec(), 0.0),
    }
}

fn element_count(shape: &[usize]) -> Result<usize, String> {
    if shape.contains(&0) {
        return Err("tensor extents must be greater than zero".to_string());
    }
    shape.iter().try_fold(1usize, |count, extent| {
        count
            .checked_mul(*extent)
            .ok_or_else(|| "tensor element count overflows usize".to_string())
    })
}

fn concat_shape(shapes: &[&[usize]], axis: usize) -> Result<Vec<usize>, String> {
    let Some(first) = shapes.first() else {
        return Err("concat requires at least one input".to_string());
    };
    if axis >= first.len() {
        return Err(format!(
            "axis {axis} is out of bounds for rank {}",
            first.len()
        ));
    }
    let mut shape = first.to_vec();
    for candidate in &shapes[1..] {
        if candidate.len() != shape.len()
            || candidate
                .iter()
                .enumerate()
                .any(|(index, extent)| index != axis && *extent != shape[index])
        {
            return Err(format!(
                "cannot concatenate shapes {:?} and {:?} along axis {axis}",
                first, candidate
            ));
        }
        shape[axis] = shape[axis]
            .checked_add(candidate[axis])
            .ok_or_else(|| "concatenated tensor extent overflows usize".to_string())?;
    }
    Ok(shape)
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for axis in (1..shape.len()).rev() {
        strides[axis - 1] = strides[axis] * shape[axis];
    }
    strides
}

fn normalize_axis(axis: isize, rank: usize) -> Result<usize, String> {
    let normalized = if axis < 0 {
        axis.checked_add(rank as isize)
            .ok_or_else(|| format!("axis {axis} is out of bounds for rank {rank}"))?
    } else {
        axis
    };
    usize::try_from(normalized)
        .ok()
        .filter(|axis| *axis < rank)
        .ok_or_else(|| format!("axis {axis} is out of bounds for rank {rank}"))
}

fn reduced_shape(shape: &[usize], axis: usize) -> Result<Vec<usize>, String> {
    if axis >= shape.len() {
        return Err(format!(
            "axis {axis} is out of bounds for tensor shape {:?}",
            shape
        ));
    }
    let mut reduced = shape.to_vec();
    reduced.remove(axis);
    Ok(reduced)
}

fn normalize_permutation(axes: Option<Vec<isize>>, rank: usize) -> Result<Vec<usize>, String> {
    let axes = axes.unwrap_or_else(|| (0..rank).rev().map(|axis| axis as isize).collect());
    if axes.len() != rank {
        return Err(format!(
            "transpose axes must have length {rank}, got {}",
            axes.len()
        ));
    }
    let axes = axes
        .into_iter()
        .map(|axis| normalize_axis(axis, rank))
        .collect::<Result<Vec<_>, _>>()?;
    validate_permutation(&axes, rank)?;
    Ok(axes)
}

fn validate_permutation(axes: &[usize], rank: usize) -> Result<(), String> {
    if axes.len() != rank {
        return Err(format!(
            "transpose axes must have length {rank}, got {}",
            axes.len()
        ));
    }
    let mut seen = vec![false; rank];
    for axis in axes {
        if *axis >= rank || std::mem::replace(&mut seen[*axis], true) {
            return Err(format!(
                "transpose axes {:?} are not a permutation of 0..{rank}",
                axes
            ));
        }
    }
    Ok(())
}

fn inverse_permutation(axes: &[usize]) -> Result<Vec<usize>, String> {
    validate_permutation(axes, axes.len())?;
    let mut inverse = vec![0; axes.len()];
    for (output_axis, input_axis) in axes.iter().enumerate() {
        inverse[*input_axis] = output_axis;
    }
    Ok(inverse)
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    let rank = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(rank);
    for offset in 0..rank {
        let lhs_extent = lhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let rhs_extent = rhs.iter().rev().nth(offset).copied().unwrap_or(1);
        let extent = if lhs_extent == rhs_extent {
            lhs_extent
        } else if lhs_extent == 1 {
            rhs_extent
        } else if rhs_extent == 1 {
            lhs_extent
        } else {
            return Err(format!(
                "cannot broadcast tensor shapes {:?} and {:?}",
                lhs, rhs
            ));
        };
        shape.push(extent);
    }
    shape.reverse();
    Ok(shape)
}

fn matmul_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    if lhs.len() < 2 || rhs.len() < 2 {
        return Err(format!(
            "matmul requires tensors with at least two dimensions, got {:?} and {:?}",
            lhs, rhs
        ));
    }
    let lhs_inner = lhs[lhs.len() - 1];
    let rhs_inner = rhs[rhs.len() - 2];
    if lhs_inner != rhs_inner {
        return Err(format!(
            "cannot matmul tensor shapes {:?} and {:?}: inner dimensions {lhs_inner} and {rhs_inner} differ",
            lhs, rhs
        ));
    }

    let mut shape = broadcast_shape(&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2])?;
    shape.push(lhs[lhs.len() - 2]);
    shape.push(rhs[rhs.len() - 1]);
    Ok(shape)
}

fn broadcast_offset(
    output_index: usize,
    output_shape: &[usize],
    input_shape: &[usize],
    input_strides: &[usize],
) -> usize {
    let mut remaining = output_index;
    let mut offset = 0;
    let rank_offset = output_shape.len() - input_shape.len();
    for axis in (0..output_shape.len()).rev() {
        let coordinate = remaining % output_shape[axis];
        remaining /= output_shape[axis];
        if axis >= rank_offset {
            let input_axis = axis - rank_offset;
            if input_shape[input_axis] != 1 {
                offset += coordinate * input_strides[input_axis];
            }
        }
    }
    offset
}

fn format_shape(shape: &[usize]) -> String {
    if shape.is_empty() {
        return "tensor<f64>".to_string();
    }
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x");
    format!("tensor<{dimensions}xf64>")
}
