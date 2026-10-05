use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::matrix::PyMatrix;

#[derive(Clone, Debug)]
struct TraceNode {
    id: usize,
    op: TraceOp,
    shape: (usize, usize),
}

#[derive(Clone, Debug)]
enum TraceOp {
    Input {
        name: String,
    },
    ScalarConstant {
        value: f64,
    },
    Add {
        lhs: usize,
        rhs: usize,
    },
    Sub {
        lhs: usize,
        rhs: usize,
    },
    Mul {
        lhs: usize,
        rhs: usize,
    },
    Div {
        lhs: usize,
        rhs: usize,
    },
    Gt {
        lhs: usize,
        rhs: usize,
    },
    Where {
        mask: usize,
        on_true: usize,
        on_false: usize,
    },
    Concat {
        inputs: Vec<usize>,
        axis: usize,
    },
    Matmul {
        lhs: usize,
        rhs: usize,
    },
    Sum {
        input: usize,
        axis: Option<usize>,
    },
    Mean {
        input: usize,
        axis: Option<usize>,
    },
    Transpose {
        input: usize,
    },
    Reshape {
        input: usize,
    },
    Powf {
        input: usize,
        exponent: f64,
    },
    Tanh {
        input: usize,
    },
    Exp {
        input: usize,
    },
    Log {
        input: usize,
    },
    Sqrt {
        input: usize,
    },
    Sin {
        input: usize,
    },
    Cos {
        input: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum IrAttrValue {
    Int(usize),
    Float(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct IrNode {
    pub id: usize,
    pub op: String,
    pub shape: (usize, usize),
    pub inputs: Vec<usize>,
    pub name: Option<String>,
    pub attrs: BTreeMap<String, IrAttrValue>,
}

impl IrNode {
    fn new(id: usize, op: &str, shape: (usize, usize), inputs: Vec<usize>) -> Self {
        Self {
            id,
            op: op.to_string(),
            shape,
            inputs,
            name: None,
            attrs: BTreeMap::new(),
        }
    }

    fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    fn with_int_attr(mut self, key: &str, value: usize) -> Self {
        self.attrs.insert(key.to_string(), IrAttrValue::Int(value));
        self
    }

    fn with_float_attr(mut self, key: &str, value: f64) -> Self {
        self.attrs
            .insert(key.to_string(), IrAttrValue::Float(value));
        self
    }
}

fn format_backend_text_attr(value: &IrAttrValue) -> String {
    match value {
        IrAttrValue::Int(value) => value.to_string(),
        IrAttrValue::Float(value) => value.to_string(),
    }
}

fn format_backend_text_shape(shape: (usize, usize)) -> String {
    format!("tensor<{}x{}xf64>", shape.0, shape.1)
}

fn format_backend_text_node(node: &IrNode) -> String {
    let shape = format_backend_text_shape(node.shape);

    if node.op == "input" {
        return match &node.name {
            Some(name) => format!("%{} = input[name={name}] : {shape}", node.id),
            None => format!("%{} = input : {shape}", node.id),
        };
    }

    let inputs = node
        .inputs
        .iter()
        .map(|input| format!("%{input}"))
        .collect::<Vec<_>>()
        .join(", ");
    let attrs = if node.attrs.is_empty() {
        String::new()
    } else {
        let attrs = node
            .attrs
            .iter()
            .map(|(key, value)| format!("{key}={}", format_backend_text_attr(value)))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" {{{attrs}}}")
    };

    format!("%{} = {}({inputs}){attrs} : {shape}", node.id, node.op)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdIrNode {
    pub id: usize,
    pub op: String,
    pub shape: (usize, usize),
    pub inputs: Vec<usize>,
    pub target: Option<usize>,
}

#[derive(Debug, Default)]
struct TraceGraphState {
    nodes: Vec<TraceNode>,
}

#[pyclass(name = "TraceGraph", skip_from_py_object)]
#[derive(Clone, Debug, Default)]
pub struct TraceGraph {
    state: Arc<Mutex<TraceGraphState>>,
}

#[pyclass(name = "TraceMatrix", from_py_object)]
#[derive(Clone, Debug)]
pub struct TraceMatrix {
    graph: TraceGraph,
    node_id: usize,
    rows: usize,
    cols: usize,
}

#[pyclass(name = "TraceResult", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct TraceResult {
    graph: TraceGraph,
    output: TraceMatrix,
}

#[pyclass(name = "CpuExecutionPlan", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct CpuExecutionPlan {
    nodes: Vec<TraceNode>,
    root_node_id: usize,
    output_node_id: usize,
}

#[pyclass(name = "GradFunction", skip_from_py_object)]
pub struct GradFunction {
    plan: CpuExecutionPlan,
    output_cotangent: PyMatrix,
}

#[pyclass(name = "GradScalarFunction", skip_from_py_object)]
pub struct GradScalarFunction {
    plan: CpuExecutionPlan,
}

#[pyclass(name = "ValueAndGradFunction", skip_from_py_object)]
pub struct ValueAndGradFunction {
    plan: CpuExecutionPlan,
    output_cotangent: PyMatrix,
}

#[pyclass(name = "VjpFunction", skip_from_py_object)]
pub struct VjpFunction {
    plan: CpuExecutionPlan,
}

#[pyclass(name = "JacobianFunction", skip_from_py_object)]
pub struct JacobianFunction {
    plan: CpuExecutionPlan,
    input_name: String,
    input_shape: (usize, usize),
}

#[pyclass(name = "JacobiansFunction", skip_from_py_object)]
pub struct JacobiansFunction {
    plan: CpuExecutionPlan,
    input_specs: Vec<(String, (usize, usize))>,
}

#[pyclass(name = "JvpFunction", skip_from_py_object)]
pub struct JvpFunction {
    plan: CpuExecutionPlan,
    input_specs: Vec<(String, (usize, usize))>,
}

#[pyclass(name = "GradScalarTransform", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct GradScalarTransform {
    input_specs: Vec<(String, (usize, usize))>,
}

#[pyclass(name = "JitFunction", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct JitFunction {
    plan: CpuExecutionPlan,
}

#[pyclass(name = "JitTransform", skip_from_py_object)]
#[derive(Clone, Debug)]
pub struct JitTransform {
    input_specs: Vec<(String, (usize, usize))>,
}

impl TraceGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&self, name: &str, shape: (usize, usize)) -> Result<TraceMatrix, String> {
        let mut state = self.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Input {
                name: name.to_string(),
            },
            shape,
        });

        Ok(TraceMatrix {
            graph: self.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn describe_nodes(&self) -> Result<Vec<String>, String> {
        let state = self.lock_state()?;

        Ok(state
            .nodes
            .iter()
            .map(|node| match &node.op {
                TraceOp::Input { name } => {
                    format!(
                        "{} input {} shape=({}, {})",
                        node.id, name, node.shape.0, node.shape.1
                    )
                }
                TraceOp::ScalarConstant { value } => {
                    format!(
                        "{} scalar_constant {} shape=({}, {})",
                        node.id, value, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Add { lhs, rhs } => {
                    format!(
                        "{} add inputs=[{}, {}] shape=({}, {})",
                        node.id, lhs, rhs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Sub { lhs, rhs } => {
                    format!(
                        "{} sub inputs=[{}, {}] shape=({}, {})",
                        node.id, lhs, rhs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Mul { lhs, rhs } => {
                    format!(
                        "{} mul inputs=[{}, {}] shape=({}, {})",
                        node.id, lhs, rhs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Div { lhs, rhs } => {
                    format!(
                        "{} div inputs=[{}, {}] shape=({}, {})",
                        node.id, lhs, rhs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Gt { lhs, rhs } => {
                    format!(
                        "{} gt inputs=[{}, {}] shape=({}, {})",
                        node.id, lhs, rhs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Where {
                    mask,
                    on_true,
                    on_false,
                } => {
                    format!(
                        "{} where inputs=[{}, {}, {}] shape=({}, {})",
                        node.id, mask, on_true, on_false, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Concat { inputs, axis } => {
                    format!(
                        "{} concat axis={} inputs={:?} shape=({}, {})",
                        node.id, axis, inputs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Matmul { lhs, rhs } => {
                    format!(
                        "{} matmul inputs=[{}, {}] shape=({}, {})",
                        node.id, lhs, rhs, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Sum { input, axis } => {
                    format!(
                        "{} sum axis={:?} inputs=[{}] shape=({}, {})",
                        node.id, axis, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Mean { input, axis } => {
                    format!(
                        "{} mean axis={:?} inputs=[{}] shape=({}, {})",
                        node.id, axis, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Transpose { input } => {
                    format!(
                        "{} transpose inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Reshape { input } => {
                    format!(
                        "{} reshape inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Powf { input, exponent } => {
                    format!(
                        "{} powf exponent={} inputs=[{}] shape=({}, {})",
                        node.id, exponent, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Tanh { input } => {
                    format!(
                        "{} tanh inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Exp { input } => {
                    format!(
                        "{} exp inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Log { input } => {
                    format!(
                        "{} log inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Sqrt { input } => {
                    format!(
                        "{} sqrt inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Sin { input } => {
                    format!(
                        "{} sin inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
                TraceOp::Cos { input } => {
                    format!(
                        "{} cos inputs=[{}] shape=({}, {})",
                        node.id, input, node.shape.0, node.shape.1
                    )
                }
            })
            .collect())
    }

    pub fn ir_nodes(&self) -> Result<Vec<IrNode>, String> {
        let state = self.lock_state()?;

        Ok(state
            .nodes
            .iter()
            .map(|node| match &node.op {
                TraceOp::Input { name } => {
                    IrNode::new(node.id, "input", node.shape, Vec::new()).with_name(name.clone())
                }
                TraceOp::ScalarConstant { value } => {
                    IrNode::new(node.id, "scalar_constant", node.shape, Vec::new())
                        .with_float_attr("value", *value)
                }
                TraceOp::Add { lhs, rhs } => {
                    IrNode::new(node.id, "add", node.shape, vec![*lhs, *rhs])
                }
                TraceOp::Sub { lhs, rhs } => {
                    IrNode::new(node.id, "sub", node.shape, vec![*lhs, *rhs])
                }
                TraceOp::Mul { lhs, rhs } => {
                    IrNode::new(node.id, "mul", node.shape, vec![*lhs, *rhs])
                }
                TraceOp::Div { lhs, rhs } => {
                    IrNode::new(node.id, "div", node.shape, vec![*lhs, *rhs])
                }
                TraceOp::Gt { lhs, rhs } => {
                    IrNode::new(node.id, "gt", node.shape, vec![*lhs, *rhs])
                }
                TraceOp::Where {
                    mask,
                    on_true,
                    on_false,
                } => IrNode::new(
                    node.id,
                    "where",
                    node.shape,
                    vec![*mask, *on_true, *on_false],
                ),
                TraceOp::Concat { inputs, axis } => {
                    IrNode::new(node.id, "concat", node.shape, inputs.clone())
                        .with_int_attr("axis", *axis)
                }
                TraceOp::Matmul { lhs, rhs } => {
                    IrNode::new(node.id, "matmul", node.shape, vec![*lhs, *rhs])
                }
                TraceOp::Sum { input, axis } => {
                    let node = IrNode::new(node.id, "sum", node.shape, vec![*input]);
                    match axis {
                        Some(axis) => node.with_int_attr("axis", *axis),
                        None => node,
                    }
                }
                TraceOp::Mean { input, axis } => {
                    let node = IrNode::new(node.id, "mean", node.shape, vec![*input]);
                    match axis {
                        Some(axis) => node.with_int_attr("axis", *axis),
                        None => node,
                    }
                }
                TraceOp::Transpose { input } => {
                    IrNode::new(node.id, "transpose", node.shape, vec![*input])
                }
                TraceOp::Reshape { input } => {
                    IrNode::new(node.id, "reshape", node.shape, vec![*input])
                }
                TraceOp::Powf { input, exponent } => {
                    IrNode::new(node.id, "powf", node.shape, vec![*input])
                        .with_float_attr("exponent", *exponent)
                }
                TraceOp::Tanh { input } => IrNode::new(node.id, "tanh", node.shape, vec![*input]),
                TraceOp::Exp { input } => IrNode::new(node.id, "exp", node.shape, vec![*input]),
                TraceOp::Log { input } => IrNode::new(node.id, "log", node.shape, vec![*input]),
                TraceOp::Sqrt { input } => IrNode::new(node.id, "sqrt", node.shape, vec![*input]),
                TraceOp::Sin { input } => IrNode::new(node.id, "sin", node.shape, vec![*input]),
                TraceOp::Cos { input } => IrNode::new(node.id, "cos", node.shape, vec![*input]),
            })
            .collect())
    }

    pub fn lower_text(&self) -> Result<String, String> {
        Ok(self
            .ir_nodes()?
            .iter()
            .map(format_backend_text_node)
            .collect::<Vec<_>>()
            .join("\n"))
    }

    pub fn vjp_ir_nodes(&self, output_node_id: usize) -> Result<Vec<AdIrNode>, String> {
        let state = self.lock_state()?;
        let output = state
            .nodes
            .get(output_node_id)
            .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

        match &output.op {
            TraceOp::Add { lhs, rhs } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "identity".to_string(),
                    shape: output.shape,
                    inputs: vec![0],
                    target: Some(*lhs),
                },
                AdIrNode {
                    id: 2,
                    op: "identity".to_string(),
                    shape: output.shape,
                    inputs: vec![0],
                    target: Some(*rhs),
                },
            ]),
            TraceOp::Sub { lhs, rhs } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "identity".to_string(),
                    shape: output.shape,
                    inputs: vec![0],
                    target: Some(*lhs),
                },
                AdIrNode {
                    id: 2,
                    op: "neg".to_string(),
                    shape: output.shape,
                    inputs: vec![0],
                    target: Some(*rhs),
                },
            ]),
            TraceOp::Mul { lhs, rhs } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, *rhs],
                    target: Some(*lhs),
                },
                AdIrNode {
                    id: 2,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![*lhs, 0],
                    target: Some(*rhs),
                },
            ]),
            TraceOp::Div { lhs, rhs } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "div".to_string(),
                    shape: output.shape,
                    inputs: vec![0, *rhs],
                    target: Some(*lhs),
                },
                AdIrNode {
                    id: 2,
                    op: "square".to_string(),
                    shape: output.shape,
                    inputs: vec![*rhs],
                    target: None,
                },
                AdIrNode {
                    id: 3,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, *lhs],
                    target: None,
                },
                AdIrNode {
                    id: 4,
                    op: "div".to_string(),
                    shape: output.shape,
                    inputs: vec![3, 2],
                    target: None,
                },
                AdIrNode {
                    id: 5,
                    op: "neg".to_string(),
                    shape: output.shape,
                    inputs: vec![4],
                    target: Some(*rhs),
                },
            ]),
            TraceOp::Where {
                mask,
                on_true,
                on_false,
            } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, *mask],
                    target: Some(*on_true),
                },
                AdIrNode {
                    id: 2,
                    op: "logical_not".to_string(),
                    shape: output.shape,
                    inputs: vec![*mask],
                    target: None,
                },
                AdIrNode {
                    id: 3,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 2],
                    target: Some(*on_false),
                },
            ]),
            TraceOp::Concat { inputs, axis } => {
                let mut nodes = Vec::with_capacity(inputs.len() + 1);
                nodes.push(AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                });

                for (index, input) in inputs.iter().enumerate() {
                    let input_node = state
                        .nodes
                        .get(*input)
                        .ok_or_else(|| format!("concat input node {input} does not exist"))?;
                    nodes.push(AdIrNode {
                        id: index + 1,
                        op: format!("slice_axis_{axis}"),
                        shape: input_node.shape,
                        inputs: vec![0],
                        target: Some(*input),
                    });
                }

                Ok(nodes)
            }
            TraceOp::Matmul { lhs, rhs } => {
                let lhs_node = state
                    .nodes
                    .get(*lhs)
                    .ok_or_else(|| format!("lhs node {lhs} does not exist"))?;
                let rhs_node = state
                    .nodes
                    .get(*rhs)
                    .ok_or_else(|| format!("rhs node {rhs} does not exist"))?;

                Ok(vec![
                    AdIrNode {
                        id: 0,
                        op: "cotangent_seed".to_string(),
                        shape: output.shape,
                        inputs: Vec::new(),
                        target: Some(output_node_id),
                    },
                    AdIrNode {
                        id: 1,
                        op: "transpose".to_string(),
                        shape: (rhs_node.shape.1, rhs_node.shape.0),
                        inputs: vec![*rhs],
                        target: None,
                    },
                    AdIrNode {
                        id: 2,
                        op: "matmul".to_string(),
                        shape: lhs_node.shape,
                        inputs: vec![0, 1],
                        target: Some(*lhs),
                    },
                    AdIrNode {
                        id: 3,
                        op: "transpose".to_string(),
                        shape: (lhs_node.shape.1, lhs_node.shape.0),
                        inputs: vec![*lhs],
                        target: None,
                    },
                    AdIrNode {
                        id: 4,
                        op: "matmul".to_string(),
                        shape: rhs_node.shape,
                        inputs: vec![3, 0],
                        target: Some(*rhs),
                    },
                ])
            }
            TraceOp::Sum { input, axis: _ } => {
                let input_node = state
                    .nodes
                    .get(*input)
                    .ok_or_else(|| format!("sum input node {input} does not exist"))?;

                Ok(vec![
                    AdIrNode {
                        id: 0,
                        op: "cotangent_seed".to_string(),
                        shape: output.shape,
                        inputs: Vec::new(),
                        target: Some(output_node_id),
                    },
                    AdIrNode {
                        id: 1,
                        op: "broadcast".to_string(),
                        shape: input_node.shape,
                        inputs: vec![0],
                        target: Some(*input),
                    },
                ])
            }
            TraceOp::Mean { input, axis } => {
                let input_node = state
                    .nodes
                    .get(*input)
                    .ok_or_else(|| format!("mean input node {input} does not exist"))?;
                let divisor = reduction_divisor(input_node.shape, *axis)?;

                Ok(vec![
                    AdIrNode {
                        id: 0,
                        op: "cotangent_seed".to_string(),
                        shape: output.shape,
                        inputs: Vec::new(),
                        target: Some(output_node_id),
                    },
                    AdIrNode {
                        id: 1,
                        op: "broadcast".to_string(),
                        shape: input_node.shape,
                        inputs: vec![0],
                        target: None,
                    },
                    AdIrNode {
                        id: 2,
                        op: format!("scale_by_{divisor}_reciprocal"),
                        shape: input_node.shape,
                        inputs: vec![1],
                        target: Some(*input),
                    },
                ])
            }
            TraceOp::Transpose { input } => {
                let input_node = state
                    .nodes
                    .get(*input)
                    .ok_or_else(|| format!("transpose input node {input} does not exist"))?;

                Ok(vec![
                    AdIrNode {
                        id: 0,
                        op: "cotangent_seed".to_string(),
                        shape: output.shape,
                        inputs: Vec::new(),
                        target: Some(output_node_id),
                    },
                    AdIrNode {
                        id: 1,
                        op: "transpose".to_string(),
                        shape: input_node.shape,
                        inputs: vec![0],
                        target: Some(*input),
                    },
                ])
            }
            TraceOp::Reshape { input } => {
                let input_node = state
                    .nodes
                    .get(*input)
                    .ok_or_else(|| format!("reshape input node {input} does not exist"))?;

                Ok(vec![
                    AdIrNode {
                        id: 0,
                        op: "cotangent_seed".to_string(),
                        shape: output.shape,
                        inputs: Vec::new(),
                        target: Some(output_node_id),
                    },
                    AdIrNode {
                        id: 1,
                        op: "reshape".to_string(),
                        shape: input_node.shape,
                        inputs: vec![0],
                        target: Some(*input),
                    },
                ])
            }
            TraceOp::Powf { input, .. } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "powf_derivative".to_string(),
                    shape: output.shape,
                    inputs: vec![*input],
                    target: None,
                },
                AdIrNode {
                    id: 2,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 1],
                    target: Some(*input),
                },
            ]),
            TraceOp::Tanh { input } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "tanh_derivative".to_string(),
                    shape: output.shape,
                    inputs: vec![output_node_id],
                    target: None,
                },
                AdIrNode {
                    id: 2,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 1],
                    target: Some(*input),
                },
            ]),
            TraceOp::Exp { input } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, output_node_id],
                    target: Some(*input),
                },
            ]),
            TraceOp::Log { input } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "reciprocal".to_string(),
                    shape: output.shape,
                    inputs: vec![*input],
                    target: None,
                },
                AdIrNode {
                    id: 2,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 1],
                    target: Some(*input),
                },
            ]),
            TraceOp::Sqrt { input } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "reciprocal".to_string(),
                    shape: output.shape,
                    inputs: vec![output_node_id],
                    target: None,
                },
                AdIrNode {
                    id: 2,
                    op: "scale_by_half".to_string(),
                    shape: output.shape,
                    inputs: vec![1],
                    target: None,
                },
                AdIrNode {
                    id: 3,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 2],
                    target: Some(*input),
                },
            ]),
            TraceOp::Sin { input } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "cos".to_string(),
                    shape: output.shape,
                    inputs: vec![*input],
                    target: None,
                },
                AdIrNode {
                    id: 2,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 1],
                    target: Some(*input),
                },
            ]),
            TraceOp::Cos { input } => Ok(vec![
                AdIrNode {
                    id: 0,
                    op: "cotangent_seed".to_string(),
                    shape: output.shape,
                    inputs: Vec::new(),
                    target: Some(output_node_id),
                },
                AdIrNode {
                    id: 1,
                    op: "sin".to_string(),
                    shape: output.shape,
                    inputs: vec![*input],
                    target: None,
                },
                AdIrNode {
                    id: 2,
                    op: "neg".to_string(),
                    shape: output.shape,
                    inputs: vec![1],
                    target: None,
                },
                AdIrNode {
                    id: 3,
                    op: "mul".to_string(),
                    shape: output.shape,
                    inputs: vec![0, 2],
                    target: Some(*input),
                },
            ]),
            TraceOp::Input { .. } | TraceOp::Gt { .. } => Err(
                "VJP is only implemented for add, sub, mul, div, where, matmul, sum, mean, transpose, reshape, powf, tanh, exp, log, sqrt, sin, and cos outputs"
                    .to_string(),
            ),
            TraceOp::ScalarConstant { .. } => Err(
                "VJP is only implemented for add, sub, mul, div, where, matmul, sum, mean, transpose, reshape, powf, tanh, exp, log, sqrt, sin, and cos outputs"
                    .to_string(),
            ),
        }
    }

    pub fn evaluate_vjp_cpu(
        &self,
        output_node_id: usize,
        inputs: HashMap<String, PyMatrix>,
        output_cotangent: &PyMatrix,
    ) -> Result<HashMap<String, PyMatrix>, String> {
        let nodes = {
            let state = self.lock_state()?;
            state.nodes.clone()
        };
        let output = nodes
            .get(output_node_id)
            .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

        if output_cotangent.dims() != output.shape {
            return Err(format!(
                "output cotangent shape {:?} does not match output shape {:?}",
                output_cotangent.dims(),
                output.shape
            ));
        }

        let values = evaluate_nodes(&nodes, output_node_id, &inputs)?;

        evaluate_vjp_from_values(&nodes, output_node_id, &values, output_cotangent)
    }

    pub fn evaluate_value_and_vjp_cpu(
        &self,
        output_node_id: usize,
        inputs: HashMap<String, PyMatrix>,
        output_cotangent: &PyMatrix,
    ) -> Result<(PyMatrix, HashMap<String, PyMatrix>), String> {
        let nodes = {
            let state = self.lock_state()?;
            state.nodes.clone()
        };
        let output = nodes
            .get(output_node_id)
            .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

        if output_cotangent.dims() != output.shape {
            return Err(format!(
                "output cotangent shape {:?} does not match output shape {:?}",
                output_cotangent.dims(),
                output.shape
            ));
        }

        let values = evaluate_nodes(&nodes, output_node_id, &inputs)?;
        let output_value = node_value(&values, output_node_id)?.clone();
        let gradients =
            evaluate_vjp_from_values(&nodes, output_node_id, &values, output_cotangent)?;

        Ok((output_value, gradients))
    }

    pub fn evaluate_jvp_cpu(
        &self,
        output_node_id: usize,
        inputs: HashMap<String, PyMatrix>,
        input_tangents: HashMap<String, PyMatrix>,
    ) -> Result<(PyMatrix, PyMatrix), String> {
        let nodes = {
            let state = self.lock_state()?;
            state.nodes.clone()
        };
        let values = evaluate_nodes(&nodes, output_node_id, &inputs)?;
        let tangents =
            evaluate_jvp_from_values(&nodes, output_node_id, &values, &input_tangents, None)?;

        Ok((node_value(&values, output_node_id)?.clone(), tangents))
    }

    pub fn compile_cpu(&self, output_node_id: usize) -> Result<CpuExecutionPlan, String> {
        let state = self.lock_state()?;
        state
            .nodes
            .get(output_node_id)
            .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

        let reachable = reachable_trace_node_ids(&state.nodes, output_node_id)?;
        let mut remapped_ids = HashMap::with_capacity(reachable.len());
        let mut nodes = Vec::with_capacity(reachable.len());
        let mut unary_cse_nodes = HashMap::new();

        for node in state.nodes.iter().take(output_node_id + 1) {
            if !reachable.contains(&node.id) {
                continue;
            }

            let id = nodes.len();
            let op = remap_trace_op(&node.op, &remapped_ids)?;
            let cse_key = unary_trace_op_cse_key(&op, node.shape);

            if let Some(existing_id) = cse_key.as_ref().and_then(|key| unary_cse_nodes.get(key)) {
                remapped_ids.insert(node.id, *existing_id);
                continue;
            }

            remapped_ids.insert(node.id, id);
            nodes.push(TraceNode {
                id,
                op,
                shape: node.shape,
            });

            if let Some(key) = cse_key {
                unary_cse_nodes.insert(key, id);
            }
        }

        let remapped_output_node_id = remapped_ids
            .get(&output_node_id)
            .copied()
            .ok_or_else(|| format!("output node {output_node_id} is not reachable"))?;

        Ok(CpuExecutionPlan {
            nodes,
            root_node_id: remapped_output_node_id,
            output_node_id,
        })
    }

    pub fn evaluate_cpu(
        &self,
        output_node_id: usize,
        inputs: HashMap<String, PyMatrix>,
    ) -> Result<PyMatrix, String> {
        let nodes = {
            let state = self.lock_state()?;
            state.nodes.clone()
        };
        evaluate_forward_nodes(&nodes, output_node_id, &inputs)
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, TraceGraphState>, String> {
        self.state
            .lock()
            .map_err(|_| "trace graph lock was poisoned".to_string())
    }
}

fn trace_op_inputs(op: &TraceOp) -> Vec<usize> {
    match op {
        TraceOp::Input { .. } | TraceOp::ScalarConstant { .. } => Vec::new(),
        TraceOp::Add { lhs, rhs }
        | TraceOp::Sub { lhs, rhs }
        | TraceOp::Mul { lhs, rhs }
        | TraceOp::Div { lhs, rhs }
        | TraceOp::Gt { lhs, rhs }
        | TraceOp::Matmul { lhs, rhs } => vec![*lhs, *rhs],
        TraceOp::Where {
            mask,
            on_true,
            on_false,
        } => vec![*mask, *on_true, *on_false],
        TraceOp::Concat { inputs, .. } => inputs.clone(),
        TraceOp::Sum { input, .. }
        | TraceOp::Mean { input, .. }
        | TraceOp::Transpose { input }
        | TraceOp::Reshape { input }
        | TraceOp::Powf { input, .. }
        | TraceOp::Tanh { input }
        | TraceOp::Exp { input }
        | TraceOp::Log { input }
        | TraceOp::Sqrt { input }
        | TraceOp::Sin { input }
        | TraceOp::Cos { input } => vec![*input],
    }
}

fn reachable_trace_node_ids(
    nodes: &[TraceNode],
    output_node_id: usize,
) -> Result<HashSet<usize>, String> {
    let mut reachable = HashSet::new();
    let mut pending = vec![output_node_id];

    while let Some(node_id) = pending.pop() {
        if !reachable.insert(node_id) {
            continue;
        }

        let node = nodes
            .get(node_id)
            .ok_or_else(|| format!("trace node {node_id} does not exist"))?;
        pending.extend(trace_op_inputs(&node.op));
    }

    Ok(reachable)
}

fn remap_trace_op(op: &TraceOp, remapped_ids: &HashMap<usize, usize>) -> Result<TraceOp, String> {
    let remap = |id: usize| {
        remapped_ids
            .get(&id)
            .copied()
            .ok_or_else(|| format!("trace node {id} was not remapped before its consumer"))
    };

    match op {
        TraceOp::Input { name } => Ok(TraceOp::Input { name: name.clone() }),
        TraceOp::ScalarConstant { value } => Ok(TraceOp::ScalarConstant { value: *value }),
        TraceOp::Add { lhs, rhs } => Ok(TraceOp::Add {
            lhs: remap(*lhs)?,
            rhs: remap(*rhs)?,
        }),
        TraceOp::Sub { lhs, rhs } => Ok(TraceOp::Sub {
            lhs: remap(*lhs)?,
            rhs: remap(*rhs)?,
        }),
        TraceOp::Mul { lhs, rhs } => Ok(TraceOp::Mul {
            lhs: remap(*lhs)?,
            rhs: remap(*rhs)?,
        }),
        TraceOp::Div { lhs, rhs } => Ok(TraceOp::Div {
            lhs: remap(*lhs)?,
            rhs: remap(*rhs)?,
        }),
        TraceOp::Gt { lhs, rhs } => Ok(TraceOp::Gt {
            lhs: remap(*lhs)?,
            rhs: remap(*rhs)?,
        }),
        TraceOp::Where {
            mask,
            on_true,
            on_false,
        } => Ok(TraceOp::Where {
            mask: remap(*mask)?,
            on_true: remap(*on_true)?,
            on_false: remap(*on_false)?,
        }),
        TraceOp::Concat { inputs, axis } => Ok(TraceOp::Concat {
            inputs: inputs
                .iter()
                .map(|input| remap(*input))
                .collect::<Result<Vec<_>, _>>()?,
            axis: *axis,
        }),
        TraceOp::Matmul { lhs, rhs } => Ok(TraceOp::Matmul {
            lhs: remap(*lhs)?,
            rhs: remap(*rhs)?,
        }),
        TraceOp::Sum { input, axis } => Ok(TraceOp::Sum {
            input: remap(*input)?,
            axis: *axis,
        }),
        TraceOp::Mean { input, axis } => Ok(TraceOp::Mean {
            input: remap(*input)?,
            axis: *axis,
        }),
        TraceOp::Transpose { input } => Ok(TraceOp::Transpose {
            input: remap(*input)?,
        }),
        TraceOp::Reshape { input } => Ok(TraceOp::Reshape {
            input: remap(*input)?,
        }),
        TraceOp::Powf { input, exponent } => Ok(TraceOp::Powf {
            input: remap(*input)?,
            exponent: *exponent,
        }),
        TraceOp::Tanh { input } => Ok(TraceOp::Tanh {
            input: remap(*input)?,
        }),
        TraceOp::Exp { input } => Ok(TraceOp::Exp {
            input: remap(*input)?,
        }),
        TraceOp::Log { input } => Ok(TraceOp::Log {
            input: remap(*input)?,
        }),
        TraceOp::Sqrt { input } => Ok(TraceOp::Sqrt {
            input: remap(*input)?,
        }),
        TraceOp::Sin { input } => Ok(TraceOp::Sin {
            input: remap(*input)?,
        }),
        TraceOp::Cos { input } => Ok(TraceOp::Cos {
            input: remap(*input)?,
        }),
    }
}

fn unary_trace_op_cse_key(op: &TraceOp, shape: (usize, usize)) -> Option<String> {
    let key = |name: &str, input: usize| format!("{name}:{}:{}:{input}", shape.0, shape.1);

    match op {
        TraceOp::Transpose { input } => Some(key("transpose", *input)),
        TraceOp::Reshape { input } => Some(key("reshape", *input)),
        TraceOp::Powf { input, exponent } => Some(format!(
            "powf:{}:{}:{}:{}",
            shape.0,
            shape.1,
            input,
            exponent.to_bits()
        )),
        TraceOp::Tanh { input } => Some(key("tanh", *input)),
        TraceOp::Exp { input } => Some(key("exp", *input)),
        TraceOp::Log { input } => Some(key("log", *input)),
        TraceOp::Sqrt { input } => Some(key("sqrt", *input)),
        TraceOp::Sin { input } => Some(key("sin", *input)),
        TraceOp::Cos { input } => Some(key("cos", *input)),
        _ => None,
    }
}

impl CpuExecutionPlan {
    pub fn evaluate_cpu(&self, inputs: HashMap<String, PyMatrix>) -> Result<PyMatrix, String> {
        evaluate_forward_nodes(&self.nodes, self.root_node_id, &inputs)
    }

    pub fn evaluate_vjp_cpu(
        &self,
        inputs: HashMap<String, PyMatrix>,
        output_cotangent: &PyMatrix,
    ) -> Result<HashMap<String, PyMatrix>, String> {
        self.validate_output_cotangent(output_cotangent)?;
        let values = evaluate_nodes(&self.nodes, self.root_node_id, &inputs)?;

        evaluate_vjp_from_values(&self.nodes, self.root_node_id, &values, output_cotangent)
    }

    pub fn evaluate_value_and_vjp_cpu(
        &self,
        inputs: HashMap<String, PyMatrix>,
        output_cotangent: &PyMatrix,
    ) -> Result<(PyMatrix, HashMap<String, PyMatrix>), String> {
        self.validate_output_cotangent(output_cotangent)?;
        let values = evaluate_nodes(&self.nodes, self.root_node_id, &inputs)?;
        let output_value = node_value(&values, self.root_node_id)?.clone();
        let gradients =
            evaluate_vjp_from_values(&self.nodes, self.root_node_id, &values, output_cotangent)?;

        Ok((output_value, gradients))
    }

    pub fn evaluate_jvp_cpu(
        &self,
        inputs: HashMap<String, PyMatrix>,
        input_tangents: HashMap<String, PyMatrix>,
    ) -> Result<(PyMatrix, PyMatrix), String> {
        let values = evaluate_nodes(&self.nodes, self.root_node_id, &inputs)?;
        let tangents = evaluate_jvp_from_values(
            &self.nodes,
            self.root_node_id,
            &values,
            &input_tangents,
            None,
        )?;

        Ok((node_value(&values, self.root_node_id)?.clone(), tangents))
    }

    fn output_shape(&self) -> (usize, usize) {
        self.nodes[self.root_node_id].shape
    }

    fn validate_output_cotangent(&self, output_cotangent: &PyMatrix) -> Result<(), String> {
        let output_shape = self.output_shape();
        if output_cotangent.dims() != output_shape {
            return Err(format!(
                "output cotangent shape {:?} does not match output shape {:?}",
                output_cotangent.dims(),
                output_shape
            ));
        }

        Ok(())
    }
}

fn evaluate_vjp_from_values(
    nodes: &[TraceNode],
    output_node_id: usize,
    values: &[PyMatrix],
    output_cotangent: &PyMatrix,
) -> Result<HashMap<String, PyMatrix>, String> {
    #[cfg(test)]
    legacy_ad_optimization_tests::VJP_EVALUATIONS.with(|count| count.set(count.get() + 1));
    let mut cotangents = vec![None; output_node_id + 1];
    cotangents[output_node_id] = Some(output_cotangent.clone());

    for node_id in (0..=output_node_id).rev() {
        // Input adjoints are returned; all other processed buffers can be released.
        if matches!(nodes[node_id].op, TraceOp::Input { .. }) {
            continue;
        }
        let Some(cotangent) = cotangents[node_id].take() else {
            continue;
        };

        match &nodes[node_id].op {
            TraceOp::Input { .. } => {}
            TraceOp::ScalarConstant { .. } => {}
            TraceOp::Add { lhs, rhs } => {
                let lhs_cotangent = reduce_cotangent_to_shape(&cotangent, nodes[*lhs].shape)?;
                let rhs_cotangent = reduce_cotangent_to_shape(&cotangent, nodes[*rhs].shape)?;

                accumulate_cotangent(&mut cotangents, nodes, *lhs, lhs_cotangent)?;
                accumulate_cotangent(&mut cotangents, nodes, *rhs, rhs_cotangent)?;
            }
            TraceOp::Sub { lhs, rhs } => {
                let negative_cotangent =
                    PyMatrix::filled(cotangent.dims().0, cotangent.dims().1, 0.0)
                        .try_sub(&cotangent)?;
                let lhs_cotangent = reduce_cotangent_to_shape(&cotangent, nodes[*lhs].shape)?;
                let rhs_cotangent =
                    reduce_cotangent_to_shape(&negative_cotangent, nodes[*rhs].shape)?;

                accumulate_cotangent(&mut cotangents, nodes, *lhs, lhs_cotangent)?;
                accumulate_cotangent(&mut cotangents, nodes, *rhs, rhs_cotangent)?;
            }
            TraceOp::Mul { lhs, rhs } => {
                let lhs_gradient = reduce_cotangent_to_shape(
                    &cotangent.try_mul(node_value(values, *rhs)?)?,
                    nodes[*lhs].shape,
                )?;
                let rhs_gradient = reduce_cotangent_to_shape(
                    &node_value(values, *lhs)?.try_mul(&cotangent)?,
                    nodes[*rhs].shape,
                )?;

                accumulate_cotangent(&mut cotangents, nodes, *lhs, lhs_gradient)?;
                accumulate_cotangent(&mut cotangents, nodes, *rhs, rhs_gradient)?;
            }
            TraceOp::Div { lhs, rhs } => {
                let lhs_value = node_value(values, *lhs)?;
                let rhs_value = node_value(values, *rhs)?;
                let lhs_gradient =
                    reduce_cotangent_to_shape(&cotangent.try_div(rhs_value)?, nodes[*lhs].shape)?;
                let rhs_squared = rhs_value.try_mul(rhs_value)?;
                let rhs_gradient = reduce_cotangent_to_shape(
                    &cotangent
                        .try_mul(lhs_value)?
                        .try_div(&rhs_squared)?
                        .try_mul_scalar(-1.0)?,
                    nodes[*rhs].shape,
                )?;

                accumulate_cotangent(&mut cotangents, nodes, *lhs, lhs_gradient)?;
                accumulate_cotangent(&mut cotangents, nodes, *rhs, rhs_gradient)?;
            }
            TraceOp::Gt { .. } => {}
            TraceOp::Where {
                mask,
                on_true,
                on_false,
            } => {
                let mask_value = node_value(values, *mask)?;
                let true_gradient = reduce_cotangent_to_shape(
                    &cotangent.try_mul(mask_value)?,
                    nodes[*on_true].shape,
                )?;
                let inverse_mask = PyMatrix::filled(mask_value.dims().0, mask_value.dims().1, 1.0)
                    .try_sub(mask_value)?;
                let false_gradient = reduce_cotangent_to_shape(
                    &cotangent.try_mul(&inverse_mask)?,
                    nodes[*on_false].shape,
                )?;

                accumulate_cotangent(&mut cotangents, nodes, *on_true, true_gradient)?;
                accumulate_cotangent(&mut cotangents, nodes, *on_false, false_gradient)?;
            }
            TraceOp::Concat { inputs, axis } => {
                let mut offset = 0;
                for input in inputs {
                    let input_node = nodes
                        .get(*input)
                        .ok_or_else(|| format!("concat input node {input} does not exist"))?;
                    let gradient = match axis {
                        0 => {
                            let slice = cotangent.try_slice_rows(offset, input_node.shape.0)?;
                            offset += input_node.shape.0;
                            slice
                        }
                        1 => {
                            let slice = cotangent.try_slice_cols(offset, input_node.shape.1)?;
                            offset += input_node.shape.1;
                            slice
                        }
                        _ => return Err(format!("axis must be 0 or 1, got {axis}")),
                    };

                    accumulate_cotangent(&mut cotangents, nodes, *input, gradient)?;
                }
            }
            TraceOp::Matmul { lhs, rhs } => {
                let lhs_value = node_value(values, *lhs)?;
                let rhs_value = node_value(values, *rhs)?;
                let lhs_gradient = cotangent.try_matmul(&rhs_value.transpose())?;
                let rhs_gradient = lhs_value.transpose().try_matmul(&cotangent)?;

                accumulate_cotangent(&mut cotangents, nodes, *lhs, lhs_gradient)?;
                accumulate_cotangent(&mut cotangents, nodes, *rhs, rhs_gradient)?;
            }
            TraceOp::Sum { input, axis } => {
                let input_node = nodes
                    .get(*input)
                    .ok_or_else(|| format!("sum input node {input} does not exist"))?;
                let input_gradient =
                    broadcast_reduction_cotangent(&cotangent, input_node.shape, *axis)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Mean { input, axis } => {
                let input_node = nodes
                    .get(*input)
                    .ok_or_else(|| format!("mean input node {input} does not exist"))?;
                let divisor = reduction_divisor(input_node.shape, *axis)?;
                let input_gradient =
                    broadcast_reduction_cotangent(&cotangent, input_node.shape, *axis)?
                        .try_div_scalar(divisor)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Transpose { input } => {
                accumulate_cotangent(&mut cotangents, nodes, *input, cotangent.transpose())?;
            }
            TraceOp::Reshape { input } => {
                let input_node = nodes
                    .get(*input)
                    .ok_or_else(|| format!("reshape input node {input} does not exist"))?;
                let input_cotangent =
                    cotangent.try_reshape(input_node.shape.0, input_node.shape.1)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_cotangent)?;
            }
            TraceOp::Powf { input, exponent } => {
                let derivative = node_value(values, *input)?
                    .elementwise_powf(exponent - 1.0)
                    .try_mul_scalar(*exponent)?;
                let input_gradient = cotangent.try_mul(&derivative)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Tanh { input } => {
                let output_value = node_value(values, node_id)?;
                let derivative = output_value
                    .try_mul(output_value)?
                    .try_mul_scalar(-1.0)?
                    .try_add_scalar(1.0)?;
                let input_gradient = cotangent.try_mul(&derivative)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Exp { input } => {
                let output_value = node_value(values, node_id)?;
                let input_gradient = cotangent.try_mul(output_value)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Log { input } => {
                let derivative = node_value(values, *input)?.elementwise_reciprocal();
                let input_gradient = cotangent.try_mul(&derivative)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Sqrt { input } => {
                let derivative = node_value(values, node_id)?
                    .elementwise_reciprocal()
                    .try_mul_scalar(0.5)?;
                let input_gradient = cotangent.try_mul(&derivative)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Sin { input } => {
                let derivative = node_value(values, *input)?.elementwise_cos();
                let input_gradient = cotangent.try_mul(&derivative)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
            TraceOp::Cos { input } => {
                let derivative = node_value(values, *input)?
                    .elementwise_sin()
                    .try_mul_scalar(-1.0)?;
                let input_gradient = cotangent.try_mul(&derivative)?;

                accumulate_cotangent(&mut cotangents, nodes, *input, input_gradient)?;
            }
        }
    }

    let mut gradients = HashMap::new();
    for (node, cotangent) in nodes.iter().take(output_node_id + 1).zip(cotangents) {
        let (TraceOp::Input { name }, Some(cotangent)) = (&node.op, cotangent) else {
            continue;
        };

        gradients.insert(name.clone(), cotangent);
    }

    Ok(gradients)
}

fn evaluate_jvp_from_values(
    nodes: &[TraceNode],
    output_node_id: usize,
    values: &[PyMatrix],
    input_tangents: &HashMap<String, PyMatrix>,
    derivative_active: Option<&[bool]>,
) -> Result<PyMatrix, String> {
    #[cfg(test)]
    legacy_ad_optimization_tests::JVP_EVALUATIONS.with(|count| count.set(count.get() + 1));
    nodes
        .get(output_node_id)
        .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

    let tangent_inputs = |node: &TraceNode| {
        if matches!(node.op, TraceOp::Gt { .. }) {
            Vec::new()
        } else {
            trace_op_inputs(&node.op)
        }
    };
    let is_active = |id: usize| derivative_active.is_none_or(|active| active[id]);
    let mut remaining_uses = vec![0usize; output_node_id + 1];
    for node in nodes
        .iter()
        .take(output_node_id + 1)
        .filter(|node| is_active(node.id))
    {
        for input in tangent_inputs(node) {
            remaining_uses[input] += 1;
        }
    }
    let mut tangents: Vec<Option<PyMatrix>> = Vec::with_capacity(output_node_id + 1);
    let tangent_value = |slots: &[Option<PyMatrix>], id: usize| {
        slots
            .get(id)
            .and_then(Option::as_ref)
            .cloned()
            .ok_or_else(|| format!("node {id} tangent has not been evaluated"))
    };
    for node in nodes.iter().take(output_node_id + 1) {
        if !is_active(node.id) {
            tangents.push(None);
            continue;
        }
        let tangent = match &node.op {
            TraceOp::Input { name } => {
                let tangent = input_tangents
                    .get(name)
                    .ok_or_else(|| format!("missing tangent matrix for {name}"))?;

                if tangent.dims() != node.shape {
                    return Err(format!(
                        "tangent {name} shape {:?} does not match traced shape {:?}",
                        tangent.dims(),
                        node.shape
                    ));
                }

                tangent.clone()
            }
            TraceOp::ScalarConstant { .. } | TraceOp::Gt { .. } => {
                PyMatrix::filled(node.shape.0, node.shape.1, 0.0)
            }
            TraceOp::Add { lhs, rhs } => tangent_value(&tangents, *lhs)?
                .try_add(&tangent_value(&tangents, *rhs)?)
                .map_err(|err| {
                    format!("failed to propagate add tangent at node {}: {err}", node.id)
                })?,
            TraceOp::Sub { lhs, rhs } => tangent_value(&tangents, *lhs)?
                .try_sub(&tangent_value(&tangents, *rhs)?)
                .map_err(|err| {
                    format!("failed to propagate sub tangent at node {}: {err}", node.id)
                })?,
            TraceOp::Mul { lhs, rhs } => {
                let lhs_term =
                    tangent_value(&tangents, *lhs)?.try_mul(node_value(values, *rhs)?)?;
                let rhs_term =
                    node_value(values, *lhs)?.try_mul(&tangent_value(&tangents, *rhs)?)?;
                lhs_term.try_add(&rhs_term).map_err(|err| {
                    format!("failed to propagate mul tangent at node {}: {err}", node.id)
                })?
            }
            TraceOp::Div { lhs, rhs } => {
                let lhs_term =
                    tangent_value(&tangents, *lhs)?.try_mul(node_value(values, *rhs)?)?;
                let rhs_term =
                    node_value(values, *lhs)?.try_mul(&tangent_value(&tangents, *rhs)?)?;
                let numerator = lhs_term.try_sub(&rhs_term)?;
                let denominator = node_value(values, *rhs)?.try_mul(node_value(values, *rhs)?)?;
                numerator.try_div(&denominator).map_err(|err| {
                    format!("failed to propagate div tangent at node {}: {err}", node.id)
                })?
            }
            TraceOp::Where {
                mask,
                on_true,
                on_false,
            } => PyMatrix::try_where(
                node_value(values, *mask)?,
                &tangent_value(&tangents, *on_true)?,
                &tangent_value(&tangents, *on_false)?,
            )
            .map_err(|err| {
                format!(
                    "failed to propagate where tangent at node {}: {err}",
                    node.id
                )
            })?,
            TraceOp::Concat { inputs, axis } => {
                let matrices = inputs
                    .iter()
                    .map(|input| tangent_value(&tangents, *input))
                    .collect::<Result<Vec<_>, _>>()?;
                PyMatrix::try_concat(&matrices, *axis).map_err(|err| {
                    format!(
                        "failed to propagate concat tangent at node {}: {err}",
                        node.id
                    )
                })?
            }
            TraceOp::Matmul { lhs, rhs } => {
                let lhs_term =
                    tangent_value(&tangents, *lhs)?.try_matmul(node_value(values, *rhs)?)?;
                let rhs_term =
                    node_value(values, *lhs)?.try_matmul(&tangent_value(&tangents, *rhs)?)?;
                lhs_term.try_add(&rhs_term).map_err(|err| {
                    format!(
                        "failed to propagate matmul tangent at node {}: {err}",
                        node.id
                    )
                })?
            }
            TraceOp::Sum { input, axis } => match axis {
                Some(axis) => tangent_value(&tangents, *input)?
                    .sum_axis(*axis)
                    .map_err(|err| {
                        format!("failed to propagate sum tangent at node {}: {err}", node.id)
                    })?,
                None => tangent_value(&tangents, *input)?.sum(),
            },
            TraceOp::Mean { input, axis } => match axis {
                Some(axis) => {
                    tangent_value(&tangents, *input)?
                        .mean_axis(*axis)
                        .map_err(|err| {
                            format!(
                                "failed to propagate mean tangent at node {}: {err}",
                                node.id
                            )
                        })?
                }
                None => tangent_value(&tangents, *input)?.mean(),
            },
            TraceOp::Transpose { input } => tangent_value(&tangents, *input)?.transpose(),
            TraceOp::Reshape { input } => tangent_value(&tangents, *input)?
                .try_reshape(node.shape.0, node.shape.1)
                .map_err(|err| {
                    format!(
                        "failed to propagate reshape tangent at node {}: {err}",
                        node.id
                    )
                })?,
            TraceOp::Powf { input, exponent } => {
                let derivative = node_value(values, *input)?
                    .elementwise_powf(exponent - 1.0)
                    .try_mul_scalar(*exponent)?;
                tangent_value(&tangents, *input)?
                    .try_mul(&derivative)
                    .map_err(|err| {
                        format!(
                            "failed to propagate powf tangent at node {}: {err}",
                            node.id
                        )
                    })?
            }
            TraceOp::Tanh { input } => {
                let derivative = node_value(values, node.id)?
                    .try_mul(node_value(values, node.id)?)?
                    .try_mul_scalar(-1.0)?
                    .try_add_scalar(1.0)?;
                tangent_value(&tangents, *input)?
                    .try_mul(&derivative)
                    .map_err(|err| {
                        format!(
                            "failed to propagate tanh tangent at node {}: {err}",
                            node.id
                        )
                    })?
            }
            TraceOp::Exp { input } => tangent_value(&tangents, *input)?
                .try_mul(node_value(values, node.id)?)
                .map_err(|err| {
                    format!("failed to propagate exp tangent at node {}: {err}", node.id)
                })?,
            TraceOp::Log { input } => tangent_value(&tangents, *input)?
                .try_div(node_value(values, *input)?)
                .map_err(|err| {
                    format!("failed to propagate log tangent at node {}: {err}", node.id)
                })?,
            TraceOp::Sqrt { input } => {
                let derivative = node_value(values, node.id)?
                    .elementwise_reciprocal()
                    .try_mul_scalar(0.5)?;
                tangent_value(&tangents, *input)?
                    .try_mul(&derivative)
                    .map_err(|err| {
                        format!(
                            "failed to propagate sqrt tangent at node {}: {err}",
                            node.id
                        )
                    })?
            }
            TraceOp::Sin { input } => tangent_value(&tangents, *input)?
                .try_mul(&node_value(values, *input)?.elementwise_cos())
                .map_err(|err| {
                    format!("failed to propagate sin tangent at node {}: {err}", node.id)
                })?,
            TraceOp::Cos { input } => {
                let derivative = node_value(values, *input)?
                    .elementwise_sin()
                    .try_mul_scalar(-1.0)?;
                tangent_value(&tangents, *input)?
                    .try_mul(&derivative)
                    .map_err(|err| {
                        format!("failed to propagate cos tangent at node {}: {err}", node.id)
                    })?
            }
        };

        if node.id == output_node_id {
            return Ok(tangent);
        }
        for input in tangent_inputs(node) {
            remaining_uses[input] -= 1;
            if remaining_uses[input] == 0 {
                tangents[input] = None;
            }
        }
        tangents.push((remaining_uses[node.id] != 0).then_some(tangent));
    }

    Err(format!(
        "node {output_node_id} tangent has not been evaluated"
    ))
}

fn evaluate_nodes(
    nodes: &[TraceNode],
    output_node_id: usize,
    inputs: &HashMap<String, PyMatrix>,
) -> Result<Vec<PyMatrix>, String> {
    #[cfg(test)]
    legacy_ad_optimization_tests::PRIMAL_EVALUATIONS.with(|count| count.set(count.get() + 1));
    nodes
        .get(output_node_id)
        .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

    let mut values = Vec::with_capacity(output_node_id + 1);
    for node in nodes.iter().take(output_node_id + 1) {
        let value = evaluate_node(node, inputs, |id| node_value(&values, id))?;

        values.push(value);
    }

    Ok(values)
}

fn evaluate_node<'a>(
    node: &TraceNode,
    inputs: &HashMap<String, PyMatrix>,
    lookup: impl Fn(usize) -> Result<&'a PyMatrix, String>,
) -> Result<PyMatrix, String> {
    let value = match &node.op {
        TraceOp::Input { name } => {
            let value = inputs
                .get(name)
                .ok_or_else(|| format!("missing input matrix for {name}"))?;

            if value.dims() != node.shape {
                return Err(format!(
                    "input {name} shape {:?} does not match traced shape {:?}",
                    value.dims(),
                    node.shape
                ));
            }

            value.clone()
        }
        TraceOp::ScalarConstant { value } => PyMatrix::filled(node.shape.0, node.shape.1, *value),
        TraceOp::Add { lhs, rhs } => lookup(*lhs)?
            .try_add(lookup(*rhs)?)
            .map_err(|err| format!("failed to evaluate add node {}: {err}", node.id))?,
        TraceOp::Sub { lhs, rhs } => lookup(*lhs)?
            .try_sub(lookup(*rhs)?)
            .map_err(|err| format!("failed to evaluate sub node {}: {err}", node.id))?,
        TraceOp::Mul { lhs, rhs } => lookup(*lhs)?
            .try_mul(lookup(*rhs)?)
            .map_err(|err| format!("failed to evaluate mul node {}: {err}", node.id))?,
        TraceOp::Div { lhs, rhs } => lookup(*lhs)?
            .try_div(lookup(*rhs)?)
            .map_err(|err| format!("failed to evaluate div node {}: {err}", node.id))?,
        TraceOp::Gt { lhs, rhs } => lookup(*lhs)?
            .try_gt(lookup(*rhs)?)
            .map_err(|err| format!("failed to evaluate gt node {}: {err}", node.id))?,
        TraceOp::Where {
            mask,
            on_true,
            on_false,
        } => PyMatrix::try_where(lookup(*mask)?, lookup(*on_true)?, lookup(*on_false)?)
            .map_err(|err| format!("failed to evaluate where node {}: {err}", node.id))?,
        TraceOp::Concat { inputs, axis } => {
            let matrices = inputs
                .iter()
                .map(|input| lookup(*input).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            PyMatrix::try_concat(&matrices, *axis)
                .map_err(|err| format!("failed to evaluate concat node {}: {err}", node.id))?
        }
        TraceOp::Matmul { lhs, rhs } => lookup(*lhs)?
            .try_matmul(lookup(*rhs)?)
            .map_err(|err| format!("failed to evaluate matmul node {}: {err}", node.id))?,
        TraceOp::Sum { input, axis } => match axis {
            Some(axis) => lookup(*input)?
                .sum_axis(*axis)
                .map_err(|err| format!("failed to evaluate sum node {}: {err}", node.id))?,
            None => lookup(*input)?.sum(),
        },
        TraceOp::Mean { input, axis } => match axis {
            Some(axis) => lookup(*input)?
                .mean_axis(*axis)
                .map_err(|err| format!("failed to evaluate mean node {}: {err}", node.id))?,
            None => lookup(*input)?.mean(),
        },
        TraceOp::Transpose { input } => lookup(*input)?.transpose(),
        TraceOp::Reshape { input } => lookup(*input)?
            .try_reshape(node.shape.0, node.shape.1)
            .map_err(|err| format!("failed to evaluate reshape node {}: {err}", node.id))?,
        TraceOp::Powf { input, exponent } => lookup(*input)?.elementwise_powf(*exponent),
        TraceOp::Tanh { input } => lookup(*input)?.elementwise_tanh(),
        TraceOp::Exp { input } => lookup(*input)?.elementwise_exp(),
        TraceOp::Log { input } => lookup(*input)?.elementwise_log(),
        TraceOp::Sqrt { input } => lookup(*input)?.elementwise_sqrt(),
        TraceOp::Sin { input } => lookup(*input)?.elementwise_sin(),
        TraceOp::Cos { input } => lookup(*input)?.elementwise_cos(),
    };

    if value.dims() != node.shape {
        return Err(format!(
            "node {} evaluated shape {:?} does not match traced shape {:?}",
            node.id,
            value.dims(),
            node.shape
        ));
    }
    Ok(value)
}

fn evaluate_forward_nodes(
    nodes: &[TraceNode],
    output_node_id: usize,
    inputs: &HashMap<String, PyMatrix>,
) -> Result<PyMatrix, String> {
    nodes
        .get(output_node_id)
        .ok_or_else(|| format!("output node {output_node_id} does not exist"))?;

    // Count every operand occurrence, including duplicates, without pruning eager work.
    let mut remaining_uses = vec![0usize; output_node_id + 1];
    for node in nodes.iter().take(output_node_id + 1) {
        for input in trace_op_inputs(&node.op) {
            if let Some(uses) = remaining_uses.get_mut(input) {
                *uses += 1;
            }
        }
    }

    let mut values: Vec<Option<PyMatrix>> = Vec::with_capacity(output_node_id + 1);
    for (node_index, node) in nodes.iter().take(output_node_id + 1).enumerate() {
        let value = evaluate_node(node, inputs, |id| {
            values
                .get(id)
                .and_then(Option::as_ref)
                .ok_or_else(|| format!("node {id} has not been evaluated"))
        })?;
        // Move the root result; differentiation still uses the complete primal tape.
        if node_index == output_node_id {
            return Ok(value);
        }
        for input in trace_op_inputs(&node.op) {
            remaining_uses[input] -= 1;
            if remaining_uses[input] == 0 {
                values[input] = None;
            }
        }
        values.push((remaining_uses[node_index] != 0).then_some(value));
    }
    Err(format!("node {output_node_id} has not been evaluated"))
}

fn node_value(values: &[PyMatrix], node_id: usize) -> Result<&PyMatrix, String> {
    values
        .get(node_id)
        .ok_or_else(|| format!("node {node_id} has not been evaluated"))
}

fn scalar_value(matrix: &PyMatrix) -> Result<f64, String> {
    if matrix.dims() != (1, 1) {
        return Err(format!(
            "expected scalar shape (1, 1), got {:?}",
            matrix.dims()
        ));
    }

    Ok(matrix.data()[0])
}

fn broadcast_extent(lhs: usize, rhs: usize) -> Option<usize> {
    if lhs == rhs {
        Some(lhs)
    } else if lhs == 1 {
        Some(rhs)
    } else if rhs == 1 {
        Some(lhs)
    } else {
        None
    }
}

fn elementwise_broadcast_shape(
    lhs: (usize, usize),
    rhs: (usize, usize),
    op: &str,
) -> Result<(usize, usize), String> {
    let rows = broadcast_extent(lhs.0, rhs.0)
        .ok_or_else(|| format!("incompatible trace {op} shapes: {lhs:?} and {rhs:?}"))?;
    let cols = broadcast_extent(lhs.1, rhs.1)
        .ok_or_else(|| format!("incompatible trace {op} shapes: {lhs:?} and {rhs:?}"))?;

    Ok((rows, cols))
}

fn reduction_shape(
    input_shape: (usize, usize),
    axis: Option<usize>,
) -> Result<(usize, usize), String> {
    match axis {
        Some(0) => Ok((1, input_shape.1)),
        Some(1) => Ok((input_shape.0, 1)),
        Some(axis) => Err(format!("axis must be 0 or 1, got {axis}")),
        None => Ok((1, 1)),
    }
}

fn reduction_divisor(input_shape: (usize, usize), axis: Option<usize>) -> Result<f64, String> {
    match axis {
        Some(0) => Ok(input_shape.0 as f64),
        Some(1) => Ok(input_shape.1 as f64),
        Some(axis) => Err(format!("axis must be 0 or 1, got {axis}")),
        None => Ok((input_shape.0 * input_shape.1) as f64),
    }
}

fn concat_shape(input_shapes: &[(usize, usize)], axis: usize) -> Result<(usize, usize), String> {
    let first = input_shapes
        .first()
        .ok_or_else(|| "concat requires at least one input".to_string())?;

    match axis {
        0 => {
            if input_shapes.iter().any(|shape| shape.1 != first.1) {
                return Err("concat axis 0 requires matching column counts".to_string());
            }

            Ok((input_shapes.iter().map(|shape| shape.0).sum(), first.1))
        }
        1 => {
            if input_shapes.iter().any(|shape| shape.0 != first.0) {
                return Err("concat axis 1 requires matching row counts".to_string());
            }

            Ok((first.0, input_shapes.iter().map(|shape| shape.1).sum()))
        }
        _ => Err(format!("axis must be 0 or 1, got {axis}")),
    }
}

fn broadcast_reduction_cotangent(
    cotangent: &PyMatrix,
    input_shape: (usize, usize),
    axis: Option<usize>,
) -> Result<PyMatrix, String> {
    match axis {
        Some(0) | Some(1) => {
            let zeros = PyMatrix::filled(input_shape.0, input_shape.1, 0.0);
            zeros.try_add(cotangent)
        }
        Some(axis) => Err(format!("axis must be 0 or 1, got {axis}")),
        None => {
            let scalar = scalar_value(cotangent)?;
            Ok(PyMatrix::filled(input_shape.0, input_shape.1, scalar))
        }
    }
}

fn reduce_cotangent_to_shape(
    cotangent: &PyMatrix,
    target_shape: (usize, usize),
) -> Result<PyMatrix, String> {
    if cotangent.dims() == target_shape {
        return Ok(cotangent.clone());
    }

    let broadcast_shape = elementwise_broadcast_shape(target_shape, cotangent.dims(), "cotangent")?;
    if broadcast_shape != cotangent.dims() {
        return Err(format!(
            "cannot reduce cotangent shape {:?} to target shape {:?}",
            cotangent.dims(),
            target_shape
        ));
    }

    let (cotangent_rows, cotangent_cols) = cotangent.dims();
    let mut rows = vec![vec![0.0; target_shape.1]; target_shape.0];
    for row in 0..cotangent_rows {
        for col in 0..cotangent_cols {
            let target_row = if target_shape.0 == 1 { 0 } else { row };
            let target_col = if target_shape.1 == 1 { 0 } else { col };
            rows[target_row][target_col] += cotangent.data()[row * cotangent_cols + col];
        }
    }

    PyMatrix::from_rows(rows)
}

fn accumulate_cotangent(
    cotangents: &mut [Option<PyMatrix>],
    nodes: &[TraceNode],
    node_id: usize,
    cotangent: PyMatrix,
) -> Result<(), String> {
    let node = nodes
        .get(node_id)
        .ok_or_else(|| format!("cotangent target node {node_id} does not exist"))?;

    if cotangent.dims() != node.shape {
        return Err(format!(
            "cotangent for node {node_id} has shape {:?}, expected {:?}",
            cotangent.dims(),
            node.shape
        ));
    }

    let slot = cotangents
        .get_mut(node_id)
        .ok_or_else(|| format!("cotangent target node {node_id} is outside the active graph"))?;

    match slot {
        Some(existing) => {
            existing.accumulate_owned_same_shape(&cotangent);
        }
        None => {
            *slot = Some(cotangent);
        }
    }

    Ok(())
}

impl TraceMatrix {
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn id(&self) -> usize {
        self.node_id
    }

    pub fn try_add(&self, rhs: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&self.graph.state, &rhs.graph.state) {
            return Err("trace add operands must belong to the same trace graph".to_string());
        }

        let shape = elementwise_broadcast_shape(self.dims(), rhs.dims(), "add")?;

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Add {
                lhs: self.node_id,
                rhs: rhs.node_id,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_add_scalar(&self, rhs: f64) -> Result<Self, String> {
        let rhs = self.try_scalar_constant(rhs)?;
        self.try_add(&rhs)
    }

    pub fn try_sub(&self, rhs: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&self.graph.state, &rhs.graph.state) {
            return Err("trace sub operands must belong to the same trace graph".to_string());
        }

        let shape = elementwise_broadcast_shape(self.dims(), rhs.dims(), "sub")?;

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Sub {
                lhs: self.node_id,
                rhs: rhs.node_id,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_sub_scalar(&self, rhs: f64) -> Result<Self, String> {
        let rhs = self.try_scalar_constant(rhs)?;
        self.try_sub(&rhs)
    }

    pub fn try_scalar_sub(&self, lhs: f64) -> Result<Self, String> {
        let lhs = self.try_scalar_constant(lhs)?;
        lhs.try_sub(self)
    }

    pub fn try_mul(&self, rhs: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&self.graph.state, &rhs.graph.state) {
            return Err("trace mul operands must belong to the same trace graph".to_string());
        }

        let shape = elementwise_broadcast_shape(self.dims(), rhs.dims(), "mul")?;

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Mul {
                lhs: self.node_id,
                rhs: rhs.node_id,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_div(&self, rhs: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&self.graph.state, &rhs.graph.state) {
            return Err("trace div operands must belong to the same trace graph".to_string());
        }

        let shape = elementwise_broadcast_shape(self.dims(), rhs.dims(), "div")?;

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Div {
                lhs: self.node_id,
                rhs: rhs.node_id,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_gt(&self, rhs: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&self.graph.state, &rhs.graph.state) {
            return Err("trace gt operands must belong to the same trace graph".to_string());
        }

        let shape = elementwise_broadcast_shape(self.dims(), rhs.dims(), "gt")?;

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Gt {
                lhs: self.node_id,
                rhs: rhs.node_id,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_where(mask: &Self, on_true: &Self, on_false: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&mask.graph.state, &on_true.graph.state)
            || !Arc::ptr_eq(&mask.graph.state, &on_false.graph.state)
        {
            return Err("trace where operands must belong to the same trace graph".to_string());
        }

        let value_shape = elementwise_broadcast_shape(on_true.dims(), on_false.dims(), "where")?;
        let shape = elementwise_broadcast_shape(mask.dims(), value_shape, "where")?;

        let mut state = mask.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Where {
                mask: mask.node_id,
                on_true: on_true.node_id,
                on_false: on_false.node_id,
            },
            shape,
        });

        Ok(Self {
            graph: mask.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_concat(inputs: &[Self], axis: usize) -> Result<Self, String> {
        let first = inputs
            .first()
            .ok_or_else(|| "trace concat requires at least one input".to_string())?;

        if inputs
            .iter()
            .any(|input| !Arc::ptr_eq(&first.graph.state, &input.graph.state))
        {
            return Err("trace concat operands must belong to the same trace graph".to_string());
        }

        let input_shapes = inputs.iter().map(Self::dims).collect::<Vec<_>>();
        let shape = concat_shape(&input_shapes, axis)?;
        let input_ids = inputs.iter().map(Self::id).collect::<Vec<_>>();

        let mut state = first.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Concat {
                inputs: input_ids,
                axis,
            },
            shape,
        });

        Ok(Self {
            graph: first.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_mul_scalar(&self, rhs: f64) -> Result<Self, String> {
        let rhs = self.try_scalar_constant(rhs)?;
        self.try_mul(&rhs)
    }

    pub fn try_div_scalar(&self, rhs: f64) -> Result<Self, String> {
        if rhs == 0.0 {
            return Err("division by zero scalar is not supported".to_string());
        }

        self.try_mul_scalar(1.0 / rhs)
    }

    pub fn try_powi(&self, exponent: u32) -> Result<Self, String> {
        if exponent == 0 {
            return self.try_scalar_constant(1.0);
        }

        let mut output = self.clone();
        for _ in 1..exponent {
            output = output.try_mul(self)?;
        }

        Ok(output)
    }

    pub fn try_powf(&self, exponent: f64) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Powf {
                input: self.node_id,
                exponent,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_sum(&self, axis: Option<usize>) -> Result<Self, String> {
        let shape = reduction_shape(self.dims(), axis)?;
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Sum {
                input: self.node_id,
                axis,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_mean(&self, axis: Option<usize>) -> Result<Self, String> {
        let shape = reduction_shape(self.dims(), axis)?;
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Mean {
                input: self.node_id,
                axis,
            },
            shape,
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: shape.0,
            cols: shape.1,
        })
    }

    pub fn try_transpose(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Transpose {
                input: self.node_id,
            },
            shape: (self.cols, self.rows),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.cols,
            cols: self.rows,
        })
    }

    pub fn try_reshape(&self, rows: usize, cols: usize) -> Result<Self, String> {
        if rows * cols != self.rows * self.cols {
            return Err(format!(
                "cannot reshape trace matrix with {} elements to ({rows}, {cols})",
                self.rows * self.cols
            ));
        }

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Reshape {
                input: self.node_id,
            },
            shape: (rows, cols),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows,
            cols,
        })
    }

    pub fn try_tanh(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Tanh {
                input: self.node_id,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_exp(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Exp {
                input: self.node_id,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_log(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Log {
                input: self.node_id,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_sqrt(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Sqrt {
                input: self.node_id,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_sin(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Sin {
                input: self.node_id,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_cos(&self) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Cos {
                input: self.node_id,
            },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }

    pub fn try_matmul(&self, rhs: &Self) -> Result<Self, String> {
        if !Arc::ptr_eq(&self.graph.state, &rhs.graph.state) {
            return Err("trace matmul operands must belong to the same trace graph".to_string());
        }

        if self.cols != rhs.rows {
            return Err(format!(
                "incompatible trace matmul shapes: ({}, {}) x ({}, {})",
                self.rows, self.cols, rhs.rows, rhs.cols
            ));
        }

        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::Matmul {
                lhs: self.node_id,
                rhs: rhs.node_id,
            },
            shape: (self.rows, rhs.cols),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: rhs.cols,
        })
    }

    fn try_scalar_constant(&self, value: f64) -> Result<Self, String> {
        let mut state = self.graph.lock_state()?;
        let id = state.nodes.len();
        state.nodes.push(TraceNode {
            id,
            op: TraceOp::ScalarConstant { value },
            shape: self.dims(),
        });

        Ok(Self {
            graph: self.graph.clone(),
            node_id: id,
            rows: self.rows,
            cols: self.cols,
        })
    }
}

impl TraceResult {
    pub fn new(graph: TraceGraph, output: TraceMatrix) -> Self {
        Self { graph, output }
    }
}

fn compile_trace_cpu(traced: &TraceResult) -> PyResult<CpuExecutionPlan> {
    traced
        .graph
        .compile_cpu(traced.output.id())
        .map_err(PyValueError::new_err)
}

pub fn trace_python_function(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<TraceResult> {
    let graph = TraceGraph::new();
    let mut inputs = Vec::with_capacity(input_specs.len());

    for (name, shape) in input_specs {
        inputs.push(
            graph
                .add_input(&name, shape)
                .map_err(PyValueError::new_err)?,
        );
    }

    let args = PyTuple::new(py, inputs)?;
    let output: TraceMatrix = function.call1(args)?.extract()?;

    Ok(TraceResult::new(graph, output))
}

#[pymethods]
impl TraceGraph {
    #[new]
    fn py_new() -> Self {
        Self::new()
    }

    fn input(&self, name: &str, shape: (usize, usize)) -> PyResult<TraceMatrix> {
        self.add_input(name, shape).map_err(PyValueError::new_err)
    }

    fn describe(&self) -> PyResult<Vec<String>> {
        self.describe_nodes().map_err(PyValueError::new_err)
    }

    fn ir(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let nodes = self.ir_nodes().map_err(PyValueError::new_err)?;
        let items = PyList::empty(py);

        for node in nodes {
            let item = PyDict::new(py);
            item.set_item("id", node.id)?;
            item.set_item("op", node.op)?;
            item.set_item("shape", node.shape)?;
            item.set_item("inputs", node.inputs)?;

            if let Some(name) = node.name {
                item.set_item("name", name)?;
            }

            if !node.attrs.is_empty() {
                let attrs = PyDict::new(py);
                for (key, value) in node.attrs {
                    match value {
                        IrAttrValue::Int(value) => attrs.set_item(key, value)?,
                        IrAttrValue::Float(value) => attrs.set_item(key, value)?,
                    }
                }
                item.set_item("attrs", attrs)?;
            }

            items.append(item)?;
        }

        Ok(items.into())
    }

    #[pyo3(name = "lower_text")]
    fn py_lower_text(&self) -> PyResult<String> {
        self.lower_text().map_err(PyValueError::new_err)
    }

    #[pyo3(name = "compile_cpu")]
    fn py_compile_cpu(&self, output_node_id: usize) -> PyResult<CpuExecutionPlan> {
        self.compile_cpu(output_node_id)
            .map_err(PyValueError::new_err)
    }

    fn vjp_ir(&self, py: Python<'_>, output_node_id: usize) -> PyResult<Py<PyList>> {
        let nodes = self
            .vjp_ir_nodes(output_node_id)
            .map_err(PyValueError::new_err)?;
        let items = PyList::empty(py);

        for node in nodes {
            let item = PyDict::new(py);
            item.set_item("id", node.id)?;
            item.set_item("op", node.op)?;
            item.set_item("shape", node.shape)?;
            item.set_item("inputs", node.inputs)?;

            if let Some(target) = node.target {
                item.set_item("target", target)?;
            }

            items.append(item)?;
        }

        Ok(items.into())
    }

    fn evaluate_vjp(
        &self,
        output_node_id: usize,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyMatrix,
    ) -> PyResult<HashMap<String, PyMatrix>> {
        let inputs = extract_matrix_map(inputs)?;

        self.evaluate_vjp_cpu(output_node_id, inputs, output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn evaluate(&self, output_node_id: usize, inputs: &Bound<'_, PyDict>) -> PyResult<PyMatrix> {
        let inputs = extract_matrix_map(inputs)?;

        self.evaluate_cpu(output_node_id, inputs)
            .map_err(PyValueError::new_err)
    }

    fn evaluate_jvp(
        &self,
        output_node_id: usize,
        inputs: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyMatrix, PyMatrix)> {
        let inputs = extract_matrix_map(inputs)?;
        let input_tangents = extract_matrix_map(input_tangents)?;

        self.evaluate_jvp_cpu(output_node_id, inputs, input_tangents)
            .map_err(PyValueError::new_err)
    }

    fn evaluate_value_and_vjp(
        &self,
        output_node_id: usize,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyMatrix,
    ) -> PyResult<(PyMatrix, HashMap<String, PyMatrix>)> {
        let inputs = extract_matrix_map(inputs)?;

        self.evaluate_value_and_vjp_cpu(output_node_id, inputs, output_cotangent)
            .map_err(PyValueError::new_err)
    }
}

#[pymethods]
impl CpuExecutionPlan {
    #[getter]
    fn output_node_id(&self) -> usize {
        self.output_node_id
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[getter(output_shape)]
    fn py_output_shape(&self) -> (usize, usize) {
        self.output_shape()
    }

    fn evaluate(&self, inputs: &Bound<'_, PyDict>) -> PyResult<PyMatrix> {
        let inputs = extract_matrix_map(inputs)?;

        self.evaluate_cpu(inputs).map_err(PyValueError::new_err)
    }

    fn evaluate_vjp(
        &self,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyMatrix,
    ) -> PyResult<HashMap<String, PyMatrix>> {
        let inputs = extract_matrix_map(inputs)?;

        self.evaluate_vjp_cpu(inputs, output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn evaluate_value_and_vjp(
        &self,
        inputs: &Bound<'_, PyDict>,
        output_cotangent: &PyMatrix,
    ) -> PyResult<(PyMatrix, HashMap<String, PyMatrix>)> {
        let inputs = extract_matrix_map(inputs)?;

        self.evaluate_value_and_vjp_cpu(inputs, output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn evaluate_jvp(
        &self,
        inputs: &Bound<'_, PyDict>,
        input_tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyMatrix, PyMatrix)> {
        let inputs = extract_matrix_map(inputs)?;
        let input_tangents = extract_matrix_map(input_tangents)?;

        self.evaluate_jvp_cpu(inputs, input_tangents)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.output_shape();
        format!(
            "CpuExecutionPlan(output_node_id={}, output_shape=({}, {}))",
            self.output_node_id, shape.0, shape.1
        )
    }
}

fn extract_matrix_map(inputs: &Bound<'_, PyDict>) -> PyResult<HashMap<String, PyMatrix>> {
    let mut matrices = HashMap::new();

    for (key, value) in inputs.iter() {
        let key = key.cast::<PyString>()?.to_str()?.to_string();
        let matrix = value.extract::<PyRef<'_, PyMatrix>>()?.clone();
        matrices.insert(key, matrix);
    }

    Ok(matrices)
}

fn trace_matrix_or_scalar_operand(
    lhs: &TraceMatrix,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<TraceMatrix> {
    if let Ok(rhs) = rhs.extract::<PyRef<'_, TraceMatrix>>() {
        return Ok(rhs.clone());
    }

    if let Ok(value) = rhs.extract::<f64>() {
        return lhs
            .try_scalar_constant(value)
            .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(
        "expected TraceMatrix or numeric scalar operand",
    ))
}

// The numeric fast path is deliberately non-expansive: unary slopes and
// constant affine factors are at most one. General products, reductions,
// division and singular domains keep the original reverse implementation.
fn legacy_forward_active(plan: &CpuExecutionPlan, primals: &[PyMatrix]) -> Option<Vec<bool>> {
    if primals
        .iter()
        .any(|value| value.data().iter().any(|value| !value.is_finite()))
    {
        return None;
    }
    let mut active = vec![false; plan.root_node_id + 1];
    active[plan.root_node_id] = true;
    for id in (0..=plan.root_node_id).rev() {
        if !active[id] {
            continue;
        }
        let node = &plan.nodes[id];
        let constant = |id: usize| matches!(plan.nodes[id].op, TraceOp::ScalarConstant { .. });
        match &node.op {
            TraceOp::Input { .. }
            | TraceOp::ScalarConstant { .. }
            | TraceOp::Gt { .. }
            | TraceOp::Reshape { .. }
            | TraceOp::Transpose { .. }
            | TraceOp::Concat { .. }
            | TraceOp::Sin { .. }
            | TraceOp::Cos { .. }
            | TraceOp::Tanh { .. } => {}
            TraceOp::Add { lhs, rhs } | TraceOp::Sub { lhs, rhs }
                if constant(*lhs) || constant(*rhs) => {}
            _ => return None,
        }
        if !matches!(node.op, TraceOp::Gt { .. }) {
            for input in trace_op_inputs(&node.op) {
                active[input] = true;
            }
        }
    }
    Some(active)
}

fn dense_jacobians_for_inputs(
    plan: &CpuExecutionPlan,
    values: &HashMap<String, PyMatrix>,
    input_specs: &[(String, (usize, usize))],
) -> Result<HashMap<String, PyMatrix>, String> {
    let output_shape = plan.output_shape();
    let output_size = output_shape.0 * output_shape.1;
    let mut blocks = input_specs
        .iter()
        .map(|_| Vec::with_capacity(output_size))
        .collect::<Vec<_>>();

    // Empty requests and empty outputs historically do not execute the primal.
    if !input_specs.is_empty() && output_size != 0 {
        let primals = evaluate_nodes(&plan.nodes, plan.root_node_id, values)?;
        let input_size = input_specs.iter().try_fold(0usize, |total, (_, shape)| {
            total.checked_add(shape.0 * shape.1)
        });
        // Shape-only graphs preserve exact 0/1 derivatives even for nonfinite
        // primals. Eligible numeric graphs use the approved F64 tolerance.
        let structural_forward = input_size.is_some_and(|size| size < output_size)
            && plan.nodes.iter().all(|node| {
                matches!(
                    node.op,
                    TraceOp::Input { .. }
                        | TraceOp::ScalarConstant { .. }
                        | TraceOp::Reshape { .. }
                        | TraceOp::Transpose { .. }
                        | TraceOp::Concat { .. }
                )
            });
        let numeric_active = input_size
            .filter(|&size| size < output_size)
            .and_then(|_| legacy_forward_active(plan, &primals));
        if structural_forward || numeric_active.is_some() {
            let mut tangents = plan
                .nodes
                .iter()
                .filter_map(|node| match &node.op {
                    TraceOp::Input { name } => Some((
                        name.clone(),
                        PyMatrix::filled(node.shape.0, node.shape.1, 0.0),
                    )),
                    _ => None,
                })
                .collect::<HashMap<_, _>>();
            let mut result = HashMap::with_capacity(input_specs.len());
            let mut failed = false;
            for (name, shape) in input_specs {
                let size = shape.0 * shape.1;
                let mut rows = vec![vec![0.0; size]; output_size];
                for column in 0..size {
                    let mut seed = vec![vec![0.0; shape.1]; shape.0];
                    seed[column / shape.1][column % shape.1] = 1.0;
                    tangents.insert(name.clone(), PyMatrix::from_rows(seed)?);
                    let propagated = evaluate_jvp_from_values(
                        &plan.nodes,
                        plan.root_node_id,
                        &primals,
                        &tangents,
                        numeric_active.as_deref(),
                    );
                    let propagated = match propagated {
                        Ok(value)
                            if structural_forward
                                || value.data().iter().all(|value| value.is_finite()) =>
                        {
                            value
                        }
                        // A speculative tangent must never introduce a new checked
                        // error or nonfinite result. Reuse primals in original VJP.
                        _ => {
                            failed = true;
                            break;
                        }
                    };
                    for (row, derivative) in rows.iter_mut().zip(propagated.data()) {
                        row[column] = *derivative;
                    }
                }
                if failed {
                    break;
                }
                tangents.insert(name.clone(), PyMatrix::filled(shape.0, shape.1, 0.0));
                result.insert(name.clone(), PyMatrix::from_rows(rows)?);
            }
            if !failed {
                return Ok(result);
            }
            // Drop candidate blocks before allocating the original reverse result.
            drop(result);
        }

        for output_index in 0..output_size {
            let mut cotangent_rows = vec![vec![0.0; output_shape.1]; output_shape.0];
            cotangent_rows[output_index / output_shape.1][output_index % output_shape.1] = 1.0;
            let output_cotangent = PyMatrix::from_rows(cotangent_rows)?;

            let gradients = evaluate_vjp_from_values(
                &plan.nodes,
                plan.root_node_id,
                &primals,
                &output_cotangent,
            )?;

            for ((input_name, input_shape), rows) in input_specs.iter().zip(&mut blocks) {
                let row = gradients
                    .get(input_name)
                    .map(|gradient| gradient.data().to_vec())
                    .unwrap_or_else(|| vec![0.0; input_shape.0 * input_shape.1]);
                rows.push(row);
            }
        }
    }

    let mut jacobians = HashMap::with_capacity(input_specs.len());
    for ((name, _), rows) in input_specs.iter().zip(blocks) {
        jacobians.insert(name.clone(), PyMatrix::from_rows(rows)?);
    }
    Ok(jacobians)
}

#[pymethods]
impl TraceMatrix {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.dims()
    }

    #[getter]
    fn node_id(&self) -> usize {
        self.id()
    }

    fn __add__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = trace_matrix_or_scalar_operand(self, rhs)?;
        self.try_add(&rhs).map_err(PyValueError::new_err)
    }

    fn add(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    fn __radd__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__add__(lhs)
    }

    fn __sub__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = trace_matrix_or_scalar_operand(self, rhs)?;
        self.try_sub(&rhs).map_err(PyValueError::new_err)
    }

    fn sub(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__sub__(rhs)
    }

    fn __rsub__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = lhs.extract::<f64>() {
            return self.try_scalar_sub(value).map_err(PyValueError::new_err);
        }

        Err(PyTypeError::new_err("expected numeric scalar operand"))
    }

    fn __mul__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = trace_matrix_or_scalar_operand(self, rhs)?;
        self.try_mul(&rhs).map_err(PyValueError::new_err)
    }

    fn mul(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    fn __rmul__(&self, lhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        self.__mul__(lhs)
    }

    fn __neg__(&self) -> PyResult<Self> {
        self.try_mul_scalar(-1.0).map_err(PyValueError::new_err)
    }

    fn __truediv__(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = rhs.extract::<f64>() {
            return self.try_div_scalar(value).map_err(PyValueError::new_err);
        }

        let rhs = rhs
            .extract::<PyRef<'_, TraceMatrix>>()
            .map_err(|_| PyTypeError::new_err("expected TraceMatrix or numeric scalar divisor"))?;

        self.try_div(&rhs).map_err(PyValueError::new_err)
    }

    fn __pow__(
        &self,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(PyTypeError::new_err(
                "modulo argument is not supported for TraceMatrix power",
            ));
        }

        if let Ok(exponent) = exponent.extract::<u32>() {
            return self.try_powi(exponent).map_err(PyValueError::new_err);
        }

        let exponent = exponent
            .extract::<f64>()
            .map_err(|_| PyTypeError::new_err("expected numeric scalar exponent"))?;

        self.try_powf(exponent).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis=None))]
    fn sum(&self, axis: Option<usize>) -> PyResult<Self> {
        self.try_sum(axis).map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (axis=None))]
    fn mean(&self, axis: Option<usize>) -> PyResult<Self> {
        self.try_mean(axis).map_err(PyValueError::new_err)
    }

    #[getter(T)]
    fn py_t(&self) -> PyResult<Self> {
        self.try_transpose().map_err(PyValueError::new_err)
    }

    fn transpose(&self) -> PyResult<Self> {
        self.try_transpose().map_err(PyValueError::new_err)
    }

    fn reshape(&self, rows: usize, cols: usize) -> PyResult<Self> {
        self.try_reshape(rows, cols).map_err(PyValueError::new_err)
    }

    fn tanh(&self) -> PyResult<Self> {
        self.try_tanh().map_err(PyValueError::new_err)
    }

    fn exp(&self) -> PyResult<Self> {
        self.try_exp().map_err(PyValueError::new_err)
    }

    fn log(&self) -> PyResult<Self> {
        self.try_log().map_err(PyValueError::new_err)
    }

    fn sqrt(&self) -> PyResult<Self> {
        self.try_sqrt().map_err(PyValueError::new_err)
    }

    fn sin(&self) -> PyResult<Self> {
        self.try_sin().map_err(PyValueError::new_err)
    }

    fn cos(&self) -> PyResult<Self> {
        self.try_cos().map_err(PyValueError::new_err)
    }

    fn gt(&self, rhs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = trace_matrix_or_scalar_operand(self, rhs)?;
        self.try_gt(&rhs).map_err(PyValueError::new_err)
    }

    fn __matmul__(&self, rhs: &Self) -> PyResult<Self> {
        self.try_matmul(rhs).map_err(PyValueError::new_err)
    }

    fn matmul(&self, rhs: &Self) -> PyResult<Self> {
        self.try_matmul(rhs).map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "TraceMatrix(node_id={}, shape=({}, {}))",
            self.node_id, self.rows, self.cols
        )
    }
}

#[pymethods]
impl TraceResult {
    #[getter]
    fn graph(&self) -> TraceGraph {
        self.graph.clone()
    }

    #[getter]
    fn output(&self) -> TraceMatrix {
        self.output.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TraceResult(output_node_id={}, output_shape=({}, {}))",
            self.output.node_id, self.output.rows, self.output.cols
        )
    }
}

#[pymethods]
impl GradFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<HashMap<String, PyMatrix>> {
        let values = extract_matrix_map(values)?;

        self.plan
            .evaluate_vjp_cpu(values, &self.output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "GradFunction(output_node_id={}, output_shape=({}, {}))",
            self.plan.output_node_id, shape.0, shape.1
        )
    }
}

#[pymethods]
impl GradScalarFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<HashMap<String, PyMatrix>> {
        let output_cotangent = PyMatrix::filled(1, 1, 1.0);
        let values = extract_matrix_map(values)?;

        self.plan
            .evaluate_vjp_cpu(values, &output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "GradScalarFunction(output_node_id={}, output_shape=({}, {}))",
            self.plan.output_node_id, shape.0, shape.1
        )
    }
}

#[pymethods]
impl ValueAndGradFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
    ) -> PyResult<(PyMatrix, HashMap<String, PyMatrix>)> {
        let values = extract_matrix_map(values)?;

        self.plan
            .evaluate_value_and_vjp_cpu(values, &self.output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "ValueAndGradFunction(output_node_id={}, output_shape=({}, {}))",
            self.plan.output_node_id, shape.0, shape.1
        )
    }
}

#[pymethods]
impl VjpFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        output_cotangent: &PyMatrix,
    ) -> PyResult<(PyMatrix, HashMap<String, PyMatrix>)> {
        let values = extract_matrix_map(values)?;

        self.plan
            .evaluate_value_and_vjp_cpu(values, output_cotangent)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "VjpFunction(output_node_id={}, output_shape=({}, {}))",
            self.plan.output_node_id, shape.0, shape.1
        )
    }
}

#[pymethods]
impl JacobianFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<PyMatrix> {
        let values = extract_matrix_map(values)?;

        let specs = [(self.input_name.clone(), self.input_shape)];
        let mut jacobians = dense_jacobians_for_inputs(&self.plan, &values, &specs)
            .map_err(PyValueError::new_err)?;
        Ok(jacobians.remove(&self.input_name).unwrap())
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "JacobianFunction(output_node_id={}, input_shape=({}, {}), output_shape=({}, {}))",
            self.plan.output_node_id, self.input_shape.0, self.input_shape.1, shape.0, shape.1
        )
    }
}

#[pymethods]
impl JacobiansFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<HashMap<String, PyMatrix>> {
        let values = extract_matrix_map(values)?;
        dense_jacobians_for_inputs(&self.plan, &values, &self.input_specs)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "JacobiansFunction(output_node_id={}, input_count={}, output_shape=({}, {}))",
            self.plan.output_node_id,
            self.input_specs.len(),
            shape.0,
            shape.1
        )
    }
}

#[pymethods]
impl JvpFunction {
    fn __call__(
        &self,
        values: &Bound<'_, PyDict>,
        tangents: &Bound<'_, PyDict>,
    ) -> PyResult<(PyMatrix, PyMatrix)> {
        let values = extract_matrix_map(values)?;
        let tangents = extract_matrix_map(tangents)?;

        self.plan
            .evaluate_jvp_cpu(values, tangents)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "JvpFunction(output_node_id={}, input_count={}, output_shape=({}, {}))",
            self.plan.output_node_id,
            self.input_specs.len(),
            shape.0,
            shape.1
        )
    }
}

#[pymethods]
impl GradScalarTransform {
    fn __call__(
        &self,
        py: Python<'_>,
        function: &Bound<'_, PyAny>,
    ) -> PyResult<GradScalarFunction> {
        let traced = trace_python_function(py, function, self.input_specs.clone())?;
        let plan = compile_trace_cpu(&traced)?;

        Ok(GradScalarFunction { plan })
    }

    fn __repr__(&self) -> String {
        format!(
            "GradScalarTransform(input_count={})",
            self.input_specs.len()
        )
    }
}

#[pymethods]
impl JitFunction {
    fn __call__(&self, values: &Bound<'_, PyDict>) -> PyResult<PyMatrix> {
        let values = extract_matrix_map(values)?;

        self.plan
            .evaluate_cpu(values)
            .map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        let shape = self.plan.output_shape();
        format!(
            "JitFunction(output_node_id={}, output_shape=({}, {}))",
            self.plan.output_node_id, shape.0, shape.1
        )
    }
}

#[pymethods]
impl JitTransform {
    fn __call__(&self, py: Python<'_>, function: &Bound<'_, PyAny>) -> PyResult<JitFunction> {
        let traced = trace_python_function(py, function, self.input_specs.clone())?;
        let plan = traced
            .graph
            .compile_cpu(traced.output.id())
            .map_err(PyValueError::new_err)?;

        Ok(JitFunction { plan })
    }

    fn __repr__(&self) -> String {
        format!("JitTransform(input_count={})", self.input_specs.len())
    }
}

#[pyfunction]
pub fn trace(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<TraceResult> {
    trace_python_function(py, function, input_specs)
}

#[pyfunction]
pub fn grad(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
    values: &Bound<'_, PyDict>,
    output_cotangent: &PyMatrix,
) -> PyResult<HashMap<String, PyMatrix>> {
    let traced = trace_python_function(py, function, input_specs)?;
    let values = extract_matrix_map(values)?;
    let plan = compile_trace_cpu(&traced)?;

    plan.evaluate_vjp_cpu(values, output_cotangent)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn grad_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
    output_cotangent: &PyMatrix,
) -> PyResult<GradFunction> {
    let traced = trace_python_function(py, function, input_specs)?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(GradFunction {
        plan,
        output_cotangent: output_cotangent.clone(),
    })
}

#[pyfunction]
pub fn grad_scalar_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<GradScalarFunction> {
    let traced = trace_python_function(py, function, input_specs)?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(GradScalarFunction { plan })
}

#[pyfunction]
pub fn value_and_grad_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
    output_cotangent: &PyMatrix,
) -> PyResult<ValueAndGradFunction> {
    let traced = trace_python_function(py, function, input_specs)?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(ValueAndGradFunction {
        plan,
        output_cotangent: output_cotangent.clone(),
    })
}

#[pyfunction]
pub fn vjp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<VjpFunction> {
    let traced = trace_python_function(py, function, input_specs)?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(VjpFunction { plan })
}

#[pyfunction]
pub fn jacobian_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<JacobianFunction> {
    if input_specs.len() != 1 {
        return Err(PyValueError::new_err(
            "jacobian_fn currently supports exactly one input spec",
        ));
    }

    let (input_name, input_shape) = input_specs[0].clone();
    let traced = trace_python_function(py, function, input_specs)?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(JacobianFunction {
        plan,
        input_name,
        input_shape,
    })
}

#[pyfunction]
pub fn jacobians_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<JacobiansFunction> {
    let traced = trace_python_function(py, function, input_specs.clone())?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(JacobiansFunction { plan, input_specs })
}

#[pyfunction]
pub fn jvp_fn(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    input_specs: Vec<(String, (usize, usize))>,
) -> PyResult<JvpFunction> {
    let traced = trace_python_function(py, function, input_specs.clone())?;
    let plan = compile_trace_cpu(&traced)?;

    Ok(JvpFunction { plan, input_specs })
}

#[pyfunction]
pub fn grad_scalar(input_specs: Vec<(String, (usize, usize))>) -> GradScalarTransform {
    GradScalarTransform { input_specs }
}

#[pyfunction]
pub fn jit(input_specs: Vec<(String, (usize, usize))>) -> JitTransform {
    JitTransform { input_specs }
}

#[cfg(test)]
mod legacy_ad_optimization_tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        pub(super) static PRIMAL_EVALUATIONS: Cell<usize> = const { Cell::new(0) };
        pub(super) static VJP_EVALUATIONS: Cell<usize> = const { Cell::new(0) };
        pub(super) static JVP_EVALUATIONS: Cell<usize> = const { Cell::new(0) };
    }

    fn plan(ops: Vec<(TraceOp, (usize, usize))>) -> CpuExecutionPlan {
        let nodes = ops
            .into_iter()
            .enumerate()
            .map(|(id, (op, shape))| TraceNode { id, op, shape })
            .collect::<Vec<_>>();
        let root_node_id = nodes.len() - 1;
        CpuExecutionPlan {
            nodes,
            root_node_id,
            output_node_id: root_node_id,
        }
    }

    fn bits(matrix: &PyMatrix) -> Vec<u64> {
        matrix.data().iter().map(|value| value.to_bits()).collect()
    }

    fn reference_jacobians(
        plan: &CpuExecutionPlan,
        values: &HashMap<String, PyMatrix>,
        specs: &[(String, (usize, usize))],
    ) -> Result<HashMap<String, PyMatrix>, String> {
        let (rows, cols) = plan.output_shape();
        let mut result = HashMap::new();
        for (name, shape) in specs {
            let mut block = Vec::new();
            for index in 0..rows * cols {
                let mut seed = vec![vec![0.0; cols]; rows];
                seed[index / cols][index % cols] = 1.0;
                let gradients =
                    plan.evaluate_vjp_cpu(values.clone(), &PyMatrix::from_rows(seed)?)?;
                block.push(
                    gradients
                        .get(name)
                        .map(|value| value.data().to_vec())
                        .unwrap_or_else(|| vec![0.0; shape.0 * shape.1]),
                );
            }
            result.insert(name.clone(), PyMatrix::from_rows(block)?);
        }
        Ok(result)
    }

    #[test]
    fn live_jvp_counts_duplicate_operands_and_validates_unused_inputs() {
        let plan = plan(vec![
            (TraceOp::Input { name: "x".into() }, (1, 2)),
            (
                TraceOp::Input {
                    name: "unused".into(),
                },
                (1, 1),
            ),
            (TraceOp::Add { lhs: 0, rhs: 0 }, (1, 2)),
            (
                TraceOp::Concat {
                    inputs: vec![2, 2, 2],
                    axis: 1,
                },
                (1, 6),
            ),
        ]);
        let inputs = HashMap::from([
            (
                "x".into(),
                PyMatrix::from_rows(vec![vec![0.1, 0.2]]).unwrap(),
            ),
            ("unused".into(), PyMatrix::filled(1, 1, 7.0)),
        ]);
        let mut tangents = HashMap::from([
            (
                "x".into(),
                PyMatrix::from_rows(vec![vec![0.3, -0.4]]).unwrap(),
            ),
            ("unused".into(), PyMatrix::filled(1, 1, 0.0)),
        ]);
        let (_, tangent) = plan
            .evaluate_jvp_cpu(inputs.clone(), tangents.clone())
            .unwrap();
        assert_eq!(tangent.data(), &[0.6, -0.8, 0.6, -0.8, 0.6, -0.8]);
        tangents.remove("unused");
        assert_eq!(
            plan.evaluate_jvp_cpu(inputs.clone(), tangents.clone())
                .unwrap_err(),
            "missing tangent matrix for unused"
        );
        tangents.insert("unused".into(), PyMatrix::filled(1, 2, 0.0));
        assert_eq!(
            plan.evaluate_jvp_cpu(inputs, tangents).unwrap_err(),
            "tangent unused shape (1, 2) does not match traced shape (1, 1)"
        );
    }

    #[test]
    fn numeric_expansion_uses_live_forward_tangents_with_original_blocks() {
        let plan = plan(vec![
            (TraceOp::Input { name: "x".into() }, (1, 2)),
            (
                TraceOp::Input {
                    name: "unused".into(),
                },
                (1, 1),
            ),
            (TraceOp::ScalarConstant { value: 0.25 }, (1, 1)),
            (TraceOp::Add { lhs: 0, rhs: 2 }, (1, 2)),
            (TraceOp::Sin { input: 3 }, (1, 2)),
            (TraceOp::Tanh { input: 4 }, (1, 2)),
            (
                TraceOp::Concat {
                    inputs: vec![5, 5, 5, 5],
                    axis: 1,
                },
                (1, 8),
            ),
        ]);
        let values = HashMap::from([
            (
                "x".into(),
                PyMatrix::from_rows(vec![vec![-0.0, 0.5]]).unwrap(),
            ),
            ("unused".into(), PyMatrix::filled(1, 1, 3.0)),
        ]);
        let specs = [("x".into(), (1, 2)), ("unused".into(), (1, 1))];
        let expected = reference_jacobians(&plan, &values, &specs).unwrap();
        PRIMAL_EVALUATIONS.with(|count| count.set(0));
        VJP_EVALUATIONS.with(|count| count.set(0));
        JVP_EVALUATIONS.with(|count| count.set(0));
        let actual = dense_jacobians_for_inputs(&plan, &values, &specs).unwrap();
        assert_eq!(PRIMAL_EVALUATIONS.with(Cell::get), 1);
        assert_eq!(VJP_EVALUATIONS.with(Cell::get), 0);
        assert_eq!(JVP_EVALUATIONS.with(Cell::get), 3);
        for (name, expected) in expected {
            assert_eq!(actual[&name].dims(), expected.dims());
            for (actual, expected) in actual[&name].data().iter().zip(expected.data()) {
                assert!((actual - expected).abs() <= 1e-12 + 1e-12 * expected.abs());
            }
        }
    }

    #[test]
    fn inactive_division_below_gt_preserves_original_reverse_result() {
        let plan = plan(vec![
            (TraceOp::Input { name: "x".into() }, (1, 1)),
            (TraceOp::Div { lhs: 0, rhs: 0 }, (1, 1)),
            (TraceOp::ScalarConstant { value: 0.0 }, (1, 1)),
            (TraceOp::Gt { lhs: 1, rhs: 2 }, (1, 1)),
            (
                TraceOp::Concat {
                    inputs: vec![3, 3],
                    axis: 1,
                },
                (1, 2),
            ),
        ]);
        let values = HashMap::from([("x".into(), PyMatrix::filled(1, 1, 1e-300))]);
        let specs = [("x".into(), (1, 1))];
        let expected = reference_jacobians(&plan, &values, &specs).unwrap();
        let primals = evaluate_nodes(&plan.nodes, plan.root_node_id, &values).unwrap();
        assert!(evaluate_jvp_from_values(
            &plan.nodes,
            plan.root_node_id,
            &primals,
            &HashMap::from([("x".into(), PyMatrix::filled(1, 1, 1.0))]),
            None
        )
        .is_err());
        let actual = dense_jacobians_for_inputs(&plan, &values, &specs).unwrap();
        assert_eq!(bits(&actual["x"]), bits(&expected["x"]));
    }

    #[test]
    fn unsafe_numeric_domains_keep_reverse_results_and_errors() {
        for op in [
            TraceOp::Div { lhs: 0, rhs: 0 },
            TraceOp::Powf {
                input: 0,
                exponent: 0.5,
            },
            TraceOp::Sqrt { input: 0 },
            TraceOp::Mul { lhs: 0, rhs: 0 },
        ] {
            for value in [0.0, -1.0, 1e-300, f64::INFINITY, f64::NAN] {
                let plan = plan(vec![
                    (TraceOp::Input { name: "x".into() }, (1, 1)),
                    (op.clone(), (1, 1)),
                    (
                        TraceOp::Concat {
                            inputs: vec![1, 1],
                            axis: 1,
                        },
                        (1, 2),
                    ),
                ]);
                let values = HashMap::from([("x".into(), PyMatrix::filled(1, 1, value))]);
                let specs = [("x".into(), (1, 1))];
                let expected = reference_jacobians(&plan, &values, &specs);
                let actual = dense_jacobians_for_inputs(&plan, &values, &specs);
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => {
                        assert_eq!(bits(&actual["x"]), bits(&expected["x"]))
                    }
                    (Err(actual), Err(expected)) => assert_eq!(actual, expected),
                    other => panic!("direction changed result/error: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn jacobians_share_primal_and_reverse_seeds_across_all_blocks() {
        let plan = plan(vec![
            (TraceOp::Input { name: "x".into() }, (1, 3)),
            (TraceOp::Input { name: "y".into() }, (1, 1)),
            (
                TraceOp::Input {
                    name: "unused".into(),
                },
                (1, 2),
            ),
            (TraceOp::Mul { lhs: 0, rhs: 0 }, (1, 3)),
            (TraceOp::Sin { input: 3 }, (1, 3)),
            (TraceOp::Add { lhs: 4, rhs: 1 }, (1, 3)),
            (TraceOp::Add { lhs: 5, rhs: 4 }, (1, 3)),
        ]);
        let values = HashMap::from([
            (
                "x".into(),
                PyMatrix::from_rows(vec![vec![-0.0, 0.5, 2.0]]).unwrap(),
            ),
            ("y".into(), PyMatrix::filled(1, 1, 0.25)),
            ("unused".into(), PyMatrix::filled(1, 2, 3.0)),
        ]);
        let specs = vec![
            ("x".into(), (1, 3)),
            ("y".into(), (1, 1)),
            ("unused".into(), (1, 2)),
        ];
        let expected = reference_jacobians(&plan, &values, &specs).unwrap();
        PRIMAL_EVALUATIONS.with(|count| count.set(0));
        VJP_EVALUATIONS.with(|count| count.set(0));
        let actual = dense_jacobians_for_inputs(&plan, &values, &specs).unwrap();
        assert_eq!(PRIMAL_EVALUATIONS.with(Cell::get), 1);
        assert_eq!(VJP_EVALUATIONS.with(Cell::get), 3);
        for (name, expected) in expected {
            assert_eq!(actual[&name].dims(), expected.dims());
            assert_eq!(bits(&actual[&name]), bits(&expected));
        }
        assert_eq!(
            bits(&values["x"]),
            vec![(-0.0f64).to_bits(), 0.5f64.to_bits(), 2.0f64.to_bits()]
        );
    }

    #[test]
    fn structural_jacobian_chooses_exact_forward_direction_for_expanding_outputs() {
        let plan = plan(vec![
            (TraceOp::Input { name: "x".into() }, (1, 2)),
            (TraceOp::Input { name: "y".into() }, (1, 1)),
            (TraceOp::Transpose { input: 0 }, (2, 1)),
            (
                TraceOp::Concat {
                    inputs: vec![2, 1, 2, 1],
                    axis: 0,
                },
                (6, 1),
            ),
            (TraceOp::Reshape { input: 3 }, (2, 3)),
        ]);
        let values = HashMap::from([
            (
                "x".into(),
                PyMatrix::from_rows(vec![vec![-0.0, f64::from_bits(0x7ff0000000000001)]]).unwrap(),
            ),
            ("y".into(), PyMatrix::filled(1, 1, f64::INFINITY)),
        ]);
        let specs = [("x".into(), (1, 2)), ("y".into(), (1, 1))];
        let expected = reference_jacobians(&plan, &values, &specs).unwrap();
        PRIMAL_EVALUATIONS.with(|count| count.set(0));
        VJP_EVALUATIONS.with(|count| count.set(0));
        JVP_EVALUATIONS.with(|count| count.set(0));
        let actual = dense_jacobians_for_inputs(&plan, &values, &specs).unwrap();
        assert_eq!(PRIMAL_EVALUATIONS.with(Cell::get), 1);
        assert_eq!(VJP_EVALUATIONS.with(Cell::get), 0);
        assert_eq!(JVP_EVALUATIONS.with(Cell::get), 3);
        for (name, expected) in expected {
            assert_eq!(actual[&name].dims(), expected.dims());
            assert_eq!(bits(&actual[&name]), bits(&expected));
        }
    }

    #[test]
    fn jacobian_preserves_empty_and_eager_error_behavior() {
        let plan = plan(vec![
            (
                TraceOp::Input {
                    name: "missing".into(),
                },
                (1, 2),
            ),
            (TraceOp::ScalarConstant { value: 1.0 }, (1, 2)),
        ]);
        let specs = [("missing".into(), (1, 2))];
        assert_eq!(
            dense_jacobians_for_inputs(&plan, &HashMap::new(), &specs).unwrap_err(),
            reference_jacobians(&plan, &HashMap::new(), &specs).unwrap_err()
        );
        assert!(dense_jacobians_for_inputs(&plan, &HashMap::new(), &[])
            .unwrap()
            .is_empty());
        let empty = self::plan(vec![(
            TraceOp::Input {
                name: "missing".into(),
            },
            (0, 2),
        )]);
        let actual = dense_jacobians_for_inputs(&empty, &HashMap::new(), &specs).unwrap();
        assert_eq!(actual["missing"].dims(), (0, 0));
    }

    #[test]
    fn reverse_retains_input_output_seed_and_duplicate_operand_gradients() {
        let identity = plan(vec![(TraceOp::Input { name: "x".into() }, (1, 2))]);
        let values = HashMap::from([("x".into(), PyMatrix::filled(1, 2, 3.0))]);
        let seed = PyMatrix::from_rows(vec![vec![-0.0, 2.0]]).unwrap();
        let gradient = identity.evaluate_vjp_cpu(values.clone(), &seed).unwrap();
        assert_eq!(bits(&gradient["x"]), bits(&seed));
        let duplicate = plan(vec![
            (TraceOp::Input { name: "x".into() }, (1, 2)),
            (TraceOp::Mul { lhs: 0, rhs: 0 }, (1, 2)),
            (TraceOp::Add { lhs: 1, rhs: 1 }, (1, 2)),
        ]);
        let gradient = duplicate.evaluate_vjp_cpu(values.clone(), &seed).unwrap();
        assert_eq!(gradient["x"].data(), &[0.0, 24.0]);
        assert_eq!(bits(&seed), vec![(-0.0f64).to_bits(), 2.0f64.to_bits()]);
        assert_eq!(values["x"].data(), &[3.0, 3.0]);
    }

    #[test]
    fn cotangent_accumulation_reuses_owned_buffer_and_preserves_bits() {
        let nodes = vec![TraceNode {
            id: 0,
            op: TraceOp::Input { name: "x".into() },
            shape: (1, 6),
        }];
        let initial = PyMatrix::from_rows(vec![vec![
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff0000000000001),
            1e16,
            1.0,
        ]])
        .unwrap();
        let incoming = PyMatrix::from_rows(vec![vec![
            -0.0,
            f64::NEG_INFINITY,
            f64::INFINITY,
            2.0,
            -1e16,
            f64::from_bits(0x7ff8000000000002),
        ]])
        .unwrap();
        let expected = initial.try_add(&incoming).unwrap();
        let snapshot = bits(&initial);
        let mut cotangents = vec![Some(initial.clone())];
        let pointer = cotangents[0].as_ref().unwrap().data().as_ptr();
        accumulate_cotangent(&mut cotangents, &nodes, 0, incoming).unwrap();
        let actual = cotangents[0].as_ref().unwrap();
        assert_eq!(actual.data().as_ptr(), pointer);
        assert_eq!(bits(actual), bits(&expected));
        assert_eq!(bits(&initial), snapshot);
        assert!(
            accumulate_cotangent(&mut cotangents, &nodes, 0, PyMatrix::filled(1, 1, 1.0))
                .unwrap_err()
                .contains("expected (1, 6)")
        );
    }
}

#[cfg(test)]
mod forward_liveness_tests {
    use super::*;

    fn node(id: usize, op: TraceOp, shape: (usize, usize)) -> TraceNode {
        TraceNode { id, op, shape }
    }

    fn inputs() -> HashMap<String, PyMatrix> {
        HashMap::from([
            (
                "x".to_string(),
                PyMatrix::from_rows(vec![vec![0.25, 0.5], vec![1.0, 2.0]]).unwrap(),
            ),
            ("y".to_string(), PyMatrix::filled(2, 2, 0.75)),
        ])
    }

    fn assert_matches_tape(nodes: &[TraceNode], root: usize, inputs: &HashMap<String, PyMatrix>) {
        let expected = evaluate_nodes(nodes, root, inputs).map(|values| values[root].clone());
        let actual = evaluate_forward_nodes(nodes, root, inputs);
        match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                assert_eq!(expected.dims(), actual.dims());
                assert_eq!(
                    expected
                        .data()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    actual
                        .data()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                );
            }
            (Err(expected), Err(actual)) => assert_eq!(expected, actual),
            (expected, actual) => panic!("tape and forward disagree: {expected:?}, {actual:?}"),
        }
    }

    #[test]
    fn forward_matches_tape_for_every_primitive() {
        let cases = vec![
            (TraceOp::ScalarConstant { value: -0.0 }, (2, 2)),
            (TraceOp::Add { lhs: 0, rhs: 1 }, (2, 2)),
            (TraceOp::Sub { lhs: 0, rhs: 1 }, (2, 2)),
            (TraceOp::Mul { lhs: 0, rhs: 1 }, (2, 2)),
            (TraceOp::Div { lhs: 0, rhs: 1 }, (2, 2)),
            (TraceOp::Gt { lhs: 0, rhs: 1 }, (2, 2)),
            (
                TraceOp::Where {
                    mask: 0,
                    on_true: 1,
                    on_false: 0,
                },
                (2, 2),
            ),
            (
                TraceOp::Concat {
                    inputs: vec![0, 1, 0],
                    axis: 0,
                },
                (6, 2),
            ),
            (
                TraceOp::Concat {
                    inputs: vec![0, 1, 0],
                    axis: 1,
                },
                (2, 6),
            ),
            (TraceOp::Matmul { lhs: 0, rhs: 1 }, (2, 2)),
            (
                TraceOp::Sum {
                    input: 0,
                    axis: None,
                },
                (1, 1),
            ),
            (
                TraceOp::Sum {
                    input: 0,
                    axis: Some(0),
                },
                (1, 2),
            ),
            (
                TraceOp::Sum {
                    input: 0,
                    axis: Some(1),
                },
                (2, 1),
            ),
            (
                TraceOp::Mean {
                    input: 0,
                    axis: None,
                },
                (1, 1),
            ),
            (
                TraceOp::Mean {
                    input: 0,
                    axis: Some(0),
                },
                (1, 2),
            ),
            (
                TraceOp::Mean {
                    input: 0,
                    axis: Some(1),
                },
                (2, 1),
            ),
            (TraceOp::Transpose { input: 0 }, (2, 2)),
            (TraceOp::Reshape { input: 0 }, (1, 4)),
            (
                TraceOp::Powf {
                    input: 0,
                    exponent: 1.5,
                },
                (2, 2),
            ),
            (TraceOp::Tanh { input: 0 }, (2, 2)),
            (TraceOp::Exp { input: 0 }, (2, 2)),
            (TraceOp::Log { input: 0 }, (2, 2)),
            (TraceOp::Sqrt { input: 0 }, (2, 2)),
            (TraceOp::Sin { input: 0 }, (2, 2)),
            (TraceOp::Cos { input: 0 }, (2, 2)),
        ];
        for (op, shape) in cases {
            let nodes = vec![
                node(
                    0,
                    TraceOp::Input {
                        name: "x".to_string(),
                    },
                    (2, 2),
                ),
                node(
                    1,
                    TraceOp::Input {
                        name: "y".to_string(),
                    },
                    (2, 2),
                ),
                node(2, op.clone(), shape),
                node(3, TraceOp::Add { lhs: 2, rhs: 2 }, shape),
                node(4, TraceOp::Add { lhs: 2, rhs: 3 }, shape),
            ];
            assert_matches_tape(&nodes, 2, &inputs());
            assert_matches_tape(&nodes, 4, &inputs());
        }
    }

    #[test]
    fn forward_uses_positions_for_noncanonical_node_ids() {
        let nodes = vec![
            node(
                99,
                TraceOp::Input {
                    name: "x".to_string(),
                },
                (2, 2),
            ),
            node(77, TraceOp::Add { lhs: 0, rhs: 0 }, (2, 2)),
        ];
        assert_matches_tape(&nodes, 0, &inputs());
        assert_matches_tape(&nodes, 1, &inputs());
    }

    #[test]
    fn forward_preserves_input_bits_and_eager_errors() {
        let mut inputs = inputs();
        inputs.insert(
            "bits".to_string(),
            PyMatrix::from_rows(vec![vec![
                -0.0,
                f64::from_bits(0x7ff0_0000_0000_0001),
                f64::INFINITY,
                f64::NEG_INFINITY,
            ]])
            .unwrap(),
        );
        let nodes = vec![node(
            0,
            TraceOp::Input {
                name: "bits".to_string(),
            },
            (1, 4),
        )];
        assert_matches_tape(&nodes, 0, &inputs);
        assert_matches_tape(&nodes, 1, &inputs);
        assert_matches_tape(&nodes, 0, &HashMap::new());
        let wrong_shape = HashMap::from([("bits".to_string(), PyMatrix::filled(1, 1, 0.0))]);
        assert_matches_tape(&nodes, 0, &wrong_shape);

        let eager = vec![
            node(
                0,
                TraceOp::Input {
                    name: "missing".to_string(),
                },
                (2, 2),
            ),
            node(
                1,
                TraceOp::Input {
                    name: "x".to_string(),
                },
                (2, 2),
            ),
        ];
        assert_matches_tape(&eager, 1, &inputs);
        let invalid = vec![
            node(0, TraceOp::ScalarConstant { value: 1.0 }, (2, 2)),
            node(1, TraceOp::Add { lhs: 0, rhs: 3 }, (2, 2)),
            node(
                2,
                TraceOp::Input {
                    name: "x".to_string(),
                },
                (2, 2),
            ),
        ];
        assert_matches_tape(&invalid, 2, &inputs);
        let wrong_result = vec![
            node(
                0,
                TraceOp::Input {
                    name: "x".to_string(),
                },
                (2, 2),
            ),
            node(1, TraceOp::Transpose { input: 0 }, (1, 4)),
            node(
                2,
                TraceOp::Input {
                    name: "y".to_string(),
                },
                (2, 2),
            ),
        ];
        assert_matches_tape(&wrong_result, 2, &inputs);
    }
}
