//! Operand binding shared by the Python operation adapters. Trace ownership
//! remains with `TraceTensor`; binding never creates a new trace or broadens
//! an operation's accepted operand types.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PySequence;
use quabla_core::tensor_ir::TensorComparison;

use crate::safe_extract;
use crate::{PyTensor, TraceTensor};

/// Array operands of a sequence operation. Eager views are accepted only
/// when a tracer supplies the trace into which they can be captured.
pub(crate) enum ArrayOperands {
    Eager(Vec<PyTensor>),
    Traced(Vec<TraceTensor>),
}

impl ArrayOperands {
    /// Reuses the adapter's length read so axis validation and sequence access
    /// keep their order. `None` leaves the adapter responsible for its type error.
    pub(crate) fn extract(items: &Bound<'_, PySequence>, len: usize) -> PyResult<Option<Self>> {
        let mut eager = safe_extract::buffer(len)?;
        for index in 0..len {
            let item = items.get_item(index)?;
            match item.extract::<PyRef<'_, PyTensor>>() {
                Ok(tensor) => eager.push(tensor.clone()),
                Err(_) => return traced_sequence(items).map(|items| items.map(Self::Traced)),
            }
        }
        Ok(Some(Self::Eager(eager)))
    }
}

/// An eager method's operation after its operands have joined a Trace.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TracedBinary {
    Arithmetic(&'static str),
    Compare(TensorComparison),
    Maximum,
    Minimum,
    Matmul,
    Solve,
    SolveTriangular { lower: bool, transpose: bool },
    Logical { and: bool },
}

impl TracedBinary {
    pub(crate) fn apply(self, lhs: &TraceTensor, rhs: &TraceTensor) -> Result<TraceTensor, String> {
        match self {
            Self::Arithmetic(op) => lhs.binary(rhs, op),
            Self::Compare(kind) => lhs.compare_tensor(rhs, kind),
            Self::Maximum => lhs.maximum_tensor(rhs),
            Self::Minimum => lhs.minimum_tensor(rhs),
            Self::Matmul => lhs.matmul_tensor(rhs),
            Self::Solve => lhs.solve_tensor(rhs),
            Self::SolveTriangular { lower, transpose } => {
                lhs.solve_triangular_tensor(rhs, lower, transpose)
            }
            Self::Logical { and } => lhs.logical_tensor(rhs, and),
        }
    }
}

/// Captures the eager operand without reversing the operation. `None` lets
/// the eager method retain its own scalar handling and type errors.
pub(crate) fn eager_traced_binary(
    eager: &PyTensor,
    operand: &Bound<'_, PyAny>,
    op: TracedBinary,
    eager_first: bool,
) -> Option<PyResult<TraceTensor>> {
    let tracer = trace_anchor([operand])?;
    let result = tracer.capture(eager).and_then(|constant| {
        if eager_first {
            op.apply(&constant, &tracer)
        } else {
            op.apply(&tracer, &constant)
        }
    });
    Some(result.map_err(PyValueError::new_err))
}

fn traced_sequence(items: &Bound<'_, PySequence>) -> PyResult<Option<Vec<TraceTensor>>> {
    let items = safe_extract::vec::<Bound<'_, PyAny>>(items.as_any())?;
    let Some(tracer) = trace_anchor(items.iter()) else {
        return Ok(None);
    };
    items
        .iter()
        .map(|item| tracer.traced_operand(item))
        .collect::<Option<PyResult<Vec<_>>>>()
        .transpose()
}

/// The first tracer anchors extraction, not ownership: the traced operation
/// still lifts active enclosing tracers into the innermost trace.
pub(crate) fn trace_anchor<'a, 'py: 'a>(
    operands: impl IntoIterator<Item = &'a Bound<'py, PyAny>>,
) -> Option<TraceTensor> {
    operands
        .into_iter()
        .find_map(|operand| operand.extract::<TraceTensor>().ok())
}

impl TraceTensor {
    /// A tracer is checked for liftability; an eager array or view becomes
    /// a constant. `None` lets the caller choose scalar support or its error.
    pub(crate) fn traced_operand(&self, operand: &Bound<'_, PyAny>) -> Option<PyResult<Self>> {
        if let Ok(tracer) = operand.extract::<PyRef<'_, TraceTensor>>() {
            return Some(
                Self::ensure_liftable(&[self, &tracer])
                    .map(|()| tracer.clone())
                    .map_err(|message| crate::errors::foreign_tracer_error(operand.py(), message)),
            );
        }
        let captured = if let Ok(tensor) = operand.extract::<PyRef<'_, PyTensor>>() {
            self.capture(&tensor)
        } else if let Ok(view) = operand.extract::<PyRef<'_, crate::PyTensorView>>() {
            view.materialize().and_then(|tensor| self.capture(&tensor))
        } else {
            return None;
        };
        Some(captured.map_err(PyValueError::new_err))
    }
}
