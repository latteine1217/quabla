//! Direct CPU kernels for the eager ops that dominate small-array latency.
//!
//! An eager array op is defined by the graph a trace of it records: the
//! Python bindings append that graph to a scratch [`TensorIr`] and evaluate
//! it, so eager and `jit` results agree bit for bit. Building and walking a
//! graph costs a few hundred nanoseconds per node, several times a small
//! `add` or `exp` itself, so the ops below can instead run the kernel that
//! the CPU evaluator runs for their node, followed by the same rounding to
//! the node dtype. A kernel applies only where the graph would hold exactly
//! that node over its operands, which already have the dtypes promotion
//! gives them, so no `Cast` is recorded. The `matches_one_node_graph` test
//! checks every kernel against the graph evaluation.
//!
//! [`TensorIr`]: super::TensorIr

use std::sync::Arc;

use super::{
    matmul_host_float_block, BinaryMathKind, DynamicTensor, HostTensorStorage, TensorComparison,
    TensorDType, UnaryMathKind,
};

/// A single-node op of the CPU evaluator that eager arrays run directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EagerKernel {
    Add,
    Sub,
    Mul,
    Div,
    /// The legacy `gt` mask: `lhs > rhs` as 0/1 values of the operand dtype.
    Greater,
    /// An IEEE comparison with a `Bool` result.
    Compare(TensorComparison),
    /// `where(condition, on_true, on_false)`: the condition selects where it
    /// is nonzero (`NaN` included).
    Where,
    Sqrt,
    /// A `UnaryMath` node: every kind runs its own `f64` rule, so a new kind
    /// needs no change here.
    UnaryMath(UnaryMathKind),
    /// A `BinaryMath` node (`pow`, `atan2`, `fmod`), likewise by its kind.
    BinaryMath(BinaryMathKind),
    Sum,
    SumAxis(usize),
    Mean,
    MeanAxis(usize),
    Matmul,
}

/// An operand of an [`EagerKernel`], borrowed from the caller's array.
#[derive(Clone, Copy, Debug)]
pub enum EagerOperand<'a> {
    Array {
        shape: &'a [usize],
        storage: &'a HostTensorStorage,
    },
    /// A rank-0 value of `dtype`, already rounded to it: a weak Python
    /// scalar after the promotion cast to the other operand's dtype.
    Scalar(f64, TensorDType),
}

impl<'a> EagerOperand<'a> {
    pub fn array(tensor: &'a DynamicTensor) -> Self {
        Self::Array {
            shape: &tensor.shape,
            storage: &tensor.data,
        }
    }

    fn dtype(self) -> TensorDType {
        match self {
            Self::Array { storage, .. } => storage.dtype(),
            Self::Scalar(_, dtype) => dtype,
        }
    }

    fn shape(self) -> &'a [usize] {
        match self {
            Self::Array { shape, .. } => shape,
            Self::Scalar(..) => &[],
        }
    }

    fn to_tensor(self) -> Result<DynamicTensor, String> {
        match self {
            Self::Array { shape, storage } => {
                DynamicTensor::from_storage(shape.to_vec(), storage.clone())
            }
            Self::Scalar(value, dtype) => DynamicTensor::with_dtype(vec![], vec![value], dtype),
        }
    }

    /// The `f64` values, broadcast when the operand has rank 0.
    fn lanes(self) -> Lanes<'a> {
        match self {
            Self::Scalar(value, _) => Lanes::Scalar(value),
            Self::Array { shape: [], storage } => Lanes::Scalar(storage.get(0)),
            Self::Array {
                storage: HostTensorStorage::F64(values),
                ..
            } => Lanes::F64(values),
            Self::Array {
                storage: HostTensorStorage::F32(values),
                ..
            } => Lanes::F32(values),
            Self::Array {
                storage: HostTensorStorage::Bool(values),
                ..
            } => Lanes::Bool(values),
        }
    }
}

impl EagerKernel {
    /// The value of this op's node over `operands`, as the CPU evaluator
    /// computes it: in `f64`, rounded to the node dtype. The arithmetic,
    /// math, reduction, and `Greater` kernels take operands of one floating
    /// dtype, which is the node dtype; `Compare` takes two operands of one
    /// dtype and gives `Bool`; `Where` takes a condition of any dtype and two
    /// values of one dtype, which is the node dtype. Other operands, and
    /// invalid shapes and axes, return an error.
    pub fn evaluate(self, operands: &[EagerOperand<'_>]) -> Result<DynamicTensor, String> {
        let dtype = self
            .output_dtype(operands)
            .ok_or_else(|| format!("eager kernel {self:?} does not accept these operand dtypes"))?;
        if let Some(value) = self.evaluate_lanes(operands, dtype) {
            return Ok(value);
        }
        if let (Self::Matmul, [lhs, rhs]) = (self, operands) {
            if let Some(value) = matmul_matrices(*lhs, *rhs, dtype) {
                return Ok(value);
            }
        }
        let operands = operands
            .iter()
            .map(|operand| operand.to_tensor())
            .collect::<Result<Vec<_>, _>>()?;
        let input = &operands[0];
        let rhs = || &operands[1];
        let value = match self {
            Self::Add => input.add(rhs()),
            Self::Sub => input.sub(rhs()),
            Self::Mul => input.mul(rhs()),
            Self::Div => input.div(rhs()),
            Self::Greater => input.greater(rhs()),
            Self::Compare(kind) => input.compare(rhs(), kind),
            Self::Where => input.where_select(&operands[1], &operands[2]),
            Self::Sqrt => input.sqrt(),
            Self::UnaryMath(kind) => input.map_f64(|x| kind.evaluate(x)),
            Self::BinaryMath(kind) => input.elementwise(rhs(), kind.function()),
            Self::Sum => input.sum_all(),
            Self::SumAxis(axis) => input.reduce_axis(axis, 1.0),
            Self::Mean => input.mean_all(),
            Self::MeanAxis(axis) => {
                let extent = *input.shape.get(axis).ok_or_else(|| {
                    format!("axis {axis} is out of bounds for shape {:?}", input.shape)
                })?;
                input.reduce_axis(axis, 1.0 / extent as f64)
            }
            Self::Matmul => input.matmul(rhs()),
        }?;
        Ok(value.into_dtype(dtype))
    }

    /// The node dtype for `operands`, or `None` when the kernel does not
    /// accept their count or dtypes.
    fn output_dtype(self, operands: &[EagerOperand<'_>]) -> Option<TensorDType> {
        let floating = |dtype: TensorDType| dtype.is_floating().then_some(dtype);
        match (self, operands) {
            (Self::Compare(_), [lhs, rhs]) => {
                (lhs.dtype() == rhs.dtype()).then_some(TensorDType::Bool)
            }
            (Self::Where, [_, on_true, on_false]) => {
                (on_true.dtype() == on_false.dtype()).then(|| on_true.dtype())
            }
            (Self::Compare(_) | Self::Where, _) => None,
            (
                Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Greater
                | Self::BinaryMath(_)
                | Self::Matmul,
                [lhs, rhs],
            ) if lhs.dtype() == rhs.dtype() => floating(lhs.dtype()),
            (
                Self::Add
                | Self::Sub
                | Self::Mul
                | Self::Div
                | Self::Greater
                | Self::BinaryMath(_)
                | Self::Matmul,
                _,
            ) => None,
            (_, [input]) => floating(input.dtype()),
            _ => None,
        }
    }

    /// The elementwise kernels for operands of one shape, or rank-0 operands
    /// against one: each lane computes the evaluator's `f64` function and
    /// rounds once to `dtype`, which equals the evaluator's `f64` result
    /// followed by the node rounding, without the `f64` intermediate array.
    /// `None` for every other case.
    fn evaluate_lanes(
        self,
        operands: &[EagerOperand<'_>],
        dtype: TensorDType,
    ) -> Option<DynamicTensor> {
        let mut shape: &[usize] = &[];
        for operand in operands {
            match (operand.shape(), shape) {
                ([], _) => {}
                (operand, []) => shape = operand,
                (operand, shape) if operand == shape => {}
                _ => return None,
            }
        }
        let count = shape.iter().product::<usize>();
        let lane = |index: usize| operands[index].lanes();
        // IEEE 754 leaves the NaN that an arithmetic op of two NaN operands
        // returns to the implementation, and the compiler may commute the
        // op, so that choice depends on how a loop is compiled. Operands
        // that both hold a NaN therefore run the evaluator's own kernel.
        if matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
            && lane(0).any_nan()
            && lane(1).any_nan()
        {
            return None;
        }
        let data = match self {
            Self::Add => binary_lanes(lane(0), lane(1), count, dtype, |x, y| x + y),
            Self::Sub => binary_lanes(lane(0), lane(1), count, dtype, |x, y| x - y),
            Self::Mul => binary_lanes(lane(0), lane(1), count, dtype, |x, y| x * y),
            Self::Div => binary_lanes(lane(0), lane(1), count, dtype, |x, y| x / y),
            Self::Greater => binary_lanes(lane(0), lane(1), count, dtype, |x, y| f64::from(x > y)),
            Self::Compare(kind) => binary_lanes(lane(0), lane(1), count, dtype, |x, y| {
                f64::from(kind.evaluate(x, y))
            }),
            Self::Where => select_lanes(lane(0), lane(1), lane(2), count, dtype),
            Self::UnaryMath(kind) => unary_lanes(lane(0), count, dtype, kind.function()),
            Self::BinaryMath(kind) => binary_lanes(lane(0), lane(1), count, dtype, kind.function()),
            // The evaluator's `sum_all`: a sequential `f64` sum in storage
            // order, and for `Mean` that sum times `1 / count`.
            Self::Sum | Self::Mean => {
                let sum = match lane(0) {
                    Lanes::F64(values) => values.iter().sum::<f64>(),
                    Lanes::F32(values) => values.iter().map(|&value| f64::from(value)).sum(),
                    lanes => (0..count).map(|index| lanes.get(index)).sum(),
                };
                let value = if self == Self::Mean {
                    sum * (1.0 / count as f64)
                } else {
                    sum
                };
                return Some(DynamicTensor {
                    shape: Vec::new(),
                    data: store(dtype, std::iter::once(value)),
                    dtype,
                });
            }
            // `Sqrt` keeps the evaluator's kernel: inlined with its constant
            // order, `powf(x, 0.5)` may compile to `fabs(sqrt(x))`, which
            // drops the sign of a NaN that the evaluator's call keeps.
            _ => return None,
        };
        Some(DynamicTensor {
            shape: shape.to_vec(),
            data,
            dtype,
        })
    }
}

/// The `f64` values of one operand, broadcast when it has rank 0.
#[derive(Clone, Copy)]
enum Lanes<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    Bool(&'a [u8]),
    Scalar(f64),
}

impl Lanes<'_> {
    fn any_nan(&self) -> bool {
        match self {
            Self::F64(values) => values.iter().any(|value| value.is_nan()),
            Self::F32(values) => values.iter().any(|value| value.is_nan()),
            Self::Bool(_) => false,
            Self::Scalar(value) => value.is_nan(),
        }
    }

    #[inline(always)]
    fn get(&self, index: usize) -> f64 {
        match self {
            Self::F64(values) => values[index],
            Self::F32(values) => f64::from(values[index]),
            Self::Bool(values) => f64::from(values[index] != 0),
            Self::Scalar(value) => *value,
        }
    }
}

/// The product of two matrices (rank 2, no batch axes) with the
/// evaluator's block kernel, without its batch bookkeeping. `None` for
/// other operands.
fn matmul_matrices(
    lhs: EagerOperand<'_>,
    rhs: EagerOperand<'_>,
    dtype: TensorDType,
) -> Option<DynamicTensor> {
    let (
        EagerOperand::Array {
            shape: &[rows, inner],
            storage: lhs,
        },
        EagerOperand::Array {
            shape: &[rhs_inner, columns],
            storage: rhs,
        },
    ) = (lhs, rhs)
    else {
        return None;
    };
    if inner != rhs_inner {
        return None;
    }
    let mut output = vec![0.0; rows * columns];
    let dimensions = [rows, inner, columns];
    match (lhs, rhs) {
        (HostTensorStorage::F64(lhs), HostTensorStorage::F64(rhs)) => {
            matmul_host_float_block(lhs, rhs, &mut output, dimensions)
        }
        (HostTensorStorage::F32(lhs), HostTensorStorage::F32(rhs)) => {
            matmul_host_float_block(lhs, rhs, &mut output, dimensions)
        }
        _ => return None,
    }
    let data = match dtype {
        TensorDType::F64 => HostTensorStorage::F64(Arc::new(output)),
        dtype => store(dtype, output.into_iter()),
    };
    Some(DynamicTensor {
        shape: vec![rows, columns],
        data,
        dtype,
    })
}

/// Binds `$name` to a function reading the lanes `$lanes` and evaluates
/// `$body` once per storage kind, so that a loop in `$body` is compiled for
/// its element type instead of matching the kind per element.
macro_rules! with_lanes {
    ($lanes:expr, $name:ident => $body:expr) => {
        match $lanes {
            Lanes::F64(values) => {
                let $name = |index: usize| values[index];
                $body
            }
            Lanes::F32(values) => {
                let $name = |index: usize| f64::from(values[index]);
                $body
            }
            Lanes::Bool(values) => {
                let $name = |index: usize| f64::from(values[index] != 0);
                $body
            }
            Lanes::Scalar(value) => {
                let $name = |_: usize| value;
                $body
            }
        }
    };
}

/// `where(condition, on_true, on_false)` lane by lane.
fn select_lanes(
    condition: Lanes<'_>,
    on_true: Lanes<'_>,
    on_false: Lanes<'_>,
    count: usize,
    dtype: TensorDType,
) -> HostTensorStorage {
    fn select(
        selected: impl Fn(usize) -> bool,
        on_true: Lanes<'_>,
        on_false: Lanes<'_>,
        count: usize,
        dtype: TensorDType,
    ) -> HostTensorStorage {
        with_lanes!(on_true, on_true => with_lanes!(on_false, on_false => store(
            dtype,
            (0..count).map(|index| if selected(index) { on_true(index) } else { on_false(index) }),
        )))
    }
    match condition {
        Lanes::Bool(mask) => select(|index| mask[index] != 0, on_true, on_false, count, dtype),
        condition => select(
            |index| condition.get(index) != 0.0,
            on_true,
            on_false,
            count,
            dtype,
        ),
    }
}

/// `function` of every lane of `input`. The common storages get their own
/// loop, so that each is compiled for its element type.
fn unary_lanes(
    input: Lanes<'_>,
    count: usize,
    dtype: TensorDType,
    function: impl Fn(f64) -> f64,
) -> HostTensorStorage {
    match input {
        Lanes::F64(values) => store(dtype, values.iter().map(|&x| function(x))),
        Lanes::F32(values) => store(dtype, values.iter().map(|&x| function(f64::from(x)))),
        input => store(dtype, (0..count).map(|index| function(input.get(index)))),
    }
}

/// `function` of every pair of lanes, with the same specialization as
/// [`unary_lanes`] for two arrays or an array and a scalar.
fn binary_lanes(
    lhs: Lanes<'_>,
    rhs: Lanes<'_>,
    count: usize,
    dtype: TensorDType,
    function: impl Fn(f64, f64) -> f64,
) -> HostTensorStorage {
    match (lhs, rhs) {
        (Lanes::F64(lhs), Lanes::F64(rhs)) => {
            store(dtype, lhs.iter().zip(rhs).map(|(&x, &y)| function(x, y)))
        }
        (Lanes::F32(lhs), Lanes::F32(rhs)) => store(
            dtype,
            lhs.iter()
                .zip(rhs)
                .map(|(&x, &y)| function(f64::from(x), f64::from(y))),
        ),
        (Lanes::F64(lhs), Lanes::Scalar(y)) => store(dtype, lhs.iter().map(|&x| function(x, y))),
        (Lanes::Scalar(x), Lanes::F64(rhs)) => store(dtype, rhs.iter().map(|&y| function(x, y))),
        (Lanes::F32(lhs), Lanes::Scalar(y)) => {
            store(dtype, lhs.iter().map(|&x| function(f64::from(x), y)))
        }
        (Lanes::Scalar(x), Lanes::F32(rhs)) => {
            store(dtype, rhs.iter().map(|&y| function(x, f64::from(y))))
        }
        (lhs, rhs) => store(
            dtype,
            (0..count).map(|index| function(lhs.get(index), rhs.get(index))),
        ),
    }
}

/// `values` rounded to `dtype` and stored at its width, as
/// `HostTensorStorage::into_dtype` rounds an `f64` array.
fn store(dtype: TensorDType, values: impl Iterator<Item = f64>) -> HostTensorStorage {
    match dtype {
        TensorDType::F64 => HostTensorStorage::F64(Arc::new(values.collect())),
        // A signaling NaN that reaches the output unchanged (a selected
        // value) is quieted, as the evaluator's round trip through an `f64`
        // array quiets it; the compiler may fold that round trip here.
        TensorDType::F32 => HostTensorStorage::F32(Arc::new(
            values
                .map(|value| {
                    let value = value as f32;
                    if value.is_nan() {
                        f32::from_bits(value.to_bits() | 0x0040_0000)
                    } else {
                        value
                    }
                })
                .collect(),
        )),
        TensorDType::Bool => HostTensorStorage::Bool(Arc::new(
            values.map(|value| u8::from(value != 0.0)).collect(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{TensorIr, TensorNodeId};
    use super::*;
    use std::collections::BTreeMap;

    type Build = fn(&mut TensorIr, &[TensorNodeId]) -> Result<TensorNodeId, String>;

    /// Special values, cycled from `offset`: signed zeros, subnormals,
    /// infinities, NaNs of both signs and a payload, and domain edges.
    fn values(count: usize, offset: usize) -> Vec<f64> {
        let special = [
            0.0,
            -0.0,
            1e-310,
            -1e-30,
            0.5,
            -1.5,
            2.5,
            1e30,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            f64::from_bits(0xfff8_0000_0000_1234),
            0.9999999,
            std::f64::consts::FRAC_PI_2,
            88.5,
            1.0,
            -1.0,
        ];
        (0..count)
            .map(|index| special[(index * 7 + offset) % special.len()])
            .collect()
    }

    fn operand(shape: &[usize], dtype: TensorDType, offset: usize) -> DynamicTensor {
        let count = shape.iter().product();
        DynamicTensor::with_dtype(shape.to_vec(), values(count, offset), dtype).unwrap()
    }

    /// An `f32` operand whose first element is a signaling NaN, which only a
    /// buffer import can create.
    fn signaling_f32(shape: &[usize]) -> DynamicTensor {
        let count: usize = shape.iter().product();
        let mut data = values(count, 5)
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        data[0] = f32::from_bits(0xff80_0123);
        DynamicTensor::from_storage(shape.to_vec(), HostTensorStorage::F32(Arc::new(data))).unwrap()
    }

    fn bits(tensor: &DynamicTensor) -> Vec<u64> {
        tensor.data().iter().map(|value| value.to_bits()).collect()
    }

    fn check(
        kernel: EagerKernel,
        build: impl Fn(&mut TensorIr, &[TensorNodeId]) -> Result<TensorNodeId, String>,
        operands: &[DynamicTensor],
    ) {
        let mut ir = TensorIr::new();
        let nodes = operands
            .iter()
            .map(|operand| ir.constant(operand.clone(), false))
            .collect::<Vec<_>>();
        let output = build(&mut ir, &nodes).unwrap();
        let expected = ir.evaluate(output, &BTreeMap::new()).unwrap();
        let borrowed = operands.iter().map(EagerOperand::array).collect::<Vec<_>>();
        let actual = kernel.evaluate(&borrowed).unwrap();
        compare(kernel, operands, &actual, &expected);
    }

    fn compare(
        kernel: EagerKernel,
        operands: &[DynamicTensor],
        actual: &DynamicTensor,
        expected: &DynamicTensor,
    ) {
        let shapes = operands
            .iter()
            .map(DynamicTensor::shape)
            .collect::<Vec<_>>();
        let context = format!("{kernel:?} {shapes:?} {:?}", operands[0].dtype());
        assert_eq!(actual.shape(), expected.shape(), "{context}");
        assert_eq!(actual.dtype(), expected.dtype(), "{context}");
        assert_eq!(bits(actual), bits(expected), "{context}");
        if let (HostTensorStorage::F32(actual), HostTensorStorage::F32(expected)) =
            (actual.storage(), expected.storage())
        {
            let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(actual), bits(expected), "{context}");
        }
    }

    #[test]
    fn matches_one_node_graph() {
        let unary: Vec<(EagerKernel, Build)> = vec![
            (EagerKernel::Sqrt, |ir, x| ir.sqrt(x[0])),
            (EagerKernel::Sum, |ir, x| ir.sum(x[0])),
            (EagerKernel::Mean, |ir, x| ir.mean(x[0])),
        ];
        let reductions: Vec<(EagerKernel, Build)> = vec![
            (EagerKernel::SumAxis(1), |ir, x| ir.sum_axis(x[0], 1)),
            (EagerKernel::MeanAxis(0), |ir, x| ir.mean_axis(x[0], 0)),
        ];
        let binary: Vec<(EagerKernel, Build)> = vec![
            (EagerKernel::Add, |ir, x| ir.add(x[0], x[1])),
            (EagerKernel::Sub, |ir, x| ir.sub(x[0], x[1])),
            (EagerKernel::Mul, |ir, x| ir.mul(x[0], x[1])),
            (EagerKernel::Div, |ir, x| ir.div(x[0], x[1])),
            (EagerKernel::Greater, |ir, x| ir.greater(x[0], x[1])),
            (EagerKernel::Compare(TensorComparison::Less), |ir, x| {
                ir.compare(x[0], x[1], TensorComparison::Less)
            }),
            (
                EagerKernel::Compare(TensorComparison::GreaterEqual),
                |ir, x| ir.compare(x[0], x[1], TensorComparison::GreaterEqual),
            ),
            (EagerKernel::Compare(TensorComparison::NotEqual), |ir, x| {
                ir.compare(x[0], x[1], TensorComparison::NotEqual)
            }),
            (EagerKernel::Compare(TensorComparison::Equal), |ir, x| {
                ir.compare(x[0], x[1], TensorComparison::Equal)
            }),
        ];
        let where_select: Build = |ir, x| ir.where_select(x[0], x[1], x[2]);
        for dtype in [TensorDType::F32, TensorDType::F64] {
            for kind in UnaryMathKind::ALL {
                let build = |ir: &mut TensorIr, x: &[TensorNodeId]| ir.unary_math(x[0], kind);
                check(
                    EagerKernel::UnaryMath(kind),
                    build,
                    &[operand(&[4, 6], dtype, 0)],
                );
                check(
                    EagerKernel::UnaryMath(kind),
                    build,
                    &[operand(&[], dtype, 3)],
                );
            }
            for kind in BinaryMathKind::ALL {
                let build =
                    |ir: &mut TensorIr, x: &[TensorNodeId]| ir.binary_math(x[0], x[1], kind);
                for (lhs, rhs) in [
                    (vec![4, 6], vec![4, 6]),
                    (vec![4, 1], vec![1, 6]),
                    (vec![4, 6], vec![]),
                    (vec![], vec![4, 6]),
                ] {
                    let operands = [operand(&lhs, dtype, 0), operand(&rhs, dtype, 3)];
                    check(EagerKernel::BinaryMath(kind), build, &operands);
                }
            }
            for (kernel, build) in &unary {
                check(*kernel, *build, &[operand(&[4, 6], dtype, 0)]);
                check(*kernel, *build, &[operand(&[], dtype, 3)]);
                for zeros in [vec![-0.0], vec![-0.0, -0.0], vec![-0.0, 0.0]] {
                    let shape = vec![zeros.len()];
                    let zeros = DynamicTensor::with_dtype(shape, zeros, dtype).unwrap();
                    check(*kernel, *build, &[zeros]);
                }
            }
            for (kernel, build) in &reductions {
                check(*kernel, *build, &[operand(&[4, 6], dtype, 0)]);
            }
            for (kernel, build) in &binary {
                for (lhs, rhs) in [
                    (vec![4, 6], vec![4, 6]),
                    (vec![4, 1], vec![1, 6]),
                    (vec![4, 6], vec![]),
                    (vec![], vec![4, 6]),
                    (vec![], vec![]),
                ] {
                    let operands = [operand(&lhs, dtype, 0), operand(&rhs, dtype, 3)];
                    check(*kernel, *build, &operands);
                }
            }
            check(
                EagerKernel::Matmul,
                |ir, x| ir.matmul(x[0], x[1]),
                &[operand(&[2, 4, 6], dtype, 0), operand(&[6, 3], dtype, 3)],
            );
            check(
                EagerKernel::Matmul,
                |ir, x| ir.matmul(x[0], x[1]),
                &[operand(&[5, 6], dtype, 1), operand(&[6, 4], dtype, 4)],
            );
            for (condition, on_true, on_false) in [
                (vec![4, 6], vec![4, 6], vec![4, 6]),
                (vec![4, 6], vec![], vec![4, 6]),
                (vec![], vec![4, 6], vec![4, 6]),
                (vec![4, 1], vec![1, 6], vec![4, 6]),
            ] {
                for condition_dtype in [TensorDType::Bool, dtype] {
                    let operands = [
                        operand(&condition, condition_dtype, 1),
                        operand(&on_true, dtype, 2),
                        operand(&on_false, dtype, 7),
                    ];
                    check(EagerKernel::Where, where_select, &operands);
                }
            }
        }
        let masks = [
            operand(&[4, 6], TensorDType::Bool, 0),
            operand(&[4, 6], TensorDType::Bool, 1),
        ];
        let selected = [masks[0].clone(), masks[0].clone(), masks[1].clone()];
        check(EagerKernel::Where, where_select, &selected);
        check(
            EagerKernel::Compare(TensorComparison::Equal),
            |ir, x| ir.compare(x[0], x[1], TensorComparison::Equal),
            &masks,
        );
        let signaling = signaling_f32(&[3, 2]);
        let plain = operand(&[3, 2], TensorDType::F32, 4);
        let mask = operand(&[3, 2], TensorDType::Bool, 0);
        let selected = [mask.clone(), signaling.clone(), plain.clone()];
        check(EagerKernel::Where, where_select, &selected);
        let selected = [mask, plain.clone(), signaling.clone()];
        check(EagerKernel::Where, where_select, &selected);
        let added = [signaling.clone(), plain];
        check(EagerKernel::Add, |ir, x| ir.add(x[0], x[1]), &added);
        check(
            EagerKernel::UnaryMath(UnaryMathKind::Exp),
            |ir, x| ir.exp(x[0]),
            std::slice::from_ref(&signaling),
        );
        check(
            EagerKernel::UnaryMath(UnaryMathKind::Floor),
            |ir, x| ir.unary_math(x[0], UnaryMathKind::Floor),
            &[signaling],
        );
    }

    /// Arithmetic with NaN operands, in every shape case (same shape, a
    /// rank-0 operand on either side), returns the evaluator's NaN.
    #[test]
    fn nan_operands_propagate_like_the_evaluator() {
        let nans = [
            f64::NAN,
            f64::from_bits(0xfff8_0000_0000_1234),
            -f64::NAN,
            f64::from_bits(0x7ff8_0000_0000_0042),
            1.5,
            -0.0,
        ];
        let pairs: Vec<(f64, f64)> = nans
            .iter()
            .flat_map(|&lhs| nans.iter().map(move |&rhs| (lhs, rhs)))
            .collect();
        let kernels: Vec<(EagerKernel, Build)> = vec![
            (EagerKernel::Add, |ir, x| ir.add(x[0], x[1])),
            (EagerKernel::Sub, |ir, x| ir.sub(x[0], x[1])),
            (EagerKernel::Mul, |ir, x| ir.mul(x[0], x[1])),
            (EagerKernel::Div, |ir, x| ir.div(x[0], x[1])),
            (EagerKernel::BinaryMath(BinaryMathKind::Pow), |ir, x| {
                ir.binary_math(x[0], x[1], BinaryMathKind::Pow)
            }),
            (EagerKernel::BinaryMath(BinaryMathKind::Atan2), |ir, x| {
                ir.binary_math(x[0], x[1], BinaryMathKind::Atan2)
            }),
            (EagerKernel::BinaryMath(BinaryMathKind::Fmod), |ir, x| {
                ir.binary_math(x[0], x[1], BinaryMathKind::Fmod)
            }),
        ];
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let lhs = pairs.iter().map(|pair| pair.0).collect::<Vec<_>>();
            let rhs = pairs.iter().map(|pair| pair.1).collect::<Vec<_>>();
            let lhs = DynamicTensor::with_dtype(vec![lhs.len()], lhs, dtype).unwrap();
            let rhs = DynamicTensor::with_dtype(vec![rhs.len()], rhs, dtype).unwrap();
            for (kernel, build) in &kernels {
                check(*kernel, *build, &[lhs.clone(), rhs.clone()]);
                for &value in &nans {
                    let scalar = DynamicTensor::with_dtype(vec![], vec![value], dtype).unwrap();
                    check(*kernel, *build, &[lhs.clone(), scalar.clone()]);
                    check(*kernel, *build, &[scalar, rhs.clone()]);
                }
            }
        }
    }

    /// A Python-number operand: the graph holds a weak scalar constant that
    /// promotion casts to the array's dtype; the kernel takes it rounded.
    #[test]
    fn scalar_operand_matches_cast_weak_constant() {
        let kernels: Vec<(EagerKernel, Build)> = vec![
            (EagerKernel::Add, |ir, x| ir.add(x[0], x[1])),
            (EagerKernel::Sub, |ir, x| ir.sub(x[0], x[1])),
            (EagerKernel::Mul, |ir, x| ir.mul(x[0], x[1])),
            (EagerKernel::Div, |ir, x| ir.div(x[0], x[1])),
            (EagerKernel::Greater, |ir, x| ir.greater(x[0], x[1])),
            (
                EagerKernel::Compare(TensorComparison::LessEqual),
                |ir, x| ir.compare(x[0], x[1], TensorComparison::LessEqual),
            ),
        ];
        for dtype in [TensorDType::F32, TensorDType::F64] {
            let array = operand(&[3, 5], dtype, 2);
            for value in [0.1, -0.0, 0.0, 1e300, -2.5, f64::NAN, f64::INFINITY] {
                for scalar_first in [false, true] {
                    for (kernel, build) in &kernels {
                        let mut ir = TensorIr::new();
                        let tensor = ir.constant(array.clone(), false);
                        let scalar = ir.scalar_constant(value);
                        let nodes = if scalar_first {
                            [scalar, tensor]
                        } else {
                            [tensor, scalar]
                        };
                        let output = build(&mut ir, &nodes).unwrap();
                        let expected = ir.evaluate(output, &BTreeMap::new()).unwrap();
                        let scalar = EagerOperand::Scalar(dtype.round(value), dtype);
                        let operands = if scalar_first {
                            [scalar, EagerOperand::array(&array)]
                        } else {
                            [EagerOperand::array(&array), scalar]
                        };
                        let actual = kernel.evaluate(&operands).unwrap();
                        compare(*kernel, std::slice::from_ref(&array), &actual, &expected);
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_operands_whose_graph_has_casts() {
        let f32 = operand(&[2], TensorDType::F32, 0);
        let f64 = operand(&[2], TensorDType::F64, 0);
        let mask = operand(&[2], TensorDType::Bool, 0);
        assert!(EagerKernel::Add
            .evaluate(&[EagerOperand::array(&f32), EagerOperand::array(&f64)])
            .is_err());
        assert!(EagerKernel::Add
            .evaluate(&[EagerOperand::array(&mask), EagerOperand::array(&mask)])
            .is_err());
        assert!(EagerKernel::Greater
            .evaluate(&[EagerOperand::array(&mask), EagerOperand::array(&mask)])
            .is_err());
        let exp = EagerKernel::UnaryMath(UnaryMathKind::Exp);
        assert!(exp.evaluate(&[EagerOperand::array(&mask)]).is_err());
        assert!(exp
            .evaluate(&[EagerOperand::array(&f32), EagerOperand::array(&f32)])
            .is_err());
        assert!(EagerKernel::BinaryMath(BinaryMathKind::Pow)
            .evaluate(&[EagerOperand::array(&f32), EagerOperand::array(&f64)])
            .is_err());
        assert!(EagerKernel::Compare(TensorComparison::Less)
            .evaluate(&[EagerOperand::array(&f32), EagerOperand::array(&f64)])
            .is_err());
        assert!(EagerKernel::Where
            .evaluate(&[
                EagerOperand::array(&mask),
                EagerOperand::array(&f32),
                EagerOperand::array(&f64)
            ])
            .is_err());
        assert!(EagerKernel::SumAxis(1)
            .evaluate(&[EagerOperand::array(&f64)])
            .is_err());
    }
}
