//! The `ExtremumAxis` op: the maximum or minimum along one axis, its CPU
//! reference, and its derivative rules.
//!
//! Values follow IEEE 754-2019 `maximum`/`minimum`: a NaN anywhere in a
//! reduced line gives NaN, and the sign of zero is ordered (`-0 < +0`), so
//! `max` of `-0.0` and `+0.0` is `+0.0` and `min` is `-0.0`. NumPy agrees on
//! NaN everywhere and on zeros in its arm64 builds; its x86 builds return
//! whichever zero comes last. Both rules make the result independent of the
//! order in which a backend visits the line, so a device tree reduction
//! returns the CPU value bit for bit.
//!
//! Derivatives follow JAX's chooser rule. With `eq` the 0/1 mask of the
//! entries equal to their line's extremum and `count` its sum over the
//! line, the JVP is `sum(t * eq) / count` and the VJP is `g * eq / count`,
//! so tied entries share the derivative equally. Both are written with IR
//! ops (`compare`, `sum_axis`, `div`), so every derivative order is again an
//! ordinary graph on every backend; the mask is piecewise constant and has a
//! zero derivative. A NaN extremum equals no entry, so `count` is zero and
//! the derivative is NaN (`0 / 0`) rather than an error.

use super::{
    normalize_axis, reduced_shape, DynamicTensor, TensorComparison, TensorIr, TensorNodeId,
    TensorOp,
};

/// Which extremum [`TensorOp::ExtremumAxis`] selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorExtremum {
    Max,
    Min,
}

impl TensorExtremum {
    pub fn name(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Min => "min",
        }
    }

    /// Whether `candidate` replaces the running extremum `current`: a NaN is
    /// kept once seen and taken when it arrives, a strictly larger (smaller)
    /// value is taken, and between equal zeros the `+0` (`-0`) is taken.
    /// Equal nonzero values are bitwise identical, so the result of folding
    /// a line with this rule does not depend on the visiting order.
    pub fn replaces(self, current: f64, candidate: f64) -> bool {
        if current.is_nan() {
            return false;
        }
        if candidate.is_nan() {
            return true;
        }
        match self {
            Self::Max => {
                candidate > current
                    || (candidate == current
                        && current.is_sign_negative()
                        && candidate.is_sign_positive())
            }
            Self::Min => {
                candidate < current
                    || (candidate == current
                        && current.is_sign_positive()
                        && candidate.is_sign_negative())
            }
        }
    }

    /// The CUDA expression of [`Self::replaces`] folding the identifier
    /// `candidate` into the identifier `current`. A NaN `current` is never
    /// replaced by a number (every comparison with it is false); replacing
    /// it by another NaN keeps the result NaN.
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn cuda_fold(self, current: &str, candidate: &str) -> String {
        let (order, zero) = match self {
            Self::Max => ('>', format!("signbit({current}) && !signbit({candidate})")),
            Self::Min => ('<', format!("!signbit({current}) && signbit({candidate})")),
        };
        format!(
            "{current} = (isnan({candidate}) || {candidate} {order} {current} || \
             ({candidate} == {current} && {zero})) ? {candidate} : {current};"
        )
    }
}

impl TensorIr {
    /// The maximum or minimum of `input` along `axis`, which is removed; see
    /// the module documentation for the NaN, signed-zero, and derivative
    /// rules. The input must be floating point (`push_derived` rejects
    /// `Bool`) and the axis non-empty.
    pub fn extremum_axis(
        &mut self,
        input: TensorNodeId,
        axis: isize,
        kind: TensorExtremum,
    ) -> Result<TensorNodeId, String> {
        let input_shape = self.node(input)?.shape.clone();
        let axis = normalize_axis(axis, input_shape.len())?;
        if input_shape[axis] == 0 {
            return Err("max/min reduction requires a non-empty reduced axis".to_string());
        }
        self.push_derived(
            TensorOp::ExtremumAxis { input, axis, kind },
            reduced_shape(&input_shape, axis)?,
        )
    }

    /// `(eq, count)`: the mask, in the dtype of `input`, of the entries of
    /// `input` equal to their line's extremum `value` (the reduced shape),
    /// and the number of such entries per line.
    fn extremum_axis_indicators(
        &mut self,
        input: TensorNodeId,
        value: TensorNodeId,
        axis: usize,
    ) -> Result<(TensorNodeId, TensorNodeId, Vec<usize>), String> {
        let source = self.node(input)?;
        let (dtype, mut expanded_shape) = (source.dtype, source.shape.clone());
        expanded_shape[axis] = 1;
        let expanded = self.reshape(value, expanded_shape.clone())?;
        let equal = self.compare(input, expanded, TensorComparison::Equal)?;
        let mask = self.cast(equal, dtype)?;
        let count = self.sum_axis(mask, axis as isize)?;
        Ok((mask, count, expanded_shape))
    }

    /// The tangent of `value = extremum_axis(input, axis)` for the input
    /// tangent `tangent`: `sum(tangent * eq) / count` over the axis.
    pub(super) fn extremum_axis_jvp(
        &mut self,
        input: TensorNodeId,
        value: TensorNodeId,
        tangent: TensorNodeId,
        axis: usize,
    ) -> Result<TensorNodeId, String> {
        let (mask, count, _) = self.extremum_axis_indicators(input, value, axis)?;
        let selected = self.mul(tangent, mask)?;
        let total = self.sum_axis(selected, axis as isize)?;
        self.div(total, count)
    }

    /// The input cotangent of `value = extremum_axis(input, axis)` for the
    /// output cotangent `cotangent`: `cotangent * eq / count`, broadcast
    /// back along the axis.
    pub(super) fn extremum_axis_vjp(
        &mut self,
        input: TensorNodeId,
        value: TensorNodeId,
        cotangent: TensorNodeId,
        axis: usize,
    ) -> Result<TensorNodeId, String> {
        let (mask, count, expanded_shape) = self.extremum_axis_indicators(input, value, axis)?;
        let cotangent = self.reshape(cotangent, expanded_shape.clone())?;
        let count = self.reshape(count, expanded_shape)?;
        let selected = self.mul(mask, cotangent)?;
        self.div(selected, count)
    }
}

impl DynamicTensor {
    /// The CPU reference of `TensorOp::ExtremumAxis`: each line is folded
    /// from its first entry with [`TensorExtremum::replaces`].
    pub(super) fn reduce_extremum_axis(
        &self,
        axis: usize,
        kind: TensorExtremum,
    ) -> Result<Self, String> {
        let output_shape = reduced_shape(&self.shape, axis)?;
        let extent = self.shape[axis];
        if extent == 0 {
            return Err("max/min reduction requires a non-empty reduced axis".to_string());
        }
        let inner = self.shape[axis + 1..].iter().product::<usize>();
        let outer = self.shape[..axis].iter().product::<usize>();
        let mut data = Vec::with_capacity(outer * inner);
        for block in 0..outer {
            let start = block * extent * inner;
            for column in 0..inner {
                let mut current = self.data.get(start + column);
                for row in 1..extent {
                    let candidate = self.data.get(start + row * inner + column);
                    if kind.replaces(current, candidate) {
                        current = candidate;
                    }
                }
                data.push(current);
            }
        }
        Self::new(output_shape, data)
    }

    /// The runtime form of [`TensorIr::extremum_axis_indicators`] for the
    /// input `self` and its reduced extremum `value`.
    fn extremum_axis_indicators(&self, value: &Self, axis: usize) -> Result<(Self, Self), String> {
        let expanded = value.expand_reduced_axis(&self.shape, axis)?;
        let mask = self.compare(&expanded, TensorComparison::Equal)?;
        let count = mask.reduce_axis(axis, 1.0)?;
        Ok((mask, count))
    }

    /// The runtime form of [`TensorIr::extremum_axis_jvp`] for the input
    /// `self`.
    pub(super) fn extremum_axis_tangent(
        &self,
        value: &Self,
        tangent: &Self,
        axis: usize,
    ) -> Result<Self, String> {
        let (mask, count) = self.extremum_axis_indicators(value, axis)?;
        tangent.mul(&mask)?.reduce_axis(axis, 1.0)?.div(&count)
    }

    /// The runtime form of [`TensorIr::extremum_axis_vjp`] for the input
    /// `self`.
    pub(super) fn extremum_axis_cotangent(
        &self,
        value: &Self,
        cotangent: &Self,
        axis: usize,
    ) -> Result<Self, String> {
        let (mask, count) = self.extremum_axis_indicators(value, axis)?;
        mask.mul(&cotangent.expand_reduced_axis(&self.shape, axis)?)?
            .div(&count.expand_reduced_axis(&self.shape, axis)?)
    }
}
