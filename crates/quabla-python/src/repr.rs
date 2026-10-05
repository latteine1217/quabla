//! Value-showing `repr` of eager tensors, modelled on NumPy's default print
//! options so that output reads like `numpy.array` output:
//!
//! - floats print positionally with at most 8 fractional digits (fewer when
//!   the shortest round-trip spelling of the value needs fewer), with the
//!   decimal points of all printed elements aligned; they switch to
//!   scientific notation when the largest finite magnitude is at least 1e8,
//!   the smallest nonzero one is below 1e-4, or their ratio exceeds 1e3;
//! - a tensor with more than 1000 elements is summarized: every axis longer
//!   than 6 prints its first and last 3 entries around `...`;
//! - rows of the innermost axis wrap at 75 columns;
//! - the dtype is always printed, as in JAX, so the dtype of a value is never
//!   implied by a default.
//!
//! Only the printed elements are read, so printing a large tensor costs
//! O(printed elements), not O(size).

use quabla_core::tensor_ir::TensorDType;

use crate::dtype::PyDType;

/// NumPy's `threshold`: larger tensors are summarized.
const THRESHOLD: usize = 1000;
/// NumPy's `edgeitems`: entries printed at each end of a summarized axis.
const EDGE_ITEMS: usize = 3;
/// NumPy's `precision`: the most fractional digits printed.
const PRECISION: usize = 8;
/// NumPy's `linewidth`.
const LINE_WIDTH: usize = 75;
const PREFIX: &str = "Tensor(";

/// Renders `Tensor(<values>, dtype=<name>)` for a C-ordered tensor whose
/// flat element `i` is `get(i)` (bool elements as 0.0 or 1.0).
pub(crate) fn tensor_repr(
    shape: &[usize],
    dtype: TensorDType,
    get: impl Fn(usize) -> f64,
) -> String {
    let name = PyDType::from(dtype).name();
    let summarize = shape.iter().product::<usize>() > THRESHOLD;
    let mut flat = Vec::new();
    collect_printed(shape, summarize, 0, 0, &mut flat);
    let values = flat.into_iter().map(get).collect::<Vec<_>>();
    let mut items = format_elements(&values, dtype).into_iter();
    let body = render(shape, summarize, 0, PREFIX.len(), &mut items);
    let last_line = match body.rsplit_once('\n') {
        Some((_, line)) => line.len(),
        None => PREFIX.len() + body.len(),
    };
    let suffix = format!("dtype={name})");
    // NumPy moves the dtype to its own line when it does not fit.
    if last_line + 2 + suffix.len() > LINE_WIDTH {
        format!("{PREFIX}{body},\n{}{suffix}", " ".repeat(PREFIX.len()))
    } else {
        format!("{PREFIX}{body}, {suffix}")
    }
}

/// The printed positions along an axis of `len` entries, with `None` for the
/// `...` of a summarized axis.
fn axis_positions(len: usize, summarize: bool) -> Vec<Option<usize>> {
    if summarize && len > 2 * EDGE_ITEMS {
        (0..EDGE_ITEMS)
            .map(Some)
            .chain([None])
            .chain((len - EDGE_ITEMS..len).map(Some))
            .collect()
    } else {
        (0..len).map(Some).collect()
    }
}

/// Appends the flat index of every printed element of the sub-tensor at
/// `axis` starting at flat index `base`, in printing order.
fn collect_printed(
    shape: &[usize],
    summarize: bool,
    axis: usize,
    base: usize,
    out: &mut Vec<usize>,
) {
    if axis == shape.len() {
        out.push(base);
        return;
    }
    let stride = shape[axis + 1..].iter().product::<usize>();
    for position in axis_positions(shape[axis], summarize).into_iter().flatten() {
        collect_printed(shape, summarize, axis + 1, base + position * stride, out);
    }
}

/// Renders the sub-tensor at `axis` whose opening bracket sits at column
/// `column`, taking its formatted elements from `items` in printing order.
fn render(
    shape: &[usize],
    summarize: bool,
    axis: usize,
    column: usize,
    items: &mut impl Iterator<Item = String>,
) -> String {
    if axis == shape.len() {
        return items
            .next()
            .expect("one formatted item per printed element");
    }
    let positions = axis_positions(shape[axis], summarize);
    if axis + 1 == shape.len() {
        let words = positions
            .iter()
            .map(|position| match position {
                Some(_) => items
                    .next()
                    .expect("one formatted item per printed element"),
                None => "...".to_string(),
            })
            .collect::<Vec<_>>();
        return wrap_row(&words, column);
    }
    // One newline between rows, plus one blank line per further axis, as
    // in NumPy.
    let separator = format!(
        ",{}{}",
        "\n".repeat(shape.len() - axis - 1),
        " ".repeat(column + 1)
    );
    let children = positions
        .iter()
        .map(|position| match position {
            Some(_) => render(shape, summarize, axis + 1, column + 1, items),
            None => "...".to_string(),
        })
        .collect::<Vec<_>>();
    format!("[{}]", children.join(&separator))
}

/// `[a, b, ...]`, wrapped so no line passes `LINE_WIDTH`; continuation lines
/// align with the first element.
fn wrap_row(words: &[String], column: usize) -> String {
    let mut out = String::from("[");
    let mut width = column + 1;
    for (index, word) in words.iter().enumerate() {
        if index > 0 {
            out.push(',');
            width += 1;
            // Leave room for the closing bracket and the comma after it.
            if width + 1 + word.len() + 2 > LINE_WIDTH {
                out.push('\n');
                out.push_str(&" ".repeat(column + 1));
                width = column + 1;
            } else {
                out.push(' ');
                width += 1;
            }
        }
        out.push_str(word);
        width += word.len();
    }
    out.push(']');
    out
}

/// Formats every printed element to a common width.
fn format_elements(values: &[f64], dtype: TensorDType) -> Vec<String> {
    if dtype == TensorDType::Bool {
        // NumPy pads bools to the width of `False`.
        return values
            .iter()
            .map(|value| format!("{:>5}", if *value != 0.0 { "True" } else { "False" }))
            .collect();
    }
    let single = dtype == TensorDType::F32;
    let magnitudes = values
        .iter()
        .filter(|value| value.is_finite() && **value != 0.0)
        .map(|value| value.abs());
    let (min, max) = magnitudes.fold((f64::INFINITY, 0.0f64), |(min, max), value| {
        (min.min(value), max.max(value))
    });
    let scientific = max > 0.0 && (max >= 1e8 || min < 1e-4 || max / min > 1e3);
    let parts = values
        .iter()
        .map(|value| {
            value.is_finite().then(|| {
                if scientific {
                    scientific_parts(*value, single)
                } else {
                    positional_parts(*value, single)
                }
            })
        })
        .collect::<Vec<_>>();
    let finite = parts.iter().flatten();
    let integer_width = finite.clone().map(|part| part.0.len()).max().unwrap_or(0);
    let fraction_width = finite.clone().map(|part| part.1.len()).max().unwrap_or(0);
    let exponent_width = finite.map(|part| part.2.len()).max().unwrap_or(0);
    let padded = values
        .iter()
        .zip(&parts)
        .map(|(value, part)| match part {
            Some((integer, fraction, exponent)) => {
                let fraction = if scientific {
                    // NumPy zero-pads mantissas (`1.0e-05`) and exponents.
                    let exponent = format!(
                        "{}{:0>width$}",
                        &exponent[..2],
                        &exponent[2..],
                        width = exponent_width - 2
                    );
                    format!("{fraction:0<fraction_width$}{exponent}")
                } else {
                    format!("{fraction:<fraction_width$}")
                };
                format!("{integer:>integer_width$}.{fraction}")
            }
            None => non_finite(*value).to_string(),
        })
        .collect::<Vec<_>>();
    let width = padded.iter().map(String::len).max().unwrap_or(0);
    padded
        .into_iter()
        .map(|item| format!("{item:>width$}"))
        .collect()
}

fn non_finite(value: f64) -> &'static str {
    if value.is_nan() {
        "nan"
    } else if value > 0.0 {
        "inf"
    } else {
        "-inf"
    }
}

/// `(integer part with sign, fractional digits, "")` of a finite value:
/// its shortest round-trip spelling for its dtype when that has at most
/// `PRECISION` fractional digits, otherwise rounded to `PRECISION` digits
/// with trailing zeros removed.
fn positional_parts(value: f64, single: bool) -> (String, String, String) {
    let shortest = if single {
        format!("{}", value as f32)
    } else {
        format!("{value}")
    };
    let text = match shortest.split_once('.') {
        Some((_, fraction)) if fraction.len() > PRECISION => format!("{value:.PRECISION$}")
            .trim_end_matches('0')
            .to_string(),
        _ => shortest,
    };
    let (integer, fraction) = text.split_once('.').unwrap_or((&text, ""));
    (integer.to_string(), fraction.to_string(), String::new())
}

/// `(mantissa integer part with sign, mantissa fractional digits, "e±NN")`
/// of a finite value, with the same digit rule as `positional_parts`.
fn scientific_parts(value: f64, single: bool) -> (String, String, String) {
    let shortest = if single {
        format!("{:e}", value as f32)
    } else {
        format!("{value:e}")
    };
    let (mantissa, _) = shortest.split_once('e').expect("`{:e}` has an exponent");
    let text = match mantissa.split_once('.') {
        Some((_, fraction)) if fraction.len() > PRECISION => format!("{value:.PRECISION$e}"),
        _ => shortest,
    };
    let (mantissa, exponent) = text.split_once('e').expect("`{:e}` has an exponent");
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let exponent = exponent
        .parse::<i32>()
        .expect("`{:e}` exponent is an integer");
    let sign = if exponent < 0 { '-' } else { '+' };
    (
        integer.to_string(),
        fraction.trim_end_matches('0').to_string(),
        format!("e{sign}{:02}", exponent.unsigned_abs()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repr(shape: &[usize], dtype: TensorDType, values: &[f64]) -> String {
        tensor_repr(shape, dtype, |index| values[index])
    }

    #[test]
    fn prints_values_with_numpy_float_rules() {
        assert_eq!(
            repr(&[2], TensorDType::F32, &[1.0, 2.0]),
            "Tensor([1., 2.], dtype=float32)"
        );
        assert_eq!(
            repr(&[2], TensorDType::F64, &[1.5, 100.25]),
            "Tensor([  1.5 , 100.25], dtype=float64)"
        );
        // f32 values print with their own shortest spelling, not the f64 one.
        assert_eq!(
            repr(
                &[2],
                TensorDType::F32,
                &[f64::from(0.1f32), f64::from(0.3f32)]
            ),
            "Tensor([0.1, 0.3], dtype=float32)"
        );
        assert_eq!(
            repr(&[1], TensorDType::F64, &[1.0 / 3.0]),
            "Tensor([0.33333333], dtype=float64)"
        );
        assert_eq!(
            repr(&[3], TensorDType::F64, &[-0.0, f64::NAN, f64::NEG_INFINITY]),
            "Tensor([ -0.,  nan, -inf], dtype=float64)"
        );
        assert_eq!(
            repr(&[], TensorDType::F64, &[2.5]),
            "Tensor(2.5, dtype=float64)"
        );
    }

    #[test]
    fn switches_to_scientific_notation_like_numpy() {
        assert_eq!(
            repr(&[2], TensorDType::F64, &[1e-5, 2.5e-7]),
            "Tensor([1.0e-05, 2.5e-07], dtype=float64)"
        );
        assert_eq!(
            repr(&[2], TensorDType::F64, &[1.0, 1e4]),
            "Tensor([1.e+00, 1.e+04], dtype=float64)"
        );
        assert_eq!(
            repr(&[2], TensorDType::F64, &[1e-5, 1e100]),
            "Tensor([1.e-005, 1.e+100], dtype=float64)"
        );
        // A ratio of exactly 1e3 stays positional.
        assert_eq!(
            repr(&[2], TensorDType::F64, &[1.0, 1000.0]),
            "Tensor([   1., 1000.], dtype=float64)"
        );
    }

    #[test]
    fn prints_bools_padded_to_a_common_width() {
        assert_eq!(
            repr(&[2], TensorDType::Bool, &[1.0, 0.0]),
            "Tensor([ True, False], dtype=bool)"
        );
        assert_eq!(
            repr(&[2], TensorDType::Bool, &[1.0, 1.0]),
            "Tensor([ True,  True], dtype=bool)"
        );
    }

    #[test]
    fn nests_rows_and_blocks() {
        let values = (0..8).map(f64::from).collect::<Vec<_>>();
        assert_eq!(
            repr(&[2, 2], TensorDType::F64, &values[..4]),
            "Tensor([[0., 1.],\n        [2., 3.]], dtype=float64)"
        );
        assert_eq!(
            repr(&[2, 2, 2], TensorDType::F64, &values),
            "Tensor([[[0., 1.],\n         [2., 3.]],\n\n        [[4., 5.],\n         [6., 7.]]], dtype=float64)"
        );
    }

    #[test]
    fn summarizes_large_tensors_and_reads_only_printed_elements() {
        let read = std::cell::RefCell::new(Vec::new());
        let text = tensor_repr(&[1001], TensorDType::F64, |index| {
            read.borrow_mut().push(index);
            index as f64
        });
        assert_eq!(
            text,
            "Tensor([   0.,    1.,    2., ...,  998.,  999., 1000.], dtype=float64)"
        );
        assert_eq!(*read.borrow(), [0, 1, 2, 998, 999, 1000]);

        let text = tensor_repr(&[40, 40], TensorDType::F64, |_| 0.0);
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 7);
        assert_eq!(lines[0], "Tensor([[0., 0., 0., ..., 0., 0., 0.],");
        assert_eq!(lines[3], "        ...,");
    }

    #[test]
    fn wraps_long_rows_at_the_line_width() {
        let values = (0..32)
            .map(|index| f64::from(index) + 0.5)
            .collect::<Vec<_>>();
        let text = repr(&[32], TensorDType::F64, &values);
        assert!(text.lines().all(|line| line.len() <= LINE_WIDTH), "{text}");
        let continuation = text.lines().nth(1).expect("the row wraps");
        assert!(continuation.starts_with("        "), "{text}");
        assert!(text.ends_with(",\n       dtype=float64)"), "{text}");
        assert_eq!(
            text.replace(['\n', ' '], ""),
            format!(
                "Tensor([{}],dtype=float64)",
                values
                    .iter()
                    .map(|value| format!("{value:.1}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        );
    }
}
