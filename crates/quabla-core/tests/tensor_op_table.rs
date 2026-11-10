//! One row per `TensorOp` variant (and per region kind): the program text,
//! the compiled CPU plan, its kernel op names and fusion regions, the value,
//! and the symbolic VJP and JVP of each row are compared with a recorded
//! snapshot. The rows go through the public `TensorIr` interface only, so the
//! snapshot pins what callers observe of each op: its name, its operands, how
//! plan compilation remaps and deduplicates it, and its derivative graphs.
//!
//! Every row builds its op twice and adds the copies, so common-subexpression
//! elimination has a duplicate to merge.
//!
//! Regenerate the snapshot after an intended change with
//! `QUABLA_BLESS=1 cargo test -p quabla-core --test tensor_op_table` and
//! review the diff.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::Arc;

use quabla_core::tensor_ir::{
    BinaryMathKind, DynamicTensor, LinalgKind, TensorComparison, TensorCondExecutionPlan,
    TensorCustomRule, TensorDType, TensorExtremum, TensorForiExecutionPlan, TensorIr, TensorNodeId,
    TensorScanExecutionPlan, TensorWhileExecutionPlan, UnaryMathKind,
};

const SNAPSHOT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/snapshots/tensor_op_table.txt"
);

/// A program under construction together with the values of its inputs.
struct Row {
    graph: TensorIr,
    inputs: BTreeMap<String, DynamicTensor>,
}

type Built = Result<(Row, TensorNodeId), String>;

impl Row {
    fn new() -> Self {
        Self {
            graph: TensorIr::new(),
            inputs: BTreeMap::new(),
        }
    }

    /// An input whose elements are distinct, non-integer, and of both signs,
    /// so no comparison ties and no derivative vanishes by accident.
    fn input(&mut self, name: &str, shape: Vec<usize>) -> Result<TensorNodeId, String> {
        let count = shape.iter().product::<usize>();
        let data = (0..count)
            .map(|index| ((index * 7 + 3) % 11) as f64 / 4.0 - 1.15)
            .collect();
        self.input_with(name, shape, data)
    }

    fn input_with(
        &mut self,
        name: &str,
        shape: Vec<usize>,
        data: Vec<f64>,
    ) -> Result<TensorNodeId, String> {
        let id = self.graph.input(name, shape.clone())?;
        self.inputs
            .insert(name.to_string(), DynamicTensor::new(shape, data)?);
        Ok(id)
    }

    /// A symmetric positive definite 3x3 matrix.
    fn spd(&mut self, name: &str) -> Result<TensorNodeId, String> {
        self.input_with(
            name,
            vec![3, 3],
            vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0],
        )
    }

    /// `op(...) + op(...)` from two identical builds of one op.
    fn twice(mut self, mut op: impl FnMut(&mut TensorIr) -> Result<TensorNodeId, String>) -> Built {
        let first = op(&mut self.graph)?;
        let second = op(&mut self.graph)?;
        let output = self.graph.add(first, second)?;
        Ok((self, output))
    }
}

fn unary(
    shape: Vec<usize>,
    op: impl Fn(&mut TensorIr, TensorNodeId) -> Result<TensorNodeId, String>,
) -> Built {
    let mut row = Row::new();
    let x = row.input("x", shape)?;
    row.twice(|graph| op(graph, x))
}

/// A row whose inputs `setup` declares and whose op it returns.
fn built<F>(setup: impl FnOnce(&mut Row) -> Result<F, String>) -> Built
where
    F: FnMut(&mut TensorIr) -> Result<TensorNodeId, String>,
{
    let mut row = Row::new();
    let op = setup(&mut row)?;
    row.twice(op)
}

fn binary(
    shape: Vec<usize>,
    op: impl Fn(&mut TensorIr, TensorNodeId, TensorNodeId) -> Result<TensorNodeId, String>,
) -> Built {
    let mut row = Row::new();
    let x = row.input("x", shape.clone())?;
    let y = row.input("y", shape)?;
    row.twice(|graph| op(graph, x, y))
}

fn scale_body() -> Result<TensorIr, String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![2])?;
    body.input("index", vec![])?;
    let scale = body.input("scale", vec![2])?;
    let product = body.mul(carry, scale)?;
    let wave = body.tanh(product)?;
    body.add(wave, carry)?;
    Ok(body)
}

fn fori_row() -> Built {
    let body = scale_body()?;
    let output = body.node_count() - 1;
    let plan = TensorForiExecutionPlan::new(0, 3, body.compile_region(output)?, "carry", "index")?;
    let mut row = Row::new();
    let c = row.input("x", vec![2])?;
    let s = row.input("s", vec![2])?;
    row.twice(|graph| graph.fori(c, plan.clone(), vec![("scale".to_string(), s)]))
}

fn scan_row() -> Built {
    let body = scale_body()?;
    let output = body.node_count() - 1;
    let (plan, _) = body.compile_cpu_many(&[output, output])?;
    let plan = TensorScanExecutionPlan::new(0, 3, plan, "carry", "index")?;
    let mut row = Row::new();
    let c = row.input("x", vec![2])?;
    let s = row.input("s", vec![2])?;
    row.twice(|graph| {
        let (_, stacked) = graph.scan(c, plan.clone(), vec![("scale".to_string(), s)])?;
        graph.sum(stacked)
    })
}

fn cond_row() -> Built {
    let mut on_true = TensorIr::new();
    let a = on_true.input("a", vec![2])?;
    let b = on_true.input("b", vec![2])?;
    let product = on_true.mul(a, b)?;
    let mut on_false = TensorIr::new();
    let a = on_false.input("a", vec![2])?;
    let b = on_false.input("b", vec![2])?;
    let difference = on_false.sub(a, b)?;
    let branches = TensorCondExecutionPlan::new(
        on_true.compile_cpu(product)?,
        on_false.compile_cpu(difference)?,
    )?;
    let mut row = Row::new();
    let x = row.input("x", vec![2])?;
    let y = row.input("y", vec![2])?;
    let predicate = row.graph.scalar_constant(1.0);
    row.twice(|graph| {
        graph.cond_with_captures(
            predicate,
            branches.clone(),
            vec![("a".to_string(), x), ("b".to_string(), y)],
        )
    })
}

fn while_row() -> Built {
    let mut predicate = TensorIr::new();
    let carry = predicate.input("carry", vec![2])?;
    let squared = predicate.mul(carry, carry)?;
    let norm = predicate.sum(squared)?;
    let limit = predicate.scalar_constant(0.1);
    let running = predicate.compare(norm, limit, TensorComparison::Greater)?;
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![2])?;
    let half = body.scalar_constant(0.5);
    let next = body.mul(carry, half)?;
    let plan = TensorWhileExecutionPlan::new(
        predicate.compile_cpu(running)?,
        body.compile_cpu(next)?,
        "carry",
    )?;
    let mut row = Row::new();
    let x = row.input("x", vec![2])?;
    row.twice(|graph| graph.while_loop(x, plan.clone(), vec![]))
}

/// A `custom_vjp`-style rule for `x * x` whose backward graph is `3 * g`.
fn custom_row() -> Built {
    let mut forward = TensorIr::new();
    let x = forward.input("x", vec![2])?;
    let square = forward.mul(x, x)?;
    let mut backward = TensorIr::new();
    backward.input("residual", vec![2])?;
    let g = backward.input("g", vec![2])?;
    let three = backward.scalar_constant(3.0);
    let scaled = backward.mul(g, three)?;
    let rule = Arc::new(TensorCustomRule::new(
        "custom_square".to_string(),
        forward,
        vec!["x".to_string()],
        1,
        vec![square, x],
        backward,
        vec!["residual".to_string()],
        vec!["g".to_string()],
        vec![Some(scaled)],
        None,
        false,
    )?);
    let mut row = Row::new();
    let x = row.input("x", vec![2])?;
    row.twice(|graph| {
        let value = graph.mul(x, x)?;
        Ok(graph.custom(rule.clone(), &[value], &[x])?[0])
    })
}

/// The rows, by the op (or region kind) each one exercises. Derivative-only
/// ops (`SqrtDerivative`, `PadSlice`, `CholeskyAd`, and the derivative region
/// kinds) appear in the VJP and JVP sections of the rows of their primal op.
fn rows() -> Vec<(&'static str, Built)> {
    vec![
        (
            "input+scalar_constant",
            unary(vec![2, 3], |graph, x| {
                let c = graph.scalar_constant(0.75);
                graph.mul(x, c)
            }),
        ),
        (
            "constant",
            unary(vec![2], |graph, x| {
                let c = graph.constant(DynamicTensor::new(vec![2], vec![0.5, -1.25])?, false);
                graph.mul(x, c)
            }),
        ),
        (
            "cast",
            unary(vec![2, 2], |graph, x| graph.cast(x, TensorDType::F32)),
        ),
        ("add", binary(vec![3], |graph, x, y| graph.add(x, y))),
        ("sub", binary(vec![3], |graph, x, y| graph.sub(x, y))),
        ("mul", binary(vec![3], |graph, x, y| graph.mul(x, y))),
        ("div", binary(vec![3], |graph, x, y| graph.div(x, y))),
        (
            "greater",
            binary(vec![3], |graph, x, y| {
                let mask = graph.greater(x, y)?;
                graph.mul(mask, x)
            }),
        ),
        (
            "compare+where",
            binary(vec![3], |graph, x, y| {
                let mask = graph.compare(x, y, TensorComparison::LessEqual)?;
                graph.where_select(mask, x, y)
            }),
        ),
        ("sum", unary(vec![2, 3], |graph, x| graph.sum(x))),
        (
            "sum_axis",
            unary(vec![2, 3], |graph, x| graph.sum_axis(x, 1)),
        ),
        ("mean", unary(vec![2, 3], |graph, x| graph.mean(x))),
        (
            "mean_axis",
            unary(vec![2, 3], |graph, x| graph.mean_axis(x, 0)),
        ),
        (
            "extremum_axis",
            unary(vec![2, 3], |graph, x| {
                graph.extremum_axis(x, -1, TensorExtremum::Max)
            }),
        ),
        (
            "matmul",
            built(|row| {
                let x = row.input("x", vec![2, 3])?;
                let y = row.input("y", vec![3, 2])?;
                Ok(move |graph: &mut TensorIr| graph.matmul(x, y))
            }),
        ),
        (
            "solve",
            built(|row| {
                let a = row.spd("x")?;
                let b = row.input("y", vec![3, 2])?;
                Ok(move |graph: &mut TensorIr| graph.solve(a, b))
            }),
        ),
        (
            "cholesky",
            built(|row| {
                let a = row.spd("x")?;
                Ok(move |graph: &mut TensorIr| graph.cholesky(a))
            }),
        ),
        (
            "triangular",
            unary(vec![3, 3], |graph, x| graph.triangular(x, true)),
        ),
        (
            "linalg:eigh_values",
            built(|row| {
                let a = row.spd("x")?;
                Ok(move |graph: &mut TensorIr| graph.linalg(a, LinalgKind::EighValues))
            }),
        ),
        (
            "linalg:svd_s",
            unary(vec![3, 2], |graph, x| graph.linalg(x, LinalgKind::SvdS)),
        ),
        (
            "sqrt",
            built(|row| {
                let x = row.input_with("x", vec![3], vec![0.25, 1.5, 4.0])?;
                Ok(move |graph: &mut TensorIr| graph.sqrt(x))
            }),
        ),
        (
            "reshape",
            unary(vec![2, 3], |graph, x| graph.reshape(x, vec![3, 2])),
        ),
        ("powi", unary(vec![3], |graph, x| graph.powi(x, 3))),
        (
            "transpose",
            unary(vec![2, 3], |graph, x| graph.transpose(x, None)),
        ),
        (
            "unary_math:tanh",
            unary(vec![3], |graph, x| graph.unary_math(x, UnaryMathKind::Tanh)),
        ),
        (
            "unary_math:erf",
            unary(vec![3], |graph, x| graph.unary_math(x, UnaryMathKind::Erf)),
        ),
        (
            "binary_math:pow",
            built(|row| {
                let x = row.input_with("x", vec![3], vec![0.5, 1.25, 2.0])?;
                let y = row.input("y", vec![3])?;
                Ok(move |graph: &mut TensorIr| graph.binary_math(x, y, BinaryMathKind::Pow))
            }),
        ),
        (
            "binary_math:atan2",
            binary(vec![3], |graph, x, y| graph.atan2(x, y)),
        ),
        (
            "stop_gradient",
            unary(vec![3], |graph, x| {
                let stopped = graph.stop_gradient(x)?;
                graph.mul(stopped, x)
            }),
        ),
        (
            "cumsum",
            unary(vec![2, 3], |graph, x| graph.cumsum(x, 1, false)),
        ),
        (
            "concat",
            binary(vec![2], |graph, x, y| graph.concat(vec![x, y, x], 0)),
        ),
        (
            "slice_axis",
            unary(vec![2, 4], |graph, x| graph.slice_axis(x, 1, 1, 3)),
        ),
        (
            "gather",
            unary(vec![4, 2], |graph, x| {
                graph.gather(x, Arc::from(vec![2, 0, 2]), 0)
            }),
        ),
        (
            "scatter_add",
            built(|row| {
                let base = row.input("x", vec![4, 2])?;
                let updates = row.input("y", vec![3, 2])?;
                Ok(move |graph: &mut TensorIr| {
                    graph.scatter_add(base, updates, Arc::from(vec![1, 3, 1]), 0)
                })
            }),
        ),
        (
            "broadcast",
            unary(vec![1, 3], |graph, x| graph.broadcast_to(x, vec![2, 3])),
        ),
        ("region:fori", fori_row()),
        ("region:scan", scan_row()),
        ("region:cond", cond_row()),
        ("region:while", while_row()),
        ("custom", custom_row()),
    ]
}

fn values(tensor: &DynamicTensor) -> String {
    format!("{:?} {:?}", tensor.shape(), tensor.data().as_ref())
}

/// The cotangent of `output`: distinct non-zero entries of its shape.
fn seed(graph: &TensorIr, output: TensorNodeId) -> Result<DynamicTensor, String> {
    let shape = graph.node_shape(output)?;
    let count = shape.iter().product::<usize>();
    DynamicTensor::new(
        shape,
        (0..count).map(|index| 0.5 + index as f64 * 0.25).collect(),
    )
}

/// The compiled plan of `outputs`, or the error that compilation reports.
fn plan_section(out: &mut String, graph: &TensorIr, outputs: &[TensorNodeId]) {
    match graph.compile_cpu_many(outputs) {
        Ok((plan, _)) => {
            out.push_str(&plan.lower_text());
            out.push('\n');
            let ops: Vec<_> = plan
                .kernel_ir()
                .nodes
                .into_iter()
                .map(|node| node.op)
                .collect();
            let _ = writeln!(out, "kernel ops: {ops:?}");
            let _ = writeln!(out, "fusion: {:?}", plan.fusion_regions());
        }
        Err(error) => {
            let _ = writeln!(out, "error: {error}");
        }
    }
}

fn render_row(out: &mut String, name: &str, built: Built) {
    let _ = writeln!(out, "=== {name}");
    let (row, output) = match built {
        Ok(built) => built,
        Err(error) => {
            let _ = writeln!(out, "build error: {error}");
            return;
        }
    };
    let Row { graph, inputs } = row;
    let _ = writeln!(out, "--- source\n{}", graph.lower_text());
    out.push_str("--- plan\n");
    plan_section(out, &graph, &[output]);
    match graph.evaluate(output, &inputs) {
        Ok(value) => {
            let _ = writeln!(out, "value: {}", values(&value));
        }
        Err(error) => {
            let _ = writeln!(out, "value error: {error}");
        }
    }

    out.push_str("--- vjp\n");
    let vjp = seed(&graph, output).and_then(|ct| {
        let vjp = graph.symbolic_vjp(output, "ct")?;
        let mut vjp_inputs = inputs.clone();
        vjp_inputs.insert("ct".to_string(), ct);
        Ok((vjp, vjp_inputs))
    });
    match vjp {
        Ok((vjp, vjp_inputs)) => {
            let gradients: Vec<_> = vjp.gradients.values().copied().collect();
            plan_section(out, &vjp.graph, &gradients);
            for (input, gradient) in &vjp.gradients {
                match vjp.graph.evaluate(*gradient, &vjp_inputs) {
                    Ok(value) => {
                        let _ = writeln!(out, "d/d{input}: {}", values(&value));
                    }
                    Err(error) => {
                        let _ = writeln!(out, "d/d{input} error: {error}");
                    }
                }
            }
            // Forward over reverse: the derivative regions of loops differentiated twice.
            if let Some(&gradient) = vjp.gradients.get("x") {
                out.push_str("--- jvp of vjp\n");
                match vjp.graph.symbolic_jvp(gradient, "x") {
                    Ok(jvp) => plan_section(out, &jvp.graph, &[jvp.tangent]),
                    Err(error) => {
                        let _ = writeln!(out, "error: {error}");
                    }
                }
            }
        }
        Err(error) => {
            let _ = writeln!(out, "error: {error}");
        }
    }

    out.push_str("--- jvp\n");
    match graph.symbolic_jvp(output, "x") {
        Ok(jvp) => plan_section(out, &jvp.graph, &[jvp.tangent]),
        Err(error) => {
            let _ = writeln!(out, "error: {error}");
        }
    }
}

#[test]
fn every_op_matches_the_recorded_snapshot() {
    let mut actual = String::new();
    for (name, built) in rows() {
        render_row(&mut actual, name, built);
    }
    if std::env::var("QUABLA_BLESS").as_deref() == Ok("1") {
        std::fs::create_dir_all(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/snapshots"))
            .expect("create the snapshot directory");
        std::fs::write(SNAPSHOT_PATH, &actual).expect("write the snapshot");
        return;
    }
    let expected = std::fs::read_to_string(SNAPSHOT_PATH).unwrap_or_default();
    if actual != expected {
        let first = actual
            .lines()
            .zip(expected.lines())
            .position(|(left, right)| left != right)
            .unwrap_or_else(|| actual.lines().count().min(expected.lines().count()));
        let context = |text: &str| {
            text.lines()
                .skip(first.saturating_sub(3))
                .take(8)
                .collect::<Vec<_>>()
                .join("\n")
        };
        panic!(
            "the op table differs from {SNAPSHOT_PATH} at line {}; rerun with QUABLA_BLESS=1 \
             only if the change is intended\n--- expected\n{}\n--- actual\n{}",
            first + 1,
            context(&expected),
            context(&actual)
        );
    }
}
