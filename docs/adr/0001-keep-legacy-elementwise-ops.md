# Keep the legacy elementwise ops as their own variants

`Add`, `Sub`, `Mul`, `Div`, `Greater`, `Sqrt`, `SqrtDerivative`, `Powi`, and
`Cast` predate the `UnaryMath` and `BinaryMath` op families and stay separate
`TensorOp` variants instead of becoming kinds of those families. Folding them
in would change the public program text (`qb.jit(f).lower(...).as_text()` and
`stablehlo_text()`), the CSE keys and node structure that the golden
baselines pin bit for bit, and the CPU fusion and CUDA kernel grouping that
depends on them, while the per-op rules would only move from the variant to
the kind rather than disappear. Per-op knowledge is instead concentrated by
giving every variant its answers in one op module (`tensor_ir/op.rs`).

Revisit this only with a separate reason, such as wider StableHLO coverage,
that justifies a program-text change in a minor release.
