# 14. Contributing and Extending

This chapter is a practical companion to [CONTRIBUTING.md](https://github.com/latteine1217/quabla/blob/main/CONTRIBUTING.md),
which states the project policy (compatibility, commit format, pull-request
requirements). Here we cover the development loop, the test suites, and how
to add functionality.

## 14.1 Development Setup

```sh
git clone https://github.com/latteine1217/quabla.git
cd quabla
python3 -m venv .venv && source .venv/bin/activate
python -m pip install "maturin>=1.9.3,<2" numpy ruff
maturin develop --release                  # add --features mlx or --features cuda
```

The package is a mixed Rust/Python project. `maturin develop` installs
`python/quabla/` in editable mode and writes the compiled extension next to
it as `python/quabla/_quabla.*.so`, so:

- edits to `python/quabla/*.py` take effect immediately;
- edits to any Rust crate need `maturin develop --release` again.

If `quabla.__file__` does not point into `python/quabla/` after a build
(for example because an older wheel is installed), run
`python -m pip uninstall quabla` and build again.

## 14.2 Development Gates

Run the gates that cover your change before opening a pull request; the
complete list is in [CONTRIBUTING.md](https://github.com/latteine1217/quabla/blob/main/CONTRIBUTING.md#development-gates).
In short:

```sh
cargo fmt --all --check
ruff check tests examples python
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
python tests/python/test_api.py          # and the other tests/python/*.py files
```

Each Python test file is a standalone script: it runs every `test_*`
function and prints `PASS <name>`. Run one file directly while iterating.

GPU suites are opt-in, because CI has no GPUs:

```sh
QUABLA_MLX_TEST=1 python tests/python/test_matrix.py          # MLX build on Apple silicon
QUABLA_CUDA_TEST=1 python tests/python/test_matrix.py         # CUDA build
QUABLA_CUDA_TEST=1 cargo test -p quabla-core --features cuda
QUABLA_CUDA_NCCL_TEST=1 cargo test -p quabla-core --features cuda-nccl   # two GPUs
cargo test --workspace --features quabla-core/mlx              # Rust MLX tests (macOS)
```

CUDA code only compiles on Linux. If you change a GPU backend without the
hardware to test it, say so in the pull request.

## 14.3 Where Tests Live

| Location | Covers |
| --- | --- |
| `crates/quabla-core/tests/tensor_ir.rs` | The main IR suite: ops, symbolic and runtime AD, regions, batching, the compiler facade, device parity |
| `crates/quabla-core/tests/tensor_ir_linalg.rs` | Batched solve and decompositions, their AD and batching |
| `crates/quabla-core/tests/compiler_liveness.rs`, `cpu_execution_memory.rs` | Folding, CSE, pruning, buffer liveness and reuse |
| `crates/quabla-core/tests/cpu_loop_checkpoint.rs` | Checkpointed loop VJP is bit-identical to the full tape |
| `crates/quabla-core/tests/higher_order_optimization.rs` | Symbolic HVP/Hessian against the mixed-dual reference |
| `crates/quabla-core/tests/cuda_*.rs` | CUDA-only: codegen size, `float64`, Cholesky, retained losses, solve profiles |
| `crates/quabla-core/src/tensor_ir/*_tests.rs` | In-crate unit tests (region batching, MLX checkpoints, Cholesky, solve replay) |
| `tests/python/test_api.py` | Public API, exports, operations, AD |
| `tests/python/test_matrix.py` | Broad tensor and trace behavior, legacy API, device parity under the opt-in flags |
| `tests/python/test_devices_api.py`, `test_lower_program.py` | Device `jit` and ahead-of-time lowering |
| `tests/python/test_control_api.py`, `test_custom_api.py` | Control flow; custom rules |
| `tests/python/test_optim_api.py`, `test_linalg_api.py`, `test_solvers_api.py`, `test_ode_api.py`, `test_random_api.py`, `test_numpy_api.py` | The libraries |
| `tests/python/test_packaging_api.py` | Version, exported names, the type stub |

## 14.4 Adding a Function: Choose the Layer

Most new functions do not need a new IR op. Choose the lowest-cost layer that
gives correct values and derivatives:

1. **Composition in Python** (`python/quabla/_ops.py`, `linalg.py`, ...).
   If the function can be written with existing operations, write it in
   Python. It then works eagerly, under every transform, and on every
   backend with no Rust change. `square`, `hypot`, `logaddexp`, `softmax`,
   `pad`, `interp`, and most of `qb.linalg` are compositions.
2. **A custom rule** (`qb.custom_vjp`/`custom_jvp` in Python). If the
   composition is right but its automatic derivative is unstable or
   expensive, wrap it. The Krylov solvers and Newton's method work this way.
   The rule also applies inside `cond` and loop bodies.
3. **A new IR op.** Only when the function needs its own kernel on each
   backend for accuracy or performance (special functions, a fused
   reduction, a decomposition).

### Writing a composition

Follow the conventions of `_ops.py`:

```python
def square(x):
    """Elementwise `x * x`."""
    x = _array(x)
    return x * x
```

- Convert inputs with the module's helpers (`_array`, `_operands`) so that
  numbers, lists, and NumPy arrays work, and Python scalars stay weak.
- Use only operations that work on both `Tensor` and `TraceTensor`.
- State the numerical method in the docstring when it matters (scaling,
  stable formulation, derivative conventions), as `hypot` does.
- Use `stop_gradient` for scaling factors that should not change the
  gradient, so the derivative stays exact.
- Add the name to `__all__`, and avoid shadowing Python builtins in
  `__all__` (see how `sum` and `max` are handled).

## 14.5 Adding an IR Op

A native op must be implemented end to end: every transform and every
backend, or an explicit rejection.

An elementwise math function of one or two operands is a new kind of
`UnaryMathKind` or `BinaryMathKind` in `tensor_ir/elementwise.rs`, not a new
op. Adding the variant makes the compiler ask for every rule of the
function there (name, `f64` value, numeric and symbolic derivatives, CUDA
spelling) and for its MLX lowering in `mlx_unary_math` or
`mlx_binary_math`; the questions with a default (CUDA fusion and loop
admission, folding, StableHLO) are answered next to them. Nothing else in
steps 1 to 9 below changes; the Python surface and the tests (steps 10 to
15) still do.

**Core IR** (`crates/quabla-core/src/tensor_ir.rs`):

1. Add the `TensorOp` variant.
2. A builder method on `TensorIr` that computes the output shape and dtype.
3. CPU evaluation in `evaluate_tensor_nodes_with_outputs`, computed in `f64`
   and rounded through `TensorDType::round`.
4. Symbolic JVP rule (`symbolic_jvp_many_with_seed`) and symbolic VJP rule
   (`symbolic_vjp_many_impl`), written with IR ops so higher orders work.
5. Runtime JVP/VJP (`jvp_many`, `value_and_vjp_many`) and the mixed
   second-order evaluator (`evaluate_mixed_with_input_mixed`).
6. Batching in `push_batched` (and `specialize_mapped_axis_zero`).
7. The plumbing that every op passes through: `tensor_op_inputs`,
   `remap_tensor_op`, `tensor_op_name`, `lower_text`, constant folding
   (`fold_scalar_constant_op`, `fold_tensor_constant_op`), the CSE key
   (`pure_tensor_op_cse_key`; include every attribute that distinguishes two
   instances), placement inference, and buffer reuse.

**Backends**:

8. CUDA (`tensor_ir/cuda.rs`): per-node launch, the elementwise formula
   that every CUDA generator shares (`cuda_elementwise_formula` in
   `tensor_ir.rs`), fusion
   admission (`is_fusable_elementwise_compute_op`), loop-body lowerability,
   and the `float64` forms.
9. MLX (`tensor_ir/mlx.rs`): lowering in `lower_arrays_with_retained` and
   the op name. If MLX lacks the function, implement it from MLX ops within
   a stated accuracy (as `cbrt` and `erfc` are) or reject it explicitly.

**Python**:

10. The eager method on `Tensor` (`crates/quabla-python/src/tensor.rs`) and
    the traced method on `TraceTensor`
    (`crates/quabla-python/src/tensor_trace.rs`).
11. The module-level function in `python/quabla/_ops.py`.
12. The type stub `python/quabla/_quabla.pyi`; `test_packaging_api.py` fails
    while an exported name is missing from it.

**Tests and docs**:

13. Rust tests in `crates/quabla-core/tests/tensor_ir.rs`: values against
    a reference, derivatives against finite differences or a closed form,
    every order you claim, and batching.
14. Python tests in `tests/python/test_api.py` and device parity in
    `tests/python/test_devices_api.py`.
15. `docs/api.md` (contract and numerics), `CHANGELOG.md`, and where it
    applies `docs/jax_like_roadmap.md`.

Region-level code (`region_batching.rs`, `cuda_host_loop.rs`) works on whole
regions and normally needs no per-op change.

## 14.6 Changing Public Behavior

- The compatibility policy is in
  [CONTRIBUTING.md](https://github.com/latteine1217/quabla/blob/main/CONTRIBUTING.md#scope-and-status): every v0.1 name
  keeps working through 0.x; replacements deprecate with a once-per-name
  `DeprecationWarning`. Backends, the IR, and Rust APIs are not covered and
  may change.
- Raise an existing exception class from `python/quabla/_errors.py` rather
  than inventing a new one; each subclasses the builtin exception that
  existing handlers catch.
- Prefer an explicit error to a silent fallback or a silently different
  result.
- The package version lives only in `crates/quabla-python/Cargo.toml`.

## 14.7 Writing Documentation

- Code comments, docstrings, commit messages, and documents are written in
  English.
- `docs/api.md` is the contract: exact signatures, numerical methods,
  derivative conventions, and backend differences. This guide teaches usage
  and links to it.
- Run every code example you add and paste its real output.
- Commits use [Conventional Commits](https://www.conventionalcommits.org/)
  with a lowercase subject; the body explains what was wrong, why the change
  fixes it, and which verification was run.
