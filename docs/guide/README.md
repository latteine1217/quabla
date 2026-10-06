# Quabla Developer Guide

This guide explains how to use Quabla and how it works inside. It is written
as a sequence of chapters: the first ones teach the Python API from the
ground up, the middle ones cover each library area in depth, and the last
ones describe the implementation for contributors.

The guide complements, and does not replace, the other documents:

- [API Reference](../api.md) is the exhaustive statement of every function's
  contract, numerical convention, and backend behavior. The guide links to
  it whenever a detail is too fine-grained for a tutorial.
- [README](https://github.com/latteine1217/quabla/blob/main/README.md) is the project overview and the build matrix.
- [jax_like_roadmap.md](../jax_like_roadmap.md) records per-feature status
  and validation runs.
- [CHANGELOG](https://github.com/latteine1217/quabla/blob/main/CHANGELOG.md) lists what changed in each release.

Every code example in the guide was run against Quabla v0.5.0 on the CPU
backend unless it says otherwise; printed output is shown in a `text` block
right after the code.

## Chapters

### Using Quabla

1. [Getting Started](01-getting-started.md): installation, the first
   gradient, and the mental model of eager arrays, traced values, and
   transforms.
2. [Arrays and Operations](02-arrays.md): creating arrays, dtypes and
   promotion, indexing, broadcasting, reductions, the math library, and NumPy
   interoperability.
3. [Function Transforms](03-transforms.md): `grad`, `value_and_grad`, `jvp`,
   `vjp`, `jacobian`, `hessian`, `vmap`, and `jit`, and how they compose.
4. [Control Flow](04-control-flow.md): `cond`, `fori_loop`, `scan`, and
   `while_loop` as differentiable IR regions.
5. [Custom Derivatives and Checkpointing](05-custom-derivatives.md):
   `custom_vjp`, `custom_jvp`, `checkpoint`, and `stop_gradient`.
6. [Pytrees, Random Numbers, and Serialization](06-pytrees-random-io.md):
   structured parameters, keyed random sampling, and `save`/`load`.
7. [Optimization](07-optimization.md): Adam, AdamW, SGD, learning-rate
   schedules, gradient clipping, L-BFGS, and the retained-buffer `Trainer`.
8. [Linear Algebra and Solvers](08-linear-algebra.md): `quabla.linalg`,
   Krylov solvers, and Newton's method with implicit gradients.
9. [Differential Equations](09-ode.md): `quabla.ode.odeint` with fixed-step,
   adaptive, and stiff integrators.
10. [Devices and Compilation](10-devices.md): running on CUDA and MLX,
    precision, ahead-of-time lowering, the compiler facade, and data
    parallelism.
11. [Tutorial: Training a PINN](11-pinn-tutorial.md): an end-to-end
    physics-informed neural network, from model to Adam-then-L-BFGS training
    and device execution.
12. [Performance, Debugging, and Limitations](12-performance-debugging.md):
    tracing costs, error types, common pitfalls, and known limits.

### Inside Quabla

13. [Architecture](13-architecture.md): crates, the tensor IR, autodiff
    transforms, plan compilation, and the CPU, CUDA, and MLX backends.
14. [Contributing and Extending](14-contributing.md): the development
    workflow, the test suites, and how to add a new operation.
15. [Migrating From the v0.1 API](15-migration.md): mapping the v0.1
    helpers to the core API.

## Conventions

- `import quabla as qb` is assumed in every example.
- "The core API" means the JAX-style API introduced in v0.2 (`qb.grad`,
  `qb.vmap`, `qb.jit`, ...). "The v0.1 API" means the spec-based helpers
  (`trace_tensor`, `tensor_*_fn`) and the 2D `Matrix` API, which keep working
  but are not the recommended entry point.
- Paths such as `python/quabla/optim.py` are relative to the repository
  root.
