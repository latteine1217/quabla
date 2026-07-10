# Nabla Core Design

## Goal

This project starts as a Rust-native SciML runtime experiment. The long-term
target is not a literal JAX clone, but a JAX-like stack with Rust-specific
answers to typed arrays, autodiff transformation, backend lowering, and
eventual sharding.

The first milestone is a small, verifiable inverse-problem vertical slice:

1. define a scientific model in Rust,
2. integrate it with a deterministic ODE solver,
3. compute a scalar loss,
4. differentiate model parameters,
5. update those parameters with an optimizer,
6. verify the derivative against finite differences and the fit loop against
   loss reduction.

## Initial Scope

The current crate focuses on:

- scalar forward-mode automatic differentiation with `Dual`,
- a compile-time shaped `Tensor2<ROWS, COLS>` with shape-checked matmul,
- a generic fixed-step RK4 ODE integrator,
- a final-state RK4 path for scalar losses that do not need full trajectories,
- a Lotka-Volterra parameter-inference example,
- a `LotkaVolterraProblem` value that keeps solver and target settings together,
- full four-parameter Lotka-Volterra gradients,
- a generic fixed-size gradient descent fitting loop,
- finite-difference gradient checks.

This is intentionally narrow. The solver is treated as one differentiable model
inside a learning workflow, not as the main product by itself. The next core
work should move toward typed tensors and transformable program
representations, not more solver-only functionality.

## Non-Goals For The First Milestone

- Matching the full NumPy API surface.
- Matching JAX transformations such as `jit`, `vmap`, or arbitrary higher-order
  autodiff.
- Adding GPU support.
- Adding adaptive or implicit solvers.
- Designing a public stable API.

## Verification

The minimum verification gate is:

```sh
cargo test
```

The tests should cover:

- elementary `Dual` derivative rules,
- typed tensor shape and matmul behavior,
- RK4 integration on a known exponential-growth ODE,
- Lotka-Volterra parameter-gradient agreement with central finite differences,
- optimizer convergence on a simple convex objective,
- Lotka-Volterra fitting reducing inverse-problem loss.

## Next Decisions

After the first milestone passes, the next design decision should be made from
evidence:

- If shape-safe tensor ergonomics are weak, add dynamic tensors and explicit
  views/strides.
- If operator-overloading AD is too limiting, add a tensor expression IR and
  generated AD transforms.
- If inverse-problem examples are too small, add observation time series and
  batching.
- If execution becomes the bottleneck, define backend traits before adding GPU
  or sharding.
