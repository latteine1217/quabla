# Rust SciML Runtime Roadmap

## v0.6 Plan (2026-10-07)

v0.6 lets custom derivative rules compose with control flow and pays down
two structural costs the 2026-10-07 architecture review found. Scope chosen
by the owner on 2026-10-07; integer dtypes, complex dtypes with FFT, sparse
matrices, CUDA fusion unification, and MLX fused loop kernels are not in it.

| # | Item | Status |
| --- | --- | --- |
| 1 | One region node: the nine region variants (`Fori`, `ForiJvp`, `ForiVjp`, `ForiVjpJvp`, `Scan`, `ScanVjp`, `ScanVjpJvp`, `While`, `Cond`) become one `RegionNode` with a kind, operand slots, captures, plans, and group result, so the sites that treat regions alike (operands, remapping, CSE, placement, validation, batching) handle them once; migrated kind by kind against a bitwise golden corpus, with the `lower_text` format unchanged. A shared loop driver across backends is not part of it | Done: `TensorOp::Region(RegionNode)` replaces the nine variants (`tensor_ir/region.rs`); the generic sites use `RegionNode`, evaluation, differentiation, and batching match on `RegionKind`; the golden corpus and `lower_text` are byte-identical on CPU and MLX; the CUDA runtime suites still need a CUDA host |
| 2 | A fused CPU evaluator: chains of elementwise nodes run as one pass over the elements with no intermediate buffers, bit for bit equal to node-by-node evaluation, for CPU `jit` and the eager composite ops (`relu`, `sigmoid`, `softplus`, `abs`, ...) that became slower when eager moved onto the evaluator | Planned |
| 3 | Custom rules inside control flow: `custom_vjp` and `custom_jvp` functions may run inside `cond`, `fori_loop`, `scan`, and `while_loop` bodies; `linalg.cg`, `linalg.gmres`, and `newton` gain forward mode; `hessian` through a `custom_vjp` solver composed with a loop (reverse mode over a loop VJP) works | Done: region bodies keep their custom rule nodes (`compile_region`), so loop JVP, VJP, HVP, and `vmap` apply the rules on CPU, MLX, and CUDA; `cg`, `gmres`, and `newton` carry an implicit-function-theorem tangent beside their unchanged `custom_vjp` rule (every two-pass combination, and a third reverse pass); reverse mode over `ForiVjp`, `ScanVjp`, and `ForiJvp` is built from `ForiVjpJvp`/`ScanVjpJvp` and the loop JVP by Hessian symmetry. Open: forward mode over a `ForiJvp` and third derivatives through loops |

Order: item 1 lands before item 3, which builds on the region node; item 2
is independent. Carried release work: the two-GPU NCCL validation with
NVRTC 12.6 and the first PyPI upload.

## Before v1.0: Retiring the v0.1 Derivative Engines (2026-10-07)

Releases stay at 0.x until the items below are done; v1.0 follows them.
The 0.x compatibility promise keeps every v0.1 call form with unchanged
behaviour, so these changes, which alter v0.1 results at the ulp level or
change where non-finite values appear, wait for v1.0. Decided by the owner
on 2026-10-07 after an architecture review.

The modern API differentiates with the symbolic JVP/VJP engines. Three more
derivative engines exist only for deprecated v0.1 entry points:

| # | Item | Effect on v0.1 results |
| --- | --- | --- |
| S3 | The runtime VJP (`value_and_vjp_many`) and JVP (`jvp_many`) engines in `tensor_ir.rs` serve only the deprecated `tensor_*` functions and pyclasses (`TensorValueAndGradFunction`, `TensorVjpFunction`, `TensorJvpFunction`, `TensorJacobianFunction`, and the others listed in `_compat.py`). Rebuild those entry points as adapters over compiled symbolic plans, make the Rust `TensorIr::vjp`/`jvp` test oracles symbolic adapters, and delete the runtime engines (about 1,900 lines). | ulp-level changes in `float64`; `float32` derivatives round per operation instead of once |
| S4 | The mixed second-order evaluator (`hessian_scalar`/`hvp_scalar`, `evaluate_mixed*`, `MixedTangent`, `SolveReplayPlan`) serves only the deprecated `tensor_hessian_scalar_fn`/`tensor_hvp_scalar_fn`. Move them onto symbolic second derivatives and delete it (about 1,470 lines). | At a non-finite or domain-edge input, NaN appears only in the affected entries instead of every entry |
| S5 | The legacy `Matrix`/`TraceGraph` engine in `crates/quabla-python/src/trace.rs` (its own trace, AD, and evaluation, about 780 lines of AD) is independent of `TensorIr`. Lower `quabla.legacy` onto `TensorIr` or remove it with the deprecated names. | Depends on the `quabla.legacy` decision for v1.0 |
| A1 | `_quabla.Adam.step` (and so `optim.Adam.step`) and the legacy CUDA `cuda_adam_*` kernels keep the v0.1 update order and, on CUDA, `float32` bias corrections, as an explicitly named variant of the shared Adam rule. Switch them to the canonical rule. | ulp-level parameter changes; `optim.Adam.step` and `optim.Adam.update` then agree bit for bit |

## v0.5 Plan (2026-10-06)

v0.5 makes `vmap` compose with control flow. In v0.4, `vmap` rejects any
`cond`, `fori_loop`, `scan`, or `while_loop` region whose operands depend on
a mapped argument, so `hessian` or a forward-mode `jacobian` through a loop,
`vmap` of `odeint`, and `vmap` of `linalg.cg`, `linalg.gmres`, or `newton`
all raise. Scope chosen by the owner on 2026-10-06; custom differentiation
rules inside loop bodies, integer dtypes, and complex dtypes with FFT are
not in it.

| # | Item | Status |
| --- | --- | --- |
| 1 | Batching rule for `fori_loop` and `scan` regions and their derivative regions (JVP, VJP, forward-over-reverse HVP): mapped captures and carries batch the body region, unmapped operands stay unbatched | Done: every loop region node batches on CPU, CUDA, and MLX, with the carry fixed point; `vmap`, `jacobian`, and `hessian` through `fori_loop`/`scan` and `vmap` of `odeint` (every method, `saveat`) work |
| 2 | Batching rule for `cond`: an unmapped predicate batches both branch regions; a mapped predicate evaluates both branches and selects per element, as JAX does | Done: `cond` and its JVP and VJP regions batch on CPU, CUDA, and MLX; a mapped predicate is a `where` of both batched branches that keeps the unselected branch's NaN out of values and mapped-operand derivatives; `vmap`, `jacobian`, and `hessian` work through `cond`, conds in loop bodies, and loops in branches |
| 3 | Batching rule for `while_loop`: an unmapped predicate batches the body; a mapped predicate runs until every element is done and freezes finished elements' carries, as JAX does | Done: on CPU, CUDA, and MLX, including the forward-mode while regions of `jvp` and forward `jacobian`; finished elements keep their carry and tangent bit for bit, and CUDA still reads back one flag per iteration |
| 4 | What the rules unlock, on CPU, CUDA, and MLX: `hessian` and `jacobian` through loops, `vmap` of `odeint` for every method, and `vmap` of `linalg.cg`, `linalg.gmres`, and `newton` | Done: `hessian` and `jacobian` through loops and `vmap` of `odeint` with item 1; `vmap` of `cg`, `gmres`, and `newton` over `b`, `x0`, and `args` with item 3, each element stopping at its own tolerance with per-element `info`, composed with their derivatives (`vmap(grad(...))`, `grad` of `vmap`), and the reverse-mode `jacobian`/`hessian` of `cg` and `gmres` solutions |

All items shipped in v0.5.0, validated on the CPU, MLX, and a single-GPU
CUDA host (NVRTC 13.1). Still open from the release process: the two-GPU
NCCL validation with NVRTC 12.6, skipped for v0.4.0 and v0.5.0, and the
first PyPI upload once TestPyPI registration works. Known gap found on the
way: `hessian` through a `custom_vjp` solver composed with a loop (for
example `newton` with `args` computed by `odeint`) needs reverse mode over
a loop VJP, which is not implemented.

## v0.4 Plan (2026-10-06)

v0.4 fills the jax.numpy surface a PINN code base reaches for, adds the
stiff and implicit solvers scientific ML needs, and removes the backend gaps
v0.3 left. Scope chosen by the owner on 2026-10-06; integer dtypes and
traced integer indexing (`sort`, `argmax`, `x[idx]`, `.at[]`) are not in it.

| # | Item | Status |
| --- | --- | --- |
| 1 | Native elementwise ops: `tan`, `arcsin`, `arccos`, `arctan`, `sinh`, `cosh`, `arcsinh`, `arccosh`, `arctanh`, `log2`, `log10`, `cbrt`, `floor`, `ceil`, `round`, `fmod` (with floor `mod`), on every backend with derivatives | Done |
| 2 | jax.numpy compositions: `flip`, `roll`, `pad`, `tile`, `repeat`, `moveaxis`, `swapaxes`, `ravel`, `diag`, `diagonal`, `trace`, `outer`, `dot`, `tensordot`, `kron`, `cross`, `diff`, `trapezoid`, `interp`, `polyval`, `logaddexp`, `hypot`, `exp2`, `isinf`, `nan_to_num`; `linalg.norm` (`ord`, `axis`), `linalg.matrix_power`, `linalg.pinv` | Done: Python compositions on every backend (CPU, CUDA, MLX) |
| 3 | Stiff ODEs and `saveat`: an adaptive L-stable Rosenbrock method in `odeint`, and states at requested times through each method's dense output | Done: `method="rosenbrock23"` and `saveat` for every method, on every backend |
| 4 | Iterative and implicit solvers: matrix-free `linalg.cg` and `linalg.gmres`, Newton root finding, each with an implicit-function-theorem gradient instead of differentiating the iterations | `cg` and `gmres` done (reverse mode, twice composable; no forward mode or `vmap`, and not inside `cond`/`fori_loop`/`scan` bodies); `quabla.newton` done with the same derivative support |
| 5 | MLX factorizations: `solve`, `det`/`slogdet`, `inv`, `eigh`, `qr`, `svd`, `lstsq` through MLX's CPU stream instead of an error | Done: LAPACK factorizations on the CPU stream with the CPU's conventions applied on the GPU stream, `solve` by a Metal LU substitution kernel |
| 6 | Device `Trainer` parity: AdamW and global-norm clipping on CUDA and MLX, `float64` on CUDA | Done: AdamW, SGD, and `clip_norm` on CUDA and MLX with the CPU's update order and the global norm reduced on the device; `Trainer(precision="float64")` on CUDA |
| 7 | CUDA loops: lower the per-iteration cost of host-driven loops (about 80 us in v0.3) and an HVP fallback for Scan bodies that are not elementwise | Done: one CUDA graph launch per iteration (about fourfold faster, bit-identical); host-driven `ScanVjpJvp` |

Also fixed during v0.4, outside the plan: traced `max`/`min` reductions are
one native IR op instead of a chain of per-element comparisons (a 600-element
`jit` `max` went from 3,597 nodes and 0.47 ms to 2 nodes and 2.3 µs on the
CPU), with JAX's tie-splitting derivative and an order-independent `-0 < +0`.

All items shipped in v0.4.0. The release was validated on the CPU, MLX, and
a single-GPU CUDA host (NVRTC 13.1); the two-GPU NCCL run with NVRTC 12.6
was not repeated for v0.4.0 and is the first follow-up, together with the
first PyPI upload, which waits on TestPyPI registration.

## v0.3 Plan (2026-10-06)

v0.3 closes the gaps a PINN or scientific-ML user meets when moving from
JAX. The list comes from an audit of v0.2.2 against jax.numpy, jax.lax,
optax, equinox and diffrax; the numerical and performance defects that audit
confirmed were fixed in v0.2.3 (division derivatives without squared
denominators, IEEE log and division on the CPU, stable sigmoid and softplus
with a native `log1p`, NaN-propagating `max`/`maximum`/`relu`, native
`Gather`/`ScatterAdd`, node-by-node CPU elementwise execution, and MLX
`Trainer` parameters the loss ignores). Status of the work items:

| # | Item | Status |
| --- | --- | --- |
| 1 | Optimizers: L-BFGS, AdamW, schedules, global-norm clipping, device `Trainer` schedules | Done; AdamW and clipping are CPU-only in `Trainer` |
| 2 | Elementwise ops and reductions | Done: native `expm1`, `erf`, `erfc`, `atan2`, `cumsum`; `prod`; compositions `softmax`, `logsumexp`, `var`/`std`, `silu`, `gelu`, `clip`, `sign` |
| 3 | `stop_gradient` | Done |
| 4 | Shape ergonomics | Done |
| 5 | Pytrees: NamedTuple, dataclasses, registration | Done |
| 6 | `qb.random` | Done |
| 7 | `while_loop`, ODE solvers | Done on the CPU, CUDA, and MLX: `while_loop` (forward mode), fixed-step and adaptive `dopri5` `odeint` |
| 8 | Linear algebra | Done: batched `solve`, `qb.linalg` with `slogdet`/`det`/`inv`/`cho_solve`/`eigh`/`qr`/`svd`/`lstsq` (CPU and CUDA; MLX rejects them) |
| 9 | `custom_vjp`/`custom_jvp`, `checkpoint`, `jit` keywords | Done; custom rules cannot run inside control-flow bodies |
| 10 | Native `float64` on CUDA | Done as opt-in `jit(..., precision="float64")`; `Trainer` and NCCL stay `float32` |
| 11 | Packaging and tooling | Done: `__version__`, value `repr`, `save`/`load`, type stubs, wheel workflow; PyPI publishing of `quabla` and `quabla-mlx` added after v0.3.0 |

All items shipped in v0.3.0. Its open follow-ups are items 6 and 7 of the
v0.4 plan and the first PyPI upload.

Decisions taken by the owner on 2026-10-06 and implemented:

- Trace caches evict their least recently used trace with a
  `RetraceWarning` beyond `max_traces` instead of raising
  `RetraceLimitError`, so a training loop whose collocation count changes
  every epoch keeps running.
- `norm` scales by `max|x|`, so it no longer overflows for float32
  magnitudes above about 1e19; results change in the last bits.

Smaller observations kept for later: `abs` has gradient -1 at 0 (JAX uses
+1), and CPU float32 reductions accumulate in f64, so device float32 sums
can differ from the CPU reference beyond roundoff for very large or
cancelling inputs. `max` used to give the whole gradient of a tie to the
last element; it splits it as JAX does since v0.4.

## v0.2 API Increment (2026-10-03)

S5-S9 are implemented on the current working tree: explicit device `jit` and
abstract lowering, pure optimizers and retained-buffer `Trainer`, control-flow
wrappers, experimental distributed value-and-gradient, and compatibility
warnings/legacy namespace. Dtype phases D2's integer indices and D3-D6 remain
separate future work. The inherited backend restrictions are unchanged.

The accepted API and implementation refinements are in `api_v0_2_design.md`.
The macOS CPU extension, workspace tests, clippy, fmt, ruff, old Python matrix,
new API tests, and five added Python suites pass. Ordered-output facade tests
also preserve the original single-output `Compiler.trace` behavior.

On the GTX 1660 SUPER CUDA host, workspace tests with CUDA runtime enabled,
clippy with `cuda-nccl`, the existing Python matrix/API suites, and the new
device jit/AOT, Trainer convergence, and control-flow suites pass.
The interleaved Poisson PINN benchmark compares
1,000 steps per executor: native and Trainer checkpoint losses/weights are
identical; on the released tree the Trainer/native synchronized step ratio is
1.0028, below the 1.02 gate. This is one host/run and one retained-input workload, not a general
performance guarantee.

A two-GPU validation run of the released tree (`main` after the S5-S9
commits; 2x RTX 3090, driver 560.35.05, NVRTC 12.6, NCCL) passes the
workspace Rust tests with `QUABLA_CUDA_TEST=1 QUABLA_CUDA_NCCL_TEST=1` and
`quabla-core/cuda-nccl`, and every `tests/python/test_*.py` suite with both
variables set. Data-parallel parity against the CPU oracle passes for every
float64/float32 and Sum/Mean case (16 cases of 5 steps each, maximum loss
error 1.5e-8, maximum gradient error 2.4e-8, tolerance 1e-5). Two-GPU training matches single-GPU and CPU references: 200 steps
on 256 rows reach a maximum relative loss difference of 2.7e-7 against CPU,
and 50 steps on 65,536 rows with width 256 reach 6.0e-7 against a single GPU;
in that larger configuration the steady-state step takes 4.71 ms on two GPUs
and 8.40 ms on one. The retained Trainer benchmark reproduces the native
executor's losses and weights exactly with a synchronized step ratio of
1.0088, below the 1.02 gate. These are single runs on one node, not
performance guarantees.

Reproduction on a Linux machine with two CUDA GPUs and a loadable
`libnccl.so` (CUDA ordinals 0 and 1; select them with
`CUDA_VISIBLE_DEVICES`):

```sh
QUABLA_CUDA_TEST=1 QUABLA_CUDA_NCCL_TEST=1 \
  cargo test --workspace --features quabla-core/cuda-nccl
maturin develop --release --features cuda-nccl
QUABLA_CUDA_NCCL_TEST=1 python tests/python/test_distributed_api.py
QUABLA_CUDA_TEST=1 python tests/python/test_devices_api.py
QUABLA_CUDA_TEST=1 python tests/python/test_optim_api.py
QUABLA_CUDA_TEST=1 python tests/python/test_control_api.py
python examples/benchmark_v02_training.py --device cuda:0 --max-step-ratio 1.02
```

On a single GPU without NCCL, build with `--features cuda` and drop
`QUABLA_CUDA_NCCL_TEST`; the distributed suite then checks the
feature-unavailable contract instead of NCCL parity.

The Metal Toolchain is now installed and the release MLX extension builds
and loads on the Apple M3 / macOS 27.0.1 host. All five new Python suites,
the existing Python matrix/API suites, and Rust workspace tests with
`quabla-core/mlx` pass (201 passed, 0 failed, 1 ignored). MLX all-target
clippy with `-D warnings` passes after retrying a GitHub DNS download failure;
fmt, ruff, and diff checks also pass. The MLX solve-rejection fixture was corrected to use
a rank-two right-hand side, so it reaches backend validation.

For this local Rust/Xcode combination, the default release build linked but
failed to load with `mis-aligned LINKEDIT string pool`. Building with the
per-command `CARGO_PROFILE_RELEASE_STRIP=none` avoids the failure, matching
[Rust issue 157750](https://github.com/rust-lang/rust/issues/157750).
No project build defaults were changed. Reproduce with:

```sh
CARGO_PROFILE_RELEASE_STRIP=none maturin develop --release --features mlx
QUABLA_MLX_TEST=1 python tests/python/test_devices_api.py
QUABLA_MLX_TEST=1 python tests/python/test_optim_api.py
QUABLA_MLX_TEST=1 python tests/python/test_control_api.py
python examples/benchmark_v02_training.py --device mlx --max-step-ratio 1.02 \
  --samples 19 --steps-per-sample 500
```

The isolated 9,600-step-per-executor MLX benchmark has identical loss/weight
checkpoints and a synchronized Trainer/native step ratio of 0.81017,
passing the 1.02 gate. The first run overlapped other verification and had
ratio 1.07599 (failed performance gate, exact numerical parity). These measurements are workload/run-specific and do
not establish a general speedup. One later MLX stream initialization failed
before graph execution; a Metal device probe and the complete suite rerun
passed without code changes. The cause of that transient failure remains
unconfirmed.

### Algorithm and Memory Review (2026-10-03)

This records the pre-optimization review of the v0.2 working tree and its
inherited executors. The implementation follow-up is recorded below.
Source line references in this review refer to that earlier snapshot. The CPU
probe scripts and their outputs were kept outside the repository.
RSS observations include allocator/runtime effects, not exact buffer accounting.

Notation: V/E are graph nodes/edges; N is a uniform tensor element count;
D is the depth of the doubling DAG; T is loop iterations; C/Y are carry/output
elements per iteration; L is selected parameter leaves; P/Q are flattened
Jacobian input/output sizes; R is tensor rank; G is replica count and B/U
are full mapped-batch/replicated-input elements. Costs exclude caller-owned
inputs unless stated. Bounds describe these execution strategies, not a
claim of globally optimal graph scheduling.

| Priority | Path | Current cost | Concrete target |
| --- | --- | --- | --- |
| 1 | CPU fused shared DAG | Doubling graph time O(N * 2^D); fusion eligibility also repeats shared subgraphs | O(N * (V+E)) execution with reusable per-element memoization; O(V+E) eligibility; O(V) scratch |
| 2 | CPU frozen forward executor | Retains every node value; a uniform chain uses O(VN) array storage | Last-use release/reuse: memory proportional to the live frontier plus retained outputs; O(N) for the chain |
| 3 | CPU forward-only Fori/Scan | Fori records O(TC) carry tape and then discards it; Scan adds O(TC) to required O(TY) outputs | Separate no-tape forward path: O(C) carry storage for Fori, O(C+TY) for Scan, plus body workspace |
| 4 | Dense Jacobian basis | L full P-by-P identity allocations: O(LP^2) initialization; retained basis is O(P^2) | Direct identity blocks reduce initialization to O(P^2); chunked JVP reduces basis workspace to O(bP), plus required O(PQ) result and chunked body workspace |
| 5 | Pure CPU Adam | 14 full-size arithmetic allocations per leaf/update; identical-shape binaries still do O(NR) indexing | One fused pure leaf update with three result buffers and O(N) work; equal-shape binary fast path |
| 6 | Distributed host sharding | Clones the full batch G times before replacing it with shards: O(GB) avoidable copy work | Construct maps from shards directly: O(B) mapped-data copy work; remove one transient full-batch copy |

Evidence and correctness constraints:

- **Shared DAG:** `evaluate_fused_element` recursively evaluates each child
  occurrence (`tensor_ir.rs:11981`), including both references in `x+x`.
  With eight elements, depths 6/10/14/18 take median hot times
  0.031/0.386/5.892/94.225 ms. Each four extra levels approaches a 16x
  increase. Preserve per-node dtype rounding, broadcast indexing and lazy
  `where` semantics when memoizing. Eligibility recursion at line 11860
  also needs a visited/memo table.
- **Forward liveness:** `evaluate_tensor_nodes` retains its entire `values`
  vector (`tensor_ir.rs:7048`); `evaluate_many` additionally clones outputs.
  A 100,000-element sine chain returning its vector and sum has peak-RSS
  increases of 9.93/26.07/42.17 MB at 10/30/50 nodes. The existing
  `buffer_plan` only protects the first output; extend it to every output
  and storage alias before reuse. Keep the AD tape evaluator distinct.
  Test duplicate outputs, early outputs consumed later, shared DAGs,
  reshape aliases and grouped loop outputs.
- **Unneeded forward tape:** `TensorForiLoopPlan::evaluate` calls
  `evaluate_with_tape` and drops its tape (`tensor_ir.rs:9489`); Scan does
  the same at line 10210. Forward Fori with a 100,000-element carry has
  peak-RSS increases of 14.63 MB for 10 iterations and 46.91 MB for 50.
  Keep tape-producing paths for reverse-mode consumers; validate zero
  iterations, explicit captures, nested loops and scan stacking. Reverse
  differentiation may trade tape memory for recomputation via checkpointing;
  that is a separate measured design decision.
- **Jacobian:** `_Jacobian._stage` creates `eye(total)` inside the selected
  leaf loop (`_transforms.py:1493`). Eight leaves with 64 total elements
  create eight 64-by-64 identities (32,768 initialized elements), although
  the basis blocks together contain only 4,096. Repeated allocation volume
  is not peak memory: retained blocks are quadratic, and one full identity
  is transient. A dense P-input/Q-output Jacobian inherently requires
  Omega(PQ) result storage; a dense Hessian requires Omega(P^2). Use JVP,
  VJP or forward-over-reverse HVP when only products are needed. A scalar
  output Jacobian can potentially use reverse mode rather than a quadratic
  forward basis, subject to derivative/shape/dtype equivalence tests.
- **Adam:** `optim.py:98-105` materializes m, v, delta and parameter
  expressions as separate eager arrays. Fourteen f64 arithmetic allocations
  total 112N bytes per update (allocation volume, not peak); float32 output
  conversion adds a copy. Widening to float64 shares host storage and is not
  itself an elementwise copy. A pure update must preserve the old state and
  return independent parameters/m/v; do not substitute the stateful legacy
  optimizer. Test varying gradients, long trajectories, input-state
  immutability, NaNs and float32 rounding. Host tensors use f64 backing for
  every logical dtype; real f32 storage is a separate dtype/storage migration.
- **Distributed sharding:** native `tensor_trace.rs:6959` clones the entire
  input map per replica and then replaces mapped tensors with copied slices.
  Build each map from sliced mapped inputs and cloned replicated inputs.
  Retained maps still occupy O(B+GU); achieving shared replicated backing
  requires a separate storage/interface decision. Verify exact shard
  coverage, multiple mapped tensors, replica input validation and both
  NCCL reductions. This finding is source-derived; no new GPU memory or
  throughput measurement was run for this review.

Secondary cold-path candidates: constant folding holds newly folded arrays
until final DCE (potential O(VN) peak on a constant chain), and separately
created Lowered objects do not share the normal JIT executable cache.
Prioritize the confirmed hot-path complexity and memory issues above;
closure snapshot semantics must be specified before combining caches.
CUDA retained Adam `loss()` also executes the shared backward plan
(`tensor_trace.rs:8001`, `cuda.rs:1616`) before returning its loss output;
a cached forward-only loss plan is worth evaluating when loss is requested
frequently. It does not eliminate the backward computation needed by `step()`.

### Optimization Implementation and Verification (2026-10-03)

The six prioritized items are implemented; public call forms, dtype policies
and legacy stateful optimizers remain unchanged.

- CPU fusion eligibility is topological O(V+E). Scalar evaluation memoizes
  each reachable node per element with reusable O(V) scratch, preserving
  lazy `where` and node rounding. At fixed rank, work is O(N(V+E)); input
  broadcast indexing still includes rank-dependent work and the evaluator
  retains an O(depth) recursion stack.
- Frozen CPU forward execution releases values after their final consumer,
  protects every ordered/duplicate output, moves final output occurrences,
  and releases grouped-region caches after their last sibling. The buffer
  plan now protects all output storage roots. Reverse tape evaluation keeps
  its required values. This is last-use release, not in-place buffer reuse.
- Forward-only Fori and Scan no longer build a carry tape. Fori uses O(C)
  carry storage plus body workspace; Scan also needs O(TY) returned outputs.
  Reverse consumers still use the explicit tape-producing paths.
- Jacobian uses bounded 64-direction JVP chunks. Runtime basis generation
  compares linear-size radix coordinates, with digits exactly representable
  in GPU float32, rather than building full identities. Basis workspace is
  O(bP), plus chunked body workspace and the required O(PQ) result. The graph
  grows as O(ceil(P/b)V). Forward AD work remains O(P*C_f), where C_f is the
  primal evaluation cost; basis comparison can still cost O(P^2). No adaptive
  reverse-mode Jacobian or matrix-free result representation was added.
- Pure CPU Adam computes each leaf in one native pass, producing independent
  params/m/v backing buffers (24N output bytes with current f64 host storage).
  It preserves old state, operation order and final parameter rounding.
  Equal-shape eager binaries use zipped loops; broadcast fallback remains.
- Distributed maps are built directly from mapped slices and replicated
  clones. Mapped copy work is O(B), removing the O(GB) full-batch clones.
  Replica maps still retain O(B+GU), in addition to original caller inputs.

Same CPU probes, with outputs preserved separately from the baseline:

| Probe | Before | After |
| --- | --- | --- |
| Eight elements, depth-18 shared doubling DAG, median hot time | 94.225 ms | 0.007375 ms |
| 100,000 elements, 50-node sine chain, peak RSS increase | 42.17 MB | 1.85 MB |
| 100,000-element carry, 50 Fori iterations, peak RSS increase | 46.91 MB | 5.88 MB |

After optimization, the sine-chain RSS increase stays at 1.85 MB for
10/30/50 nodes; Fori increases are 5.85/5.88 MB for 10/50 iterations.
These are process-RSS observations for these fixtures, not allocator-exact
bounds or general speedup promises. A scalar quadratic Jacobian with P=8192
has observed compilation/execution RSS increases of 4.91/17.33 MiB;
the original P-by-P f64 basis alone would require 512 MiB. Its runtime is
2.495 s, consistent with remaining forward quadratic work for this loss.

Verification passes: MLX-feature workspace Rust tests (212 passed, 0 failed,
1 ignored), MLX all-target clippy with `-D warnings`, fmt, ruff and diff checks;
the existing Python matrix/API and all five v0.2 suites; new fused-Adam and
chunked-Jacobian Python suites. Deterministic Rust regressions check a depth-40
shared DAG, a 100-node forward chain's actual retained-value peak, all-output
storage roots, lazy branches, no-tape/taped equivalence and grouped outputs.
Python regressions check 250-step exact eager Adam trajectories, immutable
input state, chunk boundaries, captures and nested transforms.
An additional forced-radix, float32 65-by-65 Jacobian identity probe passes
on both MLX and CUDA, exercising the GPU-safe multi-digit basis coordinates.

The CUDA host passes runtime-enabled workspace Rust tests and CUDA/NCCL
all-target clippy, the existing/new Python suites and the two added suites.
An unavailable-feature fixture initially assumed one error wording; it now
checks the common CUDA error target and typed exception metadata, covering
CPU-only and CUDA-without-NCCL builds without changing production messages.
The affected suites were rerun and passed.

The final two-GPU validation run on the same kind of node (2x RTX 3090,
NCCL) completed with exit status 0 in 13 seconds. All seven Python suites and
float32/float64 NCCL Sum/Mean parity pass. The retained Trainer benchmark has
exact trajectory parity and Trainer/native step ratio 1.000583, passing the
1.02 gate. Source hashes and output JSON for that run, and the original and
Jacobian CPU probes, were kept outside the repository; no baseline was
overwritten.

At the close of the October 3 batch, opportunities still included secondary
cold-path items, adaptive Jacobian direction, typed host storage, and
reverse-loop checkpointing. The October 4 status below supersedes that list. These require separate compatibility and
time/memory tradeoff decisions; the six completed optimizations do not claim
globally optimal graph scheduling or memory use for every workload.

### Comprehensive Optimization Audit (2026-10-03)

This follow-up reviews production computation paths beyond the six completed
optimizations: AD/compiler passes, CPU kernels/storage, CUDA/MLX/distributed
execution, Python transforms, legacy Matrix execution, scalar AD/macros, ODEs
and benchmark boundaries.
This section records candidates, not newly implemented improvements. References
below describe the current working-tree snapshot and may move after edits.

#### Coverage and evidence limits

| Area | Reviewed source families and paths |
| --- | --- |
| Graphs and AD | Core `compiler.rs`, `tensor_ir.rs`: freezing, folding, CSE, fusion metadata, symbolic/numeric JVP/VJP, Hessian/HVP, region derivatives, liveness and buffer planning |
| CPU arrays | Core `tensor.rs`, `tensor_ir.rs`; native `tensor.rs`, `matrix.rs`, `interop.rs`: physical dtype storage, boundary conversion, broadcasting, reductions, indexing, matmul, solve, triangular solve, Cholesky and NumPy exports |
| Device execution | `tensor_ir/cuda.rs`, `tensor_ir/mlx.rs`, native `tensor_trace.rs`: generated kernels, regions, pooling, solver workspace, retained Adam, host transfers, synchronization and NCCL execution |
| Python orchestration | `_transforms.py`, `tree.py`, `_array.py`, `_ops.py`, `_control.py`, `optim.py`, `distributed.py`: signatures, trace caches, lowering, pytrees, dense derivatives, VJP pullbacks, eager loops and optimizer updates |
| Legacy/scalar paths | Native `trace.rs`, `optim.rs`, compiler facade; core `autodiff.rs`, `ode.rs`, `models.rs`, `optim.rs`; macro expansion in `quabla-macros/src/lib.rs` |
| Surface and measurement | Package/legacy/compat/device/dtype/error facades and retained-training benchmark: no separate numerical kernel bottleneck established in these facades; existing benchmark is a workload-specific parity/overhead gate |

Source analysis establishes extra work and allocation, but not the speedup of a
proposed implementation. The four probes below were actually executed locally.
No new GPU execution, NVRTC compilation, two-GPU run, production edit or
baseline replacement is part of this audit. Device throughput, allocator
peaks, register pressure and collective overlap still need targeted profiling.
Coverage of current paths is not proof that every possible optimization or a
globally optimal schedule has been found.

Notation: V/E are graph nodes/edges; N is array elements; P/Q are differentiated
input/output elements; C_f is primal evaluation cost; T is loop iterations; C is
carry size; U is external capture elements; n is a square system dimension;
k is RHS columns; r is tensor rank.
Priorities below order investigation; they are not correctness-bug severities.

#### Highest-priority algorithm and compiler candidates

| ID | Original audit evidence/cost | Proposed improvement and verification |
| --- | --- | --- |
| A1 | CUDA expression builders recursively expand each child occurrence (`tensor_ir.rs:11798`, `11878`; `cuda.rs:4193`, `4305`, `4443`). A shared doubling DAG has O(d) nodes but Theta(2^d) generated text. The local source-only probe reaches 9,437,779 bytes with just 19 nodes. | Emit topological scalar temporaries or bounded fusion regions. Merely memoizing strings still duplicates expanded text at parents. Verify source-size scaling and GPU parity; preserve lazy `where`, operation order and per-node dtype semantics. |
| A2 | Non-region native `hvp_scalar` repeats `evaluate_mixed` P times (`tensor_ir.rs:6481`); work is O(P*C_f), although only P results are returned. | Reuse the existing symbolic VJP/JVP machinery, where supported, with a cached derivative plan. Verify all supported operators, bool/F32 rounding, singular/error behavior and region/nested AD. Eager derivative intermediates use F64 whereas symbolic nodes can round, so replacement is not automatically equivalent. |
| A3 | Non-region native `hessian_scalar` has P-by-P mixed evaluations (`tensor_ir.rs:6430`), O(P^2*C_f) work plus the necessary P^2 result. | Build Hessian columns from batched/chunked JVP of a gradient, targeting O(P*C_f) derivative work. Preserve native return shape and numerical conventions; compare analytic, finite-difference and nested-transform fixtures. |
| A4 | Eager/traced triangular solve masks/transposes A then calls generic LU (`tensor.rs:1000`, `tensor_trace.rs:1586`, core solve `tensor_ir.rs:2048`): O(n^3+n^2*k), with matrix-sized temporaries. | Dedicated forward/back substitution uses O(n^2*k) work. Recognizing the existing graph pattern can preserve the public API. Verify lower/upper/transposed systems, multiple RHS, dtype and singularity policies; changed arithmetic order requires an explicit tolerance decision. |
| A5 | Folding creates fresh array constants and retains them through final DCE (`tensor_ir.rs:5433`, `5462`, `5481`). An N-element, V-operation constant chain can use Theta(VN) cold-path storage. | Use folding last-use information or a separate temporary-value arena, retaining only final needed constants. Measure cold compile peak/time on constant chains, fan-out and multiple outputs; keep constant rounding/error behavior. |
| A6 | Fusion user construction checks `Vec::contains` for every new user (`tensor_ir.rs:13125`). One node with d distinct users causes Theta(d^2) membership work. | Since nodes are visited in order, deduplicate repeated operands with a last-user check or a set. Verify identical region boundaries/output protection and linear scaling of high-fan-out graph metadata. |
| A7 | Symbolic JVP/VJP constructs/clones source graph state before output reachability pruning (`tensor_ir.rs:2811`, `3284`). Dead traced work costs graph storage and transformation time. | Prune before transformation, preserving requested retained outputs and the contract returning gradients for all named inputs. Verify unused inputs, auxiliaries, captures, errors and transformed node counts. |
| A8 | Numeric Scan JVP calls the full body JVP separately for next carry and output every iteration (`tensor_ir.rs:10398`, `10400`, `10402`); both compute the primal/tangent graph. | Share one multi-output numeric JVP per step, preserving current F64 derivative arithmetic. The cost remains O(T*C_body), but duplicated traversal and workspace can be removed. Verify shared-prefix invocation counts, captures, nested scans and carry/output tangent parity. |
| A9 | Fori body input/tangent construction clones dense external captures every iteration (`tensor_ir.rs:9756`, `9767`), including reverse replay: O(TU) copy volume for U immutable captured elements. | Borrow/share immutable capture storage separately from changing carry/index bindings. Count copied bytes over T and verify capture lifetime/immutability and nested derivative behavior; forward tape removal did not address these copies. |

#### Storage, CPU kernels and legacy execution

| ID | Original audit evidence/cost | Proposed improvement and verification |
| --- | --- | --- |
| B1 | Host `DynamicTensor` and native `PyTensor` physically store F64 (`tensor_ir.rs:401`, `tensor.rs:58`): F32 and bool cost 8N bytes too. F32/bool exports additionally materialize 4N/N bytes (`interop.rs:301`); F64 export shares its Arc. | Typed host storage can reduce resident bytes; this is a larger internal representation/interop change. Verify promotion, rounding, immutable aliases, views and buffer formats. Do not label F64 exports or Arc clones as full copies. |
| B2 | `PyTensor` to `DynamicTensor` clones data; consuming conversion back still uses `to_vec` (`tensor.rs:466`, `474`). Staged calls perform both conversions (`tensor_trace.rs:2763`), and interpreter input binding clones again (`tensor_ir.rs:13723`). | First move owned outputs through an internal consuming conversion; consider shared/borrowed immutable input storage separately. Count copied bytes and test output lifetime, aliases, dtype rounding and original-input immutability. |
| B3 | Numeric reverse AD retains every primal and clones/keeps processed cotangents (`tensor_ir.rs:5534`, `5559`; legacy `trace.rs:1446`). Primal tape plus adjoint retention can both scale with graph array volume. | Take processed non-input cotangents and reuse owned accumulation buffers; retain only primal values actually required by backward consumers. Verify fan-out, repeated operands, aliases and accumulation order. Tape reduction needs its own liveness analysis. |
| B4 | The fixed rank-N forward executor now releases dead values, but does not use reusable CPU slots for arithmetic outputs (`tensor_ir.rs:11393`; `TensorBufferPlan` at `1428`). | Consume safe last-use buffers or add an internal pool. Memory is already independent of chain depth for the fixed fixture; target allocator traffic, not a second claim of fixing Theta(VN) forward retention. Protect all outputs/aliases and immutable inputs. |
| B5 | Core/native generic broadcast/reduction paths perform per-element rank-dependent coordinate/div/mod work (`tensor_ir.rs:1751`, `2220`; native `tensor.rs:1110`, `1267`). Only native equal-shape binary arithmetic already has a zipped fast path. | Add contiguous/equal-shape and outer/reduced/inner block kernels where indexing allows O(N) traversal. Preserve reduction order, NaN/tie behavior and broadcasting for arbitrary ranks; benchmark equal shapes separately from true broadcasts. |
| B6 | CPU matmul uses scalar loops. Core dynamic matmul already traverses row/inner/column with contiguous RHS/output (`tensor_ir.rs:2006`); native eager (`tensor.rs:870`), legacy (`matrix.rs:454`) and static (`core/tensor.rs:45`) use row/column/inner with strided RHS. Dense work remains O(mnk), or O(Bmnk) over batches. | Improve locality in the latter paths and evaluate blocking/size-gated SIMD or BLAS. Measure small/large and batched/broadcast products with memory; do not promise a new asymptotic bound or silently change reduction precision/order. |
| B7 | Mixed Solve AD factors the same A for value, first, second and mixed results (`tensor_ir.rs:8217`). JVP/VJP also solve/refactor related systems (`6248`, `5824`). | Factor once and solve several RHS using that factor; use transpose solves without a full matrix transpose where safe. Verify pivot choices, singular diagnostics and F32/F64 gradients. Reuse within a call first; cross-call factor caching must not assume A is unchanged. |
| B8 | Traced Cholesky expands scalar loops into O(n^3) graph operations (`tensor_trace.rs:1602`), beyond the O(n^2) matrix result. | A native IR operation and its AD/backend rules avoid graph/launch expansion while retaining O(n^3) arithmetic. This crosses modules; test SPD failures, lower-triangle conventions, batching and higher derivatives before adoption. |
| B9 | Native multi-axis indexing materializes successive slices (`tensor.rs:1497`); gather materializes chunks and then concatenates them (`1520`). Full slices across r axes can copy O(rN) data; gather retains a result-sized chunk set before final output. | Compose validated layouts before one materialization; write gather directly into its final buffer. Verify noncontiguous/reversed views, bounds, repeated indices, output ordering and alias lifetime. Returned gather data itself remains necessary. |
| B10 | Legacy Matrix forward execution retains all node values and clones its output (`trace.rs:1836`, `1358`). The depth-50 sine probe retains about 41.14 MB of additional process RSS. The rank-N fix did not modify this separate executor. | Apply separate forward liveness/output moves with legacy error/shape protection. Check old Matrix tests and chain memory scaling; do not change reverse tape semantics by applying forward release indiscriminately. |
| B11 | Legacy dense Matrix Jacobian builds a seed and calls full VJP for every output row, repeating primal execution and cloning the input map (`trace.rs:2886`). `jacobians_fn` repeats this for each selected input. | Share one primal evaluation, derive all requested input blocks per seed, and choose forward/reverse direction by dimensions where supported. Keep the O(PQ) dense result and immutable legacy interface; compare all blocks and error behavior. |
| B12 | Core DynamicTensor reshape clones the data (`tensor_ir.rs:1837`), taking O(N) time/storage; native eager reshape shares its Arc (`tensor.rs:1053`). Buffer-plan alias metadata does not make interpreter reshape zero-copy. | Share or move the backing storage for shape-only changes. Verify aliases, duplicate outputs, rounding and reverse tapes; preserve native eager sharing and do not conflate metadata with executed allocation behavior. |
| B13 | Typed buffer import first copies into a typed temporary then widens into F64 (`interop.rs:221`); F32 input has an additional 4N-byte temporary. Owned import itself is part of the contract. | Gather/widen directly into final owned storage where buffer layout/lifetime permits. Test noncontiguous buffers, conversion failures and caller mutation isolation; avoid replacing a safe copy with mutable aliasing. |
| B14 | Native eager abs builds a mask, negative tensor and where output (`tensor.rs:1323`), allocating three array results for O(N) work. | Fuse the existing comparison/negation selection into one pass. Verify NaN and signed-zero behavior explicitly; replacing it with generic `f64::abs` is not automatically equivalent. |

#### Device runtime candidates requiring profiling

| ID | Original audit evidence/cost | Proposed improvement and verification |
| --- | --- | --- |
| C1 | CUDA retained Adam `loss()` executes the shared loss-plus-gradient plan (`tensor_trace.rs:7996`, `cuda.rs:1616`), discarding gradients. MLX `loss()` lowers the shared graph but lazily evaluates only the requested loss (`mlx.rs:331`); it is not evidence of MLX GPU backward work. | Cache a forward-only loss plan, preserving current parameters/latest batch and retained input bindings. Measure loss separately from step, check unchanged state and loss parity on both devices. |
| C2 | CUDA axis reduction assigns one output thread a serial loop over the reduction extent (`cuda.rs:6878`). Work stays O(N), but a few outputs leave little parallelism. | Evaluate warp/block/two-stage reductions for long axes and retain the current path for small reductions. Verify reduction tolerances, extreme values and small/large shapes; tree reductions alter summation order. |
| C3 | CUDA free buffers are pooled by exact element count (`cuda.rs:1462`, `1513`). A fixed graph is bounded, but retained free storage across many sizes can substantially exceed peak live tensor bytes. | Profile size histograms/high-water marks; compare planned slots, capacity classes or bounded eviction. Measure both resident bytes and repeated-call allocation/time before changing retention policy. |
| C4 | CUDA Solve allocates factor/RHS/pivot/status/workspace storage per call and reads status back (`cuda.rs:3663`, `3770`). | Cache appropriately sized solver workspaces per plan/context. Keep solver error observability and stream ordering; separately measure allocation, factorization and status synchronization. |
| C5 | MLX region execution evaluates body outputs across iterations; reverse regions retain carry tapes (`mlx.rs:1127`, `1622`). Fewer host dispatch/eval boundaries may improve short bodies; batching more lazy work can increase memory. | Measure loop-length/body-size curves and bounded execution windows. CUDA and MLX VJP loops already use a forward tape plus reverse traversal; this review did not find quadratic prefix replay. Preserve lazy branch and failure behavior. |
| C6 | Distributed host shard copying is fixed, but replicated transfers, per-gradient collectives, readback and host optimizer updates remain workflow costs. | Profile gradient-size/count and upload/collective/readback intervals; evaluate packed gradient collectives and a retained distributed optimizer only for measured workloads. Account for packing memory, reduction order and current per-parameter output contracts. |

CUDA matmul already has specialized/tiled paths and fusion epilogues; no claim
that every product uses a naive kernel is made. MLX delegates dense kernels to
its runtime. Shape-specific dispatch, launch count, register pressure and library
selection are profiling candidates. MLX host `Vec<Array>` references during
lowering do not prove that all graph GPU buffers are simultaneously resident.

#### Python transforms, cache and smaller candidates

| ID | Original audit evidence/cost | Proposed improvement and verification |
| --- | --- | --- |
| D1 | Python dense Jacobian remains input-direction forward AD (`_transforms.py:1493`). Chunking bounds basis workspace, but derivative work is O(P*C_f) and dense comparison basis work can be O(P^2), even for Q much smaller than P. | Evaluate reverse directions for Q<P, structured seeds or matrix-free JVP/VJP when callers need products. Dense Hessian/Jacobian outputs still require P^2/PQ storage. Preserve pytree blocks, bool policy and nested transforms; compare direction-dependent rounding. |
| D2 | Python VJP pullbacks recompute the primal on each call and keep no residuals (`_transforms.py:1101`, `1199`). | Optional/internal residual reuse can save repeated C_f work at the price of retained tape memory. Benchmark one versus many cotangents; keep nested tracing and captured-array snapshots correct. Recomputation can be the appropriate low-memory choice. |
| D3 | `jit` cache and independently constructed lower/compile objects do not share entries (`_transforms.py:863`, `928`). The local identical-signature probe records three traces and three compilations. One Lowered object already reuses its compiled program. | Share only snapshot-equivalent staged/compiled programs or make reuse explicit internally. Test changing captured constants between lower calls; cache-key equality alone does not establish equal closure snapshots. |
| D4 | Legacy native batch execution clones cached plans on hot calls (`tensor_trace.rs:4554`, `4659`); numeric plan AD also clones nodes via `as_ir` (`tensor_ir.rs:11622`). | Borrow immutable plans or share them with Arc, and operate directly on frozen nodes. Verify cache growth limits, locks and repeated/concurrent calls; target O(V) metadata traffic separately from tensor work. |
| D5 | Adam validates/flatten parameters repeatedly; dictionary pytrees sort keys per traversal (`optim.py:87`, `tree.py:34`). Distributed hot calls also recompute names using `inspect.signature` (`distributed.py:81`, `_transforms.py:282`). | Reuse validated leaf structure/input names per cached signature while still validating live leaves and mutation-sensitive container structure. Measure many-small-leaf workloads; avoid identity-only caches for mutable dict/list arguments. |
| D6 | Pure SGD builds separate scale/subtract tensors, with a final dtype cast when needed (`optim.py:130`). | A fused native leaf update can reduce allocation volume while preserving F64 intermediate arithmetic and final rounding. Compare exact F32/F64 old behavior and input immutability; it remains O(N) work. |
| D7 | RK4 already has O(C) final-state storage, but allocates four derivative, three intermediate-state and one next-state vectors per step (`ode.rs:57`). Full trajectory storage is required by the separate trajectory-returning API. | Reuse an eight-buffer workspace, preserving arithmetic order and `FnMut` RHS calls. Count allocations and compare final/full trajectory values; do not claim the final-state path stores the trajectory. |
| D8 | The alpha-only Lotka-Volterra gradient calls the full four-parameter gradient then selects element zero (`models.rs:73`). Macro `forward_gradient!` similarly expands one body evaluation per active parameter (`quabla-macros/src/lib.rs:110`). | The alpha helper can call its single-parameter implementation. For macro/general gradients, benchmark vector-tangent or reverse approaches separately: removing repeated primal work does not remove the gradient's dimensional work, and source/code-size changes have tradeoffs. |
| D9 | Core input insertion checks names by scanning prior nodes; repeated insertion can cost O(I^2) for I inputs, and frozen shape/dtype lookup scans the graph (`tensor_ir.rs:4074`, `10913`, `10925`). | Add a name-to-node index where many-input profiles justify it. Verify duplicate-name diagnostics/order and captures. Index storage is extra O(I); small graphs may not benefit. |
| D10 | CSE formats allocated string keys including operation, dtype and weakness (`tensor_ir.rs:13325`, `5454`); dense node remapping uses hash maps. | Typed structural keys and dense vectors/bitsets can reduce metadata allocation. Existing fixed-rank passes remain expected O(V+E); measure cold time/allocation before claiming an improvement, and preserve weak typing, float bit identity and mutable-trace snapshots. |

Reverse-loop checkpointing is an additional explicit time/memory decision:
the current O(TC) carry tape can be exchanged for recomputation/checkpoints.
Fori forward execution already avoids that tape; Scan's returned O(TY) outputs
cannot be removed without changing its result contract. Eager array imports
intentionally copy caller-owned buffers; future sharing must protect the current
ownership and immutability contract. Changes to public dtype/storage/export
semantics, native IR operations or distributed training require a reviewed
compatibility plan before implementation.

#### Executed audit probes and next verification

| Local probe | Observed result |
| --- | --- |
| Same signature: two normal JIT calls, two compile calls on one Lowered object, then a second Lowered object | Three traces/compilations total; the second compile on the first Lowered reuses its program |
| Legacy sine chain, 100,000 elements, independent processes at depths 10/50 | Correct outputs; peak RSS increments 8,896,512 / 41,140,224 bytes |
| CPU cubic HVP, warm median of five calls, P=64/256/1024 | Native 0.269/2.757/39.725 ms; existing `jit(jvp(grad))` composition 0.0151/0.0253/0.0701 ms; both return exactly 3 in each coordinate |
| CUDA source generation on CPU, shared DAG depths 6/10/14/18 | Nodes 7/11/15/19; generated bytes 2,899/37,459/590,419/9,437,779; no GPU or NVRTC run |

Probe sources and JSON outputs were kept outside the repository. RSS includes
runtime/allocator effects; source bytes establish code-generation growth, not GPU speed; the HVP
timing compares two existing paths on one analytic fixture, not an implemented
replacement or a general speedup. The first legacy probe used the Rust-facing
method name instead of its Python `evaluate` wrapper and failed before execution;
the corrected script produced the successful observations above.

Suggested implementation order: A1, A2/A3, A4, B2, A5/A6, then backward
liveness/factor reuse and measured device-runtime candidates. Keep the larger
storage/checkpointing/AD-direction work as separate compatibility and Pareto
decisions. Every implementation needs a focused before/after cost probe plus
relevant numerical/API regression checks; the earlier 212-test and GPU/NCCL
results are previous validation, not newly executed tests for this audit.

#### Optimization implementation follow-up (2026-10-03)

The following changes implement selected candidates from the audit above.
They preserve the public API, graph schema, dtype policy and default settings.
The remaining candidates are still proposals; this batch does not establish
globally optimal memory use or scheduling.

| Candidate | Implemented scope and cost | Compatibility and limits |
| --- | --- | --- |
| A1, partial | Whole-plan and fusion-region CUDA elementwise code now emits one scalar SSA temporary per reachable node. Emitted statement/reference counts are linear in reachable nodes/edges; traversal also visits the full node list. Source bytes include identifier lengths and broadcast offset expressions. | `where` keeps ternary value selection; its pure CUDA operand calculations can execute before selection. An unused NaN does not propagate through selection, and checked-domain operators remain excluded. Lazy `Cond` regions are unchanged. Fori/Scan body expression builders remain a separate candidate. |
| A2/A3, partial | Smooth F64 native HVP uses forward-over-reverse, O(C_f) derivative work instead of P mixed-dual evaluations. Hessian construction reuses one transformed graph for P columns, O(P*C_f), and writes directly into the necessary P-by-P result. | F32, casts, checked/singular/discontinuous operators and exceptional arithmetic retain the old mixed-dual route. Conservative bounds protect against intermediate derivative overflow even when a symbolic zero would hide it. F64 derivative arithmetic can reorder; verification uses numerical tolerance rather than a bitwise claim. The HVP transform is still constructed per invocation. |
| A4 | Core and native eager solve detect exact finite lower/upper triangular structure and use substitution: O(n^2*k), without the LU factor scratch matrix. Existing traced triangular patterns and reverse transpose solves benefit on CPU. | General, zero-diagonal, non-finite and overflowing systems use the existing pivoted LU route. Mask/transpose temporaries and GPU lowering are unchanged. Finite results are compared with LU and residuals using tolerance. |
| A6 | Fusion user-list construction deduplicates repeated operands using the last visited user ID: O(V+E) for this construction step. | Region boundaries and retained outputs stay covered by fan-out/duplicate-operand tests. Other fusion passes have their own costs. |
| B3, partial | Numeric reverse execution takes and releases processed non-input cotangents. Scan sibling seeds are taken jointly. A chain no longer retains every cotangent buffer. | Input gradients, duplicate-output accumulation and joint Scan seeds are preserved. The primal tape still occupies O(V*N); this change does not make the entire reverse pass O(N) space. |
| B6, partial | Native eager Tensor, legacy Matrix and static Tensor2 matmul traverse contiguous RHS/output rows while preserving each output's inner-index accumulation order. | Work remains O(m*k*n), with the same output allocation and final dtype rounding. Core DynamicTensor matmul already used this traversal; larger blocked/BLAS changes remain separate. |
| B14 | Native eager abs constructs one result buffer instead of mask, negation and selection buffers. | Signed zero, NaN quieting, F32 intermediate/final rounding, bool errors and weak-to-strong behavior preserve the old implementation. |
| D7/D8 | RK4 reuses five workspace buffers and updates the state in place; alpha-only model differentiation seeds one parameter rather than all four. | Callback order, partially written derivatives, arithmetic order and zero-step behavior are covered by tests. Final-state storage is O(C); a requested trajectory still needs O(T*C). |

Changed production files are `quabla-core/src/tensor_ir.rs`, `tensor.rs`, `ode.rs`,
`models.rs`, and `quabla-python/src/tensor.rs`, `matrix.rs`. Regression coverage
was added to the existing CPU memory and ODE suites, private native/core unit
tests, and `cuda_codegen_memory.rs` / `higher_order_optimization.rs`.

Measurements and command logs were kept outside the repository; the original
audit probes and baseline results were preserved. CPU probes use release builds on Apple M3/macOS,
seven warmed timing samples (five for HVP) and fresh processes for RSS deltas.
RSS is the increase in process high-water memory for the specified call,
not the total process memory or a portable allocator metric. GPU checks use
the single-GPU Linux host's GTX 1660 SUPER, CUDA 13.1 and opt-in runtime tests.
No new two-GPU NCCL result is claimed for this batch.

Archived local verification for the October 3 batch records 209 passing Rust
tests and the rebuilt extension's cubic HVP median of 0.103208 ms at P=1024.
The final shared-DAG source probe records 2,022 bytes at depth 18 (19 nodes).
These are recorded October 3 measurements, not new GPU timing results.

#### First continuation measurements (2026-10-04, archived batch)

This continuation preserves Python interfaces, IR operations, dtype defaults,
and numerical/error contracts. Each row describes its executed scope; the
audit's other proposals and profiling questions remain open.

| Candidate | Implemented scope | Compatibility and limits |
| --- | --- | --- |
| B7, partial | Finite dense CPU mixed Solve factors the coefficient matrix once and replays elimination for value, first, second, and mixed RHS. | Historical pivot swaps/subtractions and back substitution keep their order. Triangular, singular, non-finite, or overflowing cases use the prior path. Dense work is still O(n^3+n^2*k); this removes three factorizations, not an asymptotic order. Factor scratch remains O(n^2). JVP/VJP factor reuse is a separate candidate. |
| B9 | Eager Gather writes contiguous inner blocks directly into one final buffer. Multi-axis indexing composes layouts before a single materialization. | Repeated indices, axis normalization, dropped axes, scalar outputs, dtype/weakness, and error behavior are covered. Empty indexing still shares the original tensor. Coordinate mapping in final index materialization remains rank-dependent. |
| B13, partial | C-contiguous typed buffer import widens directly into final owned F64 storage, removing the typed temporary for F32/integer sources. | PyO3 validates format/alignment and pins the exporter during copying. Caller writes cannot affect imported values. Non-contiguous imports and the bool-specific route keep their existing gathering paths; resident host storage is still F64. |
| A8 | Numeric Scan JVP shares one primal/tangent traversal for carry and output each step. | The original F64 derivative intermediates and final-output rounding are preserved, including Bool zero tangents, captures and nested Scan/Fori/Cond. Returned trajectory storage and per-step graph/capture copies remain. |
| B5, partial | Core equal-shape and scalar elementwise arithmetic uses contiguous iteration. Axis reduction traverses outer/reduced/inner blocks. | Broadcasting is validated before dispatch. Reduction visits values in the original source order, retaining +0 initialization and multiplication before addition. These paths remove per-element rank-coordinate work; arbitrary broadcasts and native eager reductions remain separate. |
| B10 | Legacy Matrix pure-forward execution releases each matrix after its final operand use and moves the root result. | Every node before the root still executes, preserving eager errors. Duplicate operands and fan-out are counted. JVP/VJP keep their full primal tapes. Live matrix storage follows the graph's frontier; O(V) metadata, caller/input copies, and concat operand copies remain. |

Sources and raw measurements, including an unchanged copy of the previous
native extension, were kept outside the repository. Comparisons use the same Python 3.14.7 interpreter
and Apple M3 CPU. Native timings exclude interpreter startup and input creation.
RSS is a fresh-process high-water increase, not the size of live tensor buffers.
Scan's allocation counter measures cumulative allocation requests, not peak RSS.

Integrated verification passed: 228 Rust tests and nine Python test scripts,
release native-extension build with MLX enabled, workspace formatting, default
and MLX Clippy checks, and Ruff checks. CUDA execution was not measured in this
batch. The legacy before/after compatibility comparison matched all 107 output
bit-pattern and error records.

| Controlled CPU case | Before | After | Observed benefit |
| --- | --- | --- | --- |
| Gather, 256x128x32 input | 1.159 ms | 0.089 ms | 13.08x; RSS increase 8.33 MB to 4.28 MB |
| Multi-axis indexing, 1,048,576 elements | 5.894 ms | 1.966 ms | 3.00x; RSS increase 16.84 MB to 8.47 MB |
| Contiguous F32 import, 1,048,576 elements | 0.336 ms | 0.239 ms | 1.40x; RSS increase 12.78 MB to 8.55 MB |
| Mixed Solve, n=192, four RHS columns | 2.931 ms | 1.464 ms | 2.00x; smaller tested systems gave 1.33x to 1.55x |
| Scan JVP, 512 lanes, 24 shared layers, 20 steps | 2.637 ms | 1.370 ms | 1.93x; cumulative requested heap bytes 30,198,888 to 15,951,248 |
| Core contiguous elementwise, 1,048,576 elements | 8.90 to 10.62 ms | 0.40 to 0.54 ms | 19.71x to 23.81x across equal-shape and scalar cases |
| Core axis reductions, shape 16x64x1024 | 3.63 to 3.92 ms | 0.14 to 0.75 ms | 5.25x to 25.37x across the three axes |
| Legacy graph forward, depth 50, 100,000 lanes | 9.347 ms | 7.494 ms | RSS increase 41.21 MB to 0.95 MB; total peak 69.12 MB to 28.83 MB |

Native, Solve, Scan, and kernel timings use 12 balanced rounds; legacy timings
use eight. MB denotes decimal megabytes. These are case-specific medians;
speedup ratios across stages cannot be multiplied without a workload model.
Scan uses a two-traversal reference in the same binary; it measures traversal
sharing rather than a full old-extension comparison. The fixtures,
individual measurements and source hashes were recorded outside the
repository.

#### Integrated optimization status (2026-10-05)

B1 and B2 are approved and implemented. Core and native tensors now share
immutable typed storage: F64 uses 8N bytes, F32 uses 4N bytes, and Bool uses N
bytes, excluding shape and ownership metadata. `DynamicTensor::data()` returns
`Cow<[f64]>`: F64 borrows its storage; narrow types widen only on request.
`storage()` provides typed access, and consuming `into_parts()` transfers
shape, storage, and dtype. Python interfaces are unchanged. Shape-only views
share their backing buffer; numerical kernels preserve F64 intermediates and
rounding at the original boundaries. Reverse tapes keep required aliases alive.

| Candidates | Current executed scope | Remaining work or decision |
| --- | --- | --- |
| A1 | Flat CUDA SSA emission includes control-flow regions. | Integrated release CUDA tests and Linux CUDA/NCCL compilation pass; no current two-GPU parity claim. |
| A2/A3/A4/A6/A8 | Earlier higher-order, triangular, fusion-user, and Scan changes remain integrated. | Preserve numerical fallbacks; do not multiply local ratios. |
| A5/A7 | Constant folding releases last-use values; single symbolic transforms prune unused nodes before differentiation. | Full multi-VJP primal-map contract still retains all requested primals. |
| A9/B1/B2/B12 | Typed immutable shared storage, consuming outputs, and shape-only aliases. | Metadata copies and transient conversion buffers are separate costs. |
| B3/B4 | Reverse evaluation retains only active-rule primals and releases processed primals/cotangents. Last-use F64/F32 unary and same-shape/scalar binary buffers are reused with immutable alias protection. | CPU and MLX first-order and forward-over-reverse Fori/Scan now use block checkpoints; CUDA covers independent lanes with same-shape captures. Short loops and CUDA broadcast captures preserve the full-tape fallback. Owned F64 gradient additions reuse unique storage; NaN payload cases retain the original kernel. |
| B5/B6/B9/B10/B11/B13/B14 | Contiguous paths and stack-only carry offsets for core/native binary, broadcast-to, and where; hoisted typed matmul; gather/indexing; legacy liveness/shared Jacobian primals; direct typed buffer imports; fused abs. | Native unary writes directly to dtype-sized output. Integer-to-F32/Bool preserves conversion through F64 per lane without a widened array. Eligible finite legacy numeric P<Q graphs use live single-direction forward tangents; general arithmetic and Q<=P retain reverse. |
| B7 | Mixed Solve and direct-input single-Solve JVP reuse finite dense factors. | Eight balanced complete JVP rounds at n=8/64/192 give 1.247/1.390/1.649x with slightly lower incremental peak heap. General graphs and VJP preserve their previous numerical path. |
| B8 | Approved native Cholesky is integrated on CPU/CUDA/Metal with bounded logical-F64 first/second-order IR, leading-batch support, O(n³) arithmetic and O(n²) numerical state. | F32 AD, third or higher derivatives, and exceptional CPU symbolic cases preserve scalar expansion. Native and reference composition tests pass; final Python/heap/device measurements were recorded outside the repository. |
| C1/C4 | Forward-only retained CUDA and MLX loss; CUDA Solve retains only the current shape workspace. | Runtime checks pass; MLX loss probe improves 0.379 to 0.358 ms. Multi-shape Solve cache stays at one entry rather than four, with identical output checksums. Same-shape reuse remains; no reliable Solve timing gain is claimed. |
| C2 | Approved SumAxis/MeanAxis block reduction is integrated for axis length >=256. Each output uses 256 threads, 2,052 shared bytes, and no global scratch. | Work stays O(N); normal per-block dependency depth becomes O(axis_length/256 + log 256). Small reductions and finite overflow-risk inputs preserve the serial path. Full CUDA gates pass; complete upload/execute/export speedups are 1.032–6.050x on the measured long-axis fixtures. |
| C3 | Bounded CUDA free-buffer pool adopted for memory priority. | Eight balanced release rounds: pooled bytes 10,321,920 to 782,336, latency 0.900 to 1.495 ms. This memory saving has a measured latency cost. |
| C5 | Pure MLX loop windows profiled; per-step evaluation remains the default. | Window 16 speeds the fixtures by 3.7–9.1x but fresh-process peak RSS rises by 974,848 bytes. Batching is not adopted for the memory target. |
| C6 | Existing distributed execution preserved. | Packed collectives add a packed buffer and can change reduction order; current one-GPU validation cannot establish two-GPU numerical parity. |
| D1 | Approved adaptive Jacobian selects reverse mode for eligible Q<P outputs, with bounded output seed batches. F64 outputs in graphs containing F32 nodes keep forward mode. | Pytree, dtype, nested-transform, and analytic regressions pass. Approved tolerances: F64 1e-12 + 1e-12*abs(reference); F32 1e-5 + 1e-5*abs(reference). Dense O(PQ) result storage remains. |
| D2/D3 | VJP recomputation and snapshot-local caches retained for memory and correctness. | Sixteen pullbacks cost approximately sixteen single calls; caching residuals would retain extra buffers. Same-signature snapshots with different captured constants produce different values, so signature-only cache sharing is unsafe. |
| D4/D5/D6/D9 | Shared immutable graph snapshots, Adam initialization leaf-structure reuse, fused SGD, and input-name index. Trainer batch specifications retain shape/dtype metadata rather than the initial Tensor. | Additional mutable-pytree caches would retain metadata and require container revalidation; no unmeasured cache is adopted. |
| D7/D8 | Earlier ODE workspace and single-alpha changes remain integrated. | Final default, MLX, and CUDA workspace regression suites pass. |
| D10 | Compact typed CSE keys and adaptive dense/sparse remapping preserve operation identity and exact scalar bits. | At 20,000 nodes, twelve paired release rounds reduce incremental peak heap from 23,057,523 to 21,404,557 bytes and median compilation from 14.567 to 8.987 ms; expected O(V+E) remains. |

The pre-October-5-continuation CPU comparison uses the retained October 3
extension, which already contains earlier optimizations, against the release
extension recorded in the October 4 artifacts. Twelve balanced fresh-process rounds include construction,
execution, and exports; warm medians exclude the first invocation:

| Complete CPU workload | October 3 baseline (ms) | October 4 snapshot (ms) | Measured speedup |
| --- | ---: | ---: | ---: |
| F32 eager pipeline | 0.212834 | 0.169198 | 1.258x |
| Twelve-step training | 0.642645 | 0.427802 | 1.502x |
| Legacy Jacobians | 0.345459 | 0.138427 | 2.496x |
| Scalar Jacobian, P=2048/Q=1 | 71.350073 | 0.110761 | 644.183x |
| Vector Jacobian, P=128/Q=16 | 0.357781 | 0.097531 | 3.668x |

All five outputs match bitwise. Raw samples and extension hashes were
recorded outside the repository, as were the raw records of every measurement
in this section. A combined invocation of these five workloads measures
72.961823 to 0.973969 ms (74.912x), also bitwise equal. The scalar Jacobian
dominates that mix; local ratios are never multiplied.

The expanded combined suite includes those five workloads, Fori VJP T512/N512,
Scan VJP T64/N512 with outputs, Adam initialization of 2,048 four-lane leaves,
i64-to-F32/Bool imports of 250,000 lanes, cold tracing/compilation/execution of
2,000 sin nodes, and numeric legacy P2/Q512 expansion. One invocation of each
runs sequentially in one process, including checkpoint replay costs. Twelve
balanced fresh-process rounds measure 145.884573 to 80.316355 ms (1.816x).
Peak process RSS decreases from 57,483,264 to 52,781,056 bytes (8.18 percent);
RSS increment decreases from 16,637,952 to 11,886,592 bytes. Maximum absolute
output difference is 4.44e-16, within the approved F64 tolerance. This is not
an original-pre-optimization-to-final project-wide ratio; each ratio applies
to its specified mix. Training alone has slightly higher RSS, so no universal
memory reduction is claimed. GPU results are separate host measurements.

Six balanced resident-array rounds write nonzero storage pages: F32 process RSS
increment falls from 101,072,896 to 59,269,120 bytes, Bool from 101,072,896 to
21,151,744, and F64 is effectively unchanged. These process peaks include
factory transients and allocator overhead; they are not exact payload sizes.
The two-million-lane native F32 sin+tanh pipeline reduces process peak increment
from 24,231,936 to 8,142,848 bytes and improves 14.497 to 12.810 ms (1.132x).
Explicit i64 buffer import to F32/Bool improves 2.486/2.457x while reducing
import peak increments by 66.5/88.3 percent. Complete-operation broadcast
probes reduce rank-dependent decode costs without adding heap coordinates;
worst-case O(N*rank) remains for frequent carries/singleton axes.

Selective VJP plus last-use reuse reduces incremental live heap in the
128-operation linear fixture from 34,097,840 to 799,424 bytes. Physical
storage remains 8N/4N/N for F64/F32/Bool. Allocator probes and fresh-process
RSS measure different costs and are reported separately.

Legacy numeric Jacobians choose P live forward sweeps for eligible P<Q graphs,
rather than Q reverse sweeps. Sin/Cos/Tanh, constant Add/Sub, shape operations,
and comparison derivative stops are eligible; other arithmetic and nonfinite
primals preserve reverse. Tangent slots release at last use, and candidate
errors/nonfinite results fall back to reverse using the existing primals.
Dense result storage remains O(PQ). Six balanced complete factory/call/export
rounds give 4.759 to 0.129 ms (36.85x) for P2/Q512 and 54.592 to 2.672 ms
(20.43x) for P1/Q128/depth2048; maximum errors are 4.44e-16 and 5.72e-35.
The Q<=P control is bitwise equal. The long-chain process RSS increment is
unchanged, so no general RSS saving is claimed. The fixed-size forward-gradient
macro keeps its documented callback count and scalar Dual API: packing all
parameter directions would increase live derivative storage and change that
contract.

Owned same-shape F64 gradient accumulation reuses unique storage, cloning
shared seeds only when required. Narrow/broadcast cases and NaN operands
retain the original addition kernel, preserving dtype, errors and NaN payloads
on macOS and Linux. Twelve balanced complete VJP rounds at N=32768 and
fanout depth 8/128 give 1.105–1.115x, with 262,184 fewer peak live heap bytes
and fewer allocations. NaN compatibility scanning reduces the earlier
pre-guard speedup. An earlier comparison that shared one Cargo target was
invalid because both labels ran the same binary; the final before/after
binaries are distinct, as their SHA-256 hashes confirm.

CPU loop checkpoints retain approximately ceil(T/B)+B carry entries with
B=ceil(sqrt(T)), reducing carry-tape space from O(TC) to O(sqrt(T)C) while
preserving O(T) work for a fixed body. Replayed steps retain the original dtype
rounding; reverse gradients accumulate in their original descending order.
Public full-tape APIs remain unchanged. Public Scan still returns O(TY) output
storage; internal gradient-only consumers omit an unused stacked primal output.
Six balanced CPU F64 rounds at T=4096/N=512 reduce incremental live heap from
17,209,763 to 568,380 bytes (96.7 percent), while complete VJP latency rises
from 248.264 to 439.357 ms (1.770x). This implements the memory priority and
has a measured replay cost, rather than a speedup.

Real Metal bitwise tests compare first-order and forward-over-reverse Fori/Scan
against full-tape execution at T=1/8/17/64. Block boundaries evaluate lazy
capture-gradient accumulation, avoiding an unbounded accumulation history.
Six balanced, separate fresh-process rounds at N=8192 reduce absolute active
Metal peaks from 2,457,620 to 851,988 bytes at T=64 and 17,137,688 to 1,835,024
at T=512. Complete VJP latency rises from 17.235 to 23.860 ms and 144.945 to
198.344 ms. Allocator cache and Python RSS are separate costs. Earlier
incremental Metal measurements are unsuitable because their baseline moves
while prior command buffers release; the fresh-process absolute-peak report
supersedes them.

CUDA checkpoints retain O(sqrt(T)C) carry tape for independent lanes and
same-shape captures, including paired carry/tangent state for higher-order AD.
Six balanced fresh-process rounds at N=8192 reduce actual carry-tape allocations
from 2,129,920 to 524,288 bytes at T=64 and 16,809,984 to 1,507,328 bytes at
T=512; paired directional state doubles both counts. These are tape bytes,
not total device peaks. At T=512, complete Fori/Scan VJP and VJP-JVP execution
costs rise by 17–43 percent. Compilation is excluded; host upload and all
sibling gradient downloads are included, and each full/checkpoint pair checks
all result bits. Required Scan outputs/cotangents remain O(TY). The run
produced 96 raw records.

Trainer's replacement-batch regression first reproduced retention of the original
Tensor through its shape/dtype specification; metadata-only specifications now
release that alias. Twelve balanced complete Adam initialization rounds with
2,048 four-lane leaves use the same native extension and reduce 2.498 to
1.984 ms (1.259x), with three flatten traversals reduced to one. Independent
F64 zero moments and optimizer behavior are preserved.

The pre-continuation root verification passed 300 Rust tests in the MLX workspace, default
and MLX all-target Clippy, fmt, Ruff, all nine required Python scripts plus
the dedicated legacy Jacobian regression, and the opt-in Metal Python matrix.
The rebuilt release extension is used for the final complete-workload reports.
The root CUDA release suite passes 259 tests on GTX 1660 SUPER; Linux
CUDA/NCCL all-target Clippy also passes after the final legacy test additions.
The first Linux gate exposed two-NaN payload differences in owned addition;
NaN operands now use the original kernel and the full gate passes. Those October 4 gates did not establish NCCL numerical parity for that snapshot.

The October 4 adoptable checklist changes were integrated and measured.
The October 5 continuation explicitly approves C2 tree-reduction numerical
order, B8 native Cholesky IR/schema changes, and C6 two-GPU validation.
Two-GPU NCCL parity has not been rerun on the final integrated source; the
earlier two-GPU results remain valid only for their recorded source snapshots.
Candidates rejected for extra retained memory, contract changes or missing
two-GPU validation remain documented above. No globally optimal
memory/scheduling claim is made.

JAX can be summarized as:

```text
JAX = NumPy-like arrays + composable autodiff + XLA compilation
```

This project should not compete by rebuilding that stack one feature at a time.
The Rust-specific bet is to move more correctness and transformation work into
types, macros, compiler passes, and explicit backend contracts.

## Layer 1: Autodiff Transformation

Current state:

- Scalar forward-mode AD with `Dual`.
- Model gradients are computed by seeding one parameter at a time.
- `quabla-core::tensor_ir::TensorIr` now provides a pure-Rust dynamic rank-N IR
  for input, broadcasted add/multiply, and scalar sum; it evaluates on CPU and
  generates both direct JVPs and VJPs that reduce cotangents over broadcast
  axes.

Target direction:

- Represent differentiable programs in a transformable IR instead of only
  runtime operator overloading.
- Support Jacobians, Hessians, and mixed higher-order derivatives needed by
  PINNs and inverse problems.
- Explore source-to-source AD through procedural macros before attempting a
  compiler plugin.
- Keep an LLVM/Enzyme-style path as a possible backend for low-level kernels.

Key risk:

- Loops, branches, mutation, and borrowing make source transformation hard.
  Rust's borrow checker is an advantage only if the AD transform preserves
  aliasing and lifetime semantics instead of bypassing them.

Near-term milestone:

- Extend `TensorIr` with reductions, reshape, matmul, and unary primitives.
- Generate forward-mode transforms from the same IR.
- Compare generated derivatives against finite differences.

## Layer 2: Typed Arrays And Layout

Current state:

- `Tensor2<ROWS, COLS>` stores row-major `f64` data.
- Matrix multiplication enforces the inner dimension at compile time:
  `(M, K) x (K, N) -> (M, N)`.
- Python-facing `Tensor` provides eager rank-N contiguous row-major storage,
  `zeros`/`ones`/`full`/`arange`/`linspace`/`eye` creation, rank-N concat,
  positive runtime shape validation, trailing-axis broadcasting for
  add/subtract/multiply/divide and scalar powers, NumPy-style batched matmul,
  permutation-validated transpose, global or single-axis sum/mean, and
  common elementwise math, `gt(...)` masks, broadcasted `where`, and
  materialized `broadcast_to`, and element-count-preserving reshape.
  `Tensor.slice(...)` creates a zero-copy,
  read-only `TensorView` with explicit stride and offset metadata; the shared
  backing allocation is immutable, so no mutable aliasing is exposed.

Target direction:

- Use const generics for static shapes where possible.
- Provide dynamic-shape tensors for data-dependent workloads.
- Make layout explicit: contiguous, strided, transposed, sliced, and device
  views should be different states with clear aliasing rules.
- Avoid promising zero-cost slicing until the stride/view representation is
  proven in tests and benchmarks.

Key risk:

- Static shapes alone are not enough. Real SciML workloads need batching,
  adaptive grids, irregular observation sets, and dynamic shapes.

Near-term milestone:

- Carry rank-N tensor operations and stride-aware views into a generic trace
  representation without weakening their shape and aliasing invariants.
- Add compile-time checked `matmul` for static tensors.
- Extend dynamic tensor operations with reductions and explicit stride-aware
  views before carrying them into the trace IR.

## Layer 3: Backend, Compilation, And Sharding

Current state:

- CPU loops remain the default execution path, with frozen execution plans but
  no general machine-code JIT.
- Linux has an NVRTC CUDA path for rank-N plans, including device-resident
  symbolic VJP/JVP evaluation and parameter updates.
- Apple silicon has an MLX backend that executes supported frozen Tensor IR
  plans, including reverse-mode and device-resident Adam training (P6), on
  `StreamOrDevice::gpu()`, and reports unsupported nodes rather than falling
  back to CPU.
- The optional Linux `cuda-nccl` feature provides single-node data-parallel
  execution with NCCL all-reduce, verified on two GPUs (P7); there is no
  multi-node or tensor-parallel runtime.

Target direction:

- Separate frontend tensor expressions from backend execution.
- Start with a CPU backend contract.
- Add a lowering path to a kernel IR before choosing LLVM, MLIR, CUDA, WGSL, or
  another backend.
- Model sharding as a typed placement/layout property, not as an afterthought.
- Use `Send` and `Sync` boundaries to make cross-device execution explicit.

Key risk:

- Distributed sharding is a systems project. It should not be started before the
  tensor IR and backend contract are stable.

Near-term milestone:

- Define backend traits for allocation, copy, elementwise map, and matmul.
- Add a CPU backend implementation.
- Add an execution trace that can later be lowered or partitioned.

## Product Direction

The scientific learning workflow should be:

```text
Python API -> Rust Matrix/Tensor objects -> trace/IR -> AD transform
           -> optimizer or backend lowering -> CPU/GPU/sharded execution
```

ODE/PDE solvers are one kind of differentiable model, not the whole product.

### Compiler Facade

The public compiler lifecycle is now explicit instead of being distributed
across tracing and backend-specific execution-plan classes:

```text
frontend trace -> Program -> JVP/VJP transform -> compile(target) -> Executable
```

`quabla_core::compiler` owns the Rust-facing `QuablaCompiler`, `QuablaProgram`,
and `QuablaExecutable` contracts. The Python bridge exposes the same lifecycle
as `quabla.Compiler`, `Program`, and `Executable`. Existing function-specific
helpers remain compatibility APIs while they migrate internally.

This facade is a boundary, not a claim that every IR operation lowers on every
target. It distinguishes build availability from per-program support: CUDA and
MLX can reject unsupported graphs explicitly, and incomplete structured GPU
HVP lowering must not be represented as a general supported capability.

Implementation status (2026-07-29): the core facade now owns immutable
single-output programs and their `freeze`, symbolic tangent-input JVP, symbolic
VJP, target selection, and execution contracts. `QuablaTarget::is_built()` is a
compile-time feature/platform check; it is deliberately separate from
per-program lowering validation. The Python facade traces rank-N functions,
exposes `Program.jvp(input_name)` and `Program.vjp(cotangent_name)`, then
returns a uniform `Executable` for CPU, CUDA, or MLX. It does not yet replace
the legacy 2D `TraceGraph` API or make the existing function-specific helpers
delegate internally.

Verification boundary (2026-07-29): the core lifecycle test checks CPU
execution and generated JVP/VJP program structure. The installed PyO3
extension test checks Python `Compiler.trace`, `Program.compile`, execution,
and re-compilation of a VJP result. On a single-GPU Linux host (GTX 1660
SUPER), the same facade test compiles a nonlinear scalar loss, its symbolic
coordinate JVP, and one symbolic VJP program per input through CUDA, then
compares all results to CPU at `1e-5` tolerance. The same GTX 1660 SUPER run
validates a nonlinear fixed-bound Scan HVP through symbolic VJP then JVP:
CUDA's paired carry result matches CPU at `2e-5` tolerance. A second parity
case verifies an output reshape whose rank differs from the carry while
preserving element count. The full CUDA Python matrix executes both acceptance
paths and an indexed unequal-lane rejection path on that GPU; no Scan test is
left outside the matrix runner. Workspace tests and all-feature Clippy pass on
the local Apple-silicon host. MLX facade execution still requires a build with
the `mlx` feature and target-host validation for each supported operation set.

Implementation status (2026-07-29, continued): CUDA primal Scan separates
carry-lane and per-step-output-lane counts. For broadcast-compatible bodies,
each output lane replays the pure elementwise carry recurrence from its mapped
initial carry value, so a `[2, 1]` carry can emit `[2, 3]` outputs without CPU
fallback. First-order `ScanVjp` now supports the verifiable unequal-count
subset where the per-step output directly broadcasts a carry-shaped value: the
kernel aggregates all mapped output cotangents for a carry lane before running
the body VJP. GTX 1660 SUPER parity covers initial-carry and explicit-capture
gradients for that path. `ScanVjpJvp` now implements the same direct-broadcast
subset, aggregating both primal and tangent output cotangents before its
forward-over-reverse recurrence. It also accepts broadcast-compatible explicit
captures and atomically reduces their directional gradients. GTX 1660 SUPER
parity covers nonlinear HVPs for both same-shaped and broadcast captures.
General unequal-count output graphs remain explicit rejections, rather
than silently treating one output lane as one carry lane.

Helper migration plan (2026-09-27): every rank-N Python helper is classified
against the current single-output facade contract. Category (a) helpers
compile one traced output; migrated targets delegate through
`TensorTraceGraph::compile_{cpu,mlx,cuda}_plan`, which build a
`QuablaProgram` from a snapshot of the traced IR and call
`QuablaCompiler::compile_without_build_check`. `Program.compile(...)` and the
legacy `compile_cpu`/`compile_mlx`/`compile_cuda` methods share the same path.
`QuablaCompiler::compile` rejects a target missing from the build up front with
`<target> target is unavailable in this build`; the bridge skips only that
check, so the Python entrypoints keep their pre-facade errors in such a build
(CUDA fails at compile time and MLX on first execution, each with its backend's
build instructions) while the lowering stays in one place. The snapshot is one
extra IR clone per compilation: constructing a
`tensor_jit_fn` over a 2,000-step `tanh` chain moved from 2.38 ms to 2.55 ms
median on the local Apple-silicon host; execution is unchanged. CPU
derivative helpers keep evaluating their derivative with the frozen plan's
runtime AD (`TensorExecutionPlan::vjp`, `jvp`, `hessian_scalar`,
`hvp_scalar`); replacing that with symbolic `Program.vjp`/`jvp` would change
numerics and operation coverage, so it is not part of this migration.

| Category | Helpers | Status |
| --- | --- | --- |
| (a) CPU | `tensor_jit_fn`, `tensor_grad_scalar_fn`, `tensor_value_and_grad_fn`, `tensor_hessian_scalar_fn`, `tensor_hvp_scalar_fn`, `tensor_vjp_fn`, `tensor_jvp_fn`, `tensor_jacobian_fn`, `tensor_vmap_fn`, `tensor_vmap_vjp_fn`, `tensor_vmap_jvp_fn`, `tensor_vmap_hvp_scalar_fn`, `tensor_jit_batch_fn`, `tensor_value_and_grad_batch_fn` (per specialization), `tensor_cond_fn`, `tensor_cond_value_and_grad_fn`, `tensor_cond_jvp_fn` (one program per branch) | Delegates through the facade |
| (a) MLX | `tensor_vmap_mlx_fn` | Delegates through the facade |
| (a) CUDA | `tensor_jit_cuda_fn`, `tensor_vmap_cuda_fn`, `tensor_vmap_hvp_scalar_cuda_fn`, `tensor_jit_batch_cuda_fn` (per specialization) | Delegates through the facade |
| (b) MLX multi-output | `tensor_value_and_grad_mlx_fn`, `tensor_value_and_grad_batch_mlx_fn` (per specialization), `tensor_vmap_vjp_mlx_fn`, `tensor_vmap_jvp_mlx_fn`, `mlx_adam_loss_optimizer` | Delegates through the facade (`compile_many`); retained inputs stay in the bridge plan |
| (b) CUDA multi-output | `tensor_value_and_grad_cuda_fn`, `tensor_value_and_grad_batch_cuda_fn` (per specialization), `tensor_vmap_vjp_cuda_fn`, `tensor_vmap_jvp_cuda_fn`, `cuda_adam_vjp_optimizer`, `cuda_adam_loss_optimizer` | Delegates through the facade (`compile_many`); retained device buffers and Adam state stay in the bridge executors |
| (c) Separate | `tensor_value_and_grad_data_parallel_cuda_fn`, `cuda_adam_optimizer`, `cuda_adam_step`, trace-time region builders (`tensor_cond`, `tensor_fori_loop*`, `tensor_scan*`), eager graph evaluation methods, legacy 2D `TraceGraph` | Stays outside the facade |

Verification (2026-09-27): the local Apple-silicon host runs workspace tests
and Clippy with and without `quabla-core/mlx`, plus the Python matrix with and
without `QUABLA_MLX_TEST=1`. On a single-GPU Linux host (GTX 1660 SUPER),
`QUABLA_CUDA_TEST=1 cargo test -p quabla-core --features cuda`, workspace
Clippy with `quabla-core/cuda`, and the release CUDA extension's Python matrix
with and without `QUABLA_CUDA_TEST=1` pass after the migration; a
facade-routed `tensor_jit_cuda_fn` runs on the GPU (`backend == "cublas"`) and
matches CPU within `2e-8`.

Multi-output programs (2026-09-27): category (b) helpers freeze a value
together with selected gradients or a primal/tangent pair into one plan and
consume the frozen output node ids. `QuablaMultiOutputProgram::new(ir,
outputs)` records that output order next to, not inside, the single-output
`QuablaProgram`, whose API and symbolic JVP/VJP transforms are unchanged.
`QuablaCompiler::compile_many` freezes it through the same lowering step as
`compile` and returns a `QuablaMultiOutputExecutable`;
`compile_many_without_build_check` is the matching bypass, again reserved for
the Python compatibility helpers. Freezing prunes and deduplicates nodes, so
the executable owns the source-to-frozen remapping: `output_node_ids()[i]` is
the frozen id of program output `i`, and `execute` returns every output in
program order from one evaluation. Retained state stays with its executors:
MLX retained input arrays (`TensorMlxExecutionPlan`, `MlxAdamPlan`),
device-resident CUDA buffers, and Adam moments are not part of
`QuablaExecutable`. `into_executable()` hands those executors the backend plan
they already consume, which keeps one lowering path without redesigning
retained execution; moving retained state into the facade executable remains a
separate design step. Rust tests cover output order, frozen-id remapping,
rejection of empty or unknown outputs, the build-check contract, CPU parity
with per-output single-output compilation, and MLX and CUDA value-and-gradient
parity against CPU (`1e-5`, CUDA on the GTX 1660 SUPER).

Every category (b) helper now compiles through
`TensorTraceGraph::compile_multi_output_executable`, which builds the program
from the transformed IR and calls `compile_many_without_build_check`, then
reads its loss, gradient, value, and tangent node ids from the executable by
position; the bridge-side `compile_{cpu,mlx,cuda}_multi_plan` functions are
gone. In a build without the backend, the MLX helpers still construct and fail
on first execution and the CUDA helpers still fail at construction. A
transformed graph is moved into the program rather than cloned; only
`cuda_adam_vjp_optimizer`, whose graph is shared with Python VJP results,
snapshots it once. Interleaved runs of `examples/benchmark_pinn_mlx.py` before
and after the MLX switch (three each, medians) stay within run-to-run noise:
33.1 vs 33.3 ms compile and 0.815 vs 0.837 ms/step for the one-parameter
model, 5.49 vs 5.53 ms/step for the 3x64 tanh MLP, where single runs vary by
up to 0.12 ms/step. On the GTX 1660 SUPER, interleaved
`examples/benchmark_pinn_cuda.py` runs before and after the CUDA switch (four
each, medians) are 1.12 vs 1.14 ms/step and 327 vs 312 ms compile at the
default size, 1.17 vs 1.10 ms/step at `--width 64 --collocation 256`, with
single runs spanning 1.03 to 1.40 ms/step.

Category (c) helpers either target a replica set rather than one `QuablaTarget`
(data-parallel CUDA, which still freezes its value-and-gradient program with
`QuablaMultiOutputProgram::freeze` before `compile_data_parallel`), only
consume already-compiled plans (CUDA Adam), build IR regions during tracing
rather than executables (single-output `tensor_cond` and
`tensor_fori_loop_region` bodies still freeze through the facade CPU path),
interpret the mutable trace graph without freezing it, or belong to the
separate 2D tracer.

## Python Bridge

Current state:

- `quabla-python` exposes a minimal `Matrix` class through PyO3.
- Python can use Rust-owned matrix storage and call Rust add/sub/mul/div/powf/tanh/exp/log/sqrt/sin/cos/gt/where/concat/transpose/reshape/matmul/sum/mean.
- `TraceGraph` and `TraceMatrix` can record input, scalar constant, add,
  subtract, multiply, divide, greater-than, `where`, concat, powf, tanh, exp,
  log, sqrt, sin, cos, transpose, reshape, and matmul nodes, plus sum/mean reductions.
  Numeric scalar literals are supported for eager and traced add/subtract/multiply. Unary negation and
  division by numeric scalars lower to scalar multiplication; non-negative
  integer powers lower to repeated multiplication, while floating-point powers
  lower to a traced `powf` primitive with reverse-mode support for scalar
  exponents. Eager and traced elementwise add/subtract/multiply/divide support
  2D NumPy-style broadcasting where each axis is equal or one side is `1`,
  including reverse-mode cotangent reduction back to the original input shapes.
  Sum/mean reductions accept `axis=0` or `axis=1` and keep reduced rank as
  `(1, cols)` or `(rows, 1)`, with reverse-mode broadcast back to the input
  shape. Concat accepts `axis=0` or `axis=1`, validates non-concatenated
  dimensions, and splits reverse-mode cotangents back to each input. Reshape
  preserves row-major storage order and requires element count to stay unchanged.
  Eager `Tensor` supports general rank-N trailing-axis broadcasting and
  NumPy-style batched matmul, but it is intentionally not part of the legacy
  `TraceGraph`, AD transforms, or CPU plan lowering; those paths remain 2D
  `Matrix`-based. The separate `TensorTraceGraph` provides a rank-N
  add/multiply/sum AD subset.
- `quabla-core::TensorIr` is the rank-N compiler-core path, but it is not yet
  exposed through the legacy Python tracer. `trace_tensor(...)` now exposes its
  add/subtract/multiply/divide/matmul/rank-N-concat/rank-N-transpose/tanh/exp/sin/cos/sqrt/non-negative-integer-powi/log/reshape/global-or-single-axis-sum/mean subset through separate `TensorTraceGraph` and
  `TraceTensor` classes, with Python-facing CPU VJP and JVP evaluation. This
  preserves the established 2D Python transform API while migration proceeds
  operation by operation.
- `TraceTensor` add/subtract/multiply/divide accept numeric scalar literals on either
  side; these lower to rank-0 constant nodes with zero JVP tangent and no VJP
  input gradient.
- `TraceTensor.reshape(shape)` preserves row-major element order, validates the
  element count at trace time, and reshapes primal values and every AD tangent
  or cotangent consistently.
- `TraceTensor.slice(axis, start, stop)` selects one contiguous rank-N range.
  Negative axes are normalized at trace time; VJP reconstructs the original
  shape with a zero-padded internal node, so CPU and CUDA reverse graphs remain
  differentiable without host fallback.
- `TraceTensor.broadcast_to(shape)` records shape expansion instead of
  materializing it in Python. Its reverse transform reduces broadcast axes to
  the original shape; the CUDA backend lowers it to a rank-N indexed copy.
- `TraceTensor.mean()` performs a global mean reduction to a rank-0 scalar;
  its VJP broadcasts the scalar cotangent back over the input with `1/N`
  scaling.
- `TraceTensor.sum(axis=...)` and `TraceTensor.mean(axis=...)` reduce one
  normalized axis (including negative Python-style indices), remove that axis
  from the result shape, and expand cotangents back over the reduced axis.
- `TraceTensor.transpose(axes=None)` permutes every rank-N axis; omitting
  `axes` reverses the axis order, while a provided list must be a complete
  permutation. VJP uses the inverse permutation to restore input layout.
- `TraceTensor.log()` follows IEEE 754 like the devices: a zero input gives
  `-inf` and a negative input gives `NaN`.
- `TensorTraceGraph.hessian_scalar(output, input_name, inputs)` computes an
  exact dense Hessian for a scalar output and one named input using a mixed
  second-direction forward transform. Its O(n^2) basis evaluation is a
  correctness-oriented implementation, not yet an optimized HVP API.
- `TensorTraceGraph.hvp_scalar(...)` reuses the same exact mixed transform to
  compute a Hessian-vector product in O(n) basis evaluations without allocating
  the dense Hessian.
- `TensorTraceResult.symbolic_jvp(input_name)` is the composable coordinate
  derivative path: it constructs a new IR with primal/tangent pairs rather than
  returning a runtime tensor. Reapplying it constructs higher coordinate
  derivatives while leaving parameter inputs available to downstream VJP.
- `tensor_grad_scalar_fn(fn, input_specs)` traces and compiles a rank-N scalar
  loss once, returning a reusable Python gradient callable backed by the frozen
  plan rather than retracing on every invocation.
- `tensor_value_and_grad_fn(fn, input_specs)` traces and compiles a rank-N
  scalar loss once, returning its scalar value and named VJP gradients from one
  frozen-plan execution.
- `tensor_jit_fn(fn, input_specs)` traces and compiles a rank-N function once,
  returning a reusable primal callable backed by that same frozen-plan boundary.
- `tensor_jit_cuda_fn(fn, input_specs, device_ordinal=0)` traces and compiles
  a fixed-shape rank-N function to a reusable callable CUDA plan on Linux.
- `tensor_value_and_grad_cuda_fn(fn, input_specs, parameter_names,
  device_ordinal=0)` traces a scalar function once and compiles its primal
  value plus requested named gradients into one CUDA union plan. It owns its
  scalar cotangent input rather than requiring it from Python.
- `cuda_adam_loss_optimizer(loss, parameter_names, inputs, learning_rate, ...)`
  accepts a scalar `TraceTensor` or `TensorTraceResult`, creates the same union
  plan, and retains static inputs, parameters, and Adam state on device.
  `step()` has no parameter or gradient host readback; `loss()` is an explicit
  diagnostic readback.
- `tensor_vmap_fn(fn, input_specs, batch_size, in_axes=None, out_axis=0)`
  traces one fixed-size batched plan from per-example input shapes. It supports
  mapped normalized positive/negative axes, unmapped (`None`) inputs, and a
  requested output batch axis; `tensor_vmap_cuda_fn(...)` and
  `tensor_vmap_mlx_fn(...)` lower that same canonical trace. Batch extent
  remains static.
- `tensor_hessian_scalar_fn(fn, input_specs, input_name)` and
  `tensor_hvp_scalar_fn(fn, input_specs, input_name)` freeze a scalar rank-N
  trace for dense Hessian or Hessian-vector-product evaluation. Dense Hessian
  materialization remains a correctness-first O(n^2) CPU path.
- `tensor_vmap_hvp_scalar_fn(fn, input_specs, batch_size, input_name, ...)`
  and `tensor_vmap_hvp_scalar_cuda_fn(...)` trace one scalar loss per mapped
  example, differentiate their sum through symbolic VJP then runtime-tangent
  JVP, and return the selected mapped input's HVP in its declared public axis
  layout. The selected input must be mapped; unmapped parameter HVPs and MLX
  lowering are not part of this interface yet.
- `tensor_vjp_fn(fn, input_specs)` traces and compiles once, returning a
  reusable rank-N VJP callable whose output cotangent is supplied at invocation
  time.
- `tensor_jvp_fn(fn, input_specs)` traces and compiles once, returning a
  reusable rank-N JVP callable whose input tangents are supplied at invocation
  time.
- `tensor_jacobian_fn(fn, input_specs, input_name)` traces and compiles once,
  then constructs a dense Jacobian for one named input using direct JVP basis
  directions. It is correctness-oriented rather than a large-scale Jacobian
  materialization strategy.
- `TensorTraceGraph.compile_cpu(output_node_id)` produces an immutable
  `TensorCpuExecutionPlan` after output-reachability DCE, operand-id remap, and
  exact structural CSE for pure nodes. It provides the rank-N CPU execution
  boundary for later native code lowering; it does not yet generate machine
  code. Its `lower_text()` is the deterministic optimized artifact to map onto
  a later kernel IR. The same immutable plan
  now exposes CPU primal, VJP, and JVP evaluation. `kernel_ir()` exposes
  structured op, operand, shape, and input-name records without parsing text.
- `TensorBackend` is the rank-N execution contract and `CpuBackend` is its
  default implementation. The Linux-only `cuda` feature adds an
  NVRTC-compiled FP32 `CudaBackend` plus an immutable `CudaExecutionPlan` used
  by Python `compile_cuda()`. The retained context/module remove repeated NVRTC
  compilation across evaluations. Current lowering covers rank-two and
  broadcast-batched matmul,
  rank-N concat with GPU-resident slice/pad-slice reverse nodes, global and axis reductions, transpose, broadcasting, `where`, `div`, `log`,
  and the supported unary primitives. Pure elementwise graphs use one fused
  CUDA kernel; general graphs also lower maximal elementwise tails after
  materialized leaves through `fusion_regions()`. Intermediate CUDA
  buffers are recycled by last-use liveness, while retained inputs, the output,
  and Adam state stay allocated. `TensorCudaExecutionPlan.device_buffer_count`
  exposes the current allocation count for diagnostics. It is still preview
  infrastructure: host inputs and outputs transfer per general evaluation. A
  gradient-output plan can retain one input and apply device-side SGD or Adam
  updates without materializing that parameter, gradient, or Adam moments on
  the host. Separate gradient plans can synchronize retained inputs through
  same-device GPU-to-GPU copies after each update. Python
  `cuda_adam_step(...)` batches the per-parameter update and synchronization
  sequence, while leaving symbolic gradient-plan construction explicit.
  `cuda_adam_optimizer(...)` stores inputs across steps; passing a dictionary
  to `step(...)` refreshes only mini-batch tensors without rebuilding plans or
  resetting GPU-resident parameter and Adam state. For symbolic VJP results
  from one graph, `cuda_adam_vjp_optimizer(...)` constructs one union plan,
  executes shared forward/reverse nodes once, and updates every parameter from
  that one device-resident gradient snapshot. It still lacks fused GEMM and
  distributed AD.
- `TensorIr.symbolic_vjp(output, cotangent_name)` now emits a transformable
  reverse graph with explicit cotangent input and one output node per original
  input gradient. `TensorTraceResult.symbolic_vjp(cotangent_name)` and
  `TraceTensor.symbolic_vjp(cotangent_name)` expose that graph to Python as
  named gradient trace results. This gives backend lowering
  an AD graph instead of a CPU-only reverse evaluator.
- `trace(fn, input_specs)` can execute a Python function with traced inputs and
  return a `TraceResult`.
- `TraceGraph.ir()` exposes structured node dictionaries with ids, ops, shapes,
inputs, input names, and typed `attrs` for lowering-sensitive parameters such
as scalar constants, `powf` exponents, reduction axes, and concat axes.
- `TraceGraph.lower_text()` renders that same trace as deterministic backend
  text with SSA-like ids, operand references, tensor shapes, and typed
  attributes. It is an inspection and lowering-contract prototype, not an
  MLIR or StableHLO-compatible serialization.
- `TraceGraph.vjp_ir(output_node_id)` can generate a minimal reverse-mode VJP IR
  for add, subtract, multiply, divide, `where`, concat, powf, tanh, exp, log,
  sqrt, sin, cos, transpose, reshape, matmul, and sum/mean outputs.
- `TraceGraph.evaluate(output_node_id, inputs)` can execute the traced primal
  graph on CPU.
- `TraceGraph.evaluate_vjp(...)` can execute reverse cotangent accumulation over
  the current add/sub/mul/div/where/concat/powf/tanh/exp/log/sqrt/sin/cos/transpose/reshape/matmul/sum/mean
  trace graphs on CPU. Greater-than mask nodes are treated as non-differentiable
  control values.
- `grad(fn, specs, values, cotangent)` wraps trace and VJP evaluation for Python
  users.
- `grad_fn(fn, specs, cotangent)` returns a reusable Python callable over value
  dictionaries by tracing once at transform creation.
- `grad_scalar_fn(fn, specs)` returns a reusable Python callable for scalar-loss
  gradients with an implicit `(1, 1)` cotangent seed and a cached trace.
- `grad_scalar(specs)` can be used as a decorator factory for scalar-loss
  gradient functions that trace once when the decorator is applied.
- `value_and_grad_fn(fn, specs, cotangent)` returns both the primal output and
  VJP gradients from one cached trace evaluation.
- `vjp_fn(fn, specs)` returns a reusable VJP callable that accepts an output
  cotangent at call time and returns primal output plus input cotangents.
- `jacobian_fn(fn, specs)` returns a reusable single-input dense Jacobian
  callable by applying repeated VJPs to one cached trace.
- `jacobians_fn(fn, specs)` returns reusable per-input dense Jacobian matrices
  as a dictionary keyed by input name, using repeated VJPs over one cached trace.
- `TraceGraph.evaluate_jvp(output_node_id, inputs, input_tangents)` evaluates
  primal values and tangents together through direct CPU forward-mode AD.
- `jvp_fn(fn, specs)` returns a reusable one-or-more-input JVP callable using
  that direct forward-mode evaluator rather than materializing dense Jacobians.
- `TraceGraph.compile_cpu(output_node_id)` freezes only the output-reachable
  traced subgraph into an immutable `CpuExecutionPlan`, eliminating unused
  nodes, remapping internal operands, and commoning identical pure unary
  subexpressions. It supports reusable CPU primal, JVP, VJP, and value-and-VJP
  evaluation; AD transforms and `jit` retain this plan. It establishes the
  execution boundary for later code generation, but is not a machine-code
  compiler yet.
- `jit(specs)` can be used as a decorator factory that traces a function once
  when the decorator is applied, then evaluates a frozen CPU execution plan for
  value dictionaries.
- `TensorTraceGraph.compile_mlx()`, `TraceTensor.compile_mlx()`, and
  `TensorTraceResult.compile_mlx()` expose the experimental MLX primal backend
  to Python. It supports elementwise arithmetic, unary math, matmul, global
  sum/mean, axis reductions, reshape, transpose, concat, and broadcast.
  Slice and internal zero-padding are rejected explicitly
  when a plan uses them.
- The Python bridge does not yet provide general compiled JIT lowering.
- `Tensor.split_key(...)`, `Tensor.random_normal(...)`, and
  `Tensor.glorot_normal(...)` provide deterministic stateless initialization
  without a global RNG.

Target direction:

- Keep Python as the user-facing authoring layer.
- Use Rust-owned tensor objects as tracer values under decorators such as
  `jit` and `grad`.
- Prefer tracing first. AST parsing can be explored later for a smaller,
  explicitly supported Python subset.
- Make Python object lifetime and Rust/device memory ownership explicit.

Near-term milestone:

- Add compiled lowering behind the current Python `jit` decorator factory.
- Expand IR-to-AD beyond the current primitive set.
- Add IR-to-backend lowering passes.

## Execution Roadmap

This section is the implementation order. A phase is not complete until its
acceptance checks pass on every backend it claims to support. The sequence
prioritizes a usable GPU-resident PINN training path before broader API surface
area, distributed execution, or source-to-source AD research.

### P0. Numerical Contract And Reproducible Baseline

Goal: make backend correctness and performance regressions observable before
adding new compiler behavior.

- Define the public precision contract: CPU may retain its `f64` reference
  behavior while CUDA and MLX execute `f32`; backend comparison tests must use
  documented, operation-appropriate tolerances.
- Maintain differential tests for CPU, CUDA, and MLX primal execution, JVP,
  VJP, and the supported higher-order transforms.
- Keep reproducible benchmarks for batched MLP forward, `value_and_grad`, a
  Poisson residual, batched matmul, and an optimizer step. Record device,
  dtype, shapes, warm-up policy, synchronization policy, and transfer policy.
- Preserve explicit failure for unsupported backend operations. No backend may
  silently fall back to host execution.

Acceptance checks:

- `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
  and the Python test suite pass on the default build.
- The MLX feature passes its Rust and Python suites on Apple silicon.
- The CUDA Rust and Python suites pass on the configured GTX 1660 SUPER,
  including retained device execution, cuBLAS matmul, CUSOLVER solve,
  symbolic VJP/JVP, multi-output materialization, and device optimizer checks.
- `git diff --check` passes and benchmark artifacts report enough metadata to
  reproduce a result.

### P1. GPU-Resident Value-And-Grad Training Path

Goal: make a small PINN train through one high-level CUDA API without copying
parameters or gradients to the host on every step.

Status: implementation and Rust runtime validation complete on the configured
GTX 1660 SUPER. The 2026-07-28 driver-backed suite passed all 47 CUDA core
tests and the release Python suite, including retained device buffers,
symbolic VJP/JVP, multi-output materialization, cuBLAS, CUSOLVER, SGD, and
Adam. Benchmark execution remains separate from correctness validation.

- Add `tensor_value_and_grad_cuda_fn` for a scalar loss and named parameters.
  It must compile a shared forward/reverse union plan rather than one plan per
  parameter.
- Expose a reusable CUDA training callable that owns parameter, optimizer, and
  static collocation buffers. Dynamic mini-batch tensors may be refreshed
  explicitly without retracing or resetting optimizer state.
- Add a narrow, documented initializer and stateless random-key API needed to
  construct MLP parameters reproducibly.
- Provide a CUDA PINN integration example and test for Poisson/Helmholtz:
  loss decreases, boundary residual is bounded, PDE residual is bounded, and
  GPU allocation count stabilizes across training steps.
- Measure and eliminate avoidable H2D/D2H copies in the training loop.

Acceptance checks:

- A Python example trains a two-layer PINN using only Quabla tensor APIs after
  initialization.
- Profiling confirms no per-step parameter or gradient device-to-host copy.
- CPU reference and CUDA loss/gradient checks agree within the documented
  `f32` tolerance.

### P2. Composable `vmap`

Goal: replace fixed axis-0 shape lifting with a transform that composes with
AD and backend lowering.

- Introduce per-primitive batching rules for elementwise operators, matmul,
  reductions, transpose, broadcast, concat, slice, and `where`.
- Support `in_axes`, `out_axes`, unmapped (`None`) arguments, and normalized
  negative axes. Keep batch extent static until shape polymorphism exists.
- Validate `vmap(grad)`, `grad(vmap)`, `vmap(jvp)`, and `vmap(vjp)` against a
  loop reference while asserting that execution does not use a Python
  per-example loop.
- Include the batching signature in compilation cache identity.

Acceptance checks:

- Per-example gradients for a batched MLP agree with the loop reference on CPU
  and CUDA.
- CUDA and MLX execute one batched plan for their supported primitive subsets.

Implementation status (2026-07-15): batch-axis propagation and public
`in_axes`/`out_axis`/`None` layout normalization are implemented for the
existing primitive surface. CPU coverage validates non-leading axes, unmapped
arguments, reductions, transpose, batched JVP, and batched VJP. CUDA now
lowers batched VJP primal and gradient outputs into one union plan; CPU and
CUDA tests compare per-example gradients of a batched MLP with a loop reference.
MLX now lowers the same batched VJP primal and gradient outputs into one
multi-output GPU plan; its CPU-parity coverage includes a non-leading mapped
axis and an unmapped parameter. MLX also lowers batched primal/JVP pairs from
explicit runtime tangent IR inputs in one multi-output GPU plan, with
non-leading input and output axes covered against the CPU reference.
CUDA now lowers the same explicit-tangent batched JVP union plan; its runtime
parity test is gated on `QUABLA_CUDA_TEST` until the configured CUDA host exposes
its NVIDIA driver and runtime libraries.

### P3. SciML Array And Linear-Algebra Surface

Goal: cover the array semantics required by PINNs, operator learning, and
inverse problems without attempting full NumPy compatibility.

- Add multi-axis reductions, `keepdims`, extrema, norms, gather/scatter, and
  an explicit indexing/view model with differentiability rules.
- Add a scoped `einsum` subset, `solve`, Cholesky, triangular solve, and their
  AD rules where numerically well-defined.
- Add common neural primitives: `relu`, `sigmoid`, `softplus`, `abs`,
  `maximum`, and `minimum` with explicit subgradient policy.
- Add dtype and device-placement APIs. Implicit cross-device copies are an
  error.

Implementation status (2026-07-15): multi-axis `sum`/`mean` and `keepdims`
are available for eager and traced tensors. They preserve symbolic AD and
lower through the existing CPU, CUDA, and MLX reduction/reshape paths.
Elementwise `maximum`/`minimum` are also available on eager and traced tensors;
they compose `greater` and `where`, with equality subgradients routed to the
right operand. `max`/`min` reductions now accept the same axis/keepdims
contract and lower static traces through slice plus selection nodes; repeated
extrema route the derivative to the final row-major coordinate. Native `sqrt`
and L2 `norm` are available across CPU, CUDA, and MLX; negative values follow
IEEE floating-point `NaN` semantics and all
derivative orders at zero use an explicit zero subgradient. Native fused extrema
reductions and other norm variants remain pending. Static integer-index
`gather` and functional `scatter_add` are available with repeated-index VJP
accumulation; dynamic tensor indices, masks, and assignment-style scatter
remain pending the dtype/index IR design. A scoped matrix-product `einsum`
subset lowers directly to rank-N `matmul`. Rank-2 `solve(A, B)` now has a
partial-pivot LU CPU reference implementation, eager and traced Python APIs,
and direct/symbolic JVP/VJP plus mixed-tangent rules; it rejects non-square,
rank-mismatched, and singular inputs explicitly. CUDA now has a CUSOLVER
`Sgetrf`/`Sgetrs` rank-2 lowering that keeps factorization and right-hand-side
buffers on device; it passed driver-backed CUSOLVER validation on the configured
GTX 1660 SUPER on 2026-07-28.
`cholesky()` is available as a rank-2 eager factorization with symmetry/SPD
validation and as a static trace-time recurrence, so existing JVP/VJP machinery
provides verified derivatives. Native CUDA/MLX factorization lowering and
broader contractions remain pending. MLX 0.32.2
only exposes `linalg::solve` on a CPU stream, so Quabla rejects it on the MLX GPU
backend rather than silently falling back. `relu`, `abs`, `sigmoid`, and a
numerically stable `softplus` are available on eager and traced tensors; `relu`
uses a zero subgradient at zero, while `abs` follows the existing `where`
tie-rule and has derivative -1 at zero. Dtype phase D1 (`float32`/`float64`)
and the `Bool` half of D2 (comparisons, logical ops, `isfinite`/`isnan`,
`any`/`all`; see Dtype Support below) are implemented; device-placement APIs,
`I32` index tensors, and the remaining dtype phases are pending.
`tril` and `triu` now project the last two axes of rank-N tensors across CPU,
CUDA, and MLX, with direct/symbolic JVP/VJP and mixed-tangent propagation; they
provide the gradient masking primitive required before triangular solve and
Cholesky can be added.
`solve_triangular(rhs, lower=True, transpose=False)` is now available for
rank-2 eager and traced tensors. Its reference lowering composes triangular
projection with `solve`, preserving existing AD semantics; unit-diagonal and a
dedicated CUDA TRSM lowering remain pending performance work.
Eager and traced tensors now share a
differentiable integer/contiguous-slice indexing subset; it lowers through
Slice/PadSlice on CPU, CUDA, and MLX. Strided/empty slices, ellipsis, and
advanced indexing remain pending the explicit gather/scatter design.

Acceptance checks:

- Reference SciML examples use no NumPy operation inside the differentiated
  model body.
- Each new primitive has CPU finite-difference or analytic AD checks and
  backend parity coverage.

### Dtype Support

Goal: make element types a checked property of every Tensor IR node instead of
an implicit backend lowering, with a CPU reference for each dtype a device
executes.

- D1. `F32`/`F64` logical dtypes on IR nodes and host tensors, an explicit
  `Cast` op, strict tensor promotion with weak Python/AD scalars, CPU `f32`
  reference semantics, native `f32` execution on CUDA and MLX with `f64`
  programs still lowered to `f32`, and Python dtype objects.
- D2. `Bool` and `I32` dtypes for masks and indices: comparisons produce
  `Bool`, `where` consumes it, and gather/scatter take integer index tensors
  (the dtype/index IR that P3's dynamic indexing waits for).
- D3. `F16`/`BF16` storage and compute on MLX and CUDA with an explicit
  accumulation dtype; backends must materialize weak constants in the
  consumer's dtype (MLX `from_f32` constants must not promote half arrays).
- D4. Native device `f64` as an opt-in execution dtype (CUDA double kernels,
  MLX where supported); `execution_dtype(F64)` then stops mapping to `f32`.
- D5. Typed host storage (implemented 2026-10-04): `DynamicTensor`/`Tensor`
  use shared F64/F32/Bool buffers rather than permanently widening narrow
  values. Typed interop preserves immutable aliases and owned imports.
  Device upload borrows compatible F32 storage; narrow device readback adopts
  typed buffers. Dtype-aware batch specialization remains follow-up work.
- D6. Mixed-precision training policies: master weights, loss scaling, and
  dtype-aware optimizer state on device.

Implementation status (2026-09-27): D1 is implemented. `TensorNode` carries a
dtype and a JAX-style weak flag inferred at construction; `TensorIr::input`
keeps the `f64` default and `input_typed` declares other dtypes. Combining
strong `f32` and `f64` tensors is a construction error that names `astype`.
When a weak scalar meets a strong operand of another dtype the builder inserts
an explicit `Cast` of the scalar, so existing AD rules built from
`scalar_constant` are unchanged and `f32` graphs are never promoted. `Cast`
takes its target from the node dtype; its JVP casts the tangent to the target
and its VJP casts the cotangent back to the source. Plan compilation removes
strong same-dtype casts, folds constant casts with the target rounding (a
`0.1` round trip through `f32` yields `0.10000000149011612`), and keys CSE on
dtype and weakness. Loop regions require matching body-output and carry
dtypes; Python loop regions give the index the carry dtype, and region
captures must match the region input dtype. The CPU keeps `f64` storage and
rounds every node result to its dtype in all interpreters; eager CPU AD does
its derivative arithmetic in `f64` and rounds at casts and returned gradients,
while symbolic AD follows node dtypes exactly.
`TensorDeviceBackend::execution_dtype` maps `F64` and `F32` to `f32` on CUDA
and MLX; both reject any node whose dtype does not lower to `f32`, execute
`Cast` as an identity, and tag readbacks with the node dtype. Python exposes
`quabla.float32`/`quabla.float64`, `Tensor(..., dtype=...)`, `astype`, `dtype`
properties, and `(name, shape, dtype)` input specs; eager `Tensor` ops follow
the same promotion and rounding rules.

Verification (2026-09-27): Rust tests cover weak-scalar adoption and `f32`
lowering text, strict mixing errors, bitwise `f32` agreement of CPU
`x * y + z`, `x / y`, and `sqrt` on both interpreter paths, cast round trips
and same-dtype cast removal, cast JVP/VJP dtypes with finite-difference
checks, CSE separation of casts, `f32` `kernel_ir()` validation with
unchanged `f64` goldens, input rounding, and an `f32` loop region. A float32
MLP value-and-gradient program matches the CPU `f32` reference with a scaled
error of 2.2e-8 on MLX (Apple silicon) and on CUDA (GTX 1660 SUPER), within
the 1e-6 bound; the same device runs stay within the existing 1e-5 bound
against the `f64` reference. `benchmark_pinn_mlx.py` shows no `f64` slowdown
(medians of five interleaved runs: step 0.836 -> 0.826 ms, compile 36.10 ->
36.32 ms).

Implementation status (2026-09-28): the `Bool` part of D2 is implemented;
`I32` dtypes and integer index tensors for gather/scatter remain pending.
`TensorDType::Bool` stores `0.0`/`1.0`, and rounding to it is the float to
bool conversion (nonzero and `NaN` become true). The only new IR op is
`Compare` with IEEE `greater`/`greater_equal`/`less`/`less_equal`/`equal`/
`not_equal` kinds and a `Bool` result; the legacy `Greater` float mask and
the Python `gt()` it backs are unchanged. The other operations compose
existing primitives, so no backend needs a new kernel: `logical_and`/`or`
are `where(a, b, a)`/`where(a, a, b)`, `logical_not` is `equal(a, false)`,
`isnan` is `x != x`, `isfinite` is `x - x == 0`, `any` is
`sum(cast(b)) > 0`, `all` is `sum(cast(!b)) == 0` (a sum of non-negative 0/1
terms is zero only when all are, even in `f32`), and `cast(x, bool)` is
`x != 0`, so a `Cast` never targets `Bool` from a float. Promotion treats
`Bool` as 0/1 of the other operand's float dtype, or as a weak `f64` next to
weak scalars only (`mask * 2.0` then adopts a later `f32` operand); bool is
accepted by comparisons, `where`, and data movement only, and arithmetic,
math, or reductions on it name `astype`. `where` and `cond` keep accepting
0/1 float predicates. Comparisons have zero tangents, no cotangent reaches a
bool node, bool inputs are omitted from gradient maps, and differentiating
a bool output or with respect to a bool input is an error. Cond regions may
capture bool masks; cond results and loop-region inputs must stay floating
until region AD pairs non-differentiable values. Region AD now retains
captures with `where(0, x, 0)` instead of `x - x`, so a `NaN`/`inf` capture no
longer poisons cond tangents and gradients. `execution_dtype(Bool)` is `f32`
on CUDA and MLX; CUDA lowers `Compare` in per-node, fused, region, and loop
body kernels, and MLX casts its native bool comparison back to `f32` at the
node. `lower_text`/`stablehlo_text` print `i1` and `kernel_ir()` reports
`"bool"`. Python exposes `quabla.bool_`, comparison methods and functions,
`<`/`<=`/`>`/`>=`, `&`/`|`/`~`, `isfinite`/`isnan`, `any`/`all` with
`axis`/`keepdims`, scalar `where` branches, and NumPy-style truthiness for
single-element eager bool tensors; `==`/`!=` keep identity semantics.

Verification (2026-09-28): Rust tests cover every comparison with `NaN`,
`inf`, and ties on the per-node and fused CPU interpreters, logical ops and
axis reductions, the promotion and rejection rules, `i1` lowering text and
`kernel_ir()` validation, bool and legacy float predicates in `where` and
`cond`, zero derivatives and the bool input/output errors, cond regions with
bool captures, and a `NaN`-guarded masked loss whose symbolic and eager
gradients match analytic values and central differences. On MLX (Apple
silicon) and CUDA (GTX 1660 SUPER) the same program matches the CPU exactly
for masks, an `all(isfinite(x))` cond and its gradient match exactly, and
the masked loss and gradients agree within a scaled error of 4.8e-8. The
Python matrix repeats these checks for eager and traced tensors, including
the MLX/CUDA parity cases. `benchmark_pinn_mlx.py` shows no slowdown
(medians of seven interleaved runs against the D1 build: step 0.687 ->
0.686 ms and compile 20.73 -> 20.78 ms by default; 3.879 -> 3.741 ms and
22.12 -> 22.49 ms with `--hidden-layers 3 --hidden-width 64`).

### P4. Compiler Passes And Kernel Performance

Goal: turn the current frozen plan into a proper backend-neutral compilation
pipeline.

- Formalize `kernel_ir()` with typed shapes, dtypes, layouts, placements,
  alias information, and effects.
- Add canonicalization, constant folding, CSE, DCE, broadcast/layout
  propagation, fusion partitioning, and buffer planning passes.
- Prioritize CUDA fusion for GEMM+bias+activation, residual/loss reductions,
  activation backward chains, and batched MLP blocks.
- Evaluate CUDA Graphs for static training steps. Treat MLIR/StableHLO export
  as an optional interoperability path, not a prerequisite for useful JIT.

Implementation status (2026-07-29): `kernel_ir()` now carries a typed
`TensorDType` contract rather than a stringly-typed dtype field. Eager tensors
currently emit logical `f64`, while CUDA/MLX `f32` execution remains an
explicit backend lowering decision rather than an implicit metadata rewrite.
`TensorDeviceBackend::precision()` records this boundary as logical/execution
dtype pairs: CPU is `f64`/`f64`, CUDA and MLX are `f64`/`f32`.
`TensorIr::stablehlo_text(output)` is an explicit interoperability probe for
static `f64` input/add/multiply/tanh graphs. It emits deterministic textual
StableHLO and rejects every operation outside that verified subset; it is not
an executable MLIR pipeline, a fallback backend, or a claim of broad StableHLO
coverage.
The IR also carries scalar/row-major-contiguous layout, backend-neutral `unplaced`
placement, and input/pure effect metadata. The validator enforces these current
invariants; physical placement, aliases, effects beyond inputs, and buffer
planning remain pending. `reshape` additionally exposes a logical alias
candidate in the IR; physical buffer reuse is still deferred to buffer planning.
Scalar-only constant subgraphs are folded during CPU-plan compilation when the
fold produces the value execution would; a log of a non-positive constant and
an invalid powi exponent remain runtime operations. A final DCE pass then removes
the unreachable source constants and intermediates introduced by folding or CSE
before any backend sees the plan. `buffer_plan()` now provides a backend-neutral
exact-size temporary-slot schedule from final-use liveness; inputs remain
external bindings and reshape aliases preserve their backing storage. CUDA has
not yet been refactored to consume this common schedule. Canonicalization also
removes identity reshape/broadcast nodes and collapses reshape chains before
backend lowering, avoiding unnecessary layout copies.
The base plan remains `Unplaced`, while `kernel_ir_with_placements(...)` can
apply validated typed compiler metadata without changing execution: a
`TensorDeviceId`, shaped `TensorDeviceMesh`, and
`TensorPartitionSpec::{Replicated, Sharded}`. This is an IR contract only; no
backend lowers it to multi-device work yet.

`examples/benchmark_pinn_cuda.py` records reproducible compilation, warm-up,
and synchronized device-resident Poisson PINN Adam-step timings, including
shape, transfer, and buffer-stability metadata. The single-GPU Linux host has
a visible GTX 1660 SUPER and the CUDA runtime needed for functional parity
tests. On 2026-07-30, the release extension ran its width-16,
four-collocation-point two-layer Poisson PINN for 1,000 timed steps after 100
warm-up steps at 711.02 steps/s (1.406 ms/step). Compilation took 374.02 ms,
loss decreased from `1.07943933e3` to `6.26469124e-3`, and the retained CUDA
buffer count remained 58. Timed steps exclude diagnostic readback but are
synchronized by the following `loss()` call. This is a small fixed-shape GTX
baseline, not a cross-backend comparison or a large-model throughput claim.
The same run with width 64 compiled in 1,248.56 ms and reached 693.11 steps/s
(1.443 ms/step), with loss `4.54379500e5 -> 8.85865356e2` and the same 58
buffers. The 2.6% step-time difference at four collocation points shows that
this harness is dominated by fixed launch/dispatch work; a larger-collocation
benchmark is required before attributing throughput changes to GEMM or fusion
scaling. The benchmark now accepts `--collocation` while retaining its
fixed-shape/device resident contract. At width 64, 512 collocation points ran
at 657.96 steps/s (1.520 ms/step; 1,338.54 ms compile) and 8,192 points ran at
368.81 steps/s (2.711 ms/step; 1,605.91 ms compile), both with 58 retained
buffers. These measurements establish that the larger workload is no longer
dominated solely by fixed dispatch cost; they remain one-device baselines, not
evidence of a fusion speedup until a before/after comparison exists.

`examples/benchmark_pinn_mlx.py` records the corresponding MLX compile,
warm-up, fixed-shape training-step, dtype, device, synchronization, and
transfer metadata. On the local Apple silicon host (2026-07-28), its
1,000-step baseline was 1,906.15 steps/s (0.525 ms/step) for the one-parameter,
seven-point Poisson PINN; loss decreased from `5.51` to `1.28e-12`. This is a
local small-graph baseline, not a cross-backend or large-model performance
claim.
`examples/benchmark_loop_value_and_grad_mlx.py` records fixed-bound `Fori`
Adam steps through the same device-resident reverse path. It times only
`optimizer.step()` after warm-up and excludes diagnostic loss readback.
`examples/benchmark_mlp_value_and_grad_mlx.py` additionally records the
compile, warm-up, and steady-state timing of a batched two-layer MLP scalar
value-and-gradient specialization. It explicitly includes dynamic input upload
and loss/gradient diagnostic readback in each timed call, separating that
end-to-end measurement from the device-resident optimizer benchmark. On the
same local Apple silicon host (2026-07-28), its batch-256, width-64 baseline
was 2,699.41 calls/s (0.370 ms/call); this is likewise not a cross-backend
comparison.
`examples/benchmark_forward_mlx.py` records corresponding end-to-end batched
matmul and two-layer MLP forward baselines. Its explicit host transfer and
materialization metadata prevents those diagnostics from being interpreted as
device-only kernel throughput. `TensorMlxExecutionPlan.evaluate_device(...)`
also evaluates and synchronizes a primal output without materializing it on
the host, so the same script can exclude output readback while still reporting
the dynamic host-input upload that remains in each invocation.
`TensorMlxExecutionPlan.retain_inputs(...)` now retains selected static
bindings across primal calls, and `benchmark_forward_mlx.py
--retain-static-inputs` records that mode separately from the all-dynamic
baseline. On the same host (2026-07-28), retaining static weights improved a
batch-256, feature-64, output-64 matmul from 0.182 to 0.172 ms/call and a
two-layer width-128 MLP from 0.231 to 0.221 ms/call. These device-only
measurements exclude output readback but still include dynamic-input upload.

`fusion_regions()` now exposes maximal elementwise partitions with explicit
materialized inputs. It prevents unsafe fusion across values consumed by a
non-elementwise operation, such as a matmul, and is the lowering contract for
partial CUDA fusion. The configured GTX 1660 SUPER passed driver-backed fusion,
buffer reuse, symbolic VJP, and device optimizer coverage on 2026-07-28.

Acceptance checks:

- Each pass has IR-level golden tests and preserves CPU reference results.
- Benchmarks demonstrate an attributable speedup for at least one end-to-end
  MLP/PINN training step, not only an isolated microkernel.

### P5. Shape Polymorphism And Structured Control Flow

Goal: support practical variable batch sizes and iterative differentiable
programs without tracing arbitrary Python control flow.

- Add symbolic dimensions and shape constraints with bounded specialization.
- Add explicit IR operations analogous to `cond`, `scan`, and `fori_loop`.
- Define JVP/VJP semantics for each control-flow operation and reject
  data-dependent Python branches during tracing.

Implementation status (2026-07-27): `tensor_jit_batch_fn(...)` provides the
first bounded-specialization path for CPU plans. It traces lazily on the first
observed mapped batch size, caches a fixed number of batch-size plans, and
rejects changes to non-batch dimensions or unmapped inputs.
`tensor_value_and_grad_batch_fn(...)` uses the same cache contract for scalar
losses and performs their VJP from the frozen plan, so variable collocation
batch sizes now have a CPU reference for both values and gradients.
`tensor_value_and_grad_batch_mlx_fn(...)` mirrors that bounded specialization
contract on MLX for requested scalar-loss gradients, with one multi-output GPU
plan per batch size and explicit result readback.
`tensor_jit_batch_cuda_fn(...)` and
`tensor_value_and_grad_batch_cuda_fn(...)` now provide the same bounded cache
contract for CUDA primal and scalar value-and-gradient plans. Their Rust
runtime coverage now passes on the configured Linux host with `libnvrtc.so`.
Symbolic
dimensions and explicit control-flow IR remain pending.
`TraceTensor.__bool__` now rejects data-dependent Python branches explicitly,
so tracing cannot silently specialize to one host-side branch.
`tensor_cond_fn(...)` now freezes true/false CPU branch regions and evaluates
only the host-boolean-selected branch; its scalar value-and-gradient variant
selects the matching branch VJP, while `tensor_cond_jvp_fn(...)` selects its
matching JVP. This establishes lazy branch semantics without misrepresenting
`where` as control flow. `TensorRegion` and `TensorCondExecutionPlan` now own
the identical-capture validation and selected-branch primal/JVP/VJP execution
in `quabla-core`, rather than leaving this contract in the Python bridge.
`TensorOp::Cond` now owns two frozen `TensorRegion` plans plus ordered,
explicit named capture bindings to parent node IDs. CPU execution evaluates the
scalar predicate first and materializes only the selected branch; DCE, remap,
buffer planning, direct JVP, and direct VJP preserve those bindings. The
current regression covers a capture from a parent intermediate, lazy inactive
branch failure avoidance, and branch-selected first-order derivatives.
`tensor_cond(predicate, on_true, on_false, operands)` exposes the same
contract to Python tracing, including nested CPU regions. Its operands are the
only legal branch captures. Symbolic JVP and VJP now rewrite `Cond` regions:
they preserve the predicate and use branch-selected primal/tangent or gradient
regions. Region transform outputs retain a stable capture ABI even if DCE
would otherwise remove an unused capture in one branch. Public scalar
Hessian/HVP calls through `Cond` compose symbolic VJP with symbolic JVP, so
second coordinate derivatives needed by PINNs work on CPU. General mixed and
higher-order region transforms remain pending. Status (2026-09-27): MLX now
lowers device-predicate `Cond`. The executor evaluates the scalar predicate,
reads it back to the host once (one GPU stream synchronization per `Cond`
evaluation), and executes only the selected region plan on the GPU stream with
device-resident captures; nested regions and the `Cond` nodes produced by
symbolic JVP/VJP (and their composition for second derivatives) use the same
path. Evaluating both branches with `where` was rejected because an inactive
branch such as `log(x)` for `x <= 0` would still inject NaN into gradients
(`0 * NaN`). Non-finite predicates fail like CPU. CUDA lowers `Cond` with the same host-sync
boundary: `CudaBackend::compile` compiles both regions as per-node device
programs in the parent CUDA context (one extra NVRTC module per region), and
execution copies one `f32` predicate to the host, then launches only the
selected region's kernels with parent buffers bound through device-to-device
copies; the region writes into a buffer from the parent pool, so repeated
executions neither copy the result to the host nor grow device memory. `Cond` inside a fused CUDA
`Fori`/`Scan` body stays an explicit compile-time rejection, because those
bodies are single device kernels without a host boundary. No backend eagerly
materializes both branches. Vmappable conditional predicates remain pending
and are rejected at trace time.
`tensor_fori_loop(...)` remains the compatibility API that statically unrolls
fixed host-integer bounds with a same-graph, shape-preserving TraceTensor
carry. `TensorForiExecutionPlan` now owns a frozen fixed-bounds body region:
CPU primal execution iterates its carry at runtime, JVP propagates carry and
explicit-capture tangents, and VJP records a carry tape then replays body VJPs
in reverse. `TensorOp::Fori` embeds that plan in parent Tensor IR with explicit
capture bindings; DCE/remap, CPU execution, direct JVP/VJP, text lowering, and
CPU compilation preserve the region boundary. `tensor_fori_loop_region(lower,
upper, body, init, operands)` exposes one-time Python tracing of
`body(index, carry, *operands)`, where the scalar index is a TraceTensor and
all external captures are explicit. Batch specialization rejects `Fori` rather
than silently unrolling or changing layout. `tensor_vmap_fn` supports
structured `Fori` with a canonical leading mapped carry and mapped captures
of the same batch extent; unmapped captures remain shared across examples.
The loop body must preserve the carry batch axis. CPU `vmap` JVP and VJP reuse
the structural loop transforms and preserve per-example carry/capture
gradients. MLX executes the same vmapped `Fori` VJP plan for its supported
first-order region path. MLX now executes the primal
fixed-bound `Fori` body by
host-dispatching device-resident MLX arrays: carry and captures remain on the
GPU between iterations, but this is not yet a fused Metal control-flow kernel.
CUDA lowers a restricted primal `Fori` body to one device kernel: each thread
owns one carry element and iterates the fixed bounds in device code. It also
lowers first-order `ForiVjp` to a single device kernel when the body is pure
elementwise and every explicit capture broadcasts to the carry shape. That
kernel writes a global-memory carry tape during its forward recurrence, then
replays the body VJP in reverse in the same thread; it does not materialize
carries or gradients on the host. Body reductions, matmul/solve, nested
regions, and broadcasted captures for `ForiVjpJvp` remain explicit rejections.
CUDA also lowers `Scan`
primal and first-order `ScanVjp` when its per-step output matches the carry
shape, plus the verified direct-broadcast output subset for unequal lane
counts, and explicit captures broadcast to it. A shared CUDA Scan
execution writes final carry and time-major outputs in one kernel, retaining
the sibling result in a device-buffer cache; its VJP reuses a carry tape and
combines final-carry/output cotangents during reverse replay. CUDA `ForiVjp`
and `ScanVjp` support scalar and legal trailing-axis broadcast captures by
emitting per-element contributions, mapping each carry index to its broadcast
target offset, and atomically reducing on device. `ScanVjpJvp` supports the
same direct-broadcast unequal-lane subset and broadcast-compatible captures;
arbitrary unequal carry/output graphs remain explicit rejections. First-order VJP source groups now compile
to one multi-output kernel, so all requested gradient targets reuse one reverse
traversal and device carry tape; for the direct-broadcast `ScanVjp` subset the
grouped kernel sums each step's output cotangents onto the carry lane, as the
single-target kernel does. Linux CUDA
compilation and GTX 1660 SUPER CPU-parity coverage exercise nonlinear loops
for both initial-carry and same-shape capture gradients.
Symbolic JVP through `Fori` emits a paired primal `Fori` and structural
`ForiJvp` tangent node, preserving separate primal/tangent carries through the
runtime loop rather than statically expanding it. Symbolic VJP through `Fori` uses a
fixed-bound carry tape and grouped `ForiVjp` result nodes, so gradients of the
initial carry and explicit captures reuse one reverse loop. JVP through a
`ForiVjp` result now emits a grouped `ForiVjpJvp` node. Its execution plan
records primal and directional carry tapes, then replays compiled JVPs of each
body-gradient in reverse. Public `hessian_scalar` and `hvp_scalar` therefore
select this forward-over-reverse transform for fixed `Fori` regions rather than
the CPU mixed-direction fallback. `ForiVjpJvp` has no further derivative rule.
CUDA lowers its restricted pure-elementwise, same-shape-capture subset with
primal and tangent carry tapes in one device kernel; batch specialization
still rejects it. MLX executes first-order `ForiVjp` results by retaining the
forward carry tape and replaying a precompiled body VJP plan on the GPU; carry
and capture gradients remain device-resident until the requested plan output
is materialized. MLX also executes `ForiVjpJvp` by retaining primal/tangent
carry tapes, then replaying precompiled gradient-JVP plans on the GPU. `tensor_scan(...)` remains the compatibility API that statically unrolls a fixed-length
carry/output loop. `TensorScanExecutionPlan` executes one frozen two-result
body region and returns final carry plus a leading-time-axis output stack.
`TensorOp::Scan` embeds the paired results in parent Tensor IR with a shared
execution group, so CPU primal evaluation executes the body once even when
both results are consumed. Direct JVP and VJP likewise process both results
together: JVP caches paired tangents, while VJP jointly replays carry/output
cotangents through the forward carry tape. `tensor_scan_region(lower, upper,
body, init, operands)` exposes one-time Python tracing of that region with
explicit captures. Symbolic JVP packs primal/tangent carries and per-step
outputs into a paired Scan region, so the transformed graph retains runtime
scan control flow. Symbolic VJP emits grouped `ScanVjp` result nodes: final
carry and output cotangents are replayed jointly through one reverse scan tape,
then initial-carry and explicit-capture gradients reuse that cached result.
MLX executes primal `Scan` with the same device-resident host-dispatch model
as `Fori`, including one shared execution cache when final carry and stacked
outputs are both consumed. `tensor_vmap_fn` applies the same leading-axis
contract to `Scan`: final carry retains batch axis zero and stacked outputs
are normalized from the native time-major stack to `[batch, time, ...]`
before the requested `out_axis` layout is applied. Direct CUDA vmap tracing
now has GTX 1660 SUPER CPU-parity coverage for Scan primal, JVP, nonlinear
VJP with initial-carry and capture gradients, and nonlinear per-example HVP
with respect to mapped initial carry and mapped capture. The HVP wrapper also
restores a selected input's non-leading public `in_axes` layout for an
elementwise loss. It requires one scalar loss per example and lowers the sum
through the existing structural `ScanVjpJvp` path; its CPU result is checked
against a central finite-difference VJP gradient. `TensorExecutionPlan::specialize_batch` still rejects structured Scan
for the bounded JIT-batch API rather than unrolling or falling back to CPU.
MLX executes first-order `ScanVjp` results
with a device-resident forward carry tape and separate precompiled body VJP
plans for final-carry and per-step-output cotangents; their carry/capture
contributions are accumulated on the GPU. `ScanVjpJvp` now binds primal/tangent
initial carry, final-carry cotangent and tangent, stacked-output cotangent and
tangent, plus primal/tangent explicit captures. Its CPU and MLX plans retain
primal/tangent carry tapes, slice each output-cotangent pair during reverse
replay, and add directional contributions from both body outputs. The
acceptance gate is a nonlinear Scan loss that consumes both final carry and
stacked outputs: CPU HVP matches a central finite-difference reference, and
MLX matches CPU without materializing intermediate carries or gradients on the
host. Further derivatives of `ScanVjpJvp` remain explicit errors.

Control-flow completion status:

`TensorRegion` now provides explicit capture bindings for `Cond`, `Fori`, and
`Scan`. CPU evaluates only the selected conditional region; structural JVP,
VJP, Hessian, and HVP paths preserve that laziness. Kernel IR, canonicalization,
and buffer planning retain the region boundary. Python exposes
`tensor_cond(...)` after lazy inactive-branch, nested-condition, and AD
coverage. This closes the original CPU control-flow increment. Status
(2026-09-27): MLX and CUDA lower device-predicate `Cond` through a single host
predicate readback and selected-region execution; CUDA rejects `Cond` inside
fused device-loop bodies explicitly.

Next implementation order:

Completed: symbolic JVP now has a multi-output result contract while preserving
the single-output API. Retained sibling `ScanVjpJvp` targets lower as one CUDA
source group; the first node launches the shared kernel, caches sibling device
buffers, and GTX 1660 SUPER parity covers carry plus explicit-capture targets.
P6's larger MLP/PINN and fixed-bound loop MLX validation (2026-09-27) reports
compile, device-step, and diagnostic-readback timing separately; results are
recorded in P6.

1. Keep arbitrary indexed or non-elementwise unequal-lane Scan bodies as
   explicit CUDA rejections until their reduction structure is represented in
   device reverse IR; add a rejection test for every unsupported form.

   Status (2026-09-27): the rejection matrix is covered at CUDA compile
   time. `test_compiler_facade_cuda_rejects_every_unsupported_scan_lane_form`
   covers primal Scan (reduced output, output-shaped capture on unequal
   lanes, capture incompatible with an equal-count output reshape, matmul,
   in-body reduction, transpose, indexed gather, layout-changing reshape),
   `ScanVjp` (non-direct-broadcast unequal output, layout-changing reshape,
   in-body capture reduction, VJP of a packed-pair Scan JVP), and
   `ScanVjpJvp` (layout-changing reshape, in-body capture reduction); the
   existing indexed unequal-lane HVP test remains.
   `test_compiler_facade_rejects_scan_derivatives_beyond_forward_over_reverse`
   covers VJP of `ScanVjp` and JVP of `ScanVjpJvp`. The audit found two
   forms that CUDA had accepted with wrong numbers. A reshape that moves a
   broadcast axis, such as a `[2, 1]` capture reshaped to `[1, 2]` or a
   reshape before an unequal-lane broadcast, is now rejected unless it only
   adds or drops leading unit axes, or has equal lanes with scalar or
   carry-sized captures. Packed-pair bodies (symbolic Scan JVP) read the
   partner half through slice/concat. The primal kernel now holds both
   halves in registers per thread instead of rejecting them, because
   accepted CUDA HVP programs keep a packed JVP Scan live. GTX 1660 SUPER
   parity covers nonlinear Scan JVP and a row-swap body.
2. P7 needs a host exposing at least two CUDA devices and NCCL. When that
   prerequisite is available, connect the typed sharding schedule to the
   Python training interface, add two-GPU loss/gradient parity, then measure
   collective and readback boundaries separately. Status (2026-09-27):
   a two-GPU node (2x RTX 3090) with NCCL is available; single-call two-GPU
   parity passes (see P7). Multi-step training and steady-state collective
   timing remain.

Implementation note (2026-07-29): `TensorForiTape` retains the fixed-bound
forward carry sequence and is reused by direct and symbolic reverse paths.
`TensorMultiRegion` and `TensorForiMultiExecutionPlan` retain shared
multi-output body plans, execute independently shaped carries in one loop, and
reverse all body outputs jointly in each iteration. Parent `ForiVjp` nodes
select the initial-carry or explicit-capture gradient and share one cached
reverse pass per source loop. `ForiVjpJvp` reuses a shared directional reverse
pass for all selected gradients in that source group. CUDA now fuses each
restricted first-order `ForiVjp` and `ScanVjp` source group into one
multi-output device kernel. CUDA `ForiVjpJvp` currently lowers one selected
directional-gradient result per kernel. `ScanVjpJvp` also lowers the verified
direct-broadcast unequal-lane subset and emits every selected
directional-gradient result of a source group from one kernel.
MLX supports first-order reverse and forward-over-reverse `Fori`
paths, but not higher derivatives of `ForiVjpJvp`.

Acceptance checks:

- A time-stepping differentiable model and multiple collocation batch sizes run
  with bounded recompilation and verified gradients.

### P6. MLX Reverse-Mode Parity

Goal: make the Apple backend a usable training backend rather than primal-only
execution.

Implementation status (2026-07-24): `tensor_value_and_grad_mlx_fn(...)`
lowers a scalar loss and its requested symbolic VJP gradients into one
multi-output MLX plan. `mlx_adam_loss_optimizer(...)` retains selected static
inputs, parameters, gradient outputs, and bias-corrected Adam moments across
steps; the MLX executor evaluates updates on `StreamOrDevice::gpu()` and does
not materialize parameters or gradients on the host during `step()`. `loss()`
and `parameters()` are explicit diagnostic readbacks. The training API has
scalar-loss parity for its supported Tensor IR operations. The fixed-shape
one-parameter Poisson PINN in `examples/pinn_poisson_mlx.py` combines PDE and
boundary terms into one graph, converges on MLX, and checks its final parameter
against the CPU reference within `2e-3`. The two-layer Poisson MLP integration
in `examples/pinn_mlp_mlx.py` covers second coordinate derivatives, four
device-resident Adam parameters, residual and boundary losses, and CPU parity.
Its MLX-gated integration test reaches residual below `1e-5`, boundary loss
below `1e-5`, and maximum CPU/MLX parameter difference below `2e-3` after
1,500 fixed-shape steps. Release MLX performance validation records three
separate boundaries: a 1,024-collocation Poisson PINN compiles in `37.86 ms`,
runs device-resident Adam at `0.935 ms/step`, and performs diagnostic loss
readback at `0.486 ms/call` on Apple silicon. The batched two-layer MLP
value-and-gradient diagnostic path at batch `4,096` and width `256` compiles
in `402.97 ms` and takes `1.753 ms/call`; its host loss/gradient materialization
is deliberately not compared with device-resident optimizer throughput.
Unretained mini-batch inputs can be supplied to `step(inputs)` and replace only
their host binding; retained parameters and Adam moments stay on the device.
Contiguous rank-N `Slice` and reverse-mode `PadSlice` now lower to MLX
device-side take/pad operations; primal and symbolic-VJP parity are covered
against the CPU reference.

Status: complete. Larger-workload performance validation (2026-09-27) ran on
an Apple M3 (16 GB, macOS 27.0 build 26A428) with the release
`--features mlx` extension, `f32`, and `mlx.gpu`. Each configuration ran in
five separate processes, interleaved round-robin across configurations; the
1-minute load average before each run was 3.0-5.0, with the resident
`dasd` system daemon near one core. Values are median (min-max) in
milliseconds. Compile is the wall time to build the optimizer from Python
(trace, symbolic derivatives, and `mlx_adam_loss_optimizer`); the first loss
evaluation and warm-up steps are excluded. Step is the mean over the timed
`optimizer.step()` calls; each evaluates the updated parameters and Adam
moments on the GPU stream and performs no host readback. Readback is the mean
of 20 `loss()` calls; each re-evaluates the loss on the device from retained
state and copies one scalar to the host. `examples/benchmark_pinn_mlx.py
--hidden-layers N --hidden-width W` replaces the one-parameter model with an
N-hidden-layer tanh MLP on the same Poisson residual and boundary loss (Adam
learning rate `0.01`, Glorot-normal weights from key `2026`, zero biases);
the default command is unchanged. `examples/benchmark_loop_value_and_grad_mlx.py`
now reports compile and 20-call readback timing beside its step timing.

| Workload | Shape | Warm-up / timed steps | Compile | Step | Readback |
| --- | --- | --- | --- | --- | --- |
| One-parameter Poisson PINN | 65,536 collocation + 2 boundary | 100 / 1,000 | 31.40 (30.79-32.20) | 1.317 (1.180-1.490) | 0.956 (0.722-1.068) |
| MLP PINN, 3 x 128 tanh (33,409 parameters) | 4,096 + 2 | 20 / 200 | 21.93 (21.59-23.20) | 15.466 (14.927-17.173) | 6.365 (4.330-6.415) |
| MLP PINN, 3 x 128 tanh (33,409 parameters) | 65,536 + 2 | 5 / 30 | 33.10 (31.43-33.66) | 245.669 (241.050-246.310) | 68.353 (66.353-68.669) |
| MLP PINN, 4 x 256 tanh (198,145 parameters) | 16,384 + 2 | 10 / 50 | 28.75 (28.20-32.83) | 186.647 (181.991-188.397) | 53.500 (51.002-53.615) |
| MLP PINN, 4 x 512 tanh (789,505 parameters) | 4,096 + 2 | 10 / 50 | 39.36 (38.16-40.51) | 121.024 (119.548-123.108) | 34.772 (33.147-35.120) |
| Fixed-bound `fori` loop Adam | width 256, length 16 | 20 / 100 | 22.59 (20.65-22.85) | 7.162 (6.489-7.222) | 6.736 (6.241-6.977) |
| Fixed-bound `fori` loop Adam | width 65,536, length 16 | 20 / 100 | 21.62 (20.85-21.78) | 7.995 (7.784-8.611) | 7.714 (7.163-8.258) |
| Fixed-bound `fori` loop Adam | width 65,536, length 64 | 10 / 50 | 21.49 (21.09-22.53) | 30.721 (29.229-31.444) | 30.409 (28.001-31.576) |

Loop step time grows with loop length (16 to 64: 8.0 to 30.7 ms) and changes
little with width (256 to 65,536 at length 16: 7.2 to 8.0 ms). A 4 x 512 MLP
PINN at 65,536 collocation points did not fit: system-wide free memory fell to
0% and the process was killed with exit status 137. Final losses were
identical to the printed precision across the five runs of every
configuration. Loss fell from
`48.83` to `4.62e-3` (3 x 128, 4,096 points, 220 steps), from `48.61` to
`0.113` (4 x 256, 16,384 points, 60 steps), and from `48.85` to `3.23` (3 x 128,
65,536 points, 35 steps). At learning rate `0.01` the 4 x 512 MLP settles on
the `u'' = 0` plateau (`48.68`, the mean squared forcing over the `N + 2`
samples) within 60 steps and stays there through 510 steps. At 256 points the
CPU and MLX losses after 30 steps agree (`48.3435`) and MLX reaches that
size's plateau (`48.3270`) by step 300, so this is an optimizer setting rather
than a backend difference.
CPU parity used the same graph and initialization with CPU Adam: for 3 x 128 at
4,096 points after 220 steps, the maximum parameter difference was `3.57e-4`
and the losses were `4.6176e-3` (CPU) and `4.6187e-3` (MLX); for 4 x 512 at
256 points after 30 steps, the maximum parameter difference was `5.43e-4`.
Both are within the documented `2e-3` `f32` tolerance. Continuing the
3 x 128, 4,096-point run to 3,000 MLX steps gave loss `1.60e-2`, so Adam at
this learning rate is not monotone after step 220. These are single-device
Apple silicon measurements; no MLX result is presented as a CUDA or JAX
comparison.

Acceptance checks:

- The P1-style one-parameter Poisson PINN runs on MLX and agrees with CPU
  within its `f32` tolerance. Completed by
  `test_mlx_poisson_pinn_matches_cpu_reference`.
- Representative larger MLP/PINN and fixed-bound loop workloads report
  compile, device-step, and diagnostic-readback time separately over at least
  three processes, and a larger MLP PINN matches CPU within `2e-3`. Completed
  by the 2026-09-27 validation above.

### P7. Distributed Sharding

Goal: add multi-device execution only after single-device plans, layout, and
memory contracts are stable.

Implementation status (2026-07-30): typed `TensorPlacement` metadata now
models unplaced, single-device, and mesh placement. Mesh validation rejects
duplicate devices, invalid topology, unknown mesh axes, out-of-range tensor
axes, and static extents that cannot be evenly sharded. Existing Python
`kernel_ir()` output continues to project the default placement as
`"unplaced"`; there is no distributed execution, implicit copy, or collective.
`kernel_ir_with_placement_propagation(...)` is an opt-in compiler check that
propagates supported local placements through elementwise operations,
transpose, and broadcast. It rejects mixing a placed tensor with an unplaced
non-scalar, reductions over a sharded axis, sharded matmul/solve/triangular
operations, and reshape, slice, or concat cases that require redistribution,
such as concatenation along a sharded axis. It does not insert copies or
collectives.
`TensorExecutionPlan::sharding_plan(...)` extends that contract for legal
sharded reductions: it emits an ordered backend-neutral `TensorAllReduce`
schedule after each replica-local reduction and changes that result's
placement to replicated on the same mesh. The schedule distinguishes `Sum`
from `Mean`; a CUDA/NCCL backend must still lower and execute it. Sharded
matmul, solve, triangular, reshape, slice, and concat redistribution remain
rejected.
`TensorCpuExecutionPlan.value_and_grad_data_parallel(...)` exposes the same
reference from Python. `TensorExecutionPlan::value_and_vjp_data_parallel(...)`
is the deterministic CPU scalar-loss reference: it rebuilds a shape-specialized
IR for equal axis-zero shards, aggregates replica-local `Sum` or `Mean` losses
explicitly, concatenates mapped-input gradients, and adds replicated gradients
in increasing shard order. It has single-device VJP parity coverage but does
not yet spawn CPU workers.

The optional Linux `cuda-nccl` feature now adds
`CudaBackend::compile_data_parallel(plan, device_ordinals)`. It creates one
identical retained-output CUDA plan per explicitly selected GPU;
`CudaDataParallelExecutionPlan::execute_replicas(...)` accepts already-sharded
input maps, enqueues all replicas, NCCL-all-reduces every retained output in
place with `Sum` or `Mean`, and reports separate host-observed enqueue,
collective, and rank-zero readback durations. Ordinary `cuda` builds neither
link nor load NCCL, and the API returns a feature error instead of silently
falling back to one GPU. The execution primitive does not yet consume a
`TensorShardingPlan`, infer general input sharding, or implement cross-node
communicator setup. Python now exposes the same restricted contract through
`tensor_value_and_grad_data_parallel_cuda_fn(...)`: full-batch mapped inputs
are split on axis zero, requested parameters must be replicated, and the
callable returns only all-reduced parameter gradients plus timing diagnostics.
Mapped-input gradients are deliberately rejected rather than incorrectly
all-reduced. On the single-GPU Linux host, Linux feature compilation, the core
device contract, and the Python constructor contract pass; the host exposes
one GTX 1660 SUPER and no NCCL library, so no collective or two-GPU numerical
result is claimed.

Implementation status (2026-09-27): remaining item 2 is implemented for the
first data-parallel subset. `TensorExecutionPlan::cuda_data_parallel_program(
sharding, device_ordinals)` is a backend-neutral check and lowering: it
re-derives the schedule from the plan to prove provenance, then returns the
axis-zero shard specialization with unchanged node ids plus the ordered
`(node, Sum|Mean)` collectives taken from `TensorShardingPlan::all_reduces`.
It rejects schedules without a collective; meshes that are not 1-D, contain
non-CUDA devices, or whose ordinals differ from `device_ordinals` in rank
order or count; inputs sharded on a non-zero axis or with different batch
extents; all-reduces on non-output or consumed nodes, which would need a
mid-graph collective; retained outputs without an all-reduce; and replica
shapes that disagree with placement, such as a replicated full-batch operand
combined with a shard. Concatenation along the sharded axis is already
rejected by the re-derived propagation.
`CudaBackend::compile_data_parallel_sharded(plan, &sharding, device_ordinals)`
binds that list, and `CudaDataParallelExecutionPlan::execute_sharded(...)`
issues one NCCL all-reduce per entry in schedule order, so mixed `Sum`/`Mean`
schedules are executable. `compile_data_parallel` and `execute_replicas(...,
reduction)` keep their caller-selected contract; on a schedule-bound plan,
`execute_replicas` rejects a reduction that disagrees with any scheduled
collective. Each collective's per-rank NCCL calls are now issued inside an
NCCL group, because one thread drives every rank; this also changes the
existing caller-reduction path. The Python value-and-gradient callable still
uses its explicit `reduction`: its parameter gradients are replica-local
partial sums, which `TensorShardingPlan` cannot express because partial
placements and sharded matmul propagation are not modeled.
Verification boundary: macOS CPU tests cover the lowering, a CPU simulation of
the mixed schedule against the global CPU plan, and every rejection above.
On the single-GPU Linux host (GTX 1660 SUPER), the `cuda` suite, `cuda` and
`cuda-nccl` clippy, the `cuda-nccl` test build, and the Python matrix pass;
that host has one GPU and no NCCL library, so the NCCL-gated two-GPU test
returns early there.

Two-GPU verification (2026-09-27): the two-GPU parity run on a single
Linux node (2x RTX 3090, driver 560.35.05, CUDA 12.6) tested commit
`6ea5e7d`. The Python caller-reduction callable matches the deterministic
CPU oracle exactly for both reductions (`Sum`: loss 35.75, weight gradient
-45.0; `Mean`: loss 17.875, gradient -22.5; absolute error 0.0). The
mixed-schedule Rust test matches the global CPU plan within `1.7e-7`
(`Sum` output) and `5.0e-9` (`Mean` output). Each is a single call, so the
reported collective durations (0.63-1.08 s) are not steady-state collective
costs. An earlier run of the same Python parity against the pre-`6ea5e7d`
extension, whose per-rank NCCL calls were not grouped, hit its 20-minute
time limit; with grouping, the parity run finished in 7 s. Commit `07aa493`
later changed how the data-parallel path freezes its program; a rerun of
both checks at `17cbc02` on the same node reproduced the same values and
errors.

Two-GPU training verification (2026-09-27): the data-parallel plan created its
NCCL communicators (`ncclCommInitAll`) on every call and dropped them, issuing
`ncclCommAbort`, before synchronizing the replicas; the single-call collective
durations above were mostly that initialization. Commit `424bf82` creates the
communicators once in `compile_data_parallel` and keeps them for the plan's
lifetime behind a mutex that serializes invocations.
`examples/validate_data_parallel_training_cuda.py` trains a 2-32-32-1 tanh MLP
on a fixed 16x16 grid (256 rows, mapped `x`/`target`) for 200 host-side SGD
steps (learning rate 0.1) from identical f32-rounded Glorot weights and zero
biases in four modes: the two-GPU callable with `mean`, the same with `sum`
(loss and learning rate divided by 2), the single-GPU callable on the full
batch, and the full-batch f64 CPU value-and-gradient. A pair fails if any
step's loss differs by more than `1e-4` relative to the second mode of the
pair, or if the final parameters differ by more than `1e-4` in `max|a - b| /
max|b|` over all parameters. `1e-4` is about 6.5 times the f32 worst-case
rounding bound for the 256-row batch mean (`256 * 2^-24`), while reduction
bugs such as a `Sum`/`Mean` mix-up give O(1) differences. The two-GPU training
run on the same node (2x RTX 3090, driver 560.35.05, CUDA 12.6, NCCL 2.24.3)
tested the tree of `1fadc03` and passed; an earlier run produced identical
differences but exited non-zero because its 50-step baseline run did not meet
the loss-halving check. Loss fell from 0.3024 to 0.1335 in every mode.

| Pair | Max loss rel. error | Max param abs. error | Param normwise rel. error |
| --- | --- | --- | --- |
| two-GPU mean vs single GPU | 1.9e-7 | 1.9e-8 | 2.1e-8 |
| two-GPU mean vs CPU (f64) | 2.7e-7 | 1.4e-7 | 1.6e-7 |
| single GPU vs CPU (f64) | 3.1e-7 | 1.4e-7 | 1.6e-7 |
| two-GPU sum vs two-GPU mean | 0 | 0 | 0 |

Timing is host-observed per call; steady state is steps 10-199 (median,
min-max). The baseline row is the pre-fix extension (`9251fee`) run in the
same run with the same 200 steps; it reproduced the final loss exactly.

| Two-GPU `mean` | First step | Steady median | Steady min-max |
| --- | --- | --- | --- |
| collective, pre-fix | 1.03 s | 0.634 s | 0.619-0.653 s |
| collective | 62.7 ms | 0.106 ms | 0.103-0.325 ms |
| replica enqueue | 98.2 ms | 0.401 ms | 0.382-3.91 ms |
| output readback | 0.087 ms | 0.049 ms | 0.047-0.142 ms |
| call wall | 161 ms | 0.581 ms | 0.557-4.51 ms |

The single-GPU call takes 0.239 ms (steady median) on the full batch, so
at this problem size two GPUs are slower per step; the check establishes
correctness and the collective cost, not a speedup. Parameter updates run
on the host, gradients return to the host every step, and only one node
with two GPUs was measured.

Failure handling and communicator lifecycle (2026-09-27): a failed
data-parallel call kept its communicators, although a failure while
enqueuing one rank's all-reduce inside the NCCL group can leave the
collective launched on the other ranks only. Commit `32baba1` defines the
contract: any error after the communicator lock is taken is returned with
the note that the communicators were aborted, the failed call drops them
(`ncclCommAbort`), and the next call on the plan or any clone creates new
ones before enqueuing; a panic while the lock is held still poisons it and
rejects every later call. The natural trigger is a replica whose input map
lacks its shard: it fails input validation after replica 0 has enqueued
its work. No fault-injection seam was added, so an error returned by NCCL
itself takes the same path but is not exercised. The two-GPU robustness run
on the same node (2x RTX 3090, driver 560.35.05, NCCL 2.24.3) tested the
tree of `f1122c7`:

- Recovery: after a successful call, two consecutive failed calls each
  returned `missing input "x"` with the reset note (the second first
  recreated the communicators); the next three calls through a clone
  matched the CPU plan within `1e-5`. The first recovered call took
  109 ms wall, which includes creating the communicators and a 67.6 ms
  first collective on them (first collectives of fresh plans in the same
  run take 43-71 ms); the next two took 0.33 ms and 0.25 ms.
- Lifecycle: 50 compile, execute, drop cycles in 37.9 s, each matching
  the CPU plan within `1e-5`; odd cycles dropped the plan right after a
  failed call that left replica 0's work enqueued. With the test holding
  both primary contexts, free device memory after every cycle was
  24,929,959,936 bytes on both GPUs: zero drift, 116 MiB below the level
  before the first plan. `nvidia-smi` reported 2 MiB used on both GPUs
  before and after the tests. No call or drop hung; every step ran under
  a 300 s timeout.

Larger-scale training (2026-09-27): the same run executed the validation at
16,384 and 65,536 rows (`--grid 128`/`256`) and widths 256 and 512 for 50
SGD steps (learning rate 0.1, steady state steps 10-49), comparing the
two-GPU callable with the single-GPU callable; the f64 CPU mode is too
slow at these sizes. The tolerance, fixed before the run, keeps its ratio
to the worst-case rounding bound of the batch mean and so grows linearly
above 256 rows (`6.4e-3` at 16,384, `2.56e-2` at 65,536). At that level a
50-step trajectory whose loss barely moves would not expose a `Sum`/`Mean`
mix-up (a CPU simulation with doubled learning rate stays at `7.0e-3`
loss and `2.1e-2` parameter error), so every pair now also compares the
first step's gradients, evaluated at identical inputs, where such a bug is
an O(1) normwise error. Plain SGD at these widths only leaves its initial
plateau after a few hundred steps, so these runs require only that the
loss does not increase (`--max-loss-ratio 1.0`). The default 256-row run
in the same run reproduced the training run (loss 0.3024 to 0.1335, identical
error levels, first-step gradient error at most `2.0e-7`).

| Rows | Width | Loss | Max loss rel. error | Param normwise rel. error | First-step gradient rel. error |
| --- | --- | --- | --- | --- | --- |
| 16,384 | 256 | 0.2656 to 0.2501 | 3.6e-7 | 2.0e-8 | 3.2e-6 |
| 16,384 | 512 | 0.2527 to 0.2498 | 3.6e-7 | 3.6e-8 | 5.0e-6 |
| 65,536 | 256 | 0.2657 to 0.2501 | 7.1e-7 | 3.8e-8 | 3.2e-6 |
| 65,536 | 512 | 0.2526 to 0.2499 | 7.8e-7 | 2.9e-8 | 5.3e-6 |

The errors are two-GPU `mean` against the single GPU. Two-GPU `sum`
against `mean` gives zero parameter and first-step gradient error in
every configuration and at most `2.4e-7` loss error, consistent with the
atomic accumulation of the scalar loss.

| Rows | Width | Single-GPU call | Two-GPU call | Enqueue | Collective | Readback | Two-GPU speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | 32 | 0.250 ms | 0.591 ms | 0.402 ms | 0.107 ms | 0.049 ms | 0.42x |
| 16,384 | 256 | 3.04 ms | 2.84 ms | 0.76 ms | 1.01 ms | 0.17 ms | 1.07x |
| 16,384 | 512 | 8.43 ms | 14.8 ms | 1.66 ms | 2.07 ms | 1.61 ms | 0.57x |
| 65,536 | 256 | 10.8 ms | 6.68 ms | 1.03 ms | 3.79 ms | 0.18 ms | 1.62x |
| 65,536 | 512 | 24.9 ms | 22.6 ms | 1.85 ms | 7.78 ms | 1.64 ms | 1.10x |

All values are steady-state medians of host-observed call time for
two-GPU `mean`; speedup is the single-GPU call divided by the two-GPU
call. The collective interval includes waiting for the replicas' compute,
because enqueue returns before the kernels finish, so at these sizes it is
not the NCCL cost alone. Two GPUs are faster at 16,384 rows and width 256
and at both 65,536-row sizes, and slower at 16,384 rows and width 512,
where about 9 ms of the two-GPU call lies outside the three recorded
intervals and is not attributed. One node, one run, and host-side
parameter updates were measured.

Remaining implementation order:

1. Validate the Python scalar value-and-gradient callable on two GPUs against
   the deterministic CPU oracle for both `Sum` and `Mean`; only then consider
   mapped-input gradient concatenation as a separate output contract.
   Single-call parity verified (2026-09-27, two-GPU parity run); 200-step
   training parity and steady-state collective timing verified (2026-09-27,
   two-GPU training run).
2. Bind `TensorShardingPlan::all_reduces` to CUDA lowering, preserving its
   operation order and rejecting schedules not represented by the first
   data-parallel subset. Implemented and two-GPU parity verified
   (2026-09-27, two-GPU parity run).
3. Add failure-handling and communicator-lifecycle coverage before considering
   multi-node transport or tensor-parallel matmul. Communicators are now
   created once per compiled plan (2026-09-27). Failure recovery and 50
   compile/execute/drop cycles verified on two GPUs (2026-09-27, two-GPU
   robustness run);
   an error returned by NCCL itself and a drop while a collective is in
   flight are not exercised, because the public API cannot reach them
   without fault injection.

Acceptance checks:

- Two-GPU data-parallel training matches the single-GPU reference within
  documented tolerance and records collective timing separately. Met
  (2026-09-27, two-GPU training run) for the host-SGD MLP check above.

### Research Track: Rust Source-To-Source AD

This track is intentionally not on the critical path. Start with a procedural
macro over a pure, restricted Rust expression subset and compare it against
Tensor IR AD and Enzyme. Do not claim support for arbitrary borrowing,
mutation, dynamic dispatch, async code, or Python callbacks until their
semantics are explicitly modeled and tested.

Implementation status (2026-07-27): the `quabla-macros` procedural-macro crate
exports `forward_diff!(|x| expression)`. It parses a one-argument closure and
rewrites the parameter to a `Dual::variable` seed at expansion time, so Rust's
operator and method resolution produces forward-mode value/derivative pairs
without a runtime tape. Analytic tests cover captured scalar constants, block
expressions, elementary methods, and `powi`. This is a source-level prototype,
not general source-to-source AD: mutation, arbitrary control flow, multiple
outputs, and type-generic captures remain unsupported.
`forward_gradient!(|x, y, ...| scalar_expression)` now adds an ordered
multi-input scalar gradient by expanding one Dual seed per input and returning
`ForwardGradient`. It is intentionally O(input dimension); dense Jacobian,
HVP, and reverse-mode source transforms remain future work.
