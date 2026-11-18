# Keep one execution plan interpreter per backend

The CPU evaluator, the CUDA device program, and the MLX lowering each walk an
execution plan with their own loop instead of sharing one plan walker behind a
per-op backend adapter. What they share is small: visit the nodes in order,
keep one value per node, release a value after its last use, and evaluate a
region group once at its first member. Each also hides a different model of
value lifetime:

- The CPU releases values only when the caller names its outputs, reuses an
  operand's buffer in place when that use is its last, and re-runs a fused
  plan node by node when a fused program meets the NaN case it cannot match.
- CUDA counts remaining uses, returns released buffers to a pool, keeps its
  value table across calls, protects retained inputs, runs separate fusion
  regions, and computes every member of a region group up front, moving each
  result out when its member is reached.
- MLX arrays are lazy, so a released handle frees memory only for an array
  that is already evaluated; it has no fusion and copies each member's result
  out of a group evaluation.

A shared walker would carry all of this in its adapter interface: reuse and
pool hooks, two fusion models, and one result type per region kind and
backend, while the logic it absorbs is a node loop and a map of group
results. The interpreters share the rules instead: the MLX lowering uses the
CPU's last-use analysis (`tensor_forward_last_uses` and
`release_consumed_values`) to release values and group results after their
last use.

Revisit this if the interpreters converge on one value-lifetime model, for
example if MLX gains fusion or CUDA stops keeping its value table across calls.
