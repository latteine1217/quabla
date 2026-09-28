# Repository Conventions

These conventions apply to AI assistants and human contributors alike.

## Language

Write every code comment, docstring, commit message, pull-request text, and
document in this repository in English. This repository rule takes precedence
over any personal or global instruction to write comments or docs in another
language; conversational replies to the user may still follow that user's
preference.

Comments state what the code does and why it is designed that way, in the
style of the surrounding code.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/) with a
lowercase English subject (`fix: ...`, `docs: ...`). The body explains what
was wrong, why the change fixes it, and which verification was actually run.

## Verification

The required gates (fmt, clippy, tests, ruff, the Python matrix) are listed
under "Development Gates" in `README.md`; run the ones that cover the change.
GPU runtime suites are opt-in through `QUABLA_MLX_TEST=1`,
`QUABLA_CUDA_TEST=1`, and `QUABLA_CUDA_NCCL_TEST=1`, and CUDA code only
compiles on Linux.

Every `unsafe` block or impl carries a `// SAFETY:` comment naming the
invariant that makes it sound.
