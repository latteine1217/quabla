## Summary

<!-- What changes and why. Link the related issue if there is one. -->

## Verification

Gates run (see CONTRIBUTING.md):

- [ ] `cargo fmt --all --check`
- [ ] `ruff check tests examples python`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo clippy --workspace --all-targets --features quabla-core/mlx -- -D warnings` (macOS)
- [ ] `cargo clippy --workspace --all-targets --features quabla-core/cuda-nccl -- -D warnings` (Linux)
- [ ] `cargo test --workspace`
- [ ] `python tests/python/test_matrix.py` and `python tests/python/test_api.py`
- [ ] GPU suites: `QUABLA_MLX_TEST=1` / `QUABLA_CUDA_TEST=1` / `QUABLA_CUDA_NCCL_TEST=1` (name which)

Hardware used:

<!-- OS, CPU or Apple silicon model, GPU model, driver and CUDA/NCCL versions.
     Name any gate you could not run and why. -->
