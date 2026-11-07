# 1.1.0 robustness audit — 2026-11-08

Status: rejected. The local full suite first exposed loss of offset
precision; the later candidate still changed ordinary zero-cancellation
rounding in Rust. Job 6944 was cancelled after that reproducible Rust failure
was observed. It is not a passing CUDA acceptance run. Raw post-termination
logs are in `remote-final/`; accounting is in `failed-job-accounting.txt`.
The current candidate is [snapshot 04](../v110-audit-20261108-04/README.md).

Source snapshots, earlier local logs and timing samples in this directory
remain historical evidence; they do not describe current acceptance.
