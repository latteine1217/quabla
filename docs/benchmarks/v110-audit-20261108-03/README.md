# 1.1.0 robustness audit, current snapshot — 2026-11-08

Status: superseded by [snapshot 04](../v110-audit-20261108-04/README.md).
Local Python checks passed for its standard-deviation gradient repair, but
this source retained snapshot 02's CUDA-only constant-table declaration
error. Dependent job 6946 was cancelled without starting after 6945 failed;
it supplies no CUDA result. Snapshot 04 corrects that declaration and runs
complete native and Python CUDA/NCCL acceptance.

Source snapshots, earlier local logs and timing samples in this directory
remain historical evidence; they do not describe current acceptance.
