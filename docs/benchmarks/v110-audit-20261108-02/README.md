# 1.1.0 robustness audit, current snapshot — 2026-11-08

Status: rejected. Job 6945 FAILED with exit 1 during the CUDA build:
a four-entry constant table still declared an array length of three. It
never ran CUDA numerical acceptance. A later CPU analytical probe also
exposed NaN standard-deviation gradients for adjacent values near 2^1000;
`spread-cpu-failure.log` preserves that reproduction using this snapshot's
Python source and an identical CPU implementation from the final extension.
The current candidate is [snapshot 04](../v110-audit-20261108-04/README.md).

Source snapshots, earlier local logs and timing samples in this directory
remain historical evidence; they do not describe current acceptance.
