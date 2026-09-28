# Security Policy

## Supported Versions

Quabla is a research project at version 0.x. Only the latest release series
receives fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Reporting a Vulnerability

Please report vulnerabilities privately through GitHub private vulnerability
reporting: open the repository's **Security** tab and select
**Report a vulnerability**
(<https://github.com/latteine1217/quabla/security/advisories/new>). If you
cannot use GitHub, email felix.tc.tw@gmail.com instead.

Do not open a public issue, pull request, or discussion for a suspected
vulnerability.

A useful report includes:

- the affected version or commit and the build features (`cpu`, `mlx`,
  `cuda`, or `cuda-nccl`);
- the operating system, Python version, and, for GPU builds, the driver and
  CUDA/NCCL versions;
- a minimal reproduction and the observed impact (for example memory
  corruption, a crash reachable from Python, or unintended code execution).

## Response

Quabla is maintained on a best-effort basis. Reports are acknowledged and
investigated as time allows; there is no guaranteed response or fix time.
When a report is confirmed, the fix and a GitHub security advisory are
published together, crediting the reporter unless they ask otherwise.
