"""Verify frozen source identity, terminal jobs and numerical acceptance logs."""
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path

run = Path(__file__).resolve().parent
root = run.parents[2]
remote = run / 'remote-final'
source = json.loads((run / 'source.json').read_text())
# These files capture pre-existing workspace context, not runtime changes in
# this commit. The complete archive and original compute-node checks retain
# their original identities; README gained the audit index after validation.
context_only = {'README.md', 'AGENTS.md', 'examples/benchmark_frameworks.py'}
for name, digest in source.items():
    if name not in context_only:
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
with tarfile.open(run / 'source-snapshot.tar.gz') as archive:
    assert set(archive.getnames()) == set(source)
    for name, digest in source.items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, name
accounting = {}
for line in (run / 'accounting.txt').read_text().splitlines():
    fields = line.split('|')
    if fields[0] in ('6947', '6948', '6949', '6950'):
        assert fields[1:3] == ['COMPLETED', '0:0'], fields
        accounting[fields[0]] = dict(state=fields[1], exit_code=fields[2], elapsed=fields[3], node=fields[4])
assert len(accounting) == 4

def suites(path):
    text = path.read_text()
    sections = re.split(r'^=== (test_\S+\.py) ===\n', text, flags=re.M)
    totals = [0, 0, 0]
    for index in range(1, len(sections), 2):
        name, body = sections[index:index + 2]
        matches = re.findall(r'^' + re.escape(name) + r': (\d+) passed, (\d+) skipped, (\d+) failed$', body, re.M)
        assert len(matches) == 1, (path, name)
        totals = [a + int(b) for a, b in zip(totals, matches[0])]
    assert len(sections) == 41 and totals[2] == 0 and text.rstrip().endswith('all suites passed'), path
    return dict(suites=20, passed=totals[0], skipped=totals[1], failed=totals[2])
python = {f'cpu-py{v}': suites(run / f'python-py{v}.log') for v in ('310', '311', '312', '313', '314')}
python['mlx-py313'] = suites(run / 'python-py313-mlx-rebuilt.log')
python['cuda-py313'] = suites(remote / 'python-6947.log')
for filename in ('source-before-6947.log', 'source-after-6947.log',
                 'curvature-source-before-6948.log', 'curvature-source-after-6948.log',
                 'statistics-source-before-6949.log', 'statistics-source-after-6949.log',
                 'nccl-timing-source-before-6950.log', 'nccl-timing-source-after-6950.log'):
    assert (remote / filename).read_text().splitlines() == [f'{name}: OK' for name in source], filename
completion = json.loads((remote / 'completion-6947.json').read_text())
assert completion['status'] == 'passed' and completion['job_id'] == '6947'
identity = (run / 'head-final-identity.log').read_text().splitlines()
assert identity[:-1] == [f'{name}: OK' for name in source]
assert identity[-1].split()[0] == completion['extension_sha256']
for name in ('validate.sbatch', 'device_probes.py', 'cuda_float64_probes.py',
             'spread_probes.py', 'curvature_probes.py', 'statistics_probes.py',
             'nccl_timing_probes.py', 'curvature.sbatch', 'statistics.sbatch', 'nccl-timing.sbatch'):
    assert (run / name).read_bytes() == (remote / name).read_bytes(), name
assert '0 errors, 0 warnings' in (run / 'pyright.log').read_text()
assert 'All checks passed!' in (run / 'ruff.log').read_text()
assert 'advisories ok, bans ok, licenses ok, sources ok' in (run / 'deny.log').read_text()
assert 'Finished' in (run / 'clippy.log').read_text()
assert 'Finished' in (remote / 'clippy-6947.log').read_text()
probe_records = {}
for filename, script, count in (
    ('probes-cuda-6947.json', 'device_probes.py', 140),
    ('float64-6947.json', 'cuda_float64_probes.py', 41),
    ('spread-cuda-6947.json', 'spread_probes.py', 57),
    ('curvature-6948.json', 'curvature_probes.py', 14),
    ('statistics-6949.json', 'statistics_probes.py', 480),
    ('nccl-timing-6950.json', 'nccl_timing_probes.py', 6),
):
    record = json.loads((remote / filename).read_text())
    assert record['status'] == 'passed' and record['checks'] == count, record
    assert record['probe_sha256'] == hashlib.sha256((run / script).read_bytes()).hexdigest(), filename
    probe_records[filename] = record
timings = probe_records['nccl-timing-6950.json']['results']
assert len(timings) == 6
for case in timings:
    assert set(case['timing']) == {'replica_enqueue', 'collective', 'output_readback'}
    assert all(math.isfinite(v) and v >= 0 for v in case['timing'].values())
parity = json.loads((remote / 'parity-6947.json').read_text())
assert parity['status'] == 'passed' and parity['tolerance'] == 1e-5
assert parity['extension_sha256'] == completion['extension_sha256']
assert len(parity['results']) == 8
assert [case['reduction'] for case in parity['results']].count('sum') == 4
assert [case['reduction'] for case in parity['results']].count('mean') == 4
loss_errors, gradient_errors = [], []
for case in parity['results']:
    assert case['reduction'] in ('sum', 'mean') and len(case['steps']) == 5
    for step in case['steps']:
        assert set(step['gradient_abs_errors']) == {'weight', 'bias'}
        assert abs(step['cpu_loss'] - step['cuda_loss']) == step['loss_abs_error']
        loss_errors.append(step['loss_abs_error'])
        gradient_errors.extend(step['gradient_abs_errors'].values())
assert all(math.isfinite(error) and error <= 1e-5 for error in loss_errors + gradient_errors)
rust = {}
for label, path in (('mlx', run / 'rust-mlx.log'), ('cuda-nccl', remote / 'rust-6947.log')):
    text = path.read_text()
    matches = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', text)
    assert matches and 'test result: FAILED' not in text and all(int(m[1]) == 0 for m in matches), path
    rust[label] = dict(passed=sum(int(m[0]) for m in matches), ignored=sum(int(m[2]) for m in matches))
for path in remote.glob('cuda_data_parallel_*-6947.log'):
    assert 'test result: ok. 1 passed; 0 failed;' in path.read_text(), path
assert len(list(remote.glob('cuda_data_parallel_*-6947.log'))) == 3
result = dict(status='passed', source_files=len(source), current_tree_files_checked=len(source) - len(context_only),
              archive_context_only=sorted(context_only), accounting=accounting,
              target_release='1.1.0', package_version_under_test=completion['package_version'], release_published=False,
              actual_remote_extension_sha256=completion['extension_sha256'],
              verification_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              python=python, rust=rust, cuda_probes=probe_records,
              parity=dict(cases=8, steps_per_case=5, max_loss_error=max(loss_errors), max_gradient_error=max(gradient_errors), tolerance=1e-5),
              source_archive_sha256=hashlib.sha256((run / 'source-snapshot.tar.gz').read_bytes()).hexdigest(),
              limitations=['Finite tests cannot prove the absence of all bugs.', 'Metal flushes float32 subnormal results.', 'Reduction timing overhead is observed; resident PINN training was not remeasured.', 'Linux aarch64 and the full Linux wheel/Python matrix remain unverified.'])
(run / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
