"""Check release files and preserve hashes for registry verification."""
from email.parser import BytesParser
from pathlib import Path
import hashlib
import json
import sys
import tarfile
import zipfile

root = Path(sys.argv[1])
files = sorted(p for p in root.rglob('*') if p.suffix == '.whl' or p.name.endswith('.tar.gz'))
assert len(files) == 21, [p.name for p in files]
assert sum(p.suffix == '.whl' for p in files) == 20
assert len({p.name for p in files}) == 21
records = []
combinations = set()
for path in files:
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
            assert len(metadata) == 1
            info = BytesParser().parsebytes(archive.read(metadata[0]))
            name, version = info['Name'], info['Version']
            assert version == '1.0.3' and name in ('quabla', 'quabla-mlx')
            interpreter = path.name.split('-')[2]
            if name == 'quabla-mlx':
                platform = 'macos-mlx'
                assert 'quabla/mlx.metallib' in names
                assert any(n.endswith('/MLX.txt') for n in names)
            else:
                assert 'quabla/mlx.metallib' not in names
                platform = ('linux-aarch64' if 'aarch64' in path.name else
                            'linux-x86_64' if 'x86_64' in path.name else 'macos-cpu')
            if platform.startswith('macos'):
                assert 'macosx_' in path.name and 'arm64' in path.name
            combinations.add((platform, interpreter))
    else:
        with tarfile.open(path) as archive:
            entries = [n for n in archive.getnames() if n.endswith('/PKG-INFO')]
            assert len(entries) == 1
            info = BytesParser().parsebytes(archive.extractfile(entries[0]).read())
            name, version = info['Name'], info['Version']
            assert (name, version) == ('quabla', '1.0.3')
    records.append({'filename': path.name, 'distribution': name, 'version': version,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
assert combinations == {(p, i) for p in ('linux-aarch64', 'linux-x86_64', 'macos-cpu', 'macos-mlx')
                         for i in ('cp310', 'cp311', 'cp312', 'cp313', 'cp314')}
print(json.dumps({'status': 'passed', 'files': records}, indent=2))
