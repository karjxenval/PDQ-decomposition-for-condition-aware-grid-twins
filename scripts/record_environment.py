from __future__ import annotations
import hashlib, json, platform, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def git_commit():
    try:
        return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        return None

out = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'python': sys.version,
    'platform': platform.platform(),
    'git_commit': git_commit(),
    'validation_config_sha256': sha256(ROOT/'configs'/'validation.json'),
    'source_catalog_sha256': sha256(ROOT/'configs'/'sources.json'),
}
path = ROOT/'data'/'provenance'/'environment.json'
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2), encoding='utf-8')
print(path)
