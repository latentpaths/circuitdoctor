"""Fetch three official checker files at a pinned Google Research revision; preserve their license."""

import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "83f1e7b59e762648c953de751d31f75720683622"
ROOT = Path(__file__).resolve().parents[1] / "src/circuitdoctor/vendor/ifeval"
ROOT.mkdir(parents=True, exist_ok=True)
manifest = {"revision": REVISION, "license": "Apache-2.0", "files": {}}
for name in ("instructions.py", "instructions_registry.py", "instructions_util.py"):
    url = f"https://raw.githubusercontent.com/google-research/google-research/{REVISION}/instruction_following_eval/{name}"
    original = urllib.request.urlopen(url, timeout=30).read()
    content = original.decode().replace(
        "from instruction_following_eval import", "from circuitdoctor.vendor.ifeval import"
    )
    (ROOT / name).write_text(content)
    manifest["files"][name] = {
        "url": url,
        "upstream_sha256": hashlib.sha256(original).hexdigest(),
        "local_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "modification": "Package import path only",
    }
(ROOT / "__init__.py").write_text('"""Official IFEval checkers; see provenance.json and LICENSE."""\n')
(ROOT.parent / "__init__.py").touch()
(ROOT / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
license_url = f"https://raw.githubusercontent.com/google-research/google-research/{REVISION}/LICENSE"
(ROOT / "LICENSE").write_bytes(urllib.request.urlopen(license_url, timeout=30).read())
print(json.dumps({"revision": REVISION, "files": list(manifest["files"])}))
