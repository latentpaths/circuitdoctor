"""Export the canonical System skill as a generated distributable, preserving one editable source."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("dist/circuitdoctor-skill.zip"))
args = parser.parse_args()
if not (args.source / "SKILL.md").is_file():
    parser.error("Source must contain SKILL.md")
args.output.parent.mkdir(parents=True, exist_ok=True)
manifest = {}
with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(args.source.rglob("*")):
        if path.is_file() and "backlog" not in path.parts:
            name = "circuitdoctor/" + str(path.relative_to(args.source))
            archive.write(path, name)
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    archive.writestr("export-manifest.json", json.dumps(manifest, indent=2))
print(json.dumps({"output": str(args.output), "files": len(manifest)}))
