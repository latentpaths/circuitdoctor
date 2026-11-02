"""Run isolated adapter checks sequentially, preserving failed logs and allowing core work to continue."""

import os
import subprocess
from pathlib import Path

from circuitdoctor.storage import write_json

root = Path("runs/adapter-smokes")
root.mkdir(parents=True, exist_ok=True)
results = {}
commands = {
    "jacobian-lens": [
        ".venv-lens/bin/python",
        "examples/jacobian_lens.py",
        "--output",
        "runs/jacobian-lens-v3",
    ],
    "circuit-trace": [
        ".venv-tracing/bin/python",
        "examples/circuit_trace.py",
        "--output",
        "runs/circuit-trace",
    ],
}
for name, command in commands.items():
    log = root / (name + ".log")
    with log.open("w") as output:
        completed = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        )
    results[name] = {"exit_code": completed.returncode, "log": str(log)}
    write_json(root / "status.json", results)
print(results, flush=True)
