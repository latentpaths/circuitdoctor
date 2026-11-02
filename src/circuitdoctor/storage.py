from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def read_examples(path: Path):
    from .schema import Example

    return [
        Example.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()
    ]


def instrumentation_digest() -> str:
    root = Path(__file__).parent
    names = (
        "runtime.py",
        "storage.py",
        "tools.py",
        "patch.py",
        "metrics.py",
        "schema.py",
        "agent.py",
        "selection.py",
        "baselines.py",
        "budget.py",
        "prime.py",
        "evaluate.py",
        "vllm_runtime.py",
    )
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names})


def environment_versions() -> dict:
    import platform
    from importlib.metadata import PackageNotFoundError, version

    result = {"python": platform.python_version()}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "safetensors",
        "math-verify",
        "vllm",
        "flash-linear-attention",
        "fla-core",
        "causal-conv1d",
        "triton",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result
