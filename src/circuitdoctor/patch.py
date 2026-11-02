"""Exact BF16 edits and rollback; no lossy float delta reconstruction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .schema import PatchManifest, PatchTensor
from .storage import read_json, write_json


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def export_patch(
    path: Path,
    base: dict[str, torch.Tensor],
    edited: dict[str, torch.Tensor],
    *,
    model: str,
    revision: str,
    label: str,
    provenance: dict | None = None,
    evaluation: list[str] | None = None,
) -> PatchManifest:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite patch: {path}")
    if not base or set(base) != set(edited):
        raise ValueError("Patch requires matching nonempty tensor sets")
    entries = []
    for name, original in base.items():
        changed = edited[name]
        if original.shape != changed.shape or original.dtype != changed.dtype:
            raise ValueError(f"Shape/dtype mismatch: {name}")
        if not torch.isfinite(changed).all():
            raise ValueError(f"Nonfinite tensor: {name}")
        entries.append(
            PatchTensor(
                name=name,
                shape=list(original.shape),
                dtype=str(original.dtype),
                base_sha256=tensor_hash(original),
                edited_sha256=tensor_hash(changed),
                changed_entries=int((original != changed).sum()),
            )
        )
    manifest = PatchManifest(
        base_model=model,
        revision=revision,
        label=label,
        tensors=entries,
        provenance=provenance or {},
        evaluation=evaluation if evaluation is not None else ["evidence.json"],
    )
    path.mkdir(parents=True)
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in base.items()}, path / "original.safetensors"
    )
    save_file(
        {k: v.detach().cpu().contiguous().clone() for k, v in edited.items()}, path / "edited.safetensors"
    )
    write_json(path / "manifest.json", manifest.model_dump())
    write_json(path / "evidence.json", {"status": "not evaluated", "references": []})
    return manifest


def apply_patch(
    tensors: dict[str, torch.Tensor],
    path: Path,
    *,
    revert: bool = False,
    model: str | None = None,
    revision: str | None = None,
):
    manifest = PatchManifest.model_validate(read_json(path / "manifest.json"))
    if model is not None and model != manifest.base_model:
        raise ValueError("Patch model does not match")
    if revision is not None and revision != manifest.revision:
        raise ValueError("Patch revision does not match")
    source = load_file(path / ("original.safetensors" if revert else "edited.safetensors"))
    if set(source) != {t.name for t in manifest.tensors}:
        raise ValueError("Payload tensor set differs from manifest")
    # Validate every tensor before mutating any: rejected patches cannot partially apply.
    for t in manifest.tensors:
        if t.name not in tensors:
            raise ValueError(f"Missing tensor: {t.name}")
        current, replacement = tensors[t.name], source[t.name]
        expected = t.edited_sha256 if revert else t.base_sha256
        target = t.base_sha256 if revert else t.edited_sha256
        if (
            list(current.shape) != t.shape
            or str(current.dtype) != t.dtype
            or replacement.shape != current.shape
            or replacement.dtype != current.dtype
        ):
            raise ValueError(f"Shape/dtype mismatch: {t.name}")
        if tensor_hash(current) != expected or tensor_hash(replacement) != target:
            raise ValueError(f"Tensor hash mismatch: {t.name}")
    with torch.no_grad():
        for t in manifest.tensors:
            tensors[t.name].copy_(source[t.name].to(tensors[t.name].device))
    return manifest


def suppress_weights(
    weight: torch.Tensor, direction: torch.Tensor, strength: float, columns: list[int] | None = None
) -> torch.Tensor:
    u = direction.to(device=weight.device, dtype=torch.float32)
    if u.ndim != 1 or u.numel() != weight.shape[0] or not torch.isfinite(u).all() or u.norm() < 1e-8:
        raise ValueError("Invalid suppression direction")
    u = u / u.norm()
    out = weight.float().clone()
    selected = list(range(out.shape[1])) if not columns else sorted(set(columns))
    if min(selected) < 0 or max(selected) >= out.shape[1]:
        raise ValueError("Column out of range")
    w = out[:, selected]
    out[:, selected] = w - strength * torch.outer(u, u @ w)
    return out.to(weight.dtype)


def record_evidence(bundle: Path, report: Path, kind: str):
    """Attach a portable derived report without changing the frozen weight manifest."""
    if kind not in ("validation", "test"):
        raise ValueError("Unknown evaluation role")
    content = report.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    target = bundle / "evaluations" / f"{kind}-{sha}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != content:
        raise ValueError("Evidence artifact collision")
    target.write_bytes(content)
    index_path = bundle / "evidence.json"
    index = read_json(index_path) if index_path.exists() else {"references": []}
    reference = {"kind": kind, "file": str(target.relative_to(bundle)), "sha256": sha}
    if reference not in index["references"]:
        index["references"].append(reference)
    index["status"] = (
        "test evaluation attached"
        if any(r["kind"] == "test" for r in index["references"])
        else "validation only"
    )
    write_json(index_path, index)
