from __future__ import annotations

from contextlib import ExitStack
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .patch import export_patch, tensor_hash
from .schema import RunConfig
from .storage import read_json


def reproduce(base_file: Path | None, out: Path, source: Path | None = None):
    source = source or Path(str(files("circuitdoctor") / "resources/published-p50"))
    manifest = read_json(source / "manifest.json")
    if base_file is None:
        from huggingface_hub import hf_hub_download

        config = RunConfig()
        index = Path(hf_hub_download(config.model, "model.safetensors.index.json", revision=config.revision))
        mapping = read_json(index)["weight_map"]
        for file in {mapping[t["name"]] for t in manifest["tensors"]}:
            hf_hub_download(config.model, file, revision=config.revision)
        base_file = index.parent
    base, edited = {}, {}
    if base_file.is_dir():
        index = base_file / "model.safetensors.index.json"
        mapping = (
            read_json(index)["weight_map"]
            if index.exists()
            else {t["name"]: "model.safetensors" for t in manifest["tensors"]}
        )
        tensor_files = {name: base_file / file for name, file in mapping.items()}
    else:
        tensor_files = {t["name"]: base_file for t in manifest["tensors"]}
    with np.load(source / "edit.npz", allow_pickle=False) as z, ExitStack() as stack:
        opened = {}
        for i, entry in enumerate(manifest["tensors"]):
            name = entry["name"]
            shard = tensor_files[name]
            if shard not in opened:
                opened[shard] = stack.enter_context(safe_open(shard, framework="pt"))
            tensor = opened[shard].get_tensor(name)
            if tensor.dtype != torch.bfloat16 or tensor_hash(tensor) != entry["base_sha256"]:
                raise ValueError(f"Published base tensor hash mismatch: {name}")
            if entry["form"] != "sparse":
                raise ValueError("This reproduction imports the sparse p50 patch only")
            changed = tensor.clone()
            bits = changed.view(torch.int16).reshape(-1)
            values = torch.from_numpy(z[f"t{i}_val"].copy().view(np.int16))
            bits[torch.from_numpy(z[f"t{i}_idx"].astype(np.int64))] = values
            if tensor_hash(changed) != entry["edited_sha256"]:
                raise ValueError(f"Published edited tensor hash mismatch: {name}")
            base[name], edited[name] = tensor, changed
    config = RunConfig()
    if out.exists():
        from safetensors.torch import load_file

        from .schema import PatchManifest

        existing = PatchManifest.model_validate(read_json(out / "manifest.json"))
        if existing.base_model != config.model or existing.revision != config.revision:
            raise ValueError("Existing reproduction is for another model/revision")
        old, new = load_file(out / "original.safetensors"), load_file(out / "edited.safetensors")
        if (
            set(old) != set(base)
            or set(new) != set(edited)
            or any(
                tensor_hash(old[k]) != tensor_hash(base[k]) or tensor_hash(new[k]) != tensor_hash(edited[k])
                for k in base
            )
        ):
            raise ValueError("Existing reproduction payload differs")
        return existing
    return export_patch(
        out,
        base,
        edited,
        model=config.model,
        revision=config.revision,
        label="published patch reproduction",
        provenance={
            "source": "https://rikiyat.github.io/blog/weight-space-programming/",
            "experiment": "8, Qwen3.5-0.8B, p50c",
            "license": "CC BY 4.0",
            "authors": "Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab",
            "verification": "All base and rebuilt tensor SHA256 values match the supplied manifest",
            "touched_entries_reported": 131072,
            "note": "Bitwise changed entries exclude touched entries rounded back to original BF16 values",
        },
    )
