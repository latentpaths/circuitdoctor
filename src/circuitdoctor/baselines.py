from __future__ import annotations

import random
import uuid

import torch
from safetensors.torch import save_file

from .schema import Intervention
from .tools import CaptureArgs, CompareArgs, ExportArgs, InspectArgs, ObserveArgs, ValidateArgs


def scripted(registry, *, random_search: bool = False, seed: int = 0):
    rng = random.Random(seed)
    observation = registry.execute("observe", ObserveArgs(max_new_tokens=2048).model_dump())
    if observation.status == "error":
        return {"patch_id": None, "reason": observation.data}
    layers = list(range(len(registry.runtime.layers)))
    failed_id = None
    if random_search:
        proposals = []
        for layer in rng.sample(layers, min(5, len(layers))):
            width = registry.runtime.layers[layer].mlp.down_proj.weight.shape[0]
            generator = torch.Generator().manual_seed(seed * 100 + layer)
            direction = torch.randn(width, generator=generator)
            direction /= direction.norm()
            key = "random_" + uuid.uuid4().hex[:12]
            file = f"directions/{key}.safetensors"
            (registry.root / file).parent.mkdir(parents=True, exist_ok=True)
            save_file({"direction": direction}, registry.root / file)
            registry.runtime._directions[key] = direction
            registry.directions[key] = {"file": file, "layer": layer, "method": "random", "seed": seed}
            proposals.append({"layer": layer, "direction_id": key})
        registry.save()
    else:
        obs = observation.data["observation_id"]
        failed = registry.execute("capture", CaptureArgs(observation_id=obs).model_dump())
        reference = registry.execute(
            "capture", CaptureArgs(observation_id=obs, source="reference").model_dump()
        )
        if failed.status == "error" or reference.status == "error":
            return {"patch_id": None, "reason": "No usable contrast"}
        failed_id = failed.data["capture_id"]
        contrasted = registry.execute(
            "compare",
            CompareArgs(
                failed_capture_id=failed_id, reference_capture_id=reference.data["capture_id"]
            ).model_dump(),
        )
        if contrasted.status == "error":
            return {"patch_id": None, "reason": contrasted.data}
        proposals = contrasted.data["candidates"][:5]
    candidates = []
    for proposal in proposals:
        layer, direction_id = proposal["layer"], proposal["direction_id"]
        width = registry.runtime.layers[layer].mlp.down_proj.weight.shape[1]
        if random_search:
            columns = rng.sample(range(width), min(64, width))
        else:
            ranking = registry.execute(
                "inspect",
                InspectArgs(
                    direction_id=direction_id, layer=layer, capture_id=failed_id, top_k=min(64, width)
                ).model_dump(),
            )
            if ranking.status == "error":
                continue
            columns = [n["column"] for n in ranking.data["neurons"]]
        for strength in (0.5, 1.0):
            spec = Intervention(layer=layer, direction_id=direction_id, strength=strength, columns=columns)
            exported = registry.execute(
                "export_patch",
                ExportArgs(
                    interventions=[spec],
                    hypothesis="Random location/direction/columns"
                    if random_search
                    else "Screen all layers; suppress strongest signed writers of the contrast direction",
                ).model_dump(),
            )
            if exported.status == "error":
                continue
            key = exported.data["patch_id"]
            validated = registry.execute("validate", ValidateArgs(patch_id=key).model_dump())
            if validated.status == "ok":
                comparison = validated.data["comparison"]
                size = exported.data["changed_entries"]
                candidates.append(
                    (comparison["accuracy_delta_pp"], -comparison["edited"]["hit_token_limit"], -size, key)
                )
    chosen = max(candidates) if candidates else None
    return {
        "patch_id": chosen[-1] if chosen and chosen[0] > 0 else None,
        "method": "random" if random_search else "scripted",
        "seed": seed,
        "selection": "positive validation accuracy; ties: fewer token-limit hits, then smaller edit",
    }
