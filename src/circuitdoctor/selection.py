"""Selection uses validation only; neither final-test scores nor editor confidence enter the ranking."""

from __future__ import annotations

from .storage import digest, read_json


def select_patch(registry):
    candidates = []
    for key, info in registry.patches.items():
        path = registry.root / f"validation/{key}.json"
        if not path.exists():
            continue
        validation = read_json(path)
        if validation.get("patch_manifest_sha256") != digest(
            read_json(registry.root / info["file"] / "manifest.json")
        ):
            raise ValueError("Candidate changed after validation")
        comparison = validation["comparison"]
        gain = comparison["accuracy_delta_pp"]
        if gain <= 0:
            continue
        size = sum(t["changed_entries"] for t in info["manifest"]["tensors"])
        candidates.append((gain, -comparison["edited"]["hit_token_limit"], -size, key))
    chosen = max(candidates) if candidates else None
    return {
        "patch_id": chosen[-1] if chosen else None,
        "selection_rule": "positive validation accuracy; ties: fewer token-limit hits, then fewer changed entries",
        "validated_candidates": len(candidates),
        "explanation": "Frozen from validation only; final evaluation required"
        if chosen
        else "No positive validated candidate; retain unchanged weights",
    }
