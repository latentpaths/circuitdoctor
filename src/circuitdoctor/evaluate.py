from __future__ import annotations

import time
from pathlib import Path

from .metrics import paired_comparison
from .schema import Example, Generation, Sampling
from .storage import digest, environment_versions, instrumentation_digest, read_json, write_json


def evaluate(
    runtime,
    examples: list[Example],
    root: Path,
    *,
    patch: Path | None = None,
    samples: int = 1,
    sampling: Sampling | None = None,
    baseline_cache: Path | None = None,
):
    if not examples or any(e.split != "test" for e in examples):
        raise ValueError("Final evaluation requires a nonempty test split")
    sampling = sampling or runtime.config.sampling
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "instrumentation_sha256": instrumentation_digest(),
        "environment": environment_versions(),
        "device": runtime.device,
        "engine": type(runtime).__name__,
        "runtime_config": runtime.config.model_dump(),
        "model": runtime.config.model,
        "revision": runtime.config.revision,
        "sampling": sampling.model_dump(),
        "samples": samples,
        "examples_sha256": digest([e.model_dump() for e in examples]),
        "patch": digest(read_json(patch / "manifest.json")) if patch else None,
    }
    protocol = root / "protocol.json"
    if protocol.exists() and read_json(protocol) != identity:
        raise ValueError("Existing evaluation belongs to a different immutable protocol")
    write_json(protocol, identity)
    cache_root = None
    if baseline_cache is not None:
        cache_identity = {k: v for k, v in identity.items() if k != "patch"}
        cache_root = baseline_cache / digest(cache_identity)
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_protocol = cache_root / "protocol.json"
        if cache_protocol.exists() and read_json(cache_protocol) != cache_identity:
            raise ValueError("Baseline cache protocol mismatch")
        write_json(cache_protocol, cache_identity)
    rows = {}
    started = time.monotonic()
    # Checkpoint after every batch. Unknown/partial batches are never scored as completed.
    for mode in ["baseline", "edited"] if patch else ["baseline"]:
        mode_rows = []
        for sample in range(samples):
            for start in range(0, len(examples), runtime.config.batch_size):
                batch = examples[start : start + runtime.config.batch_size]
                path = root / f"{mode}/{sample:02d}-{start:05d}.json"
                shared = cache_root / path.name if cache_root is not None and mode == "baseline" else None
                if not path.exists() and shared is not None and shared.exists():
                    write_json(path, {**read_json(shared), "reused_baseline": str(shared)})
                if path.exists():
                    data = read_json(path)
                    result = [Generation.model_validate(r) for r in data["rows"]]
                    if [r.id for r in result] != [e.id for e in batch]:
                        raise ValueError("Checkpoint ID mismatch")
                else:
                    if mode == "edited":
                        with runtime.patched(patch):
                            result = runtime.generate(batch, sampling, sample=sample)
                    else:
                        result = runtime.generate(batch, sampling, sample=sample)
                    write_json(path, {"rows": [r.model_dump() for r in result], "usage": runtime.last_usage})
                    if shared is not None:
                        write_json(shared, read_json(path))
                mode_rows.extend(result)
        rows[mode] = mode_rows
    comparisons = {}
    if patch:
        for suite in sorted({e.suite for e in examples}):
            ids = {e.id for e in examples if e.suite == suite}
            comparisons[suite] = paired_comparison(
                [r for r in rows["baseline"] if r.id in ids],
                [r for r in rows["edited"] if r.id in ids],
                sampling.seed,
            )
    result = {
        "protocol": identity,
        "label": "subset evaluation; not numerical parity with the published report",
        "comparisons": comparisons,
        "elapsed_this_invocation_seconds": time.monotonic() - started,
    }
    if comparisons:
        gsm = comparisons.get("gsm8k")
        ifeval = comparisons.get("ifeval")
        if gsm and ifeval:
            baseline_cut = gsm["baseline"]["hit_token_limit"]
            result["repair_target_met"] = (
                gsm["accuracy_delta_pp"] >= 10
                and gsm["ci95_pp"][0] > 0
                and baseline_cut > 0
                and gsm["edited"]["hit_token_limit"] <= baseline_cut * 0.5
                and ifeval["accuracy_delta_pp"] >= -2
            )
        else:
            result["repair_target_met"] = None
    write_json(root / "results.json", result)
    if patch is not None:
        from .patch import record_evidence

        record_evidence(patch, root / "results.json", "test")
    return result
