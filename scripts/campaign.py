"""Resume a frozen campaign. Run investigator and evaluator phases as separate processes/environments.

The investigator receives only development/validation data through its registry. Test data is consumed
only by the evaluator phase. Published patches are never supplied through investigator tools.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from circuitdoctor.budget import Ledger
from circuitdoctor.schema import RunConfig
from circuitdoctor.storage import digest, read_examples, read_json, write_json


def jobs():
    for seed in (0, 1, 2):
        for editor in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            for behavioral in (False, True):
                yield {
                    "id": f"{'behavioral' if behavioral else 'mechanistic'}-{editor.rsplit('/', 1)[1]}-seed{seed}",
                    "seed": seed,
                    "editor": editor,
                    "behavioral": behavioral,
                    "method": "agent",
                }
        for method in ("scripted", "random"):
            yield {
                "id": f"{method}-seed{seed}",
                "seed": seed,
                "editor": None,
                "behavioral": False,
                "method": method,
            }


def investigators(args):
    from circuitdoctor.agent import investigate
    from circuitdoctor.baselines import scripted
    from circuitdoctor.prime import PrimeClient
    from circuitdoctor.runtime import Runtime
    from circuitdoctor.selection import select_patch
    from circuitdoctor.tools import ToolRegistry

    config = RunConfig.model_validate(read_json(args.config))
    examples = read_examples(Path("runs/data/development.jsonl"))
    runtime = Runtime(config)
    for index, job in enumerate(jobs()):
        if args.limit_jobs is not None and index >= args.limit_jobs:
            break
        root = args.output / job["id"]
        if (root / "selection.json").exists() and "selection_rule" in read_json(root / "selection.json"):
            continue
        runtime.config.sampling.seed = job["seed"]
        identity = {
            "job": job,
            "config": runtime.config.model_dump(),
            "data_sha256": digest([e.model_dump() for e in examples]),
        }
        if (root / "identity.json").exists() and read_json(root / "identity.json") != identity:
            raise RuntimeError("Campaign condition changed; use a fresh output directory")
        write_json(root / "identity.json", identity)
        registry = ToolRegistry(
            runtime, examples, root, Ledger(args.ledger), behavioral_only=job["behavioral"]
        )
        print(json.dumps({"event": "start", **job, "time": time.time()}), flush=True)
        if job["method"] == "agent":
            editor_result = investigate(
                registry, PrimeClient(registry.ledger, "experiments", model=job["editor"])
            )
        else:
            editor_result = scripted(registry, random_search=job["method"] == "random", seed=job["seed"])
        selected = {**select_patch(registry), "editor_result": editor_result, "job": job}
        write_json(root / "selection.json", selected)
        write_json(root / "spend.json", registry.ledger.summary())
        print(json.dumps({"event": "frozen", "id": job["id"], "patch_id": selected["patch_id"]}), flush=True)
        runtime._directions.clear()
        runtime._captures.clear()
        gc.collect()


def evaluation(args):
    from circuitdoctor.evaluate import evaluate
    from circuitdoctor.vllm_runtime import VLLMRuntime

    config = RunConfig.model_validate(read_json(args.config))
    config.batch_size = 64
    config.sampling.seed = 0
    runtime = VLLMRuntime(config)
    examples = read_examples(Path("runs/data/evaluation.jsonl"))
    cache = args.output / "baseline-cache"
    published = Path("runs/reproduction/patches/published-p50")
    if not published.exists():
        raise RuntimeError("Rebuild the published patch first")
    evaluate(runtime, examples, Path("runs/reproduction/evaluation"), patch=published, baseline_cache=cache)
    # Both deployment changes are measured against the same unchanged-weight baseline.
    from circuitdoctor.metrics import paired_comparison
    from circuitdoctor.schema import Generation

    def completed_rows(directory):
        return [
            Generation.model_validate(row)
            for file in sorted(directory.glob("*.json"))
            for row in read_json(file)["rows"]
        ]

    original_rows = completed_rows(Path("runs/reproduction/evaluation/baseline"))
    settings = {
        "presence-penalty": config.sampling.model_copy(update={"presence_penalty": 2.0}),
        "non-thinking": config.sampling.model_copy(
            update={"thinking": False, "temperature": 0.7, "top_p": 0.8, "max_new_tokens": 8192}
        ),
    }
    for name, sampling in settings.items():
        root = args.output / "simple" / name
        evaluate(runtime, examples, root / "generated", sampling=sampling, baseline_cache=cache)
        alternate_rows = completed_rows(root / "generated/baseline")
        comparisons = {}
        for suite in sorted({e.suite for e in examples}):
            ids = {e.id for e in examples if e.suite == suite}
            comparisons[suite] = paired_comparison(
                [r for r in original_rows if r.id in ids], [r for r in alternate_rows if r.id in ids]
            )
        comparison_root = root / "comparison"
        comparison_root.mkdir(parents=True, exist_ok=True)
        import os

        for link, target in (
            ("baseline", Path("runs/reproduction/evaluation/baseline")),
            ("edited", root / "generated/baseline"),
        ):
            path = comparison_root / link
            if not path.exists():
                path.symlink_to(
                    os.path.relpath(target.resolve(), comparison_root.resolve()), target_is_directory=True
                )
        write_json(
            comparison_root / "results.json",
            {
                "label": name + " deployment baseline",
                "comparisons": comparisons,
                "sampling": sampling.model_dump(),
                "note": "Changed decoding protocol; weights unchanged",
            },
        )
    for job in jobs():
        root = args.output / job["id"]
        selection = root / "selection.json"
        if not selection.exists():
            continue
        chosen = read_json(selection)["patch_id"]
        if chosen is None:
            write_json(
                root / "final-disposition.json", {"weights": "unchanged", "reason": "No validation winner"}
            )
            continue
        patch = root / "patches" / chosen
        frozen = digest(read_json(patch / "manifest.json"))
        write_json(root / "frozen.json", {"manifest_sha256": frozen, "selected_before_final_test": True})
        evaluate(runtime, examples, root / "evaluation", patch=patch, baseline_cache=cache)
        if digest(read_json(patch / "manifest.json")) != frozen:
            raise RuntimeError("Frozen candidate changed during final evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["investigators", "evaluation"])
    parser.add_argument("--config", type=Path, default=Path("configs/qwen35.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/campaign"))
    parser.add_argument("--ledger", type=Path, default=Path("runs/project.sqlite"))
    parser.add_argument("--limit-jobs", type=int)
    args = parser.parse_args()
    investigators(args) if args.phase == "investigators" else evaluation(args)
