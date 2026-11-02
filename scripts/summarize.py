"""Derive campaign accounting from preserved evidence without rerunning a model."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from circuitdoctor.storage import read_json, write_json


def summarize(root: Path, hourly_usd: float):
    conditions = []
    for selection in sorted(root.glob("*/selection.json")):
        run = selection.parent
        selected = read_json(selection)
        if "selection_rule" not in selected:
            continue
        events_path = run / "events.jsonl"
        import json

        events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
        calls = [read_json(p) for p in sorted((run / "editor_calls").glob("*.json"))]
        usage = Counter()

        def count_usage(result, usage=usage):
            for key, value in result.get("usage", {}).items():
                if key in {
                    "sequences",
                    "capture_sequences",
                    "generated_tokens",
                    "prompt_tokens",
                    "capture_tokens",
                    "gpu_seconds",
                    "wall_seconds",
                }:
                    usage[key] += value
            for nested in result.get("data", {}).get("results", []):
                count_usage(nested)

        for event in events:
            count_usage(event["result"])
        identity = read_json(run / "tools.json")
        ledger = read_json(run / "spend.json")
        counters = {c["name"]: c["value"] for c in ledger["counters"] if c["run"] == identity["run_id"]}
        for key in ("sequences", "generated_tokens", "prompt_tokens", "capture_sequences", "capture_tokens"):
            usage[key] = counters.get(key, 0)
        validations = []
        for path in sorted((run / "validation").glob("patch*.json")):
            validation = read_json(path)
            validations.append({"file": str(path.relative_to(root)), "comparison": validation["comparison"]})
        condition = {
            "id": run.name,
            "job": selected.get("job"),
            "patch_id": selected["patch_id"],
            "selection_evidence": str(selection.relative_to(root)),
            "tool_calls": len(events),
            "tool_errors": sum(e["result"]["status"] != "ok" for e in events),
            "tool_counts": dict(Counter(e["tool"] for e in events)),
            "target_usage": dict(usage),
            "editor_requests": len(calls),
            "editor_prompt_tokens": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
            "editor_completion_tokens": sum(c["usage"].get("completion_tokens", 0) for c in calls),
            "editor_accounted_usd": sum(c["usage"]["usd"] for c in calls),
            "active_target_gpu_estimate_usd": usage["gpu_seconds"] * hourly_usd / 3600,
            "validated_candidates": len(validations),
            "positive_validation_candidates": sum(
                v["comparison"]["accuracy_delta_pp"] > 0 for v in validations
            ),
            "validation_evidence": validations,
            "catalog_evidence": str((run / "editor.json").relative_to(root)) if calls else None,
        }
        evaluation = run / "evaluation/results.json"
        if evaluation.exists():
            condition["final_evaluation"] = {
                "file": str(evaluation.relative_to(root)),
                "data": read_json(evaluation),
            }
        conditions.append(condition)
    result = {
        "summary": f"{len(conditions)} completed search conditions; {sum(c['patch_id'] is not None for c in conditions)} validation-selected patches. Final test results are separate from validation selection.",
        "conditions": conditions,
        "accounting_note": "Target GPU seconds are allocated wall time during generation/capture, not CUDA kernel time. Active GPU cost estimates exclude setup, editor waits and idle time and are already part of the whole lease; do not add them again to project spending. Editor dollars use the conservative maximum of catalog estimates and provider-reported costs.",
        "hourly_usd": hourly_usd,
        "selection_note": "Frozen selection.json 'validated_candidates' counts positive candidates in v0.1; this summary counts every validation evidence file explicitly.",
    }
    write_json(root / "findings.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--hourly-usd", type=float, default=4.29)
    args = parser.parse_args()
    result = summarize(args.root, args.hourly_usd)
    print(result["summary"])
