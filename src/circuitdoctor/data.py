from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

from .schema import Example
from .storage import digest, write_json


def prepare(root: Path, seed: int = 0, development_n: int = 32, *, full: bool = False):
    root.mkdir(parents=True, exist_ok=True)
    dev = load_dataset("openai/gsm8k", "main", split="train")
    ids = random.Random(seed).sample(range(len(dev)), development_n * 2)
    development = []
    for j, i in enumerate(ids):
        row = dev[i]
        development.append(
            Example(
                id=f"gsm8k_train_{i}",
                prompt=row["question"],
                answer=row["answer"].split("####")[-1].strip().replace(",", ""),
                suite="gsm8k",
                split="development" if j < development_n else "validation",
                metadata={"solution": row["answer"]},
            )
        )
    test = load_dataset("openai/gsm8k", "main", split="test")
    ids = list(range(len(test))) if full else sorted(random.Random(seed).sample(range(len(test)), 256))
    evaluation = [
        Example(
            id=f"gsm8k_test_{i}",
            prompt=test[i]["question"],
            answer=test[i]["answer"].split("####")[-1].strip().replace(",", ""),
            suite="gsm8k",
            split="test",
        )
        for i in ids
    ]
    math = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ids = list(range(len(math))) if full else sorted(random.Random(seed).sample(range(len(math)), 128))
    evaluation += [
        Example(
            id="math500_" + str(math[i]["unique_id"]),
            prompt=math[i]["problem"],
            answer=math[i]["answer"],
            suite="math500",
            split="test",
        )
        for i in ids
    ]
    instructions = load_dataset("google/IFEval", split="train")
    evaluation += [
        Example(
            id=f"ifeval_{r['key']}",
            prompt=r["prompt"],
            suite="ifeval",
            split="test",
            metadata={"instruction_id_list": r["instruction_id_list"], "kwargs": r["kwargs"]},
        )
        for r in instructions
    ]
    for filename, rows in (("development.jsonl", development), ("evaluation.jsonl", evaluation)):
        serialized = [e.model_dump() for e in rows]
        (root / filename).write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in serialized))
    manifest = {
        "seed": seed,
        "full": full,
        "counts": {"development": len(development), "evaluation": len(evaluation)},
        "fingerprints": {
            "gsm8k_train": dev._fingerprint,
            "gsm8k_test": test._fingerprint,
            "math500": math._fingerprint,
            "ifeval": instructions._fingerprint,
        },
        "development_sha256": digest([e.model_dump() for e in development]),
        "evaluation_sha256": digest([e.model_dump() for e in evaluation]),
        "evaluation_ids": [e.id for e in evaluation],
    }
    write_json(root / "manifest.json", manifest)
    return manifest
