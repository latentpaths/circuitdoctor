from __future__ import annotations

import numpy as np

from .schema import Example, Generation


def repetition_score(text: str, n: int = 8) -> float:
    tokens = text.split()
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def last_boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    level, begin = 1, start + 7
    for i in range(begin, len(text)):
        level += (text[i] == "{") - (text[i] == "}")
        if level == 0:
            return text[begin:i]
    return None


def grade(example: Example, text: str, *, thinking: bool = True) -> tuple[bool, str | None]:
    if example.suite == "ifeval":
        # Official instructions, not an LLM judge or simplified approximation.
        from langdetect import DetectorFactory

        from .vendor.ifeval import instructions_registry as registry

        DetectorFactory.seed = 0

        answer = text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else ("" if thinking else text)
        followed = []
        for iid, kwargs in zip(
            example.metadata["instruction_id_list"], example.metadata["kwargs"], strict=True
        ):
            inst = registry.INSTRUCTION_DICT[iid](iid)
            inst.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
            args = inst.get_instruction_args()
            if args and "prompt" in args:
                inst.build_description(prompt=example.prompt)
            followed.append(bool(answer) and bool(inst.check_following(answer)))
        return bool(followed) and all(followed), None
    prediction = last_boxed(text)
    if prediction is None:
        return False, None
    from math_verify import parse, verify

    try:
        expected = parse("$" + example.answer + "$")
        actual = parse("$" + prediction + "$")
        return bool(expected and actual and verify(expected, actual, timeout_seconds=3)), prediction
    except Exception:
        return False, prediction


def summarize(rows: list[Generation]) -> dict:
    if not rows:
        return {"n": 0}
    scored = [r for r in rows if r.correct is not None]
    return {
        "n": len(rows),
        "accuracy": np.mean([r.correct for r in scored]).item() if scored else None,
        "hit_token_limit": np.mean([r.hit_token_limit for r in rows]).item(),
        "repetition": np.mean([r.repetition for r in rows]).item(),
        "generated_tokens": sum(r.generated_tokens for r in rows),
        "prompt_tokens": sum(r.prompt_tokens for r in rows),
        "mean_generated_tokens": np.mean([r.generated_tokens for r in rows]).item(),
    }


def paired_comparison(base: list[Generation], edited: list[Generation], seed: int = 0) -> dict:
    def grouped(rows):
        result = {}
        for r in rows:
            if r.correct is None:
                raise ValueError("Unscored rows in comparison")
            key = (r.id, r.sample)
            if key in result:
                raise ValueError("Duplicate problem/sample")
            result[key] = r
        return result

    a, b = grouped(base), grouped(edited)
    if not a or a.keys() != b.keys():
        raise ValueError("Paired evaluation requires identical nonempty ID/sample sets")
    # Cluster bootstrap by problem: multiple samples are not independent problems.
    ids = sorted({k[0] for k in a})
    changes = np.array(
        [np.mean([int(b[k].correct) - int(a[k].correct) for k in a if k[0] == i]) for i in ids]
    )
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(changes, len(changes), replace=True).mean() for _ in range(5000)])
    return {
        "baseline": summarize(base),
        "edited": summarize(edited),
        "n_problems": len(ids),
        "accuracy_delta_pp": float(changes.mean() * 100),
        "ci95_pp": (np.quantile(boot, [0.025, 0.975]) * 100).tolist(),
        "interval": "paired bootstrap, resampled by problem, 5000 replicates",
        "fixed": sum(not a[k].correct and b[k].correct for k in a),
        "broken": sum(a[k].correct and not b[k].correct for k in a),
    }
