import pytest

from circuitdoctor.evaluate import evaluate
from circuitdoctor.schema import Example, Generation


def test_completed_batches_resume_and_changed_protocol_rejected(runtime, tmp_path, monkeypatch):
    rows = [Example(id=str(i), prompt="2+3", answer="5", split="test") for i in range(4)]
    calls = []

    def generate(examples, sampling, sample=0):
        calls.append([e.id for e in examples])
        return [
            Generation(
                id=e.id,
                text=r"\boxed{5}",
                generated_tokens=4,
                prompt_tokens=3,
                hit_token_limit=False,
                repetition=0,
                correct=True,
                sample=sample,
            )
            for e in examples
        ]

    monkeypatch.setattr(runtime, "generate", generate)
    evaluate(runtime, rows, tmp_path / "eval")
    assert len(calls) == 2
    evaluate(runtime, rows, tmp_path / "eval")
    assert len(calls) == 2
    with pytest.raises(ValueError, match="immutable protocol"):
        evaluate(runtime, rows[:-1], tmp_path / "eval")
    with pytest.raises(ValueError, match="test split"):
        evaluate(runtime, [Example(id="dev", prompt="x")], tmp_path / "bad")


def test_shared_baseline_reuse(runtime, tmp_path, monkeypatch):
    rows = [Example(id="a", prompt="x", split="test")]
    calls = []

    def generate(examples, sampling, sample=0):
        calls.append(1)
        return [
            Generation(
                id=e.id,
                text="x",
                generated_tokens=1,
                prompt_tokens=1,
                hit_token_limit=False,
                repetition=0,
                correct=False,
            )
            for e in examples
        ]

    monkeypatch.setattr(runtime, "generate", generate)
    evaluate(runtime, rows, tmp_path / "first", baseline_cache=tmp_path / "cache")
    evaluate(runtime, rows, tmp_path / "second", baseline_cache=tmp_path / "cache")
    assert len(calls) == 1
