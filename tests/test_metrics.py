import pytest

from circuitdoctor.metrics import grade, last_boxed, paired_comparison, repetition_score
from circuitdoctor.schema import Example, Generation


def test_grading_and_loop_metrics():
    assert last_boxed(r"first \boxed{1}, finally \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert grade(Example(id="x", prompt="half", answer="0.5"), r"\boxed{\frac{1}{2}}")[0]
    assert not grade(Example(id="x", prompt="half", answer="0.5"), "Final answer: 0.5")[0]
    assert repetition_score("let us check again " * 100) > 0.9


def test_paired_problem_bootstrap_rejects_mismatches():
    def row(i, correct, sample=0):
        return Generation(
            id=str(i),
            text="",
            generated_tokens=2,
            prompt_tokens=2,
            hit_token_limit=False,
            repetition=0,
            correct=correct,
            sample=sample,
        )

    base = [row(i, False, j) for i in range(8) for j in range(2)]
    edited = [row(i, True, j) for i in range(8) for j in range(2)]
    comparison = paired_comparison(base, edited)
    assert comparison["n_problems"] == 8
    assert comparison["accuracy_delta_pp"] == 100
    assert comparison["ci95_pp"] == [100, 100]
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(base, edited[:-1])


def test_official_ifeval_checker():
    example = Example(
        id="ifeval",
        prompt="Say hello in all caps.",
        suite="ifeval",
        metadata={"instruction_id_list": ["change_case:english_capital"], "kwargs": [{}]},
    )
    assert grade(example, "</think>THIS IS A COMPLETE SENTENCE WRITTEN ENTIRELY IN ENGLISH.")[0]
    assert not grade(example, "Still thinking in ALL CAPS")[0]
