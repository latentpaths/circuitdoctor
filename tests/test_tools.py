import pytest
import torch

from circuitdoctor.budget import Ledger
from circuitdoctor.schema import Example, Generation
from circuitdoctor.storage import write_json
from circuitdoctor.tools import ToolRegistry


def test_generation_parity_and_capture(runtime):
    result = runtime.parity([Example(id="fixture", prompt="What is 2 + 3?")], max_tokens=4)
    assert result["passed"] and result["hook_calls"] > 0
    captures = runtime.capture(["What is 2 + 3?", "The answer is 5"], [0, 1])
    assert captures["0.residual.0.mean"].shape == (16,)
    assert captures["1.neurons.1.mean"].shape == (32,)


def test_test_split_is_inaccessible_and_invalid_tool_cannot_read_paths(runtime, tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    with pytest.raises(ValueError, match="Final evaluation"):
        ToolRegistry(runtime, [Example(id="secret", prompt="test", split="test")], tmp_path / "bad", ledger)
    registry = ToolRegistry(runtime, [Example(id="dev", prompt="2+3")], tmp_path / "good", ledger)
    assert registry.execute("read_file", {"path": "/etc/passwd"}).status == "error"
    assert registry.execute("observe", {"ids": ["../../secret"]}).status == "error"
    assert registry.execute("observe", {"path": "/etc/passwd"}).status == "error"
    behavior = ToolRegistry(
        runtime, [Example(id="dev", prompt="2+3")], tmp_path / "behavior", ledger, behavioral_only=True
    )
    assert behavior.execute("capture", {"observation_id": "x", "layers": [0]}).status == "error"


def test_contrast_export_validation_and_resume(runtime, tmp_path):
    examples = [
        Example(id="dev", prompt="What is 2 + 3?", answer="5"),
        Example(id="val", prompt="What is 3 + 2?", answer="5", split="validation"),
    ]
    root = tmp_path / "investigation"
    registry = ToolRegistry(runtime, examples, root, Ledger(tmp_path / "ledger.sqlite"))
    row = Generation(
        id="dev",
        text="The answer is 3",
        generated_tokens=4,
        prompt_tokens=4,
        hit_token_limit=True,
        repetition=0.9,
        correct=False,
    )
    write_json(root / "observations/failure.json", {"rows": [row.model_dump()]})
    registry.observations["obs"] = {"file": "observations/failure.json"}
    failed = registry.execute("capture", {"observation_id": "obs", "layers": [0, 1]})
    reference = registry.execute(
        "capture", {"observation_id": "obs", "layers": [0, 1], "source": "reference"}
    )
    assert failed.status == reference.status == "ok"
    contrast = registry.execute(
        "compare",
        {
            "failed_capture_id": failed.data["capture_id"],
            "reference_capture_id": reference.data["capture_id"],
            "layer": 0,
        },
    )
    assert contrast.status == "ok"
    screened = registry.execute(
        "compare",
        {
            "failed_capture_id": failed.data["capture_id"],
            "reference_capture_id": reference.data["capture_id"],
        },
    )
    assert screened.status == "ok"
    assert {c["layer"] for c in screened.data["candidates"]} == {0, 1}
    direction_id = contrast.data["direction_id"]
    inspected = registry.execute("inspect", {"direction_id": direction_id, "layer": 0, "top_k": 8})
    assert inspected.status == "ok", inspected
    exported = registry.execute(
        "export_patch",
        {"interventions": [{"layer": 0, "direction_id": direction_id}], "hypothesis": "test suppression"},
    )
    assert exported.status == "ok", exported
    original = runtime.layers[0].mlp.down_proj.weight.detach().clone()
    validated = registry.execute("validate", {"patch_id": exported.data["patch_id"]})
    assert validated.status == "ok", validated
    repeated = registry.execute("validate", {"patch_id": exported.data["patch_id"]})
    assert repeated.usage["cached"] is True
    assert repeated.data["comparison"] == validated.data["comparison"]
    assert torch.equal(original, runtime.layers[0].mlp.down_proj.weight)
    resumed = ToolRegistry(runtime, examples, root, registry.ledger)
    assert direction_id in resumed.runtime._directions
    assert exported.data["patch_id"] in resumed.patches


def test_completed_tool_call_is_idempotent_across_resume(runtime, tmp_path):
    examples = [Example(id="dev", prompt="What is 2 + 3?", answer="5")]
    ledger = Ledger(tmp_path / "ledger.sqlite")
    registry = ToolRegistry(runtime, examples, tmp_path / "run", ledger)
    result = registry.execute("observe", {"max_new_tokens": 16}, call_id="call-1")
    resumed = ToolRegistry(runtime, examples, tmp_path / "run", ledger)
    repeated = resumed.execute("observe", {"max_new_tokens": 16}, call_id="call-1")
    assert result == repeated
    counts = {r["name"]: r["value"] for r in ledger.summary()["counters"]}
    assert counts["diagnostic_batches"] == 1
    with pytest.raises(ValueError, match="changed arguments"):
        resumed.execute("observe", {"max_new_tokens": 32}, call_id="call-1")
    with pytest.raises(ValueError, match="configuration or data changed"):
        ToolRegistry(runtime, [Example(id="other", prompt="x")], tmp_path / "run", ledger)
