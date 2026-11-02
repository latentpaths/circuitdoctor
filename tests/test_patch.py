import pytest
import torch
from safetensors.torch import load_file, save_file

from circuitdoctor.patch import apply_patch, export_patch, suppress_weights, tensor_hash
from circuitdoctor.schema import Intervention


def test_exact_bfloat16_roundtrip_and_atomic_rejection(tmp_path):
    base = {"a": torch.randn(4, 6).bfloat16(), "b": torch.randn(4, 6).bfloat16()}
    edited = {k: v * 0.5 for k, v in base.items()}
    bundle = tmp_path / "bundle"
    export_patch(bundle, base, edited, model="fixture", revision="abc", label="test")
    current = {k: v.clone() for k, v in base.items()}
    current["b"][0, 0] += 1
    a_before = current["a"].clone()
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_patch(current, bundle)
    assert torch.equal(current["a"], a_before)
    current = {k: v.clone() for k, v in base.items()}
    with pytest.raises(ValueError, match="revision"):
        apply_patch(current, bundle, revision="wrong")
    apply_patch(current, bundle)
    assert all(torch.equal(current[k], edited[k]) for k in current)
    apply_patch(current, bundle, revert=True)
    assert all(tensor_hash(current[k]) == tensor_hash(base[k]) for k in current)
    payload = load_file(bundle / "edited.safetensors")
    payload["a"][0, 0] += 1
    save_file(payload, bundle / "edited.safetensors")
    with pytest.raises(ValueError, match="hash mismatch"):
        apply_patch(current, bundle)


@pytest.mark.parametrize("columns", [[], [0, 3, 8]])
def test_temporary_suppression_matches_exported_weight(runtime, columns, tmp_path):
    mlp = runtime.layers[0].mlp
    direction = torch.randn(16)
    runtime._directions["test"] = direction
    spec = Intervention(layer=0, direction_id="test", columns=columns, strength=0.75)
    x = torch.randn(2, 5, 16)
    original = mlp.down_proj.weight.detach().clone()
    with torch.no_grad(), runtime.intervention(spec):
        hooked = mlp(x)
    assert torch.equal(original, mlp.down_proj.weight)
    changed = suppress_weights(original, direction, 0.75, columns)
    with torch.no_grad():
        mlp.down_proj.weight.copy_(changed)
        persistent = mlp(x)
        mlp.down_proj.weight.copy_(original)
    torch.testing.assert_close(hooked, persistent, atol=2e-7, rtol=2e-5)
    runtime.export(tmp_path / "patch", spec, {})
    with runtime.patched(tmp_path / "patch"):
        torch.testing.assert_close(mlp(x), persistent)
    assert torch.equal(original, mlp.down_proj.weight)


def test_hooks_removed_after_exception(runtime):
    runtime._directions["test"] = torch.randn(16)
    before = len(runtime.layers[0].mlp._forward_hooks)
    with pytest.raises(RuntimeError):
        with runtime.intervention(Intervention(layer=0, direction_id="test")):
            raise RuntimeError("interrupted")
    assert len(runtime.layers[0].mlp._forward_hooks) == before


def test_evaluation_attachment_preserves_frozen_manifest(tmp_path):
    from circuitdoctor.patch import record_evidence
    from circuitdoctor.storage import read_json, write_json

    bundle = tmp_path / "bundle"
    base = {"a": torch.randn(2, 3)}
    export_patch(bundle, base, {"a": base["a"] * 0.5}, model="fixture", revision="abc", label="test")
    before = (bundle / "manifest.json").read_bytes()
    report = tmp_path / "result.json"
    write_json(report, {"accuracy_delta_pp": 20, "n": 50})
    record_evidence(bundle, report, "test")
    record_evidence(bundle, report, "test")
    index = read_json(bundle / "evidence.json")
    assert len(index["references"]) == 1
    assert (bundle / index["references"][0]["file"]).read_bytes() == report.read_bytes()
    assert (bundle / "manifest.json").read_bytes() == before
