"""Attribute and intervene with released Qwen3-0.6B transcoders in an isolated environment."""

import argparse
import time
from pathlib import Path

import torch
from circuit_tracer import ReplacementModel, attribute
from huggingface_hub import HfApi, snapshot_download

from circuitdoctor.storage import write_json


def main(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    api = HfApi()
    model_revision = api.model_info("Qwen/Qwen3-0.6B").sha
    transcoder_revision = api.model_info("mwhanna/qwen3-0.6b-transcoders-lowl0").sha
    transcoders = snapshot_download("mwhanna/qwen3-0.6b-transcoders-lowl0", revision=transcoder_revision)
    model = ReplacementModel.from_pretrained(
        "Qwen/Qwen3-0.6B",
        transcoders,
        revision=model_revision,
        dtype=torch.bfloat16,
        backend="transformerlens",
    )
    prompt = "The capital of France is"
    graph = attribute(
        prompt=prompt,
        model=model,
        max_n_logits=2,
        desired_logit_prob=0.8,
        batch_size=32,
        max_feature_nodes=128,
        offload="cpu",
        verbose=True,
    )
    graph.to_pt(output / "graph.pt")
    if not torch.isfinite(graph.adjacency_matrix).all():
        raise RuntimeError("Nonfinite attribution graph")
    if len(graph.selected_features) == 0:
        raise RuntimeError("No features selected")
    features = [tuple(map(int, graph.active_features[int(i)].tolist())) for i in graph.selected_features]
    final_position = len(graph.input_tokens) - 1
    candidates = sorted((f for f in features if f[1] == final_position), reverse=True)
    if not candidates:
        raise RuntimeError("No selected feature at final position")
    with torch.inference_mode():
        original, _ = model.feature_intervention(prompt, [])
        unchanged, _ = model.feature_intervention(prompt, [])
        torch.testing.assert_close(original, unchanged)
        effect = 0.0
        for layer, _, feature in candidates[:12]:
            # Negative position avoids differing BOS conventions between attribution and intervention helpers.
            for value in (0.0, 10.0):
                intervened, _ = model.feature_intervention(prompt, [(layer, -1, feature, value)])
                if not torch.isfinite(intervened).all():
                    raise RuntimeError("Nonfinite intervention logits")
                effect = float((original - intervened).abs().max())
                if effect > 1e-6:
                    break
            if effect > 1e-6:
                break
    if effect <= 1e-6:
        raise RuntimeError("Feature manipulation produced no measurable effect")
    position = -1
    write_json(
        output / "results.json",
        {
            "method": "circuit-tracer",
            "status": "compatibility smoke passed",
            "model": "Qwen/Qwen3-0.6B",
            "transcoders": "mwhanna/qwen3-0.6b-transcoders-lowl0",
            "model_revision": model_revision,
            "transcoder_revision": transcoder_revision,
            "derivatives": True,
            "weight_updates": False,
            "prompt": prompt,
            "feature_nodes": len(graph.selected_features),
            "intervention": {"layer": layer, "position": position, "feature": feature, "value": value},
            "max_logit_difference": effect,
            "unchanged_parity": True,
            "seconds": time.monotonic() - started,
            "limitation": "A feature intervention on one prompt is not a demonstrated model repair",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/circuit-trace"))
    main(parser.parse_args().output)
