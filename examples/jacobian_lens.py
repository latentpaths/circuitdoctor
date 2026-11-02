"""Numerical compatibility smoke test using Anthropic's reference Jacobian lens.

Run in an isolated environment; this computes derivatives, without updating model weights.
Four short fitting prompts test integration, not lens quality or a behavioral repair.
"""

import argparse
import time
from pathlib import Path

import jlens
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuitdoctor.patch import tensor_hash
from circuitdoctor.storage import digest, read_json, write_json

PROMPTS = [
    "A red apple falls from a tree onto the green grass. A gardener picks it up and puts it in a basket beside several other pieces of ripe fruit from the orchard.",
    "Two plus three is five, and five minus two is three. A teacher uses these simple equations to explain how addition and subtraction undo each other when working with whole numbers.",
    "The capital of France is Paris. Paris is a city in Europe with museums, schools, parks, and many streets. A traveler arrives by train and follows a map to the river.",
    "A Python function returns a value to its caller. A programmer writes a small function, supplies some arguments, checks the returned result, and adds a test that documents the expected behavior.",
]


def main(output: Path):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    revision = HfApi().model_info("Qwen/Qwen3-0.6B").sha
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", revision=revision, torch_dtype=torch.float32, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    before = digest({name: tensor_hash(value) for name, value in model.named_parameters()})
    wrapped = jlens.from_hf(model, tokenizer)
    layers = [model.config.num_hidden_layers // 2]
    fit_spec = {
        "prompts": PROMPTS,
        "model_revision": revision,
        "layers": layers,
        "dim_batch": 32,
        "max_seq_len": 64,
    }
    if (output / "fit-spec.json").exists() and read_json(output / "fit-spec.json") != fit_spec:
        raise ValueError("Fitting protocol changed; use a fresh output directory")
    if (output / "fit-checkpoint.pt").exists() and not (output / "fit-spec.json").exists():
        raise ValueError("Unversioned fitting checkpoint; use a fresh output directory")
    write_json(output / "fit-spec.json", fit_spec)
    lens = jlens.fit(
        wrapped,
        prompts=PROMPTS,
        source_layers=layers,
        dim_batch=32,
        max_seq_len=64,
        checkpoint_path=str(output / "fit-checkpoint.pt"),
    )
    lens.save(str(output / "jacobian_lens.pt"))
    prompt = "The opposite of hot is"
    lens_logits, actual_logits, _ = lens.apply(wrapped, prompt, positions=[-1], max_seq_len=64)
    encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        native = model(**encoded).logits[0, -1].float()
    torch.testing.assert_close(actual_logits.float().reshape(-1).cpu(), native.cpu(), atol=2e-4, rtol=2e-4)
    finite = all(torch.isfinite(value).all().item() for value in lens_logits.values())
    if not finite:
        raise RuntimeError("Nonfinite lens readout")
    if before != digest({name: tensor_hash(value) for name, value in model.named_parameters()}):
        raise RuntimeError("Model weights changed during fitting")
    top = {
        str(layer): [
            {"token": tokenizer.decode([int(i)]), "logit": float(v)}
            for v, i in zip(*logits[-1].topk(8), strict=True)
        ]
        for layer, logits in lens_logits.items()
    }
    write_json(
        output / "results.json",
        {
            "method": "Jacobian Lens",
            "status": "compatibility smoke passed",
            "model": "Qwen/Qwen3-0.6B",
            "revision": revision,
            "derivatives": True,
            "weight_updates": False,
            "fit_prompts": PROMPTS,
            "layers": layers,
            "native_logits_parity": True,
            "finite": finite,
            "top_tokens": top,
            "seconds": time.monotonic() - started,
            "limitation": "Four fitting prompts establish execution and numerical compatibility, not lens quality",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/jacobian-lens"))
    main(parser.parse_args().output)
