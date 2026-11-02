"""Run real target hooks, captures, an intervention, and published patch replay on development data."""

import argparse
import time
from pathlib import Path

import torch

from circuitdoctor.metrics import paired_comparison
from circuitdoctor.patch import tensor_hash
from circuitdoctor.runtime import Runtime
from circuitdoctor.schema import Intervention, RunConfig
from circuitdoctor.storage import read_examples, read_json, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=Path("configs/qwen35.json"))
parser.add_argument("--output", type=Path, default=Path("runs/pilot"))
parser.add_argument("--max-tokens", type=int, default=512)
parser.add_argument("--numerics-only", action="store_true")
args = parser.parse_args()
config = RunConfig.model_validate(read_json(args.config))
runtime = Runtime(config)
examples = read_examples(Path("runs/data/development.jsonl"))[:4]
parity = runtime.parity(examples[:1])
captures = runtime.capture([e.prompt + "\n" + e.metadata["solution"] for e in examples[:2]], [0, 15, 19])
finite = all(torch.isfinite(v).all().item() for v in captures.values())
if not finite:
    raise RuntimeError("Nonfinite real-model activations")
weight = runtime.layers[15].mlp.down_proj.weight
before = tensor_hash(weight)
u = torch.randn(weight.shape[0], generator=torch.Generator().manual_seed(0))
runtime._directions["pilot"] = u / u.norm()
spec = Intervention(layer=15, direction_id="pilot", columns=list(range(8)), strength=0.5)
args.output.mkdir(parents=True, exist_ok=True)
runtime.export(args.output / "patches/temporary-parity", spec, {"label": "random integration check"})
x = torch.randn(2, 8, weight.shape[0], device=runtime.device, dtype=weight.dtype)
with torch.no_grad(), runtime.intervention(spec):
    temporary = runtime.layers[15].mlp(x)
with torch.no_grad(), runtime.patched(args.output / "patches/temporary-parity"):
    persistent = runtime.layers[15].mlp(x)
relative_error = float(
    (temporary.float() - persistent.float()).norm() / persistent.float().norm().clamp_min(1e-8)
)
if relative_error > 0.03:
    raise RuntimeError("Temporary/persistent BF16 edit difference exceeded numerical tolerance")
if args.numerics_only:
    result = {
        "label": "real-target numerical integration check",
        "parity": parity,
        "finite_captures": finite,
        "temporary_persistent_relative_error": relative_error,
        "tolerance": 0.03,
        "rollback_exact": tensor_hash(weight) == before,
    }
    write_json(args.output / "results.json", result)
    print(result, flush=True)
    raise SystemExit(0)
sampling = config.sampling.model_copy(update={"max_new_tokens": args.max_tokens})
started = time.monotonic()
baseline = runtime.generate(examples, sampling)
usage = dict(runtime.last_usage)
with runtime.patched(Path("runs/reproduction/patches/published-p50")):
    edited = runtime.generate(examples, sampling)
if tensor_hash(weight) != before:
    raise RuntimeError("Original weights did not restore")
write_json(args.output / "evaluation/baseline/00-00000.json", {"rows": [r.model_dump() for r in baseline]})
write_json(args.output / "evaluation/edited/00-00000.json", {"rows": [r.model_dump() for r in edited]})
result = {
    "label": "development integration pilot; not held-out repair evaluation",
    "target": runtime.describe(),
    "parity": parity,
    "finite_captures": finite,
    "capture_shapes": {k: list(v.shape) for k, v in captures.items()},
    "rollback_exact": True,
    "sampling": sampling.model_dump(),
    "comparisons": {"gsm8k": paired_comparison(baseline, edited)},
    "seconds": time.monotonic() - started,
    "baseline_usage": usage,
}
write_json(args.output / "results.json", result)
print(
    {
        "device": runtime.device,
        "finite_captures": finite,
        "rollback_exact": True,
        "generated_tokens": usage["generated_tokens"],
        "seconds": result["seconds"],
    }
)
