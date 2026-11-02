"""GPU compatibility check for the optional vLLM evaluator and worker-side patch mapping."""

import argparse
from pathlib import Path

from circuitdoctor.metrics import paired_comparison
from circuitdoctor.schema import RunConfig
from circuitdoctor.storage import read_examples, read_json, write_json
from circuitdoctor.vllm_runtime import VLLMRuntime


def main(output: Path):
    if (output / "results.json").exists():
        raise FileExistsError("Use a fresh pilot output directory")
    config = RunConfig.model_validate(read_json(Path("configs/qwen35.json")))
    config.batch_size = 64
    runtime = VLLMRuntime(config)
    examples = read_examples(Path("runs/data/development.jsonl"))[:4]
    sampling = config.sampling.model_copy(update={"max_new_tokens": 512})
    base = runtime.generate(examples, sampling)
    base_usage = dict(runtime.last_usage)
    with runtime.patched(Path("runs/reproduction/patches/published-p50")):
        edited = runtime.generate(examples, sampling)
    restored = runtime.generate(examples, sampling)
    if [r.text for r in restored] != [r.text for r in base]:
        raise RuntimeError("vLLM generation did not restore after patch rollback")
    if [r.text for r in edited] == [r.text for r in base]:
        raise RuntimeError(
            "Published patch did not affect generation; reject possible stale compiled weights"
        )
    result = {
        "label": "vLLM development integration pilot; not held-out evaluation",
        "comparison": paired_comparison(base, edited),
        "baseline_usage": base_usage,
        "edited_usage": runtime.last_usage,
        "worker_hash_checks": "passed on apply and rollback",
        "generation_rollback_exact": True,
    }
    write_json(output / "results.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/vllm-pilot"))
    main(parser.parse_args().output)
