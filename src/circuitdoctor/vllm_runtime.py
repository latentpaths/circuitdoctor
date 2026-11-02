"""Optional isolated high-throughput evaluator. No investigator hooks or tools exposed."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

from transformers import AutoTokenizer

from .metrics import grade, repetition_score
from .schema import Generation, RunConfig, Sampling


def _worker_patch(model, bundle: str, revert: bool):
    # This callable runs inside vLLM's worker, with the same tensor/hash checks as Transformers.
    import re

    from circuitdoctor.patch import apply_patch
    from circuitdoctor.storage import read_json

    path = Path(bundle)
    manifest = read_json(path / "manifest.json")
    parameters = dict(model.named_parameters())
    aliases = {}
    for t in manifest["tensors"]:
        name = t["name"]
        suffix = re.search(r"layers\.\d+\.mlp\.down_proj\.weight$", name)
        if suffix is None:
            raise ValueError("vLLM adapter supports MLP down-projection patches only")
        matches = [p for k, p in parameters.items() if k.endswith(suffix.group())]
        if len(matches) != 1:
            raise ValueError("Ambiguous or missing vLLM tensor mapping")
        aliases[name] = matches[0]
    apply_patch(aliases, path, revert=revert)
    return {"verified_tensors": len(aliases), "reverted": revert}


class PatchWorkerExtension:
    """Installed worker method: only primitive RPC arguments, with default serialization protections."""

    def apply_circuitdoctor_patch(self, bundle: str, revert: bool):
        return _worker_patch(self.model_runner.model, bundle, revert)


class VLLMRuntime:
    def __init__(self, config: RunConfig):
        from vllm import LLM

        self.config = config
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        self.engine = LLM(
            model=config.model,
            revision=config.revision,
            dtype=config.dtype,
            max_model_len=34816,
            gpu_memory_utilization=0.85,
            seed=config.sampling.seed,
            language_model_only=True,
            enforce_eager=config.evaluation_eager,
            max_num_seqs=64,
            compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64]},
            enable_prefix_caching=False,
            worker_extension_cls="circuitdoctor.vllm_runtime.PatchWorkerExtension",
        )
        self.last_usage = {}

    def prompt(self, example, sampling):
        text = (
            example.prompt
            if example.suite == "ifeval"
            else (
                example.prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            )
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=sampling.thinking,
        )

    def generate(self, examples, sampling: Sampling | None = None, sample: int = 0):
        from vllm import SamplingParams

        sampling = sampling or self.config.sampling
        started = time.monotonic()
        outputs = self.engine.generate(
            [self.prompt(e, sampling) for e in examples],
            SamplingParams(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                top_k=sampling.top_k,
                presence_penalty=sampling.presence_penalty,
                max_tokens=sampling.max_new_tokens,
                seed=sampling.seed + sample,
            ),
            use_tqdm=False,
        )
        rows = []
        for example, output in zip(examples, outputs, strict=True):
            result = output.outputs[0]
            if result.finish_reason not in ("stop", "length"):
                raise RuntimeError("Incomplete vLLM generation rejected")
            correct, prediction = grade(example, result.text, thinking=sampling.thinking)
            rows.append(
                Generation(
                    id=example.id,
                    text=result.text,
                    generated_tokens=len(result.token_ids),
                    prompt_tokens=len(output.prompt_token_ids),
                    hit_token_limit=result.finish_reason == "length",
                    repetition=repetition_score(result.text),
                    correct=correct,
                    prediction=prediction,
                    sample=sample,
                )
            )
        self.last_usage = {
            "sequences": len(rows),
            "generated_tokens": sum(r.generated_tokens for r in rows),
            "prompt_tokens": sum(r.prompt_tokens for r in rows),
            "gpu_seconds": time.monotonic() - started,
            "engine": "vllm",
        }
        return rows

    @contextlib.contextmanager
    def patched(self, bundle: Path):
        self.engine.collective_rpc("apply_circuitdoctor_patch", args=(str(bundle.resolve()), False))
        self.engine.reset_prefix_cache()
        try:
            yield
        finally:
            self.engine.collective_rpc("apply_circuitdoctor_patch", args=(str(bundle.resolve()), True))
            self.engine.reset_prefix_cache()
