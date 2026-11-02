from __future__ import annotations

import contextlib
import hashlib
import inspect
import time
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from .metrics import grade, repetition_score
from .patch import apply_patch, export_patch, suppress_weights, tensor_hash
from .schema import Example, Generation, Intervention, RunConfig, Sampling


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length: int, penalty: float):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, input_ids[:, self.prompt_length :], True)
        return scores - seen.to(scores.dtype) * self.penalty


class Runtime:
    def __init__(self, config: RunConfig, model=None, tokenizer=None):
        self.config = config
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else config.device
        )
        dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
        if model is None:
            hfconfig = AutoConfig.from_pretrained(
                config.model, revision=config.revision, trust_remote_code=False
            )
            factory = (
                AutoModelForImageTextToText if hasattr(hfconfig, "vision_config") else AutoModelForCausalLM
            )
            model = factory.from_pretrained(
                config.model,
                revision=config.revision,
                torch_dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=False,
            )
        self.model = model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.model, revision=config.revision, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        root = self.model.model
        self.transformer = getattr(root, "language_model", root)
        self.layers = self.transformer.layers
        self.last_usage = {}
        self._directions: dict[str, torch.Tensor] = {}
        self._captures: dict[str, dict] = {}

    def describe(self):
        return {
            "model": self.config.model,
            "revision": self.config.revision,
            "device": self.device,
            "layers": len(self.layers),
            "residual_width": self.layers[0].mlp.down_proj.weight.shape[0],
            "mlp_widths": [block.mlp.down_proj.weight.shape[1] for block in self.layers],
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "mlp_modules": [f"{i}:{type(block.mlp).__name__}" for i, block in enumerate(self.layers)],
            "dtype": str(next(self.model.parameters()).dtype),
        }

    def weight_name(self, layer: int):
        target = self.layers[layer].mlp.down_proj.weight
        return next(name for name, p in self.model.named_parameters() if p is target)

    def prompt(self, example: Example, sampling: Sampling) -> str:
        prompt = (
            example.prompt
            if example.suite == "ifeval"
            else (
                example.prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}."
            )
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=sampling.thinking,
        )

    @contextlib.contextmanager
    def intervention(self, spec: Intervention | None):
        if spec is None:
            yield
            return
        if spec.layer >= len(self.layers):
            raise ValueError("Layer out of range")
        mlp = self.layers[spec.layer].mlp
        if spec.kind == "scale_columns":
            original = mlp.down_proj.weight.detach().clone()
            columns = sorted(set(spec.columns))
            if not columns or min(columns) < 0 or max(columns) >= original.shape[1]:
                raise ValueError("Valid columns required")
            with torch.no_grad():
                mlp.down_proj.weight[:, columns] *= spec.strength
            try:
                yield
            finally:
                with torch.no_grad():
                    mlp.down_proj.weight.copy_(original)
            return
        module = mlp.down_proj if spec.columns else mlp
        direction = None
        if spec.kind == "suppress":
            if spec.direction_id not in self._directions:
                raise ValueError("Unknown direction")
            direction = self._directions[spec.direction_id]
        if spec.kind == "replace" and spec.capture_id not in self._captures:
            raise ValueError("Unknown replacement capture")

        def hook(_module, args, output):
            if spec.kind == "zero":
                return output * (1 - spec.strength)
            if spec.kind == "replace":
                reference = self._captures[spec.capture_id][f"mlp.{spec.layer}.mean"].to(output)
                return output * (1 - spec.strength) + reference * spec.strength
            u = direction.to(output).float()
            u = u / u.norm()
            if spec.columns:
                x = args[0].float()
                cols = spec.columns
                if min(cols) < 0 or max(cols) >= mlp.down_proj.weight.shape[1]:
                    raise ValueError("Column out of range")
                contribution = x[..., cols] @ mlp.down_proj.weight[:, cols].float().T
            else:
                contribution = output.float()
            return (output.float() - spec.strength * (contribution @ u).unsqueeze(-1) * u).to(output.dtype)

        handle = module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate(
        self,
        examples: list[Example],
        sampling: Sampling | None = None,
        intervention: Intervention | None = None,
        sample: int = 0,
        *,
        score: bool = True,
    ):
        sampling = sampling or self.config.sampling
        results, started = [], time.monotonic()
        with self.intervention(intervention):
            for start in range(0, len(examples), self.config.batch_size):
                batch = examples[start : start + self.config.batch_size]
                encoded = self.tokenizer(
                    [self.prompt(e, sampling) for e in batch], return_tensors="pt", padding=True
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                width = encoded["input_ids"].shape[1]
                seed_offset = int(hashlib.sha256("|".join(e.id for e in batch).encode()).hexdigest()[:8], 16)
                torch.manual_seed((sampling.seed + sample + seed_offset) % (2**32))
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=sampling.max_new_tokens,
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=sampling.top_k,
                    pad_token_id=self.tokenizer.pad_token_id,
                    logits_processor=LogitsProcessorList([PresencePenalty(width, sampling.presence_penalty)]),
                )
                eos = self.model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos or [])
                for i, e in enumerate(batch):
                    ids = output[i, width:].tolist()
                    stop = next((j + 1 for j, x in enumerate(ids) if x in eos), len(ids))
                    ids = ids[:stop]
                    text = self.tokenizer.decode(
                        ids[:-1] if ids and ids[-1] in eos else ids, skip_special_tokens=False
                    )
                    correct, prediction = (
                        grade(e, text, thinking=sampling.thinking) if score else (None, None)
                    )
                    results.append(
                        Generation(
                            id=e.id,
                            text=text,
                            generated_tokens=len(ids),
                            prompt_tokens=int(encoded["attention_mask"][i].sum()),
                            hit_token_limit=len(ids) >= sampling.max_new_tokens
                            and not (ids and ids[-1] in eos),
                            repetition=repetition_score(text),
                            correct=correct,
                            prediction=prediction,
                            sample=sample,
                        )
                    )
        self.last_usage = {
            "sequences": len(results),
            "generated_tokens": sum(r.generated_tokens for r in results),
            "prompt_tokens": sum(r.prompt_tokens for r in results),
            "wall_seconds": time.monotonic() - started,
            "gpu_seconds": time.monotonic() - started if self.device.startswith("cuda") else 0.0,
        }
        return results

    @torch.inference_mode()
    def capture(self, texts: list[str], layers: list[int], limit: int | None = None):
        limit = min(limit or self.config.capture_token_limit, self.config.capture_token_limit)
        if not layers or any(i < 0 or i >= len(self.layers) for i in layers):
            raise ValueError(f"Layers must be integers from 0 through {len(self.layers) - 1}")
        tensors = {}
        started, prefill_tokens = time.monotonic(), 0
        for index, text in enumerate(texts):
            encoded = self.tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=False)
            encoded = {k: v[:, -limit:].to(self.device) for k, v in encoded.items()}
            prefill_tokens += int(encoded["attention_mask"].sum())
            hooks = []

            def record(prefix, output, index=index):
                x = (output[0] if isinstance(output, tuple) else output).detach().float()[0]
                # Store bounded summaries, not a dense position x layer x vocabulary array.
                tensors[f"{index}.{prefix}.mean"] = x[-64:].mean(0).cpu()
                tensors[f"{index}.{prefix}.last"] = x[-1].cpu()

            for layer in layers:
                block = self.layers[layer]
                hooks.append(
                    block.register_forward_hook(lambda m, a, o, idx=layer: record(f"residual.{idx}", o))
                )
                hooks.append(
                    block.mlp.register_forward_hook(lambda m, a, o, idx=layer: record(f"mlp.{idx}", o))
                )
                hooks.append(
                    block.mlp.down_proj.register_forward_pre_hook(
                        lambda m, a, idx=layer: record(f"neurons.{idx}", a[0])
                    )
                )
            try:
                kwargs = (
                    {"logits_to_keep": 1}
                    if "logits_to_keep" in inspect.signature(self.model.forward).parameters
                    else {}
                )
                self.model(**encoded, use_cache=False, **kwargs)
            finally:
                for handle in hooks:
                    handle.remove()
        self.last_usage = {
            "capture_sequences": len(texts),
            "capture_tokens": prefill_tokens,
            "wall_seconds": time.monotonic() - started,
            "gpu_seconds": time.monotonic() - started if self.device.startswith("cuda") else 0.0,
        }
        return tensors

    def export(self, path: Path, spec: Intervention, provenance: dict):
        if spec.kind not in ("suppress", "zero", "scale_columns"):
            raise ValueError("Activation replacement cannot be exported as an exact weight edit")
        weight = self.layers[spec.layer].mlp.down_proj.weight.detach()
        name = self.weight_name(spec.layer)
        if spec.kind == "suppress":
            changed = suppress_weights(
                weight, self._directions[spec.direction_id], spec.strength, spec.columns
            )
        else:
            changed = weight.clone()
            if spec.kind == "zero":
                changed *= 1 - spec.strength
            else:
                if not spec.columns or min(spec.columns) < 0 or max(spec.columns) >= changed.shape[1]:
                    raise ValueError("Valid columns required")
                changed[:, spec.columns] *= spec.strength
        return export_patch(
            path,
            {name: weight},
            {name: changed},
            model=self.config.model,
            revision=self.config.revision,
            label=provenance.get("label", "agent candidate"),
            provenance=provenance,
        )

    @contextlib.contextmanager
    def patched(self, path: Path):
        params = dict(self.model.named_parameters())
        apply_patch(params, path, model=self.config.model, revision=self.config.revision)
        try:
            yield
        finally:
            apply_patch(params, path, revert=True, model=self.config.model, revision=self.config.revision)

    def parity(self, examples: list[Example], max_tokens: int = 16):
        cfg = self.config.sampling.model_copy(update={"max_new_tokens": max_tokens})
        original = self.generate(examples, cfg, score=False)
        calls = []
        handle = self.layers[0].mlp.register_forward_hook(lambda m, a, o: calls.append(tuple(o.shape)))
        try:
            hooked = self.generate(examples, cfg, score=False)
        finally:
            handle.remove()
        ok = [r.text for r in original] == [r.text for r in hooked] and bool(calls)
        if not ok:
            raise RuntimeError("Unchanged-model hook parity failed")
        return {
            "passed": ok,
            "hook_calls": len(calls),
            "first_shape": calls[0],
            "weight_sha256": tensor_hash(self.layers[0].mlp.down_proj.weight),
        }
