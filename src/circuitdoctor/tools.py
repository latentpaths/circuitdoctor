"""The investigator's entire capability surface: opaque IDs, never arbitrary paths/code."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from safetensors.torch import load_file, save_file

from .budget import Ledger
from .metrics import paired_comparison, summarize
from .patch import export_patch, record_evidence, suppress_weights
from .runtime import Runtime
from .schema import Example, Generation, Intervention, StrictModel, ToolResult
from .storage import digest, environment_versions, instrumentation_digest, read_json, write_json


class ObserveArgs(StrictModel):
    ids: list[str] = Field(default_factory=list, max_length=32)
    max_new_tokens: int = Field(default=2048, ge=16, le=32768)


class CaptureArgs(StrictModel):
    observation_id: str
    layers: list[int] = Field(default_factory=list, max_length=24)
    source: Literal["generated", "reference"] = "generated"


class CompareArgs(StrictModel):
    failed_capture_id: str
    reference_capture_id: str
    layer: int | None = Field(default=None, ge=0)
    method: Literal["mean_difference", "pca"] = "mean_difference"


class InspectArgs(StrictModel):
    direction_id: str
    layer: int = Field(ge=0)
    capture_id: str | None = None
    top_k: int = Field(default=16, ge=1, le=64)


class InterveneArgs(ObserveArgs):
    intervention: Intervention


class SweepArgs(ObserveArgs):
    interventions: list[Intervention] = Field(min_length=1, max_length=5)


class ExportArgs(StrictModel):
    interventions: list[Intervention] = Field(min_length=1, max_length=4)
    hypothesis: str = Field(min_length=1, max_length=2000)


class ValidateArgs(StrictModel):
    patch_id: str


class ToolRegistry:
    descriptions = {
        "observe": "Generate on development examples and measure failures; returns compact summaries and trace IDs.",
        "capture": "Capture layer/MLP/neuron summaries from generated or reference-solution traces. Omit layers to capture all layers.",
        "compare": "Contrast failing generated traces with reference traces. Omit layer to screen all captured layers; returns candidate direction IDs.",
        "inspect": "Decode a candidate direction with a logit lens and rank MLP neurons writing that direction.",
        "intervene": "Causally test one temporary MLP intervention on development examples.",
        "sweep": "Compare up to five interventions; each consumes one diagnostic batch.",
        "export_patch": "Export up to four MLP weight interventions as one exact reversible candidate patch.",
        "validate": "Test a candidate against the original on validation data; cannot access final evaluation data.",
    }
    types = {
        "observe": ObserveArgs,
        "capture": CaptureArgs,
        "compare": CompareArgs,
        "inspect": InspectArgs,
        "intervene": InterveneArgs,
        "sweep": SweepArgs,
        "export_patch": ExportArgs,
        "validate": ValidateArgs,
    }

    def __init__(
        self,
        runtime: Runtime,
        examples: list[Example],
        root: Path,
        ledger: Ledger,
        *,
        behavioral_only: bool = False,
    ):
        if any(e.split == "test" for e in examples):
            raise ValueError("Final evaluation examples may not enter an investigator registry")
        self.runtime, self.root, self.ledger = runtime, root, ledger
        self.examples = {e.id: e for e in examples}
        if len(self.examples) != len(examples):
            raise ValueError("Duplicate example IDs")
        self.behavioral_only = behavioral_only
        root.mkdir(parents=True, exist_ok=True)
        identity = {
            "instrumentation_sha256": instrumentation_digest(),
            "environment": environment_versions(),
            "device": runtime.device,
            "config": runtime.config.model_dump(),
            "behavioral_only": behavioral_only,
            "examples_sha256": digest([e.model_dump() for e in examples]),
        }
        if (root / "run-identity.json").exists() and read_json(root / "run-identity.json") != identity:
            raise ValueError("Run configuration or data changed; use a fresh output directory")
        write_json(root / "run-identity.json", identity)
        self.allowed = (
            {"observe", "intervene", "sweep", "export_patch", "validate"}
            if behavioral_only
            else set(self.types)
        )
        # Resume only artifacts created by this run; opaque IDs never become filesystem paths.
        state = read_json(root / "tools.json") if (root / "tools.json").exists() else {}
        self.run_id = state.get("run_id", uuid.uuid4().hex)
        self.observations = state.get("observations", {})
        self.captures = state.get("captures", {})
        self.directions = state.get("directions", {})
        self.patches = state.get("patches", {})
        self.completed_calls = state.get("completed_calls", {})
        for key, info in self.directions.items():
            runtime._directions[key] = load_file(root / info["file"])["direction"]
        for key, info in self.captures.items():
            tensors = load_file(root / info["file"])
            runtime._captures[key] = {k.split(".", 1)[1]: v for k, v in tensors.items() if k.startswith("0.")}

        self.save()

    def schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": self.descriptions[name],
                    "parameters": self.types[name].model_json_schema(),
                },
            }
            for name in self.types
            if name in self.allowed
        ]

    def save(self):
        write_json(
            self.root / "tools.json",
            {
                "run_id": self.run_id,
                "observations": self.observations,
                "captures": self.captures,
                "directions": self.directions,
                "patches": self.patches,
                "completed_calls": self.completed_calls,
            },
        )

    def execute(self, name: str, arguments: dict, call_id: str | None = None) -> ToolResult:
        identity = digest({"name": name, "arguments": arguments})
        if call_id is not None and call_id in self.completed_calls:
            cached = self.completed_calls[call_id]
            if cached["identity"] != identity:
                raise ValueError("Tool call ID reused with changed arguments")
            return ToolResult.model_validate(cached["result"])
        started = time.monotonic()
        if name not in self.allowed:
            return ToolResult(status="error", data={"error": "Tool unavailable"})
        try:
            args = self.types[name].model_validate(arguments)
            result = getattr(self, name)(args)
        except Exception as exc:
            # No stack traces, secrets, arbitrary directory listings, or exception repr to the editor.
            result = ToolResult(status="error", data={"error": str(exc)[:1000], "type": type(exc).__name__})
        counts = {r["name"]: r["value"] for r in self.ledger.summary()["counters"] if r["run"] == self.run_id}
        result.usage["remaining_diagnostic_batches"] = self.runtime.config.diagnostic_batches - counts.get(
            "diagnostic_batches", 0
        )
        result.usage["remaining_candidate_patches"] = self.runtime.config.candidate_patches - counts.get(
            "candidate_patches", 0
        )
        result.usage["elapsed_seconds"] = time.monotonic() - started
        event = {"time": time.time(), "tool": name, "arguments": arguments, "result": result.model_dump()}
        with (self.root / "events.jsonl").open("a") as f:
            import json

            f.write(json.dumps(event) + "\n")
        if call_id is not None:
            self.completed_calls[call_id] = {"identity": identity, "result": result.model_dump()}
        self.save()
        return result

    def selected(self, ids: list[str], split: str = "development"):
        candidates = [e for e in self.examples.values() if e.split == split]
        if ids:
            if any(i not in self.examples or self.examples[i].split != split for i in ids):
                raise ValueError("Unknown example or inaccessible split")
            candidates = [self.examples[i] for i in ids]
        if not candidates:
            raise ValueError(f"No {split} examples")
        return candidates[:16]

    def _batch(self):
        self.ledger.consume(self.run_id, "diagnostic_batches", self.runtime.config.diagnostic_batches)

    def _generate(self, args: ObserveArgs, intervention=None):
        self._batch()
        examples = self.selected(args.ids)
        sampling = self.runtime.config.sampling.model_copy(update={"max_new_tokens": args.max_new_tokens})
        rows = self.runtime.generate(examples, sampling, intervention)
        key = "obs_" + uuid.uuid4().hex[:12]
        file = f"observations/{key}.json"
        write_json(
            self.root / file,
            {
                "rows": [r.model_dump() for r in rows],
                "sampling": sampling.model_dump(),
                "intervention": intervention.model_dump() if intervention else None,
            },
        )
        self.observations[key] = {"file": file}
        self._count(rows)
        return ToolResult(
            evidence=[file],
            data={
                "observation_id": key,
                "summary": summarize(rows),
                "examples": [
                    {
                        "id": r.id,
                        "correct": r.correct,
                        "hit_token_limit": r.hit_token_limit,
                        "repetition": r.repetition,
                        "tail": r.text[-1400:],
                    }
                    for r in rows
                ],
            },
            usage=dict(self.runtime.last_usage),
        )

    def _count(self, rows):
        for name, count in (
            ("sequences", len(rows)),
            ("generated_tokens", sum(r.generated_tokens for r in rows)),
            ("prompt_tokens", sum(r.prompt_tokens for r in rows)),
        ):
            self.ledger.consume(self.run_id, name, 10**12, count)

    def observe(self, args: ObserveArgs):
        return self._generate(args)

    def capture(self, args: CaptureArgs):
        self._batch()
        if args.observation_id not in self.observations:
            raise ValueError("Unknown observation")
        data = read_json(self.root / self.observations[args.observation_id]["file"])
        rows = [Generation.model_validate(r) for r in data["rows"]]
        if args.source == "generated":
            rows = [r for r in rows if r.hit_token_limit or r.repetition > 0.2 or not r.correct]
            texts = [self.examples[r.id].prompt + "\n" + r.text[-12000:] for r in rows]
        else:
            texts = [
                self.examples[r.id].prompt
                + "\n"
                + self.examples[r.id].metadata.get(
                    "solution", "The answer is \\boxed{" + self.examples[r.id].answer + "}."
                )
                for r in rows
            ]
        if not texts:
            raise ValueError("No failing traces; try a longer generation before capture")
        layers = args.layers or list(range(len(self.runtime.layers)))
        tensors = self.runtime.capture(texts[:8], layers)
        key, file = "cap_" + uuid.uuid4().hex[:12], f"captures/{uuid.uuid4().hex}.safetensors"
        (self.root / file).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, self.root / file)
        self.captures[key] = {
            "file": file,
            "source": args.source,
            "layers": layers,
            "n": min(8, len(texts)),
            "observation_id": args.observation_id,
            "window": f"last {self.runtime.config.capture_token_limit} tokens; mean/last summaries over final 64 positions",
        }
        self.runtime._captures[key] = {
            k.split(".", 1)[1]: v for k, v in tensors.items() if k.startswith("0.")
        }
        self.ledger.consume(self.run_id, "capture_sequences", 10**12, min(8, len(texts)))
        self.ledger.consume(self.run_id, "capture_tokens", 10**12, self.runtime.last_usage["capture_tokens"])
        return ToolResult(
            evidence=[file],
            data={"capture_id": key, **self.captures[key]},
            usage=dict(self.runtime.last_usage),
        )

    def compare(self, args: CompareArgs):
        if args.failed_capture_id not in self.captures or args.reference_capture_id not in self.captures:
            raise ValueError("Unknown capture")
        a, b = self.captures[args.failed_capture_id], self.captures[args.reference_capture_id]
        if a["source"] != "generated" or b["source"] != "reference":
            raise ValueError("Compare failing generated traces against reference traces")
        if args.layer is None:
            layers = sorted(set(a["layers"]) & set(b["layers"]))
            results = []
            evidence = []
            for layer in layers:
                result = self.compare(args.model_copy(update={"layer": layer}))
                results.append({"layer": layer, **result.data})
                evidence.extend(result.evidence)
            results.sort(key=lambda r: r["relative_contrast"], reverse=True)
            return ToolResult(
                evidence=evidence,
                data={
                    "candidates": results,
                    "interpretation": "Screening associations across layers; causal intervention required",
                },
            )
        suffix = f"residual.{args.layer}.mean"

        def matrix(info):
            tensors = load_file(self.root / info["file"])
            values = [v for k, v in tensors.items() if k.endswith("." + suffix)]
            if not values:
                raise ValueError("Layer not captured")
            return torch.stack(values)

        x, y = matrix(a), matrix(b)
        difference = x.mean(0) - y.mean(0)
        if args.method == "pca":
            centered = x - y.mean(0)
            _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
            difference = vectors[0]
            if torch.dot(difference, x.mean(0) - y.mean(0)) < 0:
                difference = -difference
        norm = difference.norm()
        if norm < 1e-8:
            raise ValueError("No measurable contrast")
        direction = difference / norm
        key, file = "dir_" + uuid.uuid4().hex[:12], f"directions/{uuid.uuid4().hex}.safetensors"
        (self.root / file).parent.mkdir(parents=True, exist_ok=True)
        save_file({"direction": direction}, self.root / file)
        self.runtime._directions[key] = direction
        self.directions[key] = {"file": file, "layer": args.layer, "method": args.method}
        return ToolResult(
            evidence=[file],
            data={
                "direction_id": key,
                "contrast_norm": float(norm),
                "relative_contrast": float(
                    norm / ((x.norm(dim=1).mean() + y.norm(dim=1).mean()) / 2).clamp_min(1e-8)
                ),
                "interpretation": "Association only; intervention required to test behavioral effect",
            },
        )

    def inspect(self, args: InspectArgs):
        if args.direction_id not in self.runtime._directions or args.layer >= len(self.runtime.layers):
            raise ValueError("Unknown direction or layer")
        u = self.runtime._directions[args.direction_id].to(self.runtime.device)
        weight = self.runtime.layers[args.layer].mlp.down_proj.weight.float()
        contributions = u @ weight
        if args.capture_id:
            if args.capture_id not in self.captures:
                raise ValueError("Unknown capture")
            tensors = load_file(self.root / self.captures[args.capture_id]["file"])
            activations = [v for k, v in tensors.items() if k.endswith(f".neurons.{args.layer}.mean")]
            if not activations:
                raise ValueError("Neuron activations not captured")
            contributions *= torch.stack(activations).mean(0).to(weight.device)
        values, indices = contributions.topk(args.top_k)
        with torch.no_grad():
            norm = self.runtime.transformer.norm
            decoded = self.runtime.model.get_output_embeddings()(
                norm(u.to(self.runtime.model.get_output_embeddings().weight.dtype))
            )
            scores, tokens = decoded.float().topk(args.top_k)
        return ToolResult(
            data={
                "neurons": [
                    {"column": int(i), "score": float(v)} for i, v in zip(indices, values, strict=True)
                ],
                "logit_lens": [
                    {"token": self.runtime.tokenizer.decode([int(i)]), "score": float(v)}
                    for i, v in zip(tokens, scores, strict=True)
                ],
                "interpretation": "Signed scores rank neurons writing the failure direction; labels and rankings are hypotheses, not a mechanism",
            }
        )

    def intervene(self, args: InterveneArgs):
        return self._generate(args, args.intervention)

    def sweep(self, args: SweepArgs):
        results = [
            self._generate(ObserveArgs(ids=args.ids, max_new_tokens=args.max_new_tokens), s)
            for s in args.interventions
        ]
        return ToolResult(
            evidence=[e for r in results for e in r.evidence],
            data={"results": [r.model_dump() for r in results]},
        )

    def export_patch(self, args: ExportArgs):
        self.ledger.consume(self.run_id, "candidate_patches", self.runtime.config.candidate_patches)
        base, edited = {}, {}
        for spec in args.interventions:
            if spec.layer >= len(self.runtime.layers) or spec.kind == "replace":
                raise ValueError("Only valid MLP weight interventions can be exported")
            name = self.runtime.weight_name(spec.layer)
            original = self.runtime.layers[spec.layer].mlp.down_proj.weight.detach().cpu()
            base.setdefault(name, original.clone())
            w = edited.get(name, original).clone()
            if spec.kind == "suppress":
                if spec.direction_id not in self.runtime._directions:
                    raise ValueError("Unknown direction")
                w = suppress_weights(
                    w, self.runtime._directions[spec.direction_id], spec.strength, spec.columns
                )
            elif spec.kind == "zero":
                w *= 1 - spec.strength
            elif spec.kind == "scale_columns":
                if not spec.columns or min(spec.columns) < 0 or max(spec.columns) >= w.shape[1]:
                    raise ValueError("Valid columns required")
                w[:, spec.columns] *= spec.strength
            edited[name] = w
        key = "patch_" + uuid.uuid4().hex[:12]
        file = f"patches/{key}"
        manifest = export_patch(
            self.root / file,
            base,
            edited,
            model=self.runtime.config.model,
            revision=self.runtime.config.revision,
            label="candidate weight patch",
            provenance={
                "hypothesis": args.hypothesis,
                "investigation": self.root.name,
                "run_id": self.run_id,
                "interventions": [s.model_dump() for s in args.interventions],
            },
        )
        self.patches[key] = {"file": file, "manifest": manifest.model_dump()}
        return ToolResult(
            evidence=[file + "/manifest.json"],
            data={"patch_id": key, "changed_entries": manifest.changed_entries},
        )

    def validate(self, args: ValidateArgs):
        if args.patch_id not in self.patches:
            raise ValueError("Unknown patch")
        path = self.root / self.patches[args.patch_id]["file"]
        manifest_hash = digest(read_json(path / "manifest.json"))
        file = f"validation/{args.patch_id}.json"
        if (self.root / file).exists():
            cached = read_json(self.root / file)
            if cached.get("patch_manifest_sha256") != manifest_hash:
                raise ValueError("Validated patch changed")
            return ToolResult(
                evidence=[file],
                data={
                    "patch_id": args.patch_id,
                    "comparison": cached["comparison"],
                    "note": "Cached validation; final evaluation remains sealed",
                },
                usage={"cached": True},
            )
        self._batch()
        examples = self.selected([], "validation")
        sampling = self.runtime.config.sampling.model_copy(
            update={"max_new_tokens": self.runtime.config.validation_max_tokens}
        )
        baseline_file = self.root / "validation/baseline.json"
        if baseline_file.exists():
            base = [Generation.model_validate(row) for row in read_json(baseline_file)["rows"]]
            base_usage = {}
        else:
            base = self.runtime.generate(examples, sampling)
            base_usage = dict(self.runtime.last_usage)
            self._count(base)
            write_json(
                baseline_file,
                {
                    "rows": [r.model_dump() for r in base],
                    "sampling": sampling.model_dump(),
                    "usage": base_usage,
                },
            )
        with self.runtime.patched(path):
            edited = self.runtime.generate(examples, sampling)
        self._count(edited)
        comparison = paired_comparison(base, edited)
        file = f"validation/{args.patch_id}.json"
        write_json(
            self.root / file,
            {
                "comparison": comparison,
                "patch_manifest_sha256": manifest_hash,
                "baseline": [r.model_dump() for r in base],
                "edited": [r.model_dump() for r in edited],
                "sampling": sampling.model_dump(),
            },
        )
        record_evidence(path, self.root / file, "validation")
        return ToolResult(
            evidence=[file],
            data={
                "patch_id": args.patch_id,
                "comparison": comparison,
                "note": "Development validation only; final evaluation remains sealed",
            },
            usage={
                key: base_usage.get(key, 0) + self.runtime.last_usage.get(key, 0)
                for key in set(base_usage) | set(self.runtime.last_usage)
            },
        )
