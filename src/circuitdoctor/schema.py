from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sampling(StrictModel):
    thinking: bool = True
    temperature: float = Field(default=0.6, gt=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    presence_penalty: float = Field(default=1.5, ge=-2, le=2)
    max_new_tokens: int = Field(default=32768, ge=1, le=32768)
    seed: int = 0


class Example(StrictModel):
    id: str
    prompt: str
    answer: str = ""
    suite: str = "gsm8k"
    split: Literal["development", "validation", "test"] = "development"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Generation(StrictModel):
    id: str
    text: str
    generated_tokens: int
    prompt_tokens: int
    hit_token_limit: bool
    repetition: float
    correct: bool | None = None
    prediction: str | None = None
    sample: int = 0


class Intervention(StrictModel):
    layer: int = Field(ge=0)
    kind: Literal["suppress", "zero", "replace", "scale_columns"] = "suppress"
    strength: float = Field(default=1.0, ge=-2, le=2)
    direction_id: str | None = None
    capture_id: str | None = None
    columns: list[int] = Field(default_factory=list)


class ToolResult(StrictModel):
    status: Literal["ok", "error"] = "ok"
    evidence: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class RunConfig(StrictModel):
    model: str = "Qwen/Qwen3.5-0.8B"
    revision: str = "2fc06364715b967f1860aea9cf38778875588b17"
    device: str = "auto"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    batch_size: int = Field(default=4, ge=1, le=64)
    evaluation_eager: bool = False
    sampling: Sampling = Field(default_factory=Sampling)
    diagnostic_batches: int = Field(default=25, ge=1)
    candidate_patches: int = Field(default=10, ge=1)
    validation_max_tokens: int = Field(default=4096, ge=16, le=32768)
    capture_token_limit: int = Field(default=2048, ge=1, le=8192)


class PatchTensor(StrictModel):
    name: str
    shape: list[int]
    dtype: str
    base_sha256: str
    edited_sha256: str
    changed_entries: int


class PatchManifest(StrictModel):
    schema_version: int = 1
    base_model: str
    revision: str
    label: str
    tensors: list[PatchTensor]
    provenance: dict[str, Any] = Field(default_factory=dict)
    evaluation: list[str] = Field(default_factory=list)

    @property
    def changed_entries(self) -> int:
        return sum(t.changed_entries for t in self.tensors)
