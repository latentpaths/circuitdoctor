from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from .budget import Ledger
from .storage import write_json


class PrimeClient:
    def __init__(
        self,
        ledger: Ledger,
        stage: str,
        *,
        model: str = "openai/gpt-oss-20b",
        key: str | None = None,
        team: str | None = None,
        base_url: str | None = None,
    ):
        key = key or os.environ.get("PRIME_API_KEY")
        if not key:
            raise ValueError("Set PRIME_API_KEY in the process environment")
        self.model, self.ledger, self.stage = model, ledger, stage
        self.client = httpx.Client(
            base_url=base_url or "https://api.pinference.ai/api/v1",
            timeout=300,
            headers={
                "Authorization": f"Bearer {key}",
                **(
                    {"X-Prime-Team-ID": team or os.environ["PRIME_TEAM_ID"]}
                    if team or os.environ.get("PRIME_TEAM_ID")
                    else {}
                ),
            },
        )
        response = self.client.get("/models")
        response.raise_for_status()
        catalog = response.json()
        self.entry = next(
            (m for m in catalog.get("data", catalog.get("models", [])) if m["id"] == model), None
        )
        if self.entry is None:
            raise ValueError("Model not in current Prime catalog")
        if "tools" not in (self.entry.get("supported_parameters") or []):
            raise ValueError("Catalog does not declare tool support for this model")

    def complete(self, messages: list, tools: list, max_tokens: int = 4096):
        pricing = self.entry["pricing"]
        # UTF-8 bytes + chat/schema overhead is deliberately conservative for pre-dispatch reservation.
        input_bound = (
            len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()) + 4096
        )
        upper = (
            input_bound * pricing["input_usd_per_mtok"] + max_tokens * pricing["output_usd_per_mtok"]
        ) / 1e6
        reservation = self.ledger.reserve(self.stage, "editor:" + self.model, upper)
        started = time.monotonic()
        response = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": max_tokens,
                "temperature": 0.6,
            },
        )
        # For unknown request outcomes retain the reservation. No unmetered automatic retries.
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        estimated = (
            usage.get("prompt_tokens", usage.get("input_tokens", input_bound)) * pricing["input_usd_per_mtok"]
            + usage.get("completion_tokens", usage.get("output_tokens", max_tokens))
            * pricing["output_usd_per_mtok"]
        ) / 1e6
        reported = usage.get("cost")
        if isinstance(reported, dict):
            reported = reported.get("total_cost")
        # A provider may round sub-cent costs to zero. Keep the catalog estimate accounted until billing reconciliation.
        amount = max(estimated, float(reported)) if reported is not None else estimated
        self.ledger.settle(reservation, amount)
        return data, {
            **usage,
            "usd": amount,
            "provider_reported_usd": reported,
            "catalog_estimated_usd": estimated,
            "cost_source": "conservative catalog estimate / provider maximum",
            "seconds": time.monotonic() - started,
        }

    def record(self, path: Path):
        write_json(
            path,
            {
                "catalog_entry": self.entry,
                "base_url": str(self.client.base_url),
                "recorded_at": time.time(),
                "team_billing": "X-Prime-Team-ID" in self.client.headers,
            },
        )
