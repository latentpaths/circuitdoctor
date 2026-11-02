from __future__ import annotations

import copy
import json
import time

from .prime import PrimeClient
from .storage import read_json, write_json
from .tools import ToolRegistry

SYSTEM = """You investigate repetitive reasoning in an open language model. Discover and test a targeted,
reversible MLP weight repair using the supplied tools. You may examine development examples and reference
solutions and validate candidates on a separate validation split. Final tests are inaccessible.
Use observation -> contrast -> causal intervention -> patch -> validation.
Omit layers in capture to capture all blocks. Omit layer in compare to screen all captured layers.
Use returned candidate IDs exactly. Tool results report authoritative remaining budgets.
Explore several promising layers and intervention strengths. Do not assume the budget is exhausted after one
failed hypothesis. Export and validate promising candidates; if a validation fails, use remaining budget to improve it. Logit-lens labels are hypotheses,
not causal evidence. Distinguish task correctness from merely ending generation. The task and tool results specify the diagnostic and candidate limits; captures and validations also consume batches. Return a JSON object with patch_id
(or null), explanation, and limitations. Never claim success without validation. No training is allowed.
You are not given the published repair. Similarity to a known result alone does not establish novel discovery.
"""


def investigate(registry: ToolRegistry, client: PrimeClient, max_turns: int = 48):
    path = registry.root / "conversation.json"
    checkpoint = read_json(path) if path.exists() else {"messages": [], "turn": 0, "complete": False}
    if checkpoint["complete"]:
        return checkpoint["final"]
    if not checkpoint["messages"]:
        checkpoint["messages"] = [
            {
                "role": "system",
                "content": SYSTEM
                + (
                    " Internal capture/compare/inspect tools are unavailable in this behavioral condition. "
                    "Use observations, MLP ablations or column scaling, export and validate; do not invent direction IDs."
                    if registry.behavioral_only
                    else ""
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target": registry.runtime.describe(),
                        "development_examples": [
                            {"id": e.id, "prompt": e.prompt, "answer": e.answer}
                            for e in registry.selected([])
                        ],
                        "limits": {
                            "diagnostic_batches": registry.runtime.config.diagnostic_batches,
                            "candidate_patches": registry.runtime.config.candidate_patches,
                        },
                        "task": "Find a small persistent repair; retain the original if no candidate is validated.",
                    }
                ),
            },
        ]
    client.record(registry.root / "editor.json")
    while checkpoint["turn"] < max_turns:
        turn = checkpoint["turn"]
        if checkpoint.get("pending") is None:
            response_path = registry.root / f"editor_calls/{turn:03d}.json"
            if response_path.exists():
                saved = read_json(response_path)
                data = saved["response"]
            else:
                data, usage = client.complete(checkpoint["messages"], registry.schemas())
                write_json(response_path, {"response": data, "usage": usage, "time": time.time()})
            message = data["choices"][0]["message"]
            checkpoint["messages"].append(
                {
                    k: v
                    for k, v in message.items()
                    if k in ("role", "content", "tool_calls", "reasoning_content") and v is not None
                }
            )
            calls = message.get("tool_calls") or []
            checkpoint["pending"] = copy.deepcopy(calls)
            if not calls:
                content = message.get("content") or ""
                try:
                    final = json.loads(content.removeprefix("```json").removesuffix("```").strip())
                    if not isinstance(final, dict):
                        raise ValueError("Final selection must be an object")
                except ValueError:
                    final = {
                        "patch_id": None,
                        "explanation": content,
                        "limitations": "No structured final selection",
                    }
                selected = final.get("patch_id")
                if selected is not None and selected not in registry.patches:
                    final = {
                        "patch_id": None,
                        "explanation": content,
                        "limitations": "Unknown patch ID rejected",
                    }
                checkpoint.update(complete=True, final=final, turn=turn + 1, pending=None)
                write_json(path, checkpoint)
                write_json(registry.root / "selection.json", final)
                return final
            # Persist the assistant's call IDs before any side-effectful experiment.
            write_json(path, checkpoint)
        while checkpoint["pending"]:
            call = checkpoint["pending"][0]
            try:
                arguments = json.loads(call["function"]["arguments"])
                result = registry.execute(
                    call["function"]["name"], arguments, call_id=call["id"]
                ).model_dump()
            except (ValueError, KeyError, TypeError) as exc:
                result = {"status": "error", "data": {"error": str(exc)}}
            checkpoint["messages"].append(
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
            )
            checkpoint["pending"].pop(0)
            if not checkpoint["pending"]:
                checkpoint.update(pending=None, turn=turn + 1)
            write_json(path, checkpoint)
            if checkpoint["pending"] is None:
                break
    final = {"patch_id": None, "explanation": "Turn limit reached", "limitations": "No final selection"}
    write_json(registry.root / "selection.json", final)
    return final
