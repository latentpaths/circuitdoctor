"""Small, metered live Prime tool-calling integration check (no model repair claim)."""

import json
from pathlib import Path

from circuitdoctor.budget import Ledger
from circuitdoctor.prime import PrimeClient
from circuitdoctor.storage import write_json

root = Path("runs/prime-smoke")
client = PrimeClient(Ledger(Path("runs/project.sqlite")), "pilot")
client.record(root / "editor.json")
tools = [
    {
        "type": "function",
        "function": {
            "name": "observe",
            "description": "Observe model failures",
            "parameters": {
                "type": "object",
                "properties": {"max_new_tokens": {"type": "integer"}},
                "required": ["max_new_tokens"],
                "additionalProperties": False,
            },
        },
    }
]
response, usage = client.complete(
    [
        {"role": "system", "content": "Use the observe tool to inspect the target."},
        {"role": "user", "content": "Call observe with max_new_tokens 128."},
    ],
    tools,
    max_tokens=512,
)
write_json(root / "response.json", {"response": response, "usage": usage})
calls = response["choices"][0]["message"].get("tool_calls") or []
if not any(
    c["function"]["name"] == "observe" and json.loads(c["function"]["arguments"]) == {"max_new_tokens": 128}
    for c in calls
):
    raise RuntimeError("Live tool-calling check did not produce the requested call")
print(
    json.dumps(
        {
            "model": client.model,
            "serving": client.entry.get("serving"),
            "tool_calling": "passed",
            "usage": usage,
        }
    )
)
