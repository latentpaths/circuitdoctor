import pytest

from circuitdoctor.agent import investigate
from circuitdoctor.budget import Ledger
from circuitdoctor.schema import Example
from circuitdoctor.tools import ToolRegistry


def test_resume_reuses_saved_response_and_completed_tool(runtime, tmp_path, monkeypatch):
    class Editor:
        calls = 0

        def record(self, path):
            pass

        def complete(self, messages, schemas):
            self.calls += 1
            if self.calls > 1:
                assistant = next(m for m in messages if m["role"] == "assistant")
                assert assistant["tool_calls"][0]["id"] == "call-observe"
            if self.calls == 1:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-observe",
                            "type": "function",
                            "function": {"name": "observe", "arguments": '{"max_new_tokens":16}'},
                        }
                    ],
                }
            else:
                message = {"role": "assistant", "content": '{"patch_id":null,"explanation":"No repair"}'}
            return {"choices": [{"message": message}]}, {"usd": 0}

    editor = Editor()
    examples = [Example(id="dev", prompt="What is 2+3?", answer="5")]
    ledger = Ledger(tmp_path / "ledger.sqlite")
    registry = ToolRegistry(runtime, examples, tmp_path / "run", ledger)
    execute = registry.execute

    def interrupt_after_tool(*args, **kwargs):
        execute(*args, **kwargs)
        raise RuntimeError("Simulated interruption after durable tool completion")

    monkeypatch.setattr(registry, "execute", interrupt_after_tool)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        investigate(registry, editor)
    resumed = ToolRegistry(runtime, examples, tmp_path / "run", ledger)
    result = investigate(resumed, editor)
    assert result["patch_id"] is None
    assert editor.calls == 2
    counts = {r["name"]: r["value"] for r in ledger.summary()["counters"]}
    assert counts["diagnostic_batches"] == 1
