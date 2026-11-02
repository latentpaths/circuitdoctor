import httpx
import pytest

from circuitdoctor.budget import Ledger
from circuitdoctor.prime import PrimeClient


def test_catalog_billing_context_and_usage_accounting(tmp_path, monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["X-Prime-Team-ID"] == "team-fixture"
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "fixture",
                            "serving": "hosted",
                            "pricing": {"input_usd_per_mtok": 1, "output_usd_per_mtok": 2},
                            "supported_parameters": ["tools"],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: real_client(**kw, transport=httpx.MockTransport(handle))
    )
    ledger = Ledger(tmp_path / "ledger.sqlite")
    client = PrimeClient(ledger, "experiments", model="fixture", key="test-key", team="team-fixture")
    _, usage = client.complete([{"role": "user", "content": "hello"}], [], max_tokens=32)
    assert usage["usd"] == pytest.approx(20 / 1e6)
    assert ledger.summary()["committed_usd"] == pytest.approx(20 / 1e6)
    assert len(calls) == 2
