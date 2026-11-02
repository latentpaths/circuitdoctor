from concurrent.futures import ThreadPoolExecutor

import pytest

from circuitdoctor.budget import BudgetExceeded, Ledger


def test_reservations_survive_resume_and_are_atomic(tmp_path):
    path = tmp_path / "budget.sqlite"
    ledger = Ledger(path, cap=10)
    key = ledger.reserve("pilot", "unknown request", 6)
    resumed = Ledger(path, cap=10)
    with pytest.raises(BudgetExceeded):
        resumed.reserve("pilot", "overspend", 5)
    resumed.settle(key, 1)

    def reserve(_):
        try:
            return Ledger(path, 10).reserve("pilot", "concurrent", 4)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(pool.map(reserve, range(4)))
    assert sum(x is not None for x in accepted) == 2
    assert resumed.summary()["committed_usd"] == 9


def test_stage_and_tool_limits(tmp_path):
    ledger = Ledger(tmp_path / "budget.sqlite")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("pilot", "too large", 40.01)
    ledger.consume("run", "diagnostic_batches", 1)
    with pytest.raises(BudgetExceeded):
        ledger.consume("run", "diagnostic_batches", 1)
    assert Ledger(tmp_path / "budget.sqlite").summary()["counters"][0]["value"] == 1


def test_remote_merge_preserves_local_settlement_and_deduplicates(tmp_path):
    local = Ledger(tmp_path / "local.sqlite")
    key = local.reserve("pilot", "GPU lease", 10)
    local.snapshot(tmp_path / "remote.sqlite")
    remote = Ledger(tmp_path / "remote.sqlite")
    inference = remote.reserve("experiments", "editor", 2)
    remote.settle(inference, 1)
    local.settle(key, 3)
    local.merge(tmp_path / "remote.sqlite")
    local.merge(tmp_path / "remote.sqlite")
    assert local.summary()["committed_usd"] == 4
