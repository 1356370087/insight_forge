from __future__ import annotations

import pytest

from open_deep_research.budgets import (
    BudgetDimension,
    BudgetExhausted,
    RunBudgetLedger,
    RunBudgetPolicy,
)


def test_budget_ledger_reserves_settles_and_is_idempotent(tmp_path):
    ledger = RunBudgetLedger(
        "run-budget",
        runs_dir=str(tmp_path),
        policy=RunBudgetPolicy(max_model_calls=2, max_tool_calls=3),
    )

    first = ledger.reserve("model:1", BudgetDimension.MODEL_CALLS, 1)
    duplicate = ledger.reserve("model:1", BudgetDimension.MODEL_CALLS, 1)
    settled = ledger.settle("model:1", actual=1)

    assert duplicate.reservation_id == first.reservation_id
    assert settled.status == "settled"
    assert ledger.snapshot().model_calls == 1


def test_budget_ledger_rejects_over_reservation(tmp_path):
    ledger = RunBudgetLedger(
        "run-budget-exhaust",
        runs_dir=str(tmp_path),
        policy=RunBudgetPolicy(max_tool_calls=1),
    )
    ledger.reserve("tool:1", BudgetDimension.TOOL_CALLS, 1)

    with pytest.raises(BudgetExhausted) as raised:
        ledger.reserve("tool:2", BudgetDimension.TOOL_CALLS, 1)

    assert raised.value.dimension is BudgetDimension.TOOL_CALLS
    assert ledger.snapshot().exhausted is True


def test_budget_ledger_survives_reconstruction(tmp_path):
    policy = RunBudgetPolicy(max_input_tokens=100)
    first = RunBudgetLedger("run-budget-reload", runs_dir=str(tmp_path), policy=policy)
    first.reserve("input:1", BudgetDimension.INPUT_TOKENS, 40)
    first.settle("input:1", actual=35)

    second = RunBudgetLedger("run-budget-reload", runs_dir=str(tmp_path), policy=policy)

    assert second.snapshot().input_tokens == 35
