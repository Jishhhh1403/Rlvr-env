"""Statement Reconciliation tests."""

from __future__ import annotations

import json
import re

import pytest

from ledgergym.env.families.reconciliation import ReconciliationFamily
from ledgergym.parsing import parse_response

FAMILY = ReconciliationFamily()


def _answer(gt, **overrides):
    payload = dict(gt.answer)
    payload.update(gt.intermediate)
    payload.update(overrides)
    return f"<answer>{json.dumps(payload)}</answer>"


@pytest.mark.parametrize("difficulty", range(1, 11))
def test_correct_answer_passes_at_every_difficulty(difficulty):
    _, gt = FAMILY.generate(seed=500 + difficulty, difficulty=difficulty)
    result = FAMILY.verify(gt, parse_response(_answer(gt)))
    assert result.final_correct
    assert result.intermediate_score == 1.0


def test_deterministic_by_seed():
    assert FAMILY.generate(seed=99, difficulty=5) == FAMILY.generate(seed=99, difficulty=5)


def test_duplicate_variant_has_exactly_one_duplicated_id():
    prompt, gt = FAMILY.generate(seed=3, difficulty=2)
    ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
    dup_id = gt.answer["duplicate_txn_id"]
    assert ids.count(dup_id) == 2
    for other in set(ids) - {dup_id}:
        assert ids.count(other) == 1


def test_duplicate_second_occurrence_index_matches_prompt_position():
    prompt, gt = FAMILY.generate(seed=8, difficulty=3)
    ids = re.findall(r'"id":\s*"([^"]+)"', prompt)
    dup_id = gt.answer["duplicate_txn_id"]
    positions = [i + 1 for i, value in enumerate(ids) if value == dup_id]
    assert gt.intermediate["second_occurrence_index"] == positions[-1]


def test_missing_deposit_id_is_absent_from_bank_statement():
    prompt, gt = FAMILY.generate(seed=21, difficulty=5)
    expected_block, bank_block = prompt.split("Bank statement:")
    missing_id = gt.answer["missing_txn_id"]
    assert missing_id in expected_block
    assert missing_id not in bank_block


def test_missing_deposit_amount_is_ambiguous_by_amount():
    """Reconciling must require comparing IDs, not just matching the difference."""
    prompt, gt = FAMILY.generate(seed=25, difficulty=5)
    _, expected_block = prompt.split("Customer ledger (expected):")
    expected_block, bank_block = expected_block.split("Bank statement:")
    exp_ids = re.findall(r'"id":\s*"([^"]+)"', expected_block)
    bank_ids = set(re.findall(r'"id":\s*"([^"]+)"', bank_block))
    # Salaries repeat, so at least two expected credits share the missing amount
    from ledgergym.fmt import parse_number

    amounts = re.findall(r'"amount":\s*([0-9.]+)', expected_block)
    missing_amount = gt.meta.get("difference")
    matching = [a for a in amounts if parse_number(a) == pytest.approx(missing_amount)]
    assert len(matching) >= 2
    # Exactly one expected ID is absent from the bank statement.
    assert len([i for i in exp_ids if i not in bank_ids]) == 1


def test_balance_mismatch_first_divergence_is_correct():
    prompt, gt = FAMILY.generate(seed=33, difficulty=8)
    opening = float(re.search(r"Opening balance:\s*([0-9,]+\.\d\d)", prompt).group(1).replace(",", ""))
    rows = re.findall(r"\{[^}]+\}", prompt.split("Bank statement:")[1])
    running = opening
    first_divergent = None
    for row in rows:
        data = json.loads(row)
        running += data["amount"] if data["type"] == "credit" else -data["amount"]
        if abs(round(running, 2) - data["stated_balance"]) > 0.005:
            first_divergent = data["id"]
            break
    assert first_divergent == gt.answer["first_divergent_txn_id"]


def test_wrong_id_is_rejected():
    _, gt = FAMILY.generate(seed=40, difficulty=2)
    wrong = dict(gt.answer)
    key = next(iter(wrong))
    wrong[key] = "TXN-0000"
    assert not FAMILY.verify(gt, parse_response(_answer(gt, **wrong))).final_correct


def test_difficulty_maps_to_variants():
    assert FAMILY.variant_for_difficulty(1) == "duplicate_charge"
    assert FAMILY.variant_for_difficulty(5) == "missing_deposit"
    assert FAMILY.variant_for_difficulty(9) == "balance_mismatch"


def test_leak_check_disabled_for_id_selection_tasks():
    _, gt = FAMILY.generate(seed=1, difficulty=1)
    assert FAMILY.leak_strings(gt) == []
