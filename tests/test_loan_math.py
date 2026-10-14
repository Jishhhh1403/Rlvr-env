"""Loan Math tests: independent cross-checks, tolerances, and boundaries."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ledgergym.env.families.loan_math import generator
from ledgergym.env.families.loan_math import LoanMathFamily
from ledgergym.parsing import parse_response

FAMILY = LoanMathFamily()


def _answer(gt, **overrides):
    payload = dict(gt.answer)
    payload.update(gt.intermediate)
    payload.update(overrides)
    return f"<answer>{json.dumps(payload)}</answer>"


# -- independent implementation cross-check --------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    principal=st.floats(min_value=10_000, max_value=5_000_000),
    rate=st.floats(min_value=0.5, max_value=30.0),
    months=st.integers(min_value=1, max_value=360),
)
def test_closed_form_emi_pays_off_loan(principal, rate, months):
    """The closed-form EMI must amortize the loan to ~zero in exactly `months`."""
    payment = generator.emi(principal, rate, months)
    r = rate / 12.0 / 100.0
    balance = principal
    for _ in range(months):
        balance = balance * (1 + r) - payment
    assert abs(balance) <= max(1.0, principal * 1e-6)


@settings(max_examples=200, deadline=None)
@given(
    principal=st.floats(min_value=50_000, max_value=5_000_000),
    rate=st.floats(min_value=1.0, max_value=25.0),
    months=st.integers(min_value=2, max_value=240),
)
def test_closed_form_matches_simulation(principal, rate, months):
    payment = generator.emi(principal, rate, months)
    k = months // 2
    closed = generator.closed_form_balance(principal, rate, k, payment)
    sim = generator.amortize(principal, rate, months, emi_value=payment)
    simulated = sim["schedule"][k - 1]["balance"]
    assert abs(closed - simulated) <= max(1.0, principal * 1e-6)


@given(
    prepay=st.floats(min_value=1_000, max_value=200_000),
    month=st.integers(min_value=1, max_value=40),
)
@settings(max_examples=100, deadline=None)
def test_prepayment_reduces_total_interest(prepay, month):
    principal, rate, months = 2_000_000.0, 11.0, 60
    payment = generator.emi(principal, rate, months)
    base = generator.amortize(principal, rate, months, emi_value=payment)
    with_prepay = generator.amortize(
        principal, rate, months, prepayments=[(month, prepay)], emi_value=payment
    )
    if with_prepay["payoff_month"] < 0:
        pytest.skip("prepayment did not amortize within horizon")
    assert with_prepay["total_interest"] <= base["total_interest"] + 1e-6


# -- generator / verifier behavior -----------------------------------------------


@pytest.mark.parametrize("difficulty", range(1, 11))
def test_correct_answer_passes_at_every_difficulty(difficulty):
    prompt, gt = FAMILY.generate(seed=100 + difficulty, difficulty=difficulty)
    assert "ground" not in prompt.lower()
    result = FAMILY.verify(gt, parse_response(_answer(gt)))
    assert result.final_correct
    assert result.intermediate_score == 1.0


def test_deterministic_by_seed():
    assert FAMILY.generate(seed=7, difficulty=6) == FAMILY.generate(seed=7, difficulty=6)


def test_wrong_answer_is_rejected():
    _, gt = FAMILY.generate(seed=11, difficulty=1)
    wrong = gt.answer.copy()
    key = next(iter(wrong))
    wrong[key] = float(wrong[key]) * 2 + 5000
    result = FAMILY.verify(gt, parse_response(_answer(gt, **wrong)))
    assert not result.final_correct


def test_malformed_answer_scores_zero_correctness():
    _, gt = FAMILY.generate(seed=12, difficulty=1)
    result = FAMILY.verify(gt, parse_response("no answer block here"))
    assert not result.final_correct
    assert result.intermediate_score == 0.0


def test_tolerance_boundary_accepts_inside_rejects_outside():
    """Numeric tolerance is `<= tol` and near-misses outside it must fail."""
    _, gt = FAMILY.generate(seed=13, difficulty=1)
    emi_key = "emi"
    truth = float(gt.answer[emi_key])
    tol = gt.tolerances[emi_key]

    inside = FAMILY.verify(gt, parse_response(_answer(gt, **{emi_key: truth + tol * 0.99})))
    outside = FAMILY.verify(gt, parse_response(_answer(gt, **{emi_key: truth + tol * 1.01})))
    assert inside.final_correct
    assert not outside.final_correct


def test_accepts_indian_grouped_string_answers():
    _, gt = FAMILY.generate(seed=14, difficulty=1)
    from ledgergym.fmt import format_inr

    truth = gt.answer["emi"]
    grouped = format_inr(truth, decimals=2)
    result = FAMILY.verify(gt, parse_response(_answer(gt, emi=grouped)))
    assert result.final_correct


def test_rate_comparison_has_a_unique_winner():
    for seed in range(50):
        _, gt = FAMILY.generate(seed=seed, difficulty=9)
        assert gt.answer["cheaper_offer"] in {"A", "B"}
        assert gt.answer["savings"] > 0
        assert abs(gt.intermediate["total_cost_a"] - gt.intermediate["total_cost_b"]) > 0


def test_prepayment_answers_are_positive():
    for seed in range(50):
        _, gt = FAMILY.generate(seed=seed, difficulty=6)
        assert gt.answer["new_tenure_months"] >= 1
        assert gt.answer["total_interest"] > 0
