"""Trend task tests: rule reproducibility and label validity."""

from __future__ import annotations

import json
import re

import pytest

from ledgergym.env.families.trend import TrendFamily, generator
from ledgergym.parsing import parse_response

FAMILY = TrendFamily()


def _answer(gt, **overrides):
    payload = dict(gt.answer)
    payload.update(gt.intermediate)
    payload.update(overrides)
    return f"<answer>{json.dumps(payload)}</answer>"


def test_slope_of_line_is_exact():
    assert generator.slope([0.0, 2.0, 4.0, 6.0]) == pytest.approx(2.0)
    assert generator.slope([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_label_thresholds():
    assert generator.liquidity_label([0, 5, 10, 15]) == "improving"
    assert generator.liquidity_label([15, 10, 5, 0]) == "deteriorating"
    assert generator.liquidity_label([10, 10, 10, 10]) == "stable"


@pytest.mark.parametrize("difficulty", range(1, 11))
def test_correct_answer_passes_at_every_difficulty(difficulty):
    _, gt = FAMILY.generate(seed=700 + difficulty, difficulty=difficulty)
    result = FAMILY.verify(gt, parse_response(_answer(gt)))
    assert result.final_correct
    assert result.intermediate_score == 1.0


def test_prompt_rule_reproduces_label():
    """Recompute the label straight from the prompt table and compare to truth."""
    prompt, gt = FAMILY.generate(seed=55, difficulty=4)
    rows = re.findall(r"(2026-\d\d) \| (\d+) \| (-?[\d,]+\.\d\d) \| (-?[\d,]+\.\d\d)", prompt)
    assert rows
    buffers = []
    for _month, days, expenses, balance in rows:
        buffers.append(
            float(balance.replace(",", ""))
            * int(days)
            / float(expenses.replace(",", ""))
        )
    assert generator.liquidity_label(buffers) == gt.answer["liquidity"]
    assert gt.intermediate["buffer_days_start"] == pytest.approx(round(buffers[0], 2), abs=0.01)


def test_label_distribution_is_balanced():
    from collections import Counter

    counts = Counter(
        FAMILY.generate(seed=s, difficulty=3)[1].answer["liquidity"] for s in range(300)
    )
    assert set(counts) == {"improving", "stable", "deteriorating"}
    assert all(count > 40 for count in counts.values()), counts


def test_shocks_do_not_flip_label():
    """When a shock note is present the label must still follow the rule."""
    seen_shock = False
    for seed in range(200):
        prompt, gt = FAMILY.generate(seed=seed, difficulty=6)
        if "Note:" in prompt:
            seen_shock = True
            rows = re.findall(r"(2026-\d\d) \| (\d+) \| (-?[\d,]+\.\d\d) \| (-?[\d,]+\.\d\d)", prompt)
            buffers = [
                float(b.replace(",", "")) * int(d) / float(e.replace(",", ""))
                for _m, d, e, b in rows
            ]
            assert generator.liquidity_label(buffers) == gt.answer["liquidity"]
    assert seen_shock, "expected at least one shock episode in 200 seeds"


def test_wrong_label_is_rejected():
    _, gt = FAMILY.generate(seed=61, difficulty=3)
    wrong = {"improving": "stable", "stable": "deteriorating", "deteriorating": "improving"}
    assert not FAMILY.verify(
        gt, parse_response(_answer(gt, liquidity=wrong[gt.answer["liquidity"]]))
    ).final_correct


def test_deterministic_by_seed():
    assert FAMILY.generate(seed=5, difficulty=7) == FAMILY.generate(seed=5, difficulty=7)


def test_leak_check_disabled_for_rule_tasks():
    _, gt = FAMILY.generate(seed=2, difficulty=2)
    assert FAMILY.leak_strings(gt) == []
