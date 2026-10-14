"""Reward-hacking defense and monitor tests."""

from __future__ import annotations

import pytest

from ledgergym.rewards.hacking_checks import (
    LeakError,
    assert_no_leak,
    count_answer_blocks,
    format_farming_rate,
    hack_monitor,
    label_entropy,
    length_drift,
    near_miss_values,
)


def test_assert_no_leak_raises_on_hit():
    with pytest.raises(LeakError):
        assert_no_leak("the answer is 42 here", ["42"])
    assert_no_leak("nothing forbidden", ["42", "hidden"])


def test_count_answer_blocks():
    assert count_answer_blocks("<answer>1</answer>") == 1
    assert count_answer_blocks("<answer>1</answer><answer>2</answer>") == 2
    assert count_answer_blocks("no blocks") == 0


def test_label_entropy_high_for_uniform_low_for_constant():
    uniform = ["a", "b", "c", "d"] * 25
    constant = ["a"] * 100
    assert label_entropy(uniform) == pytest.approx(1.386, abs=0.01)
    assert label_entropy(constant) == pytest.approx(0.0)
    assert label_entropy([]) == 0.0


def test_length_drift_summary():
    stats = length_drift([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    assert stats["mean"] == 55.0
    assert stats["max"] == 100.0
    assert stats["p50"] == 50.0 or stats["p50"] == 60.0
    assert length_drift([])["mean"] == 0.0


def test_format_farming_rate_detects_valid_but_wrong():
    infos = [
        {"format_valid": True, "passed": False},
        {"format_valid": True, "passed": False},
        {"format_valid": True, "passed": True},
        {"format_valid": False, "passed": False},
    ]
    assert format_farming_rate(infos) == pytest.approx(0.5)


def test_near_miss_values_straddle_tolerance():
    inside = near_miss_values(1000.0, tol=10.0, inside=True)
    outside = near_miss_values(1000.0, tol=10.0, inside=False)
    assert all(abs(v - 1000.0) < 10.0 for v in inside)
    assert all(abs(v - 1000.0) > 10.0 for v in outside)


def test_hack_monitor_aggregates():
    infos = [
        {"format_valid": True, "passed": False, "char_len": 100, "multiple_answers": False},
        {"format_valid": True, "passed": True, "char_len": 200, "multiple_answers": True},
    ]
    monitor = hack_monitor(infos)
    assert monitor["episodes"] == 2
    assert monitor["format_validity_rate"] == 1.0
    assert monitor["multiple_answer_rate"] == 0.5
    assert monitor["format_farming_rate"] == 0.5
    assert monitor["length"]["mean"] == 150.0
