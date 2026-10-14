"""Reward aggregation tests: dominance rules, modes, and anti-farming behavior."""

from __future__ import annotations

import pytest

from ledgergym.env.families.base import GroundTruth, VerifyResult
from ledgergym.parsing import parse_response
from ledgergym.rewards.components import RewardConfig, compute_reward

VALID = '<answer>{"emi": 100.0}</answer>'


def _parsed(valid: bool):
    return parse_response(VALID if valid else "garbage without a block")


def _verify(final_correct: bool, intermediate: float = 0.0, rel_error=None) -> VerifyResult:
    return VerifyResult(
        final_correct=final_correct, intermediate_score=intermediate, rel_error=rel_error
    )


def test_shaped_correct_answer_gets_full_correctness_weight():
    reward, components = compute_reward(_parsed(True), _verify(True, 1.0), RewardConfig())
    assert reward == pytest.approx(1.0)
    assert components["correctness"] == 1.0
    assert components["format"] == 1.0
    assert components["intermediate"] == 1.0


def test_correctness_dominates_format_and_intermediate():
    """Format + intermediate alone (max 0.20) must never beat a correct answer."""
    shaped = RewardConfig()
    farming_reward, _ = compute_reward(_parsed(True), _verify(False, 1.0), shaped)
    correct_reward, _ = compute_reward(_parsed(True), _verify(True, 0.0), shaped)
    assert farming_reward == pytest.approx(0.20)
    assert correct_reward == pytest.approx(0.90)
    assert correct_reward > farming_reward


def test_malformed_output_zeroes_correctness_and_length_penalty_applies():
    """Defense: format cannot be farmed; malformed output can never score correctness."""
    short_garbage = parse_response("nope")
    reward, components = compute_reward(short_garbage, _verify(True, 1.0), RewardConfig())
    assert components["correctness"] == 0.0
    assert components["intermediate"] == 0.0
    assert components["length_penalty"] == pytest.approx(-0.10)  # empty/short output penalty
    assert reward == pytest.approx(-0.10)


def test_binary_mode_is_one_or_zero():
    cfg = RewardConfig(mode="binary")
    assert compute_reward(_parsed(True), _verify(True), cfg)[0] == 1.0
    assert compute_reward(_parsed(True), _verify(False), cfg)[0] == 0.0
    assert compute_reward(_parsed(False), _verify(True), cfg)[0] == 0.0


def test_gradient_mode_gives_decaying_partial_credit():
    cfg = RewardConfig(mode="gradient", gradient_scale=0.05)
    near = compute_reward(_parsed(True), _verify(False, 0.0, rel_error=0.01), cfg)[0]
    far = compute_reward(_parsed(True), _verify(False, 0.0, rel_error=0.20), cfg)[0]
    assert far < near < 0.80 + 0.10  # never reaches a fully-correct answer
    # A correct answer still gets full correctness even in gradient mode.
    exact = compute_reward(_parsed(True), _verify(True, 1.0), cfg)[0]
    assert exact == pytest.approx(1.0)


def test_length_penalty_applies_to_overlong_output():
    cfg = RewardConfig(max_length_chars=10)
    reward, components = compute_reward(_parsed(True), _verify(True, 1.0), cfg)
    assert components["length_penalty"] == pytest.approx(-cfg.length_penalty)
    assert reward == pytest.approx(1.0 - cfg.length_penalty)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        RewardConfig(mode="nonsense")


def test_reward_is_clamped():
    cfg = RewardConfig(mode="gradient", gradient_scale=0.0)
    reward, _ = compute_reward(_parsed(True), _verify(False, 0.0, rel_error=0.5), cfg)
    assert -0.10 <= reward <= 1.0


def test_ground_truth_default_leak_strings_cover_numeric_answers():
    gt = GroundTruth(
        family="x",
        variant="v",
        difficulty=1,
        seed=0,
        answer={"emi": 16012.45, "label": "improving"},
    )
    forbidden = GroundTruth.leak_forbidden_strings(gt)
    # Base model has no family, so only the documented default applies (empty here).
    assert isinstance(forbidden, list)
    from ledgergym.env.families.base import TaskFamily

    class _F(TaskFamily):
        name = "f"

        def generate(self, seed, difficulty, variant=None):  # pragma: no cover
            raise NotImplementedError

    strings = _F().leak_strings(gt)
    assert "16012.45" in strings
    assert "improving" not in strings
