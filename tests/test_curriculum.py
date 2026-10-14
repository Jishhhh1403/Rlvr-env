"""Curriculum controller tests: thresholds, clamping, and easier mixing."""

from __future__ import annotations

from ledgergym.config import CurriculumSection
from ledgergym.env.curriculum import CurriculumController


def _controller(**kw) -> CurriculumController:
    params = {"window": 10, "up_threshold": 0.7, "down_threshold": 0.3, "easier_mix": 0.0}
    params.update(kw)
    cfg = CurriculumSection(**params)
    return CurriculumController(cfg, seed=0)


def test_holds_at_start():
    controller = _controller()
    assert controller.next_difficulty("loan_math") == 1


def test_increases_on_high_pass_rate():
    controller = _controller()
    for _ in range(10):
        controller.record("loan_math", 1, passed=True)
    assert controller.pass_rate("loan_math") == 1.0
    assert controller.next_difficulty("loan_math") == 2


def test_decreases_on_low_pass_rate():
    controller = _controller()
    controller._difficulty["loan_math"] = 5
    for _ in range(10):
        controller.record("loan_math", 5, passed=False)
    assert controller.next_difficulty("loan_math") == 4


def test_holds_in_middle_band():
    controller = _controller()
    controller._difficulty["loan_math"] = 5
    for i in range(10):
        controller.record("loan_math", 5, passed=i < 5)  # 50% pass rate
    assert controller.next_difficulty("loan_math") == 5


def test_clamps_to_bounds():
    controller = _controller(min_difficulty=1, max_difficulty=3)
    for _ in range(10):
        controller.record("loan_math", 3, passed=True)
    assert controller.next_difficulty("loan_math") <= 3

    controller2 = _controller(min_difficulty=1, max_difficulty=3)
    controller2._difficulty["loan_math"] = 1
    for _ in range(10):
        controller2.record("loan_math", 1, passed=False)
    assert controller2.next_difficulty("loan_math") >= 1


def test_window_limits_history():
    controller = _controller(window=5)
    for _ in range(20):
        controller.record("loan_math", 1, passed=False)
    for _ in range(10):
        controller.record("loan_math", 1, passed=True)
    assert controller.pass_rate("loan_math") == 1.0


def test_easier_mix_reduces_difficulty_sometimes():
    controller = _controller(easier_mix=1.0, easier_mix_span=2)
    controller._difficulty["loan_math"] = 6
    seen_easier = False
    for _ in range(20):
        controller._difficulty["loan_math"] = 6
        level = controller.next_difficulty("loan_math")
        seen_easier = seen_easier or level < 6
    assert seen_easier


def test_state_snapshot():
    controller = _controller()
    controller.record("trend", 2, passed=True)
    state = controller.state()
    assert state["difficulty"] == {"trend": 2}
    assert state["pass_rate"]["trend"] == 1.0
