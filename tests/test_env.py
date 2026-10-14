"""Environment-level tests: determinism, ground-truth secrecy, and the Gym API."""

from __future__ import annotations

import json

import pytest

from ledgergym import LedgerEnv
from ledgergym.config import load_config
from ledgergym.env.base import episode_seed
from ledgergym.env.curriculum import CurriculumController
from ledgergym.rewards.hacking_checks import LeakError

FAMILIES = ["loan_math", "reconciliation", "trend"]


def _perfect_response(gt):
    payload = {}
    for key, value in {**gt.answer, **gt.intermediate}.items():
        payload[key] = value
    return f"<reasoning>worked it out</reasoning><answer>{json.dumps(payload)}</answer>"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("difficulty", range(1, 11))
def test_perfect_response_gets_full_reward(family, difficulty):
    env = LedgerEnv()
    env.reset(seed=family.__len__() + difficulty * 13, family=family, difficulty=difficulty)
    result = env.step(_perfect_response(env.current_truth))
    assert result.info["passed"]
    assert result.reward == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_determinism_same_seed_same_prompt_and_reward(family):
    env = LedgerEnv()
    task_a = env.reset(seed=1234, family=family, difficulty=4)
    first = env.step("<answer>{}</answer>")
    task_b = env.reset(seed=1234, family=family, difficulty=4)
    second = env.step("<answer>{}</answer>")
    assert task_a.prompt == task_b.prompt
    assert first.reward == second.reward


@pytest.mark.parametrize("family", FAMILIES)
def test_ground_truth_never_leaks_into_prompts(family):
    env = LedgerEnv()
    family_obj = env._families[family]
    for seed in range(200):
        difficulty = (seed % 10) + 1
        env.reset(seed=seed, family=family, difficulty=difficulty)
        gt = env.current_truth
        leaked = [s for s in family_obj.leak_strings(gt) if s in task_prompt(env)]
        assert not leaked, (family, seed, leaked)
    assert task_prompt(env)


def task_prompt(env: LedgerEnv) -> str:
    assert env.current_task is not None
    return env.current_task.prompt


@pytest.mark.parametrize("family", FAMILIES)
def test_public_task_exposes_no_ground_truth_fields(family):
    env = LedgerEnv()
    task = env.reset(seed=5, family=family, difficulty=2)
    dumped = task.model_dump()
    assert set(dumped) == {
        "prompt",
        "family",
        "variant",
        "difficulty",
        "seed",
        "expected_keys",
        "intermediate_keys",
    }
    # expected_keys are field *names* only, never values
    assert all(isinstance(k, str) for k in dumped["expected_keys"])


def test_multiple_answer_blocks_are_penalized():
    env = LedgerEnv()
    env.reset(seed=1, family="loan_math", difficulty=1)
    result = env.step(
        '<answer>{"emi": 1}</answer> <answer>{"emi": 2}</answer>'
    )
    assert result.info["multiple_answers"]
    assert not result.info["format_valid"]
    assert result.components["correctness"] == 0.0


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        LedgerEnv().step("<answer>{}</answer>")


def test_unknown_family_raises():
    with pytest.raises(ValueError):
        LedgerEnv().reset(seed=0, family="does_not_exist", difficulty=1)


def test_leak_guard_is_wired_into_reset(monkeypatch):
    env = LedgerEnv()
    family_obj = env._families["loan_math"]
    # Plant a "ground truth" string that is guaranteed to appear in the prompt
    # ("EMI") and prove reset() refuses to serve the task.
    monkeypatch.setattr(type(family_obj), "leak_strings", lambda self, gt: ["EMI"])
    with pytest.raises(LeakError):
        env.reset(seed=0, family="loan_math", difficulty=1)


def test_curriculum_driven_reset():
    controller = CurriculumController(load_config().curriculum, seed=1)
    env = LedgerEnv(curriculum=controller)
    for i in range(20):
        env.reset(seed=i, family="loan_math")
        assert env.current_task is not None


def test_episode_seed_is_stable_and_family_specific():
    assert episode_seed(0, 3, "loan_math") == episode_seed(0, 3, "loan_math")
    assert episode_seed(0, 3, "loan_math") != episode_seed(0, 3, "trend")
    assert episode_seed(0, 3, "trend") != episode_seed(0, 4, "trend")
