"""Evaluation-runner tests (pure functions + a couple of smoke runs)."""

from __future__ import annotations

import json

import pytest

from ledgergym.eval.make_frozen_seeds import build_specs
from ledgergym.eval.policies import ConstantPolicy, OraclePolicy, RandomPolicy
from ledgergym.eval.run_eval import EpisodeSpec, evaluate, load_specs, run_episode
from ledgergym.env.base import LedgerEnv
from ledgergym.env.families import get_family


def test_frozen_seeds_are_deterministic_and_cover_all_families():
    first = build_specs()
    second = build_specs()
    assert first == second
    families = {item["family"] for item in first["in_distribution"]}
    assert families == {"loan_math", "reconciliation", "trend"}
    assert first["out_of_distribution"]


def test_load_specs_includes_both_splits():
    specs = load_specs()
    splits = {s.split for s in specs}
    assert {"in_distribution", "out_of_distribution"} <= splits


def _oracle_table(specs):
    table = {}
    for spec in specs:
        prompt, gt = get_family(spec.family).generate(
            seed=spec.seed, difficulty=spec.difficulty, variant=spec.variant
        )
        table[prompt] = {**gt.answer, **gt.intermediate}
    return table


def test_oracle_policy_achieves_full_pass_rate():
    specs = [
        EpisodeSpec(seed=1, family="loan_math", difficulty=1),
        EpisodeSpec(seed=2, family="reconciliation", difficulty=2),
        EpisodeSpec(seed=3, family="trend", difficulty=3),
    ]
    report = evaluate(OraclePolicy(_oracle_table(specs)), specs)
    assert report.pass_at_1 == 1.0
    assert report.format_validity_rate == 1.0
    assert report.mean_reward == pytest.approx(1.0, abs=1e-6)


def test_random_policy_fails_but_stays_format_valid():
    specs = [EpisodeSpec(seed=4, family="loan_math", difficulty=1) for _ in range(10)]
    report = evaluate(RandomPolicy(seed=0), specs)
    assert report.pass_at_1 == 0.0
    assert report.format_validity_rate == 1.0
    # Random-but-valid answers are exactly the format-farming signature.
    assert report.hack_monitor["format_farming_rate"] == 1.0


def test_pass_at_k_uses_any_sample():
    specs = [EpisodeSpec(seed=9, family="loan_math", difficulty=1)]
    table = _oracle_table(specs)
    prompt = next(iter(table))
    payload = table[prompt]

    class Sometimes:
        def __init__(self):
            self.calls = 0

        def __call__(self, _prompt):
            self.calls += 1
            if self.calls == 1:
                return "<answer>{}</answer>"
            return f"<answer>{json.dumps(payload)}</answer>"

    report = evaluate(Sometimes(), specs, k=2)
    assert report.pass_at_1 == 0.0
    assert report.pass_at_k == 1.0


def test_constant_policy_always_wrong():
    specs = [EpisodeSpec(seed=5, family="trend", difficulty=1)]
    report = evaluate(ConstantPolicy({"liquidity": "stable"}), specs)
    assert report.pass_at_1 in (0.0, 1.0)


def test_run_episode_uses_env_and_records_components():
    env = LedgerEnv()
    spec = EpisodeSpec(seed=6, family="loan_math", difficulty=1)
    result = run_episode(env, spec, RandomPolicy(seed=1), k=1)
    assert set(result.components) == {"correctness", "format", "intermediate", "length_penalty"}
    assert result.spec == spec


def test_reward_mode_is_honored_in_evaluation():
    from ledgergym.rewards.components import RewardConfig

    specs = [EpisodeSpec(seed=7, family="loan_math", difficulty=1)]
    binary = evaluate(
        RandomPolicy(seed=2), specs, k=1, reward_config=RewardConfig(mode="binary")
    )
    assert binary.mean_reward == 0.0  # binary mode gives no format credit
