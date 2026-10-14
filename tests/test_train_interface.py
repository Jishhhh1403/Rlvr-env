"""Config loading and training-interface tests (no GPU/TRL required)."""

from __future__ import annotations

import json
from pathlib import Path

from ledgergym import RewardConfig
from ledgergym.config import load_config
from ledgergym.env.families import get_family
from ledgergym.train.grpo_train import build_reward_function, load_train_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "ledgergym" / "train" / "configs"


def test_default_yaml_loads_with_expected_weights():
    config = load_config()
    assert config.reward.mode == "shaped"
    total = (
        config.reward.weights.correctness
        + config.reward.weights.format
        + config.reward.weights.intermediate
    )
    assert total == 1.0
    assert config.reward.weights.correctness > config.reward.weights.format
    assert config.curriculum.up_threshold > config.curriculum.down_threshold
    assert set(config.families) >= {"loan_math", "reconciliation", "trend"}


def test_train_configs_exist_and_parse():
    for name in ("grpo_shaped.yaml", "grpo_binary.yaml"):
        cfg = load_train_config(CONFIG_DIR / name)
        assert "model_name" in cfg and "grpo" in cfg
    assert load_train_config(CONFIG_DIR / "grpo_binary.yaml")["reward_mode"] == "binary"
    assert load_train_config(CONFIG_DIR / "grpo_shaped.yaml")["reward_mode"] == "shaped"


def test_reward_function_scores_a_correct_completion():
    config = load_config()
    reward_fn = build_reward_function(config, RewardConfig.from_section(config.reward))

    prompt, gt = get_family("loan_math").generate(seed=77, difficulty=2)
    completion = "<reasoning>...</reasoning>" + "<answer>" + json.dumps(gt.answer) + "</answer>"

    rewards = reward_fn(
        [completion],
        seed=[77],
        family=["loan_math"],
        difficulty=[2],
        variant=[""],
    )
    # Only the final answer is supplied, so intermediate credit is 0:
    # correctness (0.80) + format (0.10) = 0.90.
    assert rewards == [0.9]

    wrong = reward_fn(
        ['<answer>{"emi": 1}</answer>'],
        seed=[77],
        family=["loan_math"],
        difficulty=[2],
        variant=[""],
    )
    assert wrong[0] < 1.0


def test_reward_function_handles_chat_style_completions():
    config = load_config()
    reward_fn = build_reward_function(config, RewardConfig.from_section(config.reward))
    prompt, gt = get_family("trend").generate(seed=88, difficulty=3)
    chat = [
        {"role": "assistant", "content": "<answer>" + json.dumps(gt.answer) + "</answer>"}
    ]
    rewards = reward_fn([chat], seed=[88], family=["trend"], difficulty=[3], variant=[""])
    assert rewards == [0.9]
