"""Reward modules: components and reward-hacking defenses."""

from ledgergym.rewards.components import RewardConfig, RewardWeights, compute_reward
from ledgergym.rewards.hacking_checks import (
    LeakError,
    assert_no_leak,
    hack_monitor,
    label_entropy,
)

__all__ = [
    "RewardConfig",
    "RewardWeights",
    "compute_reward",
    "LeakError",
    "assert_no_leak",
    "hack_monitor",
    "label_entropy",
]
