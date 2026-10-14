"""Reward components and aggregation (Section 5 of the design doc).

Reward modes:

* ``binary``   -- 1.0 if the answer is correct *and* well-formed, else 0.0
* ``shaped``   -- correctness + format + intermediate credit, minus length penalty
* ``gradient`` -- shaped, but the correctness term gives partial credit decaying
                  with relative error on the primary numeric field

Rule: correctness dominates. Format + intermediate credit alone
(0.20 max) can never beat a correct answer, and malformed output zeroes
correctness so format cannot be farmed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ledgergym.parsing import ParsedResponse
from ledgergym.env.families.base import VerifyResult

__all__ = ["RewardWeights", "RewardConfig", "compute_reward"]

VALID_MODES = ("binary", "shaped", "gradient")


@dataclass
class RewardWeights:
    correctness: float = 0.80
    format: float = 0.10
    intermediate: float = 0.10

    def total(self) -> float:
        return self.correctness + self.format + self.intermediate


@dataclass
class RewardConfig:
    mode: str = "shaped"
    weights: RewardWeights = field(default_factory=RewardWeights)
    length_penalty: float = 0.10
    min_length_chars: int = 20
    max_length_chars: int = 4000
    gradient_scale: float = 0.05
    clamp_min: float = -0.10
    clamp_max: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")

    @classmethod
    def from_section(cls, section) -> RewardConfig:
        """Build from a :class:`ledgergym.config.RewardSection`."""
        weights = RewardWeights(
            correctness=section.weights.correctness,
            format=section.weights.format,
            intermediate=section.weights.intermediate,
        )
        return cls(
            mode=section.mode,
            weights=weights,
            length_penalty=section.length_penalty,
            min_length_chars=section.min_length_chars,
            max_length_chars=section.max_length_chars,
            gradient_scale=section.gradient_scale,
        )


def _length_component(parsed: ParsedResponse, cfg: RewardConfig) -> float:
    """Return a non-positive length contribution (0 = no penalty)."""
    if parsed.char_len < cfg.min_length_chars:
        return -cfg.length_penalty
    if parsed.char_len > cfg.max_length_chars:
        return -cfg.length_penalty
    return 0.0


def compute_reward(
    parsed: ParsedResponse, verify: VerifyResult, cfg: RewardConfig
) -> tuple[float, dict[str, float]]:
    """Aggregate verification output into a scalar reward plus a component breakdown."""
    w = cfg.weights
    format_valid = 1.0 if parsed.format_valid else 0.0
    length = _length_component(parsed, cfg)

    # Malformed output zeroes correctness and intermediate credit: format cannot
    # be farmed, and a wrong-format answer can never score as a correct one.
    if not parsed.format_valid:
        correctness = 0.0
        intermediate = 0.0
    else:
        correctness = 1.0 if verify.final_correct else 0.0
        intermediate = float(verify.intermediate_score)

        if cfg.mode == "gradient" and not verify.final_correct and verify.rel_error is not None:
            correctness = math.exp(-verify.rel_error / max(cfg.gradient_scale, 1e-9))

    components = {
        "correctness": correctness,
        "format": format_valid,
        "intermediate": intermediate,
        "length_penalty": length,
    }

    if cfg.mode == "binary":
        reward = 1.0 if (verify.final_correct and parsed.format_valid) else 0.0
        components = {"correctness": reward, "format": format_valid, "intermediate": 0.0, "length_penalty": 0.0}
        return reward, components

    reward = (
        w.correctness * components["correctness"]
        + w.format * components["format"]
        + w.intermediate * components["intermediate"]
        + components["length_penalty"]
    )
    reward = max(cfg.clamp_min, min(cfg.clamp_max, reward))
    return reward, components
