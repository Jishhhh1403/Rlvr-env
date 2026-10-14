"""YAML configuration loading for LedgerGym."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

__all__ = ["RewardSection", "CurriculumSection", "FamilySection", "LedgerConfig", "load_config"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class RewardWeights(BaseModel):
    correctness: float = 0.80
    format: float = 0.10
    intermediate: float = 0.10


class RewardSection(BaseModel):
    mode: str = "shaped"
    weights: RewardWeights = RewardWeights()
    length_penalty: float = 0.10
    min_length_chars: int = 20
    max_length_chars: int = 4000
    gradient_scale: float = 0.05


class CurriculumSection(BaseModel):
    window: int = 50
    up_threshold: float = 0.70
    down_threshold: float = 0.30
    easier_mix: float = 0.15
    easier_mix_span: int = 2
    min_difficulty: int = 1
    max_difficulty: int = 10


class FamilySection(BaseModel):
    min_difficulty: int = 1
    max_difficulty: int = 10


class LedgerConfig(BaseModel):
    reward: RewardSection = RewardSection()
    curriculum: CurriculumSection = CurriculumSection()
    families: dict[str, FamilySection] = Field(default_factory=dict)


def load_config(path: str | Path | None = None) -> LedgerConfig:
    """Load a :class:`LedgerConfig` from YAML; defaults to ``configs/default.yaml``."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return LedgerConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return LedgerConfig.model_validate(raw)
