"""Generate the frozen held-out evaluation seed sets.

Two splits are produced (Section 8.1 of the design doc):

* ``in_distribution``  -- the natural variant for each difficulty level.
* ``out_of_distribution`` -- unseen format/size combinations, e.g. a variant at a
  difficulty where it is normally never sampled, to test generalization.

The files are committed so evaluation is reproducible across runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from ledgergym.env.families import available_families, get_family

__all__ = ["build_specs", "write_frozen_seeds"]

DEFAULT_OUT = Path(__file__).resolve().parent / "frozen_seeds" / "frozen_seeds.json"
FROZEN_BASE_SEED = 20_261_014
PER_FAMILY_DIFFICULTY = 5  # episodes per (family, difficulty) in the in-distribution set
PER_OOD_EPISODE = 8


def _stable_offset(name: str, modulus: int) -> int:
    """Stable across processes (unlike the salted builtin ``hash``)."""
    digest = hashlib.sha256(name.encode()).digest()
    return int.from_bytes(digest[:4], "big") % modulus


def build_specs() -> dict:
    in_distribution = []
    out_of_distribution = []

    for family_name in available_families():
        family = get_family(family_name)
        for difficulty in range(family.min_difficulty, family.max_difficulty + 1):
            for i in range(PER_FAMILY_DIFFICULTY):
                seed = FROZEN_BASE_SEED + difficulty * 1_000 + i * 7 + _stable_offset(family_name, 97)
                in_distribution.append(
                    {
                        "seed": abs(seed),
                        "family": family_name,
                        "difficulty": difficulty,
                        "variant": None,
                        "split": "in_distribution",
                    }
                )

    # Out-of-distribution: force a variant that is off the natural difficulty
    # schedule, exercising unseen prompt shapes and sizes.
    ood_variants = {
        "loan_math": ("rate_comparison", 5),
        "reconciliation": ("balance_mismatch", 2),
        "trend": ("liquidity", 10),
    }
    for family_name, (variant, difficulty) in ood_variants.items():
        for i in range(PER_OOD_EPISODE):
            seed = FROZEN_BASE_SEED + 500_000 + i * 13 + _stable_offset(family_name, 89)
            out_of_distribution.append(
                {
                    "seed": abs(seed),
                    "family": family_name,
                    "difficulty": difficulty,
                    "variant": variant,
                    "split": "out_of_distribution",
                }
            )

    return {
        "version": 1,
        "created": date.today().isoformat(),
        "base_seed": FROZEN_BASE_SEED,
        "in_distribution": in_distribution,
        "out_of_distribution": out_of_distribution,
    }


def write_frozen_seeds(path: str | Path | None = None) -> Path:
    out = Path(path) if path is not None else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_specs(), indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze held-out evaluation seeds")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    path = write_frozen_seeds(args.out)
    blob = json.loads(path.read_text(encoding="utf-8"))
    print(
        f"wrote {path} "
        f"({len(blob['in_distribution'])} in-distribution, "
        f"{len(blob['out_of_distribution'])} out-of-distribution)"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
