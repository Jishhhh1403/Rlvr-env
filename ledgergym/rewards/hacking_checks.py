"""Reward-hacking defenses and monitors (Section 5.3 of the design doc).

These are used two ways:

* **In training/eval loops** as runtime monitors (length drift, label entropy,
  format-farming rate, multiple-answer rate).
* **In tests** as adversarial checks (ground-truth leakage, near-miss answers
  that must not be accepted, extra answer blocks).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = [
    "LeakError",
    "assert_no_leak",
    "label_entropy",
    "count_answer_blocks",
    "length_drift",
    "format_farming_rate",
    "near_miss_values",
    "hack_monitor",
]


class LeakError(AssertionError):
    """Raised when ground-truth-derived text appears in a prompt."""


def assert_no_leak(prompt: str, forbidden: Iterable[str]) -> None:
    """Fail loudly if any forbidden ground-truth string appears in ``prompt``."""
    hits = [s for s in forbidden if s and s in prompt]
    if hits:
        raise LeakError(f"ground truth leaked into prompt: {hits!r}")


def count_answer_blocks(text: str) -> int:
    """Count ``<answer>`` blocks in a raw response (defense: emit only one)."""
    import re

    return len(re.findall(r"<answer>", text, flags=re.IGNORECASE))


def label_entropy(labels: Sequence[str]) -> float:
    """Shannon entropy (nats) of the label distribution; low entropy = degeneracy."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def length_drift(lengths: Sequence[int]) -> dict[str, float]:
    """Summary of output-length distribution for verbosity/hack-drift monitoring."""
    if not lengths:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    ordered = sorted(lengths)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(ordered[idx])

    return {
        "mean": sum(lengths) / n,
        "p10": pct(0.10),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "max": float(ordered[-1]),
    }


def format_farming_rate(infos: Sequence[dict[str, Any]]) -> float:
    """Fraction of episodes that earned format credit but answered incorrectly.

    A high rate is the signature of format farming.
    """
    if not infos:
        return 0.0
    farming = sum(
        1
        for info in infos
        if info.get("format_valid") and not info.get("passed")
    )
    return farming / len(infos)


def near_miss_values(truth: float, tol: float, *, inside: bool = True) -> list[float]:
    """Adversarial values just inside/outside a verifier tolerance.

    Used by tests to prove the tolerance boundary behaves as intended.
    """
    delta = tol * 0.5 if inside else tol * 1.5
    return [truth - delta, truth + delta]


def hack_monitor(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate hack-monitoring signals over a batch of episode infos."""
    lengths = [int(i.get("char_len", 0)) for i in infos]
    multiple = sum(1 for i in infos if i.get("multiple_answers"))
    return {
        "episodes": len(infos),
        "format_validity_rate": (
            sum(1 for i in infos if i.get("format_valid")) / len(infos) if infos else 0.0
        ),
        "format_farming_rate": format_farming_rate(infos),
        "multiple_answer_rate": (multiple / len(infos)) if infos else 0.0,
        "length": length_drift(lengths),
    }
