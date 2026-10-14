"""Plot training/eval curves from LedgerGym reports.

Usage::

    python notebooks/reward_curves.py --report eval_report.json --out curves.png

Reads a JSON report produced by ``ledgergym.eval.run_eval`` (or a W&B/MLflow export
with the same shape) and renders pass rate per family plus the reward-component mix.
Matplotlib is optional: the script prints a summary table when it is unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize(report: dict) -> str:
    lines = [
        f"episodes: {report.get('n_episodes')}",
        f"pass@1: {report.get('pass_at_1', 0):.3f}",
        f"mean reward: {report.get('mean_reward', 0):.3f}",
        "components:",
    ]
    for name, value in sorted(report.get("mean_components", {}).items()):
        lines.append(f"  {name:14s} {value:.3f}")
    lines.append("per family:")
    for family, stats in sorted(report.get("per_family", {}).items()):
        lines.append(f"  {family:15s} pass={stats['pass_rate']:.3f} reward={stats['mean_reward']:.3f}")
    return "\n".join(lines)


def plot(report: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; showing summary instead\n")
        print(summarize(report))
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for family, by_diff in sorted(report.get("per_family_difficulty", {}).items()):
        diffs = sorted(by_diff, key=int)
        ax1.plot(diffs, [by_diff[d]["pass_rate"] for d in diffs], marker="o", label=family)
    ax1.set_title("Pass rate by difficulty")
    ax1.set_xlabel("difficulty")
    ax1.set_ylabel("pass rate")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(alpha=0.3)

    components = report.get("mean_components", {})
    ax2.bar(list(components), list(components.values()))
    ax2.set_title("Mean reward components")
    ax2.set_ylabel("value")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}\n")
    print(summarize(report))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", default="curves.png")
    args = parser.parse_args()
    plot(load_report(Path(args.report)), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
