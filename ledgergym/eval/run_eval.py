"""Evaluation runner: pass rate, reward components, pass@k, and hack monitors."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ledgergym.config import load_config
from ledgergym.env.base import LedgerEnv
from ledgergym.rewards.components import RewardConfig
from ledgergym.rewards.hacking_checks import hack_monitor, label_entropy

__all__ = ["EpisodeSpec", "EpisodeResult", "EvalReport", "run_episode", "evaluate", "load_specs"]

FROZEN_SEEDS_PATH = Path(__file__).resolve().parent / "frozen_seeds" / "frozen_seeds.json"


class EpisodeSpec(BaseModel):
    seed: int
    family: str
    difficulty: int
    variant: str | None = None
    split: str = "in_distribution"


class EpisodeResult(BaseModel):
    spec: EpisodeSpec
    reward: float
    passed: bool
    pass_at_1: bool = False
    format_valid: bool
    components: dict[str, float]
    char_len: int
    label: str | None = None
    info: dict[str, Any] = Field(default_factory=dict)


class EvalReport(BaseModel):
    n_episodes: int
    k: int
    pass_at_1: float
    pass_at_k: float
    mean_reward: float
    mean_components: dict[str, float]
    format_validity_rate: float
    per_family: dict[str, dict[str, float]]
    per_family_difficulty: dict[str, dict[str, dict[str, float]]]
    label_entropy: float
    length: dict[str, float]
    hack_monitor: dict[str, Any]
    result_file: str | None = None

    def to_markdown(self) -> str:
        lines = [
            "## LedgerGym evaluation",
            "",
            f"- episodes: {self.n_episodes} (k = {self.k})",
            f"- pass@1: {self.pass_at_1:.3f}   pass@{self.k}: {self.pass_at_k:.3f}",
            f"- mean reward: {self.mean_reward:.3f}",
            f"- format validity: {self.format_validity_rate:.3f}",
            f"- label entropy: {self.label_entropy:.3f}",
            "- components: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted(self.mean_components.items())),
            "",
            "| family | difficulty | pass rate | mean reward | format |",
            "|---|---|---|---|---|",
        ]
        for fam, by_diff in sorted(self.per_family_difficulty.items()):
            for diff, stats in sorted(by_diff.items(), key=lambda kv: int(kv[0])):
                lines.append(
                    f"| {fam} | {diff} | {stats['pass_rate']:.3f} | "
                    f"{stats['mean_reward']:.3f} | {stats['format_validity']:.3f} |"
                )
        return "\n".join(lines) + "\n"


def load_specs(path: str | Path | None = None) -> list[EpisodeSpec]:
    """Load frozen evaluation seeds (defaults to the bundled file)."""
    spec_path = Path(path) if path is not None else FROZEN_SEEDS_PATH
    with spec_path.open("r", encoding="utf-8") as fh:
        blob = json.load(fh)
    specs = [EpisodeSpec.model_validate(item) for item in blob["in_distribution"]]
    specs += [EpisodeSpec.model_validate(item) for item in blob.get("out_of_distribution", [])]
    return specs


def run_episode(
    env: LedgerEnv, spec: EpisodeSpec, policy, k: int = 1
) -> EpisodeResult:
    """Run one episode, sampling the policy ``k`` times; pass if any sample passes."""
    env.reset(seed=spec.seed, family=spec.family, difficulty=spec.difficulty, variant=spec.variant)
    task = env.current_task
    assert task is not None

    best = None
    any_pass = False
    first_pass = False
    for sample_index in range(max(1, k)):
        result = env.step(policy(task.prompt))
        if best is None or result.reward > best.reward:
            best = result
        if sample_index == 0:
            first_pass = bool(result.info["passed"])
        any_pass = any_pass or result.info["passed"]
    assert best is not None

    label = None
    truth = env.current_truth
    if truth is not None:
        raw_label = truth.answer.get("liquidity")
        label = str(raw_label) if raw_label is not None else None

    return EpisodeResult(
        spec=spec,
        reward=best.reward,
        passed=any_pass,
        pass_at_1=first_pass,
        format_valid=best.info["format_valid"],
        components=best.components,
        char_len=best.info["char_len"],
        label=label,
        info=best.info,
    )


def _per_group(results: list[EpisodeResult], key_fn) -> dict[str, dict[str, float]]:
    groups: dict[str, list[EpisodeResult]] = defaultdict(list)
    for r in results:
        groups[key_fn(r)].append(r)
    out: dict[str, dict[str, float]] = {}
    for key, items in groups.items():
        out[key] = {
            "pass_rate": sum(1 for i in items if i.passed) / len(items),
            "mean_reward": statistics.fmean(i.reward for i in items),
            "format_validity": sum(1 for i in items if i.format_valid) / len(items),
        }
    return out


def evaluate(
    policy,
    specs: list[EpisodeSpec] | None = None,
    *,
    env: LedgerEnv | None = None,
    k: int = 1,
    reward_config: RewardConfig | None = None,
) -> EvalReport:
    """Evaluate ``policy`` on ``specs`` and return an aggregated :class:`EvalReport`."""
    specs = specs if specs is not None else load_specs()
    if env is None:
        config = load_config()
        env = LedgerEnv(config, reward_config=reward_config)

    results = [run_episode(env, spec, policy, k=k) for spec in specs]

    labels = [r.label for r in results if r.label is not None]
    monkey = hack_monitor([r.info for r in results])
    per_family_difficulty = _per_group(
        results, lambda r: f"{r.spec.family}\u0000{r.spec.difficulty}"
    )
    nested: dict[str, dict[str, float]] = defaultdict(dict)
    for key, stats in per_family_difficulty.items():
        fam, diff = key.split("\u0000")
        nested[fam][diff] = stats

    component_keys = sorted({k2 for r in results for k2 in r.components})
    mean_components = {
        key: statistics.fmean(r.components.get(key, 0.0) for r in results) if results else 0.0
        for key in component_keys
    }

    return EvalReport(
        n_episodes=len(results),
        k=k,
        pass_at_1=sum(1 for r in results if r.pass_at_1) / len(results) if results else 0.0,
        pass_at_k=sum(1 for r in results if r.passed) / len(results) if results else 0.0,
        mean_reward=statistics.fmean(r.reward for r in results) if results else 0.0,
        mean_components=mean_components,
        format_validity_rate=(
            sum(1 for r in results if r.format_valid) / len(results) if results else 0.0
        ),
        per_family=_per_group(results, lambda r: r.spec.family),
        per_family_difficulty={k2: dict(v) for k2, v in nested.items()},
        label_entropy=label_entropy(labels),
        length=monkey["length"],
        hack_monitor=monkey,
    )


def _build_policy(args):
    from ledgergym.eval.policies import ConstantPolicy, ModelPolicy, RandomPolicy

    if args.policy == "random":
        return RandomPolicy(seed=args.policy_seed)
    if args.policy == "constant":
        return ConstantPolicy({"emi": 12345.67})
    if args.policy == "model":
        return ModelPolicy(args.model_name)
    raise ValueError(f"unknown policy {args.policy!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a policy on frozen LedgerGym seeds")
    parser.add_argument("--policy", choices=["random", "constant", "model"], default="random")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--seeds", default=None, help="path to frozen seeds JSON")
    parser.add_argument("--k", type=int, default=1, help="samples per prompt for pass@k")
    parser.add_argument("--reward-mode", choices=["binary", "shaped", "gradient"], default="shaped")
    parser.add_argument("--out", default=None, help="where to write the JSON report")
    args = parser.parse_args(argv)

    config = load_config()
    config.reward.mode = args.reward_mode
    reward_config = RewardConfig.from_section(config.reward)

    policy = _build_policy(args)
    specs = load_specs(args.seeds)
    report = evaluate(policy, specs, reward_config=reward_config, k=args.k)

    print(report.to_markdown())
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
