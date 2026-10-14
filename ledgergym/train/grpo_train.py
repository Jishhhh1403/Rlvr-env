"""GRPO training for LedgerGym (Section 9 of the design doc).

Requires the training extras::

    pip install -e ".[train]"

The reward function wraps :class:`~ledgergym.env.base.LedgerEnv`: training rows
carry only ``(seed, family, difficulty, variant)`` and the prompt -- never the
ground truth -- so correctness is regenerated locally at reward time.

Example::

    python -m ledgergym.train.grpo_train --config ledgergym/train/configs/grpo_shaped.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from ledgergym.config import LedgerConfig, load_config
from ledgergym.env.base import LedgerEnv, episode_seed
from ledgergym.env.curriculum import CurriculumController
from ledgergym.rewards.components import RewardConfig

__all__ = ["load_train_config", "build_dataset", "build_reward_function", "main"]

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def load_train_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _completion_to_text(completion: Any) -> str:
    """Normalize a TRL completion (str, or list of chat messages) to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                parts.append(str(message.get("content", "")))
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return str(completion)


def build_reward_function(
    ledger_config: LedgerConfig,
    reward_config: RewardConfig,
    *,
    on_device_seed: int = 0,
):
    """Return a TRL-compatible reward function backed by a fresh LedgerEnv."""
    env = LedgerEnv(ledger_config, reward_config=reward_config)

    def reward_fn(completions: list[Any], **columns: Any) -> list[float]:
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            seed = _column(columns, "seed", index, on_device_seed)
            family = _column(columns, "family", index, "loan_math")
            difficulty = int(_column(columns, "difficulty", index, 1))
            variant = _column(columns, "variant", index, None)
            if variant == "":
                variant = None
            env.reset(
                seed=int(seed), family=str(family), difficulty=difficulty, variant=variant
            )
            result = env.step(_completion_to_text(completion))
            rewards.append(float(result.reward))
        return rewards

    return reward_fn


def _column(columns: dict[str, Any], name: str, index: int, default: Any) -> Any:
    values = columns.get(name)
    if values is None:
        return default
    try:
        return values[index]
    except (IndexError, TypeError):
        return default


def build_dataset(config: dict[str, Any], ledger_config: LedgerConfig):
    """Build a prompt dataset with per-row episode metadata and no ground truth."""
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("build_dataset requires the train extras: pip install -e '.[train]'") from exc

    size = int(config.get("dataset_size", 2048))
    base_seed = int(config.get("seed", 0))
    families = config.get("families", ["loan_math", "reconciliation", "trend"])
    use_curriculum = bool(config.get("curriculum", False))

    controller = CurriculumController(ledger_config.curriculum, seed=base_seed) if use_curriculum else None
    difficulty_plan = config.get("difficulty_plan", {})

    rows: list[dict[str, Any]] = []
    for index in range(size):
        family = families[index % len(families)]
        if controller is not None:
            difficulty = controller.next_difficulty(family)
        else:
            difficulty = int(
                difficulty_plan.get(family, difficulty_plan.get("default", 1))
            )
        seed = episode_seed(base_seed, index, family)
        task_family = LedgerEnv(ledger_config)._families[family]  # registry lookup
        prompt, _truth = task_family.generate(seed=seed, difficulty=difficulty)
        rows.append(
            {
                "prompt": prompt,
                "seed": seed,
                "family": family,
                "difficulty": difficulty,
                "variant": "",
            }
        )
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs a GPU stack
    parser = argparse.ArgumentParser(description="GRPO-train a small model on LedgerGym")
    parser.add_argument("--config", default=str(CONFIG_DIR / "grpo_shaped.yaml"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    train_config = load_train_config(args.config)
    ledger_config = load_config(train_config.get("ledger_config"))

    reward_section = ledger_config.reward
    reward_section.mode = train_config.get("reward_mode", reward_section.mode)
    reward_config = RewardConfig.from_section(reward_section)

    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "GRPO training requires the train extras: pip install -e '.[train]'"
        ) from exc

    model_name = train_config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    lora_cfg = train_config.get("lora", {})
    peft_config = None
    if lora_cfg.get("enabled", True):
        peft_config = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
        )

    dataset = build_dataset(train_config, ledger_config)
    grpo_cfg = train_config.get("grpo", {})
    output_dir = args.output_dir or train_config.get("output_dir", "runs/ledgergym-grpo")

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        num_generations=int(grpo_cfg.get("num_generations", 4)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 4)),
        max_prompt_length=int(grpo_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(grpo_cfg.get("max_completion_length", 1024)),
        max_steps=int(grpo_cfg.get("max_steps", 500)),
        beta=float(grpo_cfg.get("beta", 0.02)),
        temperature=float(grpo_cfg.get("temperature", 1.0)),
        logging_steps=int(grpo_cfg.get("logging_steps", 5)),
        save_steps=int(grpo_cfg.get("save_steps", 100)),
        report_to=train_config.get("report_to", "none"),
        seed=int(train_config.get("seed", 0)),
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=build_reward_function(ledger_config, reward_config),
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"saved policy to {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
