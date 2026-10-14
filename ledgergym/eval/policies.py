"""Policies used for evaluation and baselines.

A *policy* is simply ``Callable[[str], str]`` mapping a prompt to a raw response.
Policies here are intentionally simple so an untrained baseline can be measured
without a GPU; :class:`ModelPolicy` is the real (lazy, optional) one.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

__all__ = ["Policy", "RandomPolicy", "ConstantPolicy", "OraclePolicy", "ModelPolicy"]

Policy = Callable[[str], str]


class RandomPolicy:
    """Baseline: emit a valid-format answer with random plausible values."""

    def __init__(self, seed: int = 0, answer: str = "0") -> None:
        self.rng = random.Random(seed)
        self.answer = answer

    def __call__(self, prompt: str) -> str:
        value = round(self.rng.uniform(100.0, 5_000_000.0), 2)
        keys = _guess_keys(prompt)
        payload = {k: (value if i == 0 else "improving") for i, k in enumerate(keys)}
        return (
            "<reasoning>A quick estimate.</reasoning>"
            f"<answer>{json.dumps(payload)}</answer>"
        )


class ConstantPolicy:
    """Baseline: always emit the same answer payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __call__(self, prompt: str) -> str:
        return f"<reasoning>constant</reasoning><answer>{json.dumps(self.payload)}</answer>"


class OraclePolicy:
    """Upper-bound reference: returns the ground truth for a known prompt.

    Built from a prompt -> answer table produced by regenerating tasks locally.
    Used in tests and as an ideal-policy sanity check.
    """

    def __init__(self, table: dict[str, dict[str, Any]]) -> None:
        self.table = table

    def __call__(self, prompt: str) -> str:
        payload = self.table.get(prompt)
        if payload is None:
            return "<answer>{}</answer>"
        return f"<reasoning>oracle</reasoning><answer>{json.dumps(payload)}</answer>"


def _guess_keys(prompt: str) -> list[str]:
    """Best-effort extraction of the JSON keys named in a prompt (baseline only)."""
    import re

    keys = re.findall(r'"([a-z_]+)"\s*:', prompt)
    seen: list[str] = []
    for key in keys:
        if key not in seen:
            seen.append(key)
    return seen or ["answer"]


class ModelPolicy:
    """Wrap a Hugging Face causal LM as a policy (requires the train extras).

    The model is treated as a chat model if a chat template exists; generation is
    sampled so pass@k is meaningful.
    """

    def __init__(
        self,
        model_name: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ModelPolicy requires the train extras: pip install -e '.[train]'"
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device or ("auto" if torch.cuda.is_available() else None),
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, prompt: str) -> str:
        torch = self.torch
        messages = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
