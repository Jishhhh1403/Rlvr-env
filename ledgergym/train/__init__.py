"""Training entry points.

Kept import-light so ``python -m ledgergym.train.grpo_train`` does not double-import
the runner module. Import the helpers from the module directly, or via ``__getattr__``.
"""

from __future__ import annotations

__all__ = ["build_dataset", "build_reward_function", "load_train_config"]


def __getattr__(name: str):
    if name in __all__:
        from ledgergym.train import grpo_train

        return getattr(grpo_train, name)
    raise AttributeError(name)
