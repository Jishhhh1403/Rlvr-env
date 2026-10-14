# Reproducible LedgerGym environment (generators, verifiers, tests, eval).
# Use as the base for CPU-only work; add the [train] extra for GRPO on a GPU host.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ledgergym ./ledgergym
COPY configs ./configs
COPY tests ./tests

RUN pip install --upgrade pip && pip install -e ".[dev]"

# Freeze the held-out evaluation seeds (deterministic) and run the suite.
RUN python -m ledgergym.eval.make_frozen_seeds && pytest

CMD ["python", "-m", "ledgergym.eval.run_eval", "--policy", "random"]
