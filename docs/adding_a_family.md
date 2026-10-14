# Adding a Task Family

A family is two things: a **generator** (returns a prompt + private ground truth from a
`(seed, difficulty)` pair) and a **verifier**. Verification is generic and tolerance-based,
so most families only write the generator. Target: ~50 lines.

## 1. Create the package

```
ledgergym/env/families/myfamily/
├── __init__.py     # MyFamily(TaskFamily) -- metadata + wiring
├── generator.py    # generate(seed, difficulty, variant) -> (prompt, GroundTruth)
└── verifier.py     # optional specialist verification (else use the default)
```

## 2. Write the generator

```python
# ledgergym/env/families/myfamily/generator.py
import random
from ledgergym.env.families.base import GroundTruth

FORMAT_HINT = "<reasoning>...</reasoning><answer>{\"key\": value}</answer>"

def generate(seed: int, difficulty: int, variant: str | None = None):
    rng = random.Random(seed)                 # deterministic: seed fully determines the task
    n = 5 + difficulty                        # difficulty knob
    value = rng.randrange(100, 100 * n)       # ...compute the *hidden* truth...
    prompt = (
        f"Some question with {n} items. "
        'Answer as {"total": <number>}. ' + FORMAT_HINT
    )
    gt = GroundTruth(
        family="myfamily",
        variant="total",
        difficulty=difficulty,
        seed=seed,
        answer={"total": value},                          # scored fields
        intermediate={"n_items": n},                       # optional sub-answers
        tolerances={"total": 0.01, "n_items": 0.0},        # absolute tolerance per field
    )
    return prompt, gt
```

## 3. Wire up the family

```python
# ledgergym/env/families/myfamily/__init__.py
from ledgergym.env.families.base import GroundTruth, TaskFamily
from ledgergym.env.families.myfamily import generator

class MyFamily(TaskFamily):
    name = "myfamily"
    variants = ("total",)
    default_variant = "total"
    numeric_fields = {"total", "n_items"}   # which keys compare numerically
    primary_numeric_field = "total"          # drives the gradient reward

    def generate(self, seed, difficulty, variant=None):
        return generator.generate(seed, self.clamp_difficulty(difficulty), variant)
```

## 4. Register it

```python
# ledgergym/env/families/__init__.py
from ledgergym.env.families.myfamily import MyFamily
register_family(MyFamily())
```

Add a `myfamily:` block to `configs/default.yaml` with its difficulty bounds.

## 5. Rules that keep the reward trustworthy

- **Never put ground truth in the prompt.** `reset()` runs `assert_no_leak` automatically.
  If your answer is a *selection* from items printed in the prompt (an ID, a label),
  override `leak_strings(gt) -> []` and rely on the prompt carrying no selection signal.
- **Malformed output must be wrong.** Return nothing special: the reward layer zeroes
  correctness whenever the strict parser rejects the output.
- **Tolerances are absolute and explicit.** Put every non-exact field in `tolerances`.
- **Determinism.** Use only `random.Random(seed)`; no `random`, `time`, or set iteration
  order in outputs (sort anything you iterate).

## 6. Test it

```python
# tests/test_myfamily.py
from ledgergym.parsing import parse_response
from ledgergym.env.families.myfamily import MyFamily

def test_minimal():
    fam = MyFamily()
    prompt, gt = fam.generate(seed=1, difficulty=3)
    payload = {**gt.answer, **gt.intermediate}
    import json
    assert fam.verify(gt, parse_response(f"<answer>{json.dumps(payload)}</answer>")).final_correct
```

Run `pytest`. The shared env tests in `tests/test_env.py` already cover determinism,
leakage, and the Gym interface for every registered family.
