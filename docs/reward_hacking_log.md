# Reward-Hacking Log

Every known or plausible way the policy could earn reward without solving the task,
how it is detected, and its current status. This file is a living document (Section 12,
deliverable 4): append a dated entry whenever a new exploit is found, including the
episode seed that reproduced it and the patch that closed it.

**Status legend:** `defended` (mitigation in code + test), `monitored` (signal logged,
no hard block yet), `open` (known, not yet handled).

## Built-in defenses (present from day one)

| # | Potential hack | Detection | Defense | Status |
|---|---|---|---|---|
| H1 | Guess a plausible number without doing the work | pass rate vs. reasoning length; `length_drift` | randomized parameters + held-out seeds make guessing effectively impossible | defended |
| H2 | Emit valid format but a wrong/empty answer to farm the format weight | `format_farming_rate` monitor | format weight capped at 0.10; correctness weight 0.80; malformed output zeroes correctness | defended |
| H3 | Emit several `<answer>` blocks to cover options | `ParsedResponse.multiple_answers`; `multiple_answer_rate` | parser accepts exactly one block; extra blocks -> format invalid + correctness 0 | defended (`tests/test_env.py`) |
| H4 | Degenerate single-label answers on the trend task | `label_entropy` of outputs | generator samples the label uniformly *first*, then builds a matching series | defended (`tests/test_trend.py::test_label_distribution_is_balanced`) |
| H5 | Exploit verifier tolerance with near-miss numbers | boundary tests, adversarial near-misses | per-field absolute tolerances; `near_miss_values` proves the boundary rejects just-outside answers | defended (`tests/test_loan_math.py`) |
| H6 | Read the answer out of the prompt (leakage) | `assert_no_leak` on every `reset` | ground truth is generated privately and never interpolated into prompts; attempted leaks raise `LeakError` | defended (`tests/test_env.py`, `tests/test_reconciliation.py`) |
| H7 | Read ground truth from the public `Task` object | API surface review + test | `Task` exposes only prompt/family/variant/difficulty/seed/key-names; truth lives on the env (`current_truth`), never serialized | defended (`tests/test_env.py::test_public_task_exposes_no_ground_truth_fields`) |
| H8 | Verbosity drift ("think longer to get credit") | `length_drift` monitor | hard max length with penalty; length only ever subtracts reward | defended |
| H9 | Match a reconciliation answer by amount instead of reconciling IDs | generator design | salary credits repeat an identical amount, so the discrepancy amount never identifies the answer | defended (`tests/test_reconciliation.py::test_missing_deposit_amount_is_ambiguous_by_amount`) |
| H10 | Copy a near-duplicate decoy transaction | generator design | decoys share amounts but never share IDs; only an exact duplicated ID is correct | defended (`tests/test_reconciliation.py`) |

## Reward-mode-specific risks (ablation awareness)

| # | Potential hack | Context | Notes | Status |
|---|---|---|---|---|
| H11 | Gradient reward gives partial credit to far-off values | only in `gradient` mode | decay is exponential (`exp(-rel_err / scale)`); a wrong answer can never exceed `0.10 + 0.10` credit | monitored |
| H12 | Format + intermediate credit without correctness | ablation comparison | max 0.20, strictly below the 0.90 floor of a correct answer | defended (`tests/test_rewards.py::test_correctness_dominates_format_and_intermediate`) |

## Investigation protocol

1. Inspect samples manually after every run (the design doc requires this).
2. Watch the four monitors: `format_farming_rate`, `multiple_answer_rate`,
   `length_drift`, `label_entropy` (see `ledgergym.rewards.hacking_checks.hack_monitor`).
3. When a hack is suspected, freeze a held-out seed that reproduces it and add a
   regression test **before** patching.
4. Patch the verifier/generator, re-run the frozen seed, and log a dated row below.

## Findings

_No policy-level exploits observed yet (environment and verifiers only; training runs
not yet executed). Add dated rows here as runs produce evidence._
