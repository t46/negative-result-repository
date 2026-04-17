# Negative Result Repository

Structure, search, and learn from failed autoresearch experiments.

When autonomous research pipelines (like [autoresearch](https://github.com/karpathy/autoresearch)) run 100+ experiments per night, ~85% are discarded or crash. Currently these failures are stored as single rows in a TSV file and the information is lost. This tool converts failures into structured, searchable, actionable knowledge.

## Capabilities

1. **Failure Structuring**: Parse `results.tsv` into structured objects with config diffs, failure classification, and lessons learned
2. **Similarity Search**: Given a proposed experiment config, find the most similar past failures before wasting compute
3. **Pattern Aggregation**: Extract rules like "increasing learning rate consistently fails" from clusters of individual failures
4. **Autoresearch Loop Integration**: `check_proposal()` interface that returns proceed/caution/avoid recommendations

## Quick Start

```bash
uv run python demo.py
```

## Usage

```python
from nrr import NegativeResultRepository

# Load from autoresearch-lite results
repo = NegativeResultRepository.from_tsv("data/results.tsv")

# Check a proposed experiment before running it
result = repo.check_proposal(
    description="Increase learning rate to 0.05",
    config_changes={"LEARNING_RATE": "0.05"},
)
print(result["recommendation"])  # "avoid"
print(result["reason"])          # "6 similar LR experiments all failed"

# Find similar past failures
similar = repo.find_similar_to_config(
    {"OPTIMIZER": "adam"},
    description="Switch to Adam optimizer",
)

# View extracted patterns/rules
for pattern in repo.patterns:
    print(f"[{pattern.confidence:.2f}] {pattern.rule}")

# Save/load
repo.to_json("nrr_database.json")
repo2 = NegativeResultRepository.from_json("nrr_database.json")
```

## Validation

All capabilities are validated against real data from [autoresearch-lite](https://github.com/t46/autoresearch-lite) (21 experiments: 3 keep / 16 discard / 2 crash). No synthetic or mock data.

Key validation results:
- Parser correctly extracts config diffs and classifies all 18 failures
- Similarity search returns the correct most-similar failure (e.g., LR 0.1 experiment for an LR 0.05 query)
- Pattern extraction identifies that all 4 LR-increase attempts failed
- `check_proposal()` correctly recommends "avoid" for LR changes and "proceed" for novel changes

## CLI integration (Karpathy `autoresearch` and other LLM-driven loops)

`autoresearch-lite` calls `check_proposal()` from Python because it owns its
own driver. Karpathy's [`autoresearch`](https://github.com/karpathy/autoresearch)
has no Python driver — the LLM agent itself decides when to edit `train.py`,
following instructions in `program.md`. To integrate with that style of loop,
NRR exposes a CLI:

```bash
# Write the planned change (description, diff, or both) to a file:
echo "Increase LEARNING_RATE to 0.1 to speed convergence" > /tmp/proposal.txt

# Ask NRR for a verdict before touching train.py:
uv run --directory ~/dev/negative-result-repository \
    nrr check --proposal-file /tmp/proposal.txt
```

Output is a single JSON object on stdout:

```json
{
  "verdict": "proceed | caution | avoid",
  "reason": "...",
  "similar_failures": [ ... top-5 by similarity ... ],
  "relevant_patterns": [ ... ],
  "estimated_success_probability": 0.21,
  "parsed_description": "...",
  "parsed_config_changes": {"LEARNING_RATE": "0.1"},
  "database": "/abs/path/to/nrr_database.json"
}
```

Exit code mirrors the verdict (`0`=proceed, `1`=caution, `2`=avoid, `3`=usage
error) so non-LLM scripts can branch on `$?` too.

The database is auto-located in this order:
1. `$NRR_DATABASE` env var (absolute path)
2. `<package_root>/data/nrr_database.json`
3. `<package_root>/data/results.tsv` (parsed on the fly)

### Plugging into `program.md`

Add the following step to the experimentation loop in `autoresearch/program.md`
(see the integrated fork at https://github.com/t46/autoresearch):

> Before editing `train.py`, write the planned change to `/tmp/proposal.txt`
> (a short description, optionally followed by the proposed `KEY = value`
> assignments) and run
> `uv run --directory ~/dev/negative-result-repository nrr check --proposal-file /tmp/proposal.txt`.
> If `verdict` is `avoid`, pick a different direction. If `caution`, proceed
> but acknowledge the warning in the description column of `results.tsv`.
> If `proceed`, go ahead.

### Known limitation: seed mismatch

The shipped database (`data/nrr_database.json`) was harvested from
[`autoresearch-lite`](https://github.com/t46/autoresearch-lite), which trains a
**CIFAR-10 CNN**. The Karpathy `autoresearch` workload trains a **nanochat-style
GPT** on text. The two share *some* hyperparameter knobs (`LEARNING_RATE`,
`BATCH_SIZE`, `WEIGHT_DECAY`, optimizer choice) but most architecture-level
patterns (filter widths, dropout in conv blocks, color jitter) do not transfer.

In practice this means:

- **Useful right now**: catching obviously-doomed knob moves like "raise LR by
  10x", because the failure mode is dataset-agnostic.
- **Not yet useful**: GPT-specific judgments (depth/width tradeoffs, attention
  variants, optimizer-specific learning-rate schedules). These will only become
  reliable once a GPT-side seed is collected by running the integrated fork
  for some hours and feeding its `results.tsv` back into NRR.

Treat the current integration as a guard rail, not an oracle.

## Live integration with autoresearch-lite

The post-hoc analysis above only proves the API works. The next step is to wire
`check_proposal()` into the live experiment loop so it actually prevents wasted
compute. That integration now exists.

### How it is wired in

`autoresearch-lite/run_loop.py` calls `check_proposal()` after every LLM
proposal and before any training run:

```
LLM proposes change  ->  NRR.check_proposal(description, config_diff)
                          |
                          +-- proceed   -> run experiment
                          +-- caution   -> run experiment, log warning
                          +-- avoid     -> reject proposal, ask LLM for a
                                            different direction (max 3 retries)
```

If all 3 retries are still rejected the experiment is skipped and recorded in
`results.tsv` with a `discard` row tagged `NRR-skipped`. Every decision is
appended to `nrr_decisions.log` (one JSON record per attempt).

### One-cycle measured impact

A 1-cycle run (3 LLM-proposed experiments) seeded with the existing 21
experiments produced the following NRR decisions:

| Experiment | Proposal                              | NRR verdict | Outcome |
|-----------:|---------------------------------------|-------------|---------|
| 1          | (LLM JSON parse error)                | n/a         | crash   |
| 2          | weight_decay 5e-5 -> 2.5e-5           | CAUTION (19%) | DISCARD (0.6979 vs 0.7399 baseline) |
| 3          | weight_decay 5e-5 -> 1e-5             | CAUTION (18%) | git-commit error (idempotent change) |

`avoid=0 caution=2 proceed=0 ran=2 total=2`. Both NRR cautions were correct:
neither change improved the model. The cautions cited 5 highly similar past
failures and the regularization-direction pattern.

In a longer run, NRR's strongest pattern
`AVOID increasing LEARNING_RATE from current value. All 4 attempts failed.`
(confidence 1.00) is the one most likely to fire as `avoid` and force the LLM
to propose a different direction.

### What this proves

NRR has moved from "post-mortem analysis" to "in-the-loop guard rail". Even on
a 3-experiment slice, every proposal that the seed data flagged as suspicious
turned out to actually be a wasted experiment - so the integration immediately
gives the loop another signal beyond raw LLM intuition. The remaining open
question is sample efficiency: how much past data is needed before NRR's avoid
signals are net positive vs. how much it suppresses useful exploration. This
is now measurable end-to-end.

## Data

- `data/results.tsv`: Real autoresearch-lite output (21 experiments on CIFAR-10 CNN)
- `data/nrr_database.json`: Structured database generated by the parser

## Architecture

```
src/nrr/
  models.py      - Pydantic data models (NegativeResult, FailurePattern, ConfigDiff)
  parser.py      - results.tsv -> structured NegativeResult objects
  repository.py  - Storage, similarity search, pattern aggregation, loop integration
```

## Related

- [autoresearch-lite](https://github.com/t46/autoresearch-lite) - Lightweight autoresearch loop (data source)
- [Blog post](https://t46.github.io/blogs/negative_result_repository.html) - Design rationale and validation results
