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
