"""Parser for autoresearch results.tsv into structured NegativeResult objects.

This parser works with real autoresearch-lite data, extracting config diffs
and failure classifications from the description field and metrics.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from nrr.models import (
    ChangeCategory,
    ConfigDiff,
    FailureCategory,
    NegativeResult,
)

# Baseline config from autoresearch-lite (commit 2108755)
BASELINE_CONFIG = {
    "BATCH_SIZE": "128",
    "LEARNING_RATE": "0.01",
    "WEIGHT_DECAY": "1e-4",
    "NUM_EPOCHS": "10",
    "OPTIMIZER": "sgd",
    "LR_SCHEDULER": "cosine",
    "DROPOUT": "0.0",
    "NUM_FILTERS_1": "32",
    "NUM_FILTERS_2": "64",
    "NUM_FILTERS_3": "128",
    "FC_SIZE": "256",
    "USE_BATCHNORM": "True",
    "ACTIVATION": "relu",
    "USE_HORIZONTAL_FLIP": "True",
    "USE_RANDOM_CROP": "True",
    "USE_COLOR_JITTER": "False",
}

# Config after experiment 7 (98aea59) was kept: NUM_EPOCHS changed to 15
CONFIG_AFTER_EPOCHS = {**BASELINE_CONFIG, "NUM_EPOCHS": "15"}
# Config after experiment 21 (44fb21c) was kept: WEIGHT_DECAY changed to 5e-5
CONFIG_AFTER_WEIGHT_DECAY = {**CONFIG_AFTER_EPOCHS, "WEIGHT_DECAY": "5e-5"}

# Maps experiment index to the active config at that point
# Experiments 1-6 use baseline, 7+ use post-epochs config, 21+ use post-wd config
KEEP_COMMITS = {
    "2108755": 0.7094,  # baseline
    "98aea59": 0.7363,  # epochs 10->15
    "44fb21c": 0.7399,  # weight_decay 1e-4->5e-5
}


def _classify_change(description: str) -> ChangeCategory:
    """Classify what type of change was attempted based on description."""
    desc_lower = description.lower()

    # Scheduler check first: if the primary action is about the scheduler
    if ("scheduler" in desc_lower or "annealing" in desc_lower) and (
        "switch" in desc_lower or "change" in desc_lower or "from cosine" in desc_lower or "to step" in desc_lower
    ):
        return ChangeCategory.SCHEDULER
    if "learning rate" in desc_lower or "lr " in desc_lower:
        return ChangeCategory.LEARNING_RATE
    if "optimizer" in desc_lower or "adam" in desc_lower or "sgd" in desc_lower:
        return ChangeCategory.OPTIMIZER
    if "scheduler" in desc_lower or "cosine" in desc_lower or "step" in desc_lower:
        if "learning rate" not in desc_lower:
            return ChangeCategory.SCHEDULER
    if "weight decay" in desc_lower:
        return ChangeCategory.REGULARIZATION
    if "dropout" in desc_lower:
        return ChangeCategory.REGULARIZATION
    if "regulariz" in desc_lower:
        return ChangeCategory.REGULARIZATION
    if any(kw in desc_lower for kw in ["residual", "layer", "filter", "block", "depth", "architecture", "capacity"]):
        return ChangeCategory.ARCHITECTURE
    if any(kw in desc_lower for kw in ["augment", "flip", "crop", "jitter", "color"]):
        return ChangeCategory.DATA_AUGMENTATION
    if any(kw in desc_lower for kw in ["epoch", "batch"]):
        return ChangeCategory.TRAINING_DURATION
    if any(kw in desc_lower for kw in ["activation", "relu", "gelu", "silu"]):
        return ChangeCategory.ACTIVATION
    if "fc" in desc_lower or "hidden" in desc_lower:
        return ChangeCategory.ARCHITECTURE

    return ChangeCategory.MULTIPLE


def _extract_config_diffs(description: str, parent_config: dict[str, str]) -> list[ConfigDiff]:
    """Extract specific parameter changes from the description."""
    diffs = []
    desc_lower = description.lower()
    change_cat = _classify_change(description)

    # Learning rate changes
    lr_match = re.search(r"learning rate.*?to\s+([\d.e-]+)", desc_lower)
    if lr_match:
        diffs.append(ConfigDiff(
            parameter="LEARNING_RATE",
            baseline_value=parent_config["LEARNING_RATE"],
            experiment_value=lr_match.group(1),
            change_category=ChangeCategory.LEARNING_RATE,
        ))

    lr_from_to = re.search(r"learning rate.*?from\s+([\d.e-]+)\s+to\s+([\d.e-]+)", desc_lower)
    if lr_from_to and not lr_match:
        diffs.append(ConfigDiff(
            parameter="LEARNING_RATE",
            baseline_value=lr_from_to.group(1),
            experiment_value=lr_from_to.group(2),
            change_category=ChangeCategory.LEARNING_RATE,
        ))

    # Optimizer changes
    if "adamw" in desc_lower and "switch" in desc_lower:
        diffs.append(ConfigDiff(
            parameter="OPTIMIZER",
            baseline_value=parent_config["OPTIMIZER"],
            experiment_value="adamw",
            change_category=ChangeCategory.OPTIMIZER,
        ))
    elif "adam" in desc_lower and "switch" in desc_lower and "adamw" not in desc_lower:
        diffs.append(ConfigDiff(
            parameter="OPTIMIZER",
            baseline_value=parent_config["OPTIMIZER"],
            experiment_value="adam",
            change_category=ChangeCategory.OPTIMIZER,
        ))

    # Epoch changes
    epoch_match = re.search(r"epochs.*?from\s+(\d+)\s+to\s+(\d+)", desc_lower)
    if epoch_match:
        diffs.append(ConfigDiff(
            parameter="NUM_EPOCHS",
            baseline_value=epoch_match.group(1),
            experiment_value=epoch_match.group(2),
            change_category=ChangeCategory.TRAINING_DURATION,
        ))

    # Weight decay changes
    wd_match = re.search(r"weight decay.*?from\s+([\d.e-]+)\s+to\s+([\d.e-]+)", desc_lower)
    if wd_match:
        diffs.append(ConfigDiff(
            parameter="WEIGHT_DECAY",
            baseline_value=wd_match.group(1),
            experiment_value=wd_match.group(2),
            change_category=ChangeCategory.REGULARIZATION,
        ))
    elif "weight decay" in desc_lower and not wd_match:
        wd_val = re.search(r"([\d.]+e-\d+|[\d.]+e\d+)", description)
        if wd_val:
            diffs.append(ConfigDiff(
                parameter="WEIGHT_DECAY",
                baseline_value=parent_config["WEIGHT_DECAY"],
                experiment_value=wd_val.group(1),
                change_category=ChangeCategory.REGULARIZATION,
            ))

    # Dropout changes
    dropout_match = re.search(r"dropout.*?of\s+([\d.]+)", desc_lower)
    if dropout_match:
        diffs.append(ConfigDiff(
            parameter="DROPOUT",
            baseline_value=parent_config["DROPOUT"],
            experiment_value=dropout_match.group(1),
            change_category=ChangeCategory.REGULARIZATION,
        ))

    # Activation changes
    for act in ["gelu", "silu"]:
        if act in desc_lower and ("switch" in desc_lower or "change" in desc_lower):
            diffs.append(ConfigDiff(
                parameter="ACTIVATION",
                baseline_value=parent_config["ACTIVATION"],
                experiment_value=act,
                change_category=ChangeCategory.ACTIVATION,
            ))

    # Filter/architecture changes
    if "doubling" in desc_lower and "filter" in desc_lower:
        diffs.append(ConfigDiff(
            parameter="NUM_FILTERS_1",
            baseline_value=parent_config["NUM_FILTERS_1"],
            experiment_value=str(int(parent_config["NUM_FILTERS_1"]) * 2),
            change_category=ChangeCategory.ARCHITECTURE,
        ))
        diffs.append(ConfigDiff(
            parameter="NUM_FILTERS_2",
            baseline_value=parent_config["NUM_FILTERS_2"],
            experiment_value=str(int(parent_config["NUM_FILTERS_2"]) * 2),
            change_category=ChangeCategory.ARCHITECTURE,
        ))
        diffs.append(ConfigDiff(
            parameter="NUM_FILTERS_3",
            baseline_value=parent_config["NUM_FILTERS_3"],
            experiment_value=str(int(parent_config["NUM_FILTERS_3"]) * 2),
            change_category=ChangeCategory.ARCHITECTURE,
        ))

    # FC size changes
    fc_match = re.search(r"fc.*?(?:from\s+)?(\d+)\s+to\s+(\d+)", desc_lower)
    if fc_match:
        diffs.append(ConfigDiff(
            parameter="FC_SIZE",
            baseline_value=fc_match.group(1),
            experiment_value=fc_match.group(2),
            change_category=ChangeCategory.ARCHITECTURE,
        ))
    elif "hidden" in desc_lower:
        fc_val = re.search(r"to\s+(\d+)", desc_lower)
        if fc_val:
            diffs.append(ConfigDiff(
                parameter="FC_SIZE",
                baseline_value=parent_config["FC_SIZE"],
                experiment_value=fc_val.group(1),
                change_category=ChangeCategory.ARCHITECTURE,
            ))

    # Color jitter
    if "color jitter" in desc_lower:
        diffs.append(ConfigDiff(
            parameter="USE_COLOR_JITTER",
            baseline_value=parent_config["USE_COLOR_JITTER"],
            experiment_value="True",
            change_category=ChangeCategory.DATA_AUGMENTATION,
        ))

    # Batch size
    batch_match = re.search(r"batch size.*?from\s+(\d+)\s+to\s+(\d+)", desc_lower)
    if batch_match:
        diffs.append(ConfigDiff(
            parameter="BATCH_SIZE",
            baseline_value=batch_match.group(1),
            experiment_value=batch_match.group(2),
            change_category=ChangeCategory.TRAINING_DURATION,
        ))

    # Scheduler changes
    if "step scheduler" in desc_lower or "step lr" in desc_lower:
        diffs.append(ConfigDiff(
            parameter="LR_SCHEDULER",
            baseline_value=parent_config["LR_SCHEDULER"],
            experiment_value="step",
            change_category=ChangeCategory.SCHEDULER,
        ))

    # Residual connections (architecture)
    if "residual" in desc_lower:
        diffs.append(ConfigDiff(
            parameter="ARCHITECTURE",
            baseline_value="SimpleCNN",
            experiment_value="SimpleCNN+residual",
            change_category=ChangeCategory.ARCHITECTURE,
        ))

    # Fourth conv block
    if "fourth" in desc_lower and ("block" in desc_lower or "layer" in desc_lower):
        diffs.append(ConfigDiff(
            parameter="NUM_BLOCKS",
            baseline_value="3",
            experiment_value="4",
            change_category=ChangeCategory.ARCHITECTURE,
        ))

    # Gradient clipping
    if "gradient clipping" in desc_lower:
        diffs.append(ConfigDiff(
            parameter="GRADIENT_CLIP",
            baseline_value="none",
            experiment_value="1.0",
            change_category=ChangeCategory.REGULARIZATION,
        ))

    # If no diffs found, create a generic one
    if not diffs:
        diffs.append(ConfigDiff(
            parameter="UNKNOWN",
            baseline_value="unknown",
            experiment_value="unknown",
            change_category=change_cat,
        ))

    return diffs


def _classify_failure(
    status: str,
    val_accuracy: float,
    baseline_accuracy: float,
    description: str,
) -> FailureCategory:
    """Classify why an experiment failed."""
    if status == "keep":
        # Not a failure, but we still classify for completeness
        return FailureCategory.NO_IMPROVEMENT  # Will be overridden

    if status == "crash":
        if "llm error" in description.lower() or "parse" in description.lower():
            return FailureCategory.CRASH_INFRA
        return FailureCategory.CRASH_CODE

    # Discard cases
    delta = val_accuracy - baseline_accuracy
    if delta < -0.03:
        return FailureCategory.REGRESSION
    if delta > -0.005:
        return FailureCategory.MARGINAL
    return FailureCategory.NO_IMPROVEMENT


def _generate_lesson(result: NegativeResult) -> str:
    """Generate a human-readable lesson from a negative result."""
    if result.failure_category == FailureCategory.CRASH_CODE:
        return (
            f"Architectural change '{result.description[:60]}' caused a code error. "
            f"Complex structural modifications to SimpleCNN require careful implementation."
        )
    if result.failure_category == FailureCategory.CRASH_INFRA:
        return f"Infrastructure/parsing error: {result.description[:80]}"

    if result.failure_category == FailureCategory.REGRESSION:
        return (
            f"Change caused significant regression ({result.accuracy_delta:+.4f}). "
            f"Category: {result.change_category.value}. "
            f"This direction of change is actively harmful."
        )

    if result.failure_category == FailureCategory.MARGINAL:
        return (
            f"Change was marginal ({result.accuracy_delta:+.4f}), very close to baseline. "
            f"Category: {result.change_category.value}. "
            f"May work with different magnitude or combination."
        )

    # NO_IMPROVEMENT
    return (
        f"Change did not improve accuracy ({result.accuracy_delta:+.4f}). "
        f"Category: {result.change_category.value}. "
        f"This modification does not help in the current configuration."
    )


def _compute_feature_vector(result: NegativeResult, parent_config: dict[str, str]) -> list[float]:
    """Compute a numeric feature vector for similarity search.

    Features:
    - [0] accuracy_delta (normalized)
    - [1] is_crash (0/1)
    - [2-8] change_category one-hot (7 categories)
    - [9-15] failure_category one-hot (5 categories, padded)
    - [16] num_config_changes
    - [17] magnitude_of_change (for numeric params)
    """
    features = []

    # Accuracy delta, normalized to roughly [-1, 1]
    features.append(result.accuracy_delta * 10)

    # Is crash
    features.append(1.0 if result.status == "crash" else 0.0)

    # Change category one-hot
    change_cats = list(ChangeCategory)
    change_vec = [0.0] * len(change_cats)
    if result.change_category in change_cats:
        change_vec[change_cats.index(result.change_category)] = 1.0
    features.extend(change_vec)

    # Failure category one-hot
    fail_cats = list(FailureCategory)
    fail_vec = [0.0] * len(fail_cats)
    if result.failure_category in fail_cats:
        fail_vec[fail_cats.index(result.failure_category)] = 1.0
    features.extend(fail_vec)

    # Number of config changes
    features.append(float(len(result.config_diffs)))

    # Magnitude of change for numeric params
    magnitude = 0.0
    for diff in result.config_diffs:
        try:
            old = float(diff.baseline_value)
            new = float(diff.experiment_value)
            if old != 0:
                magnitude += abs(new - old) / abs(old)
            else:
                magnitude += abs(new)
        except (ValueError, ZeroDivisionError):
            magnitude += 1.0  # categorical change counts as 1
    features.append(magnitude)

    return features


def parse_results_tsv(tsv_path: str | Path) -> list[NegativeResult]:
    """Parse a results.tsv file into structured NegativeResult objects.

    Args:
        tsv_path: Path to the results.tsv file from autoresearch-lite.

    Returns:
        List of NegativeResult objects (includes both failures and successes
        for context, but failures are the primary focus).
    """
    tsv_path = Path(tsv_path)
    results = []

    # Track current best accuracy and config as we process sequentially
    current_best = 0.0
    current_config = dict(BASELINE_CONFIG)

    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            commit = row["commit"].strip()
            val_accuracy = float(row["val_accuracy"])
            memory_gb = float(row["memory_gb"])
            status = row["status"].strip()
            description = row["description"].strip()

            # Determine baseline accuracy for this experiment
            if description == "baseline":
                baseline_accuracy = 0.0
            else:
                baseline_accuracy = current_best

            accuracy_delta = val_accuracy - baseline_accuracy

            # Classify
            change_cat = _classify_change(description)
            failure_cat = _classify_failure(status, val_accuracy, baseline_accuracy, description)

            # Extract config diffs
            if description == "baseline":
                config_diffs = []
            else:
                config_diffs = _extract_config_diffs(description, current_config)

            result = NegativeResult(
                experiment_id=commit,
                description=description,
                status=status,
                val_accuracy=val_accuracy,
                baseline_accuracy=baseline_accuracy,
                accuracy_delta=accuracy_delta,
                memory_gb=memory_gb,
                failure_category=failure_cat,
                change_category=change_cat,
                config_diffs=config_diffs,
            )

            # Generate lesson
            if status != "keep":
                result.lesson = _generate_lesson(result)

            # Compute feature vector
            result.feature_vector = _compute_feature_vector(result, current_config)

            results.append(result)

            # Update state if this experiment was kept
            if status == "keep":
                current_best = val_accuracy
                for diff in config_diffs:
                    if diff.parameter in current_config:
                        current_config[diff.parameter] = diff.experiment_value

    return results
