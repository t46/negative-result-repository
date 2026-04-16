"""Data models for the Negative Result Repository."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FailureCategory(str, Enum):
    """High-level failure categories for experiment results."""

    NO_IMPROVEMENT = "no_improvement"  # Ran but didn't beat baseline
    REGRESSION = "regression"  # Significantly worse than baseline
    CRASH_CODE = "crash_code"  # Code error (syntax, runtime)
    CRASH_INFRA = "crash_infra"  # Infrastructure error (timeout, OOM, parse error)
    MARGINAL = "marginal"  # Very close to baseline but not enough


class ChangeCategory(str, Enum):
    """What kind of change was attempted."""

    LEARNING_RATE = "learning_rate"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    REGULARIZATION = "regularization"  # weight_decay, dropout
    ARCHITECTURE = "architecture"  # layers, filters, connections
    DATA_AUGMENTATION = "data_augmentation"
    TRAINING_DURATION = "training_duration"  # epochs, batch_size
    ACTIVATION = "activation"
    MULTIPLE = "multiple"  # Multiple changes at once


class ConfigDiff(BaseModel):
    """A single parameter change between baseline and experiment."""

    parameter: str = Field(description="Parameter name (e.g., LEARNING_RATE)")
    baseline_value: str = Field(description="Value in the baseline/parent config")
    experiment_value: str = Field(description="Value in this experiment")
    change_category: ChangeCategory = Field(description="Category of the change")


class NegativeResult(BaseModel):
    """A structured negative result from an autoresearch experiment."""

    experiment_id: str = Field(description="Unique identifier (commit hash)")
    description: str = Field(description="What was changed and why")
    status: str = Field(description="Original status: discard, crash, keep")

    # Metrics
    val_accuracy: float = Field(description="Validation accuracy achieved")
    baseline_accuracy: float = Field(description="Best accuracy before this experiment")
    accuracy_delta: float = Field(description="val_accuracy - baseline_accuracy")
    memory_gb: float = Field(default=0.0, description="Peak memory in GB")

    # Failure classification
    failure_category: FailureCategory = Field(description="Why this experiment failed")
    change_category: ChangeCategory = Field(description="What type of change was attempted")

    # Config diff
    config_diffs: list[ConfigDiff] = Field(
        default_factory=list,
        description="Specific parameter changes from baseline",
    )

    # For similarity search
    feature_vector: Optional[list[float]] = Field(
        default=None,
        description="Numeric feature vector for similarity computation",
    )

    # Lesson learned
    lesson: str = Field(
        default="",
        description="Human/LLM-readable lesson from this failure",
    )


class FailurePattern(BaseModel):
    """An aggregated pattern across multiple failures."""

    pattern_id: str = Field(description="Unique pattern identifier")
    change_category: ChangeCategory = Field(description="Category of changes in this pattern")
    description: str = Field(description="What this pattern means")
    evidence: list[str] = Field(description="Experiment IDs supporting this pattern")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score based on number of supporting experiments",
    )
    rule: str = Field(description="Actionable rule for future experiments")
    avg_accuracy_delta: float = Field(description="Average accuracy change in this pattern")
    num_experiments: int = Field(description="Number of experiments in this pattern")
