"""Negative Result Repository: Structure, search, and learn from failed experiments."""

from nrr.models import NegativeResult, FailureCategory, ConfigDiff
from nrr.parser import parse_results_tsv
from nrr.repository import NegativeResultRepository

__all__ = [
    "NegativeResult",
    "FailureCategory",
    "ConfigDiff",
    "parse_results_tsv",
    "NegativeResultRepository",
]
