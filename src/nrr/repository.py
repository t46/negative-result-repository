"""Negative Result Repository: store, search, and aggregate failed experiments.

Core capabilities:
1. Store structured negative results
2. Find similar past failures given a new experiment config
3. Aggregate individual failures into actionable patterns/rules
4. Provide an interface for autoresearch loop integration
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from nrr.models import (
    ChangeCategory,
    ConfigDiff,
    FailureCategory,
    FailurePattern,
    NegativeResult,
)
from nrr.parser import parse_results_tsv


class NegativeResultRepository:
    """Repository for storing, searching, and analyzing negative results."""

    def __init__(self) -> None:
        self._results: list[NegativeResult] = []
        self._patterns: list[FailurePattern] = []

    @classmethod
    def from_tsv(cls, tsv_path: str | Path) -> NegativeResultRepository:
        """Create a repository from an autoresearch-lite results.tsv file."""
        repo = cls()
        all_results = parse_results_tsv(tsv_path)
        for r in all_results:
            repo.add(r)
        repo._patterns = repo._extract_patterns()
        return repo

    def add(self, result: NegativeResult) -> None:
        """Add a negative result to the repository."""
        self._results.append(result)

    @property
    def failures(self) -> list[NegativeResult]:
        """Return only failed experiments (discard + crash)."""
        return [r for r in self._results if r.status != "keep"]

    @property
    def all_results(self) -> list[NegativeResult]:
        """Return all results including successes."""
        return list(self._results)

    @property
    def patterns(self) -> list[FailurePattern]:
        """Return extracted failure patterns."""
        return list(self._patterns)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query: NegativeResult | dict,
        top_k: int = 5,
        failures_only: bool = True,
    ) -> list[tuple[NegativeResult, float]]:
        """Find the most similar past results to a query.

        Args:
            query: A NegativeResult or a dict with experiment config to look up.
                   If dict, it should contain keys like {"LEARNING_RATE": "0.1", ...}
            top_k: Number of results to return.
            failures_only: If True, only return failed experiments.

        Returns:
            List of (result, similarity_score) tuples, sorted by similarity descending.
        """
        if isinstance(query, dict):
            query = self._config_to_query(query)

        query_vec = np.array(query.feature_vector).reshape(1, -1)

        candidates = self.failures if failures_only else self._results
        if not candidates:
            return []

        # Build matrix of feature vectors
        vecs = []
        valid_candidates = []
        for c in candidates:
            if c.feature_vector is not None and c.experiment_id != query.experiment_id:
                vecs.append(c.feature_vector)
                valid_candidates.append(c)

        if not vecs:
            return []

        mat = np.array(vecs)
        similarities = cosine_similarity(query_vec, mat)[0]

        # Sort by similarity
        ranked = sorted(
            zip(valid_candidates, similarities),
            key=lambda x: x[1],
            reverse=True,
        )

        return ranked[:top_k]

    def find_similar_to_config(
        self,
        config_changes: dict[str, str],
        description: str = "",
        top_k: int = 5,
    ) -> list[tuple[NegativeResult, float]]:
        """Find similar past failures given a proposed config change.

        This is the main interface for autoresearch loop integration.

        Args:
            config_changes: Dict of parameter -> new_value for proposed changes.
                           e.g., {"LEARNING_RATE": "0.05", "OPTIMIZER": "adam"}
            description: Optional text description of the proposed change.
            top_k: Number of results to return.

        Returns:
            List of (result, similarity_score) tuples.
        """
        query = self._config_to_query(config_changes, description)
        return self.find_similar(query, top_k=top_k, failures_only=True)

    def _config_to_query(
        self,
        config_changes: dict[str, str],
        description: str = "",
    ) -> NegativeResult:
        """Convert a config change dict into a NegativeResult for similarity search."""
        from nrr.parser import _classify_change, _compute_feature_vector, BASELINE_CONFIG

        # Use description if provided, otherwise construct from config
        if not description:
            parts = [f"{k}={v}" for k, v in config_changes.items()]
            description = "Proposed: " + ", ".join(parts)

        change_cat = _classify_change(description)

        # Build config diffs
        diffs = []
        for param, value in config_changes.items():
            param_upper = param.upper()
            baseline_val = BASELINE_CONFIG.get(param_upper, "unknown")
            diffs.append(ConfigDiff(
                parameter=param_upper,
                baseline_value=baseline_val,
                experiment_value=value,
                change_category=change_cat,
            ))

        result = NegativeResult(
            experiment_id="query",
            description=description,
            status="query",
            val_accuracy=0.0,
            baseline_accuracy=0.0,
            accuracy_delta=0.0,
            failure_category=FailureCategory.NO_IMPROVEMENT,
            change_category=change_cat,
            config_diffs=diffs,
        )

        # Get current best config for feature computation
        current_config = dict(BASELINE_CONFIG)
        for r in self._results:
            if r.status == "keep":
                for d in r.config_diffs:
                    if d.parameter in current_config:
                        current_config[d.parameter] = d.experiment_value

        result.feature_vector = _compute_feature_vector(result, current_config)
        return result

    # ------------------------------------------------------------------
    # Pattern aggregation
    # ------------------------------------------------------------------

    def _extract_patterns(self) -> list[FailurePattern]:
        """Extract failure patterns from the repository.

        Groups failures by change category and analyzes whether certain
        types of changes consistently fail.
        """
        patterns = []
        failures = self.failures

        if not failures:
            return patterns

        # Group by change category
        by_category: dict[ChangeCategory, list[NegativeResult]] = defaultdict(list)
        for f in failures:
            by_category[f.change_category].append(f)

        # Also look at parameter-level patterns
        by_parameter: dict[str, list[NegativeResult]] = defaultdict(list)
        for f in failures:
            for diff in f.config_diffs:
                by_parameter[diff.parameter].append(f)

        # Generate category-level patterns
        for cat, results in by_category.items():
            if not results:
                continue

            deltas = [r.accuracy_delta for r in results if r.status != "crash"]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
            crash_count = sum(1 for r in results if r.status == "crash")

            # Confidence: more evidence = higher confidence
            n = len(results)
            total_failures = len(failures)
            confidence = min(1.0, n / max(total_failures * 0.3, 1))

            # All negative deltas = strong signal
            if deltas and all(d < 0 for d in deltas):
                confidence = min(1.0, confidence * 1.3)

            rule = self._generate_rule(cat, results, avg_delta, crash_count)
            description = self._generate_pattern_description(cat, results, avg_delta)

            patterns.append(FailurePattern(
                pattern_id=f"pat_{cat.value}",
                change_category=cat,
                description=description,
                evidence=[r.experiment_id for r in results],
                confidence=confidence,
                rule=rule,
                avg_accuracy_delta=avg_delta,
                num_experiments=n,
            ))

        # Generate parameter-level patterns (more specific)
        for param, results in by_parameter.items():
            if len(results) < 2:
                continue  # Need at least 2 experiments for a pattern

            deltas = [r.accuracy_delta for r in results if r.status != "crash"]
            avg_delta = sum(deltas) / len(deltas) if deltas else 0.0

            # Direction analysis for numeric parameters
            direction_info = self._analyze_direction(param, results)

            if direction_info:
                confidence = min(1.0, len(results) / max(len(failures) * 0.2, 1))
                patterns.append(FailurePattern(
                    pattern_id=f"pat_param_{param.lower()}",
                    change_category=results[0].change_category,
                    description=direction_info["description"],
                    evidence=[r.experiment_id for r in results],
                    confidence=confidence,
                    rule=direction_info["rule"],
                    avg_accuracy_delta=avg_delta,
                    num_experiments=len(results),
                ))

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    def _analyze_direction(
        self,
        param: str,
        results: list[NegativeResult],
    ) -> dict | None:
        """Analyze whether increasing or decreasing a parameter consistently fails."""
        increases = []
        decreases = []

        for r in results:
            for diff in r.config_diffs:
                if diff.parameter != param:
                    continue
                try:
                    old = float(diff.baseline_value)
                    new = float(diff.experiment_value)
                    if new > old:
                        increases.append(r)
                    elif new < old:
                        decreases.append(r)
                except ValueError:
                    continue

        info = None

        if len(increases) >= 2:
            inc_deltas = [r.accuracy_delta for r in increases if r.status != "crash"]
            if inc_deltas and all(d <= 0 for d in inc_deltas):
                avg = sum(inc_deltas) / len(inc_deltas)
                info = {
                    "description": (
                        f"Increasing {param} consistently fails "
                        f"(n={len(increases)}, avg delta={avg:+.4f})"
                    ),
                    "rule": (
                        f"AVOID increasing {param} from current value. "
                        f"All {len(increases)} attempts failed."
                    ),
                }

        if len(decreases) >= 2:
            dec_deltas = [r.accuracy_delta for r in decreases if r.status != "crash"]
            if dec_deltas and all(d <= 0 for d in dec_deltas):
                avg = sum(dec_deltas) / len(dec_deltas)
                dec_info = {
                    "description": (
                        f"Decreasing {param} consistently fails "
                        f"(n={len(decreases)}, avg delta={avg:+.4f})"
                    ),
                    "rule": (
                        f"AVOID decreasing {param} from current value. "
                        f"All {len(decreases)} attempts failed."
                    ),
                }
                if info is None:
                    info = dec_info

        return info

    def _generate_rule(
        self,
        cat: ChangeCategory,
        results: list[NegativeResult],
        avg_delta: float,
        crash_count: int,
    ) -> str:
        """Generate an actionable rule from a failure pattern."""
        n = len(results)

        if crash_count > 0 and crash_count == n:
            return f"BLOCK: All {n} {cat.value} changes crashed. Do not attempt without safeguards."

        if crash_count > 0:
            crash_pct = crash_count / n * 100
            rule = f"CAUTION: {crash_pct:.0f}% of {cat.value} changes crash. "
        else:
            rule = ""

        if avg_delta < -0.02:
            rule += (
                f"AVOID: {cat.value} changes cause significant regression "
                f"(avg {avg_delta:+.4f}). {n} experiments tried."
            )
        elif avg_delta < -0.005:
            rule += (
                f"UNLIKELY: {cat.value} changes show modest negative effect "
                f"(avg {avg_delta:+.4f}). {n} experiments tried."
            )
        else:
            rule += (
                f"MARGINAL: {cat.value} changes have near-zero effect "
                f"(avg {avg_delta:+.4f}). {n} experiments tried, none improved."
            )

        return rule

    def _generate_pattern_description(
        self,
        cat: ChangeCategory,
        results: list[NegativeResult],
        avg_delta: float,
    ) -> str:
        """Generate a description for a failure pattern."""
        n = len(results)
        statuses = defaultdict(int)
        for r in results:
            statuses[r.failure_category.value] += 1

        parts = [f"{n} failed {cat.value} experiments"]
        for fc, count in sorted(statuses.items(), key=lambda x: -x[1]):
            parts.append(f"{count} {fc}")

        return "; ".join(parts) + f". Avg accuracy delta: {avg_delta:+.4f}"

    # ------------------------------------------------------------------
    # Autoresearch loop integration
    # ------------------------------------------------------------------

    def check_proposal(
        self,
        description: str,
        config_changes: dict[str, str] | None = None,
    ) -> dict:
        """Check a proposed experiment against the failure database.

        This is the primary integration point for autoresearch loops.
        Call this before running a new experiment.

        Args:
            description: Text description of the proposed change.
            config_changes: Optional dict of parameter -> value changes.

        Returns:
            Dict with:
                - "recommendation": "proceed" | "caution" | "avoid"
                - "reason": Explanation
                - "similar_failures": List of similar past failures
                - "relevant_patterns": List of relevant patterns
                - "estimated_success_probability": float
        """
        # Find similar failures
        if config_changes:
            similar = self.find_similar_to_config(config_changes, description, top_k=5)
        else:
            # Create a minimal query from description
            from nrr.parser import _classify_change
            cat = _classify_change(description)
            query = NegativeResult(
                experiment_id="query",
                description=description,
                status="query",
                val_accuracy=0.0,
                baseline_accuracy=0.0,
                accuracy_delta=0.0,
                failure_category=FailureCategory.NO_IMPROVEMENT,
                change_category=cat,
                config_diffs=[],
            )
            from nrr.parser import _compute_feature_vector, BASELINE_CONFIG
            query.feature_vector = _compute_feature_vector(query, BASELINE_CONFIG)
            similar = self.find_similar(query, top_k=5, failures_only=True)

        # Find relevant patterns
        from nrr.parser import _classify_change
        change_cat = _classify_change(description)
        relevant_patterns = [
            p for p in self._patterns
            if p.change_category == change_cat
        ]

        # Estimate success probability
        total_in_category = sum(
            1 for r in self._results if r.change_category == change_cat
        )
        successes_in_category = sum(
            1 for r in self._results
            if r.change_category == change_cat and r.status == "keep"
        )

        if total_in_category > 0:
            base_prob = successes_in_category / total_in_category
        else:
            base_prob = 0.15  # Default: ~15% keep rate from autoresearch

        # Adjust based on similar failures
        if similar:
            avg_sim = sum(s for _, s in similar) / len(similar)
            # High similarity to failures = lower probability
            adjusted_prob = base_prob * (1 - avg_sim * 0.5)
        else:
            adjusted_prob = base_prob

        # Determine recommendation
        if adjusted_prob < 0.05 or any(p.confidence > 0.7 for p in relevant_patterns):
            recommendation = "avoid"
            reason = self._build_avoid_reason(similar, relevant_patterns)
        elif adjusted_prob < 0.15 or any(p.confidence > 0.4 for p in relevant_patterns):
            recommendation = "caution"
            reason = self._build_caution_reason(similar, relevant_patterns, adjusted_prob)
        else:
            recommendation = "proceed"
            reason = f"No strong signals against this change. Estimated success probability: {adjusted_prob:.0%}"

        return {
            "recommendation": recommendation,
            "reason": reason,
            "similar_failures": [
                {
                    "experiment_id": r.experiment_id,
                    "description": r.description,
                    "accuracy_delta": r.accuracy_delta,
                    "similarity": round(sim, 3),
                    "lesson": r.lesson,
                }
                for r, sim in similar
            ],
            "relevant_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "rule": p.rule,
                    "confidence": p.confidence,
                    "num_experiments": p.num_experiments,
                }
                for p in relevant_patterns
            ],
            "estimated_success_probability": round(adjusted_prob, 3),
        }

    def _build_avoid_reason(
        self,
        similar: list[tuple[NegativeResult, float]],
        patterns: list[FailurePattern],
    ) -> str:
        """Build explanation for an 'avoid' recommendation."""
        reasons = []
        for p in patterns:
            if p.confidence > 0.5:
                reasons.append(f"Pattern: {p.rule}")
        if similar:
            top = similar[0]
            reasons.append(
                f"Most similar failure: '{top[0].description[:60]}' "
                f"(similarity={top[1]:.2f}, delta={top[0].accuracy_delta:+.4f})"
            )
        return " | ".join(reasons) if reasons else "Multiple similar failures found."

    def _build_caution_reason(
        self,
        similar: list[tuple[NegativeResult, float]],
        patterns: list[FailurePattern],
        prob: float,
    ) -> str:
        """Build explanation for a 'caution' recommendation."""
        reasons = [f"Estimated success probability: {prob:.0%}"]
        if patterns:
            reasons.append(f"Related pattern: {patterns[0].rule[:100]}")
        if similar:
            n_high = sum(1 for _, s in similar if s > 0.7)
            if n_high > 0:
                reasons.append(f"{n_high} highly similar past failures")
        return " | ".join(reasons)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Save the repository to a JSON file."""
        data = {
            "results": [r.model_dump() for r in self._results],
            "patterns": [p.model_dump() for p in self._patterns],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def from_json(cls, path: str | Path) -> NegativeResultRepository:
        """Load a repository from a JSON file."""
        data = json.loads(Path(path).read_text())
        repo = cls()
        for r_data in data["results"]:
            repo._results.append(NegativeResult(**r_data))
        for p_data in data["patterns"]:
            repo._patterns.append(FailurePattern(**p_data))
        return repo

    # ------------------------------------------------------------------
    # Summary / display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a human-readable summary of the repository."""
        total = len(self._results)
        failures = len(self.failures)
        keeps = total - failures
        crashes = sum(1 for r in self._results if r.status == "crash")
        discards = sum(1 for r in self._results if r.status == "discard")

        lines = [
            "=" * 60,
            "Negative Result Repository Summary",
            "=" * 60,
            f"Total experiments: {total}",
            f"  Keep: {keeps} ({keeps/total*100:.0f}%)",
            f"  Discard: {discards} ({discards/total*100:.0f}%)",
            f"  Crash: {crashes} ({crashes/total*100:.0f}%)",
            "",
            "Failure categories:",
        ]

        from collections import Counter
        cat_counts = Counter(r.failure_category.value for r in self.failures)
        for cat, count in cat_counts.most_common():
            lines.append(f"  {cat}: {count}")

        lines.append("")
        lines.append("Change categories (failures only):")
        change_counts = Counter(r.change_category.value for r in self.failures)
        for cat, count in change_counts.most_common():
            lines.append(f"  {cat}: {count}")

        if self._patterns:
            lines.append("")
            lines.append(f"Extracted patterns: {len(self._patterns)}")
            for p in self._patterns[:5]:
                lines.append(f"  [{p.confidence:.2f}] {p.rule[:80]}")

        return "\n".join(lines)
