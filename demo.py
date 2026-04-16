"""Demo: Negative Result Repository with real autoresearch-lite data.

This script demonstrates all four capabilities:
1. Parsing and structuring failures from results.tsv
2. Similarity search: finding similar past failures
3. Pattern aggregation: extracting rules from failure clusters
4. Autoresearch loop integration: checking proposed experiments
"""

from pathlib import Path

from nrr import NegativeResultRepository


DATA_PATH = Path(__file__).parent / "data" / "results.tsv"


def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_parsing(repo: NegativeResultRepository) -> None:
    """Demo 1: Structured failure parsing."""
    section("1. STRUCTURED FAILURE PARSING")

    print(repo.summary())

    print("\n--- Individual Failures ---\n")
    for r in repo.failures:
        print(f"  [{r.experiment_id[:7]}] {r.status:8s} | "
              f"acc={r.val_accuracy:.4f} (delta={r.accuracy_delta:+.4f}) | "
              f"{r.failure_category.value:16s} | {r.change_category.value}")
        if r.config_diffs:
            for d in r.config_diffs:
                print(f"    {d.parameter}: {d.baseline_value} -> {d.experiment_value}")
        if r.lesson:
            print(f"    Lesson: {r.lesson[:100]}")
        print()


def demo_similarity_search(repo: NegativeResultRepository) -> None:
    """Demo 2: Similar failure search."""
    section("2. SIMILARITY SEARCH")

    # Scenario 1: Someone wants to try increasing learning rate
    print("--- Query: Increase learning rate to 0.05 ---\n")
    similar = repo.find_similar_to_config(
        {"LEARNING_RATE": "0.05"},
        description="Increase learning rate to 0.05 for faster convergence",
        top_k=5,
    )
    for result, sim in similar:
        print(f"  sim={sim:.3f} | [{result.experiment_id[:7]}] "
              f"delta={result.accuracy_delta:+.4f} | {result.description[:60]}")

    # Scenario 2: Someone wants to try Adam optimizer
    print("\n--- Query: Switch to Adam optimizer ---\n")
    similar = repo.find_similar_to_config(
        {"OPTIMIZER": "adam"},
        description="Switch optimizer from SGD to Adam",
        top_k=5,
    )
    for result, sim in similar:
        print(f"  sim={sim:.3f} | [{result.experiment_id[:7]}] "
              f"delta={result.accuracy_delta:+.4f} | {result.description[:60]}")

    # Scenario 3: Architecture change
    print("\n--- Query: Add residual connections ---\n")
    similar = repo.find_similar_to_config(
        {"ARCHITECTURE": "residual"},
        description="Add residual connections to CNN blocks",
        top_k=5,
    )
    for result, sim in similar:
        print(f"  sim={sim:.3f} | [{result.experiment_id[:7]}] "
              f"delta={result.accuracy_delta:+.4f} | {result.description[:60]}")


def demo_pattern_aggregation(repo: NegativeResultRepository) -> None:
    """Demo 3: Failure pattern aggregation."""
    section("3. FAILURE PATTERN AGGREGATION")

    for p in repo.patterns:
        print(f"  Pattern: {p.pattern_id}")
        print(f"  Category: {p.change_category.value}")
        print(f"  Confidence: {p.confidence:.2f}")
        print(f"  Evidence: {p.num_experiments} experiments")
        print(f"  Avg delta: {p.avg_accuracy_delta:+.4f}")
        print(f"  Description: {p.description}")
        print(f"  Rule: {p.rule}")
        print()


def demo_autoresearch_integration(repo: NegativeResultRepository) -> None:
    """Demo 4: Autoresearch loop integration."""
    section("4. AUTORESEARCH LOOP INTEGRATION")

    proposals = [
        {
            "description": "Increase learning rate to 0.02 for faster convergence",
            "config": {"LEARNING_RATE": "0.02"},
        },
        {
            "description": "Switch to AdamW optimizer with default settings",
            "config": {"OPTIMIZER": "adamw"},
        },
        {
            "description": "Add residual connections to all conv blocks",
            "config": {"ARCHITECTURE": "residual"},
        },
        {
            "description": "Reduce batch size to 64 for better generalization",
            "config": {"BATCH_SIZE": "64"},
        },
        {
            "description": "Enable mixed precision training with torch.amp",
            "config": {},  # Novel change, no direct config mapping
        },
    ]

    for p in proposals:
        print(f"--- Proposal: {p['description'][:60]} ---\n")
        result = repo.check_proposal(p["description"], p["config"] or None)

        icon = {"proceed": "GO", "caution": "!!", "avoid": "XX"}
        print(f"  [{icon[result['recommendation']]}] Recommendation: {result['recommendation'].upper()}")
        print(f"  Success probability: {result['estimated_success_probability']:.0%}")
        print(f"  Reason: {result['reason'][:120]}")

        if result["similar_failures"]:
            print(f"  Similar failures ({len(result['similar_failures'])}):")
            for sf in result["similar_failures"][:3]:
                print(f"    - [{sf['experiment_id'][:7]}] sim={sf['similarity']:.2f} "
                      f"delta={sf['accuracy_delta']:+.4f}")

        if result["relevant_patterns"]:
            print(f"  Relevant patterns:")
            for rp in result["relevant_patterns"]:
                print(f"    - [{rp['confidence']:.2f}] {rp['rule'][:80]}")

        print()


def main() -> None:
    print("Negative Result Repository - Demo with real autoresearch-lite data")
    print(f"Data source: {DATA_PATH}")
    print()

    # Load repository from real data
    repo = NegativeResultRepository.from_tsv(DATA_PATH)

    # Run all demos
    demo_parsing(repo)
    demo_similarity_search(repo)
    demo_pattern_aggregation(repo)
    demo_autoresearch_integration(repo)

    # Save to JSON for inspection
    output_path = Path(__file__).parent / "data" / "nrr_database.json"
    repo.to_json(output_path)
    print(f"\nDatabase saved to: {output_path}")

    # Verify round-trip
    repo2 = NegativeResultRepository.from_json(output_path)
    assert len(repo2.all_results) == len(repo.all_results)
    assert len(repo2.patterns) == len(repo.patterns)
    print("Round-trip JSON serialization: OK")


if __name__ == "__main__":
    main()
