"""Production Data Quality and Distribution Drift Monitoring Example.

Demonstrates baseline-versus-production monitoring using both:
1. The modern unified ``cleanalytix.analyze()`` API with reference baseline comparison.
2. The backward-compatible ``Run_DQ_Pipeline()`` multi-dataset interface.

Strictly non-mutating: user data is never modified, transformed, or imputed.
"""

import pandas as pd
import cleanalytix as ca
from cleanalytix import Run_DQ_Pipeline


def valid_age(value):
    """Custom business validation rule: age must be in [0, 120] when present."""
    if pd.isna(value):
        return True
    try:
        return 0 <= float(value) <= 120
    except (TypeError, ValueError):
        return False


def main():
    # Historical baseline reference dataset
    base = pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 28, 32],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
            "city": ["NY", "NY", "LA", "SF", "NY", "LA"],
            "income": [50000, 52000, 48000, 55000, 51000, 49000],
        }
    )

    # Newly arrived production dataset to monitor against baseline
    prod = pd.DataFrame(
        {
            "age": [26, 39, 141, None, 29, 31],
            "name": ["Alice ", " Bob", "Charlie", "David", "Eve", "Grace"],
            "city": ["NY", "LA", "LA", "SF", "Chicago", "SF"],
            "income": [51000, 49500, 1000000, 47000, 53000, 50000],
        }
    )

    print("=" * 60)
    print("1. Modern Unified Analysis (cleanalytix.analyze)")
    print("=" * 60)

    # Evaluate production dataset with baseline drift comparison & custom rules
    result = ca.analyze(
        prod,
        baseline_df=base,
        rules={"age": valid_age},
    )

    print(f"\nOverall Production DQ Score: {result.overall_score:.2f} / 100")

    print("\nQuality Dimension Scores:")
    for dim_name, dim_score in result.quality_dimensions.items():
        status = "Applicable" if dim_score.applicable else f"N/A ({dim_score.reason})"
        print(f"  - {dim_name.title()}: {dim_score.score:.2f}% (weight: {dim_score.effective_weight:.4f}) [{status}]")

    print("\nStability (Distribution Drift) Dimension:")
    stability = result.quality_dimensions.get("stability")
    if stability and stability.applicable:
        print(f"  Score: {stability.score:.2f}%")
        for metric_name, score in stability.metric_scores.items():
            print(f"    * {metric_name}: {score:.2f}%")

    if not result.drift_profile.empty:
        print("\nDrift Profile Summary:")
        cols = [c for c in ["psi", "js_divergence", "drift_detected", "drift_severity"] if c in result.drift_profile.columns]
        print(result.drift_profile[cols])

    print("\nActionable Recommendations:")
    for rec in result.recommendations[:5]:
        col = rec.affected_column or "dataset"
        print(f"  [{rec.severity}] {col} ({rec.issue_name}):")
        print(f"    --> Action: {rec.suggested_action}")

    print("\n" + "=" * 60)
    print("2. Legacy Pipeline Compatibility (Run_DQ_Pipeline)")
    print("=" * 60)

    # Evaluate baseline and production datasets via legacy pipeline
    pipeline_result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[base],
        new_dataset_list=[prod],
        rules={"age": valid_age},
        cleaning=False,  # strictly non-mutating
    )

    print("\nBaseline Dataset Health:")
    print(pipeline_result["base_data"]["dirty_scores"])

    print("\nProduction Dataset Health:")
    print(pipeline_result["prod_data"]["dirty_scores"])


if __name__ == "__main__":
    main()
