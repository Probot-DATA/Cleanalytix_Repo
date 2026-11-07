# Cleanalytix

**Data Quality Intelligence & Advanced Analytics Library for Python & pandas.**

[![PyPI](https://img.shields.io/pypi/v/cleanalytix.svg)](https://pypi.org/project/cleanalytix/)
[![Python Versions](https://img.shields.io/pypi/pyversions/cleanalytix.svg)](https://pypi.org/project/cleanalytix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Cleanalytix is a non-mutating, multidimensional Data Quality Intelligence library designed for data scientists, ML engineers, analytics teams, and data platform developers.

Unlike black-box profiling reports or destructive auto-cleaning scripts, Cleanalytix provides **mathematically rigorous, multidimensional quality scores**, **deep statistical diagnostics**, and **actionable, deterministic advisory recommendations**—while strictly guaranteeing **zero modification** of the user's data.

---

## Non-Negotiable Core Principle: Strict Non-Mutation

Cleanalytix operates under a strict **Zero-Mutation Guarantee**:
- **No data alteration:** The library never mutates, transforms, imputes, normalizes, or drops user records.
- **Pure diagnostics:** Input DataFrames are treated as strictly immutable.
- **Retired auto-cleaning:** Destructive, heuristic automated cleaning has been completely retired in v0.2.0. Calling `cleaning=True` raises `FeatureRetiredError` with guidance toward advisory remediation workflows.

---

## Key Features

- **Multidimensional Data Quality Scoring:** Rigorous two-level evaluation across 6 core quality dimensions with authoritative default weights strictly summing to 1.0:
  1. **Completeness (0.20):** Cell-level null density, row-level completeness, and missingness co-occurrence patterns.
  2. **Validity (0.20):** Semantic type conformity, whitespace cleanliness, domain business rules, and statistical outlier plausibility.
  3. **Uniqueness (0.15):** Dataset duplicate rows and primary/identifier column uniqueness.
  4. **Consistency (0.15):** Safe cross-column relational business rules.
  5. **Timeliness (0.15):** Ingestion freshness, future date containment, and optional sequence monotonicity.
  6. **Stability (0.15):** Distribution drift detection (Population Stability Index, Jensen-Shannon divergence, Wasserstein distance, KS test).
- **Dynamic Applicability Re-Weighting:** Non-applicable dimensions (e.g., Stability without a baseline, or Timeliness without datetime columns) are dynamically excluded from the denominator with their reasons exposed, and remaining applicable weights are proportionally re-normalized to 1.0.
- **Exact Information-Theoretic Metrics:** Mathematically rigorous Kullback-Leibler and Jensen-Shannon divergences without heuristic shortcuts, and histogram-based continuous Shannon entropy via the Freedman-Diaconis rule.
- **Primary Single-DataFrame API (`analyze`):** Returns a rich, structured `CleanalytixResult` supporting both attribute and dictionary access.
- **Deterministic Actionable Recommendations:** Generates structured `InsightItem`s with severity classification (Critical, Warning, Info, Optimization).
- **Deep Modular Metrics Package (`cleanalytix.metrics`):** Standalone, importable mathematical and statistical functions.
- **Backward-Compatible Pipeline (`Run_DQ_Pipeline`):** Supports existing batch and production-monitoring workflows with `cleaning=False`.

---

## Installation

```bash
pip install cleanalytix
```

For development:
```bash
git clone https://github.com/Probot-DATA/Cleanalytix_Repo
cd Cleanalytix_Repo
pip install -e .
```

---

## Quick Start: Modern Analysis API

```python
import pandas as pd
import cleanalytix as ca

# Load your dataset
df = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 105],
    "age": [25, 30, 35, 140, None],
    "name": [" Alice ", "Bob", "Charlie", "David", "Eve"],
    "is_active": [True, False, True, True, False],
})

# Run comprehensive data quality analysis
result = ca.analyze(df)

# Access high-level health score and quality dimensions
print(f"Overall DQ Score: {result.overall_score:.2f} / 100")
for dim_name, dim_score in result.quality_dimensions.items():
    status = "Applicable" if dim_score.applicable else f"N/A ({dim_score.reason})"
    print(f"  - {dim_name.title()}: {dim_score.score:.2f}% (weight: {dim_score.effective_weight:.4f}) [{status}]")

# Inspect actionable recommendations
print("\nActionable Recommendations:")
for rec in result.recommendations:
    print(f"  [{rec.severity.value.upper()}] {rec.column}: {rec.message}")
    print(f"    --> Recommendation: {rec.recommendation}")
```

Sample output:
```text
Overall DQ Score: 75.79 / 100
  - Completeness: 52.50% (weight: 0.3636) [Applicable]
  - Validity: 80.95% (weight: 0.3636) [Applicable]
  - Uniqueness: 100.00% (weight: 0.2727) [Applicable]
  - Consistency: 0.00% (weight: 0.0000) [N/A (No cross-column consistency rules configured)]
  - Timeliness: 0.00% (weight: 0.0000) [N/A (No datetime columns detected)]
  - Stability: 0.00% (weight: 0.0000) [N/A (No baseline reference dataset provided)]

Actionable Recommendations:
  [WARNING] age: Statistical outliers: 1 values (25.00%) in 'age' are statistical outliers.
    --> Recommendation: Inspect the distribution of 'age'. Verify whether outliers represent measurement errors or genuine extreme events.
  [INFO] name: Unstripped whitespace: 1 values (20.00%) in 'name' contain leading/trailing whitespace.
    --> Recommendation: Apply trimming (`str.strip()`) to column 'name' to prevent duplicate category keys.
```

---

## Production Monitoring & Distribution Drift

Compare newly arrived production datasets against baseline distributions:

```python
result = ca.analyze(production_df, baseline_df=baseline_df)

# Check stability dimension
stability = result.quality_dimensions["stability"]
print(f"Stability Score: {stability.score:.2f}%")

# Inspect drift findings
for insight in result.insights:
    if insight.dimension == ca.QualityDimension.STABILITY:
        print(f"Drift alert on {insight.column}: {insight.message}")
```

---

## Modular Metrics Suite

Cleanalytix provides direct access to individual analytical modules:

```python
from cleanalytix.metrics import (
    compute_completeness_metrics,
    detect_candidate_keys,
    compute_numeric_statistics,
    detect_outliers,
    compute_categorical_distribution,
    compute_numeric_drift,
    compute_numeric_correlations,
)

# Robust statistics with Bessel's correction
stats = compute_numeric_statistics(df["age"])
print("Median:", stats["median"], "Sample Std:", stats["std"])

# Discover single and composite primary keys
keys = detect_candidate_keys(df)
print("Candidate keys:", keys["single_keys"])
```

---

## Custom Business Validation Rules

Pass domain-specific rules as functions returning `True` for valid values:

```python
def valid_age(val):
    return pd.isna(val) or (0 <= float(val) <= 120)

result = ca.analyze(df, rules={"age": valid_age})
```

---

## Legacy Pipeline Compatibility

The existing `Run_DQ_Pipeline` interface is fully supported for profiling and scoring:

```python
from cleanalytix import Run_DQ_Pipeline

result = Run_DQ_Pipeline(
    dataset_names=["customers"],
    dataset_list=[df],
    cleaning=False,  # strictly non-mutating
)
print(result["base_data"]["dirty_scores"])
```

---

## License

MIT License. See [LICENSE](./LICENSE) for details.
