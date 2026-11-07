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

## 10-Domain Comprehensive Analytical Architecture

Cleanalytix organizes Data Quality Intelligence across 10 structured analytical domains, exposed both as object properties and exportable DataFrames via `result.to_dataframe(view=...)`:

| Domain | Property / Method | Description |
| :--- | :--- | :--- |
| **1. Dataset Profiling** | `result.dataset_profile` | Executive dataset metrics: row/column counts, memory usage (MB, per-row), null density, complete/empty row counts, duplicate rows/columns, candidate keys. |
| **2. Column Profiling** | `result.column_profile` | Deep per-column metadata: inferred semantic roles, physical dtypes, distinct values, uniqueness, constant flags, mode frequencies, and conformance error counts. |
| **3. Metric Transparency** | `result.scoring_matrix`<br>`result.metrics_table`<br>`result.metric_inventory` | Granular per-column penalty matrix, standardized percentage DQ metrics table, and exhaustive audit ledger linking each metric to its catalog definition. |
| **4. Dimension Transparency** | `result.dimension_summary`<br>`result.scoring_audit_trail` | 6-dimension scoreboard with configured weights, applicability status, effective weights, weighted contributions, and human-readable audit trail. |
| **5. Statistical Intelligence** | `result.statistical_profile` | Bessel sample variance ($ddof=1$), standard deviation, Fisher excess kurtosis, skewness, MAD, IQR, percentiles ($p01$–$p99$), zero/negative ratios, HHI, and Gini-Simpson diversity. |
| **6. Relationship Analysis** | `result.relationships`<br>`result.correlations`<br>`result.candidate_keys` | Pearson/Spearman correlation matrices, Cramér's V associations, collinearity pairs ($|r| \ge 0.85$), missingness empirical Jaccard co-occurrence, and single/composite keys. |
| **7. Temporal Analysis** | `result.temporal_profile` | Datetime parse rate, time span, freshness days/status, detected cadence (e.g. Daily, Hourly), median step, max gap, sequence monotonicity, and future timestamps. |
| **8. Drift Intelligence** | `result.drift_profile` | Baseline distribution drift tests: 2-sample Kolmogorov-Smirnov test ($p$-value, statistic), normalized Wasserstein distance, Jeffreys PSI, JSD, and unseen categories. |
| **9. Anomaly Diagnostics** | `result.anomalies`<br>`result.insights` | Structured tabular diagnostics of detected anomalies categorized by severity (Critical, Warning, Info, Optimization), dimension, metric, and threshold. |
| **10. Actionable Recommendations**| `result.recommendations_table`<br>`result.recommendations` | Deterministic, ranked remediation recommendations with 1-based priority ranks, rule IDs (1–10), affected columns, analytical rationale, and code fixes. |

---

## Unified Analytical View Selector (`to_dataframe`)

Export any analytical domain directly to a pandas DataFrame using `result.to_dataframe(view)`:

```python
# Column-level profiling
df_cols = result.to_dataframe("columns")

# Deep statistical moments and percentiles
df_stats = result.to_dataframe("statistics")

# Per-column scoring penalty matrix
df_scoring = result.to_dataframe("scoring")

# 6-dimension scoreboard and weights
df_dims = result.to_dataframe("dimensions")

# Metric audit inventory
df_inv = result.to_dataframe("inventory")

# Cross-column Pearson correlations
df_corr = result.to_dataframe("relationships")

# Temporal coverage and cadence
df_temp = result.to_dataframe("temporal")

# Baseline drift diagnostics
df_drift = result.to_dataframe("drift")

# Anomaly diagnostics
df_anom = result.to_dataframe("anomalies")

# Ranked recommendations table
df_recs = result.to_dataframe("recommendations")

# Executive dataset summary (1 row)
df_dataset = result.to_dataframe("dataset")
```

---

## Standardized Metric Registry & Catalog

Cleanalytix features a centralized, thread-safe `MetricRegistry` maintaining definitions, domains, scopes, units, calculation formulas, and scoring contributions for all metrics:

```python
from cleanalytix import MetricRegistry, MetricDomain, QualityDimension

# Query metric definition by name or standardized alias
m = MetricRegistry.get("cell_missingness_percent")
print(f"Metric: {m.display_name} ({m.unit})")
print(f"Domain: {m.domain.value} | Dimension: {m.dimension.value}")
print(f"Formula: {m.calculation_method}")

# Query all metrics contributing to quality scoring
scoring_metrics = MetricRegistry.scoring_metrics()
print(f"Scoring metrics count: {len(scoring_metrics)}")

# Query all metrics within a specific domain
validity_metrics = MetricRegistry.by_domain(MetricDomain.VALIDITY)
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
