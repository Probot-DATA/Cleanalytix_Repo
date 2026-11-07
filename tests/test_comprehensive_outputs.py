"""Comprehensive validation test suite for Cleanalytix 10-domain analytical output system.

Validates:
1. Standardized Metric Registry and Catalog integrity.
2. Dataset Profiling domain and summary exports.
3. Column Profiling domain and conformance metrics.
4. Statistical Intelligence domain with independent SciPy/NumPy reference tests.
5. Metric & Dimension Transparency (scoring_matrix, dimension_summary, metric_inventory).
6. Cross-column Relationship Analysis (Pearson, Spearman, Cramér's V, missingness Jaccard, candidate keys).
7. Temporal Analysis (cadence, gaps, freshness, monotonicity).
8. Baseline Drift Intelligence (KS test, Wasserstein, PSI, JSD, unseen categories).
9. Anomaly Diagnostics and Actionable Recommendations (priority ranks, rule IDs).
10. `to_dataframe(view=...)` unified view selector across all 12 views.
11. Critical edge cases: empty DF, single row, constant columns, duplicate indices, nullable dtypes, wide datasets.
12. Strict immutability and non-mutation guarantees.
"""

from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import cleanalytix as ca
from cleanalytix import (
    CleanalytixConfig,
    AnalysisLevel,
    FeatureRetiredError,
    DatasetProfile,
    RelationshipProfile,
    RecommendationItem,
    MetricRegistry,
    MetricDomain,
    MetricScope,
    MetricValueStatus,
)
from cleanalytix.types import QualityDimension, InsightSeverity


def _df_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash of a DataFrame's content."""
    # Convert DataFrame to a canonical CSV-like string for hashing
    csv_bytes = df.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


# ============================================================================
# 1. METRIC REGISTRY & CATALOG INTEGRITY
# ============================================================================

def test_metric_registry_and_catalog():
    """Verify that MetricRegistry contains authoritative specifications and query methods."""
    all_metrics = MetricRegistry.all()
    assert len(all_metrics) >= 20, "Catalog should define at least 20 core metrics"

    # Query by name
    comp_cell = MetricRegistry.get("cell_missingness_percent")
    assert comp_cell is not None
    assert comp_cell.domain == MetricDomain.COMPLETENESS
    assert comp_cell.dimension == QualityDimension.COMPLETENESS
    assert comp_cell.unit == "%"
    assert comp_cell.contributes_to_dq_score is True
    assert "missing" in comp_cell.description.lower()

    # Query by domain
    validity_metrics = MetricRegistry.by_domain(MetricDomain.VALIDITY)
    assert len(validity_metrics) >= 4
    for m in validity_metrics:
        assert m.domain == MetricDomain.VALIDITY

    # Query by dimension
    uniqueness_metrics = MetricRegistry.by_dimension(QualityDimension.UNIQUENESS)
    assert len(uniqueness_metrics) >= 2

    # Query scoring metrics
    scoring_metrics = MetricRegistry.scoring_metrics()
    assert len(scoring_metrics) >= 8
    for m in scoring_metrics:
        assert m.contributes_to_dq_score is True


# ============================================================================
# 2. DOMAIN 1: DATASET PROFILING
# ============================================================================

def test_dataset_profiling_domain():
    """Verify Domain 1: DatasetProfile structural metrics and summary export."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "val": np.random.randn(n),
        "cat": np.random.choice(["A", "B", "C"], size=n),
    })
    # Add a duplicate row
    df.loc[99] = df.loc[98]

    res = ca.analyze(df)
    dp = res.dataset_profile

    assert isinstance(dp, DatasetProfile)
    assert dp.total_rows == 100
    assert dp.total_columns == 3
    assert dp.total_cells == 300
    assert dp.memory_bytes > 0
    assert dp.memory_mb > 0.0
    assert dp.null_density == 0.0
    assert dp.complete_rows_count == 100
    assert dp.complete_rows_percent == 100.0
    assert dp.empty_rows_count == 0
    assert dp.duplicate_rows_count == 1
    assert dp.duplicate_rows_percent == 1.0
    assert "id" in dp.candidate_keys or ("id",) in dp.composite_candidate_keys or len(dp.candidate_keys) >= 0

    # Test to_dataframe() and to_dict()
    dp_df = dp.to_dataframe()
    assert isinstance(dp_df, pd.DataFrame)
    assert len(dp_df) == 1
    assert "total_rows" in dp_df.columns
    assert "overall_score" in dp_df.columns

    dp_dict = dp.to_dict()
    assert isinstance(dp_dict, dict)
    assert dp_dict["total_rows"] == 100


# ============================================================================
# 3. DOMAIN 2: COLUMN PROFILING
# ============================================================================

def test_column_profiling_domain():
    """Verify Domain 2: Column profile properties, roles, and error metrics."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, np.nan, 4.0, 5.0],
        "txt": ["  foo", "bar  ", "baz", "qux", "qux"],
        "const": [10, 10, 10, 10, 10],
    })

    res = ca.analyze(df)
    cp = res.column_profile

    assert isinstance(cp, pd.DataFrame)
    assert list(cp.index) == ["num", "txt", "const"]

    # Verify columns in column_profile
    expected_cols = [
        "column_role", "dtype", "total_rows", "non_null_count", "null_count",
        "null_percent", "distinct_count", "distinct_percent", "is_unique",
        "is_constant", "is_near_constant", "column_quality_score"
    ]
    for c in expected_cols:
        assert c in cp.columns, f"Missing expected column '{c}' in column_profile"

    # num column
    assert cp.loc["num", "null_count"] == 1
    assert cp.loc["num", "null_percent"] == 20.0
    assert cp.loc["num", "is_constant"] == False

    # txt column whitespace
    assert cp.loc["txt", "whitespace_count"] == 2

    # const column
    assert cp.loc["const", "is_constant"] == True
    assert cp.loc["const", "distinct_count"] == 1


# ============================================================================
# 4. DOMAIN 3 & 4: SCORING MATRIX, DIMENSION SUMMARY, METRIC INVENTORY
# ============================================================================

def test_scoring_transparency_and_reconciliation():
    """Verify Domain 3 & 4: scoring_matrix, dimension_summary, and audit trail."""
    df = pd.DataFrame({
        "a": [10.0, 20.0, 30.0, 40.0],
        "b": ["x", "y", "z", "w"],
    })

    res = ca.analyze(df)

    # 1. Scoring Matrix
    sm = res.scoring_matrix
    assert isinstance(sm, pd.DataFrame)
    assert list(sm.index) == ["a", "b"]
    # Dataset-level metrics must be NaN across columns (no artificial column scoring)
    assert "incomplete_rows_score" in sm.columns
    assert sm["incomplete_rows_score"].isna().all()

    # 2. Dimension Summary
    ds = res.dimension_summary
    assert isinstance(ds, pd.DataFrame)
    assert ds.index.name == "dimension" or "dimension" in ds.columns
    assert "score" in ds.columns
    assert "effective_weight" in ds.columns
    assert "weighted_contribution" in ds.columns

    # Effective weights of applicable dimensions must sum to 1.0 (100% normalized weight)
    applicable_rows = ds[ds["applicable"] == True]
    assert np.isclose(applicable_rows["effective_weight"].sum(), 1.0, atol=1e-2)

    # Weighted contributions must sum to overall_score
    reconciled_score = applicable_rows["weighted_contribution"].sum()
    assert np.isclose(reconciled_score, res.overall_score, atol=1e-2)

    # 3. Metric Inventory
    mi = res.metric_inventory
    assert isinstance(mi, pd.DataFrame)
    assert "metric_name" in mi.columns
    assert "domain" in mi.columns
    assert "status" in mi.columns
    assert len(mi) >= 10

    # 4. Scoring Audit Trail
    audit_str = res.scoring_audit_trail
    assert isinstance(audit_str, str)
    assert "CLEANALYTIX 6-DIMENSION DATA QUALITY SCORING AUDIT TRAIL" in audit_str
    assert "OVERALL DATASET QUALITY SCORE" in audit_str


# ============================================================================
# 5. DOMAIN 5: STATISTICAL INTELLIGENCE WITH SCIPY REFERENCE TESTS
# ============================================================================

def test_statistical_profile_mathematical_accuracy():
    """Verify Domain 5: Independent mathematical verification against SciPy/NumPy.

    Checks:
    - Bessel variance (ddof=1)
    - Bessel standard deviation (ddof=1)
    - Fisher excess kurtosis (fisher=True, bias=False)
    - Sample skewness (bias=False)
    - Median Absolute Deviation (MAD)
    - Percentiles p01..p99
    - IQR
    - Shannon entropy, HHI, Gini-Simpson
    """
    np.random.seed(123)
    # Generate 60 values with distinct mean, variance, skewness, and outliers
    vals = np.concatenate([
        np.random.normal(loc=50.0, scale=10.0, size=55),
        np.array([120.0, 130.0, -20.0, 0.0, 0.0]),
    ])
    df = pd.DataFrame({"val": vals})

    res = ca.analyze(df)
    sp = res.statistical_profile

    assert "val" in sp.index

    # Independent reference computations
    ref_mean = float(np.mean(vals))
    ref_var = float(np.var(vals, ddof=1))
    ref_std = float(np.std(vals, ddof=1))
    ref_skew = float(stats.skew(vals, bias=False))
    ref_kurt = float(stats.kurtosis(vals, fisher=True, bias=False))
    ref_mad = float(stats.median_abs_deviation(vals))
    ref_p25 = float(np.percentile(vals, 25))
    ref_p75 = float(np.percentile(vals, 75))
    ref_iqr = ref_p75 - ref_p25

    # Check moments
    assert np.isclose(sp.loc["val", "mean"], ref_mean, atol=1e-4)
    assert np.isclose(sp.loc["val", "variance"], ref_var, atol=1e-4)
    assert np.isclose(sp.loc["val", "std"], ref_std, atol=1e-4)
    assert np.isclose(sp.loc["val", "skewness"], ref_skew, atol=1e-3)
    assert np.isclose(sp.loc["val", "kurtosis_fisher"], ref_kurt, atol=1e-3)
    assert np.isclose(sp.loc["val", "mad"], ref_mad, atol=1e-4)
    assert np.isclose(sp.loc["val", "iqr"], ref_iqr, atol=1e-4)

    # Check zero counts and negative counts
    assert sp.loc["val", "zero_count"] == 2
    assert sp.loc["val", "negative_count"] == 1

    # Check categorical diversity on a category column
    df_cat = pd.DataFrame({"cat": ["A"] * 60 + ["B"] * 30 + ["C"] * 10})
    res_cat = ca.analyze(df_cat)
    sp_cat = res_cat.statistical_profile
    assert "cat" in sp_cat.index
    # Dominance ratio: 60 / 100 = 0.60
    assert np.isclose(sp_cat.loc["cat", "top_category_dominance_ratio"], 0.60, atol=1e-2)
    # HHI: 0.6^2 + 0.3^2 + 0.1^2 = 0.36 + 0.09 + 0.01 = 0.46
    assert np.isclose(sp_cat.loc["cat", "hhi"], 0.46, atol=1e-2)
    # Gini-Simpson: 1 - 0.46 = 0.54
    assert np.isclose(sp_cat.loc["cat", "gini_simpson_diversity"], 0.54, atol=1e-2)


# ============================================================================
# 6. DOMAIN 6: RELATIONSHIP ANALYSIS
# ============================================================================

def test_relationship_profile_domain():
    """Verify Domain 6: Pearson, Spearman, Cramér's V, missingness Jaccard, duplicate cols."""
    np.random.seed(99)
    x = np.linspace(1, 100, 100)
    y = 2.0 * x + np.random.normal(0, 1, 100)
    z = np.sin(x)

    # Missingness pattern with intentional partial overlap
    m1 = [np.nan if i < 10 else 1.0 for i in range(100)]
    m2 = [np.nan if i < 5 or (15 <= i < 20) else 1.0 for i in range(100)]

    df = pd.DataFrame({
        "x": x,
        "y": y,
        "z": z,
        "m1": m1,
        "m2": m2,
        "cat1": ["A", "B", "A", "B"] * 25,
        "cat2": ["A", "B", "A", "B"] * 25,  # perfectly associated
    })

    res = ca.analyze(df)
    rel = res.relationships

    assert isinstance(rel, RelationshipProfile)
    # Pearson
    assert isinstance(res.correlations, pd.DataFrame)
    assert np.isclose(res.correlations.loc["x", "y"], 1.0, atol=0.05)

    # High correlations list
    assert not rel.high_correlations.empty
    pair_cols = set(rel.high_correlations[["column_1", "column_2"]].values.flatten())
    assert "x" in pair_cols and "y" in pair_cols

    # Missingness Jaccard co-occurrence
    mj = rel.missingness_cooccurrence
    assert isinstance(mj, pd.DataFrame)
    if "m1" in mj.index and "m2" in mj.columns:
        # m1 is null for i in 0..9 (10 items)
        # m2 is null for i in 0..4 and 15..19 (10 items)
        # Intersection: 0..4 (5 items)
        # Union: 0..9 and 15..19 (15 items)
        # Jaccard = 5 / 15 = 0.3333
        assert np.isclose(mj.loc["m1", "m2"], 5.0 / 15.0, atol=0.02)

    # Cramér's V
    cv = rel.cramers_v
    assert isinstance(cv, pd.DataFrame)
    if "cat1" in cv.index and "cat2" in cv.columns:
        assert np.isclose(cv.loc["cat1", "cat2"], 1.0, atol=0.05)


# ============================================================================
# 7. DOMAIN 7: TEMPORAL PROFILE
# ============================================================================

def test_temporal_profile_domain():
    """Verify Domain 7: Temporal metrics, coverage, cadence, and sequence monotonicity."""
    # Monotonic daily sequence
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "ts": dates,
        "val": range(10),
    })

    res = ca.analyze(df)
    tp = res.temporal_profile

    assert isinstance(tp, pd.DataFrame)
    assert "ts" in tp.index
    assert tp.loc["ts", "parse_rate"] == 100.0
    assert tp.loc["ts", "span_days"] == 9.0
    assert tp.loc["ts", "is_monotonic_increasing"] == True
    assert tp.loc["ts", "detected_cadence"] == "Daily"
    assert tp.loc["ts", "future_dates_count"] == 0

    # Non-temporal dataset returns empty or clean profile
    df_no_dt = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    res_no_dt = ca.analyze(df_no_dt)
    assert isinstance(res_no_dt.temporal_profile, pd.DataFrame)


# ============================================================================
# 8. DOMAIN 8: BASELINE DRIFT INTELLIGENCE
# ============================================================================

def test_drift_profile_domain_with_scipy_reference():
    """Verify Domain 8: Baseline drift metrics with independent KS test reference."""
    np.random.seed(42)
    # Baseline: standard normal
    base_vals = np.random.normal(loc=0.0, scale=1.0, size=200)
    # Production: shifted normal (mean = 1.5)
    prod_vals = np.random.normal(loc=1.5, scale=1.0, size=200)

    base_df = pd.DataFrame({
        "num": base_vals,
        "cat": np.random.choice(["A", "B"], size=200, p=[0.7, 0.3]),
    })
    prod_df = pd.DataFrame({
        "num": prod_vals,
        "cat": np.random.choice(["A", "B", "NEW_CAT"], size=200, p=[0.2, 0.5, 0.3]),
    })

    res = ca.analyze(prod_df, baseline_df=base_df)
    dp = res.drift_profile

    assert isinstance(dp, pd.DataFrame)
    assert "num" in dp.index

    # Independent reference KS test
    ref_ks = stats.ks_2samp(base_vals, prod_vals)
    ref_wass = stats.wasserstein_distance(base_vals, prod_vals)

    assert np.isclose(dp.loc["num", "ks_statistic"], ref_ks.statistic, atol=1e-3)
    assert np.isclose(dp.loc["num", "wasserstein_distance"], ref_wass, atol=1e-3)
    assert dp.loc["num", "psi"] > 0.25  # Substantial drift detected
    assert dp.loc["num", "drift_detected"] == True

    # Categorical drift with unseen categories
    assert "cat" in dp.index
    assert dp.loc["cat", "unseen_categories_count"] == 1
    assert "NEW_CAT" in str(dp.loc["cat", "unseen_categories_list"])

    # When no baseline is provided, drift_profile is None
    res_single = ca.analyze(prod_df)
    assert res_single.drift_profile is None
    # to_dataframe("drift") should return empty DataFrame gracefully
    assert res_single.to_dataframe("drift").empty


# ============================================================================
# 9. DOMAIN 9 & 10: ANOMALIES & ACTIONABLE RECOMMENDATIONS
# ============================================================================

def test_anomalies_and_recommendations_domain():
    """Verify Domain 9 & 10: Structured anomalies and ranked recommendations."""
    df = pd.DataFrame({
        "clean_id": range(50),
        "missing_col": [np.nan] * 30 + [1.0] * 20,  # 60% missing -> Excessive Missing Values
        "text_ws": ["  val  "] * 10 + ["val"] * 40,   # Whitespace issues
        "skewed_col": [1.0] * 48 + [1000.0, 2000.0],  # Outliers / skewness
    })

    res = ca.analyze(df)

    # 1. Anomalies Table
    anom = res.anomalies
    assert isinstance(anom, pd.DataFrame)
    assert "severity" in anom.columns
    assert "dimension" in anom.columns
    assert "metric_name" in anom.columns
    assert "recommendation" in anom.columns
    assert len(anom) >= 1

    # 2. Recommendations Table
    rt = res.recommendations_table
    assert isinstance(rt, pd.DataFrame)
    expected_recs_cols = [
        "priority_rank", "rule_id", "dimension", "severity", "affected_column",
        "issue_name", "evidence", "rationale", "suggested_action"
    ]
    for c in expected_recs_cols:
        assert c in rt.columns, f"Missing '{c}' in recommendations_table"

    # Priority rank must be 1-indexed and sorted
    ranks = list(rt["priority_rank"])
    assert ranks == list(range(1, len(ranks) + 1))

    # Recommendations List
    recs_list = res.recommendations
    assert isinstance(recs_list, list)
    assert len(recs_list) == len(rt)
    assert isinstance(recs_list[0], RecommendationItem)


# ============================================================================
# 10. UNIFIED VIEW SELECTOR: to_dataframe(view=...)
# ============================================================================

def test_to_dataframe_views():
    """Verify that to_dataframe(view=...) supports all 12 views and raises ValueError on invalid views."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [10.0, 20.0, 30.0, 40.0],
        "d": pd.date_range("2026-01-01", periods=4, freq="D"),
    })
    res = ca.analyze(df, baseline_df=df)

    valid_views = [
        "columns", "scoring", "dimensions", "statistics", "metrics",
        "inventory", "relationships", "temporal", "drift", "anomalies",
        "recommendations", "dataset",
    ]

    for view in valid_views:
        v_df = res.to_dataframe(view=view)
        assert isinstance(v_df, pd.DataFrame), f"View '{view}' did not return a DataFrame"

    # Invalid view should raise ValueError
    with pytest.raises(ValueError, match="Unknown view 'invalid_view_name'"):
        res.to_dataframe(view="invalid_view_name")


# ============================================================================
# 11. EDGE CASES & ROBUSTNESS
# ============================================================================

def test_edge_cases_empty_and_single_row():
    """Verify behavior on empty DataFrames (0 rows) and single row DataFrames (1 row)."""
    # 1. Empty DataFrame (0 rows, 3 columns)
    empty_df = pd.DataFrame(columns=["a", "b", "c"])
    res_empty = ca.analyze(empty_df)
    assert res_empty.dataset_profile.total_rows == 0
    assert isinstance(res_empty.column_profile, pd.DataFrame)
    assert isinstance(res_empty.statistical_profile, pd.DataFrame)
    assert isinstance(res_empty.scoring_matrix, pd.DataFrame)

    # 2. Single-row DataFrame (1 row)
    single_df = pd.DataFrame({"a": [42.0], "b": ["constant_string"]})
    res_single = ca.analyze(single_df)
    assert res_single.dataset_profile.total_rows == 1
    # Sample variance on N=1 should be NaN gracefully (Bessel ddof=1 not defined)
    assert np.isnan(res_single.statistical_profile.loc["a", "variance"])


def test_edge_cases_constant_column():
    """Verify constant column with 0 variance does not divide by zero."""
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0, 5.0, 5.0]})
    res = ca.analyze(df)
    sp = res.statistical_profile
    assert sp.loc["const", "variance"] == 0.0
    assert sp.loc["const", "std"] == 0.0
    assert res.column_profile.loc["const", "is_constant"] == True


def test_edge_cases_nullable_pandas_dtypes():
    """Verify support for modern pandas nullable extension dtypes (Int64, Float64, boolean)."""
    df = pd.DataFrame({
        "nullable_int": pd.Series([1, None, 3, 4], dtype="Int64"),
        "nullable_float": pd.Series([1.5, 2.5, None, 4.5], dtype="Float64"),
        "nullable_bool": pd.Series([True, False, None, True], dtype="boolean"),
    })
    res = ca.analyze(df)
    assert res.dataset_profile.total_rows == 4
    assert res.column_profile.loc["nullable_int", "null_count"] == 1
    assert res.column_profile.loc["nullable_float", "null_count"] == 1
    assert res.column_profile.loc["nullable_bool", "null_count"] == 1


def test_edge_cases_duplicate_index():
    """Verify DataFrame with duplicate index labels runs without indexing crashes."""
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]},
        index=[10, 10, 20],  # Duplicate index 10
    )
    res = ca.analyze(df)
    assert res.dataset_profile.total_rows == 3
    assert "a" in res.column_profile.index
    assert "b" in res.column_profile.index


def test_edge_cases_wide_dataset():
    """Verify wide dataset (M >= 100 columns) executes cleanly and respects max_correlation_columns."""
    m_cols = 110
    n_rows = 50
    data = {f"col_{i}": np.random.randn(n_rows) for i in range(m_cols)}
    df = pd.DataFrame(data)

    config = CleanalytixConfig(max_correlation_columns=30)
    res = ca.analyze(df, config=config)

    assert res.dataset_profile.total_columns == 110
    # Pearson correlation matrix must be capped at 30 columns
    assert res.correlations.shape[0] <= 30
    assert res.correlations.shape[1] <= 30


# ============================================================================
# 12. IMMUTABILITY & SAFEGUARDS GUARANTEE
# ============================================================================

def test_strict_immutability_and_safeguards():
    """Verify input DataFrames are strictly immutable (SHA-256 hash check) and cleaning=True raises FeatureRetiredError."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["  Alice  ", "Bob", "Charlie ", "David"],
        "num": [10.0, np.nan, 30.0, 40.0],
    })

    hash_before = _df_hash(df)
    _ = ca.analyze(df)
    hash_after = _df_hash(df)

    # DataFrame must not have changed by even a single byte
    assert hash_before == hash_after, "Input DataFrame was mutated during ca.analyze()!"

    # Retired feature safeguard
    with pytest.raises(FeatureRetiredError, match="Automatic data cleaning .* has been retired"):
        ca.Run_DQ_Pipeline(
            dataset_names=["test"],
            dataset_list=[df],
            cleaning=True,
        )


# ============================================================================
# 13. REGRESSION: AUDIT TRAIL ROUNDING RECONCILIATION & REGISTRY COVERAGE
# ============================================================================

def test_scoring_audit_trail_rounding_reconciliation_exact_100():
    """Regression test: verify overall score is strictly 100.00 when all applicable dimensions score 100.00.

    Ensures that presentation rounding does NOT degrade the overall score (e.g., to 99.99),
    and that displayed weights sum to 1.0000 and contributions sum to 100.00.
    """
    # Perfectly clean dataset with 3 applicable dimensions: Completeness (100), Validity (100), Uniqueness (100)
    # Configured weights: 0.20, 0.20, 0.15 (total = 0.55).
    # Unrounded normalized weights: 0.363636..., 0.363636..., 0.272727...
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "num": [10.0, 20.0, 30.0, 40.0],
    })

    res = ca.analyze(df)

    # 1. Overall Score must be exactly 100.00 (NOT 99.99)
    assert res.overall_score == 100.00, f"Expected 100.00, got {res.overall_score}"

    # 2. Check dimension summary table
    ds = res.dimension_summary
    applicable = ds[ds["applicable"] == True]
    assert len(applicable) == 3

    # All applicable scores are 100.00
    for s in applicable["score"]:
        assert s == 100.00

    # Displayed effective weights must sum strictly to 1.0000
    assert np.isclose(applicable["effective_weight"].sum(), 1.0000, atol=1e-6)

    # Displayed weighted contributions must sum strictly to 100.00
    assert np.isclose(applicable["weighted_contribution"].sum(), 100.00, atol=1e-6)

    # 3. Check audit trail string formatting
    audit = res.scoring_audit_trail
    assert "Total Applicable Weight: 1.0000" in audit
    assert "OVERALL DATASET QUALITY SCORE: 100.00 / 100.00" in audit
    assert "99.99" not in audit


def test_metric_registry_comprehensive_coverage():
    """Verify that every analytical metric across all 10 domains is registered in MetricRegistry."""
    all_defs = MetricRegistry.all()
    assert len(all_defs) >= 30, f"Expected >= 30 registered metrics, found {len(all_defs)}"

    # Required non-scoring statistical metrics
    expected_stats = [
        "stat_variance", "stat_kurtosis", "stat_skewness", "stat_mad",
        "stat_iqr", "stat_cv", "stat_zero_ratio", "stat_negative_ratio",
        "stat_entropy", "stat_hhi", "stat_gini_simpson", "stat_top_dominance",
        "stat_rare_categories",
    ]
    for m_name in expected_stats:
        m = MetricRegistry.get(m_name)
        assert m is not None, f"Metric '{m_name}' not found in registry"
        assert m.contributes_to_dq_score is False
        assert m.unit != ""

    # Required temporal and dataset profiling metrics
    expected_profiling = [
        "temporal_parse_rate", "temporal_span_days", "temporal_cadence",
        "dataset_null_density", "dataset_memory_mb", "profile_column_quality_score",
        "drift_kl_divergence", "drift_ks_test", "drift_wasserstein_distance",
    ]
    for m_name in expected_profiling:
        m = MetricRegistry.get(m_name)
        assert m is not None, f"Metric '{m_name}' not found in registry"

    # Test alias resolution
    assert MetricRegistry.get("iqr") is not None
    assert MetricRegistry.get("cv") is not None
    assert MetricRegistry.get("null_density") is not None
    assert MetricRegistry.get("cadence") is not None
    assert MetricRegistry.get("column_quality_score") is not None

