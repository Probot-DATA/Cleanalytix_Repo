"""Targeted validation tests for Mathematical, Statistical, and Scoring Remediation.

Covers:
1. Exact numeric and categorical KL / JS / PSI divergence calculations.
2. Histogram-based continuous entropy using Freedman-Diaconis / Scott's rule.
3. Constant-baseline drift without artificial 1.0 jumps.
4. Authoritative 6-dimension scoring framework with default weights summing to 1.0.
5. Strict dimension weight validation and partial override normalization.
6. Applicability-aware aggregation (excluding non-applicable dimensions with transparent reasons).
7. Timeliness quality dimension (freshness, future date containment, sequence integrity).
8. Consistency quality dimension (safe cross-column business rules + outlier plausibility).
9. Uniqueness dimension with identifier column duplicate sensitivity.
10. Deterministic hand-calculated scoring benchmark.
"""

import math
import datetime
import numpy as np
import pandas as pd
import pytest

from cleanalytix.types import (
    QualityDimension,
    DimensionScore,
    CleanalytixConfig,
    ConsistencyRule,
    TimelinessConfig,
)
from cleanalytix.scoring import (
    DEFAULT_DIMENSION_WEIGHTS,
    DEFAULT_METRIC_WEIGHTS,
    resolve_dimension_weights,
    resolve_thresholds,
    calculate_metric_score,
    score_dataset,
)
from cleanalytix.metrics.drift import (
    compute_numeric_drift,
    compute_categorical_drift,
)
from cleanalytix.metrics.distribution import (
    compute_continuous_distribution,
    compute_categorical_distribution,
)
from cleanalytix.metrics.uniqueness import (
    compute_identifier_uniqueness,
    compute_duplicate_row_metrics,
)
from cleanalytix.metrics.temporal import compute_temporal_metrics
from cleanalytix.generate_meta import get_entropy, KL_PSI_and_JS
import cleanalytix as ca


def test_authoritative_default_weights_sum_to_one():
    """Verify that all 6 quality dimensions exist in default weights and sum strictly to 1.0."""
    expected_dimensions = {
        QualityDimension.COMPLETENESS,
        QualityDimension.VALIDITY,
        QualityDimension.UNIQUENESS,
        QualityDimension.CONSISTENCY,
        QualityDimension.TIMELINESS,
        QualityDimension.STABILITY,
    }
    assert set(DEFAULT_DIMENSION_WEIGHTS.keys()) == expected_dimensions
    assert math.isclose(sum(DEFAULT_DIMENSION_WEIGHTS.values()), 1.0, rel_tol=1e-9)

    assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.COMPLETENESS] == 0.20
    assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.VALIDITY] == 0.20
    assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.UNIQUENESS] == 0.15
    assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.CONSISTENCY] == 0.15
    assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.TIMELINESS] == 0.15
    assert DEFAULT_DIMENSION_WEIGHTS[QualityDimension.STABILITY] == 0.15


def test_dimension_weight_validation():
    """Verify weight validation: rejection of negatives, zero sum, and proper resolution."""
    # Negative weight raises ValueError
    with pytest.raises(ValueError, match="non-negative"):
        resolve_dimension_weights({"completeness": -0.1, "validity": 0.5})

    # All zeros raises ValueError
    with pytest.raises(ValueError, match="cannot all be zero"):
        resolve_dimension_weights({dim: 0.0 for dim in QualityDimension})

    # Unknown dimension raises ValueError
    with pytest.raises(ValueError, match="Unknown quality dimension"):
        resolve_dimension_weights({"invalid_dim": 0.5})

    # Full override with all 6 dimensions
    custom = {
        QualityDimension.COMPLETENESS: 0.30,
        QualityDimension.VALIDITY: 0.20,
        QualityDimension.UNIQUENESS: 0.10,
        QualityDimension.CONSISTENCY: 0.10,
        QualityDimension.TIMELINESS: 0.15,
        QualityDimension.STABILITY: 0.15,
    }
    resolved = resolve_dimension_weights(custom)
    assert math.isclose(sum(resolved.values()), 1.0, rel_tol=1e-9)
    assert resolved[QualityDimension.COMPLETENESS] == 0.30

    # Partial override summing to 1.0: unspecified become 0.0
    partial_exact = {"completeness": 0.60, "validity": 0.40}
    resolved_pe = resolve_dimension_weights(partial_exact)
    assert math.isclose(sum(resolved_pe.values()), 1.0, rel_tol=1e-9)
    assert resolved_pe[QualityDimension.COMPLETENESS] == 0.60
    assert resolved_pe[QualityDimension.VALIDITY] == 0.40
    assert resolved_pe[QualityDimension.UNIQUENESS] == 0.0
    assert resolved_pe[QualityDimension.TIMELINESS] == 0.0

    # Partial override not summing to 1.0: unspecified inherit defaults and everything is normalized
    partial_relative = {"completeness": 0.50}
    resolved_pr = resolve_dimension_weights(partial_relative)
    assert math.isclose(sum(resolved_pr.values()), 1.0, rel_tol=1e-9)
    assert resolved_pr[QualityDimension.COMPLETENESS] > resolved_pr[QualityDimension.VALIDITY]


def test_applicability_aware_aggregation_and_renormalization():
    """Verify that non-applicable dimensions have effective weight 0 and remaining are re-normalized."""
    # Tabular dataset with numeric and text, but NO datetime and NO baseline
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [85.0, 90.0, 78.0, 92.0, 88.0],
    })

    result = ca.analyze(df)

    dims = result.quality_dimensions
    assert len(dims) == 6

    # Stability is NOT applicable (no baseline)
    assert dims["stability"].applicable is False
    assert dims["stability"].effective_weight == 0.0
    assert dims["stability"].weight == 0.0
    assert "No baseline" in str(dims["stability"].reason)

    # Timeliness is NOT applicable (no datetime)
    assert dims["timeliness"].applicable is False
    assert dims["timeliness"].effective_weight == 0.0
    assert dims["timeliness"].weight == 0.0
    assert "No datetime" in str(dims["timeliness"].reason)

    # Consistency is NOT applicable (no consistency rules configured)
    assert dims["consistency"].applicable is False
    assert dims["consistency"].effective_weight == 0.0
    assert dims["consistency"].weight == 0.0
    assert "No cross-column consistency rules" in str(dims["consistency"].reason)

    # Completeness, Validity, Uniqueness are applicable
    applicable_dims = [ds for ds in dims.values() if ds.applicable]
    assert len(applicable_dims) == 3

    # Effective weights of applicable dimensions must sum strictly to 1.0
    eff_weight_sum = sum(ds.effective_weight for ds in applicable_dims)
    assert math.isclose(eff_weight_sum, 1.0, rel_tol=1e-4)

    # Hand-calculate expected overall score: sum(ds.score * ds.effective_weight)
    expected_overall = sum(ds.score * ds.effective_weight for ds in applicable_dims)
    assert math.isclose(result.overall_score, round(expected_overall, 2), abs_tol=0.02)


def test_exact_numeric_divergence_math():
    """Verify exact mathematical formulas for KL, JS, and PSI on continuous distributions."""
    np.random.seed(42)
    b = pd.Series(np.random.normal(0, 1, 2000))
    p = pd.Series(np.random.normal(0.6, 1.2, 2000))  # shifted and scaled

    drift = compute_numeric_drift(b, p)
    kl = drift["kl_divergence"]
    rev_kl = drift["reverse_kl_divergence"]
    psi = drift["psi"]
    jsd = drift["js_divergence"]

    # 1. Asymmetry of Kullback-Leibler: D_KL(P||Q) != D_KL(Q||P)
    assert kl != rev_kl
    assert abs(kl - rev_kl) > 0.01

    # 2. Symmetric Jeffreys relation: PSI == D_KL(P||Q) + D_KL(Q||P)
    assert math.isclose(psi, kl + rev_kl, rel_tol=1e-4)

    # 3. Jensen-Shannon divergence bounded in [0, ln(2)]
    assert 0.0 <= jsd <= math.log(2.0)
    assert jsd > 0.0  # shifted distributions have strictly positive divergence

    # 4. Identical series has zero drift
    self_drift = compute_numeric_drift(b, b)
    assert self_drift["ks_statistic"] == 0.0
    assert self_drift["wasserstein_distance"] == 0.0
    assert self_drift["psi"] == 0.0
    assert self_drift["kl_divergence"] == 0.0
    assert self_drift["js_divergence"] == 0.0


def test_constant_baseline_drift_continuity():
    """Verify constant baseline drift does not artificially jump to KS=1.0, PSI=1.0 for small shift."""
    b = pd.Series(np.full(1000, 10.0))
    # Production: 999 points identical, 1 point shifted by 0.1
    p_slight = pd.Series([10.0] * 999 + [10.1])

    res_slight = compute_numeric_drift(b, p_slight)
    # KS should be exactly 1/1000 = 0.001
    assert math.isclose(res_slight["ks_statistic"], 0.001, abs_tol=1e-4)
    # PSI should be small (< 0.10, severity 'none')
    assert res_slight["psi"] < 0.10
    assert res_slight["drift_severity"] == "none"

    # Production completely shifted: all 20.0
    p_complete = pd.Series(np.full(1000, 20.0))
    res_complete = compute_numeric_drift(b, p_complete)
    assert res_complete["ks_statistic"] == 1.0
    assert res_complete["psi"] >= 0.25
    assert res_complete["drift_severity"] == "critical"


def test_continuous_entropy_fd_scott_binning():
    """Verify that continuous entropy distinguishes distribution shapes and does not collapse to 1.0."""
    np.random.seed(42)
    uniform_s = pd.Series(np.random.uniform(0, 100, 5000))
    normal_s = pd.Series(np.random.normal(50, 10, 5000))
    exp_s = pd.Series(np.random.exponential(2.0, 5000))
    const_s = pd.Series(np.full(5000, 42.0))

    df = pd.DataFrame({
        "uniform": uniform_s,
        "normal": normal_s,
        "exp": exp_s,
        "const": const_s,
    })

    h_uniform = get_entropy(df, "uniform")
    h_normal = get_entropy(df, "normal")
    h_exp = get_entropy(df, "exp")
    h_const = get_entropy(df, "const")

    # Constant data has strictly zero entropy
    assert h_const == 0.0

    # Uniform distribution has highest entropy (approaching 1.0)
    assert h_uniform > 0.98

    # Peaked normal distribution has strictly lower entropy than uniform
    assert h_uniform > h_normal

    # Skewed exponential distribution has strictly lower entropy than normal
    assert h_normal > h_exp > 0.40

    # Verify specialized module matches
    dist_res = compute_continuous_distribution(normal_s)
    assert math.isclose(dist_res["normalized_entropy"], h_normal, abs_tol=0.01)


def test_timeliness_quality_dimension():
    """Verify Timeliness dimension scoring on fresh, stale, and future dates."""
    base_time = pd.Timestamp("2026-11-01 12:00:00")

    # 1. Fresh dataset (dates up to 5 days before reference)
    df_fresh = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-25", periods=5, freq="D"),
        "val": [1, 2, 3, 4, 5],
    })
    res_fresh = ca.analyze(
        df_fresh,
        config=CleanalytixConfig(
            timeliness_config=TimelinessConfig(reference_time=base_time, freshness_days=30.0)
        ),
    )
    t_dim = res_fresh.quality_dimensions["timeliness"]
    assert t_dim.applicable is True
    assert t_dim.score > 90.0

    # 2. Stale dataset (data is 60 days old, exceeds 30-day freshness threshold)
    df_stale = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-01", periods=5, freq="D"),
        "val": [1, 2, 3, 4, 5],
    })
    res_stale = ca.analyze(
        df_stale,
        config=CleanalytixConfig(
            timeliness_config=TimelinessConfig(reference_time=base_time, freshness_days=30.0)
        ),
    )
    t_dim_stale = res_stale.quality_dimensions["timeliness"]
    assert t_dim_stale.applicable is True
    # Score must be lower due to staleness decay
    assert t_dim_stale.score < t_dim.score

    # 3. Future dates violation
    df_future = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-10-20"),
            pd.Timestamp("2026-10-21"),
            pd.Timestamp("2026-12-01"),  # in the future
        ],
        "val": [1, 2, 3],
    })
    res_future = ca.analyze(
        df_future,
        config=CleanalytixConfig(
            timeliness_config=TimelinessConfig(reference_time=base_time, allow_future_dates=False)
        ),
    )
    t_future = res_future.quality_dimensions["timeliness"]
    assert t_future.metrics["timestamp_future_dates_percent"] > 0.0
    assert any(i.dimension == QualityDimension.TIMELINESS for i in res_future.insights)

    # 4. Unsorted timestamps - default require_monotonic=False MUST NOT penalize sequence
    df_unsorted = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2026-10-25"),
            pd.Timestamp("2026-10-23"),
            pd.Timestamp("2026-10-24"),
        ],
        "val": [1, 2, 3],
    })
    res_unsorted_default = ca.analyze(
        df_unsorted,
        config=CleanalytixConfig(
            timeliness_config=TimelinessConfig(reference_time=base_time, freshness_days=30.0)
        ),
    )
    t_unsorted_def = res_unsorted_default.quality_dimensions["timeliness"]
    assert t_unsorted_def.metric_scores["sequence_score"] == 100.0
    assert t_unsorted_def.score == 87.5

    # 5. Unsorted timestamps - explicit require_monotonic=True MUST penalize sequence
    res_unsorted_strict = ca.analyze(
        df_unsorted,
        config=CleanalytixConfig(
            timeliness_config=TimelinessConfig(
                reference_time=base_time,
                freshness_days=30.0,
                require_monotonic=True,
            )
        ),
    )
    t_unsorted_strict = res_unsorted_strict.quality_dimensions["timeliness"]
    assert t_unsorted_strict.metric_scores["sequence_score"] == 50.0
    assert t_unsorted_strict.score == 77.5
    assert t_unsorted_strict.score < t_unsorted_def.score


def test_consistency_rules_evaluation():
    """Verify Consistency dimension evaluates custom cross-column business rules without eval()."""
    df = pd.DataFrame({
        "start_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"]),
        "end_date": pd.to_datetime(["2026-01-10", "2026-02-10", "2026-02-01", "2026-04-10"]),  # row 3 violated!
        "price": [10.0, 20.0, 30.0, 40.0],
        "total": [12.0, 24.0, 36.0, 48.0],
    })

    # Rule: end_date must be >= start_date
    rule = ConsistencyRule(
        name="end_after_start",
        condition=lambda d: d["end_date"] >= d["start_date"],
    )

    res = ca.analyze(df, config=CleanalytixConfig(consistency_rules=[rule]))
    c_dim = res.quality_dimensions["consistency"]

    assert c_dim.applicable is True
    assert "cross_column_rules_score" in c_dim.metric_scores
    # 1 out of 4 rows violated (25% violation)
    assert c_dim.details["rule_details"]["end_after_start"]["violations"] == 1
    # Check that an insight was generated for consistency rule violation
    cons_insights = [i for i in res.insights if i.dimension == QualityDimension.CONSISTENCY]
    assert len(cons_insights) >= 1
    assert any("end_after_start" in i.message for i in cons_insights)


def test_uniqueness_with_identifier_duplicates():
    """Verify Uniqueness dimension is sensitive to duplicates in primary/identifier columns."""
    # Entire rows are unique, but the 'user_id' identifier column has duplicate keys!
    df = pd.DataFrame({
        "user_id": [101, 102, 101, 104],  # duplicate user_id 101
        "event": ["login", "login", "purchase", "logout"],
    })

    res = ca.analyze(df, config=CleanalytixConfig(identifier_columns=["user_id"]))
    u_dim = res.quality_dimensions["uniqueness"]

    assert u_dim.applicable is True
    assert u_dim.metrics["duplicate_rows_percent"] == 0.0  # 0 whole-row duplicates
    assert u_dim.metrics["identifier_duplicate_percent"] == 25.0  # 1 duplicate in 4 keys = 25%
    # Score should be penalized below 100.0 due to identifier duplicates
    assert u_dim.score < 100.0
    assert any("identifier" in i.message.lower() for i in res.insights)


def test_hand_calculated_deterministic_score():
    """Benchmark test verifying exact hand-calculated score on deterministic toy dataset."""
    # 10 rows, 2 columns: 'measurement' has 2 missing (20% missing), 0 duplicates
    df = pd.DataFrame({
        "measurement": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, np.nan],
        "cat": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    })

    # Linear mode, default thresholds: missing threshold 20.0%
    # Cell missing: 2 / 20 = 10.0%. Linear score: 1.0 - (10.0 / 20.0) = 0.50 -> 50.0%
    # Row missing: 2 / 10 = 20.0%. Linear score: 1.0 - (20.0 / 20.0) = 0.00 -> 0.0%
    # Completeness: 0.70 * 50.0 + 0.30 * 0.0 = 35.0
    # Duplicate rows: 0.0%. Score: 100.0
    # Validity: 0 errors, no outliers in 8 values. Score: 100.0
    #
    # Case A: No consistency rules configured
    # Consistency: not applicable (applicable=False). Effective weight: 0.0
    # Timeliness: not applicable (0 datetime). Effective weight: 0.0
    # Stability: not applicable (0 baseline). Effective weight: 0.0
    # Applicable dimensions: Completeness (0.20), Validity (0.20), Uniqueness (0.15)
    # Sum of configured weights: 0.20 + 0.20 + 0.15 = 0.55
    # Effective weights:
    # W(Completeness) = 0.20 / 0.55 = 4/11
    # W(Validity) = 0.20 / 0.55 = 4/11
    # W(Uniqueness) = 0.15 / 0.55 = 3/11
    # Expected overall = (35.0 * 4 + 100.0 * 4 + 100.0 * 3) / 11
    #                  = (140.0 + 400.0 + 300.0) / 11 = 840.0 / 11 ≈ 76.3636 -> 76.36

    res = ca.analyze(df)
    assert res.quality_dimensions["completeness"].score == 35.0
    assert res.quality_dimensions["uniqueness"].score == 100.0
    assert res.quality_dimensions["validity"].score == 100.0
    assert res.quality_dimensions["consistency"].applicable is False
    assert res.quality_dimensions["consistency"].score == 0.0
    assert math.isclose(res.overall_score, 76.36, abs_tol=0.05)

    # Case B: With cross-column consistency rule configured
    # Rule: when measurement is not null, measurement > 0
    rule = ConsistencyRule(
        name="positive_measurement",
        condition=lambda d: d["measurement"].isna() | (d["measurement"] > 0),
    )
    res_with_rule = ca.analyze(df, config=CleanalytixConfig(consistency_rules=[rule]))
    assert res_with_rule.quality_dimensions["consistency"].applicable is True
    assert res_with_rule.quality_dimensions["consistency"].score == 100.0
    # Applicable dimensions: Completeness (0.20), Validity (0.20), Uniqueness (0.15), Consistency (0.15)
    # Sum of configured weights = 0.70
    # Expected overall = (35.0 * 2 + 100.0 * 2 + 100.0 * 1.5 + 100.0 * 1.5) / 7 = 570.0 / 7 ≈ 81.43
    assert math.isclose(res_with_rule.overall_score, 81.43, abs_tol=0.05)


def test_custom_metric_weights_change_dimension_score():
    """Verify that custom metric_weights directly alter the relevant dimension score as calculated."""
    # Construct a dataset where cell missingness and row missingness yield distinct subscores
    # 10 rows, 2 columns. Total cells: 20.
    # Row 0: [NaN, NaN] -> 2 cells missing
    # Row 1: [1.0, NaN] -> 1 cell missing
    # Rows 2..9: [1.0, 1.0] -> 0 cells missing
    # Missing cells: 3 / 20 = 15.0%. Linear threshold = 20.0%. Subscore = 1 - (15/20) = 0.25 -> 25.0%
    # Incomplete rows: 2 / 10 = 20.0%. Linear threshold = 20.0%. Subscore = 1 - (20/20) = 0.00 -> 0.0%
    df = pd.DataFrame({
        "col_a": [np.nan, 1.0] + [1.0] * 8,
        "col_b": [np.nan, np.nan] + [1.0] * 8,
    })

    # Baseline with default weights: 70% cell (25.0) + 30% row (0.0) = 17.5
    res_default = ca.analyze(df)
    c_default = res_default.quality_dimensions["completeness"]
    assert math.isclose(c_default.metric_scores["cell_completeness_score"], 25.0, abs_tol=0.01)
    assert math.isclose(c_default.metric_scores["row_completeness_score"], 0.0, abs_tol=0.01)
    assert math.isclose(c_default.score, 17.5, abs_tol=0.01)

    # 100% cell weight, 0% row weight -> score must be exactly 25.0
    res_cell_only = ca.analyze(
        df,
        config=CleanalytixConfig(
            metric_weights={"completeness_cell": 1.0, "completeness_row": 0.0}
        ),
    )
    assert math.isclose(res_cell_only.quality_dimensions["completeness"].score, 25.0, abs_tol=0.01)

    # 0% cell weight, 100% row weight -> score must be exactly 0.0
    res_row_only = ca.analyze(
        df,
        config=CleanalytixConfig(
            metric_weights={"completeness_cell": 0.0, "completeness_row": 1.0}
        ),
    )
    assert math.isclose(res_row_only.quality_dimensions["completeness"].score, 0.0, abs_tol=0.01)

    # 50% cell weight, 50% row weight -> score must be exactly 12.5
    res_equal = ca.analyze(
        df,
        config=CleanalytixConfig(
            metric_weights={"completeness_cell": 0.5, "completeness_row": 0.5}
        ),
    )
    assert math.isclose(res_equal.quality_dimensions["completeness"].score, 12.5, abs_tol=0.01)


def test_scipy_dependency_import():
    """Verify that scipy is installed and required submodules are functional."""
    import scipy
    from scipy import stats
    from scipy.stats import ks_2samp, wasserstein_distance

    ver = tuple(int(x) for x in scipy.__version__.split(".")[:2])
    assert ver >= (1, 10), f"scipy version must be >= 1.10.0, got {scipy.__version__}"


def test_strict_non_mutation_across_entry_points():
    """Verify that input DataFrames are NEVER mutated across all public entry points."""
    df_orig = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": [" Alice ", "Bob", " Charlie", "David "],
        "val": [10.5, np.nan, 30.2, 40.1],
        "date": pd.date_range("2026-01-01", periods=4, freq="D"),
    })
    df_test = df_orig.copy(deep=True)
    baseline_orig = df_orig.copy(deep=True)
    baseline_test = baseline_orig.copy(deep=True)

    # 1. analyze() without baseline
    _ = ca.analyze(df_test)
    pd.testing.assert_frame_equal(df_test, df_orig)

    # 2. analyze() with baseline
    _ = ca.analyze(df_test, baseline_df=baseline_test)
    pd.testing.assert_frame_equal(df_test, df_orig)
    pd.testing.assert_frame_equal(baseline_test, baseline_orig)

    # 3. Run_DQ_Pipeline()
    _ = ca.Run_DQ_Pipeline(["ds"], [baseline_test], new_dataset_list=[df_test], cleaning=False)
    pd.testing.assert_frame_equal(df_test, df_orig)
    pd.testing.assert_frame_equal(baseline_test, baseline_orig)
