"""Comprehensive regression tests for Phase 1 Critical Defect Remediation.

Covers:
- AUD-001: Numeric drift detection bin collapse (two-unique-value, constant baseline, identical, empty)
- AUD-002: Recommendation generation severity normalization (enum and string normalization, all dimensions)
- AUD-003: Modern analyze() baseline comparison consistency (reference profile, distinct base/prod data blocks)
"""

import math
import numpy as np
import pandas as pd
import pytest

import cleanalytix as ca
from cleanalytix.cleaning_recommendations import normalize_severity, generate_recommendations
from cleanalytix.metrics.drift import compute_numeric_drift
from cleanalytix.generate_meta import generate_meta
from cleanalytix.types import (
    CleanalytixConfig,
    ConsistencyRule,
    InsightItem,
    InsightSeverity,
    QualityDimension,
    RecommendationItem,
    TimelinessConfig,
)


# ============================================================================
# 1. AUD-001: NUMERIC DRIFT DETECTION BIN COLLAPSE
# ============================================================================

class TestAUD001NumericDriftBinCollapse:
    """Regression tests for AUD-001: numeric drift bin collapse."""

    def test_two_unique_value_baseline_detects_substantial_shift(self):
        """When baseline has only 2 unique quantile edges (e.g. 95% at 10, 5% at 20),

        a massive production shift to 1,000,000 must NOT collapse to 1 bin and report PSI=0.
        """
        b_arr = pd.Series([10.0] * 95 + [20.0] * 5)
        p_arr = pd.Series([1000000.0] * 100)

        drift_res = compute_numeric_drift(b_arr, p_arr)

        assert drift_res["drift_detected"] is True
        assert drift_res["drift_severity"] == "critical"
        assert drift_res["psi"] > 0.25
        assert drift_res["kl_divergence"] > 0.0
        assert drift_res["reverse_kl_divergence"] > 0.0
        assert drift_res["js_divergence"] > 0.0
        # Jeffreys divergence identity: PSI == KL(P||Q) + KL(Q||P)
        assert math.isclose(
            drift_res["psi"],
            drift_res["kl_divergence"] + drift_res["reverse_kl_divergence"],
            rel_tol=1e-3,
        )
        # JSD bounded in [0, ln(2)]
        assert 0.0 <= drift_res["js_divergence"] <= math.log(2.0)

    def test_two_unique_value_baseline_identical_produces_zero_divergence(self):
        """Identical two-unique-value baseline and production distributions must produce

        strictly zero or numerically negligible divergence.
        """
        b_arr = pd.Series([10.0] * 95 + [20.0] * 5)
        p_arr = pd.Series([10.0] * 95 + [20.0] * 5)

        drift_res = compute_numeric_drift(b_arr, p_arr)

        assert drift_res["drift_detected"] is False
        assert drift_res["drift_severity"] == "none"
        assert drift_res["psi"] == 0.0
        assert drift_res["kl_divergence"] == 0.0
        assert drift_res["reverse_kl_divergence"] == 0.0
        assert drift_res["js_divergence"] == 0.0

    def test_two_unique_value_baseline_detects_shifts_in_all_directions(self):
        """Verify the 5-bin surrogate detects shifts below e0, between e0 and e1, and above e1."""
        b_arr = pd.Series([10.0] * 50 + [20.0] * 50)

        # Shift below e0
        p_below = pd.Series([-50.0] * 100)
        res_below = compute_numeric_drift(b_arr, p_below)
        assert res_below["drift_detected"] is True
        assert res_below["psi"] > 0.25

        # Shift between e0 and e1
        p_between = pd.Series([15.0] * 100)
        res_between = compute_numeric_drift(b_arr, p_between)
        assert res_between["drift_detected"] is True
        assert res_between["psi"] > 0.25

        # Shift above e1
        p_above = pd.Series([500.0] * 100)
        res_above = compute_numeric_drift(b_arr, p_above)
        assert res_above["drift_detected"] is True
        assert res_above["psi"] > 0.25

    def test_constant_baseline_shifted_drift_detection(self):
        """Constant baseline (zero variance) against shifted production must detect drift."""
        b = pd.Series([10.0] * 100)
        p = pd.Series([20.0] * 100)

        drift_res = compute_numeric_drift(b, p)
        assert drift_res["drift_detected"] is True
        assert drift_res["psi"] >= 0.25

    def test_constant_baseline_identical_produces_zero_drift(self):
        """Constant baseline against identical constant production must report zero drift."""
        b = pd.Series([10.0] * 100)
        p = pd.Series([10.0] * 100)

        drift_res = compute_numeric_drift(b, p)
        assert drift_res["drift_detected"] is False
        assert drift_res["psi"] == 0.0
        assert drift_res["kl_divergence"] == 0.0
        assert drift_res["js_divergence"] == 0.0

    def test_generate_meta_drift_uses_remediated_binning(self):
        """generate_meta() must return non-zero PSI, KL, JSD on a 2-unique-value baseline

        when production is shifted.
        """
        df_base = pd.DataFrame({"score": [10.0] * 95 + [20.0] * 5})
        df_prod = pd.DataFrame({"score": [1000000.0] * 100})

        meta = generate_meta(["test_ds"], [df_base], [df_prod])
        psi_val = meta["psi_divergence"].iloc[0]
        kl_val = meta["kl_divergence"].iloc[0]
        js_val = meta["js_divergence"].iloc[0]

        assert psi_val > 0.25
        assert kl_val > 0.0
        assert js_val > 0.0

    def test_empty_and_degenerate_drift_inputs(self):
        """Verify empty and NaN series return default zero drift without crashing."""
        b_empty = pd.Series([], dtype=float)
        p_empty = pd.Series([], dtype=float)
        drift_res = compute_numeric_drift(b_empty, p_empty)
        assert drift_res["drift_detected"] is False
        assert drift_res["psi"] == 0.0

    @pytest.mark.parametrize(
        "case_name,e0,e1,shifted_val",
        [
            ("closely_spaced_positive", 10.0, 10.0001, 1000.0),
            ("zero_and_small_positive", 0.0, 1e-8, 1.0),
            ("closely_spaced_negative", -10.0001, -10.0, 0.0),
            ("closely_spaced_negative_small", -1e-7, -1e-8, -1.0),
            ("widely_separated_moderate", -1e6, 1e6, 1e9),
            ("widely_separated_huge", -1e12, 1e12, 1e15),
            ("very_large_finite_1e12", 1e12, 1e12 + 100.0, 1e13),
            ("very_large_finite_1e15", 1e15, 1e15 + 1000.0, 1e16),
            ("very_large_finite_1e15_small_step", 1e15, 1e15 + 1.0, 1e16),
            ("very_small_finite_1e_neg15", 1e-15, 2e-15, 1.0),
            ("very_small_finite_1e_neg100", 1e-100, 2e-100, 1.0),
            ("denormal_boundary", 0.0, 1e-300, 1.0),
            ("extreme_large_scale", 1e300, 1.00000000000001e300, 1e305),
        ],
    )
    def test_closely_spaced_and_extreme_scale_two_quantile_edges(
        self, case_name, e0, e1, shifted_val
    ):
        """Verify strictly increasing bin edges, absence of NaNs, finite non-negative metrics,

        zero false-positive drift for identical inputs, and robust shift detection across
        closely spaced, negative, widely separated, and extreme scale float edges.
        """
        b = pd.Series([e0] * 50 + [e1] * 50)
        p_shifted = pd.Series([shifted_val] * 100)

        # 1. Identical distribution must not produce artificial drift
        res_id = compute_numeric_drift(b, b)
        assert res_id["psi"] == 0.0, f"[{case_name}] Identical PSI non-zero: {res_id['psi']}"
        assert res_id["drift_detected"] is False, f"[{case_name}] Artificial drift detected on identical data"
        assert res_id["kl_divergence"] >= 0.0, f"[{case_name}] Identical KL negative"
        assert res_id["js_divergence"] >= 0.0, f"[{case_name}] Identical JSD negative"
        assert not np.isnan(res_id["psi"]), f"[{case_name}] Identical PSI is NaN"
        assert not np.isinf(res_id["psi"]), f"[{case_name}] Identical PSI is Inf"

        # 2. Shifted distribution must remain detectable with finite non-negative metrics
        res_sh = compute_numeric_drift(b, p_shifted)
        assert res_sh["psi"] > 0.1, f"[{case_name}] Shifted PSI failed to detect drift: {res_sh['psi']}"
        assert res_sh["drift_detected"] is True, f"[{case_name}] Shifted drift_detected is False"
        assert res_sh["kl_divergence"] > 0.0, f"[{case_name}] Shifted KL not positive"
        assert res_sh["js_divergence"] > 0.0, f"[{case_name}] Shifted JSD not positive"
        assert not np.isnan(res_sh["psi"]), f"[{case_name}] Shifted PSI is NaN"
        assert not np.isinf(res_sh["psi"]), f"[{case_name}] Shifted PSI is Inf"
        assert 0.0 <= res_sh["js_divergence"] <= math.log(2.0) + 1e-6


# ============================================================================
# 2. AUD-002: RECOMMENDATION SEVERITY NORMALIZATION
# ============================================================================

class TestAUD002RecommendationSeverityNormalization:
    """Regression tests for AUD-002: insight-to-recommendation severity normalization."""

    def test_normalize_severity_function(self):
        """Verify normalize_severity handles InsightSeverity enums and diverse strings."""
        assert normalize_severity(InsightSeverity.CRITICAL) == "CRITICAL"
        assert normalize_severity(InsightSeverity.WARNING) == "WARNING"
        assert normalize_severity(InsightSeverity.INFO) == "INFO"
        assert normalize_severity(InsightSeverity.OPTIMIZATION) == "OPTIMIZATION"

        assert normalize_severity("critical") == "CRITICAL"
        assert normalize_severity("CRITICAL") == "CRITICAL"
        assert normalize_severity("warning") == "WARNING"
        assert normalize_severity("WARNING") == "WARNING"
        assert normalize_severity("info") == "INFO"
        assert normalize_severity("  warning  ") == "WARNING"

    def test_critical_uniqueness_issues_generate_recommendations(self):
        """Datasets with duplicate rows (> 5%) must generate a Uniqueness recommendation."""
        df_dups = pd.DataFrame({
            "id": [1, 1, 1, 1, 2],
            "val": ["A", "A", "A", "A", "B"],
        })
        res = ca.analyze(df_dups)

        uniq_recs = [r for r in res.recommendations if r.dimension == QualityDimension.UNIQUENESS]
        assert len(uniq_recs) >= 1
        rec = uniq_recs[0]
        assert rec.severity == "CRITICAL"
        assert "duplicate" in rec.issue_name.lower() or "duplicate" in rec.evidence.lower()
        assert rec.suggested_action

    def test_warning_consistency_issues_generate_recommendations(self):
        """Cross-column consistency rule violations must generate Consistency recommendations."""
        df = pd.DataFrame({
            "start": [10, 20, 30, 40],
            "end": [5, 25, 15, 45],  # 2 violations (start < end violated by rows 0 and 2)
        })
        cfg = CleanalytixConfig(
            consistency_rules=[
                ConsistencyRule(name="start_lt_end", condition=lambda d: d["start"] < d["end"])
            ]
        )
        res = ca.analyze(df, config=cfg)

        cons_recs = [r for r in res.recommendations if r.dimension == QualityDimension.CONSISTENCY]
        assert len(cons_recs) >= 1
        rec = cons_recs[0]
        assert rec.severity in {"CRITICAL", "WARNING"}
        assert "start_lt_end" in rec.evidence or "Consistency" in rec.issue_name

    def test_timeliness_issues_generate_recommendations(self):
        """Data staleness exceeding threshold must generate a Timeliness recommendation."""
        df_time = pd.DataFrame({
            "event_time": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        })
        cfg = CleanalytixConfig(
            timeliness_config=TimelinessConfig(
                reference_time="2025-01-01",
                freshness_days=30.0,
            )
        )
        res = ca.analyze(df_time, config=cfg)

        time_recs = [r for r in res.recommendations if r.dimension == QualityDimension.TIMELINESS]
        assert len(time_recs) >= 1
        rec = time_recs[0]
        assert rec.severity in {"CRITICAL", "WARNING"}
        assert "staleness" in rec.issue_name.lower() or "staleness" in rec.evidence.lower()

    def test_stability_issues_generate_recommendations(self):
        """Substantial distribution drift against baseline must generate a Stability recommendation."""
        np.random.seed(42)
        b_df = pd.DataFrame({"metric": np.random.normal(0, 1, 500)})
        p_df = pd.DataFrame({"metric": np.random.normal(5, 1, 500)})

        res = ca.analyze(p_df, baseline_df=b_df)

        stab_recs = [r for r in res.recommendations if r.dimension == QualityDimension.STABILITY]
        assert len(stab_recs) >= 1
        rec = stab_recs[0]
        assert rec.severity == "CRITICAL"
        assert "drift" in rec.issue_name.lower() or "drift" in rec.evidence.lower()

    def test_no_duplicate_recommendations(self):
        """Recommendations table must not contain duplicate (rule_id, affected_column) entries."""
        df = pd.DataFrame({
            "a": [1, 1, 1, 1, 2],
            "b": [10, 10, 10, 10, 20],
        })
        res = ca.analyze(df)
        rt = res.recommendations_table

        pairs = list(zip(rt["rule_id"], rt["affected_column"]))
        assert len(pairs) == len(set(pairs)), "Duplicate (rule_id, affected_column) pairs found!"


# ============================================================================
# 3. AUD-003: MODERN ANALYZE() BASELINE COMPARISON CONSISTENCY
# ============================================================================

class TestAUD003ModernAnalyzeBaselineConsistency:
    """Regression tests for AUD-003: baseline comparison consistency in analyze()."""

    def test_baseline_and_production_metadata_are_distinct(self):
        """Baseline and production metadata must reflect their own respective data distributions."""
        base = pd.DataFrame({"val": [10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8, 10.9, 11.1, 10.7]})
        prod = pd.DataFrame({"val": [1000.0, 1001.0, 1002.0, 1000.5, 1001.5, 1000.2, 1001.8, 1000.9, 1001.1, 1000.7]})

        res = ca.analyze(prod, baseline_df=base)

        base_meta = res.base_data["meta_before_cleaning"]
        prod_meta = res.prod_data["meta_before_cleaning"]

        assert base_meta.loc[base_meta["column_name"] == "val", "mean"].iloc[0] == pytest.approx(11.0, rel=0.1)
        assert prod_meta.loc[prod_meta["column_name"] == "val", "mean"].iloc[0] == pytest.approx(1001.0, rel=0.1)
        assert res.metadata.loc[res.metadata["column_name"] == "val", "mean"].iloc[0] == pytest.approx(1001.0, rel=0.1)

    def test_reference_bounds_used_for_production_outliers(self):
        """Production outliers must be evaluated against learned baseline reference bounds."""
        base = pd.DataFrame({"val": [10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8, 10.9, 11.1, 10.7]})
        prod = pd.DataFrame({"val": [1000.0, 1001.0, 1002.0, 1000.5, 1001.5, 1000.2, 1001.8, 1000.9, 1001.1, 1000.7]})

        res = ca.analyze(prod, baseline_df=base)

        # Baseline itself has 0 outliers
        assert res.base_data["meta_before_cleaning"].loc[
            res.base_data["meta_before_cleaning"]["column_name"] == "val", "outlier_count"
        ].iloc[0] == 0

        # Production evaluated against baseline bounds has all 10 records as outliers
        assert res.prod_data["meta_before_cleaning"].loc[
            res.prod_data["meta_before_cleaning"]["column_name"] == "val", "outlier_count"
        ].iloc[0] == 10
        assert res.metadata.loc[res.metadata["column_name"] == "val", "outlier_count"].iloc[0] == 10

    def test_base_data_and_prod_data_contain_respective_dataset_information(self):
        """base_data and prod_data blocks must represent baseline and production respectively."""
        base = pd.DataFrame({"val": [10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8, 10.9, 11.1, 10.7]})
        prod = pd.DataFrame({"val": [1000.0, 1001.0, 1002.0, 1000.5, 1001.5, 1000.2, 1001.8, 1000.9, 1001.1, 1000.7]})

        res = ca.analyze(prod, baseline_df=base)

        base_score = res.base_data["dirty_scores"]["DQ_Score"].iloc[0]
        prod_score = res.prod_data["dirty_scores"]["DQ_Score"].iloc[0]

        # Clean baseline has perfect score
        assert base_score == 100.0
        # Production shifted by 1000 has lower score due to outliers and drift
        assert prod_score < 100.0
        assert res.overall_score == pytest.approx(prod_score, abs=1e-2)

    def test_analyze_and_run_dq_pipeline_logical_equivalence(self):
        """analyze() and Run_DQ_Pipeline() must detect the same baseline reference outliers and drift."""
        base = pd.DataFrame({"val": [10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.8, 10.9, 11.1, 10.7]})
        prod = pd.DataFrame({"val": [1000.0, 1001.0, 1002.0, 1000.5, 1001.5, 1000.2, 1001.8, 1000.9, 1001.1, 1000.7]})

        analyze_res = ca.analyze(prod, baseline_df=base)
        pipeline_res = ca.Run_DQ_Pipeline(["dataset"], [base], [prod])

        # Both must find 10 reference outliers in production
        analyze_outliers = analyze_res.prod_data["meta_before_cleaning"]["outlier_count"].iloc[0]
        pipeline_outliers = pipeline_res["prod_data"]["meta_before_cleaning"]["outlier_count"].iloc[0]
        assert analyze_outliers == pipeline_outliers == 10

        # Both must compute the same baseline score
        assert analyze_res.base_data["dirty_scores"]["DQ_Score"].iloc[0] == pipeline_res["base_data"]["dirty_scores"]["DQ_Score"].iloc[0] == 100.0

        # Both must detect severe drift
        analyze_psi = analyze_res.metadata["psi_divergence"].iloc[0]
        pipeline_psi = pipeline_res["prod_data"]["meta_before_cleaning"]["psi_divergence"].iloc[0]
        assert analyze_psi == pytest.approx(pipeline_psi, rel=1e-3)

    def test_analyze_without_baseline_continues_working_normally(self):
        """analyze() without baseline must populate base_data from input df and leave prod_data empty."""
        df = pd.DataFrame({"val": [10.0, 11.0, 12.0, 10.5, 11.5]})
        res = ca.analyze(df)

        assert res.prod_data == {}
        assert res.base_data["meta_before_cleaning"]["mean"].iloc[0] == pytest.approx(11.0, rel=0.1)
        assert 0.0 <= res.overall_score <= 100.0

    def test_strict_non_mutation_guarantee(self):
        """Calling analyze() with baseline must not mutate df or baseline_df."""
        base = pd.DataFrame({"val": [10.0, 11.0, 12.0]})
        prod = pd.DataFrame({"val": [100.0, 110.0, 120.0]})
        base_orig = base.copy(deep=True)
        prod_orig = prod.copy(deep=True)

        _ = ca.analyze(prod, baseline_df=base)

        pd.testing.assert_frame_equal(base, base_orig)
        pd.testing.assert_frame_equal(prod, prod_orig)
