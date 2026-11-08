"""Comprehensive Regression Test Suite for Cleanalytix v0.2.0 Phase 2 High-Severity Defect Remediation.

Covers:
- AUD-004: Consistency Rules and pd.NA Boolean Evaluation
- AUD-005: Missing Values and Double Penalties in Rule-Based Validity
- AUD-006: Inconsistent Denominators Across Validity Metrics
- AUD-007: Multi-Column Timeliness Scoring and Metric Inventory
- AUD-008: Identifier Detection False Positives
- AUD-009: Inconsistent Rare-Category Thresholds
- AUD-010: Positive and Negative Infinity Handling
- AUD-011: Metric Inventory Completeness and Scoring Consistency
"""

import numpy as np
import pandas as pd
import pytest

import cleanalytix as cx
from cleanalytix.scoring import (
    DEFAULT_METRIC_WEIGHTS,
    DEFAULT_DIMENSION_WEIGHTS,
    build_metric_inventory,
)
from cleanalytix.types import (
    CleanalytixConfig,
    ConsistencyRule,
    QualityDimension,
    TimelinessConfig,
)


# ============================================================================
# AUD-004: Consistency Rules and pd.NA Boolean Evaluation
# ============================================================================
class TestAUD004ConsistencyNullableNA:
    """Verify cross-column consistency rules safely handle pd.NA and nullable dtypes."""

    def test_aud004_pd_na_series_evaluation(self):
        """Nullable Int64 series with pd.NA should evaluate non-NA rows without crashing or false penalties."""
        df = pd.DataFrame({
            "a": pd.Series([10, 20, 30, 40, None], dtype="Int64"),
            "b": pd.Series([5, 10, 15, 20, 25], dtype="Int64"),
        })
        cfg = CleanalytixConfig(
            consistency_rules=[ConsistencyRule(name="a_gt_b", condition=lambda d: d["a"] > d["b"])]
        )
        res = cx.analyze(df, config=cfg)

        cons_dim = res.quality_dimensions["consistency"]
        assert cons_dim.applicable is True
        assert cons_dim.score == 100.0

        details = cons_dim.details.get("rule_details", {}).get("a_gt_b", {})
        assert details.get("violations") == 0
        assert details.get("violation_percent") == 0.0
        assert details.get("score") == 100.0

    def test_aud004_actual_violations_with_na(self):
        """Eligible non-NA rows should be evaluated, with violations computed strictly over eligible rows."""
        # row 0: 10 > 5 (pass)
        # row 1: 5 > 10 (violation)
        # row 2: 30 > 15 (pass)
        # row 3: None > 20 (NA, exempt)
        df = pd.DataFrame({
            "a": pd.Series([10, 5, 30, None], dtype="Int64"),
            "b": pd.Series([5, 10, 15, 20], dtype="Int64"),
        })
        cfg = CleanalytixConfig(
            consistency_rules=[ConsistencyRule(name="a_gt_b", condition=lambda d: d["a"] > d["b"])]
        )
        res = cx.analyze(df, config=cfg)

        details = res.quality_dimensions["consistency"].details.get("rule_details", {}).get("a_gt_b", {})
        assert details.get("violations") == 1
        # 1 violation out of 3 eligible rows = 33.3333%
        assert abs(details.get("violation_percent") - 33.3333) < 0.01

    def test_aud004_all_na_rows(self):
        """When all evaluated rule outcomes are NA, no violations should be counted."""
        df = pd.DataFrame({
            "a": pd.Series([None, None], dtype="Int64"),
            "b": pd.Series([None, None], dtype="Int64"),
        })
        cfg = CleanalytixConfig(
            consistency_rules=[ConsistencyRule(name="a_gt_b", condition=lambda d: d["a"] > d["b"])]
        )
        res = cx.analyze(df, config=cfg)
        assert res.quality_dimensions["consistency"].score == 100.0


# ============================================================================
# AUD-005: Missing Values and Double Penalties in Rule-Based Validity
# ============================================================================
class TestAUD005RuleBasedErrorsMissingValues:
    """Verify rule_based_errors does not penalize missing data in Validity."""

    def test_aud005_standard_predicate_no_double_penalty(self):
        """Standard lambda predicate (lambda x: x >= 0) should not fail on null values."""
        df = pd.DataFrame({"age": [25.0, 30.0, None, None, None]})
        res = cx.analyze(df, rules={"age": lambda x: x >= 0})

        meta_row = res.metadata[res.metadata["column_name"] == "age"].iloc[0]
        assert meta_row["rule_errors"] == 0
        assert res.column_profile.loc["age", "rule_error_percent"] == 0.0

        # Completeness penalizes the 3 nulls, but Validity remains 100.0
        assert res.quality_dimensions["validity"].score == 100.0
        assert res.quality_dimensions["completeness"].score < 100.0

    def test_aud005_real_rule_errors_detected_on_non_nulls(self):
        """Actual business rule failures on non-null values must be accurately counted."""
        # Non-null values: 25.0 (pass), -5.0 (fail), 130.0 (fail)
        df = pd.DataFrame({"age": [25.0, -5.0, None, 130.0, None]})
        res = cx.analyze(df, rules={"age": lambda x: 0 <= x <= 120})

        meta_row = res.metadata[res.metadata["column_name"] == "age"].iloc[0]
        assert meta_row["rule_errors"] == 2


# ============================================================================
# AUD-006: Inconsistent Denominators Across Validity Metrics
# ============================================================================
class TestAUD006ValidityDenominators:
    """Verify Validity error rates consistently use non_null_count across all outputs."""

    def test_aud006_whitespace_percent_denominator_consistency(self):
        """Whitespace error percentage must use non_null_count in column_profile, metric_inventory, and score_dataset."""
        # 50 non-null values, 1 with untrimmed whitespace -> 1 / 50 = 2.0%
        # 50 missing values -> if total_rows was used, it would report 1.0%
        df = pd.DataFrame({"code": [" A "] + ["B"] * 49 + [None] * 50})
        res = cx.analyze(df)

        col_prof_ws = res.column_profile.loc["code", "whitespace_percent"]
        assert col_prof_ws == 2.0

        ws_inv = res.metric_inventory[
            (res.metric_inventory["metric_name"] == "validity_whitespace")
            & (res.metric_inventory["target_column"] == "code")
        ].iloc[0]
        assert ws_inv["raw_value"] == 2.0

        # With whitespace threshold = 5.0%, ws_pct = 2.0% gives linear score:
        # 1.0 - (2.0 / 5.0) = 0.60 -> 60.0 score
        assert ws_inv["metric_score"] == 60.0
        assert res.scoring_matrix.loc["code", "whitespace_score"] == 60.0

    def test_aud006_type_errors_denominator_consistency(self):
        """Type errors percentage must use non_null_count across outputs."""
        # 4 non-null values: ["1", "2", "3", "bad"], 96 nulls -> 1 / 4 = 25.0%
        df = pd.DataFrame({"num": ["1", "2", "3", "bad"] + [None] * 96})
        res = cx.analyze(df)

        meta_row = res.metadata[res.metadata["column_name"] == "num"].iloc[0]
        assert meta_row["type_errors"] == 1

        col_prof_type = res.column_profile.loc["num", "type_error_percent"]
        assert col_prof_type == 25.0

        type_inv = res.metric_inventory[
            (res.metric_inventory["metric_name"] == "validity_type_errors")
            & (res.metric_inventory["target_column"] == "num")
        ].iloc[0]
        assert type_inv["raw_value"] == 25.0


# ============================================================================
# AUD-007: Multi-Column Timeliness Scoring and Metric Inventory
# ============================================================================
class TestAUD007MultiColumnTimeliness:
    """Verify multi-column timeliness evaluation, metric inventory rows, and column scoring."""

    def test_aud007_stale_and_fresh_columns_both_scored(self):
        """Stale datetime column must receive lower score, reflect in scoring matrix, and emit inventory rows."""
        df = pd.DataFrame({
            "stale_dt": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "fresh_dt": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        })
        cfg = CleanalytixConfig(
            timeliness_config=TimelinessConfig(reference_time="2025-01-03", freshness_days=30.0)
        )
        res = cx.analyze(df, config=cfg)

        time_dim = res.quality_dimensions["timeliness"]
        assert time_dim.applicable is True

        # Freshness subscore reflects the mean of stale (0.0) and fresh (100.0) = 50.0
        assert time_dim.metric_scores["freshness_score"] < 100.0

        # Column profile and column scores must penalize stale_dt
        assert res.column_scores["stale_dt"] < res.column_scores["fresh_dt"]
        assert res.scoring_matrix.loc["stale_dt", "timeliness_score"] < 100.0
        assert res.scoring_matrix.loc["fresh_dt", "timeliness_score"] == 100.0

        # Metric inventory must have distinct rows for both columns
        inv = res.metric_inventory
        stale_fresh_inv = inv[
            (inv["metric_name"] == "timeliness_freshness") & (inv["target_column"] == "stale_dt")
        ]
        assert not stale_fresh_inv.empty
        assert stale_fresh_inv.iloc[0]["raw_value"] > 1000.0  # ~1827 days stale
        assert stale_fresh_inv.iloc[0]["status"] == "WARNING"

        fresh_fresh_inv = inv[
            (inv["metric_name"] == "timeliness_freshness") & (inv["target_column"] == "fresh_dt")
        ]
        assert not fresh_fresh_inv.empty
        assert fresh_fresh_inv.iloc[0]["raw_value"] == 0.0
        assert fresh_fresh_inv.iloc[0]["status"] == "PASS"

        # Check future dates and sequence metrics present
        fut_inv = inv[inv["metric_name"] == "timeliness_future_dates"]
        assert len(fut_inv) == 2
        seq_inv = inv[inv["metric_name"] == "timeliness_sequence"]
        assert len(seq_inv) == 2


# ============================================================================
# AUD-008: Identifier Detection False Positives
# ============================================================================
class TestAUD008IdentifierDetectionFalsePositives:
    """Verify normal columns ('width', 'paid', 'country_code') are not treated as primary keys."""

    def test_aud008_normal_words_not_flagged_as_identifiers(self):
        """Columns containing 'id', 'code' as substrings must not be classified as primary identifiers."""
        df = pd.DataFrame({
            "width": [10.0, 10.0, 20.0, 20.0, 10.0],
            "paid": ["yes", "yes", "no", "yes", "no"],
            "country_code": ["US", "US", "UK", "US", "UK"],
        })
        res = cx.analyze(df)

        uniq_dim = res.quality_dimensions["uniqueness"]
        assert uniq_dim.details.get("has_identifiers") is False

        # Uniqueness score is determined solely by duplicate rows, not penalized by 60% duplicate IDs
        inv = res.metric_inventory
        id_inv = inv[inv["metric_name"] == "uniqueness_identifiers"].iloc[0]
        assert id_inv["status"] == "NOT_APPLICABLE"
        assert not bool(id_inv["contributes_to_score"])

    def test_aud008_real_identifier_correctly_recognized(self):
        """Columns with strong identifier names and high uniqueness must be identified."""
        df = pd.DataFrame({
            "customer_id": [f"CUST_{i}" for i in range(100)],
            "value": [1.0] * 100,
        })
        res = cx.analyze(df)
        assert res.quality_dimensions["uniqueness"].details.get("has_identifiers") is True


# ============================================================================
# AUD-009: Inconsistent Rare-Category Thresholds
# ============================================================================
class TestAUD009RareCategoryThresholds:
    """Verify rare-category threshold is unified to < 1.0% across all components."""

    def test_aud009_balanced_categories_between_1_and_5_percent(self):
        """25 categories at 4.0% each must not be flagged as rare categories."""
        df = pd.DataFrame({"cat": [f"c{i}" for i in range(25) for _ in range(4)]})
        res = cx.analyze(df)

        # In metadata, rare_category_percent must be 0.0
        meta_row = res.metadata[res.metadata["column_name"] == "cat"].iloc[0]
        assert meta_row["rare_category_percent"] == 0.0

        # In statistical profile, rare_categories_ratio must be 0.0
        stat_row = res.statistical_profile.loc["cat"]
        assert stat_row["rare_categories_ratio"] == 0.0

        # No recommendations for rare categories should be triggered
        rare_recs = [r for r in res.recommendations if r.issue_name == "Rare Categories"]
        assert len(rare_recs) == 0

    def test_aud009_actual_rare_categories_under_1_percent(self):
        """Categories with frequency < 1.0% must be detected and documented consistently."""
        # 1 dominant category (195 rows = 97.5%) + 5 singletons (1 row each = 0.5%)
        # total rows = 200. Each singleton frequency = 0.5% < 1.0%.
        df = pd.DataFrame({"cat": ["dominant"] * 195 + [f"rare_{i}" for i in range(5)]})
        res = cx.analyze(df)

        meta_row = res.metadata[res.metadata["column_name"] == "cat"].iloc[0]
        assert meta_row["rare_category_percent"] == 2.5  # 5 * 0.5% = 2.5%

        stat_row = res.statistical_profile.loc["cat"]
        assert round(stat_row["rare_categories_ratio"] * 100.0, 1) == 2.5


# ============================================================================
# AUD-010: Positive and Negative Infinity Handling
# ============================================================================
class TestAUD010InfinityHandling:
    """Verify np.inf and -np.inf in numeric columns are counted as type errors, anomalies, and penalized."""

    def test_aud010_infinite_values_flagged_in_validity_and_anomalies(self):
        """Infinite values must be counted as type errors, lower Validity score, and appear in anomalies."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.inf, -np.inf, 3.0]})
        res = cx.analyze(df)

        meta_row = res.metadata[res.metadata["column_name"] == "a"].iloc[0]
        assert meta_row["type_errors"] == 2

        # 2 type errors out of 5 non-null rows = 40% error rate
        assert res.quality_dimensions["validity"].score < 100.0

        # Recorded in statistical profile
        assert res.statistical_profile.loc["a", "infinite_count"] == 2

        # Appears in anomalies DataFrame
        assert not res.anomalies.empty
        anom_rows = res.anomalies[res.anomalies["column"] == "a"]
        assert not anom_rows.empty
        assert any("type" in str(msg).lower() for msg in anom_rows["message"])

        # Generates Type Errors recommendation
        type_recs = [r for r in res.recommendations if r.issue_name == "Type Errors"]
        assert len(type_recs) >= 1


# ============================================================================
# AUD-011: Metric Inventory Completeness and Scoring Consistency
# ============================================================================
class TestAUD011MetricInventoryAndWeights:
    """Verify DEFAULT_METRIC_WEIGHTS, metric inventory completeness, and weight normalization."""

    def test_aud011_default_metric_weights_normalized(self):
        """Default metric weights within Validity must sum to 1.00."""
        val_keys = ["validity_type_errors", "validity_whitespace", "validity_rules", "validity_outliers"]
        val_sum = sum(DEFAULT_METRIC_WEIGHTS[k] for k in val_keys)
        assert abs(val_sum - 1.00) < 1e-6

        time_keys = ["timeliness_freshness", "timeliness_future_dates", "timeliness_sequence"]
        time_sum = sum(DEFAULT_METRIC_WEIGHTS[k] for k in time_keys)
        assert abs(time_sum - 1.00) < 1e-6

        stab_keys = ["stability_psi", "stability_jsd"]
        stab_sum = sum(DEFAULT_METRIC_WEIGHTS[k] for k in stab_keys)
        assert abs(stab_sum - 1.00) < 1e-6

    def test_aud011_all_scored_metrics_present_in_inventory(self):
        """Every metric that participates in scoring must be present in metric_inventory."""
        base_df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(100)],
            "val": [float(i) for i in range(100)],
            "ts": pd.date_range("2025-01-01", periods=100, freq="D"),
            "txt": [" category "] * 100,
        })
        prod_df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(100)],
            "val": [float(i + 10) for i in range(100)],
            "ts": pd.date_range("2025-01-01", periods=100, freq="D"),
            "txt": [" category "] * 100,
        })
        cfg = CleanalytixConfig(
            rules={"val": lambda x: x >= 0},
            consistency_rules=[ConsistencyRule(name="val_pos", condition=lambda d: d["val"] >= 0)],
            timeliness_config=TimelinessConfig(reference_time="2025-05-01", freshness_days=30.0),
        )
        res = cx.analyze(prod_df, baseline_df=base_df, config=cfg)

        inv = res.metric_inventory
        metric_names = set(inv["metric_name"].tolist())

        # Check all 6 dimensions' participating metrics are present
        expected_metrics = {
            "completeness_row",
            "completeness_cell",
            "validity_type_errors",
            "validity_whitespace",
            "validity_rules",
            "validity_outliers",
            "uniqueness_duplicate_rows",
            "uniqueness_identifiers",
            "consistency_rules",
            "timeliness_freshness",
            "timeliness_future_dates",
            "timeliness_sequence",
            "stability_psi",
            "stability_jsd",
        }
        for em in expected_metrics:
            assert em in metric_names, f"Expected metric '{em}' missing from metric_inventory"

    def test_aud011_effective_weights_sum_to_one(self):
        """Sum of effective weights across all scored metrics in metric_inventory must equal 1.0000."""
        base_df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(100)],
            "val": [float(i) for i in range(100)],
            "ts": pd.date_range("2025-01-01", periods=100, freq="D"),
            "cat": ["A"] * 50 + ["B"] * 50,
        })
        prod_df = pd.DataFrame({
            "id": [f"ID_{i}" for i in range(100)],
            "val": [float(i) for i in range(100)],
            "ts": pd.date_range("2025-01-01", periods=100, freq="D"),
            "cat": ["A"] * 50 + ["B"] * 50,
        })
        cfg = CleanalytixConfig(
            rules={"val": lambda x: x >= 0},
            consistency_rules=[ConsistencyRule(name="val_pos", condition=lambda d: d["val"] >= 0)],
            timeliness_config=TimelinessConfig(reference_time="2025-05-01", freshness_days=30.0),
        )
        res = cx.analyze(prod_df, baseline_df=base_df, config=cfg)

        inv = res.metric_inventory
        scored_metrics = inv[inv["contributes_to_score"] == True]
        total_effective_weight = float(scored_metrics["effective_weight"].sum())

        # Total effective weight should sum to 1.0000 (tolerance: 0.01 for rounding)
        assert abs(total_effective_weight - 1.0000) < 0.02
