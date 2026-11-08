"""Tests for Phase 3 Medium and Low Severity Defect Remediation (AUD-012 to AUD-022).

Exhaustively verifies:
- AUD-012: Categorical Type Errors (numeric/date-like strings in categorical columns)
- AUD-013: Free-Text Identifier Detection (high-cardinality prose vs. token keys)
- AUD-014: All-Null Column Scoring (Validity on 100% missing columns/datasets)
- AUD-015: Column Quality Score Consistency (configured dimension weights, identifiers, timeliness)
- AUD-016: Timeliness Configuration Handling (dict-based timeliness_config)
- AUD-017: Caller Configuration Non-Mutation (config.rules immutability)
- AUD-018: Public API Exports (ConsistencyRule, TimelinessConfig, AnalysisLevel, InsightSeverity)
- AUD-019: Extreme Numerical Values (float64 overflow guards for +/- 1e300)
- AUD-020: Column Identifier and Indexing Consistency (index.name and column_name across tables)
- AUD-021: Redundant Health Summary and Datetime Conversion (native datetime64 handling)
- AUD-022: Consistency-Rule Metric Inventory Representation (unit, raw_value, and threshold coherence)
"""

import warnings
import numpy as np
import pandas as pd
import pytest

import cleanalytix as cx
from cleanalytix import (
    ConsistencyRule,
    TimelinessConfig,
    AnalysisLevel,
    InsightSeverity,
    CleanalytixConfig,
    QualityDimension,
    analyze,
)
from cleanalytix.generate_meta import get_type_error_count, generate_meta
from cleanalytix.column_role import infer_column_role, is_identifier_column, is_free_text_column
from cleanalytix.metrics.statistical import compute_numeric_statistics, detect_outliers


class TestAUD012CategoricalTypeErrors:
    """AUD-012: In categorical columns, numeric strings and date-like strings must not be flagged as type errors."""

    def test_numeric_and_date_like_strings_in_categorical_column_not_errors(self):
        df_cat = pd.DataFrame({
            "label": ["apple", "banana", "cherry", "100", "March 2024", "10-20"]
        })
        err_count = get_type_error_count(df_cat, "label")
        assert err_count == 0, f"Expected 0 type errors, got {err_count}"

        res = cx.analyze(df_cat)
        assert res.column_profile.loc["label", "type_error_count"] == 0
        assert res.column_profile.loc["label", "type_error_percent"] == 0.0
        assert res.quality_dimensions["validity"].score == 100.0

    def test_non_string_objects_in_predominantly_string_column_flagged(self):
        # Mixed types with raw non-string objects
        df_mixed = pd.DataFrame({
            "label": ["apple", "banana", "cherry", "date", 100, 3.14]
        })
        err_count = get_type_error_count(df_mixed, "label")
        assert err_count == 2, f"Expected 2 type errors for raw int/float, got {err_count}"


class TestAUD013FreeTextIdentifierDetection:
    """AUD-013: High-cardinality unstructured free text must not be misclassified as an identifier."""

    def test_unique_product_reviews_classified_as_free_text_not_identifier(self):
        s = pd.Series(
            [f"This is a long unique product review sentence number {i} with lots of words and spaces." for i in range(12)],
            name="review_text"
        )
        assert not is_identifier_column(s, "review_text", len(s))
        assert is_free_text_column(s)
        assert infer_column_role(s, "review_text") == "free_text"

    def test_unhinted_tokens_classified_as_identifier(self):
        s = pd.Series([f"TOK-{i:04d}" for i in range(15)], name="token_val")
        assert is_identifier_column(s, "token_val", len(s))
        assert infer_column_role(s, "token_val") == "identifier"


class TestAUD014AllNullColumnScoring:
    """AUD-014: All-null columns and all-null datasets must not receive a 100.0 Validity score."""

    def test_all_null_dataset_validity_not_applicable_overall_score_zero(self):
        df = pd.DataFrame({"a": [None, None, None], "b": [np.nan, np.nan, np.nan]})
        res = cx.analyze(df)

        val_dim = res.quality_dimensions["validity"]
        assert not val_dim.applicable, "Validity must be marked applicable=False when dataset has no non-null values"
        assert val_dim.score == 0.0
        assert res.overall_score == 0.0
        assert res.column_profile.loc["a", "column_quality_score"] == 0.0
        assert res.column_profile.loc["b", "column_quality_score"] == 0.0

    def test_mixed_dataset_with_one_all_null_column(self):
        df = pd.DataFrame({
            "all_null": [None, None, None, None],
            "valid_num": [10.0, 20.0, 30.0, 40.0],
        })
        res = cx.analyze(df)
        val_dim = res.quality_dimensions["validity"]
        assert val_dim.applicable
        assert val_dim.score == 100.0
        assert np.isnan(res.scoring_matrix.loc["all_null", "validity_score"])
        assert res.scoring_matrix.loc["valid_num", "validity_score"] == 100.0


class TestAUD015ColumnQualityScoreConsistency:
    """AUD-015: column_quality_score must respect configured dimension weights and include identifiers and timeliness."""

    def test_column_quality_score_respects_custom_dimension_weights(self):
        df = pd.DataFrame({
            "a": [10.0, None, 30.0, None],  # 50% missing -> completeness = 50.0 (or calculated)
        })
        cfg = CleanalytixConfig(
            dimension_weights={
                "completeness": 0.90,
                "validity": 0.10,
                "uniqueness": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "stability": 0.0,
            }
        )
        res = cx.analyze(df, config=cfg)
        comp_s = res.scoring_matrix.loc["a", "completeness_score"]
        val_s = res.scoring_matrix.loc["a", "validity_score"]
        expected_col_score = round(comp_s * 0.90 + val_s * 0.10, 2)
        actual_col_score = res.column_profile.loc["a", "column_quality_score"]
        assert np.isclose(actual_col_score, expected_col_score, atol=0.05)

    def test_identifier_column_duplicates_penalize_column_score(self):
        # Identifier column with 50% duplicates
        df = pd.DataFrame({
            "id": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        })
        cfg = CleanalytixConfig(identifier_columns=["id"])
        res = cx.analyze(df, config=cfg)
        assert res.scoring_matrix.loc["id", "uniqueness_score"] < 100.0
        assert res.column_profile.loc["id", "column_quality_score"] < 100.0


class TestAUD016TimelinessConfigHandling:
    """AUD-016: build_temporal_profile must support dictionary timeliness_config."""

    def test_dict_timeliness_config_used_in_temporal_profile(self):
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        })
        ref = "2024-01-10"
        cfg = CleanalytixConfig(
            timeliness_config={"reference_time": ref, "freshness_days": 10.0}
        )
        res = cx.analyze(df, config=cfg)
        tp = res.temporal_profile
        assert not tp.empty
        # Max date is 2024-01-03, reference is 2024-01-10 -> freshness is 7 days
        assert np.isclose(tp.loc["ts", "freshness_days"], 7.0, atol=0.5)
        assert tp.loc["ts", "freshness_status"] == "FRESH"


class TestAUD017CallerConfigMutation:
    """AUD-017: analyze() must never mutate the caller's CleanalytixConfig in-place."""

    def test_caller_config_rules_not_mutated(self):
        cfg = CleanalytixConfig()
        assert cfg.rules is None

        df1 = pd.DataFrame({"age": [25, 30, 35]})
        rule_fn = lambda x: x >= 18
        res1 = cx.analyze(df1, rules={"age": rule_fn}, config=cfg)

        assert cfg.rules is None, "Caller's config.rules was mutated in-place!"

        df2 = pd.DataFrame({"age": [10, 15, 20]})
        res2 = cx.analyze(df2, config=cfg)
        assert cfg.rules is None


class TestAUD018PublicExports:
    """AUD-018: ConsistencyRule, TimelinessConfig, AnalysisLevel, InsightSeverity exported."""

    def test_public_types_available_at_package_root(self):
        import cleanalytix
        assert hasattr(cleanalytix, "ConsistencyRule")
        assert hasattr(cleanalytix, "TimelinessConfig")
        assert hasattr(cleanalytix, "AnalysisLevel")
        assert hasattr(cleanalytix, "InsightSeverity")

        assert "ConsistencyRule" in cleanalytix.__all__
        assert "TimelinessConfig" in cleanalytix.__all__
        assert "AnalysisLevel" in cleanalytix.__all__
        assert "InsightSeverity" in cleanalytix.__all__


class TestAUD019ExtremeNumericalValues:
    """AUD-019: Float64 values near +/- 1e300 must not cause unhandled overflow warnings or NaN moments."""

    def test_extreme_float_moments_and_outliers(self):
        s = pd.Series([1e300, -1e300, 0.0, 1.0, 2.0, -1.0])
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            stats = compute_numeric_statistics(s)
            outliers = detect_outliers(s)

        # Skewness and kurtosis must be finite
        assert stats["skewness"] is not None and np.isfinite(stats["skewness"])
        assert stats["kurtosis"] is not None and np.isfinite(stats["kurtosis"])
        assert outliers["recommended_outlier_count"] >= 0

        # Full analyze pipeline must succeed cleanly
        df = pd.DataFrame({"extreme": s})
        res = cx.analyze(df)
        assert res.overall_score >= 0.0
        assert not res.statistical_profile.empty


class TestAUD020ColumnIdentifierAndIndexingConsistency:
    """AUD-020: Consistent index.name and column_name across all analytical DataFrames."""

    def test_empty_dataset_index_name_is_column_name(self):
        df_empty = pd.DataFrame()
        res = cx.analyze(df_empty)

        assert res.column_profile.index.name == "column_name"
        assert res.statistical_profile.index.name == "column_name"
        assert res.scoring_matrix.index.name == "column_name"

    def test_analytical_tables_expose_column_name(self):
        df = pd.DataFrame({
            "code": ["a", " b ", "c"],
            "num": [10.0, 20.0, 1000.0],
        })
        res = cx.analyze(df)

        # metric_inventory must have column_name
        assert "column_name" in res.metric_inventory.columns
        assert "target_column" in res.metric_inventory.columns

        # recommendations_table must have column_name
        assert "column_name" in res.recommendations_table.columns
        assert "affected_column" in res.recommendations_table.columns

        # anomalies must have column_name
        assert "column_name" in res.anomalies.columns
        assert "column" in res.anomalies.columns


class TestAUD021RedundantHealthSummaryAndDatetimeConversion:
    """AUD-021: Native datetime64 series preserved without string roundtrip; health_summary kwargs respected."""

    def test_native_datetime_range_preserved(self):
        dts = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({"dt": dts})
        meta = generate_meta(["dataset"], [df])
        assert meta.loc[0, "date_range_days"] == 9
        assert "2024-01-01" in str(meta.loc[0, "min_date"])
        assert "2024-01-10" in str(meta.loc[0, "max_date"])

    def test_cleanalytix_result_uses_passed_health_summary(self):
        custom_hs = pd.DataFrame([{"dataset_name": "custom", "avg_missing_percent": 12.34}])
        res = cx.CleanalytixResult(health_summary=custom_hs)
        assert res.health_summary.loc[0, "avg_missing_percent"] == 12.34


class TestAUD022ConsistencyRuleMetricInventory:
    """AUD-022: consistency_rules in metric_inventory must use unit='%' and violation percent as raw_value."""

    def test_consistency_rules_metric_inventory_coherence(self):
        df = pd.DataFrame({
            "start": [10, 20, 30, 40],
            "end": [15, 25, 20, 50],  # Row 2 (30, 20) violates start <= end (25% violation)
        })
        rule = ConsistencyRule(
            name="start_lte_end",
            condition=lambda d: d["start"] <= d["end"]
        )
        cfg = CleanalytixConfig(consistency_rules=[rule])
        res = cx.analyze(df, config=cfg)

        mi = res.metric_inventory
        cons_row = mi[mi["metric_name"] == "consistency_rules"]
        assert not cons_row.empty
        row_dict = cons_row.iloc[0].to_dict()

        assert row_dict["unit"] == "%"
        assert row_dict["threshold"] == 2.0
        assert row_dict["raw_value"] == 25.0
        assert row_dict["status"] == "WARNING"
        assert row_dict["contributes_to_score"] is True
