from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from cleanalytix import (
    Run_DQ_Pipeline,
    generate_meta,
    get_cleaned_data,
    clean,
    cleaning_recommendations,
    FeatureRetiredError,
)
from cleanalytix.compute_dq_score import DEFAULT_THRESHOLDS, resolve_thresholds, get_new_values
from cleanalytix.generate_meta import get_type_error_count
from cleanalytix.get_table_for_DQ_computation import get_table_for_DQ_computation
from cleanalytix.preprocess_types import infer_and_fix_types


def test_production_meta_uses_production_missingness():
    base = pd.DataFrame({"age": [10, 20, 30]})
    prod = pd.DataFrame({"age": [None, None, 300]})

    meta = generate_meta(["ds"], [base], [prod])
    assert meta.loc[meta["column_name"] == "age", "missing_percent"].iloc[0] == pytest.approx(
        66.666666, rel=1e-3
    )


def test_retired_cleaning_raises_feature_retired_error():
    df = pd.DataFrame({"a": [None] * 5 + [1], "b": [1, 2, 3, 4, 5, 6]})
    recs = cleaning_recommendations(generate_meta(["ds"], [df]))

    with pytest.raises(FeatureRetiredError) as exc_info:
        get_cleaned_data(["ds"], [df], recs)
    assert "retired" in str(exc_info.value).lower()

    with pytest.raises(FeatureRetiredError) as exc_info:
        clean(df, "a", 1)
    assert "retired" in str(exc_info.value).lower()


def test_mostly_numeric_mixed_column_counts_text_as_error():
    df = pd.DataFrame({"col": ["1", "2", "3", "4", "bad"]})
    assert get_type_error_count(df, "col") == 1


def test_binary_flag_not_treated_as_continuous():
    meta = generate_meta(["ds"], [pd.DataFrame({"flag": [0, 1, 0, 1, 0]})])
    row = meta.iloc[0]
    assert row["column_role"] == "binary"
    assert pd.isna(row["skewness_level"]) or row["skewness_level"] is None
    assert row["outlier_count"] == 0


def test_identifier_column_skips_continuous_profiling():
    df = pd.DataFrame({"customer_id": range(50), "amount": [i * 1.5 for i in range(50)]})
    meta = generate_meta(["ds"], [df])
    id_row = meta.loc[meta["column_name"] == "customer_id"].iloc[0]
    assert id_row["column_role"] == "identifier"
    assert id_row["outlier_count"] == 0


def test_duplicate_threshold_is_documented_default():
    assert DEFAULT_THRESHOLDS["duplicate_rows_percent"] == 1.0
    thresholds = resolve_thresholds()
    assert thresholds["duplicate_rows_percent"] == 1.0

    df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [3, 4, 3, 4]})
    meta = generate_meta(["ds"], [df])
    metrics = get_table_for_DQ_computation(meta, [df], ["ds"])
    assert metrics["duplicate_rows_percent"].iloc[0] == 50.0
    scored = get_new_values(metrics.iloc[:1], thresholds=thresholds)
    assert scored["duplicate_rows_percent"].iloc[0] < 1.0


def test_infer_types_for_new_false_preserves_production_dtype():
    base = pd.DataFrame({"amount": ["1", "2", "3", "bad"]})
    prod = pd.DataFrame({"amount": ["1", "2", "3", "5"]})

    result = Run_DQ_Pipeline(
        ["ds"],
        [base],
        new_dataset_list=[prod],
        infer_types=True,
        infer_types_for_new=False,
    )

    assert str(result["base_data"]["meta_before_cleaning"]["dtype"].iloc[0]) != "object"
    assert str(result["prod_data"]["meta_before_cleaning"]["dtype"].iloc[0]) == "object"


def test_generate_meta_handles_boolean_columns():
    meta = generate_meta(["ds"], [pd.DataFrame({"active": [True, False, True]})])
    row = meta.iloc[0]
    assert row["column_role"] == "binary"
    assert row["mean"] is None or pd.isna(row["mean"])


def test_zero_self_drift_in_baseline_only_mode():
    """BUG-03: Baseline profiling must not split dataset against itself or penalize self-drift."""
    df = pd.DataFrame({"score": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]})
    meta = generate_meta(["baseline"], [df])
    assert meta["psi_divergence"].iloc[0] == 0.0
    assert meta["js_divergence"].iloc[0] == 0.0
    assert meta["kl_divergence"].iloc[0] == 0.0


def test_constant_baseline_drift_detection():
    """BUG-04: Constant baseline columns must handle production divergence gracefully."""
    base_df = pd.DataFrame({"val": [5.0, 5.0, 5.0, 5.0, 5.0]})
    prod_identical = pd.DataFrame({"val": [5.0, 5.0, 5.0]})
    prod_shifted = pd.DataFrame({"val": [5.0, 6.0, 7.0]})

    meta_ident = generate_meta(["ds"], [base_df], [prod_identical])
    assert meta_ident["psi_divergence"].iloc[0] == 0.0

    meta_shifted = generate_meta(["ds"], [base_df], [prod_shifted])
    assert meta_shifted["psi_divergence"].iloc[0] > 0.0


def test_sample_variance_uses_bessel_correction():
    """BUG-09: Sample variance and standard deviation must use ddof=1."""
    # Values [2, 4]: mean = 3. Sample var (ddof=1) = ((2-3)^2 + (4-3)^2)/1 = 2.0. std = sqrt(2) ~= 1.4142
    df = pd.DataFrame({"x": [2.0, 4.0]})
    meta = generate_meta(["ds"], [df])
    computed_std = meta["std"].iloc[0]
    assert computed_std == pytest.approx(np.sqrt(2.0), rel=1e-4)


def test_nullable_types_handling():
    """Verify pandas nullable types (Int64, Float64) do not raise TypeError on pd.NA."""
    df = pd.DataFrame({
        "int_col": pd.Series([10, 20, None, 40], dtype="Int64"),
        "float_col": pd.Series([1.5, 2.5, None, 4.5], dtype="Float64"),
    })
    meta = generate_meta(["ds"], [df])
    assert meta.loc[meta["column_name"] == "int_col", "mean"].iloc[0] == pytest.approx(23.333, rel=1e-2)
    assert meta.loc[meta["column_name"] == "float_col", "mean"].iloc[0] == pytest.approx(2.833, rel=1e-2)


def test_duplicate_column_names_rejected():
    """BUG-12: Duplicate column names must raise an informative ValueError."""
    df = pd.DataFrame([[1, 2]], columns=["col", "col"])
    with pytest.raises(ValueError) as exc_info:
        generate_meta(["ds"], [df])
    assert "duplicate column names" in str(exc_info.value).lower()


def test_zero_row_dataframe_handling():
    """Verify zero-row DataFrame does not crash with ZeroDivisionError."""
    df = pd.DataFrame(columns=["a", "b"])
    meta = generate_meta(["empty"], [df])
    assert len(meta) == 2
    metrics = get_table_for_DQ_computation(meta, [df], ["empty"])
    assert len(metrics) == 2


def test_metrics_completeness_module():
    """Verify cleanalytix.metrics.completeness functions."""
    from cleanalytix.metrics.completeness import compute_completeness_metrics

    df = pd.DataFrame({
        "a": [1, None, 3, None],
        "b": [1, None, None, None],
        "c": [1, 2, 3, 4],
    })
    res = compute_completeness_metrics(df)
    assert res["complete_rows_count"] == 1
    assert res["column_missing_count"]["b"] == 3
    assert len(res["missingness_cooccurrence"]) >= 1


def test_metrics_uniqueness_module():
    """Verify cleanalytix.metrics.uniqueness functions."""
    from cleanalytix.metrics.uniqueness import (
        compute_column_uniqueness,
        compute_duplicate_row_metrics,
        detect_candidate_keys,
    )

    df = pd.DataFrame({
        "id": [101, 102, 103, 104],
        "category": ["A", "A", "B", "A"],
        "constant_col": [9, 9, 9, 9],
    })
    col_u = compute_column_uniqueness(df["constant_col"])
    assert col_u["is_constant"] is True
    assert col_u["distinct_count"] == 1

    keys = detect_candidate_keys(df)
    assert "id" in keys["single_keys"]

    dups = compute_duplicate_row_metrics(df)
    assert dups["duplicate_row_count"] == 0


def test_metrics_statistical_module():
    """Verify cleanalytix.metrics.statistical functions."""
    from cleanalytix.metrics.statistical import (
        compute_numeric_statistics,
        detect_outliers,
    )

    s = pd.Series([10.0, 12.0, 11.0, 13.0, 12.0, 100.0])  # 100 is an outlier
    stats = compute_numeric_statistics(s)
    assert stats["count"] == 6
    assert stats["median"] == 12.0
    assert stats["max"] == 100.0
    assert stats["min"] == 10.0
    assert stats["skewness"] > 1.0  # heavily right-skewed due to 100.0 outlier

    outliers = detect_outliers(s)
    assert outliers["recommended_outlier_count"] >= 1


def test_metrics_distribution_module():
    """Verify cleanalytix.metrics.distribution functions."""
    from cleanalytix.metrics.distribution import compute_categorical_distribution

    s = pd.Series(["A"] * 80 + ["B"] * 15 + ["C"] * 5)
    dist = compute_categorical_distribution(s)
    assert dist["unique_categories"] == 3
    assert 0.0 < dist["normalized_entropy"] < 1.0
    assert dist["hhi"] > 0.5


def test_metrics_drift_module():
    """Verify cleanalytix.metrics.drift functions."""
    from cleanalytix.metrics.drift import (
        compute_numeric_drift,
        compute_categorical_drift,
    )

    b_num = pd.Series(np.random.normal(0, 1, 100))
    p_num = pd.Series(np.random.normal(5, 1, 100))  # clear drift
    drift_res = compute_numeric_drift(b_num, p_num)
    assert drift_res["drift_detected"] is True
    assert drift_res["psi"] > 0.25

    b_cat = pd.Series(["apple", "banana"] * 50)
    p_cat = pd.Series(["apple", "orange"] * 50)  # "orange" is unseen
    cat_drift = compute_categorical_drift(b_cat, p_cat)
    assert "orange" in cat_drift["unseen_categories"]


def test_metrics_relationships_module():
    """Verify cleanalytix.metrics.relationships functions."""
    from cleanalytix.metrics.relationships import (
        compute_numeric_correlations,
        compute_cramers_v,
        detect_duplicate_columns,
    )

    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [2, 4, 6, 8, 10],  # perfect correlation
        "z": [1, 2, 3, 4, 5],  # duplicate of x
    })
    corrs = compute_numeric_correlations(df)
    assert len(corrs["high_correlation_pairs"]) >= 1

    dups = detect_duplicate_columns(df)
    assert ("x", "z") in dups["duplicate_columns"] or ("z", "x") in dups["duplicate_columns"]

    s1 = pd.Series(["red", "blue", "red", "blue"])
    s2 = pd.Series(["stop", "go", "stop", "go"])
    v = compute_cramers_v(s1, s2)
    assert v == pytest.approx(1.0, rel=1e-2)


def test_metrics_temporal_module():
    """Verify cleanalytix.metrics.temporal functions."""
    from cleanalytix.metrics.temporal import compute_temporal_metrics

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    temp_res = compute_temporal_metrics(pd.Series(dates))
    assert temp_res["is_temporal"] is True
    assert temp_res["is_monotonic_increasing"] is True
    assert temp_res["time_span_days"] == 9.0


def test_single_row_dataset_handling():
    """Verify single-row dataset (N=1) handles sample stats gracefully without zero variance errors."""
    from cleanalytix import analyze

    df = pd.DataFrame({"a": [42.0], "b": ["text"]})
    res = analyze(df)
    assert 0.0 <= res.overall_score <= 100.0
    meta = res.metadata
    assert meta.loc[meta["column_name"] == "a", "mean"].iloc[0] == 42.0
    # Standard deviation on N=1 must be NaN or None, not 0.0
    val_std = meta.loc[meta["column_name"] == "a", "std"].iloc[0]
    assert pd.isna(val_std) or val_std is None
