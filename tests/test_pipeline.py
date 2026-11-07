from pathlib import Path
import pandas as pd
import pytest

from cleanalytix import (
    Run_DQ_Pipeline,
    __version__,
    analyze,
    CleanalytixResult,
    FeatureRetiredError,
    QualityDimension,
    infer_and_fix_types,
)

SAMPLE_DIR = Path(__file__).parent / "sample_data"
EXPECTED_SCORE_COLUMNS = ["Dataset_name", "DQ_Score"]


def _load_sample_frames():
    base = pd.read_csv(SAMPLE_DIR / "base_customers.csv")
    prod = pd.read_csv(SAMPLE_DIR / "prod_customers.csv")
    return base, prod


def _valid_age(value):
    if pd.isna(value):
        return True
    try:
        return 0 <= float(value) <= 120
    except (TypeError, ValueError):
        return False


def _assert_pipeline_block(block, expect_cleaning=False):
    required_keys = {
        "dirty_scores",
        "cleaned_scores",
        "cleaned_datasets",
        "meta_before_cleaning",
        "meta_after_cleaning",
        "recommendations",
        "change_log",
        "summarized_before",
        "summarized_after",
        "main_metrics_before",
        "main_metrics_after",
    }

    assert required_keys.issubset(block.keys())
    assert list(block["dirty_scores"].columns) == EXPECTED_SCORE_COLUMNS
    assert not block["dirty_scores"].empty
    assert not block["meta_before_cleaning"].empty
    assert not block["recommendations"].empty
    assert not block["summarized_before"].empty
    assert not block["main_metrics_before"].empty


def test_public_import_surface():
    assert __version__
    assert callable(Run_DQ_Pipeline)
    assert callable(analyze)


def test_pipeline_base_only_runs_without_errors():
    base, _ = _load_sample_frames()

    result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[base],
    )

    assert result["prod_data"] == {}
    _assert_pipeline_block(result["base_data"], expect_cleaning=False)


def test_pipeline_with_production_options_runs_without_errors():
    base, prod = _load_sample_frames()

    result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[base],
        new_dataset_list=[prod],
        rules={"age": _valid_age},
        weights=[0.20, 0.05, 0.15, 0.10, 0.15, 0.10, 0.15, 0.10],
        thresholds={
            "missing_percent": 15.0,
            "psi_divergence": 0.20,
        },
        infer_types=True,
        infer_types_for_new=False,
        score_mode="exponential",
    )

    _assert_pipeline_block(result["base_data"], expect_cleaning=False)
    _assert_pipeline_block(result["prod_data"], expect_cleaning=False)


def test_infer_types_false_does_not_call_preprocessor_for_base_or_prod(monkeypatch):
    base, prod = _load_sample_frames()

    def fail_if_called(_df):
        raise AssertionError("type inference should not run when infer_types=False")

    monkeypatch.setattr("cleanalytix.pipeline.infer_and_fix_types", fail_if_called)

    result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[base],
        new_dataset_list=[prod],
    )

    _assert_pipeline_block(result["base_data"], expect_cleaning=False)
    _assert_pipeline_block(result["prod_data"], expect_cleaning=False)


def test_hard_type_inference_current_semantics():
    df = pd.DataFrame(
        {
            "amount": [str(i) for i in range(1, 100)] + ["not numeric"],
            "event_date": [
                f"2024-01-{(i % 28) + 1:02d}T12:00:00+05:30"
                for i in range(99)
            ]
            + ["not a date"],
            "is_active": ([True, False] * 49) + [True, "unknown"],
            "flag_01": [0, 1] * 50,
            "customer_id": [f"{i:05d}" for i in range(100)],
            "ship_date": ["01/02/2024"] * 100,
        }
    )

    fixed = infer_and_fix_types(df)

    assert pd.api.types.is_numeric_dtype(fixed["amount"])
    assert pd.isna(fixed.loc[99, "amount"])

    assert str(fixed["event_date"].dtype) == "datetime64[ns, UTC]"
    assert pd.isna(fixed.loc[99, "event_date"])

    assert str(fixed["is_active"].dtype) == "boolean"
    assert bool(fixed.loc[0, "is_active"]) is True
    assert bool(fixed.loc[1, "is_active"]) is False
    assert pd.isna(fixed.loc[99, "is_active"])
    assert str(fixed["flag_01"].dtype) == "boolean"

    pd.testing.assert_series_equal(fixed["customer_id"], df["customer_id"])
    pd.testing.assert_series_equal(fixed["ship_date"], df["ship_date"])


def test_pipeline_cleaning_raises_feature_retired_error():
    base, prod = _load_sample_frames()

    with pytest.raises(FeatureRetiredError) as exc_info:
        Run_DQ_Pipeline(
            dataset_names=["customers"],
            dataset_list=[base],
            new_dataset_list=[prod],
            cleaning=True,
        )
    assert "retired" in str(exc_info.value).lower()


def test_analyze_api_contract():
    base, _ = _load_sample_frames()
    orig = base.copy(deep=True)

    result = analyze(base)
    assert isinstance(result, CleanalytixResult)
    assert 0.0 <= result.overall_score <= 100.0
    assert QualityDimension.COMPLETENESS.value in result.quality_dimensions
    assert QualityDimension.VALIDITY.value in result.quality_dimensions
    assert QualityDimension.UNIQUENESS.value in result.quality_dimensions
    assert QualityDimension.CONSISTENCY.value in result.quality_dimensions
    assert isinstance(result.insights, list)
    assert isinstance(result.recommendations, list)
    assert not result.metadata.empty
    # Strict non-mutation check
    assert base.equals(orig)


def test_analyze_with_baseline():
    base, prod = _load_sample_frames()
    result = analyze(prod, baseline_df=base)
    assert QualityDimension.STABILITY.value in result.quality_dimensions
    assert 0.0 <= result.overall_score <= 100.0
    summary = result.summary()
    assert "dimensions" in summary
    assert "overall_score" in summary
