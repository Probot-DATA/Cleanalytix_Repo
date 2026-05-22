from pathlib import Path

import pandas as pd

from cleanalytix import Run_DQ_Pipeline, __version__

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


def _assert_pipeline_block(block, expect_cleaning):
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

    if expect_cleaning:
        assert list(block["cleaned_scores"].columns) == EXPECTED_SCORE_COLUMNS
        assert not block["cleaned_scores"].empty
        assert isinstance(block["cleaned_datasets"], list)
        assert len(block["cleaned_datasets"]) == 1
        assert isinstance(block["cleaned_datasets"][0], pd.DataFrame)
        assert not block["meta_after_cleaning"].empty
        assert not block["summarized_after"].empty
        assert not block["main_metrics_after"].empty
    else:
        assert list(block["cleaned_scores"].columns) == EXPECTED_SCORE_COLUMNS
        assert block["meta_after_cleaning"].empty
        assert block["summarized_after"].empty
        assert block["main_metrics_after"].empty


def test_public_import_surface():
    assert __version__
    assert callable(Run_DQ_Pipeline)


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
        infer_types_for_new=False,
        score_mode="exponential",
    )

    _assert_pipeline_block(result["base_data"], expect_cleaning=False)
    _assert_pipeline_block(result["prod_data"], expect_cleaning=False)


def test_pipeline_with_cleaning_and_interactive_branch_runs(monkeypatch):
    base, prod = _load_sample_frames()
    responses = iter(["no"] * 64)
    monkeypatch.setattr("builtins.input", lambda *args: next(responses))

    result = Run_DQ_Pipeline(
        dataset_names=["customers"],
        dataset_list=[base],
        new_dataset_list=[prod],
        rules={"age": _valid_age},
        cleaning=True,
        interactive=True,
    )

    _assert_pipeline_block(result["base_data"], expect_cleaning=True)
    _assert_pipeline_block(result["prod_data"], expect_cleaning=True)
