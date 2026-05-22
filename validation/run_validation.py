from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Make local package imports work whether this script is run from the repo
# root or from inside the validation directory/notebook.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleanalytix import Run_DQ_Pipeline

VALIDATION_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = VALIDATION_ROOT / "datasets"
OUTPUTS_ROOT = VALIDATION_ROOT / "outputs"
CHUNK_SIZE = 20_000

ARTIFACT_FILENAMES = {
    "dirty_scores": "dirty_scores.csv",
    "cleaned_scores": "cleaned_scores.csv",
    "recommendations": "recommendations.csv",
    "change_log": "change_log.csv",
    "meta_before_cleaning": "meta_before_cleaning.csv",
    "meta_after_cleaning": "meta_after_cleaning.csv",
    "summarized_before": "summarized_before.csv",
    "summarized_after": "summarized_after.csv",
    "main_metrics_before": "main_metrics_before.csv",
    "main_metrics_after": "main_metrics_after.csv",
}

DATASET_FILTER_COLUMNS = ("dataset_name", "Dataset_name", "dataset")


def check_age(value):
    if pd.isna(value):
        return True
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


RULES = {
    "age": check_age,
}


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _print_missing_dataset(dataset_name: str, missing_path: Path) -> None:
    relative_path = missing_path.relative_to(PROJECT_ROOT)
    print(
        "\nMissing validation dataset:\n"
        f"{dataset_name}\n\n"
        f"Expected file:\n{relative_path}\n\n"
        "Please download it following:\n"
        "validation/datasets/README.md\n\n"
        "Skipping this validation..."
    )


def _load_customer_churn() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(
            DATASETS_ROOT / "customer_churn" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
    )


def _load_uci_credit_card() -> pd.DataFrame:
    return pd.read_csv(
        _require_file(DATASETS_ROOT / "UCI_credit_card" / "UCI_Credit_Card.csv")
    )


def _load_cc_general() -> pd.DataFrame:
    return pd.read_csv(_require_file(DATASETS_ROOT / "CC_GENERAL" / "CC GENERAL.csv"))


def _load_adult() -> pd.DataFrame:
    col_names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "<=50k",
    ]
    adult = pd.read_csv(
        _require_file(DATASETS_ROOT / "adult" / "adult.data"),
        header=None,
        names=col_names,
    )
    adult_test = pd.read_csv(
        _require_file(DATASETS_ROOT / "adult" / "adult.test"),
        header=None,
        names=col_names,
        skiprows=1,
    )
    return pd.concat([adult, adult_test], ignore_index=True)


def _load_yellow_trip_data() -> pd.DataFrame:
    stems = [
        "yellow_tripdata_2015-01",
        "yellow_tripdata_2016-01",
        "yellow_tripdata_2016-02",
        "yellow_tripdata_2016-03",
    ]
    frames = []

    for stem in stems:
        path = _require_file(DATASETS_ROOT / "yellow_trip_data" / f"{stem}.csv")
        frames.append(next(pd.read_csv(path, chunksize=CHUNK_SIZE)))

    yellow = pd.concat(frames, ignore_index=True)
    yellow = yellow.rename(
        columns={
            "RateCodeID": "RateCodeID_1",
            "RatecodeID": "RateCodeID_2",
        }
    )
    yellow["RateCodeID_Final"] = yellow["RateCodeID_1"].combine_first(
        yellow["RateCodeID_2"]
    )
    yellow = yellow.drop(["RateCodeID_1", "RateCodeID_2"], axis=1)
    yellow["tpep_dropoff_datetime"] = pd.to_datetime(yellow["tpep_dropoff_datetime"])
    yellow["tpep_pickup_datetime"] = pd.to_datetime(yellow["tpep_pickup_datetime"])
    return yellow


VALIDATION_SPECS = [
    {
        "dataset_name": "yellow",
        "output_dir_name": "yellow_tripdata_2015-01",
        "loader": _load_yellow_trip_data,
    },
    {
        "dataset_name": "UCI_credit_card",
        "output_dir_name": "UCI_Credit_Card",
        "loader": _load_uci_credit_card,
    },
    {
        "dataset_name": "customer_churn",
        "output_dir_name": "WA_Fn-UseC_-Telco-Customer-Churn",
        "loader": _load_customer_churn,
    },
    {
        "dataset_name": "adult",
        "output_dir_name": "adult",
        "loader": _load_adult,
    },
    {
        "dataset_name": "CC_General",
        "output_dir_name": "CC GENERAL",
        "loader": _load_cc_general,
    },
]


def _filter_dataset_frame(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()

    for column in DATASET_FILTER_COLUMNS:
        if column in frame.columns:
            filtered = frame[frame[column] == dataset_name]
            return filtered if not filtered.empty else pd.DataFrame()

    return frame


def save_dataset_outputs(result, dataset_name: str, output_dir_name: str) -> Path:
    dataset_dir = OUTPUTS_ROOT / output_dir_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    base_data = result.get("base_data", {})
    if not isinstance(base_data, dict):
        return dataset_dir

    for result_key, filename in ARTIFACT_FILENAMES.items():
        filtered = _filter_dataset_frame(base_data.get(result_key), dataset_name)
        if filtered.empty:
            continue
        filtered.to_csv(dataset_dir / filename, index=False)

    return dataset_dir


def run_all_validations():
    """Run each configured real-world validation scenario independently."""
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    results = {}

    for spec in VALIDATION_SPECS:
        dataset_name = spec["dataset_name"]
        output_dir_name = spec["output_dir_name"]

        try:
            dataset = spec["loader"]()
            result = Run_DQ_Pipeline(
                dataset_names=[dataset_name],
                dataset_list=[dataset],
                rules=RULES,
                interactive=False,
                cleaning=True,
            )
            save_dataset_outputs(result, dataset_name, output_dir_name)
            results[dataset_name] = result
            print(f"Saved validation outputs for {dataset_name} to {OUTPUTS_ROOT / output_dir_name}")
        except FileNotFoundError as exc:
            _print_missing_dataset(dataset_name, Path(exc.filename or str(exc)))
        except Exception as exc:
            print(f"Skipping {dataset_name}: validation failed with {exc}")

    return results


if __name__ == "__main__":
    run_all_validations()
