# Cleanalytix

`Cleanalytix` is a Python library for profiling, scoring, cleaning, and monitoring the quality of tabular datasets with a single pipeline.

It is designed for pandas-first workflows and supports:

- baseline dataset profiling and scoring
- optional cleaning recommendations and automatic cleaning
- optional production/new-dataset monitoring
- optional business rules, thresholds, weights, and type inference for new data

## Installation

From a source checkout:

```bash
git clone https://github.com/Probot-DATA/Cleanalytix_Repo
cd Cleanalytix_Repo
pip install -e ".[dev]"
```

Once this project is published to PyPI, the install command will be:

```bash
pip install cleanalytix
```

Runtime requirements:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- nltk

## Quick Start

```python
import pandas as pd
from cleanalytix import Run_DQ_Pipeline

df = pd.read_csv("my_data.csv")

result = Run_DQ_Pipeline(
    dataset_names=["my_dataset"],
    dataset_list=[df],
)

print(result["base_data"]["dirty_scores"])
print(result["base_data"]["meta_before_cleaning"])
print(result["base_data"]["recommendations"])
```

## Production / Monitoring Example

```python
import pandas as pd
from cleanalytix import Run_DQ_Pipeline

train_df = pd.read_csv("train.csv")
prod_df = pd.read_csv("production.csv")

rules = {
    "age": lambda value: pd.isna(value) or 0 <= float(value) <= 120,
}

result = Run_DQ_Pipeline(
    dataset_names=["customers"],
    dataset_list=[train_df],
    new_dataset_list=[prod_df],
    rules=rules,
    cleaning=True,
    interactive=False,
    score_mode="exponential",
)

print(result["base_data"]["dirty_scores"])
print(result["base_data"]["cleaned_scores"])
print(result["prod_data"]["dirty_scores"])
print(result["prod_data"]["change_log"])
```

## Public API

The primary entrypoint is:

```python
from cleanalytix import Run_DQ_Pipeline
```

Additional building blocks are also exported:

```python
from cleanalytix import (
    Compute_DQ_Score,
    DEFAULT_THRESHOLDS,
    generate_meta,
    cleaning_recommendations,
    get_cleaned_data,
    get_table_for_DQ_computation,
    summarize_dataset_health,
    learn_reference_profile,
    adjust_prod_meta_with_reference,
    infer_and_fix_types,
)
```

## Pipeline Output Structure

`Run_DQ_Pipeline` returns:

```python
{
    "base_data": {...},
    "prod_data": {...},
}
```

Each block preserves the same keys:

- `dirty_scores`
- `cleaned_scores`
- `cleaned_datasets`
- `meta_before_cleaning`
- `meta_after_cleaning`
- `recommendations`
- `change_log`
- `summarized_before`
- `summarized_after`
- `main_metrics_before`
- `main_metrics_after`

## Examples

Runnable examples live in [examples](./examples):

```bash
python examples/simple_usage.py
python examples/production_usage.py
```

These examples assume the package has already been installed in the active environment.

## Validation

The [validation](./validation) folder contains a portable real-world validation workflow.

- Large raw datasets are intentionally not committed to the repository.
- Put the expected files under `validation/datasets/` by following
  [validation/datasets/README.md](./validation/datasets/README.md).
- Run the validation script:

```bash
python validation/run_validation.py
```

- The script saves non-empty outputs to `validation/outputs/<dataset_name>/`.
- The notebook [validation/main.ipynb](./validation/main.ipynb) uses the same relative-path workflow.

## Repository Layout

- `cleanalytix/` - installable library package
- `examples/` - small runnable examples
- `tests/` - smoke tests and lightweight sample fixtures
- `validation/` - public-friendly validation workflow and output folder
- `archive/legacy/` - historical prototype notebook/code kept for reference, not for active use

## Known Limitations

- Validation datasets are not bundled with the repository.
- The yellow taxi validation workflow samples the first `20,000` rows from each configured monthly file to match the original project workflow and to keep validation practical.
- Interactive cleaning is intended for notebook/CLI use and will prompt for input when `interactive=True`.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

[MIT](./LICENSE) (c) 2026 Probot-DATA contributors
