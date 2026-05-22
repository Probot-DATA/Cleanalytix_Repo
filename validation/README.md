# Validation

This folder contains the portable, public-friendly validation workflow for `Cleanalytix`.

## What is included

- `run_validation.py` - command-line validation runner
- `main.ipynb` - notebook wrapper around the same relative-path workflow
- `datasets/README.md` - dataset download instructions and expected filenames
- `outputs/` - generated validation outputs (not committed)

## What is not included

Large real-world validation datasets are intentionally not stored in the repository.

## Run the validation workflow

1. Install the package in your environment:

```bash
pip install -e ".[dev]"
```

2. Download the datasets described in `validation/datasets/README.md`.
3. Place the files under `validation/datasets/` using the documented names.
4. Run:

```bash
python validation/run_validation.py
```

The runner saves non-empty CSV outputs to `validation/outputs/<dataset_name>/`
and continues if one dataset is missing or fails.
