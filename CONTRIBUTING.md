# Contributing to Cleanalytix

Thanks for contributing.

## Development Setup

```bash
git clone https://github.com/Probot-DATA/Cleanalytix_Repo
cd Cleanalytix_Repo
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows activation:

```powershell
.venv\Scripts\Activate.ps1
```

## Running Tests

```bash
pytest
```

The automated test suite uses only the lightweight fixtures in `tests/sample_data/`.
Real-world validation runs live under `validation/` and are intentionally separate from unit tests.

## Linting

```bash
ruff check cleanalytix tests examples validation
```

## Validation Data

Large public validation datasets are not committed to the repository.
Use [validation/datasets/README.md](./validation/datasets/README.md) to download and place the expected files locally before running validation.

## Pull Requests

Before opening a pull request:

- make sure `pip install -e ".[dev]"` succeeds
- make sure `pytest` passes
- keep examples and docs in sync with any public API change
- avoid changing the pipeline output contract unless absolutely necessary
