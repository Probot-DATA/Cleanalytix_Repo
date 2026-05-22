# Changelog

All notable changes to `Cleanalytix` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- cleaned the repository layout for public release
- removed hardcoded local-path assumptions from the validation workflow
- aligned docs, examples, and tests with the actual repository structure
- added a portable validation script and dataset instructions
- removed generated artifacts and large tracked validation data from the public tree
- made `Run_DQ_Pipeline(..., score_mode=...)` forward the documented scoring mode to the scorer

## [0.1.0] - 2026-05-19

### Added

- `Run_DQ_Pipeline`
- `generate_meta`
- `Compute_DQ_Score`
- `cleaning_recommendations`
- `get_cleaned_data`
- `summarize_dataset_health`
- `learn_reference_profile`
- `adjust_prod_meta_with_reference`
- `infer_and_fix_types`
- `DEFAULT_THRESHOLDS`
