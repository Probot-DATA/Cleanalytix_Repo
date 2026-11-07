# Changelog

All notable changes to `Cleanalytix` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-07

### Added
- Unified primary analysis API: `cleanalytix.analyze(df, ...) -> CleanalytixResult`.
- Authoritative 6-dimension data quality scoring framework: Completeness (0.20), Validity (0.20), Uniqueness (0.15), Consistency (0.15), Timeliness (0.15), and Stability (0.15).
- Strict dimension weight validation, custom metric weight configuration, and automatic applicability-aware normalization.
- Dedicated `cleanalytix.metrics` subpackage: completeness, continuous/categorical distribution profiling, robust statistical estimation (Bessel's correction), drift detection (continuous/discrete KL, JS, PSI, Wasserstein, KS), relationship analysis (correlation, Cramér's V, mutual information), candidate primary key discovery, and temporal metrics.
- Structured advisory recommendations (`InsightItem`) with severity levels (Critical, Warning, Info, Optimization).
- Comprehensive regression and mathematical validation test suite (43 tests).

### Changed
- Strictly non-mutating architecture: all public functions treat input DataFrames as immutable.
- Continuous entropy calculation now uses adaptive Freedman-Diaconis / Scott histogram binning instead of discrete value-counts.
- Timeliness row sequence monotonicity is opt-in (`require_monotonic=False` by default) to conform to tabular set semantics.
- Statistical outlier plausibility is evaluated under Validity; Consistency is strictly reserved for cross-column relational business rules.
- Declared `scipy>=1.10.0` as an explicit core runtime dependency in `pyproject.toml` and `requirements.txt`.

### Retired
- Retired destructive heuristic automated cleaning: calling `cleaning=True` raises `FeatureRetiredError` with guidance toward advisory remediation workflows.

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
