"""Cleanalytix
===========

Cleanalytix is a Python library for Data Quality Intelligence and Advanced
Analytics on tabular datasets. It provides multidimensional data quality profiling,
statistical evaluation, divergence and drift detection, key discovery, and
deterministic, non-mutating advisory recommendations.

Strict Non-Mutation Principle
-----------------------------
Cleanalytix is strictly non-mutating. It never modifies, transforms, imputes, or
drops rows/columns in the caller's data. All analytical operations treat input
DataFrames as immutable.

Primary Entry Points
--------------------
- `cleanalytix.analyze(...)`: The modern, unified API for analyzing a DataFrame
  and receiving a structured `CleanalytixResult` with dimensional scores, insights,
  and actionable recommendations.
- `cleanalytix.Run_DQ_Pipeline(...)`: The pipeline interface for multi-dataset
  profiling, scoring, and production monitoring.

Quality Dimensions
------------------
Cleanalytix assesses data health across 6 core dimensions:
1. **Completeness**: Missingness patterns, null rates, and missingness correlation.
2. **Validity**: Semantic type conformity, regex validation, whitespace cleanliness, domain rules, and statistical outlier plausibility.
3. **Uniqueness**: Duplicate row rates, distinct ratios, and candidate primary key discovery.
4. **Consistency**: Safe cross-column relational business rules.
5. **Timeliness**: Datetime freshness, future date containment, and optional sequence monotonicity.
6. **Stability**: Population Stability Index (PSI), Wasserstein distance, and Jensen-Shannon divergence.
"""

from cleanalytix.version import __author__, __version__
from cleanalytix.pipeline import analyze, Run_DQ_Pipeline
from cleanalytix.scoring import score_dataset, DEFAULT_THRESHOLDS
from cleanalytix.compute_dq_score import Compute_DQ_Score
from cleanalytix.generate_meta import generate_meta
from cleanalytix.cleaning_recommendations import cleaning_recommendations
from cleanalytix.get_cleaned_data import get_cleaned_data, clean
from cleanalytix.get_table_for_DQ_computation import get_table_for_DQ_computation
from cleanalytix.summarize_dataset_health import summarize_dataset_health
from cleanalytix.adjust_prod_meta import learn_reference_profile, adjust_prod_meta_with_reference
from cleanalytix.preprocess_types import infer_and_fix_types, diagnose_column_types
from cleanalytix.types import (
    AnalysisLevel,
    QualityDimension,
    InsightSeverity,
    InsightItem,
    DimensionScore,
    CleanalytixConfig,
    CleanalytixResult,
    FeatureRetiredError,
    DQPipelineResult,
)
from cleanalytix import metrics

run_dq_pipeline = Run_DQ_Pipeline

__all__ = [
    # Primary Modern API
    "analyze",
    "score_dataset",
    "CleanalytixConfig",
    "CleanalytixResult",
    "QualityDimension",
    "DimensionScore",
    "InsightItem",
    "InsightSeverity",
    "AnalysisLevel",
    "FeatureRetiredError",
    "diagnose_column_types",
    "metrics",
    # Legacy & Pipeline API
    "Run_DQ_Pipeline",
    "run_dq_pipeline",
    "Compute_DQ_Score",
    "DEFAULT_THRESHOLDS",
    "DQPipelineResult",
    "generate_meta",
    "cleaning_recommendations",
    "get_cleaned_data",
    "clean",
    "get_table_for_DQ_computation",
    "summarize_dataset_health",
    "learn_reference_profile",
    "adjust_prod_meta_with_reference",
    "infer_and_fix_types",
    # Metadata
    "__author__",
    "__version__",
]
