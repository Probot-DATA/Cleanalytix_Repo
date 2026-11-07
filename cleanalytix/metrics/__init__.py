"""Cleanalytix metrics package.

Exposes specialized mathematical and statistical metrics across key data quality dimensions.
"""

from cleanalytix.metrics.completeness import (
    compute_completeness_metrics,
    compute_missingness_metrics,
)
from cleanalytix.metrics.uniqueness import (
    compute_column_uniqueness,
    compute_duplicate_row_metrics,
    detect_candidate_keys,
)
from cleanalytix.metrics.statistical import (
    compute_numeric_statistics,
    detect_outliers,
)
from cleanalytix.metrics.distribution import (
    compute_categorical_distribution,
)
from cleanalytix.metrics.drift import (
    compute_numeric_drift,
    compute_categorical_drift,
)
from cleanalytix.metrics.relationships import (
    compute_numeric_correlations,
    compute_cramers_v,
    detect_duplicate_columns,
)
from cleanalytix.metrics.temporal import (
    compute_temporal_metrics,
)

__all__ = [
    "compute_completeness_metrics",
    "compute_missingness_metrics",
    "compute_column_uniqueness",
    "compute_duplicate_row_metrics",
    "detect_candidate_keys",
    "compute_numeric_statistics",
    "detect_outliers",
    "compute_categorical_distribution",
    "compute_numeric_drift",
    "compute_categorical_drift",
    "compute_numeric_correlations",
    "compute_cramers_v",
    "detect_duplicate_columns",
    "compute_temporal_metrics",
]
