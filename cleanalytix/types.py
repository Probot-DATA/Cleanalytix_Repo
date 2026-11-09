"""Typing definitions, enums, dataclasses, and result containers for Cleanalytix.

This module defines public protocols, enums, dataclasses, and structured result
objects for Cleanalytix Data Quality Intelligence & Advanced Analytics.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, Tuple, TypedDict, Union
import numpy as np
import pandas as pd


class FeatureRetiredError(NotImplementedError):
    """Raised when retired transformational features (such as automatic cleaning) are requested."""
    pass


class QualityDimension(str, Enum):
    """Standardized Data Quality Dimensions."""
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    STABILITY = "stability"


class InsightSeverity(str, Enum):
    """Severity classification for detected quality insights."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    OPTIMIZATION = "optimization"


ColumnRole = Literal[
    "continuous",
    "discrete",
    "binary",
    "identifier",
    "categorical",
    "free_text",
    "datetime",
    "boolean",
    "unknown",
]
"""Semantic column role inferred during profiling."""

SemanticType = Literal["numeric", "datetime", "binary", "identifier", "categorical", "text"]
"""Diagnostic semantic type."""

ScoreMode = Literal["linear", "exponential"]
"""Supported score transformation modes."""


class AnalysisLevel(str, Enum):
    """Configurable analysis depth."""
    BASIC = "basic"
    STANDARD = "standard"
    DEEP = "deep"
    FULL = "full"


class BusinessRule(Protocol):
    """Callable protocol for user-defined column validation rules."""

    def __call__(self, value: Any) -> bool:
        """Return True when *value* is valid for the target column."""


Rules = Mapping[str, BusinessRule]
"""Mapping from column name to validation callable."""

Thresholds = Mapping[str, float]
"""Mapping from scoring metric name to numeric threshold."""

MetricWeights = Sequence[float]
"""Metric weights used by the DQ scorer. Values must sum to 1.0."""


class ChangeLogEntry(TypedDict, total=False):
    """Record of a transformation action applied to a dataset."""
    dataset: str
    column: str
    rule_id: int
    replacement: Any
    affected_rows: int


class ReferenceColumnProfile(TypedDict, total=False):
    """Learned reference statistics for one dataset column."""

    type: str
    column_role: str
    lower: float
    upper: float
    method: str
    skewness: float
    popular: list[str]
    mean: float
    std: float
    median: float
    mad: float


ReferenceProfile = Dict[str, Dict[str, ReferenceColumnProfile]]
"""Nested profile: ``profile[dataset_name][column_name]``."""


@dataclass
class InsightItem:
    """Structured data quality insight or anomaly finding."""

    dimension: QualityDimension
    severity: InsightSeverity
    column: Optional[str]
    message: str
    metric_name: str
    metric_value: float
    threshold: float
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value if isinstance(self.dimension, QualityDimension) else str(self.dimension),
            "severity": self.severity.value if isinstance(self.severity, InsightSeverity) else str(self.severity),
            "column": self.column,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
            "details": self.details,
        }


@dataclass
class RecommendationItem:
    """Structured, actionable recommendation for data quality remediation.

    Contains machine-readable metadata, deterministic rule IDs (matching legacy
    pipeline hooks), analytical rationale, and suggested remediation code.
    """

    priority_rank: int
    rule_id: int
    dimension: QualityDimension
    severity: str
    affected_column: Optional[str]
    issue_name: str
    evidence: str
    rationale: str
    suggested_action: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority_rank": self.priority_rank,
            "rule_id": self.rule_id,
            "dimension": self.dimension.value if isinstance(self.dimension, QualityDimension) else str(self.dimension),
            "severity": self.severity,
            "affected_column": self.affected_column,
            "issue_name": self.issue_name,
            "evidence": self.evidence,
            "rationale": self.rationale,
            "suggested_action": self.suggested_action,
            "details": self.details,
        }


@dataclass
class ConsistencyRule:
    """Cross-column consistency constraint or logical business validation rule.

    Evaluated safely without arbitrary string eval(). The condition callable accepts
    the DataFrame and returns a boolean Series or array where True indicates consistency
    and False indicates a violation.
    """

    name: str
    condition: Any
    description: Optional[str] = None
    columns: Optional[List[str]] = None


@dataclass
class TimelinessConfig:
    """Configuration for Timeliness quality dimension evaluation."""

    reference_time: Optional[Any] = None
    freshness_days: float = 30.0
    allow_future_dates: bool = False
    datetime_columns: Optional[List[str]] = None
    require_monotonic: bool = False


@dataclass
class DimensionScore:
    """Score breakdown for a specific data quality dimension."""

    dimension: QualityDimension
    score: float
    weight: float = 0.0
    configured_weight: float = 0.0
    effective_weight: float = 0.0
    applicable: bool = True
    reason: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    metric_scores: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.weight == 0.0 and self.effective_weight != 0.0:
            self.weight = self.effective_weight
        elif self.effective_weight == 0.0 and self.weight != 0.0:
            self.effective_weight = self.weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension.value if isinstance(self.dimension, QualityDimension) else str(self.dimension),
            "score": self.score,
            "weight": self.weight,
            "configured_weight": self.configured_weight,
            "effective_weight": self.effective_weight,
            "applicable": self.applicable,
            "reason": self.reason,
            "metrics": self.metrics,
            "metric_scores": self.metric_scores,
            "details": self.details,
        }


def _normalize_column_profiles(
    columns: Union[Dict[str, "ColumnProfile"], Sequence["ColumnProfile"], None],
) -> Dict[str, "ColumnProfile"]:
    """Normalize a mapping or sequence of ColumnProfile objects into an ordered dict.

    Preserves exact insertion/sequence ordering and raises ``ValueError`` if duplicate
    column names or mismatched dictionary keys are encountered, ensuring columns and
    ordering are never silently lost.
    """
    if columns is None:
        return {}
    if isinstance(columns, Mapping):
        normalized: Dict[str, ColumnProfile] = {}
        for key, profile in columns.items():
            if not isinstance(profile, ColumnProfile):
                raise TypeError(
                    f"Expected ColumnProfile instance for column '{key}', got {type(profile).__name__}."
                )
            if str(key) != profile.name:
                raise ValueError(
                    f"Column mapping key '{key}' does not match ColumnProfile.name '{profile.name}'."
                )
            normalized[str(key)] = profile
        return normalized
    if isinstance(columns, Sequence) and not isinstance(columns, (str, bytes)):
        normalized = {}
        for profile in columns:
            if not isinstance(profile, ColumnProfile):
                raise TypeError(
                    f"Expected ColumnProfile instance in columns sequence, got {type(profile).__name__}."
                )
            if profile.name in normalized:
                raise ValueError(
                    f"Duplicate column name '{profile.name}' in ColumnProfile sequence. "
                    f"Column names must be unique to prevent silent column loss."
                )
            normalized[profile.name] = profile
        return normalized
    raise TypeError(
        f"columns must be a Mapping[str, ColumnProfile] or Sequence[ColumnProfile], got {type(columns).__name__}."
    )


@dataclass
class ColumnCompleteness:
    """Column-level completeness and missingness metrics.

    Authoritative owner of per-column non-null and missing counts/percentages
    grounded in ``cleanalytix.metrics.completeness.compute_completeness_metrics()``.
    Dataset-level completeness metrics (e.g., ``complete_rows_count``,
    ``empty_rows_count``, ``null_density``) belong to ``DatasetProfile``.
    """

    total_count: int = 0
    non_null_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    completeness_ratio: float = 1.0
    empty_string_count: int = 0

    @property
    def missing_count(self) -> int:
        """Read-only alias for ``null_count``."""
        return self.null_count

    @property
    def missing_percent(self) -> float:
        """Read-only alias for ``null_percentage``."""
        return self.null_percentage


@dataclass
class ColumnUniqueness:
    """Column-level uniqueness, cardinality, and key candidacy metrics.

    Authoritative owner of per-column distinct counts, uniqueness ratios,
    duplicate value statistics, and single-column primary key candidacy grounded
    in ``cleanalytix.metrics.uniqueness.compute_column_uniqueness()`` and
    ``compute_identifier_uniqueness()``. Dataset-level duplicate row metrics
    belong exclusively to ``DatasetProfile``.
    """

    distinct_count: int = 0
    distinct_ratio: float = 0.0
    non_null_distinct_ratio: float = 0.0
    unique_percentage: float = 0.0
    duplicate_count: int = 0
    duplicate_percentage: float = 0.0
    is_unique: bool = False
    is_constant: bool = False
    is_near_constant: bool = False
    is_primary_key_candidate: bool = False
    top_frequency_ratio: float = 0.0
    top_value: Any = None
    mode: Any = None
    mode_percentage: Optional[float] = None

    @property
    def uniqueness_ratio(self) -> float:
        """Read-only alias returning ``non_null_distinct_ratio`` (or ``distinct_ratio`` if zero)."""
        return self.non_null_distinct_ratio if self.non_null_distinct_ratio > 0.0 else self.distinct_ratio

    @property
    def unique_count(self) -> int:
        """Read-only alias for ``distinct_count``."""
        return self.distinct_count

    @property
    def unique_percent(self) -> float:
        """Read-only alias for ``unique_percentage``."""
        return self.unique_percentage


@dataclass
class ColumnValidity:
    """Column-level validity metrics across type conformance, whitespace, and custom rules.

    Authoritative owner of semantic type error counts/percentages, leading/trailing
    whitespace contamination counts/percentages, and custom business-rule violation
    counts/percentages grounded in ``cleanalytix.metrics.validity``. All percentages
    are scaled relative to ``N_non_null`` (which belongs to ``ColumnCompleteness``).
    """

    inferred_type: str = "categorical"
    type_error_count: int = 0
    type_error_percentage: float = 0.0
    whitespace_count: int = 0
    whitespace_percentage: float = 0.0
    rule_error_count: int = 0
    rule_error_percentage: float = 0.0


@dataclass
class ColumnStatistical:
    """Column-level numerical statistical moments, quantiles, continuous entropy, and outliers.

    Authoritative computational owner of all numeric statistics grounded in
    ``cleanalytix.metrics.statistical.compute_numeric_statistics()``,
    ``compute_continuous_entropy()``, and ``detect_outliers()``.
    """

    count: int = 0
    infinite_count: int = 0
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p01: Optional[float] = None
    p05: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    iqr: Optional[float] = None
    mad: Optional[float] = None
    skewness: Optional[float] = None
    skewness_class: Optional[str] = None
    kurtosis: Optional[float] = None
    kurtosis_class: Optional[str] = None
    coefficient_of_variation: Optional[float] = None
    zero_count: int = 0
    zero_ratio: float = 0.0
    negative_count: int = 0
    negative_ratio: float = 0.0
    continuous_entropy: Optional[float] = 0.0
    continuous_shannon_entropy: Optional[float] = None
    continuous_hhi: Optional[float] = None
    continuous_gini_simpson: Optional[float] = None
    outlier_method: Optional[str] = "iqr"
    outlier_count: int = 0
    outlier_ratio: float = 0.0
    outlier_percentage: float = 0.0
    outlier_lower_bound: Optional[float] = None
    outlier_upper_bound: Optional[float] = None
    zscore_outliers: Dict[str, Any] = field(default_factory=dict)
    iqr_outliers: Dict[str, Any] = field(default_factory=dict)
    modified_zscore_outliers: Dict[str, Any] = field(default_factory=dict)

    @property
    def min_value(self) -> Optional[float]:
        """Read-only alias for ``min``."""
        return self.min

    @property
    def max_value(self) -> Optional[float]:
        """Read-only alias for ``max``."""
        return self.max

    @property
    def kurtosis_fisher(self) -> Optional[float]:
        """Read-only alias for ``kurtosis`` (Fisher's excess kurtosis)."""
        return self.kurtosis


@dataclass
class ColumnDistribution:
    """Column-level categorical distribution, concentration, and diversity metrics.

    Authoritative owner of categorical frequency distributions, discrete Shannon
    entropy, normalized categorical entropy, HHI, Gini-Simpson diversity, and
    rare category diagnostics grounded in
    ``cleanalytix.metrics.distribution.compute_categorical_distribution()``.
    Does not duplicate ``continuous_entropy`` (owned by ``ColumnStatistical``).
    """

    unique_categories: int = 0
    shannon_entropy: float = 0.0
    normalized_entropy: float = 0.0
    hhi: float = 0.0
    gini_simpson_diversity: float = 0.0
    top_category: Any = None
    top_category_ratio: float = 0.0
    top_category_percentage: float = 0.0
    is_dominated: bool = False
    rare_categories_count: int = 0
    rare_categories_ratio: float = 0.0
    rare_categories_percentage: float = 0.0
    top_frequencies: Dict[Any, Dict[str, Any]] = field(default_factory=dict)

    @property
    def mode(self) -> Any:
        """Read-only alias for ``top_category``."""
        return self.top_category

    @property
    def mode_percentage(self) -> float:
        """Read-only alias for ``top_category_percentage``."""
        return self.top_category_percentage


@dataclass
class ColumnTemporal:
    """Column-level temporal quality, freshness, cadence, and sequence metrics.

    Authoritative owner of datetime parsing, bounds, span, monotonicity, step/gap
    seconds, future date counts, and freshness grounded in
    ``cleanalytix.metrics.temporal.compute_temporal_metrics()``.
    """

    is_temporal: bool = False
    parse_rate: float = 0.0
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    min_timestamp: Optional[Any] = None
    max_timestamp: Optional[Any] = None
    date_range_days: Optional[int] = None
    time_span_days: Optional[float] = None
    is_monotonic_increasing: bool = False
    future_date_count: int = 0
    future_date_ratio: float = 0.0
    median_step_seconds: Optional[float] = None
    max_gap_seconds: Optional[float] = None
    freshness_days: Optional[float] = None
    inferred_cadence: Optional[str] = None
    freshness_status: Optional[str] = None


@dataclass
class ColumnDrift:
    """Column-level distribution drift and two-sample divergence metrics.

    Authoritative owner of baseline-vs-production drift outputs grounded in
    ``cleanalytix.metrics.drift.compute_numeric_drift()`` and
    ``compute_categorical_drift()``. Preserves distinct metrics for PSI, KL
    divergence, reverse KL divergence, Jensen-Shannon divergence, KS two-sample
    test, Wasserstein distance, and unseen categories.
    """

    psi: float = 0.0
    kl_divergence: float = 0.0
    reverse_kl_divergence: float = 0.0
    js_divergence: float = 0.0
    ks_statistic: Optional[float] = None
    ks_p_value: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    normalized_wasserstein: Optional[float] = None
    unseen_categories: List[Any] = field(default_factory=list)
    unseen_categories_count: int = 0
    unseen_categories_ratio: float = 0.0
    drift_detected: bool = False
    drift_severity: str = "none"
    sample_size_base: int = 0
    sample_size_prod: int = 0


@dataclass
class ColumnProfile:
    """Compositional analytical profile for a single dataset column.

    Composes domain-level structures by reference without duplicating individual
    domain metrics. Domains that do not apply to the column's semantic role or
    have not been evaluated remain ``None``.
    """

    name: str
    role: ColumnRole = "unknown"
    dtype: str = "object"
    semantic_type: Optional[SemanticType] = None
    completeness: Optional[ColumnCompleteness] = None
    uniqueness: Optional[ColumnUniqueness] = None
    validity: Optional[ColumnValidity] = None
    statistical: Optional[ColumnStatistical] = None
    distribution: Optional[ColumnDistribution] = None
    temporal: Optional[ColumnTemporal] = None
    drift: Optional[ColumnDrift] = None
    dtype_object: Optional[Any] = None

    @property
    def entropy(self) -> Optional[float]:
        """Resolve the canonical ``stat_entropy`` value from the applicable domain."""
        if self.role in {"continuous", "discrete"} and self.statistical is not None:
            return self.statistical.continuous_entropy
        if self.distribution is not None:
            return self.distribution.normalized_entropy
        if self.statistical is not None:
            return self.statistical.continuous_entropy
        return None


@dataclass
class RelationshipProfile:
    """Cross-column relationships, correlations, associations, and redundancies.

    Authoritative owner of cross-column relationship analysis grounded in
    ``cleanalytix.metrics.relationships`` and ``profiling.build_relationship_profile()``.
    """

    pearson_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    spearman_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    cramers_v: pd.DataFrame = field(default_factory=pd.DataFrame)
    high_correlations: pd.DataFrame = field(default_factory=pd.DataFrame)
    redundant_columns: pd.DataFrame = field(default_factory=pd.DataFrame)
    missingness_cooccurrence: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_keys: Dict[str, Any] = field(default_factory=dict)
    duplicate_columns: List[Tuple[str, str]] = field(default_factory=list)
    near_duplicate_columns: List[Dict[str, Any]] = field(default_factory=list)
    missingness_correlations: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pearson": self.pearson_correlations.to_dict() if not self.pearson_correlations.empty else {},
            "spearman": self.spearman_correlations.to_dict() if not self.spearman_correlations.empty else {},
            "cramers_v": self.cramers_v.to_dict() if not self.cramers_v.empty else {},
            "high_correlations": self.high_correlations.to_dict(orient="records") if not self.high_correlations.empty else [],
            "redundant_columns": self.redundant_columns.to_dict(orient="records") if not self.redundant_columns.empty else [],
            "missingness_cooccurrence": self.missingness_cooccurrence.to_dict(orient="records") if not self.missingness_cooccurrence.empty else [],
            "candidate_keys": self.candidate_keys,
        }


@dataclass
class DatasetProfile:
    """Dataset-level structural characteristics, health indicators, and quality summary.

    Authoritative owner of dataset-scoped metrics (row count, column count, memory
    footprint, complete/empty row counts, duplicate row counts, and dataset-level
    relationship and column compositions).
    """

    total_rows: int = 0
    total_columns: int = 0
    total_cells: int = 0
    memory_bytes: int = 0
    memory_mb: float = 0.0
    memory_per_row_bytes: float = 0.0
    null_density: float = 0.0
    complete_rows_count: int = 0
    complete_rows_percent: float = 0.0
    empty_rows_count: int = 0
    empty_rows_percent: float = 0.0
    duplicate_rows_count: int = 0
    duplicate_rows_percent: float = 0.0
    duplicate_columns_count: int = 0
    near_duplicate_columns_count: int = 0
    role_counts: Dict[str, int] = field(default_factory=dict)
    candidate_keys: List[str] = field(default_factory=list)
    composite_candidate_keys: List[Tuple[str, ...]] = field(default_factory=list)
    overall_score: float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    critical_anomalies_count: int = 0
    warning_anomalies_count: int = 0
    recommendations_count: int = 0
    dataset_name: str = "dataset"
    row_completeness_mean: float = 100.0
    unique_rows_count: int = 0
    overall_null_count: int = 0
    overall_null_percentage: float = 0.0
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    relationships: Optional[RelationshipProfile] = None

    def __post_init__(self) -> None:
        self.columns = _normalize_column_profiles(self.columns)

    @property
    def row_count(self) -> int:
        """Read-only alias for ``total_rows``."""
        return self.total_rows

    @property
    def column_count(self) -> int:
        """Read-only alias for ``total_columns``."""
        return self.total_columns

    @property
    def cell_count(self) -> int:
        """Read-only alias for ``total_cells``."""
        return self.total_cells

    @property
    def memory_usage_bytes(self) -> int:
        """Read-only alias for ``memory_bytes``."""
        return self.memory_bytes

    @property
    def duplicate_row_count(self) -> int:
        """Read-only alias for ``duplicate_rows_count``."""
        return self.duplicate_rows_count

    @property
    def duplicate_row_percentage(self) -> float:
        """Read-only alias for ``duplicate_rows_percent``."""
        return self.duplicate_rows_percent

    @property
    def column_order(self) -> List[str]:
        """Ordered list of column names in ``columns``."""
        return list(self.columns.keys())

    @property
    def column_profiles(self) -> List[ColumnProfile]:
        """Ordered list of ``ColumnProfile`` objects in ``columns``."""
        return list(self.columns.values())

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        """Retrieve a ``ColumnProfile`` by column name, or ``None`` if absent."""
        return self.columns.get(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "total_cells": self.total_cells,
            "memory_bytes": self.memory_bytes,
            "memory_mb": round(self.memory_mb, 4),
            "memory_per_row_bytes": round(self.memory_per_row_bytes, 2),
            "null_density": round(self.null_density, 4),
            "complete_rows_count": self.complete_rows_count,
            "complete_rows_percent": round(self.complete_rows_percent, 2),
            "empty_rows_count": self.empty_rows_count,
            "empty_rows_percent": round(self.empty_rows_percent, 2),
            "duplicate_rows_count": self.duplicate_rows_count,
            "duplicate_rows_percent": round(self.duplicate_rows_percent, 2),
            "duplicate_columns_count": self.duplicate_columns_count,
            "near_duplicate_columns_count": self.near_duplicate_columns_count,
            "role_counts": self.role_counts,
            "candidate_keys": self.candidate_keys,
            "composite_candidate_keys": [list(k) for k in self.composite_candidate_keys],
            "overall_score": round(self.overall_score, 2),
            "dimension_scores": self.dimension_scores,
            "critical_anomalies_count": self.critical_anomalies_count,
            "warning_anomalies_count": self.warning_anomalies_count,
            "recommendations_count": self.recommendations_count,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a single-row DataFrame summary of dataset-level metrics."""
        d = self.to_dict()
        # Flatten composite keys for clean tabular representation
        d["composite_candidate_keys"] = str(d["composite_candidate_keys"])
        d["candidate_keys"] = str(d["candidate_keys"])
        d["role_counts"] = str(d["role_counts"])
        d["dimension_scores"] = str(d["dimension_scores"])
        return pd.DataFrame([d])


@dataclass
class ProfileContext:
    """Unified typed internal representation composing dataset, column, and relationship profiles.

    Acts as the shared source of truth across profiling, scoring, insights,
    cleaning recommendations, and legacy compatibility adaptation in Cleanalytix v2.0.
    Preserves exact column ordering and prevents silent loss of duplicate column names.
    """

    dataset: DatasetProfile = field(default_factory=DatasetProfile)
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    relationships: Optional[RelationshipProfile] = None
    reference_profile: Optional[Union[Dict[str, Any], "ProfileContext"]] = None
    analysis_level: Union[AnalysisLevel, str] = AnalysisLevel.STANDARD

    def __post_init__(self) -> None:
        self.columns = _normalize_column_profiles(self.columns)
        if not self.columns and self.dataset.columns:
            self.columns = dict(self.dataset.columns)
        elif self.columns and not self.dataset.columns:
            self.dataset.columns = dict(self.columns)
        if self.relationships is None and self.dataset.relationships is not None:
            self.relationships = self.dataset.relationships
        elif self.relationships is not None and self.dataset.relationships is None:
            self.dataset.relationships = self.relationships

    @property
    def dataset_name(self) -> str:
        """Return the dataset identifier from ``dataset.dataset_name``."""
        return self.dataset.dataset_name

    @property
    def column_order(self) -> List[str]:
        """Ordered list of column names preserved in exact insertion/sequence order."""
        return list(self.columns.keys())

    @property
    def column_profiles(self) -> List[ColumnProfile]:
        """Ordered list of ``ColumnProfile`` objects in exact insertion/sequence order."""
        return list(self.columns.values())

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        """Retrieve a ``ColumnProfile`` by column name, or ``None`` if absent."""
        return self.columns.get(name)


class PipelineBlock(TypedDict, total=False):
    """Artifacts returned for one analysis branch."""

    dirty_scores: pd.DataFrame
    meta_before_cleaning: pd.DataFrame
    main_metrics_before: pd.DataFrame
    summarized_before: pd.DataFrame
    recommendations: pd.DataFrame
    insights: list[InsightItem]
    quality_dimensions: dict[str, DimensionScore]
    advanced_metrics: dict[str, Any]
    cleaned_scores: pd.DataFrame
    cleaned_datasets: list[pd.DataFrame]
    meta_after_cleaning: pd.DataFrame
    change_log: pd.DataFrame
    summarized_after: pd.DataFrame
    main_metrics_after: pd.DataFrame


class _LazyPipelineBlock(dict):
    """PipelineBlock dictionary that lazily materializes ``meta_before_cleaning``."""

    def __init__(
        self,
        *args: Any,
        _meta_context: Optional[ProfileContext] = None,
        _owner_result: Optional[CleanalytixResult] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._meta_context = _meta_context
        self._owner_result = _owner_result
        self._lazy_meta_pending = (
            _meta_context is not None and self.get("meta_before_cleaning") is None
        )
        if self._lazy_meta_pending:
            super().__setitem__("meta_before_cleaning", None)

    def _materialize_meta(self) -> pd.DataFrame:
        if self._lazy_meta_pending and self._meta_context is not None:
            if (
                self._owner_result is not None
                and self._owner_result._context is self._meta_context
            ):
                meta_df = self._owner_result._materialize_lazy_metadata()
            else:
                from cleanalytix.legacy.adapter import LegacyMetaAdapter

                meta_df = LegacyMetaAdapter.context_to_meta_df(self._meta_context)
            super().__setitem__("meta_before_cleaning", meta_df)
            self._lazy_meta_pending = False
            return meta_df
        val = super().get("meta_before_cleaning")
        return val if isinstance(val, pd.DataFrame) else pd.DataFrame()

    def __getitem__(self, key: Any) -> Any:
        if key == "meta_before_cleaning" and getattr(self, "_lazy_meta_pending", False):
            return self._materialize_meta()
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if key == "meta_before_cleaning" and getattr(self, "_lazy_meta_pending", False):
            return self._materialize_meta()
        return super().get(key, default)


class CleanalytixResult(dict):
    """Comprehensive Data Quality Intelligence and Advanced Analytics result object.

    Provides both dictionary-like access and typed attribute access across
    all 10 analytical output domains.
    """

    def __init__(
        self,
        overall_score: float = 0.0,
        quality_dimensions: Optional[Dict[str, DimensionScore]] = None,
        column_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[pd.DataFrame] = None,
        recommendations: Optional[Union[List[InsightItem], List[RecommendationItem]]] = None,
        insights: Optional[List[InsightItem]] = None,
        metrics_table: Optional[pd.DataFrame] = None,
        drift_summary: Optional[Dict[str, Any]] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        column_profile: Optional[pd.DataFrame] = None,
        statistical_profile: Optional[pd.DataFrame] = None,
        scoring_matrix: Optional[pd.DataFrame] = None,
        dimension_summary: Optional[pd.DataFrame] = None,
        metric_inventory: Optional[pd.DataFrame] = None,
        relationships: Optional[RelationshipProfile] = None,
        temporal_profile: Optional[pd.DataFrame] = None,
        drift_profile: Optional[pd.DataFrame] = None,
        anomalies: Optional[pd.DataFrame] = None,
        recommendations_table: Optional[pd.DataFrame] = None,
        scoring_audit_trail: Optional[str] = None,
        base_data: Optional[PipelineBlock] = None,
        prod_data: Optional[Union[PipelineBlock, Dict[str, Any]]] = None,
        analysis_level: str = "standard",
        context: Optional[ProfileContext] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._context: Optional[ProfileContext] = context
        self._lazy_metadata_pending: bool = metadata is None and context is not None
        self["overall_score"] = float(overall_score)
        self["quality_dimensions"] = quality_dimensions or {}
        self["column_scores"] = column_scores or {}
        if metadata is not None:
            super().__setitem__("metadata", metadata)
        elif context is None:
            super().__setitem__("metadata", pd.DataFrame())
        self["recommendations"] = recommendations or []
        self["insights"] = insights or []
        self["metrics_table"] = metrics_table if metrics_table is not None else pd.DataFrame()
        self["drift_summary"] = drift_summary or {}
        if dataset_profile is not None:
            self["dataset_profile"] = dataset_profile
        elif context is not None:
            self["dataset_profile"] = context.dataset
        else:
            self["dataset_profile"] = DatasetProfile()
        self["column_profile"] = column_profile if column_profile is not None else pd.DataFrame()
        if self["column_profile"].empty and self["column_profile"].index.name is None:
            self["column_profile"].index.name = "column_name"

        self["statistical_profile"] = statistical_profile if statistical_profile is not None else pd.DataFrame()
        if self["statistical_profile"].empty and self["statistical_profile"].index.name is None:
            self["statistical_profile"].index.name = "column_name"

        self["scoring_matrix"] = scoring_matrix if scoring_matrix is not None else pd.DataFrame()
        if self["scoring_matrix"].empty and self["scoring_matrix"].index.name is None:
            self["scoring_matrix"].index.name = "column_name"
        self["dimension_summary"] = dimension_summary if dimension_summary is not None else pd.DataFrame()
        self["metric_inventory"] = metric_inventory if metric_inventory is not None else pd.DataFrame()
        if relationships is not None:
            self["relationships"] = relationships
        elif context is not None and context.relationships is not None:
            self["relationships"] = context.relationships
        else:
            self["relationships"] = RelationshipProfile()
        self["temporal_profile"] = temporal_profile if temporal_profile is not None else pd.DataFrame()
        self["drift_profile"] = drift_profile
        self["anomalies"] = anomalies if anomalies is not None else pd.DataFrame()
        self["recommendations_table"] = recommendations_table if recommendations_table is not None else pd.DataFrame()
        self["scoring_audit_trail"] = scoring_audit_trail or ""
        self["analysis_level"] = analysis_level

        # Synthesize health_summary DataFrame (or use passed health_summary if provided)
        if "health_summary" in kwargs and kwargs["health_summary"] is not None:
            self["health_summary"] = kwargs.pop("health_summary")
        elif not self["dataset_profile"].to_dataframe().empty:
            dp = self["dataset_profile"]
            health_sum_df = pd.DataFrame([{
                "dataset_name": "dataset",
                "avg_missing_percent": round(dp.null_density * 100.0, 2),
                "avg_duplicate_percent": round(dp.duplicate_rows_percent, 2),
                "total_outliers": int(self["column_profile"]["outlier_count"].sum()) if not self["column_profile"].empty and "outlier_count" in self["column_profile"].columns else 0,
                "total_whitespace_issues": int(self["column_profile"]["whitespace_count"].sum()) if not self["column_profile"].empty and "whitespace_count" in self["column_profile"].columns else 0,
            }])
            self["health_summary"] = health_sum_df
        else:
            self["health_summary"] = pd.DataFrame()

        # Synthesize base_data if not provided
        if base_data is not None:
            self["base_data"] = base_data
        else:
            recs_df = self["recommendations_table"]
            if recs_df.empty and self["recommendations"]:
                recs_df = pd.DataFrame([r.to_dict() for r in self["recommendations"]])

            self["base_data"] = _LazyPipelineBlock(
                {
                    "dirty_scores": pd.DataFrame([{"Dataset_name": "dataset", "DQ_Score": overall_score}]),
                    "meta_before_cleaning": super().get("metadata"),
                    "recommendations": recs_df,
                    "insights": self["insights"],
                    "quality_dimensions": self["quality_dimensions"],
                    "cleaned_scores": pd.DataFrame(),
                    "cleaned_datasets": [],
                    "meta_after_cleaning": pd.DataFrame(),
                    "change_log": pd.DataFrame(),
                    "summarized_before": self["health_summary"],
                    "summarized_after": pd.DataFrame(),
                    "main_metrics_before": self["metrics_table"],
                    "main_metrics_after": pd.DataFrame(),
                },
                _meta_context=context if metadata is None else None,
                _owner_result=self,
            )

        self["prod_data"] = prod_data or {}

        # Store any extra kwargs
        for k, v in kwargs.items():
            self[k] = v

    def _materialize_lazy_metadata(self) -> pd.DataFrame:
        if self._lazy_metadata_pending and self._context is not None:
            from cleanalytix.legacy.adapter import LegacyMetaAdapter

            meta_df = LegacyMetaAdapter.context_to_meta_df(self._context)
            super().__setitem__("metadata", meta_df)
            self._lazy_metadata_pending = False
            base_block = super().get("base_data")
            if isinstance(base_block, _LazyPipelineBlock) and base_block._meta_context is self._context:
                dict.__setitem__(base_block, "meta_before_cleaning", meta_df)
                base_block._lazy_meta_pending = False
            elif isinstance(base_block, dict) and dict.get(base_block, "meta_before_cleaning") is None:
                base_block["meta_before_cleaning"] = meta_df
            return meta_df
        return super().__getitem__("metadata")

    def __getitem__(self, key: Any) -> Any:
        if key == "metadata" and getattr(self, "_lazy_metadata_pending", False):
            return self._materialize_lazy_metadata()
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        if key == "metadata" and getattr(self, "_lazy_metadata_pending", False):
            return self._materialize_lazy_metadata()
        return super().get(key, default)

    @property
    def context(self) -> Optional[ProfileContext]:
        return self._context

    @property
    def overall_score(self) -> float:
        return self["overall_score"]

    @property
    def quality_dimensions(self) -> Dict[str, DimensionScore]:
        return self["quality_dimensions"]

    @property
    def column_scores(self) -> Dict[str, float]:
        return self["column_scores"]

    @property
    def metadata(self) -> pd.DataFrame:
        return self["metadata"]

    @property
    def meta_df(self) -> pd.DataFrame:
        return self["metadata"]

    @property
    def dataset_profile(self) -> DatasetProfile:
        return self["dataset_profile"]

    @property
    def column_profile(self) -> pd.DataFrame:
        return self["column_profile"]

    @property
    def statistical_profile(self) -> pd.DataFrame:
        return self["statistical_profile"]

    @property
    def scoring_matrix(self) -> pd.DataFrame:
        return self["scoring_matrix"]

    @property
    def dimension_summary(self) -> pd.DataFrame:
        return self["dimension_summary"]

    @property
    def metrics_table(self) -> pd.DataFrame:
        return self["metrics_table"]

    @property
    def metric_inventory(self) -> pd.DataFrame:
        return self["metric_inventory"]

    @property
    def relationships(self) -> RelationshipProfile:
        return self["relationships"]

    @property
    def correlations(self) -> pd.DataFrame:
        """Convenience property returning Pearson linear correlation matrix."""
        return self.relationships.pearson_correlations

    @property
    def candidate_keys(self) -> Dict[str, Any]:
        """Convenience property returning detected single and composite candidate keys."""
        return self.relationships.candidate_keys

    @property
    def temporal_profile(self) -> pd.DataFrame:
        return self["temporal_profile"]

    @property
    def drift_profile(self) -> Optional[pd.DataFrame]:
        return self["drift_profile"]

    @property
    def anomalies(self) -> pd.DataFrame:
        return self["anomalies"]

    @property
    def recommendations_table(self) -> pd.DataFrame:
        return self["recommendations_table"]

    @property
    def recommendations(self) -> List[Any]:
        return self["recommendations"]

    @property
    def insights(self) -> List[InsightItem]:
        return self["insights"]

    @property
    def drift_summary(self) -> Dict[str, Any]:
        return self["drift_summary"]

    @property
    def scoring_audit_trail(self) -> str:
        return self["scoring_audit_trail"]

    @property
    def health_summary(self) -> pd.DataFrame:
        return self["health_summary"]

    @property
    def base_data(self) -> PipelineBlock:
        return self["base_data"]

    @property
    def prod_data(self) -> Union[PipelineBlock, Dict[str, Any]]:
        return self["prod_data"]

    def to_dataframe(self, view: str = "columns") -> pd.DataFrame:
        """Extract a structured analytical table by domain view name.

        Parameters
        ----------
        view : str, default="columns"
            Analytical view to export:
            - "columns" / "column_profile": Per-column profiling metrics table.
            - "scoring" / "scoring_matrix": Per-column scoring penalty matrix.
            - "dimensions" / "dimension_summary": 6-dimension scoreboard with weights and contributions.
            - "statistics" / "statistical_profile": Deep exploratory statistical moments and quantiles.
            - "metrics" / "metrics_table": Standardized percentage-scaled DQ metrics table.
            - "inventory" / "metric_inventory": Exhaustive metric audit ledger with thresholds and statuses.
            - "relationships" / "correlations": Pearson correlation matrix.
            - "temporal" / "temporal_profile": Datetime coverage, cadence, and gaps.
            - "drift" / "drift_profile": Baseline drift divergence and statistical tests.
            - "anomalies": Tabular view of detected quality anomalies.
            - "recommendations" / "recommendations_table": Tabular actionable recommendations.
            - "dataset" / "dataset_profile": Single-row dataset overview summary.

        Returns
        -------
        pd.DataFrame
            The requested analytical DataFrame.

        Raises
        ------
        ValueError
            If view name is unrecognized.
        """
        v = str(view).strip().lower()
        if v in {"columns", "column_profile"}:
            return self.column_profile
        elif v in {"scoring", "scoring_matrix"}:
            return self.scoring_matrix
        elif v in {"dimensions", "dimension_summary"}:
            return self.dimension_summary
        elif v in {"statistics", "statistical_profile"}:
            return self.statistical_profile
        elif v in {"metrics", "metrics_table"}:
            return self.metrics_table
        elif v in {"inventory", "metric_inventory"}:
            return self.metric_inventory
        elif v in {"relationships", "correlations", "pearson"}:
            return self.correlations
        elif v in {"temporal", "temporal_profile"}:
            return self.temporal_profile
        elif v in {"drift", "drift_profile"}:
            return self.drift_profile if self.drift_profile is not None else pd.DataFrame()
        elif v in {"anomalies", "insights"}:
            return self.anomalies
        elif v in {"recommendations", "recommendations_table"}:
            return self.recommendations_table
        elif v in {"dataset", "dataset_profile"}:
            return self.dataset_profile.to_dataframe()
        else:
            valid_views = [
                "columns", "scoring", "dimensions", "statistics", "metrics",
                "inventory", "relationships", "temporal", "drift", "anomalies",
                "recommendations", "dataset",
            ]
            raise ValueError(f"Unknown view '{view}'. Valid views are: {valid_views}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to JSON-serializable dictionary."""
        return {
            "overall_score": self.overall_score,
            "dimensions": {k: v.to_dict() for k, v in self.quality_dimensions.items()},
            "column_scores": self.column_scores,
            "dataset_profile": self.dataset_profile.to_dict(),
            "critical_insights_count": sum(1 for i in self.insights if getattr(i, "severity", None) == InsightSeverity.CRITICAL),
            "warning_insights_count": sum(1 for i in self.insights if getattr(i, "severity", None) == InsightSeverity.WARNING),
            "recommendations_count": len(self.recommendations),
        }

    def summary(self) -> Dict[str, Any]:
        """Return a compact JSON-serializable health and score summary."""
        dim_summary = {}
        for k, v in self.quality_dimensions.items():
            if isinstance(v, DimensionScore):
                dim_summary[k] = {
                    "score": v.score,
                    "weight": v.weight,
                    "effective_weight": v.effective_weight,
                    "applicable": v.applicable,
                    "reason": v.reason,
                }
            elif isinstance(v, dict):
                dim_summary[k] = {
                    "score": v.get("score"),
                    "weight": v.get("weight"),
                    "effective_weight": v.get("effective_weight", v.get("weight")),
                    "applicable": v.get("applicable", True),
                    "reason": v.get("reason"),
                }

        return {
            "overall_score": self.overall_score,
            "dimensions": dim_summary,
            "critical_insights_count": sum(1 for i in self.insights if getattr(i, "severity", None) == InsightSeverity.CRITICAL),
            "warning_insights_count": sum(1 for i in self.insights if getattr(i, "severity", None) == InsightSeverity.WARNING),
            "recommendations_count": len(self.recommendations),
        }


DQPipelineResult = Dict[str, Any]
"""Type alias for pipeline result dictionary."""


@dataclass
class CleanalytixConfig:
    """Configuration options for Cleanalytix analysis."""

    analysis_level: str = "standard"
    score_mode: str = "linear"
    thresholds: Dict[str, float] = field(default_factory=dict)
    weights: Optional[List[float]] = None
    dimension_weights: Optional[Dict[Union[QualityDimension, str], float]] = None
    metric_weights: Optional[Dict[str, float]] = None
    rules: Optional[Rules] = None
    consistency_rules: Optional[List[Union[ConsistencyRule, Dict[str, Any], Any]]] = None
    timeliness_config: Optional[Union[TimelinessConfig, Dict[str, Any]]] = None
    identifier_columns: Optional[List[str]] = None
    max_correlation_columns: int = 100
    random_state: int = 42
