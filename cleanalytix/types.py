"""Typing definitions, enums, dataclasses, and result containers for Cleanalytix.

This module defines public protocols, enums, dataclasses, and structured result
objects for Cleanalytix Data Quality Intelligence & Advanced Analytics.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Protocol, Sequence, TypedDict, Union
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


class CleanalytixResult(dict):
    """Structured result object for Cleanalytix analysis.

    Provides both dictionary-like access and attribute access for clean, modern integration.
    """

    def __init__(
        self,
        overall_score: float = 0.0,
        quality_dimensions: Optional[Dict[str, DimensionScore]] = None,
        column_scores: Optional[Dict[str, float]] = None,
        metadata: Optional[pd.DataFrame] = None,
        recommendations: Optional[List[InsightItem]] = None,
        insights: Optional[List[InsightItem]] = None,
        metrics_table: Optional[pd.DataFrame] = None,
        drift_summary: Optional[Dict[str, Any]] = None,
        base_data: Optional[PipelineBlock] = None,
        prod_data: Optional[Union[PipelineBlock, Dict[str, Any]]] = None,
        analysis_level: str = "standard",
        **kwargs: Any,
    ):
        super().__init__()
        self["overall_score"] = overall_score
        self["quality_dimensions"] = quality_dimensions or {}
        self["column_scores"] = column_scores or {}
        self["metadata"] = metadata if metadata is not None else pd.DataFrame()
        self["recommendations"] = recommendations or []
        self["insights"] = insights or []
        self["metrics_table"] = metrics_table if metrics_table is not None else pd.DataFrame()
        self["drift_summary"] = drift_summary or {}
        self["analysis_level"] = analysis_level

        # Synthesize base_data if not provided
        if base_data is not None:
            self["base_data"] = base_data
        else:
            self["base_data"] = {
                "dirty_scores": pd.DataFrame([{"Dataset_name": "dataset", "DQ_Score": overall_score}]),
                "meta_before_cleaning": self["metadata"],
                "recommendations": pd.DataFrame([r.to_dict() for r in self["recommendations"]]) if self["recommendations"] else pd.DataFrame(),
                "insights": self["insights"],
                "quality_dimensions": self["quality_dimensions"],
                "cleaned_scores": pd.DataFrame(),
                "cleaned_datasets": [],
                "meta_after_cleaning": pd.DataFrame(),
                "change_log": pd.DataFrame(),
                "summarized_before": pd.DataFrame(),
                "summarized_after": pd.DataFrame(),
                "main_metrics_before": self["metrics_table"],
                "main_metrics_after": pd.DataFrame(),
            }

        self["prod_data"] = prod_data or {}

        # Store any extra kwargs
        for k, v in kwargs.items():
            self[k] = v

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
    def recommendations(self) -> List[InsightItem]:
        return self["recommendations"]

    @property
    def insights(self) -> List[InsightItem]:
        return self["insights"]

    @property
    def metrics_table(self) -> pd.DataFrame:
        return self["metrics_table"]

    @property
    def drift_summary(self) -> Dict[str, Any]:
        return self["drift_summary"]

    @property
    def base_data(self) -> PipelineBlock:
        return self["base_data"]

    @property
    def prod_data(self) -> Union[PipelineBlock, Dict[str, Any]]:
        return self["prod_data"]

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
            "critical_insights_count": sum(1 for i in self.insights if i.severity == InsightSeverity.CRITICAL),
            "warning_insights_count": sum(1 for i in self.insights if i.severity == InsightSeverity.WARNING),
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
