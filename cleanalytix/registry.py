"""Standardized Metric Registry for Cleanalytix Data Quality Intelligence.

Defines the authoritative catalog of all data quality, statistical, distributional,
relational, temporal, and drift metrics supported by Cleanalytix.

Every registered metric contains consistent, structured metadata:
- Canonical metric identifier and human-readable display name.
- Analytical domain and associated Data Quality Dimension (if applicable).
- Applicable column roles and data types.
- Raw value unit and measurement scale.
- Whether it contributes directly to the 6-dimension Data Quality score.
- Mathematical calculation method and relevant algorithm parameters.
- Default threshold and interpretation semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from cleanalytix.types import QualityDimension


class MetricDomain(str, Enum):
    """Analytical domains for Cleanalytix intelligence."""
    DATASET_PROFILING = "dataset_profiling"
    COLUMN_PROFILING = "column_profiling"
    COMPLETENESS = "completeness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    STABILITY = "stability"
    STATISTICAL_INTELLIGENCE = "statistical_intelligence"
    DISTRIBUTION = "distribution"
    RELATIONSHIPS = "relationships"
    DRIFT_ANALYSIS = "drift_analysis"
    ANOMALY_DIAGNOSTICS = "anomaly_diagnostics"
    RECOMMENDATIONS = "recommendations"


class MetricScope(str, Enum):
    """Evaluation scope of a metric."""
    DATASET = "dataset"
    COLUMN = "column"
    PAIR = "pair"


class MetricValueStatus(str, Enum):
    """Evaluation and calculation status of a metric."""
    PASS = "PASS"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    INFO = "INFO"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_CALCULATED = "NOT_CALCULATED"


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable definition and metadata specification for an analytical metric."""

    name: str
    display_name: str
    description: str
    domain: MetricDomain
    dimension: Optional[QualityDimension]
    scope: MetricScope
    applicable_types: List[str]
    unit: str
    contributes_to_dq_score: bool
    calculation_method: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    default_threshold: Optional[float] = None
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert definition to dictionary."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "domain": self.domain.value,
            "dimension": self.dimension.value if self.dimension else None,
            "scope": self.scope.value,
            "applicable_types": self.applicable_types,
            "unit": self.unit,
            "contributes_to_dq_score": self.contributes_to_dq_score,
            "calculation_method": self.calculation_method,
            "parameters": self.parameters,
            "default_threshold": self.default_threshold,
            "interpretation": self.interpretation,
        }


# =============================================================================
# AUTHORITATIVE METRIC REGISTRY
# =============================================================================

METRIC_CATALOG: Dict[str, MetricDefinition] = {
    # -------------------------------------------------------------------------
    # COMPLETENESS METRICS
    # -------------------------------------------------------------------------
    "completeness_cell": MetricDefinition(
        name="completeness_cell",
        display_name="Cell Missingness Percent",
        description="Percentage of missing (null, NaN, None, NaT) cell values relative to total values.",
        domain=MetricDomain.COMPLETENESS,
        dimension=QualityDimension.COMPLETENESS,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric", "categorical", "datetime", "boolean", "text"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (null_count / max(1, total_count))",
        default_threshold=20.0,
        interpretation="Lower is better. 0% indicates fully populated data.",
    ),
    "completeness_row": MetricDefinition(
        name="completeness_row",
        display_name="Incomplete Rows Percent",
        description="Percentage of rows containing at least one missing field.",
        domain=MetricDomain.COMPLETENESS,
        dimension=QualityDimension.COMPLETENESS,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (count(rows with any null) / max(1, total_rows))",
        default_threshold=20.0,
        interpretation="Lower is better. 0% indicates all records are completely filled.",
    ),
    "complete_rows_percent": MetricDefinition(
        name="complete_rows_percent",
        display_name="Complete Rows Percent",
        description="Percentage of dataset rows that contain zero missing values across all columns.",
        domain=MetricDomain.DATASET_PROFILING,
        dimension=QualityDimension.COMPLETENESS,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="%",
        contributes_to_dq_score=False,
        calculation_method="100.0 * (count(rows with 0 nulls) / max(1, total_rows))",
        default_threshold=80.0,
        interpretation="Higher is better. 100% indicates completely filled tabular records.",
    ),
    "empty_rows_percent": MetricDefinition(
        name="empty_rows_percent",
        display_name="Completely Empty Rows Percent",
        description="Percentage of dataset rows where every single column is null.",
        domain=MetricDomain.DATASET_PROFILING,
        dimension=QualityDimension.COMPLETENESS,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="%",
        contributes_to_dq_score=False,
        calculation_method="100.0 * (count(rows where all cols are null) / max(1, total_rows))",
        default_threshold=0.0,
        interpretation="Lower is better. 0% indicates no entirely blank rows.",
    ),
    "missingness_cooccurrence_jaccard": MetricDefinition(
        name="missingness_cooccurrence_jaccard",
        display_name="Missingness Co-occurrence (Jaccard)",
        description="Observed empirical Jaccard similarity of missingness masks between column pairs. Note: Indicates empirical co-occurrence, not statistical proof of MCAR/MAR/MNAR.",
        domain=MetricDomain.RELATIONSHIPS,
        dimension=None,
        scope=MetricScope.PAIR,
        applicable_types=["all"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="both_missing_count / (either_missing_count + 1e-9)",
        default_threshold=0.20,
        interpretation="Values >= 0.20 indicate pairs of columns frequently missing in the same rows.",
    ),

    # -------------------------------------------------------------------------
    # VALIDITY METRICS
    # -------------------------------------------------------------------------
    "validity_type_errors": MetricDefinition(
        name="validity_type_errors",
        display_name="Type Conformance Error Percent",
        description="Percentage of non-null values that conflict with the column's inferred semantic type.",
        domain=MetricDomain.VALIDITY,
        dimension=QualityDimension.VALIDITY,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric", "datetime", "categorical"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (type_error_count / max(1, non_null_count))",
        default_threshold=2.0,
        interpretation="Lower is better. 0% indicates perfect data type conformity.",
    ),
    "validity_whitespace": MetricDefinition(
        name="validity_whitespace",
        display_name="Whitespace Issue Percent",
        description="Percentage of string/text values with leading or trailing whitespace.",
        domain=MetricDomain.VALIDITY,
        dimension=QualityDimension.VALIDITY,
        scope=MetricScope.COLUMN,
        applicable_types=["categorical", "text", "identifier"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (whitespace_count / max(1, non_null_count))",
        default_threshold=5.0,
        interpretation="Lower is better. 0% indicates clean, trimmed string values.",
    ),
    "validity_rules": MetricDefinition(
        name="validity_rules",
        display_name="Custom Business Rule Violation Percent",
        description="Percentage of column values that fail user-defined business validation predicates.",
        domain=MetricDomain.VALIDITY,
        dimension=QualityDimension.VALIDITY,
        scope=MetricScope.COLUMN,
        applicable_types=["all"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (rule_violation_count / max(1, non_null_count))",
        default_threshold=2.0,
        interpretation="Lower is better. 0% indicates zero business rule violations.",
    ),
    "validity_outliers": MetricDefinition(
        name="validity_outliers",
        display_name="Statistical Outlier Percent",
        description="Percentage of numeric values identified as statistical outliers via skewness-adaptive heuristics (Z-score, IQR fences, or Modified Z-score). Note: Outlier agreement reflects heuristic consensus, not ground-truth classification.",
        domain=MetricDomain.VALIDITY,
        dimension=QualityDimension.VALIDITY,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (outlier_count / max(1, non_null_count))",
        default_threshold=10.0,
        interpretation="Lower is better. High outlier ratios warrant domain plausibility review.",
    ),

    # -------------------------------------------------------------------------
    # UNIQUENESS METRICS
    # -------------------------------------------------------------------------
    "uniqueness_duplicate_rows": MetricDefinition(
        name="uniqueness_duplicate_rows",
        display_name="Duplicate Rows Percent",
        description="Percentage of redundant duplicate records in the dataset relative to total rows.",
        domain=MetricDomain.UNIQUENESS,
        dimension=QualityDimension.UNIQUENESS,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (duplicate_row_count / max(1, total_rows))",
        default_threshold=1.0,
        interpretation="Lower is better. 0% indicates strictly unique tabular rows.",
    ),
    "uniqueness_identifiers": MetricDefinition(
        name="uniqueness_identifiers",
        display_name="Identifier Column Duplication Percent",
        description="Average duplication percentage across identified or configured primary key/identifier columns.",
        domain=MetricDomain.UNIQUENESS,
        dimension=QualityDimension.UNIQUENESS,
        scope=MetricScope.COLUMN,
        applicable_types=["identifier", "numeric", "text"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (non_unique_id_count / max(1, non_null_count))",
        default_threshold=0.0,
        interpretation="Lower is better. Primary/identifier keys must be 100% unique (0% duplicate).",
    ),

    # -------------------------------------------------------------------------
    # CONSISTENCY METRICS
    # -------------------------------------------------------------------------
    "consistency_rules": MetricDefinition(
        name="consistency_rules",
        display_name="Cross-Column Consistency Rule Violations",
        description="Percentage of rows failing cross-column logical business rules or multi-column relational invariants.",
        domain=MetricDomain.CONSISTENCY,
        dimension=QualityDimension.CONSISTENCY,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (cross_column_violations / max(1, total_rows))",
        default_threshold=2.0,
        interpretation="Lower is better. 0% indicates full multi-column relational consistency.",
    ),

    # -------------------------------------------------------------------------
    # TIMELINESS METRICS
    # -------------------------------------------------------------------------
    "timeliness_freshness": MetricDefinition(
        name="timeliness_freshness",
        display_name="Data Freshness (Elapsed Days)",
        description="Number of days elapsed between the most recent timestamp in the dataset and the reference evaluation time.",
        domain=MetricDomain.TIMELINESS,
        dimension=QualityDimension.TIMELINESS,
        scope=MetricScope.COLUMN,
        applicable_types=["datetime"],
        unit="days",
        contributes_to_dq_score=True,
        calculation_method="(reference_time - max_timestamp).total_seconds() / 86400.0",
        default_threshold=30.0,
        interpretation="Lower is better. Represents data staleness / latency.",
    ),
    "timeliness_future_dates": MetricDefinition(
        name="timeliness_future_dates",
        display_name="Future Date Percent",
        description="Percentage of timestamp records that post-date the reference evaluation time.",
        domain=MetricDomain.TIMELINESS,
        dimension=QualityDimension.TIMELINESS,
        scope=MetricScope.COLUMN,
        applicable_types=["datetime"],
        unit="%",
        contributes_to_dq_score=True,
        calculation_method="100.0 * (count(timestamps > reference_time) / max(1, total_timestamps))",
        default_threshold=0.1,
        interpretation="Lower is better. 0% indicates no impossible future timestamps.",
    ),
    "timeliness_sequence": MetricDefinition(
        name="timeliness_sequence",
        display_name="Timestamp Sequence Monotonicity",
        description="Verification that timestamp records are ordered monotonically non-decreasing when sequence integrity is configured.",
        domain=MetricDomain.TIMELINESS,
        dimension=QualityDimension.TIMELINESS,
        scope=MetricScope.COLUMN,
        applicable_types=["datetime"],
        unit="boolean/score",
        contributes_to_dq_score=True,
        calculation_method="series.is_monotonic_increasing",
        default_threshold=1.0,
        interpretation="100% indicates chronological ordering without time-inversions.",
    ),

    # -------------------------------------------------------------------------
    # STABILITY / DRIFT METRICS
    # -------------------------------------------------------------------------
    "stability_psi": MetricDefinition(
        name="stability_psi",
        display_name="Population Stability Index (PSI)",
        description="Binned Population Stability Index (PSI) evaluating distribution shift using 10 quantile bins for continuous distributions and a discrete 3-bin surrogate approximation for constant baselines.",
        domain=MetricDomain.STABILITY,
        dimension=QualityDimension.STABILITY,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric", "categorical"],
        unit="PSI",
        contributes_to_dq_score=True,
        calculation_method="sum((P_b - Q_b) * ln(P_b / Q_b)) over 10 quantile bins with Laplace smoothing (or discrete 3-bin surrogate approximation for constant baselines)",
        parameters={"binning_strategy": "10_quantile_bins", "constant_baseline_method": "discrete_3bin_surrogate", "epsilon": 1e-4},
        default_threshold=0.25,
        interpretation="< 0.10: Insignificant drift. 0.10 - 0.25: Moderate shift. >= 0.25: Critical distribution drift.",
    ),
    "stability_jsd": MetricDefinition(
        name="stability_jsd",
        display_name="Jensen-Shannon Divergence (JSD)",
        description="Symmetric, bounded informational divergence between baseline and production distributions.",
        domain=MetricDomain.STABILITY,
        dimension=QualityDimension.STABILITY,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric", "categorical"],
        unit="JSD [0, ln(2)]",
        contributes_to_dq_score=True,
        calculation_method="0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = 0.5*(P+Q)",
        default_threshold=0.10,
        interpretation="Lower is better. 0 indicates identical probability distributions.",
    ),
    "drift_ks_test": MetricDefinition(
        name="drift_ks_test",
        display_name="Kolmogorov-Smirnov Two-Sample Test",
        description="Non-parametric two-sample test evaluating whether baseline and production samples come from the same continuous distribution.",
        domain=MetricDomain.DRIFT_ANALYSIS,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="KS statistic & p-value",
        contributes_to_dq_score=False,
        calculation_method="scipy.stats.ks_2samp(base, prod)",
        parameters={"alpha": 0.05},
        default_threshold=0.05,
        interpretation="p-value < 0.05 rejects null hypothesis of identical distributions.",
    ),
    "drift_wasserstein_distance": MetricDefinition(
        name="drift_wasserstein_distance",
        display_name="Wasserstein Distance (Earth Mover's)",
        description="First Wasserstein distance measuring the minimum work required to transform the baseline distribution into the production distribution.",
        domain=MetricDomain.DRIFT_ANALYSIS,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="data unit / normalized",
        contributes_to_dq_score=False,
        calculation_method="scipy.stats.wasserstein_distance(base, prod) / pooled_std",
        default_threshold=0.50,
        interpretation="Lower is better. Normalized Wasserstein > 0.50 indicates substantial shift.",
    ),
    "drift_unseen_categories": MetricDefinition(
        name="drift_unseen_categories",
        display_name="Unseen Categories in Production",
        description="Identification of discrete categories appearing in production that were never observed in the baseline.",
        domain=MetricDomain.DRIFT_ANALYSIS,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["categorical", "text"],
        unit="count & %",
        contributes_to_dq_score=False,
        calculation_method="count(distinct categories in prod - base) / len(prod)",
        default_threshold=0.02,
        interpretation="Unseen categories can break downstream ML models or categorical joins.",
    ),

    # -------------------------------------------------------------------------
    # STATISTICAL & DISTRIBUTIONAL INTELLIGENCE
    # -------------------------------------------------------------------------
    "stat_variance": MetricDefinition(
        name="stat_variance",
        display_name="Sample Variance",
        description="Unbiased sample variance using Bessel's correction (ddof=1) for sample size N >= 2.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="squared data unit",
        contributes_to_dq_score=False,
        calculation_method="np.var(clean_arr, ddof=1)",
        parameters={"ddof": 1, "min_sample_size": 2},
        interpretation="Measure of dispersion around the sample mean.",
    ),
    "stat_kurtosis": MetricDefinition(
        name="stat_kurtosis",
        display_name="Fisher Excess Kurtosis",
        description="Fourth standardized moment of distribution shape (Fisher's definition, where normal distribution = 0.0). Requires N >= 4.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="excess kurtosis",
        contributes_to_dq_score=False,
        calculation_method="pd.Series.kurt() (Fisher definition)",
        parameters={"min_sample_size": 4},
        interpretation="0 = mesokurtic (normal). > 0 = leptokurtic (heavy tails). < 0 = platykurtic (light tails).",
    ),
    "stat_skewness": MetricDefinition(
        name="stat_skewness",
        display_name="Fisher-Pearson Skewness",
        description="Third standardized moment quantifying distribution asymmetry. Requires N >= 3.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="skewness",
        contributes_to_dq_score=False,
        calculation_method="pd.Series.skew()",
        parameters={"min_sample_size": 3},
        interpretation="|skew| < 0.5: approximately symmetric. 0.5 - 1.0: moderately skewed. > 1.0: highly skewed.",
    ),
    "stat_mad": MetricDefinition(
        name="stat_mad",
        display_name="Median Absolute Deviation (MAD)",
        description="Robust measure of statistical dispersion: median of absolute deviations from the median.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="data unit",
        contributes_to_dq_score=False,
        calculation_method="median(|x - median(x)|)",
        interpretation="Robust alternative to standard deviation, resistant to extreme outliers.",
    ),
    "stat_entropy": MetricDefinition(
        name="stat_entropy",
        display_name="Normalized Shannon Entropy",
        description="Information-theoretic entropy in [0.0, 1.0]. Evaluated via Freedman-Diaconis histogram binning for continuous data and category probabilities for discrete data.",
        domain=MetricDomain.DISTRIBUTION,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric", "categorical"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="-sum(p * ln(p)) / ln(K)",
        interpretation="0 = completely constant/deterministic. 1.0 = maximum diversity/uniform spread.",
    ),
    "stat_hhi": MetricDefinition(
        name="stat_hhi",
        display_name="Herfindahl-Hirschman Index (HHI)",
        description="Measure of concentration and monopolization across discrete categories or discretized histogram bins.",
        domain=MetricDomain.DISTRIBUTION,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["categorical", "numeric"],
        unit="HHI [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="sum(p_i^2)",
        interpretation="Values near 1.0 indicate severe concentration in a single dominant category.",
    ),
    "stat_gini_simpson": MetricDefinition(
        name="stat_gini_simpson",
        display_name="Gini-Simpson Diversity Index",
        description="Probability that two randomly chosen values belong to different categories or bins.",
        domain=MetricDomain.DISTRIBUTION,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["categorical", "numeric"],
        unit="diversity [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="1.0 - sum(p_i^2)",
        interpretation="Values near 1.0 indicate high categorical diversity; values near 0 indicate homogeneity.",
    ),

    # -------------------------------------------------------------------------
    # RELATIONSHIP & REDUNDANCY METRICS
    # -------------------------------------------------------------------------
    "rel_pearson": MetricDefinition(
        name="rel_pearson",
        display_name="Pearson Linear Correlation",
        description="Bivariate linear correlation coefficient between numeric feature pairs. Note: Indicates linear association, not causation.",
        domain=MetricDomain.RELATIONSHIPS,
        dimension=None,
        scope=MetricScope.PAIR,
        applicable_types=["numeric"],
        unit="[-1.0, 1.0]",
        contributes_to_dq_score=False,
        calculation_method="cov(X, Y) / (std(X) * std(Y))",
        default_threshold=0.85,
        interpretation="|r| >= 0.85 indicates strong collinearity.",
    ),
    "rel_spearman": MetricDefinition(
        name="rel_spearman",
        display_name="Spearman Rank Correlation",
        description="Non-parametric monotonic rank correlation coefficient between feature pairs. Note: Indicates monotonic association, not causation.",
        domain=MetricDomain.RELATIONSHIPS,
        dimension=None,
        scope=MetricScope.PAIR,
        applicable_types=["numeric"],
        unit="[-1.0, 1.0]",
        contributes_to_dq_score=False,
        calculation_method="Pearson correlation between rank-transformed variables",
        default_threshold=0.85,
        interpretation="|rho| >= 0.85 indicates strong monotonic dependence.",
    ),
    "rel_cramers_v": MetricDefinition(
        name="rel_cramers_v",
        display_name="Cramér's V Categorical Association",
        description="Bias-corrected (Bergsma 2013) association measure between two categorical variables. Note: Indicates categorical association, not causation.",
        domain=MetricDomain.RELATIONSHIPS,
        dimension=None,
        scope=MetricScope.PAIR,
        applicable_types=["categorical"],
        unit="[0.0, 1.0]",
        contributes_to_dq_score=False,
        calculation_method="sqrt(phi2_corr / min(r_corr-1, k_corr-1))",
        default_threshold=0.60,
        interpretation="V >= 0.60 indicates strong categorical association.",
    ),
    "rel_duplicate_columns": MetricDefinition(
        name="rel_duplicate_columns",
        display_name="Duplicate Column Detection",
        description="Identification of pairs of columns that contain identical data values and identical null alignments.",
        domain=MetricDomain.RELATIONSHIPS,
        dimension=None,
        scope=MetricScope.PAIR,
        applicable_types=["all"],
        unit="exact / near match",
        contributes_to_dq_score=False,
        calculation_method="s1.equals(s2) or match_ratio >= 0.99",
        default_threshold=0.99,
        interpretation="Duplicate and near-duplicate columns represent schema redundancy and waste storage/compute.",
    ),
    "stat_iqr": MetricDefinition(
        name="stat_iqr",
        display_name="Interquartile Range (IQR)",
        description="Difference between upper quartile (75th percentile) and lower quartile (25th percentile). Robust non-parametric measure of statistical dispersion.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="data unit",
        contributes_to_dq_score=False,
        calculation_method="q75 - q25",
        interpretation="Spread of the central 50% of the distribution; foundation for Tukey outlier fences.",
    ),
    "stat_cv": MetricDefinition(
        name="stat_cv",
        display_name="Coefficient of Variation (CV)",
        description="Ratio of the sample standard deviation to the sample mean. Quantifies relative variability across different scales.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="ratio",
        contributes_to_dq_score=False,
        calculation_method="sample_std / mean",
        interpretation="CV < 0.15: low variability. 0.15 - 0.50: moderate variability. > 0.50: high variability.",
    ),
    "stat_zero_ratio": MetricDefinition(
        name="stat_zero_ratio",
        display_name="Zero Value Ratio",
        description="Proportion of non-null numeric values that are exactly zero. Identifies zero-inflation or sparse signals.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="count(x == 0) / max(1, non_null_count)",
        interpretation="High zero ratios indicate zero-inflated distributions or sparse feature vectors.",
    ),
    "stat_negative_ratio": MetricDefinition(
        name="stat_negative_ratio",
        display_name="Negative Value Ratio",
        description="Proportion of non-null numeric values that are strictly negative. Critical for identifying sign anomalies on physical measurements.",
        domain=MetricDomain.STATISTICAL_INTELLIGENCE,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="count(x < 0) / max(1, non_null_count)",
        interpretation="Non-zero negative ratios on non-negative domains (e.g., age, price) indicate validity defects.",
    ),
    "stat_top_dominance": MetricDefinition(
        name="stat_top_dominance",
        display_name="Top Category Dominance Ratio",
        description="Fraction of non-null records belonging to the single most frequent category or value.",
        domain=MetricDomain.DISTRIBUTION,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["categorical", "numeric", "text"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="mode_frequency / max(1, non_null_count)",
        default_threshold=0.95,
        interpretation="Ratio >= 0.95 indicates near-constant columns with low informational content.",
    ),
    "stat_rare_categories": MetricDefinition(
        name="stat_rare_categories",
        display_name="Rare Categories Ratio",
        description="Cumulative proportion of records in categories that each account for less than 1% of non-null records.",
        domain=MetricDomain.DISTRIBUTION,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["categorical", "text"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="sum(frequencies of categories < 0.01) / max(1, non_null_count)",
        default_threshold=0.05,
        interpretation="High rare category ratios cause sparse one-hot encodings and ML model overfitting.",
    ),
    "dataset_null_density": MetricDefinition(
        name="dataset_null_density",
        display_name="Dataset Null Density",
        description="Overall ratio of null/missing cells relative to the total cellular capacity of the dataset (rows * columns).",
        domain=MetricDomain.DATASET_PROFILING,
        dimension=QualityDimension.COMPLETENESS,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="ratio [0, 1]",
        contributes_to_dq_score=False,
        calculation_method="total_null_cells / max(1, total_rows * total_columns)",
        default_threshold=0.10,
        interpretation="Dataset-level cellular sparsity. Lower is better.",
    ),
    "dataset_memory_mb": MetricDefinition(
        name="dataset_memory_mb",
        display_name="Dataset Memory Utilization (MB)",
        description="Deep in-memory footprint of the tabular dataset including dynamic heap allocations for string and object types.",
        domain=MetricDomain.DATASET_PROFILING,
        dimension=None,
        scope=MetricScope.DATASET,
        applicable_types=["all"],
        unit="MB",
        contributes_to_dq_score=False,
        calculation_method="df.memory_usage(deep=True).sum() / (1024.0 * 1024.0)",
        interpretation="Accurate RAM consumption for capacity planning and pipeline sizing.",
    ),
    "temporal_parse_rate": MetricDefinition(
        name="temporal_parse_rate",
        display_name="Datetime Parse Rate",
        description="Percentage of column values that successfully parse into valid datetime timestamps.",
        domain=MetricDomain.TIMELINESS,
        dimension=QualityDimension.VALIDITY,
        scope=MetricScope.COLUMN,
        applicable_types=["datetime", "text"],
        unit="%",
        contributes_to_dq_score=False,
        calculation_method="100.0 * (parsed_datetime_count / max(1, non_null_count))",
        default_threshold=95.0,
        interpretation="Higher is better. 100% indicates full temporal conformance.",
    ),
    "temporal_span_days": MetricDefinition(
        name="temporal_span_days",
        display_name="Temporal Span (Days)",
        description="Total duration covered by the datetime column from the earliest to latest timestamp in days.",
        domain=MetricDomain.TIMELINESS,
        dimension=QualityDimension.TIMELINESS,
        scope=MetricScope.COLUMN,
        applicable_types=["datetime"],
        unit="days",
        contributes_to_dq_score=False,
        calculation_method="(max_timestamp - min_timestamp).total_seconds() / 86400.0",
        interpretation="Total temporal coverage of the dataset.",
    ),
    "temporal_cadence": MetricDefinition(
        name="temporal_cadence",
        display_name="Inferred Temporal Cadence",
        description="Classified dominant sampling frequency (e.g., Daily, Hourly, Minutely, Weekly, Monthly) based on median step size.",
        domain=MetricDomain.TIMELINESS,
        dimension=QualityDimension.TIMELINESS,
        scope=MetricScope.COLUMN,
        applicable_types=["datetime"],
        unit="classification",
        contributes_to_dq_score=False,
        calculation_method="Classify median_step_seconds into standard frequency bands",
        interpretation="Expected arrival periodicity for regular time-series monitoring.",
    ),
    "drift_kl_divergence": MetricDefinition(
        name="drift_kl_divergence",
        display_name="Forward Kullback-Leibler Divergence (P || Q)",
        description="Asymmetric information divergence measuring the relative entropy between production and baseline distributions.",
        domain=MetricDomain.DRIFT_ANALYSIS,
        dimension=QualityDimension.STABILITY,
        scope=MetricScope.COLUMN,
        applicable_types=["numeric", "categorical"],
        unit="nats",
        contributes_to_dq_score=False,
        calculation_method="sum(P_b * ln(P_b / Q_p)) over binned distributions",
        interpretation="0 indicates identical distributions. Strictly non-negative.",
    ),
    "profile_column_quality_score": MetricDefinition(
        name="profile_column_quality_score",
        display_name="Column Quality Score",
        description="Composite overall quality score for a single column aggregating all applicable dimensional subscores.",
        domain=MetricDomain.COLUMN_PROFILING,
        dimension=None,
        scope=MetricScope.COLUMN,
        applicable_types=["all"],
        unit="score [0, 100]",
        contributes_to_dq_score=False,
        calculation_method="Weighted mean of evaluated subscores for the column",
        interpretation="Higher is better. 100 indicates perfect column quality across all tested dimensions.",
    ),
}


class MetricRegistry:
    """Thread-safe registry query service for Cleanalytix metrics."""

    _catalog = METRIC_CATALOG

    _ALIASES: Dict[str, str] = {
        "cell_missingness_percent": "completeness_cell",
        "missing_percent": "completeness_cell",
        "incomplete_rows_percent": "completeness_row",
        "complete_rows_percent": "complete_rows_percent",
        "empty_rows_percent": "empty_rows_percent",
        "duplicate_rows_percent": "uniqueness_duplicate_rows",
        "identifier_duplicate_percent": "uniqueness_identifiers",
        "type_errors_percent": "validity_type_errors",
        "type_errors": "validity_type_errors",
        "whitespace_percent": "validity_whitespace",
        "whitespace_issues": "validity_whitespace",
        "rule_errors_percent": "validity_rules",
        "rule_errors": "validity_rules",
        "outlier_percent": "validity_outliers",
        "outlier_count": "validity_outliers",
        "future_dates_percent": "timeliness_future_dates",
        "freshness_days": "timeliness_freshness",
        "psi_divergence": "stability_psi",
        "psi": "stability_psi",
        "js_divergence": "stability_jsd",
        "jsd": "stability_jsd",
        "iqr": "stat_iqr",
        "cv": "stat_cv",
        "coefficient_of_variation": "stat_cv",
        "zero_ratio": "stat_zero_ratio",
        "negative_ratio": "stat_negative_ratio",
        "top_category_dominance_ratio": "stat_top_dominance",
        "dominance_ratio": "stat_top_dominance",
        "rare_categories_ratio": "stat_rare_categories",
        "null_density": "dataset_null_density",
        "memory_mb": "dataset_memory_mb",
        "time_span_days": "temporal_span_days",
        "span_days": "temporal_span_days",
        "cadence": "temporal_cadence",
        "inferred_cadence": "temporal_cadence",
        "detected_cadence": "temporal_cadence",
        "parse_rate": "temporal_parse_rate",
        "column_quality_score": "profile_column_quality_score",
        "kl_divergence": "drift_kl_divergence",
    }

    @classmethod
    def get(cls, name: str) -> Optional[MetricDefinition]:
        """Retrieve definition for a registered metric, supporting standardized aliases."""
        clean_name = str(name).strip().lower()
        target = cls._ALIASES.get(clean_name, clean_name)
        return cls._catalog.get(target)

    @classmethod
    def all(cls) -> List[MetricDefinition]:
        """Return all registered metric definitions."""
        return list(cls._catalog.values())

    @classmethod
    def by_domain(cls, domain: Union[MetricDomain, str]) -> List[MetricDefinition]:
        """Filter metric definitions by analytical domain."""
        dom_val = domain.value if isinstance(domain, MetricDomain) else str(domain)
        return [m for m in cls._catalog.values() if m.domain.value == dom_val]

    @classmethod
    def by_dimension(cls, dimension: Union[QualityDimension, str]) -> List[MetricDefinition]:
        """Filter metric definitions by Data Quality Dimension."""
        dim_val = dimension.value if isinstance(dimension, QualityDimension) else str(dimension)
        return [m for m in cls._catalog.values() if m.dimension and m.dimension.value == dim_val]

    @classmethod
    def scoring_metrics(cls) -> List[MetricDefinition]:
        """Return all metrics that directly contribute to Data Quality scoring."""
        return [m for m in cls._catalog.values() if m.contributes_to_dq_score]
