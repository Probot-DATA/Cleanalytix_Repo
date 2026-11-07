"""Comprehensive profiling and analytical characterization module for Cleanalytix.

Assembles deep dataset profiling, per-column profiling, statistical intelligence,
cross-column relationships, temporal analysis, and baseline drift comparisons
into standardized, structured DataFrames and containers.

Strict Non-Mutation & Statistical Safeguards:
- Input DataFrames are treated as strictly immutable (read-only).
- Missingness co-occurrence is documented as empirical co-occurrence, NOT proof of MCAR/MAR/MNAR.
- Outlier consensus and bounds reflect heuristic fences, NOT ground-truth verified anomalies.
- Correlations and Cramér's V reflect mathematical association, NOT causation or dependency.
- Zeros, missing values, non-applicable metrics, and insufficient-data states are strictly distinguished.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from cleanalytix.column_role import infer_column_role, harmonize_series_for_drift
from cleanalytix.metrics.completeness import compute_completeness_metrics
from cleanalytix.metrics.distribution import (
    compute_categorical_distribution,
    compute_continuous_distribution,
)
from cleanalytix.metrics.drift import compute_numeric_drift, compute_categorical_drift
from cleanalytix.metrics.relationships import (
    compute_numeric_correlations,
    compute_cramers_v,
    detect_duplicate_columns,
)
from cleanalytix.metrics.statistical import (
    compute_numeric_statistics,
    detect_outliers,
)
from cleanalytix.metrics.temporal import compute_temporal_metrics
from cleanalytix.metrics.uniqueness import (
    compute_column_uniqueness,
    compute_duplicate_row_metrics,
    detect_candidate_keys,
)
from cleanalytix.types import (
    DatasetProfile,
    RelationshipProfile,
    CleanalytixConfig,
    DimensionScore,
    InsightItem,
    InsightSeverity,
    QualityDimension,
)


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError, OverflowError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError, OverflowError):
        return default


def build_dataset_profile(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    overall_score: float = 0.0,
    dim_scores: Optional[Dict[str, DimensionScore]] = None,
    critical_count: int = 0,
    warning_count: int = 0,
    recommendations_count: int = 0,
    **kwargs: Any,
) -> DatasetProfile:
    """Construct an executive, structural dataset-level profile.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed dataset (read-only).
    meta_df : pd.DataFrame
        Column metadata DataFrame.
    overall_score : float
        Overall Data Quality Score in [0, 100].
    dim_scores : dict of str to DimensionScore, optional
        Evaluated dimensional score objects.
    critical_count : int
        Count of critical quality anomalies.
    warning_count : int
        Count of warning quality anomalies.
    recommendations_count : int
        Count of actionable recommendations.

    Returns
    -------
    DatasetProfile
        Structured dataset-level health and characteristic container.
    """
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    memory_bytes = int(df.memory_usage(deep=True).sum()) if total_rows > 0 and total_cols > 0 else 0
    memory_mb = float(memory_bytes / (1024.0 * 1024.0))
    memory_per_row = float(memory_bytes / max(1, total_rows))

    # Global completeness & empty rows
    comp_metrics = compute_completeness_metrics(df)
    null_density = float(df.isna().sum().sum() / max(1, total_cells))

    # Duplicate rows
    dup_row_metrics = compute_duplicate_row_metrics(df)

    # Column redundancies
    dup_cols_info = detect_duplicate_columns(df)
    dup_cols_count = len(dup_cols_info.get("duplicate_columns", []))
    near_dup_cols_count = len(dup_cols_info.get("near_duplicate_columns", []))

    # Column semantic role distribution
    role_counts: Dict[str, int] = {}
    if "column_role" in meta_df.columns:
        role_counts = {str(k): int(v) for k, v in meta_df["column_role"].value_counts().items()}

    # Candidate keys
    key_info = detect_candidate_keys(df)
    single_keys = key_info.get("single_keys", [])
    composite_keys = key_info.get("composite_keys", [])

    # Dimensional scores dictionary
    dimension_scores_dict: Dict[str, float] = {}
    if dim_scores:
        for k, ds in dim_scores.items():
            dimension_scores_dict[k] = float(ds.score)

    return DatasetProfile(
        total_rows=total_rows,
        total_columns=total_cols,
        total_cells=total_cells,
        memory_bytes=memory_bytes,
        memory_mb=round(memory_mb, 4),
        memory_per_row_bytes=round(memory_per_row, 2),
        null_density=round(null_density, 4),
        complete_rows_count=comp_metrics["complete_rows_count"],
        complete_rows_percent=comp_metrics["complete_rows_percent"],
        empty_rows_count=comp_metrics["empty_rows_count"],
        empty_rows_percent=comp_metrics["empty_rows_percent"],
        duplicate_rows_count=dup_row_metrics["duplicate_row_count"],
        duplicate_rows_percent=round(dup_row_metrics["duplicate_row_ratio"] * 100.0, 2),
        duplicate_columns_count=dup_cols_count,
        near_duplicate_columns_count=near_dup_cols_count,
        role_counts=role_counts,
        candidate_keys=single_keys,
        composite_candidate_keys=composite_keys,
        overall_score=round(overall_score, 2),
        dimension_scores=dimension_scores_dict,
        critical_anomalies_count=critical_count,
        warning_anomalies_count=warning_count,
        recommendations_count=recommendations_count,
    )


def build_column_profile(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    column_scores: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Build a unified, comprehensive per-column profiling table.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed dataset.
    meta_df : pd.DataFrame
        Metadata table returned by generate_meta.
    column_scores : dict of str to float, optional
        Overall quality score for each column in [0, 100].

    Returns
    -------
    pd.DataFrame
        Comprehensive profiling table indexed by column_name.
    """
    col_scores = column_scores or {}
    total_rows = len(df)
    rows = []

    for _, meta_row in meta_df.iterrows():
        col = meta_row.get("column_name")
        ser = df[col] if col in df.columns else pd.Series(dtype=object)

        role = str(meta_row.get("column_role", "unknown"))
        dtype_str = str(meta_row.get("dtype", ser.dtype))

        non_null = _safe_int(meta_row.get("non_null_count", 0))
        missing = _safe_int(meta_row.get("missing_count", 0))
        missing_pct = _safe_float(meta_row.get("missing_percent", 0.0))

        uniq_cnt = _safe_int(meta_row.get("unique_count", 0))
        uniq_pct = _safe_float(meta_row.get("unique_percent", 0.0))

        # Uniqueness properties
        is_uniq = (uniq_cnt == total_rows) and (missing == 0) and (total_rows > 0)
        is_const = (uniq_cnt <= 1) and (total_rows > 0)

        # Mode and top frequency
        top_val = meta_row.get("mode")
        top_pct = meta_row.get("mode_percent")
        is_near_const = bool(top_pct is not None and pd.notna(top_pct) and _safe_float(top_pct) >= 95.0)

        # Basic central tendency
        mean_val = meta_row.get("mean")
        std_val = meta_row.get("std")
        median_val = meta_row.get("median")
        min_val = meta_row.get("min")
        max_val = meta_row.get("max")
        skew_val = meta_row.get("skewness")
        skew_lvl = meta_row.get("skewness_level")

        entropy_val = meta_row.get("Entropy", 0.0)

        # Conformance and error counts
        type_err_cnt = _safe_int(meta_row.get("type_errors", 0))
        type_err_pct = round((type_err_cnt / max(1, non_null)) * 100.0, 2)

        ws_cnt = _safe_int(meta_row.get("whitespace_issues", 0))
        ws_pct = round((ws_cnt / max(1, non_null)) * 100.0, 2)

        rule_err_cnt = _safe_int(meta_row.get("rule_errors", 0))
        rule_err_pct = round((rule_err_cnt / max(1, non_null)) * 100.0, 2)

        out_cnt = _safe_int(meta_row.get("outlier_count", 0))
        out_pct = round((out_cnt / max(1, non_null)) * 100.0, 2) if role in {"continuous", "discrete"} else 0.0

        min_dt = meta_row.get("min_date")
        max_dt = meta_row.get("max_date")
        range_days = meta_row.get("date_range_days")

        quality_score = col_scores.get(col, 100.0)

        rows.append({
            "column_name": col,
            "column_role": role,
            "dtype": dtype_str,
            "total_rows": total_rows,
            "non_null_count": non_null,
            "null_count": missing,
            "null_percent": round(missing_pct, 2),
            "distinct_count": uniq_cnt,
            "distinct_percent": round(uniq_pct, 2),
            "is_unique": is_uniq,
            "is_constant": is_const,
            "is_near_constant": is_near_const,
            "top_value": top_val,
            "top_frequency_percent": round(top_pct, 2) if top_pct is not None else None,
            "mean": round(float(mean_val), 4) if mean_val is not None and pd.notna(mean_val) else np.nan,
            "std": round(float(std_val), 4) if std_val is not None and pd.notna(std_val) else np.nan,
            "median": round(float(median_val), 4) if median_val is not None and pd.notna(median_val) else np.nan,
            "min": round(float(min_val), 4) if min_val is not None and pd.notna(min_val) else np.nan,
            "max": round(float(max_val), 4) if max_val is not None and pd.notna(max_val) else np.nan,
            "skewness": round(float(skew_val), 4) if skew_val is not None and pd.notna(skew_val) else np.nan,
            "skewness_level": skew_lvl,
            "entropy": round(float(entropy_val), 4) if pd.notna(entropy_val) else 0.0,
            "type_error_count": type_err_cnt,
            "type_error_percent": type_err_pct,
            "whitespace_count": ws_cnt,
            "whitespace_percent": ws_pct,
            "rule_error_count": rule_err_cnt,
            "rule_error_percent": rule_err_pct,
            "outlier_count": out_cnt,
            "outlier_percent": out_pct,
            "min_date": str(min_dt) if pd.notna(min_dt) else None,
            "max_date": str(max_dt) if pd.notna(max_dt) else None,
            "date_range_days": range_days if pd.notna(range_days) else None,
            "column_quality_score": round(quality_score, 2),
        })

    col_df = pd.DataFrame(rows)
    if not col_df.empty:
        col_df.set_index("column_name", inplace=True)
    return col_df


def build_statistical_profile(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a deep statistical intelligence table with moments, quantiles, and diversity.

    Statistical Invariants:
    - Sample variance and standard deviation use Bessel's correction (ddof=1) for N >= 2.
    - Excess kurtosis uses Fisher's definition (normal distribution = 0.0) for N >= 4.
    - Skewness uses Fisher-Pearson sample skewness for N >= 3.
    - Outlier bounds and consensus counts represent heuristic fences, not ground truth.
    - Categorical diversity uses Herfindahl-Hirschman Index and Gini-Simpson index.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed dataset.
    meta_df : pd.DataFrame
        Metadata DataFrame.

    Returns
    -------
    pd.DataFrame
        Deep statistical profile table indexed by column_name.
    """
    rows = []

    for _, meta_row in meta_df.iterrows():
        col = meta_row.get("column_name")
        if col not in df.columns:
            continue

        ser = df[col]
        role = str(meta_row.get("column_role", "unknown"))
        clean_ser = ser.dropna()
        n = len(clean_ser)

        stat_entry: Dict[str, Any] = {
            "column_name": col,
            "column_role": role,
            "count": n,
            # Moments & Dispersion
            "mean": np.nan,
            "std": np.nan,
            "variance": np.nan,
            "median": np.nan,
            "iqr": np.nan,
            "mad": np.nan,
            "coefficient_of_variation": np.nan,
            # Quantiles
            "min": np.nan,
            "p01": np.nan,
            "p05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
            # Shape
            "skewness": np.nan,
            "skewness_class": None,
            "kurtosis": np.nan,
            "kurtosis_fisher": np.nan,
            "kurtosis_class": None,
            # Boundaries & Density
            "zero_count": 0,
            "zero_ratio": 0.0,
            "negative_count": 0,
            "negative_ratio": 0.0,
            "infinite_count": 0,
            # Outlier Bounds (Heuristic fences)
            "outlier_method": None,
            "outlier_lower_bound": np.nan,
            "outlier_upper_bound": np.nan,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            # Categorical Concentration & Diversity
            "shannon_entropy": np.nan,
            "normalized_entropy": np.nan,
            "hhi": np.nan,
            "gini_simpson_diversity": np.nan,
            "top_category_dominance_ratio": np.nan,
            "is_dominated": False,
            "rare_categories_count": 0,
            "rare_categories_ratio": 0.0,
        }

        # Numeric processing
        if is_numeric_dtype(ser) and not is_bool_dtype(ser) and role not in {"categorical", "binary", "datetime", "boolean"}:
            if n > 0:
                stats_dict = compute_numeric_statistics(ser)
                out_dict = detect_outliers(ser)

                stat_entry["mean"] = stats_dict["mean"]
                stat_entry["std"] = stats_dict["std"]
                stat_entry["variance"] = stats_dict["variance"]
                stat_entry["median"] = stats_dict["median"]
                stat_entry["iqr"] = stats_dict["iqr"]
                stat_entry["mad"] = stats_dict["mad"]
                stat_entry["coefficient_of_variation"] = stats_dict["coefficient_of_variation"]

                stat_entry["min"] = stats_dict["min"]
                stat_entry["p01"] = stats_dict["p01"]
                stat_entry["p05"] = stats_dict["p05"]
                stat_entry["q25"] = stats_dict["q25"]
                stat_entry["q75"] = stats_dict["q75"]
                stat_entry["p95"] = stats_dict["p95"]
                stat_entry["p99"] = stats_dict["p99"]
                stat_entry["max"] = stats_dict["max"]

                skew_val = stats_dict["skewness"]
                stat_entry["skewness"] = skew_val
                if skew_val is not None:
                    abs_s = abs(skew_val)
                    stat_entry["skewness_class"] = (
                        "symmetric" if abs_s < 0.5 else ("moderate" if abs_s < 1.0 else "highly_skewed")
                    )

                kurt_val = stats_dict["kurtosis"]
                stat_entry["kurtosis"] = kurt_val
                stat_entry["kurtosis_fisher"] = kurt_val
                if kurt_val is not None:
                    stat_entry["kurtosis_class"] = (
                        "mesokurtic" if abs(kurt_val) < 0.5 else ("leptokurtic" if kurt_val >= 0.5 else "platykurtic")
                    )

                stat_entry["zero_count"] = stats_dict["zero_count"]
                stat_entry["zero_ratio"] = stats_dict["zero_ratio"]
                stat_entry["negative_count"] = stats_dict["negative_count"]
                stat_entry["negative_ratio"] = stats_dict["negative_ratio"]
                stat_entry["infinite_count"] = stats_dict["infinite_count"]

                # Recommended outlier method bounds
                rec_meth = out_dict.get("recommended_method", "iqr")
                stat_entry["outlier_method"] = rec_meth
                stat_entry["outlier_count"] = out_dict.get("recommended_outlier_count", 0)
                stat_entry["outlier_ratio"] = out_dict.get("recommended_outlier_ratio", 0.0)

                meth_info = out_dict.get(rec_meth, {})
                stat_entry["outlier_lower_bound"] = meth_info.get("lower_bound", np.nan)
                stat_entry["outlier_upper_bound"] = meth_info.get("upper_bound", np.nan)

                # Continuous entropy & diversity
                cont_dist = compute_continuous_distribution(ser)
                stat_entry["shannon_entropy"] = cont_dist.get("shannon_entropy", np.nan)
                stat_entry["normalized_entropy"] = cont_dist.get("normalized_entropy", np.nan)
                stat_entry["hhi"] = cont_dist.get("hhi", np.nan)
                stat_entry["gini_simpson_diversity"] = cont_dist.get("gini_simpson_diversity", np.nan)

        # Categorical / text / binary / boolean
        else:
            if n > 0:
                cat_dist = compute_categorical_distribution(ser)
                stat_entry["shannon_entropy"] = cat_dist.get("shannon_entropy", np.nan)
                stat_entry["normalized_entropy"] = cat_dist.get("normalized_entropy", np.nan)
                stat_entry["hhi"] = cat_dist.get("hhi", np.nan)
                stat_entry["gini_simpson_diversity"] = cat_dist.get("gini_simpson_diversity", np.nan)
                stat_entry["top_category_dominance_ratio"] = cat_dist.get("top_category_ratio", np.nan)
                stat_entry["is_dominated"] = cat_dist.get("is_dominated", False)
                stat_entry["rare_categories_count"] = cat_dist.get("rare_categories_count", 0)
                stat_entry["rare_categories_ratio"] = cat_dist.get("rare_categories_ratio", 0.0)

        rows.append(stat_entry)

    stat_df = pd.DataFrame(rows)
    if not stat_df.empty:
        stat_df.set_index("column_name", inplace=True)
    return stat_df


def build_relationship_profile(
    df: pd.DataFrame,
    config: Optional[CleanalytixConfig] = None,
) -> RelationshipProfile:
    """Construct a comprehensive cross-column relationship profile.

    Statistical Safeguards:
    - Correlation and Cramér's V reflect mathematical association, NOT causation.
    - Missingness co-occurrence represents empirical Jaccard similarity, NOT proof of MCAR/MAR/MNAR.
    - Near-duplicate matching is capped to bounded columns to prevent combinatorial explosion.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed dataset.
    config : CleanalytixConfig, optional
        Configuration options.

    Returns
    -------
    RelationshipProfile
        Structured relationship container with correlation and association matrices.
    """
    cfg = config or CleanalytixConfig()
    max_cols = cfg.max_correlation_columns

    # 1. Numeric correlations
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not is_bool_dtype(df[c])]
    viable_numeric = [c for c in numeric_cols if df[c].dropna().nunique() > 1][:max_cols]

    if len(viable_numeric) >= 2:
        sub_num = df[viable_numeric]
        pearson_df = sub_num.corr(method="pearson").round(4)
        spearman_df = sub_num.corr(method="spearman").round(4)

        high_corr_list = []
        for c1, c2 in itertools.combinations(viable_numeric, 2):
            p = pearson_df.loc[c1, c2]
            s = spearman_df.loc[c1, c2]
            p_abs = abs(p) if pd.notna(p) else 0.0
            s_abs = abs(s) if pd.notna(s) else 0.0
            max_abs = max(p_abs, s_abs)
            if max_abs >= 0.85:
                high_corr_list.append({
                    "column_1": c1,
                    "column_2": c2,
                    "pearson": float(p) if pd.notna(p) else None,
                    "spearman": float(s) if pd.notna(s) else None,
                    "max_abs_association": float(round(max_abs, 4)),
                    "interpretation": "Strong collinear association (does not imply causation)",
                })
        high_corr_df = pd.DataFrame(high_corr_list)
        if not high_corr_df.empty:
            high_corr_df.sort_values("max_abs_association", ascending=False, inplace=True)
    else:
        pearson_df = pd.DataFrame()
        spearman_df = pd.DataFrame()
        high_corr_df = pd.DataFrame()

    # 2. Categorical Cramér's V
    cat_cols = [c for c in df.columns if c not in viable_numeric][:20]
    viable_cat = [c for c in cat_cols if 1 < df[c].dropna().nunique() <= 100]

    cramers_matrix = pd.DataFrame(index=viable_cat, columns=viable_cat, dtype=float)
    if len(viable_cat) >= 2:
        for c1 in viable_cat:
            cramers_matrix.loc[c1, c1] = 1.0
        for c1, c2 in itertools.combinations(viable_cat, 2):
            try:
                v = compute_cramers_v(df[c1], df[c2])
                cramers_matrix.loc[c1, c2] = round(v, 4)
                cramers_matrix.loc[c2, c1] = round(v, 4)
            except Exception:
                cramers_matrix.loc[c1, c2] = np.nan
                cramers_matrix.loc[c2, c1] = np.nan

    # 3. Redundant & near-duplicate columns
    dup_info = detect_duplicate_columns(df)
    red_rows = []
    for c1, c2 in dup_info.get("duplicate_columns", []):
        red_rows.append({
            "column_1": c1,
            "column_2": c2,
            "redundancy_type": "exact_duplicate",
            "match_ratio": 1.0,
        })
    for item in dup_info.get("near_duplicate_columns", []):
        red_rows.append({
            "column_1": item["column_1"],
            "column_2": item["column_2"],
            "redundancy_type": "near_duplicate",
            "match_ratio": round(item["match_ratio"], 4),
        })
    redundant_df = pd.DataFrame(red_rows)

    # 4. Missingness co-occurrence (Jaccard)
    comp_metrics = compute_completeness_metrics(df)
    jaccard_rows = []
    for item in comp_metrics.get("missingness_cooccurrence", []):
        jaccard_rows.append({
            "column_a": item["column_a"],
            "column_b": item["column_b"],
            "both_missing_count": item["both_missing_count"],
            "jaccard_similarity": item["jaccard_similarity"],
            "interpretation": "Observed empirical co-occurrence (does not prove MCAR/MAR/MNAR)",
        })
    missing_cooc_df = pd.DataFrame(jaccard_rows)

    # 5. Candidate keys
    key_info = detect_candidate_keys(df)

    return RelationshipProfile(
        pearson_correlations=pearson_df,
        spearman_correlations=spearman_df,
        cramers_v=cramers_matrix.dropna(how="all"),
        high_correlations=high_corr_df,
        redundant_columns=redundant_df,
        missingness_cooccurrence=missing_cooc_df,
        candidate_keys=key_info,
    )


def build_temporal_profile(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    config: Optional[CleanalytixConfig] = None,
) -> pd.DataFrame:
    """Construct a comprehensive temporal quality and cadence table.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed dataset.
    meta_df : pd.DataFrame
        Metadata DataFrame.
    config : CleanalytixConfig, optional
        Configuration options.

    Returns
    -------
    pd.DataFrame
        Temporal profile table indexed by datetime column name.
    """
    cfg = config or CleanalytixConfig()
    time_cfg = cfg.timeliness_config
    ref_time = getattr(time_cfg, "reference_time", None) if time_cfg else None

    # Detect datetime columns
    dt_cols = []
    for _, row in meta_df.iterrows():
        col = row.get("column_name")
        role = row.get("column_role")
        dtype = str(row.get("dtype", "")).lower()
        if role == "datetime" or "datetime" in dtype or (col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])):
            dt_cols.append(col)

    if not dt_cols:
        return pd.DataFrame()

    rows = []
    for col in dt_cols:
        ser = df[col]
        t_metrics = compute_temporal_metrics(ser, reference_time=ref_time)
        if not t_metrics.get("is_temporal", False):
            continue

        step_sec = t_metrics.get("median_step_seconds")
        if step_sec is not None:
            if step_sec < 120:
                cadence = "Minutely / Sub-minute"
            elif 3000 <= step_sec <= 4200:
                cadence = "Hourly"
            elif 80000 <= step_sec <= 95000:
                cadence = "Daily"
            elif 550000 <= step_sec <= 700000:
                cadence = "Weekly"
            elif 2400000 <= step_sec <= 2800000:
                cadence = "Monthly"
            else:
                cadence = "Irregular / Mixed"
        else:
            cadence = "Unknown / Single timestamp"

        fresh_days = t_metrics.get("freshness_days")
        if fresh_days is not None:
            if fresh_days <= 7.0:
                fresh_status = "FRESH"
            elif fresh_days <= 30.0:
                fresh_status = "ACCEPTABLE"
            else:
                fresh_status = "STALE"
        else:
            fresh_status = "UNKNOWN"

        rows.append({
            "column_name": col,
            "parse_rate": round(t_metrics["parse_rate"] * 100.0, 2),
            "min_date": t_metrics["min_date"],
            "max_date": t_metrics["max_date"],
            "time_span_days": round(t_metrics["time_span_days"], 2) if t_metrics["time_span_days"] is not None else None,
            "span_days": round(t_metrics["time_span_days"], 2) if t_metrics["time_span_days"] is not None else None,
            "freshness_days": round(fresh_days, 2) if fresh_days is not None else None,
            "freshness_status": fresh_status,
            "inferred_cadence": cadence,
            "detected_cadence": cadence,
            "median_step_seconds": round(step_sec, 2) if step_sec is not None else None,
            "max_gap_seconds": round(t_metrics["max_gap_seconds"], 2) if t_metrics["max_gap_seconds"] is not None else None,
            "is_monotonic_increasing": t_metrics["is_monotonic_increasing"],
            "future_date_count": t_metrics["future_date_count"],
            "future_dates_count": t_metrics["future_date_count"],
            "future_date_ratio": round(t_metrics["future_date_ratio"] * 100.0, 4),
        })

    temp_df = pd.DataFrame(rows)
    if not temp_df.empty:
        temp_df.set_index("column_name", inplace=True)
    return temp_df


def build_drift_profile(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame],
    meta_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    """Construct a comprehensive distribution drift table against a reference baseline.

    Statistical Invariants:
    - Binned Jeffreys PSI (10 quantile bins) and discrete 3-bin surrogate approximation for constant baselines.
    - Two-sample Kolmogorov-Smirnov asymptotic/exact test p-value and statistic.
    - Raw and pooled-std normalized Wasserstein distance.
    - Exact unseen categories list and row ratios.

    Parameters
    ----------
    df : pd.DataFrame
        Current / production dataset.
    baseline_df : pd.DataFrame, optional
        Reference / baseline dataset.
    meta_df : pd.DataFrame
        Metadata DataFrame.

    Returns
    -------
    pd.DataFrame or None
        Drift profile table indexed by column_name, or None if no baseline.
    """
    if baseline_df is None:
        return None

    shared_cols = [c for c in df.columns if c in baseline_df.columns]
    if not shared_cols:
        return pd.DataFrame()

    rows = []
    for col in shared_cols:
        role = "unknown"
        if "column_name" in meta_df.columns and "column_role" in meta_df.columns:
            match = meta_df[meta_df["column_name"] == col]
            if not match.empty:
                role = str(match["column_role"].iloc[0])

        b_ser = baseline_df[col]
        p_ser = df[col]

        # Use specialized drift modules
        is_num = (
            is_numeric_dtype(b_ser)
            and is_numeric_dtype(p_ser)
            and not is_bool_dtype(b_ser)
            and not is_bool_dtype(p_ser)
            and role not in {"categorical", "binary", "datetime", "boolean"}
        )

        if is_num:
            d_res = compute_numeric_drift(b_ser, p_ser)
            rows.append({
                "column_name": col,
                "column_role": role,
                "sample_size_base": d_res["sample_size_base"],
                "sample_size_prod": d_res["sample_size_prod"],
                "ks_statistic": round(d_res["ks_statistic"], 4),
                "ks_p_value": round(d_res["ks_p_value"], 6),
                "wasserstein_distance": round(d_res["wasserstein_distance"], 4),
                "normalized_wasserstein": round(d_res["normalized_wasserstein"], 4),
                "psi": round(d_res["psi"], 4),
                "js_divergence": round(d_res["js_divergence"], 4),
                "kl_divergence": round(d_res["kl_divergence"], 4),
                "reverse_kl_divergence": round(d_res["reverse_kl_divergence"], 4),
                "unseen_categories_count": 0,
                "unseen_categories_ratio": 0.0,
                "unseen_categories": [],
                "unseen_categories_list": [],
                "drift_detected": d_res["drift_detected"],
                "drift_severity": d_res["drift_severity"].upper(),
            })
        else:
            c_res = compute_categorical_drift(b_ser, p_ser)
            rows.append({
                "column_name": col,
                "column_role": role,
                "sample_size_base": c_res["sample_size_base"],
                "sample_size_prod": c_res["sample_size_prod"],
                "ks_statistic": np.nan,
                "ks_p_value": np.nan,
                "wasserstein_distance": np.nan,
                "normalized_wasserstein": np.nan,
                "psi": round(c_res["psi"], 4),
                "js_divergence": round(c_res["js_divergence"], 4),
                "kl_divergence": round(c_res["kl_divergence"], 4),
                "reverse_kl_divergence": round(c_res["reverse_kl_divergence"], 4),
                "unseen_categories_count": c_res["unseen_categories_count"],
                "unseen_categories_ratio": round(c_res["unseen_categories_ratio"] * 100.0, 2),
                "unseen_categories": c_res["unseen_categories"][:10],
                "unseen_categories_list": c_res["unseen_categories"][:10],
                "drift_detected": c_res["drift_detected"],
                "drift_severity": c_res["drift_severity"].upper(),
            })

    drift_df = pd.DataFrame(rows)
    if not drift_df.empty:
        drift_df.set_index("column_name", inplace=True)
    return drift_df
