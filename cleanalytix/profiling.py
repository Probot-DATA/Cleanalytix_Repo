"""Comprehensive profiling and analytical characterization module for Cleanalytix.

Assembles deep dataset profiling, per-column profiling, statistical intelligence,
cross-column relationships, temporal analysis, and baseline drift comparisons
into standardized, structured DataFrames and typed ProfileContext containers.

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
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from cleanalytix.column_role import infer_column_role, harmonize_series_for_drift
from cleanalytix.legacy.adapter import LegacyMetaAdapter
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
from cleanalytix.metrics.validity import compute_column_validity
from cleanalytix.types import (
    AnalysisLevel,
    CleanalytixConfig,
    ColumnCompleteness,
    ColumnDistribution,
    ColumnDrift,
    ColumnProfile,
    ColumnRole,
    ColumnStatistical,
    ColumnTemporal,
    ColumnUniqueness,
    ColumnValidity,
    DatasetProfile,
    DimensionScore,
    InsightItem,
    InsightSeverity,
    ProfileContext,
    QualityDimension,
    RelationshipProfile,
    Rules,
    SemanticType,
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


def _safe_nunique(series: pd.Series, dropna: bool = True) -> int:
    try:
        return int(series.nunique(dropna=dropna))
    except TypeError:
        return int(series.astype(str).nunique(dropna=dropna))


def _infer_semantic_type_from_role(role: ColumnRole) -> SemanticType:
    if role in {"continuous", "discrete"}:
        return "numeric"
    if role == "datetime":
        return "datetime"
    if role in {"binary", "boolean"}:
        return "binary"
    if role == "identifier":
        return "identifier"
    if role == "free_text":
        return "text"
    return "categorical"


def _classify_cadence(step_sec: Optional[float]) -> str:
    if step_sec is not None:
        if step_sec < 120:
            return "Minutely / Sub-minute"
        if 3000 <= step_sec <= 4200:
            return "Hourly"
        if 80000 <= step_sec <= 95000:
            return "Daily"
        if 550000 <= step_sec <= 700000:
            return "Weekly"
        if 2400000 <= step_sec <= 2800000:
            return "Monthly"
        return "Irregular / Mixed"
    return "Unknown / Single timestamp"


def _classify_freshness(fresh_days: Optional[float], fresh_thresh: float) -> str:
    if fresh_days is not None:
        if fresh_days <= 7.0:
            return "FRESH"
        if fresh_days <= fresh_thresh:
            return "ACCEPTABLE"
        return "STALE"
    return "UNKNOWN"


def build_profile_context(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    config: Optional[CleanalytixConfig] = None,
    rules: Optional[Rules] = None,
    dataset_name: str = "dataset",
    reference_profile: Optional[Union[Dict[str, Any], ProfileContext]] = None,
    include_relationships: bool = True,
) -> ProfileContext:
    """Construct the authoritative Cleanalytix v2.0 ``ProfileContext`` for a DataFrame.

    Executes each applicable analytical domain once per column/dataset and populates
    strongly typed domain dataclasses without intermediate ``meta_df`` construction
    or duplicate metric calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset to profile. Strictly read-only and never mutated.
    baseline_df : pd.DataFrame, optional
        Optional baseline/reference dataset for two-sample drift evaluation.
    config : CleanalytixConfig, optional
        Analysis configuration.
    rules : Rules, optional
        Optional column validation business rules mapping.
    dataset_name : str, default="dataset"
        Identifier for the dataset.
    reference_profile : dict or ProfileContext, optional
        Optional learned reference profile stored on the returned context.
    include_relationships : bool, default=True
        Whether to compute cross-column relationship matrices and candidate keys.

    Returns
    -------
    ProfileContext
        Unified typed profile context containing ``DatasetProfile``, ordered
        ``ColumnProfile`` objects, and ``RelationshipProfile``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame for df, got {type(df).__name__}.")
    if baseline_df is not None and not isinstance(baseline_df, pd.DataFrame):
        raise TypeError(
            f"Expected pandas DataFrame for baseline_df, got {type(baseline_df).__name__}."
        )

    if not df.columns.is_unique:
        dup_cols = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"DataFrame '{dataset_name}' contains duplicate column names: {dup_cols}. "
            f"Columns must be strictly unique for data quality analysis."
        )

    if baseline_df is not None and not baseline_df.columns.is_unique:
        dup_cols = baseline_df.columns[baseline_df.columns.duplicated()].tolist()
        raise ValueError(
            f"Baseline DataFrame '{dataset_name}' contains duplicate column names: {dup_cols}. "
            f"Columns must be strictly unique for data quality analysis."
        )

    cfg = config or CleanalytixConfig(rules=rules)
    active_rules: Dict[str, Any] = {}
    if cfg.rules:
        active_rules.update(cfg.rules)
    if rules:
        active_rules.update(rules)

    time_cfg = cfg.timeliness_config
    cfg_dt_cols: set = set()
    if time_cfg:
        if isinstance(time_cfg, dict):
            ref_time = time_cfg.get("reference_time")
            fd = time_cfg.get("freshness_days", 30.0)
            fresh_thresh = float(fd) if fd is not None else 30.0
            dt_list = time_cfg.get("datetime_columns")
        else:
            ref_time = getattr(time_cfg, "reference_time", None)
            fd = getattr(time_cfg, "freshness_days", 30.0)
            fresh_thresh = float(fd) if fd is not None else 30.0
            dt_list = getattr(time_cfg, "datetime_columns", None)
        if dt_list:
            cfg_dt_cols = {str(c) for c in dt_list} | set(dt_list)
    else:
        ref_time = None
        fresh_thresh = 30.0

    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    memory_bytes = (
        int(df.memory_usage(deep=True).sum()) if total_rows > 0 and total_cols > 0 else 0
    )
    memory_mb = float(memory_bytes / (1024.0 * 1024.0))
    memory_per_row = float(memory_bytes / max(1, total_rows))

    # Dataset-level completeness and duplicate rows (computed once)
    comp_metrics = compute_completeness_metrics(df)
    overall_null_count = int(sum(comp_metrics["column_missing_count"].values()))
    null_density = float(overall_null_count / max(1, total_cells))
    overall_null_pct = float(null_density * 100.0)

    dup_row_metrics = compute_duplicate_row_metrics(df)

    # Cross-column relationships (computed once if enabled)
    if include_relationships:
        relationships = build_relationship_profile(
            df, config=cfg, _comp_metrics=comp_metrics
        )
        dup_cols_count = len(relationships.duplicate_columns)
        near_dup_cols_count = len(relationships.near_duplicate_columns)
        single_keys = list(relationships.candidate_keys.get("single_keys", []))
        composite_keys = list(relationships.candidate_keys.get("composite_keys", []))
    else:
        relationships = None
        dup_cols_count = 0
        near_dup_cols_count = 0
        single_keys = []
        composite_keys = []

    single_key_set = set(single_keys)
    role_counts: Dict[str, int] = {}
    column_profiles: Dict[str, ColumnProfile] = {}

    for col_raw in df.columns:
        col_name = str(col_raw)
        ser = df[col_raw]
        dtype_str = str(ser.dtype)
        try:
            role = infer_column_role(ser, col_raw, total_rows)
        except TypeError:
            role = infer_column_role(ser.astype(str), col_raw, total_rows)
        role_counts[role] = role_counts.get(role, 0) + 1

        # 1. Completeness Domain
        null_count = int(comp_metrics["column_missing_count"].get(col_raw, int(ser.isna().sum())))
        non_null_count = int(total_rows - null_count)
        null_pct = float((null_count / max(1, total_rows)) * 100.0)
        comp_ratio = float(non_null_count / total_rows) if total_rows > 0 else 1.0

        empty_str_cnt = 0
        if non_null_count > 0 and (
            ser.dtype == object or pd.api.types.is_string_dtype(ser)
        ):
            empty_str_cnt = int(
                sum(1 for v in ser.dropna() if isinstance(v, str) and v == "")
            )

        col_completeness = ColumnCompleteness(
            total_count=total_rows,
            non_null_count=non_null_count,
            null_count=null_count,
            null_percentage=null_pct,
            completeness_ratio=comp_ratio,
            empty_string_count=empty_str_cnt,
        )

        # 2. Uniqueness Domain
        u_dict = compute_column_uniqueness(ser)
        distinct_count = int(u_dict["distinct_count"])
        unique_pct = float((distinct_count / max(1, total_rows)) * 100.0)
        dup_val_cnt = max(0, non_null_count - distinct_count)
        dup_val_pct = (
            float((dup_val_cnt / non_null_count) * 100.0) if non_null_count > 0 else 0.0
        )

        try:
            mode_series = ser.mode()
            mode_val = mode_series.iloc[0] if len(mode_series) > 0 else None
            if mode_val is not None and hasattr(mode_val, "item"):
                try:
                    mode_val = mode_val.item()
                except (ValueError, TypeError):
                    pass
        except Exception:
            mode_val = None

        if mode_val is not None:
            try:
                mode_pct: Optional[float] = float(
                    ((ser == mode_val).sum() / max(1, total_rows)) * 100.0
                )
            except Exception:
                mode_pct = None
        else:
            mode_pct = None

        col_uniqueness = ColumnUniqueness(
            distinct_count=distinct_count,
            distinct_ratio=float(u_dict["distinct_ratio"]),
            non_null_distinct_ratio=float(u_dict["non_null_distinct_ratio"]),
            unique_percentage=unique_pct,
            duplicate_count=dup_val_cnt,
            duplicate_percentage=dup_val_pct,
            is_unique=bool(u_dict["is_unique"]),
            is_constant=bool(u_dict["is_constant"]),
            is_near_constant=bool(u_dict["is_near_constant"]),
            is_primary_key_candidate=(
                col_raw in single_key_set
                or (
                    bool(u_dict["is_unique"])
                    and total_rows > 0
                    and null_count == 0
                )
            ),
            top_frequency_ratio=float(u_dict["top_frequency_ratio"]),
            top_value=u_dict["top_value"],
            mode=mode_val,
            mode_percentage=mode_pct,
        )

        # 3. Validity Domain (Phase 1 module)
        rule_fn = active_rules.get(col_raw) or active_rules.get(col_name)
        v_dict = compute_column_validity(ser, rule_fn=rule_fn)
        col_validity = ColumnValidity(
            inferred_type=str(v_dict["inferred_type"]),
            type_error_count=int(v_dict["type_error_count"]),
            type_error_percentage=float(v_dict["type_error_percentage"]),
            whitespace_count=int(v_dict["whitespace_count"]),
            whitespace_percentage=float(v_dict["whitespace_percentage"]),
            rule_error_count=int(v_dict["rule_error_count"]),
            rule_error_percentage=float(v_dict["rule_error_percentage"]),
        )

        # 4. Statistical Domain & 5. Distribution Domain
        is_num_domain = (
            is_numeric_dtype(ser)
            and not is_bool_dtype(ser)
            and role in {"continuous", "discrete"}
        )

        col_statistical: Optional[ColumnStatistical] = None
        col_distribution: Optional[ColumnDistribution] = None

        if is_num_domain:
            if non_null_count > 0:
                stats_dict = compute_numeric_statistics(ser)
                out_dict = detect_outliers(ser)
                cont_dist = compute_continuous_distribution(ser)

                skew_val = stats_dict["skewness"]
                skew_cls: Optional[str] = None
                if skew_val is not None:
                    abs_s = abs(skew_val)
                    skew_cls = (
                        "symmetric"
                        if abs_s < 0.5
                        else ("moderate" if abs_s < 1.0 else "highly_skewed")
                    )

                kurt_val = stats_dict["kurtosis"]
                kurt_cls: Optional[str] = None
                if kurt_val is not None:
                    kurt_cls = (
                        "mesokurtic"
                        if abs(kurt_val) < 0.5
                        else ("leptokurtic" if kurt_val >= 0.5 else "platykurtic")
                    )

                rec_meth = out_dict.get("recommended_method", "iqr")
                out_cnt = int(out_dict.get("recommended_outlier_count", 0))
                out_ratio = float(out_dict.get("recommended_outlier_ratio", 0.0))
                out_pct = float((out_cnt / non_null_count) * 100.0)
                meth_info = out_dict.get(rec_meth, {})

                col_statistical = ColumnStatistical(
                    count=int(stats_dict["count"]),
                    infinite_count=int(stats_dict["infinite_count"]),
                    mean=stats_dict["mean"],
                    median=stats_dict["median"],
                    std=stats_dict["std"],
                    variance=stats_dict["variance"],
                    min=stats_dict["min"],
                    max=stats_dict["max"],
                    p01=stats_dict["p01"],
                    p05=stats_dict["p05"],
                    q25=stats_dict["q25"],
                    q75=stats_dict["q75"],
                    p95=stats_dict["p95"],
                    p99=stats_dict["p99"],
                    iqr=stats_dict["iqr"],
                    mad=stats_dict["mad"],
                    skewness=skew_val,
                    skewness_class=skew_cls,
                    kurtosis=kurt_val,
                    kurtosis_class=kurt_cls,
                    coefficient_of_variation=stats_dict["coefficient_of_variation"],
                    zero_count=int(stats_dict["zero_count"]),
                    zero_ratio=float(stats_dict["zero_ratio"]),
                    negative_count=int(stats_dict["negative_count"]),
                    negative_ratio=float(stats_dict["negative_ratio"]),
                    continuous_entropy=float(stats_dict["continuous_entropy"]),
                    continuous_shannon_entropy=cont_dist.get("shannon_entropy"),
                    continuous_hhi=cont_dist.get("hhi"),
                    continuous_gini_simpson=cont_dist.get("gini_simpson_diversity"),
                    outlier_method=rec_meth,
                    outlier_count=out_cnt,
                    outlier_ratio=out_ratio,
                    outlier_percentage=out_pct,
                    outlier_lower_bound=meth_info.get("lower_bound"),
                    outlier_upper_bound=meth_info.get("upper_bound"),
                    zscore_outliers=dict(out_dict.get("zscore", {})),
                    iqr_outliers=dict(out_dict.get("iqr", {})),
                    modified_zscore_outliers=dict(out_dict.get("modified_zscore", {})),
                )
            else:
                col_statistical = ColumnStatistical(
                    count=0,
                    infinite_count=0,
                    continuous_entropy=0.0,
                    outlier_method=None,
                )
        else:
            cat_dist = compute_categorical_distribution(ser)
            vals_str = ser.dropna().astype(str)
            if not vals_str.empty:
                freq_pct = vals_str.value_counts(normalize=True) * 100.0
                rare_series = freq_pct[freq_pct < 1.0]
                rare_pct = float(rare_series.sum()) if not rare_series.empty else 0.0
            else:
                rare_pct = 0.0

            col_distribution = ColumnDistribution(
                unique_categories=int(cat_dist["unique_categories"]),
                shannon_entropy=float(cat_dist["shannon_entropy"]),
                normalized_entropy=float(cat_dist["normalized_entropy"]),
                hhi=float(cat_dist["hhi"]),
                gini_simpson_diversity=float(cat_dist["gini_simpson_diversity"]),
                top_category=cat_dist["top_category"],
                top_category_ratio=float(cat_dist["top_category_ratio"]),
                top_category_percentage=float(cat_dist["top_category_ratio"] * 100.0),
                is_dominated=bool(cat_dist["is_dominated"]),
                rare_categories_count=int(cat_dist["rare_categories_count"]),
                rare_categories_ratio=float(cat_dist["rare_categories_ratio"]),
                rare_categories_percentage=rare_pct,
                top_frequencies=dict(cat_dist["top_frequencies"]),
            )

        # Optional reference profile adjustment (learned baseline bounds/categories)
        if isinstance(reference_profile, dict) and role not in {"binary", "identifier"}:
            ref_ds = reference_profile.get(dataset_name, {})
            if isinstance(ref_ds, dict):
                ref_col = ref_ds.get(col_raw) or ref_ds.get(col_name)
                if isinstance(ref_col, dict) and non_null_count > 0:
                    if ref_col.get("type") == "numeric" and col_statistical is not None:
                        lb = ref_col.get("lower", np.nan)
                        ub = ref_col.get("upper", np.nan)
                        if pd.notna(lb) and pd.notna(ub):
                            s_clean = ser.dropna()
                            outliers = int(((s_clean < lb) | (s_clean > ub)).sum())
                            out_pct = float((outliers / non_null_count) * 100.0)
                            out_ratio = float(outliers / non_null_count)
                            col_statistical.outlier_count = outliers
                            col_statistical.outlier_percentage = out_pct
                            col_statistical.outlier_ratio = out_ratio
                            col_statistical.outlier_lower_bound = float(lb)
                            col_statistical.outlier_upper_bound = float(ub)
                    elif ref_col.get("type") == "categorical" and col_distribution is not None:
                        popular = ref_col.get("popular", [])
                        if len(popular) > 0:
                            vals = ser.dropna().astype(str)
                            rare_pct_ref = float((~vals.isin(popular)).mean() * 100.0)
                            col_distribution.rare_categories_percentage = rare_pct_ref
                            col_distribution.rare_categories_ratio = float(
                                rare_pct_ref / 100.0
                            )

        # 6. Temporal Domain
        is_dt_col = (
            role == "datetime"
            or "datetime" in dtype_str.lower()
            or is_datetime64_any_dtype(ser)
            or col_raw in cfg_dt_cols
            or col_name in cfg_dt_cols
        )
        col_temporal: Optional[ColumnTemporal] = None
        if is_dt_col:
            t_metrics = compute_temporal_metrics(ser, reference_time=ref_time)
            if is_datetime64_any_dtype(ser):
                ser_dt = ser
            else:
                ser_dt = pd.to_datetime(ser, errors="coerce")

            min_ts = None
            max_ts = None
            range_days: Optional[int] = None
            if ser_dt.notna().any():
                min_ts = ser_dt.min()
                max_ts = ser_dt.max()
                if pd.notna(min_ts) and pd.notna(max_ts):
                    range_days = int((max_ts - min_ts).days)

            is_temp_flag = bool(t_metrics.get("is_temporal", False))
            step_sec = t_metrics.get("median_step_seconds")
            fresh_days = t_metrics.get("freshness_days")
            cadence_str = _classify_cadence(step_sec) if is_temp_flag else None
            fresh_status_str = (
                _classify_freshness(fresh_days, fresh_thresh) if is_temp_flag else None
            )

            col_temporal = ColumnTemporal(
                is_temporal=is_temp_flag,
                parse_rate=float(t_metrics.get("parse_rate", 0.0)),
                min_date=t_metrics.get("min_date") if is_temp_flag else (str(min_ts) if min_ts is not None else None),
                max_date=t_metrics.get("max_date") if is_temp_flag else (str(max_ts) if max_ts is not None else None),
                min_timestamp=min_ts,
                max_timestamp=max_ts,
                date_range_days=range_days,
                time_span_days=t_metrics.get("time_span_days"),
                is_monotonic_increasing=bool(
                    t_metrics.get("is_monotonic_increasing", False)
                ),
                future_date_count=int(t_metrics.get("future_date_count", 0)),
                future_date_ratio=float(t_metrics.get("future_date_ratio", 0.0)),
                median_step_seconds=step_sec,
                max_gap_seconds=t_metrics.get("max_gap_seconds"),
                freshness_days=fresh_days,
                inferred_cadence=cadence_str,
                freshness_status=fresh_status_str,
            )
            col_temporal._reference_time = ref_time  # type: ignore[attr-defined]

        # 7. Drift Domain
        col_drift: Optional[ColumnDrift] = None
        if baseline_df is not None:
            if col_raw in baseline_df.columns:
                b_ser = baseline_df[col_raw]
                base_role = infer_column_role(b_ser, col_raw, len(baseline_df))
                base_h, prod_h = harmonize_series_for_drift(b_ser.dropna(), ser.dropna())

                is_num_drift = (
                    base_role in {"continuous", "discrete"}
                    and is_numeric_dtype(base_h)
                    and not is_bool_dtype(base_h)
                )

                if is_num_drift:
                    d_res = compute_numeric_drift(base_h, prod_h)
                    col_drift = ColumnDrift(
                        psi=float(d_res["psi"]),
                        kl_divergence=float(d_res["kl_divergence"]),
                        reverse_kl_divergence=float(d_res["reverse_kl_divergence"]),
                        js_divergence=float(d_res["js_divergence"]),
                        ks_statistic=float(d_res["ks_statistic"]),
                        ks_p_value=float(d_res["ks_p_value"]),
                        wasserstein_distance=float(d_res["wasserstein_distance"]),
                        normalized_wasserstein=float(d_res["normalized_wasserstein"]),
                        unseen_categories=[],
                        unseen_categories_count=0,
                        unseen_categories_ratio=0.0,
                        drift_detected=bool(d_res["drift_detected"]),
                        drift_severity=str(d_res["drift_severity"]),
                        sample_size_base=int(d_res["sample_size_base"]),
                        sample_size_prod=int(d_res["sample_size_prod"]),
                    )
                else:
                    c_res = compute_categorical_drift(base_h, prod_h)
                    col_drift = ColumnDrift(
                        psi=float(c_res["psi"]),
                        kl_divergence=float(c_res["kl_divergence"]),
                        reverse_kl_divergence=float(c_res["reverse_kl_divergence"]),
                        js_divergence=float(c_res["js_divergence"]),
                        ks_statistic=None,
                        ks_p_value=None,
                        wasserstein_distance=None,
                        normalized_wasserstein=None,
                        unseen_categories=list(c_res["unseen_categories"]),
                        unseen_categories_count=int(c_res["unseen_categories_count"]),
                        unseen_categories_ratio=float(c_res["unseen_categories_ratio"]),
                        drift_detected=bool(c_res["drift_detected"]),
                        drift_severity=str(c_res["drift_severity"]),
                        sample_size_base=int(c_res["sample_size_base"]),
                        sample_size_prod=int(c_res["sample_size_prod"]),
                    )
                    if (
                        is_numeric_dtype(b_ser)
                        and is_numeric_dtype(ser)
                        and not is_bool_dtype(b_ser)
                        and not is_bool_dtype(ser)
                        and role not in {"categorical", "binary", "datetime", "boolean"}
                    ):
                        col_drift._numeric_drift_override = compute_numeric_drift(b_ser, ser)  # type: ignore[attr-defined]
            else:
                col_drift = ColumnDrift(
                    psi=1.0,
                    kl_divergence=1.0,
                    reverse_kl_divergence=1.0,
                    js_divergence=1.0,
                    drift_detected=True,
                    drift_severity="critical",
                )

        column_profiles[col_name] = ColumnProfile(
            name=col_name,
            role=role,
            dtype=dtype_str,
            semantic_type=_infer_semantic_type_from_role(role),
            completeness=col_completeness,
            uniqueness=col_uniqueness,
            validity=col_validity,
            statistical=col_statistical,
            distribution=col_distribution,
            temporal=col_temporal,
            drift=col_drift,
            dtype_object=ser.dtype,
        )

    dataset_profile = DatasetProfile(
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
        dataset_name=dataset_name,
        row_completeness_mean=comp_metrics["row_completeness_mean"],
        unique_rows_count=dup_row_metrics["unique_row_count"],
        overall_null_count=overall_null_count,
        overall_null_percentage=round(overall_null_pct, 4),
        columns=column_profiles,
        relationships=relationships,
    )

    return ProfileContext(
        dataset=dataset_profile,
        columns=column_profiles,
        relationships=relationships,
        reference_profile=reference_profile,
        analysis_level=cfg.analysis_level,
    )


def build_dataset_profile(
    df: pd.DataFrame,
    meta_df: Optional[Union[pd.DataFrame, ProfileContext]] = None,
    overall_score: float = 0.0,
    dim_scores: Optional[Dict[str, DimensionScore]] = None,
    critical_count: int = 0,
    warning_count: int = 0,
    recommendations_count: int = 0,
    **kwargs: Any,
) -> DatasetProfile:
    """Construct an executive, structural dataset-level profile.

    Supports both a pre-built ``ProfileContext`` (via ``meta_df`` or ``context``
    kwarg) and the legacy ``(df, meta_df)`` signature.
    """
    ctx = kwargs.get("context") if isinstance(kwargs.get("context"), ProfileContext) else (
        meta_df if isinstance(meta_df, ProfileContext) else None
    )

    dimension_scores_dict: Dict[str, float] = {}
    if dim_scores:
        for k, ds in dim_scores.items():
            dimension_scores_dict[k] = float(ds.score)

    if ctx is not None:
        dp = ctx.dataset
        dp.overall_score = round(overall_score, 2)
        dp.dimension_scores = dimension_scores_dict
        dp.critical_anomalies_count = critical_count
        dp.warning_anomalies_count = warning_count
        dp.recommendations_count = recommendations_count
        return dp

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
    if isinstance(meta_df, pd.DataFrame) and "column_role" in meta_df.columns:
        role_counts = {str(k): int(v) for k, v in meta_df["column_role"].value_counts().items()}

    # Candidate keys
    key_info = detect_candidate_keys(df)
    single_keys = key_info.get("single_keys", [])
    composite_keys = key_info.get("composite_keys", [])

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
    df: Union[pd.DataFrame, ProfileContext],
    meta_df: Optional[Union[pd.DataFrame, ProfileContext]] = None,
    column_scores: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Build a unified, comprehensive per-column profiling table.

    Accepts either a ``ProfileContext`` (reading pre-computed domain objects directly)
    or the legacy ``(df, meta_df)`` arguments.
    """
    col_scores = column_scores or {}
    ctx: Optional[ProfileContext] = None
    if isinstance(df, ProfileContext):
        ctx = df
    elif isinstance(meta_df, ProfileContext):
        ctx = meta_df

    if ctx is not None:
        total_rows = ctx.dataset.total_rows
        rows = []
        for col_prof in ctx.column_profiles:
            col = col_prof.name
            role = str(col_prof.role)
            dtype_str = str(col_prof.dtype)

            comp = col_prof.completeness
            non_null = comp.non_null_count if comp is not None else 0
            missing = comp.null_count if comp is not None else 0
            missing_pct = comp.null_percentage if comp is not None else 0.0

            uniq = col_prof.uniqueness
            uniq_cnt = uniq.distinct_count if uniq is not None else 0
            uniq_pct = uniq.unique_percentage if uniq is not None else 0.0

            is_uniq = (uniq_cnt == total_rows) and (missing == 0) and (total_rows > 0)
            is_const = (uniq_cnt <= 1) and (total_rows > 0)

            top_val = uniq.mode if (uniq is not None and uniq.mode is not None) else (uniq.top_value if uniq is not None else None)
            top_pct = uniq.mode_percentage if uniq is not None else None
            is_near_const = bool(top_pct is not None and pd.notna(top_pct) and _safe_float(top_pct) >= 95.0)

            stat = col_prof.statistical
            mean_val = stat.mean if stat is not None else None
            std_val = stat.std if stat is not None else None
            median_val = stat.median if stat is not None else None
            min_val = stat.min if stat is not None else None
            max_val = stat.max if stat is not None else None
            skew_val = stat.skewness if stat is not None else None
            if skew_val is not None and pd.notna(skew_val):
                abs_s = abs(float(skew_val))
                skew_lvl = "low" if abs_s < 0.5 else ("moderate" if abs_s < 1.0 else "high")
            else:
                skew_lvl = None

            entropy_val = col_prof.entropy if col_prof.entropy is not None else 0.0

            val = col_prof.validity
            type_err_cnt = val.type_error_count if val is not None else 0
            type_err_pct = round((type_err_cnt / max(1, non_null)) * 100.0, 2)

            ws_cnt = (
                val.whitespace_count
                if (val is not None and role not in {"continuous", "discrete", "datetime"})
                else 0
            )
            ws_pct = round((ws_cnt / max(1, non_null)) * 100.0, 2)

            rule_err_cnt = val.rule_error_count if val is not None else 0
            rule_err_pct = round((rule_err_cnt / max(1, non_null)) * 100.0, 2)

            out_cnt = stat.outlier_count if (stat is not None and role in {"continuous", "discrete"}) else 0
            out_pct = round((out_cnt / max(1, non_null)) * 100.0, 2) if role in {"continuous", "discrete"} else 0.0

            temp = col_prof.temporal
            min_dt = temp.min_timestamp if (temp is not None and temp.min_timestamp is not None) else (temp.min_date if temp is not None else None)
            max_dt = temp.max_timestamp if (temp is not None and temp.max_timestamp is not None) else (temp.max_date if temp is not None else None)
            range_days = temp.date_range_days if temp is not None else None

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
        else:
            col_df.index.name = "column_name"
        return col_df

    assert isinstance(df, pd.DataFrame)
    assert isinstance(meta_df, pd.DataFrame)
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
            "top_value": top_val if pd.notna(top_val) else None,
            "top_frequency_percent": round(float(top_pct), 2) if (top_pct is not None and pd.notna(top_pct)) else None,
            "mean": round(float(mean_val), 4) if mean_val is not None and pd.notna(mean_val) else np.nan,
            "std": round(float(std_val), 4) if std_val is not None and pd.notna(std_val) else np.nan,
            "median": round(float(median_val), 4) if median_val is not None and pd.notna(median_val) else np.nan,
            "min": round(float(min_val), 4) if min_val is not None and pd.notna(min_val) else np.nan,
            "max": round(float(max_val), 4) if max_val is not None and pd.notna(max_val) else np.nan,
            "skewness": round(float(skew_val), 4) if skew_val is not None and pd.notna(skew_val) else np.nan,
            "skewness_level": skew_lvl if pd.notna(skew_lvl) else None,
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
    else:
        col_df.index.name = "column_name"
    return col_df


def build_statistical_profile(
    df: Union[pd.DataFrame, ProfileContext],
    meta_df: Optional[Union[pd.DataFrame, ProfileContext]] = None,
) -> pd.DataFrame:
    """Build a deep statistical intelligence table with moments, quantiles, and diversity.

    Accepts either a ``ProfileContext`` (reading pre-computed ``ColumnStatistical``
    and ``ColumnDistribution`` directly without recomputation) or the legacy
    ``(df, meta_df)`` arguments.
    """
    ctx: Optional[ProfileContext] = None
    if isinstance(df, ProfileContext):
        ctx = df
    elif isinstance(meta_df, ProfileContext):
        ctx = meta_df

    if ctx is not None:
        rows = []
        for col_prof in ctx.column_profiles:
            col = col_prof.name
            role = str(col_prof.role)
            n = col_prof.completeness.non_null_count if col_prof.completeness is not None else 0

            stat_entry: Dict[str, Any] = {
                "column_name": col,
                "column_role": role,
                "count": n,
                "mean": np.nan,
                "std": np.nan,
                "variance": np.nan,
                "median": np.nan,
                "iqr": np.nan,
                "mad": np.nan,
                "coefficient_of_variation": np.nan,
                "min": np.nan,
                "p01": np.nan,
                "p05": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "p95": np.nan,
                "p99": np.nan,
                "max": np.nan,
                "skewness": np.nan,
                "skewness_class": None,
                "kurtosis": np.nan,
                "kurtosis_fisher": np.nan,
                "kurtosis_class": None,
                "zero_count": 0,
                "zero_ratio": 0.0,
                "negative_count": 0,
                "negative_ratio": 0.0,
                "infinite_count": 0,
                "outlier_method": None,
                "outlier_lower_bound": np.nan,
                "outlier_upper_bound": np.nan,
                "outlier_count": 0,
                "outlier_ratio": 0.0,
                "shannon_entropy": np.nan,
                "normalized_entropy": np.nan,
                "hhi": np.nan,
                "gini_simpson_diversity": np.nan,
                "top_category_dominance_ratio": np.nan,
                "is_dominated": False,
                "rare_categories_count": 0,
                "rare_categories_ratio": 0.0,
            }

            stat = col_prof.statistical
            dist = col_prof.distribution

            if stat is not None and role not in {"categorical", "binary", "datetime", "boolean"}:
                if n > 0:
                    stat_entry["mean"] = stat.mean
                    stat_entry["std"] = stat.std
                    stat_entry["variance"] = stat.variance
                    stat_entry["median"] = stat.median
                    stat_entry["iqr"] = stat.iqr
                    stat_entry["mad"] = stat.mad
                    stat_entry["coefficient_of_variation"] = stat.coefficient_of_variation
                    stat_entry["min"] = stat.min
                    stat_entry["p01"] = stat.p01
                    stat_entry["p05"] = stat.p05
                    stat_entry["q25"] = stat.q25
                    stat_entry["q75"] = stat.q75
                    stat_entry["p95"] = stat.p95
                    stat_entry["p99"] = stat.p99
                    stat_entry["max"] = stat.max
                    stat_entry["skewness"] = stat.skewness
                    stat_entry["skewness_class"] = stat.skewness_class
                    stat_entry["kurtosis"] = stat.kurtosis
                    stat_entry["kurtosis_fisher"] = stat.kurtosis_fisher
                    stat_entry["kurtosis_class"] = stat.kurtosis_class
                    stat_entry["zero_count"] = stat.zero_count
                    stat_entry["zero_ratio"] = stat.zero_ratio
                    stat_entry["negative_count"] = stat.negative_count
                    stat_entry["negative_ratio"] = stat.negative_ratio
                    stat_entry["infinite_count"] = stat.infinite_count
                    stat_entry["outlier_method"] = stat.outlier_method
                    stat_entry["outlier_count"] = stat.outlier_count
                    stat_entry["outlier_ratio"] = stat.outlier_ratio
                    stat_entry["outlier_lower_bound"] = (
                        stat.outlier_lower_bound if stat.outlier_lower_bound is not None else np.nan
                    )
                    stat_entry["outlier_upper_bound"] = (
                        stat.outlier_upper_bound if stat.outlier_upper_bound is not None else np.nan
                    )
                    stat_entry["shannon_entropy"] = (
                        stat.continuous_shannon_entropy if stat.continuous_shannon_entropy is not None else np.nan
                    )
                    stat_entry["normalized_entropy"] = (
                        stat.continuous_entropy if stat.continuous_entropy is not None else np.nan
                    )
                    stat_entry["hhi"] = (
                        stat.continuous_hhi if stat.continuous_hhi is not None else np.nan
                    )
                    stat_entry["gini_simpson_diversity"] = (
                        stat.continuous_gini_simpson if stat.continuous_gini_simpson is not None else np.nan
                    )
            elif dist is not None:
                if n > 0:
                    stat_entry["shannon_entropy"] = dist.shannon_entropy
                    stat_entry["normalized_entropy"] = dist.normalized_entropy
                    stat_entry["hhi"] = dist.hhi
                    stat_entry["gini_simpson_diversity"] = dist.gini_simpson_diversity
                    stat_entry["top_category_dominance_ratio"] = dist.top_category_ratio
                    stat_entry["is_dominated"] = dist.is_dominated
                    stat_entry["rare_categories_count"] = dist.rare_categories_count
                    stat_entry["rare_categories_ratio"] = dist.rare_categories_ratio

            rows.append(stat_entry)

        stat_df = pd.DataFrame(rows)
        if not stat_df.empty:
            stat_df.set_index("column_name", inplace=True)
        else:
            stat_df.index.name = "column_name"
        return stat_df

    assert isinstance(df, pd.DataFrame)
    assert isinstance(meta_df, pd.DataFrame)
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
    else:
        stat_df.index.name = "column_name"
    return stat_df


def build_relationship_profile(
    df: Union[pd.DataFrame, ProfileContext],
    config: Optional[CleanalytixConfig] = None,
    *,
    _comp_metrics: Optional[Dict[str, Any]] = None,
) -> RelationshipProfile:
    """Construct a comprehensive cross-column relationship profile.

    Accepts either a ``ProfileContext`` (returning pre-computed ``RelationshipProfile``
    directly) or a ``pd.DataFrame``.

    Statistical Safeguards:
    - Correlation and Cramér's V reflect mathematical association, NOT causation.
    - Missingness co-occurrence represents empirical Jaccard similarity, NOT proof of MCAR/MAR/MNAR.
    - Near-duplicate matching is capped to bounded columns to prevent combinatorial explosion.
    """
    if isinstance(df, ProfileContext):
        return df.relationships if df.relationships is not None else RelationshipProfile()

    cfg = config or CleanalytixConfig()
    max_cols = cfg.max_correlation_columns

    # 1. Numeric correlations
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not is_bool_dtype(df[c])]
    viable_numeric = [c for c in numeric_cols if _safe_nunique(df[c].dropna()) > 1][:max_cols]

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
    viable_cat = [c for c in cat_cols if 1 < _safe_nunique(df[c].dropna()) <= 100]

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
    dup_col_list = list(dup_info.get("duplicate_columns", []))
    near_dup_col_list = list(dup_info.get("near_duplicate_columns", []))
    red_rows = []
    for c1, c2 in dup_col_list:
        red_rows.append({
            "column_1": c1,
            "column_2": c2,
            "redundancy_type": "exact_duplicate",
            "match_ratio": 1.0,
        })
    for item in near_dup_col_list:
        red_rows.append({
            "column_1": item["column_1"],
            "column_2": item["column_2"],
            "redundancy_type": "near_duplicate",
            "match_ratio": round(item["match_ratio"], 4),
        })
    redundant_df = pd.DataFrame(red_rows)

    # 4. Missingness co-occurrence (Jaccard)
    comp_metrics = _comp_metrics if _comp_metrics is not None else compute_completeness_metrics(df)
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
    missing_corr_dict = dict(comp_metrics.get("missingness_correlations", {}))

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
        duplicate_columns=dup_col_list,
        near_duplicate_columns=near_dup_col_list,
        missingness_correlations=missing_corr_dict,
    )


def build_temporal_profile(
    df: Union[pd.DataFrame, ProfileContext],
    meta_df: Optional[Union[pd.DataFrame, ProfileContext]] = None,
    config: Optional[CleanalytixConfig] = None,
) -> pd.DataFrame:
    """Construct a comprehensive temporal quality and cadence table.

    Accepts either a ``ProfileContext`` (reading ``ColumnTemporal`` directly)
    or the legacy ``(df, meta_df)`` arguments.
    """
    ctx: Optional[ProfileContext] = None
    if isinstance(df, ProfileContext):
        ctx = df
    elif isinstance(meta_df, ProfileContext):
        ctx = meta_df

    if ctx is not None:
        rows = []
        for col_prof in ctx.column_profiles:
            temp = col_prof.temporal
            if temp is None or not temp.is_temporal:
                continue

            step_sec = temp.median_step_seconds
            cadence = temp.inferred_cadence or _classify_cadence(step_sec)
            fresh_days = temp.freshness_days
            fresh_status = temp.freshness_status or "UNKNOWN"

            rows.append({
                "column_name": col_prof.name,
                "parse_rate": round(temp.parse_rate * 100.0, 2),
                "min_date": temp.min_date,
                "max_date": temp.max_date,
                "time_span_days": round(temp.time_span_days, 2) if temp.time_span_days is not None else None,
                "span_days": round(temp.time_span_days, 2) if temp.time_span_days is not None else None,
                "freshness_days": round(fresh_days, 2) if fresh_days is not None else None,
                "freshness_status": fresh_status,
                "inferred_cadence": cadence,
                "detected_cadence": cadence,
                "median_step_seconds": round(step_sec, 2) if step_sec is not None else None,
                "max_gap_seconds": round(temp.max_gap_seconds, 2) if temp.max_gap_seconds is not None else None,
                "is_monotonic_increasing": temp.is_monotonic_increasing,
                "future_date_count": temp.future_date_count,
                "future_dates_count": temp.future_date_count,
                "future_date_ratio": round(temp.future_date_ratio * 100.0, 4),
            })

        temp_df = pd.DataFrame(rows)
        if not temp_df.empty:
            temp_df.set_index("column_name", inplace=True)
        else:
            temp_df.index.name = "column_name"
        return temp_df

    assert isinstance(df, pd.DataFrame)
    assert isinstance(meta_df, pd.DataFrame)
    cfg = config or CleanalytixConfig()
    time_cfg = cfg.timeliness_config
    cfg_dt_cols: set = set()
    if time_cfg:
        if isinstance(time_cfg, dict):
            ref_time = time_cfg.get("reference_time")
            fd = time_cfg.get("freshness_days", 30.0)
            fresh_thresh = float(fd) if fd is not None else 30.0
            dt_list = time_cfg.get("datetime_columns")
        else:
            ref_time = getattr(time_cfg, "reference_time", None)
            fd = getattr(time_cfg, "freshness_days", 30.0)
            fresh_thresh = float(fd) if fd is not None else 30.0
            dt_list = getattr(time_cfg, "datetime_columns", None)
        if dt_list:
            cfg_dt_cols = {str(c) for c in dt_list} | set(dt_list)
    else:
        ref_time = None
        fresh_thresh = 30.0

    # Detect datetime columns
    dt_cols = []
    for _, row in meta_df.iterrows():
        col = row.get("column_name")
        role = row.get("column_role")
        dtype = str(row.get("dtype", "")).lower()
        if role == "datetime" or "datetime" in dtype or (col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col])):
            dt_cols.append(col)

    if not dt_cols:
        empty_temp_df = pd.DataFrame()
        empty_temp_df.index.name = "column_name"
        return empty_temp_df

    rows = []
    for col in dt_cols:
        ser = df[col]
        t_metrics = compute_temporal_metrics(ser, reference_time=ref_time)
        if not t_metrics.get("is_temporal", False):
            continue

        step_sec = t_metrics.get("median_step_seconds")
        cadence = _classify_cadence(step_sec)

        fresh_days = t_metrics.get("freshness_days")
        fresh_status = _classify_freshness(fresh_days, fresh_thresh)

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
    else:
        temp_df.index.name = "column_name"
    return temp_df


def build_drift_profile(
    df: Union[pd.DataFrame, ProfileContext],
    baseline_df: Optional[pd.DataFrame] = None,
    meta_df: Optional[Union[pd.DataFrame, ProfileContext]] = None,
) -> Optional[pd.DataFrame]:
    """Construct a comprehensive distribution drift table against a reference baseline.

    Accepts either a ``ProfileContext`` (reading pre-computed ``ColumnDrift``
    directly) or the legacy ``(df, baseline_df, meta_df)`` arguments.
    """
    ctx: Optional[ProfileContext] = None
    if isinstance(df, ProfileContext):
        ctx = df
    elif isinstance(meta_df, ProfileContext):
        ctx = meta_df

    if ctx is not None:
        has_any_drift = any(c.drift is not None for c in ctx.column_profiles)
        if not has_any_drift:
            return None

        rows = []
        for col_prof in ctx.column_profiles:
            d = col_prof.drift
            if d is None:
                continue
            role = str(col_prof.role)
            num_override = getattr(d, "_numeric_drift_override", None)
            if isinstance(num_override, dict):
                rows.append({
                    "column_name": col_prof.name,
                    "column_role": role,
                    "sample_size_base": num_override["sample_size_base"],
                    "sample_size_prod": num_override["sample_size_prod"],
                    "ks_statistic": round(num_override["ks_statistic"], 4),
                    "ks_p_value": round(num_override["ks_p_value"], 6),
                    "wasserstein_distance": round(num_override["wasserstein_distance"], 4),
                    "normalized_wasserstein": round(num_override["normalized_wasserstein"], 4),
                    "psi": round(num_override["psi"], 4),
                    "js_divergence": round(num_override["js_divergence"], 4),
                    "kl_divergence": round(num_override["kl_divergence"], 4),
                    "reverse_kl_divergence": round(num_override["reverse_kl_divergence"], 4),
                    "unseen_categories_count": 0,
                    "unseen_categories_ratio": 0.0,
                    "unseen_categories": [],
                    "unseen_categories_list": [],
                    "drift_detected": num_override["drift_detected"],
                    "drift_severity": str(num_override["drift_severity"]).upper(),
                })
            elif d.ks_statistic is not None:
                rows.append({
                    "column_name": col_prof.name,
                    "column_role": role,
                    "sample_size_base": d.sample_size_base,
                    "sample_size_prod": d.sample_size_prod,
                    "ks_statistic": round(d.ks_statistic, 4),
                    "ks_p_value": round(d.ks_p_value, 6) if d.ks_p_value is not None else np.nan,
                    "wasserstein_distance": round(d.wasserstein_distance, 4) if d.wasserstein_distance is not None else np.nan,
                    "normalized_wasserstein": round(d.normalized_wasserstein, 4) if d.normalized_wasserstein is not None else np.nan,
                    "psi": round(d.psi, 4),
                    "js_divergence": round(d.js_divergence, 4),
                    "kl_divergence": round(d.kl_divergence, 4),
                    "reverse_kl_divergence": round(d.reverse_kl_divergence, 4),
                    "unseen_categories_count": 0,
                    "unseen_categories_ratio": 0.0,
                    "unseen_categories": [],
                    "unseen_categories_list": [],
                    "drift_detected": d.drift_detected,
                    "drift_severity": d.drift_severity.upper(),
                })
            else:
                rows.append({
                    "column_name": col_prof.name,
                    "column_role": role,
                    "sample_size_base": d.sample_size_base,
                    "sample_size_prod": d.sample_size_prod,
                    "ks_statistic": np.nan,
                    "ks_p_value": np.nan,
                    "wasserstein_distance": np.nan,
                    "normalized_wasserstein": np.nan,
                    "psi": round(d.psi, 4),
                    "js_divergence": round(d.js_divergence, 4),
                    "kl_divergence": round(d.kl_divergence, 4),
                    "reverse_kl_divergence": round(d.reverse_kl_divergence, 4),
                    "unseen_categories_count": d.unseen_categories_count,
                    "unseen_categories_ratio": round(d.unseen_categories_ratio * 100.0, 2),
                    "unseen_categories": d.unseen_categories[:10],
                    "unseen_categories_list": d.unseen_categories[:10],
                    "drift_detected": d.drift_detected,
                    "drift_severity": d.drift_severity.upper(),
                })

        drift_df = pd.DataFrame(rows)
        if not drift_df.empty:
            drift_df.set_index("column_name", inplace=True)
        else:
            drift_df.index.name = "column_name"
        return drift_df

    assert isinstance(df, pd.DataFrame)
    if baseline_df is None:
        return None

    shared_cols = [c for c in df.columns if c in baseline_df.columns]
    if not shared_cols:
        return pd.DataFrame()

    rows = []
    for col in shared_cols:
        role = "unknown"
        if isinstance(meta_df, pd.DataFrame) and "column_name" in meta_df.columns and "column_role" in meta_df.columns:
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
            b_h, p_h = harmonize_series_for_drift(b_ser.dropna(), p_ser.dropna())
            c_res = compute_categorical_drift(b_h, p_h)
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
    else:
        drift_df.index.name = "column_name"
    return drift_df
