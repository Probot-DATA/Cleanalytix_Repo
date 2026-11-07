"""Authoritative 6-dimension Data Quality scoring engine with dynamic metric applicability.

Evaluates datasets across the 6 standard data quality dimensions:
1. Completeness: cell-level missingness and row-level null density
2. Validity: type conformity, whitespace cleanliness, custom business rule validation
3. Uniqueness: dataset duplicate rows and identifier/key column uniqueness
4. Consistency: cross-column logical business rules and outlier plausibility containment
5. Timeliness: freshness against reference time, future date containment, sequence integrity
6. Stability: distribution drift (PSI, JSD, KS) against reference baseline datasets

Guarantees:
- Authoritative 6 default dimension weights sum strictly to 1.0.
- Non-applicable dimensions are excluded from the score (no free 100% scores).
- Configured weights for applicable dimensions are dynamically re-normalized to 1.0.
- DimensionScore transparently exposes applicable, reason, configured_weight, and effective_weight.
- Metric-level custom weights can be configured within each dimension.
- Overall score and dimension scores are strictly bounded in [0.0, 100.0].
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple, Sequence, Union, Mapping, cast
import math
import warnings
import numpy as np
import pandas as pd

from cleanalytix.types import (
    QualityDimension,
    DimensionScore,
    CleanalytixConfig,
    ConsistencyRule,
    TimelinessConfig,
    ScoreMode,
    Thresholds,
    MetricWeights,
)
from cleanalytix.registry import (
    METRIC_CATALOG,
    MetricDefinition,
    MetricDomain,
    MetricScope,
    MetricValueStatus,
)
from cleanalytix.metrics.temporal import compute_temporal_metrics
from cleanalytix.metrics.uniqueness import (
    compute_identifier_uniqueness,
    compute_duplicate_row_metrics,
)

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "missing_percent": 20.0,
    "duplicate_rows_percent": 1.0,
    "outlier_percent": 10.0,
    "whitespace_percent": 5.0,
    "type_errors_percent": 2.0,
    "rule_errors_percent": 2.0,
    "psi_divergence": 0.25,
    "js_divergence": 0.10,
    "freshness_days": 30.0,
    "future_dates_percent": 0.1,
}

# Authoritative 6-dimension default weights strictly summing to 1.0
DEFAULT_DIMENSION_WEIGHTS: Dict[QualityDimension, float] = {
    QualityDimension.COMPLETENESS: 0.20,
    QualityDimension.VALIDITY: 0.20,
    QualityDimension.UNIQUENESS: 0.15,
    QualityDimension.CONSISTENCY: 0.15,
    QualityDimension.TIMELINESS: 0.15,
    QualityDimension.STABILITY: 0.15,
}

# Default metric-level weights within dimensions
DEFAULT_METRIC_WEIGHTS: Dict[str, float] = {
    "completeness_cell": 0.70,
    "completeness_row": 0.30,
    "validity_type_errors": 0.40,
    "validity_whitespace": 0.20,
    "validity_rules": 0.40,
    "validity_outliers": 0.30,
    "uniqueness_duplicate_rows": 0.50,
    "uniqueness_identifiers": 0.50,
    "consistency_rules": 1.00,
    "timeliness_freshness": 0.50,
    "timeliness_future_dates": 0.30,
    "timeliness_sequence": 0.20,
    "stability_psi": 0.50,
    "stability_jsd": 0.50,
}


def resolve_thresholds(user_thresholds: Optional[Thresholds] = None) -> Dict[str, float]:
    """Merge user-supplied thresholds with system defaults."""
    final = {k.lower(): float(v) for k, v in DEFAULT_THRESHOLDS.items()}
    if not user_thresholds:
        return final

    for key, val in user_thresholds.items():
        key_lower = str(key).lower()
        if key_lower in final:
            try:
                final[key_lower] = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Threshold for '{key}' must be numeric")
        else:
            warnings.warn(
                f"Unknown threshold key '{key}' — ignoring it.",
                UserWarning,
                stacklevel=3,
            )
    return final


def resolve_dimension_weights(
    user_weights: Optional[Mapping[Union[QualityDimension, str], float]] = None,
) -> Dict[QualityDimension, float]:
    """Validate and resolve configured dimension weights.

    Policy:
    - All weights must be non-negative (raises ValueError if any < 0).
    - Weights cannot all be zero (raises ValueError if sum <= 0).
    - If user provides all 6 dimensions, normalizes so sum == 1.0.
    - If user provides a partial dictionary that sums to 1.0 (within 1e-6),
      omitted dimensions are assigned 0.0 weight.
    - If user provides a partial dictionary not summing to 1.0, specified
      dimensions take their configured values, unspecified dimensions inherit
      their relative default weights, and the entire set is normalized to sum to 1.0.
    """
    if not user_weights:
        return dict(DEFAULT_DIMENSION_WEIGHTS)

    # Normalize keys to QualityDimension enum
    normalized_input: Dict[QualityDimension, float] = {}
    for k, v in user_weights.items():
        if isinstance(k, QualityDimension):
            dim = k
        else:
            try:
                dim = QualityDimension(str(k).lower())
            except ValueError:
                valid_names = [d.value for d in QualityDimension]
                raise ValueError(f"Unknown quality dimension '{k}'. Expected one of: {valid_names}")

        try:
            val = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Weight for dimension '{dim.value}' must be numeric, got {v!r}")

        if val < 0.0:
            raise ValueError(f"Dimension weights must be non-negative. Got {val} for '{dim.value}'")
        normalized_input[dim] = val

    if sum(normalized_input.values()) <= 0.0:
        raise ValueError("Dimension weights cannot all be zero.")

    all_dims = list(QualityDimension)
    resolved: Dict[QualityDimension, float] = {}

    if len(normalized_input) == len(all_dims):
        # All 6 dimensions explicitly specified
        tot = sum(normalized_input.values())
        for dim in all_dims:
            resolved[dim] = normalized_input[dim] / tot
        return resolved

    # Partial specification
    spec_sum = sum(normalized_input.values())
    if math.isclose(spec_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        # User specified an exact 1.0 allocation over their chosen dimensions
        for dim in all_dims:
            resolved[dim] = normalized_input.get(dim, 0.0)
        return resolved

    # Unspecified dimensions inherit default weights, then normalize
    for dim in all_dims:
        if dim in normalized_input:
            resolved[dim] = normalized_input[dim]
        else:
            resolved[dim] = DEFAULT_DIMENSION_WEIGHTS[dim]

    tot = sum(resolved.values())
    for dim in all_dims:
        resolved[dim] = resolved[dim] / tot
    return resolved


def resolve_metric_weights(
    user_metric_weights: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Merge user-supplied metric-level weights with defaults."""
    res = dict(DEFAULT_METRIC_WEIGHTS)
    if not user_metric_weights:
        return res

    for k, v in user_metric_weights.items():
        k_lower = str(k).lower()
        if k_lower in res:
            try:
                val = float(v)
                if val < 0.0:
                    raise ValueError(f"Metric weight for '{k}' must be non-negative")
                res[k_lower] = val
            except (TypeError, ValueError) as err:
                raise ValueError(f"Invalid metric weight for '{k}': {err}")
    return res


def calculate_metric_score(
    value: float,
    threshold: float,
    score_mode: str = "linear",
    k: float = 2.0,
) -> float:
    """Calculate single metric score in [0.0, 1.0] given error value and threshold."""
    if threshold <= 0 or not np.isfinite(threshold):
        threshold = 1.0
    v = max(0.0, float(value))

    if score_mode == "linear":
        score = 1.0 - (v / threshold)
        return float(max(0.0, min(1.0, score)))
    else:
        # Exponential mode
        severity = v / threshold
        return float(np.exp(-k * severity))


def _evaluate_consistency_rules(
    df: pd.DataFrame,
    rules: Sequence[Union[ConsistencyRule, Dict[str, Any], Any]],
    rule_threshold: float,
    score_mode: str,
) -> Tuple[float, Dict[str, Any]]:
    """Safely evaluate user-defined cross-column consistency rules without eval()."""
    total_rows = len(df)
    if total_rows == 0 or not rules:
        return 1.0, {"evaluated_rules": 0, "rule_details": {}}

    rule_scores: List[float] = []
    rule_details: Dict[str, Any] = {}

    for idx, r in enumerate(rules):
        name = f"rule_{idx+1}"
        cond_fn = None
        if isinstance(r, ConsistencyRule):
            name = r.name
            cond_fn = r.condition
        elif isinstance(r, dict):
            name = str(r.get("name", name))
            cond_fn = r.get("condition")
        elif callable(r):
            name = getattr(r, "__name__", name)
            cond_fn = r

        if not callable(cond_fn):
            continue

        try:
            res = cond_fn(df)
            if isinstance(res, (pd.Series, np.ndarray, list)):
                res_arr = np.asarray(res, dtype=bool)
                valid_count = int(np.sum(res_arr))
                violation_count = max(0, total_rows - valid_count)
            elif isinstance(res, (bool, np.bool_)):
                violation_count = 0 if bool(res) else total_rows
            else:
                violation_count = 0

            violation_pct = (violation_count / total_rows) * 100.0
            r_score = calculate_metric_score(violation_pct, rule_threshold, score_mode=score_mode)
            rule_scores.append(r_score)
            rule_details[name] = {
                "violations": violation_count,
                "violation_percent": round(violation_pct, 4),
                "score": round(r_score * 100.0, 2),
            }
        except Exception as ex:
            # Rule execution failure treats evaluated rule as violated
            rule_scores.append(0.0)
            rule_details[name] = {
                "error": str(ex),
                "violations": total_rows,
                "violation_percent": 100.0,
                "score": 0.0,
            }

    mean_score = float(np.mean(rule_scores)) if rule_scores else 1.0
    return mean_score, {"evaluated_rules": len(rule_scores), "rule_details": rule_details}


def _evaluate_timeliness(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    timeliness_config: Optional[Union[TimelinessConfig, Dict[str, Any]]],
    thresholds: Dict[str, float],
    score_mode: str,
    metric_weights: Dict[str, float],
) -> Tuple[bool, Optional[str], float, Dict[str, float], Dict[str, float], Dict[str, Any]]:
    """Evaluate Timeliness quality dimension using temporal metrics."""
    # Find candidate datetime columns
    specified_cols = None
    ref_time = None
    fresh_days_override = None
    allow_future = False
    require_monotonic = False

    if timeliness_config is not None:
        if isinstance(timeliness_config, TimelinessConfig):
            specified_cols = timeliness_config.datetime_columns
            ref_time = timeliness_config.reference_time
            fresh_days_override = timeliness_config.freshness_days
            allow_future = timeliness_config.allow_future_dates
            require_monotonic = getattr(timeliness_config, "require_monotonic", False)
        elif isinstance(timeliness_config, dict):
            specified_cols = timeliness_config.get("datetime_columns")
            ref_time = timeliness_config.get("reference_time")
            fresh_days_override = timeliness_config.get("freshness_days")
            allow_future = timeliness_config.get("allow_future_dates", False)
            require_monotonic = bool(timeliness_config.get("require_monotonic", False))

    dt_cols: List[str] = []
    if specified_cols:
        dt_cols = [c for c in specified_cols if c in df.columns]
    else:
        # Check inferred roles from meta_df or DataFrame dtypes
        if not meta_df.empty and "column_role" in meta_df.columns and "column_name" in meta_df.columns:
            for _, r in meta_df.iterrows():
                if r.get("column_role") == "datetime":
                    c = r.get("column_name")
                    if c in df.columns:
                        dt_cols.append(c)

        if not dt_cols:
            for c in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    dt_cols.append(c)

    if not dt_cols:
        return (
            False,
            "No datetime columns detected",
            0.0,
            {},
            {},
            {"detected_columns": []},
        )

    fresh_thresh = (
        float(fresh_days_override)
        if fresh_days_override is not None
        else thresholds.get("freshness_days", 30.0)
    )
    future_thresh = thresholds.get("future_dates_percent", 0.1)

    w_fresh = metric_weights.get("timeliness_freshness", 0.50)
    w_future = metric_weights.get("timeliness_future_dates", 0.30)
    w_seq = metric_weights.get("timeliness_sequence", 0.20)
    tot_w = w_fresh + w_future + w_seq
    if tot_w > 0:
        w_fresh, w_future, w_seq = w_fresh / tot_w, w_future / tot_w, w_seq / tot_w

    col_scores: List[float] = []
    all_metrics: Dict[str, float] = {}
    metric_subscores: Dict[str, float] = {}
    col_details: Dict[str, Any] = {}

    for c in dt_cols:
        t_res = compute_temporal_metrics(df[c], reference_time=ref_time)
        if not t_res.get("is_temporal", False):
            continue

        fresh_days = t_res.get("freshness_days")
        if fresh_days is None:
            s_fresh = 1.0
            actual_fresh = 0.0
        else:
            actual_fresh = max(0.0, float(fresh_days))
            s_fresh = calculate_metric_score(actual_fresh, fresh_thresh, score_mode=score_mode)

        fut_ratio = float(t_res.get("future_date_ratio", 0.0))
        fut_pct = fut_ratio * 100.0
        if allow_future:
            s_future = 1.0
        else:
            s_future = calculate_metric_score(fut_pct, future_thresh, score_mode=score_mode)

        parse_rate = float(t_res.get("parse_rate", 1.0))
        if require_monotonic:
            is_mono = 1.0 if t_res.get("is_monotonic_increasing", False) else 0.0
            s_seq = 0.5 * parse_rate + 0.5 * is_mono
        else:
            s_seq = parse_rate

        col_score = (w_fresh * s_fresh + w_future * s_future + w_seq * s_seq) * 100.0
        col_scores.append(col_score)

        all_metrics[f"{c}_freshness_days"] = actual_fresh
        all_metrics[f"{c}_future_dates_percent"] = fut_pct
        all_metrics[f"{c}_parse_rate"] = parse_rate

        col_details[c] = {
            "freshness_days": actual_fresh,
            "future_dates_count": t_res.get("future_date_count", 0),
            "future_dates_percent": round(fut_pct, 4),
            "parse_rate": round(parse_rate, 4),
            "is_monotonic_increasing": t_res.get("is_monotonic_increasing", False),
            "score": round(col_score, 2),
        }

    if not col_scores:
        return (
            False,
            "Datetime columns could not be parsed into temporal timestamps",
            0.0,
            {},
            {},
            {"detected_columns": dt_cols},
        )

    timeliness_score = float(np.mean(col_scores))
    metric_subscores["freshness_score"] = round(s_fresh * 100.0, 2)
    metric_subscores["future_dates_score"] = round(s_future * 100.0, 2)
    metric_subscores["sequence_score"] = round(s_seq * 100.0, 2)

    return (
        True,
        None,
        timeliness_score,
        all_metrics,
        metric_subscores,
        {"columns": col_details},
    )


def score_dataset(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    config: Optional[CleanalytixConfig] = None,
    has_baseline: bool = False,
    return_details: bool = False,
) -> Any:
    """Compute overall quality score, 6 dimension scores, and column-level scores.

    Applies dynamic applicability re-weighting so that non-applicable dimensions
    (such as Stability when no baseline is present, or Timeliness when no datetime
    columns exist) are excluded from scoring, with their weight proportionally
    distributed among applicable dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze (read-only).
    meta_df : pd.DataFrame
        Metadata table returned by generate_meta.
    config : CleanalytixConfig, optional
        Custom thresholds, weights, and rules.
    has_baseline : bool, default False
        Whether baseline dataset is present for Stability/drift scoring.
    return_details : bool, default False
        If True, returns an extended tuple with scoring_matrix, dimension_summary,
        metric_inventory, and scoring_audit_trail. If False (default), returns the
        backward-compatible 3-element tuple (overall_score, dimension_scores, final_col_scores).
    """
    if config is None:
        config = CleanalytixConfig()

    thresholds = resolve_thresholds(config.thresholds)
    score_mode = config.score_mode
    configured_dim_weights = resolve_dimension_weights(config.dimension_weights)
    metric_weights = resolve_metric_weights(config.metric_weights)
    total_rows = len(df)

    if total_rows == 0 or meta_df.empty:
        # Empty dataset edge case: all non-applicable
        empty_dims = {
            dim.value: DimensionScore(
                dimension=dim,
                score=0.0,
                weight=0.0,
                configured_weight=configured_dim_weights[dim],
                effective_weight=0.0,
                applicable=False,
                reason="Dataset is empty (0 rows or 0 profiled columns)",
            )
            for dim in QualityDimension
        }
        if return_details:
            return 0.0, empty_dims, {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "Empty dataset"
        return 0.0, empty_dims, {}

    col_scores_collector: Dict[str, List[Tuple[float, float]]] = {col: [] for col in df.columns}
    per_col_subscores: Dict[str, Dict[str, float]] = {col: {} for col in df.columns}

    # =========================================================================
    # 1. COMPLETENESS DIMENSION (cell completeness + row completeness)
    # =========================================================================
    comp_col_scores = []
    total_missing_cells = 0
    total_cells = total_rows * max(1, len(df.columns))

    for _, row in meta_df.iterrows():
        col = row.get("column_name")
        if col not in col_scores_collector:
            continue

        miss_pct = row.get("missing_percent", 0.0)
        miss_pct = float(miss_pct) if pd.notna(miss_pct) else 0.0
        miss_cnt = row.get("missing_count", 0)
        total_missing_cells += int(miss_cnt or 0)

        m_score = calculate_metric_score(
            miss_pct, thresholds.get("missing_percent", 20.0), score_mode=score_mode
        )
        comp_col_scores.append(m_score)
        col_scores_collector[col].append((m_score, 1.0))
        per_col_subscores[col]["completeness_score"] = round(m_score * 100.0, 2)

    avg_cell_missing_pct = (total_missing_cells / total_cells) * 100.0 if total_cells > 0 else 0.0
    cell_comp_score = calculate_metric_score(
        avg_cell_missing_pct, thresholds.get("missing_percent", 20.0), score_mode=score_mode
    )

    try:
        incomplete_rows_pct = float(df.isna().any(axis=1).mean() * 100.0)
    except Exception:
        incomplete_rows_pct = avg_cell_missing_pct

    row_comp_score = calculate_metric_score(
        incomplete_rows_pct, thresholds.get("missing_percent", 20.0), score_mode=score_mode
    )

    w_c_cell = metric_weights.get("completeness_cell", 0.70)
    w_c_row = metric_weights.get("completeness_row", 0.30)
    tot_c_w = w_c_cell + w_c_row
    if tot_c_w > 0:
        w_c_cell, w_c_row = w_c_cell / tot_c_w, w_c_row / tot_c_w

    completeness_score = (w_c_cell * cell_comp_score + w_c_row * row_comp_score) * 100.0

    completeness_dim = DimensionScore(
        dimension=QualityDimension.COMPLETENESS,
        score=round(completeness_score, 2),
        configured_weight=configured_dim_weights[QualityDimension.COMPLETENESS],
        applicable=True,
        metrics={
            "cell_missing_percent": round(avg_cell_missing_pct, 4),
            "incomplete_rows_percent": round(incomplete_rows_pct, 4),
        },
        metric_scores={
            "cell_completeness_score": round(cell_comp_score * 100.0, 2),
            "row_completeness_score": round(row_comp_score * 100.0, 2),
        },
    )

    # =========================================================================
    # 2. VALIDITY DIMENSION (type errors, whitespace, custom business rules, outliers)
    # =========================================================================
    validity_column_scores = []
    w_v_type = metric_weights.get("validity_type_errors", 0.40)
    w_v_ws = metric_weights.get("validity_whitespace", 0.20)
    w_v_rule = metric_weights.get("validity_rules", 0.40)
    w_v_out = metric_weights.get("validity_outliers", 0.30)

    for _, row in meta_df.iterrows():
        col = row.get("column_name")
        if col not in col_scores_collector:
            continue

        role = str(row.get("column_role", "unknown"))
        n_vals = int(row.get("non_null_count", 0)) + int(row.get("missing_count", 0))
        n_safe = max(1, n_vals)

        val_metric_scores = []
        val_metric_weights = []

        t_score_val = np.nan
        w_score_val = np.nan
        r_score_val = np.nan
        o_score_val = np.nan

        # Type errors
        type_err = row.get("type_errors", 0)
        if pd.notna(type_err):
            type_pct = (float(type_err) / n_safe) * 100.0
            t_score = calculate_metric_score(
                type_pct, thresholds.get("type_errors_percent", 2.0), score_mode=score_mode
            )
            val_metric_scores.append(t_score)
            val_metric_weights.append(w_v_type)
            t_score_val = round(t_score * 100.0, 2)

        # Whitespace
        if role in {"categorical", "free_text", "identifier"}:
            ws_err = row.get("whitespace_issues", 0)
            if pd.notna(ws_err):
                ws_pct = (float(ws_err) / n_safe) * 100.0
                w_score = calculate_metric_score(
                    ws_pct, thresholds.get("whitespace_percent", 5.0), score_mode=score_mode
                )
                val_metric_scores.append(w_score)
                val_metric_weights.append(w_v_ws)
                w_score_val = round(w_score * 100.0, 2)

        # Rules
        rule_err = row.get("rule_errors", 0)
        if pd.notna(rule_err) and config.rules and col in config.rules:
            rule_pct = (float(rule_err) / n_safe) * 100.0
            r_score = calculate_metric_score(
                rule_pct, thresholds.get("rule_errors_percent", 2.0), score_mode=score_mode
            )
            val_metric_scores.append(r_score)
            val_metric_weights.append(w_v_rule)
            r_score_val = round(r_score * 100.0, 2)

        # Outlier plausibility (continuous / discrete numeric columns)
        if role in {"continuous", "discrete"}:
            n_non_null = int(row.get("non_null_count", 0))
            if n_non_null >= 3:
                out_cnt = row.get("outlier_count", 0)
                out_pct = (float(out_cnt or 0) / max(1, n_non_null)) * 100.0
                o_score = calculate_metric_score(
                    out_pct, thresholds.get("outlier_percent", 10.0), score_mode=score_mode
                )
                val_metric_scores.append(o_score)
                val_metric_weights.append(w_v_out)
                o_score_val = round(o_score * 100.0, 2)

        if val_metric_scores:
            sub_w_tot = sum(val_metric_weights)
            col_val_score = sum(s * (w / sub_w_tot) for s, w in zip(val_metric_scores, val_metric_weights))
            validity_column_scores.append(col_val_score)
            col_scores_collector[col].append((col_val_score, 1.0))
            per_col_subscores[col]["validity_score"] = round(col_val_score * 100.0, 2)
        else:
            per_col_subscores[col]["validity_score"] = 100.0

        per_col_subscores[col]["type_score"] = t_score_val
        per_col_subscores[col]["whitespace_score"] = w_score_val
        per_col_subscores[col]["rule_score"] = r_score_val
        per_col_subscores[col]["outlier_score"] = o_score_val

    validity_score = float(np.mean(validity_column_scores) * 100.0) if validity_column_scores else 100.0

    validity_dim = DimensionScore(
        dimension=QualityDimension.VALIDITY,
        score=round(validity_score, 2),
        configured_weight=configured_dim_weights[QualityDimension.VALIDITY],
        applicable=True,
        metrics={"average_validity": round(validity_score, 2)},
        metric_scores={"column_validity_score": round(validity_score, 2)},
    )

    # =========================================================================
    # 3. UNIQUENESS DIMENSION (duplicate rows + identifier column uniqueness)
    # =========================================================================
    dup_percent = 0.0
    if "duplicate_rows_percent" in meta_df.columns:
        first_dup = meta_df["duplicate_rows_percent"].iloc[0]
        if pd.notna(first_dup):
            dup_percent = float(first_dup)
    else:
        dup_metrics = compute_duplicate_row_metrics(df)
        dup_percent = dup_metrics["duplicate_row_ratio"] * 100.0

    dup_thresh = thresholds.get("duplicate_rows_percent", 1.0)
    dup_row_score = calculate_metric_score(dup_percent, dup_thresh, score_mode=score_mode)

    # Check identifier uniqueness
    id_res = compute_identifier_uniqueness(df, identifier_columns=config.identifier_columns)
    has_ids = id_res.get("has_identifiers", False)

    w_u_rows = metric_weights.get("uniqueness_duplicate_rows", 0.50)
    w_u_ids = metric_weights.get("uniqueness_identifiers", 0.50)

    if has_ids:
        id_score = calculate_metric_score(
            id_res["mean_duplicate_percent"], dup_thresh, score_mode=score_mode
        )
        tot_u_w = w_u_rows + w_u_ids
        if tot_u_w > 0:
            w_u_rows, w_u_ids = w_u_rows / tot_u_w, w_u_ids / tot_u_w
        uniqueness_score = (w_u_rows * dup_row_score + w_u_ids * id_score) * 100.0
        id_score_val = round(id_score * 100.0, 2)
    else:
        uniqueness_score = dup_row_score * 100.0
        id_score_val = 100.0

    uniqueness_dim = DimensionScore(
        dimension=QualityDimension.UNIQUENESS,
        score=round(uniqueness_score, 2),
        configured_weight=configured_dim_weights[QualityDimension.UNIQUENESS],
        applicable=True,
        metrics={
            "duplicate_rows_percent": round(dup_percent, 4),
            "identifier_duplicate_percent": round(id_res.get("mean_duplicate_percent", 0.0), 4),
        },
        metric_scores={
            "duplicate_rows_score": round(dup_row_score * 100.0, 2),
            "identifier_uniqueness_score": id_score_val,
        },
        details={"identifier_columns": id_res.get("identifier_columns", [])},
    )

    # =========================================================================
    # 4. CONSISTENCY DIMENSION (cross-column logical business rules)
    # =========================================================================
    consistency_subscores: Dict[str, float] = {}
    consistency_metrics: Dict[str, float] = {}

    has_consistency_rules = bool(config.consistency_rules)

    if has_consistency_rules:
        rule_score_val, rule_eval_details = _evaluate_consistency_rules(
            df=df,
            rules=config.consistency_rules,  # type: ignore[arg-type]
            rule_threshold=thresholds.get("rule_errors_percent", 2.0),
            score_mode=score_mode,
        )
        consistency_subscores["cross_column_rules_score"] = round(rule_score_val * 100.0, 2)
        consistency_score = rule_score_val * 100.0
        consistency_applicable = True
        consistency_reason = None
    else:
        consistency_score = 0.0
        consistency_applicable = False
        consistency_reason = "No cross-column consistency rules configured"
        rule_eval_details = {}

    consistency_dim = DimensionScore(
        dimension=QualityDimension.CONSISTENCY,
        score=round(consistency_score, 2),
        configured_weight=configured_dim_weights[QualityDimension.CONSISTENCY],
        applicable=consistency_applicable,
        reason=consistency_reason,
        metrics=consistency_metrics,
        metric_scores=consistency_subscores,
        details=rule_eval_details,
    )

    # =========================================================================
    # 5. TIMELINESS DIMENSION (freshness, future date containment, sequence)
    # =========================================================================
    (
        time_applicable,
        time_reason,
        time_score_val,
        time_metrics,
        time_subscores,
        time_details,
    ) = _evaluate_timeliness(
        df=df,
        meta_df=meta_df,
        timeliness_config=config.timeliness_config,
        thresholds=thresholds,
        score_mode=score_mode,
        metric_weights=metric_weights,
    )

    timeliness_dim = DimensionScore(
        dimension=QualityDimension.TIMELINESS,
        score=round(time_score_val, 2),
        configured_weight=configured_dim_weights[QualityDimension.TIMELINESS],
        applicable=time_applicable,
        reason=time_reason,
        metrics=time_metrics,
        metric_scores=time_subscores,
        details=time_details,
    )

    # =========================================================================
    # 6. STABILITY DIMENSION (PSI, JSD, KS drift against baseline)
    # =========================================================================
    stability_scores: List[float] = []
    stability_metrics: Dict[str, float] = {}
    stability_subscores: Dict[str, float] = {}
    stability_applicable = False
    stability_reason = "No baseline reference dataset provided"
    stability_score_val = 0.0

    w_s_psi = metric_weights.get("stability_psi", 0.50)
    w_s_jsd = metric_weights.get("stability_jsd", 0.50)
    tot_s_w = w_s_psi + w_s_jsd
    if tot_s_w > 0:
        w_s_psi, w_s_jsd = w_s_psi / tot_s_w, w_s_jsd / tot_s_w

    if has_baseline:
        for _, row in meta_df.iterrows():
            col = row.get("column_name")
            if col not in col_scores_collector:
                continue

            psi_val = row.get("psi_divergence")
            js_val = row.get("js_divergence")

            if pd.notna(psi_val) and pd.notna(js_val):
                p_score = calculate_metric_score(
                    float(psi_val), thresholds.get("psi_divergence", 0.25), score_mode=score_mode, k=3.0
                )
                j_score = calculate_metric_score(
                    float(js_val), thresholds.get("js_divergence", 0.10), score_mode=score_mode, k=3.0
                )
                drift_score = w_s_psi * p_score + w_s_jsd * j_score
                stability_scores.append(drift_score)
                col_scores_collector[col].append((drift_score, 1.0))
                stability_metrics[f"{col}_psi"] = float(psi_val)
                stability_metrics[f"{col}_jsd"] = float(js_val)
                per_col_subscores[col]["stability_score"] = round(drift_score * 100.0, 2)
            else:
                per_col_subscores[col]["stability_score"] = np.nan

        if stability_scores:
            stability_score_val = float(np.mean(stability_scores) * 100.0)
            stability_applicable = True
            stability_reason = None
            stability_subscores["mean_drift_score"] = round(stability_score_val, 2)
        else:
            stability_applicable = False
            stability_reason = "No drift divergence metrics could be computed between baseline and production"
    else:
        for col in df.columns:
            per_col_subscores[col]["stability_score"] = np.nan

    stability_dim = DimensionScore(
        dimension=QualityDimension.STABILITY,
        score=round(stability_score_val, 2),
        configured_weight=configured_dim_weights[QualityDimension.STABILITY],
        applicable=stability_applicable,
        reason=stability_reason,
        metrics=stability_metrics,
        metric_scores=stability_subscores,
    )

    # Collect all 6 dimensions
    dimension_scores: Dict[str, DimensionScore] = {
        QualityDimension.COMPLETENESS.value: completeness_dim,
        QualityDimension.VALIDITY.value: validity_dim,
        QualityDimension.UNIQUENESS.value: uniqueness_dim,
        QualityDimension.CONSISTENCY.value: consistency_dim,
        QualityDimension.TIMELINESS.value: timeliness_dim,
        QualityDimension.STABILITY.value: stability_dim,
    }

    # =========================================================================
    # APPLICABILITY-AWARE AGGREGATION & RE-NORMALIZATION
    # =========================================================================
    applicable_dims = [ds for ds in dimension_scores.values() if ds.applicable]
    total_applicable_configured_w = sum(ds.configured_weight for ds in applicable_dims)

    if total_applicable_configured_w > 0:
        for ds in dimension_scores.values():
            if ds.applicable:
                eff_w = ds.configured_weight / total_applicable_configured_w
                ds.effective_weight = eff_w
                ds.weight = ds.effective_weight
            else:
                ds.effective_weight = 0.0
                ds.weight = 0.0

        # Exact mathematical score calculation with unrounded weights
        overall_score = sum(
            ds.score * (ds.configured_weight / total_applicable_configured_w)
            for ds in applicable_dims
        )
        overall_score = float(max(0.0, min(100.0, round(overall_score, 2))))
    else:
        # None applicable
        for ds in dimension_scores.values():
            ds.effective_weight = 0.0
            ds.weight = 0.0
        overall_score = 0.0

    # Compute per-column scores
    final_col_scores: Dict[str, float] = {}
    for col, pairs in col_scores_collector.items():
        if not pairs:
            final_col_scores[col] = 100.0
        else:
            tot_w = sum(w for _, w in pairs)
            weighted_s = sum(s * w for s, w in pairs)
            final_col_scores[col] = round((weighted_s / tot_w) * 100.0, 2)

    for col in df.columns:
        per_col_subscores[col]["overall_column_score"] = final_col_scores.get(col, 100.0)

    if not return_details:
        return overall_score, dimension_scores, final_col_scores

    # =========================================================================
    # BUILD EXTENDED ANALYTICAL TABLES (Scoring Matrix, Dimension Summary, Audit)
    # =========================================================================
    # 1. Scoring Matrix DataFrame
    scoring_rows = []
    for col in df.columns:
        sc = per_col_subscores.get(col, {})
        scoring_rows.append({
            "column_name": col,
            "completeness_score": sc.get("completeness_score", 100.0),
            "incomplete_rows_score": np.nan,  # Dataset-level metric: strictly NaN at column level
            "validity_score": sc.get("validity_score", 100.0),
            "type_score": sc.get("type_score", np.nan),
            "whitespace_score": sc.get("whitespace_score", np.nan),
            "rule_score": sc.get("rule_score", np.nan),
            "outlier_score": sc.get("outlier_score", np.nan),
            "duplicate_rows_score": np.nan,   # Dataset-level metric: strictly NaN at column level
            "stability_score": sc.get("stability_score", np.nan),
            "overall_column_score": sc.get("overall_column_score", 100.0),
        })
    scoring_matrix_df = pd.DataFrame(scoring_rows)
    if not scoring_matrix_df.empty:
        scoring_matrix_df.set_index("column_name", inplace=True)

    # 2. Dimension Summary DataFrame with Hare-Niemeyer display reconciliation
    disp_weights: Dict[str, float] = {}
    disp_contribs: Dict[str, float] = {}

    if applicable_dims and total_applicable_configured_w > 0:
        # Reconcile 4-decimal displayed effective weights so sum strictly equals 1.0000
        w_records = []
        for ds in applicable_dims:
            w_raw = ds.effective_weight
            w_floor = math.floor(w_raw * 10000)
            w_rem = (w_raw * 10000) - w_floor
            w_records.append({"dim": ds.dimension.value, "floor": w_floor, "rem": w_rem})
        w_diff = 10000 - sum(r["floor"] for r in w_records)
        w_records.sort(key=lambda r: r["rem"], reverse=True)
        for i in range(min(max(0, w_diff), len(w_records))):
            w_records[i]["floor"] += 1
        disp_weights = {r["dim"]: round(r["floor"] / 10000.0, 4) for r in w_records}

        # Reconcile 2-decimal displayed contributions so sum strictly equals overall_score
        target_cents = round(overall_score * 100)
        c_records = []
        for ds in applicable_dims:
            c_raw = ds.score * ds.effective_weight
            c_raw_cents = c_raw * 100.0
            c_floor = math.floor(c_raw_cents)
            c_rem = c_raw_cents - c_floor
            c_records.append({"dim": ds.dimension.value, "floor": c_floor, "rem": c_rem})
        c_diff = target_cents - sum(r["floor"] for r in c_records)
        c_records.sort(key=lambda r: r["rem"], reverse=True)
        for i in range(min(max(0, c_diff), len(c_records))):
            c_records[i]["floor"] += 1
        disp_contribs = {r["dim"]: round(r["floor"] / 100.0, 2) for r in c_records}

    dim_rows = []
    for dim_enum in QualityDimension:
        ds = dimension_scores[dim_enum.value]
        conf_w = ds.configured_weight
        eff_w = disp_weights.get(dim_enum.value, 0.0) if ds.applicable else 0.0
        contrib = disp_contribs.get(dim_enum.value, 0.0) if ds.applicable else 0.0
        dim_rows.append({
            "dimension": dim_enum.value,
            "score": round(ds.score, 2),
            "configured_weight": round(conf_w, 4),
            "effective_weight": round(eff_w, 4),
            "applicable": ds.applicable,
            "reason": ds.reason,
            "weighted_contribution": round(contrib, 2),
        })
    dimension_summary_df = pd.DataFrame(dim_rows)
    if not dimension_summary_df.empty:
        dimension_summary_df.set_index("dimension", inplace=True)

    # 3. Mathematical Scoring Audit Trail
    trail_lines = [
        "=========================================================================================",
        "CLEANALYTIX 6-DIMENSION DATA QUALITY SCORING AUDIT TRAIL",
        "=========================================================================================",
        f"Applicability Re-weighting Policy: Exclude non-applicable dimensions; re-normalize configured weights to 1.0.",
        f"Active Score Mode: {score_mode}",
        "-----------------------------------------------------------------------------------------",
    ]
    for r in dim_rows:
        dim_name = str(r["dimension"]).capitalize()
        if r["applicable"]:
            trail_lines.append(
                f"  [APPLICABLE]     {dim_name:<14} : Score = {r['score']:>6.2f} / 100.00 | Configured Weight = {r['configured_weight']:.4f} | Effective Weight = {r['effective_weight']:.4f} -> Contribution = {r['weighted_contribution']:>6.2f}"
            )
        else:
            trail_lines.append(
                f"  [NON-APPLICABLE] {dim_name:<14} : [EXCLUDED FROM SCORE] (Configured Weight = {r['configured_weight']:.4f}) Reason: {r['reason']}"
            )
    trail_lines.append("-----------------------------------------------------------------------------------------")
    tot_eff_w = sum(r["effective_weight"] for r in dim_rows if r["applicable"])
    trail_lines.append(
        f"Total Applicable Weight: {tot_eff_w:.4f} | OVERALL DATASET QUALITY SCORE: {overall_score:.2f} / 100.00"
    )
    trail_lines.append("=========================================================================================")
    scoring_audit_trail_str = "\n".join(trail_lines)

    # 4. Metric Inventory DataFrame
    metric_inventory_df = build_metric_inventory(
        df=df,
        meta_df=meta_df,
        dimension_scores=dimension_scores,
        thresholds=thresholds,
        metric_weights=metric_weights,
        has_baseline=has_baseline,
        config=config,
    )

    return (
        overall_score,
        dimension_scores,
        final_col_scores,
        scoring_matrix_df,
        dimension_summary_df,
        metric_inventory_df,
        scoring_audit_trail_str,
    )


def build_metric_inventory(
    df: pd.DataFrame,
    meta_df: pd.DataFrame,
    dimension_scores: Dict[str, DimensionScore],
    thresholds: Dict[str, float],
    metric_weights: Dict[str, float],
    has_baseline: bool,
    config: CleanalytixConfig,
) -> pd.DataFrame:
    """Build an exhaustive audit ledger of every evaluated metric across dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Analyzed dataset.
    meta_df : pd.DataFrame
        Metadata table.
    dimension_scores : dict
        DimensionScore objects.
    thresholds : dict
        Metric thresholds.
    metric_weights : dict
        Metric-level weights within dimensions.
    has_baseline : bool
        Whether baseline comparison was conducted.
    config : CleanalytixConfig
        Configuration options.

    Returns
    -------
    pd.DataFrame
        Metric inventory ledger.
    """
    rows = []
    comp_ds = dimension_scores.get(QualityDimension.COMPLETENESS.value)
    val_ds = dimension_scores.get(QualityDimension.VALIDITY.value)
    uniq_ds = dimension_scores.get(QualityDimension.UNIQUENESS.value)
    cons_ds = dimension_scores.get(QualityDimension.CONSISTENCY.value)
    time_ds = dimension_scores.get(QualityDimension.TIMELINESS.value)
    stab_ds = dimension_scores.get(QualityDimension.STABILITY.value)

    # 1. COMPLETENESS METRICS
    # Dataset-level row completeness
    incomp_pct = comp_ds.metrics.get("incomplete_rows_percent") if comp_ds else 0.0
    row_score = comp_ds.metric_scores.get("row_completeness_score") if comp_ds else 100.0
    r_thresh = thresholds.get("missing_percent", 20.0)
    w_row = metric_weights.get("completeness_row", 0.30)
    eff_w_comp = comp_ds.effective_weight if comp_ds and comp_ds.applicable else 0.0
    rows.append({
        "dimension": QualityDimension.COMPLETENESS.value,
        "metric_name": "completeness_row",
        "domain": "completeness",
        "scope": "dataset",
        "target_column": None,
        "raw_value": round(float(incomp_pct), 2) if incomp_pct is not None else 0.0,
        "unit": "%",
        "threshold": r_thresh,
        "status": "PASS" if (incomp_pct or 0) <= r_thresh else "WARNING",
        "metric_score": row_score,
        "contributes_to_score": True,
        "effective_weight": round(w_row * eff_w_comp, 4),
        "calculation_method": "Percentage of rows containing at least 1 null",
        "reason": None,
    })

    # Column-level cell missingness
    w_cell = metric_weights.get("completeness_cell", 0.70)
    for _, meta_row in meta_df.iterrows():
        col = meta_row.get("column_name")
        m_pct = meta_row.get("missing_percent", 0.0)
        m_score = calculate_metric_score(float(m_pct or 0), r_thresh, score_mode=config.score_mode) * 100.0
        rows.append({
            "dimension": QualityDimension.COMPLETENESS.value,
            "metric_name": "completeness_cell",
            "domain": "completeness",
            "scope": "column",
            "target_column": col,
            "raw_value": round(float(m_pct or 0), 2),
            "unit": "%",
            "threshold": r_thresh,
            "status": "PASS" if (m_pct or 0) <= r_thresh else ("CRITICAL" if (m_pct or 0) > 2 * r_thresh else "WARNING"),
            "metric_score": round(m_score, 2),
            "contributes_to_score": True,
            "effective_weight": round((w_cell / max(1, len(meta_df))) * eff_w_comp, 4),
            "calculation_method": "Column cell null percentage",
            "reason": None,
        })

    # 2. VALIDITY METRICS
    eff_w_val = val_ds.effective_weight if val_ds and val_ds.applicable else 0.0
    t_thresh = thresholds.get("type_errors_percent", 2.0)
    ws_thresh = thresholds.get("whitespace_percent", 5.0)
    rule_thresh = thresholds.get("rule_errors_percent", 2.0)
    out_thresh = thresholds.get("outlier_percent", 10.0)

    for _, meta_row in meta_df.iterrows():
        col = meta_row.get("column_name")
        role = str(meta_row.get("column_role", "unknown"))
        non_null = int(meta_row.get("non_null_count", 0))

        # Type errors
        t_err = meta_row.get("type_errors", 0)
        t_pct = (float(t_err or 0) / max(1, non_null)) * 100.0
        t_score = calculate_metric_score(t_pct, t_thresh, score_mode=config.score_mode) * 100.0
        rows.append({
            "dimension": QualityDimension.VALIDITY.value,
            "metric_name": "validity_type_errors",
            "domain": "validity",
            "scope": "column",
            "target_column": col,
            "raw_value": round(t_pct, 2),
            "unit": "%",
            "threshold": t_thresh,
            "status": "PASS" if t_pct <= t_thresh else "WARNING",
            "metric_score": round(t_score, 2),
            "contributes_to_score": True,
            "effective_weight": None,
            "calculation_method": "Semantic type conformance errors",
            "reason": None,
        })

        # Whitespace
        if role in {"categorical", "free_text", "identifier"}:
            ws_err = meta_row.get("whitespace_issues", 0)
            ws_pct = (float(ws_err or 0) / max(1, non_null)) * 100.0
            w_score = calculate_metric_score(ws_pct, ws_thresh, score_mode=config.score_mode) * 100.0
            rows.append({
                "dimension": QualityDimension.VALIDITY.value,
                "metric_name": "validity_whitespace",
                "domain": "validity",
                "scope": "column",
                "target_column": col,
                "raw_value": round(ws_pct, 2),
                "unit": "%",
                "threshold": ws_thresh,
                "status": "PASS" if ws_pct <= ws_thresh else "WARNING",
                "metric_score": round(w_score, 2),
                "contributes_to_score": True,
                "effective_weight": None,
                "calculation_method": "Untrimmed leading/trailing whitespace",
                "reason": None,
            })

        # Outliers
        if role in {"continuous", "discrete"}:
            out_cnt = meta_row.get("outlier_count", 0)
            out_pct = (float(out_cnt or 0) / max(1, non_null)) * 100.0
            o_score = calculate_metric_score(out_pct, out_thresh, score_mode=config.score_mode) * 100.0
            rows.append({
                "dimension": QualityDimension.VALIDITY.value,
                "metric_name": "validity_outliers",
                "domain": "validity",
                "scope": "column",
                "target_column": col,
                "raw_value": round(out_pct, 2),
                "unit": "%",
                "threshold": out_thresh,
                "status": "PASS" if out_pct <= out_thresh else "WARNING",
                "metric_score": round(o_score, 2),
                "contributes_to_score": True,
                "effective_weight": None,
                "calculation_method": "Heuristic statistical outlier containment",
                "reason": None,
            })

    # 3. UNIQUENESS METRICS
    eff_w_uniq = uniq_ds.effective_weight if uniq_ds and uniq_ds.applicable else 0.0
    dup_row_pct = uniq_ds.metrics.get("duplicate_rows_percent", 0.0) if uniq_ds else 0.0
    dup_thresh = thresholds.get("duplicate_rows_percent", 1.0)
    rows.append({
        "dimension": QualityDimension.UNIQUENESS.value,
        "metric_name": "uniqueness_duplicate_rows",
        "domain": "uniqueness",
        "scope": "dataset",
        "target_column": None,
        "raw_value": round(float(dup_row_pct), 2),
        "unit": "%",
        "threshold": dup_thresh,
        "status": "PASS" if dup_row_pct <= dup_thresh else "WARNING",
        "metric_score": uniq_ds.metric_scores.get("duplicate_rows_score", 100.0) if uniq_ds else 100.0,
        "contributes_to_score": True,
        "effective_weight": round(eff_w_uniq * 0.50, 4),
        "calculation_method": "Duplicate rows percentage",
        "reason": None,
    })

    # 4. CONSISTENCY METRICS
    eff_w_cons = cons_ds.effective_weight if cons_ds and cons_ds.applicable else 0.0
    if cons_ds and cons_ds.applicable:
        rows.append({
            "dimension": QualityDimension.CONSISTENCY.value,
            "metric_name": "consistency_rules",
            "domain": "consistency",
            "scope": "dataset",
            "target_column": None,
            "raw_value": cons_ds.score,
            "unit": "score [0-100]",
            "threshold": rule_thresh,
            "status": "PASS" if cons_ds.score >= 90.0 else "WARNING",
            "metric_score": cons_ds.score,
            "contributes_to_score": True,
            "effective_weight": round(eff_w_cons, 4),
            "calculation_method": "Cross-column relational business rule evaluation",
            "reason": None,
        })
    else:
        rows.append({
            "dimension": QualityDimension.CONSISTENCY.value,
            "metric_name": "consistency_rules",
            "domain": "consistency",
            "scope": "dataset",
            "target_column": None,
            "raw_value": None,
            "unit": "%",
            "threshold": rule_thresh,
            "status": "NOT_APPLICABLE",
            "metric_score": None,
            "contributes_to_score": False,
            "effective_weight": 0.0,
            "calculation_method": "Cross-column relational business rule evaluation",
            "reason": cons_ds.reason if cons_ds else "No consistency rules configured",
        })

    # 5. TIMELINESS METRICS
    eff_w_time = time_ds.effective_weight if time_ds and time_ds.applicable else 0.0
    if time_ds and time_ds.applicable:
        fresh_days = time_ds.metrics.get("freshness_days")
        fut_pct = time_ds.metrics.get("future_dates_percent", 0.0)
        rows.append({
            "dimension": QualityDimension.TIMELINESS.value,
            "metric_name": "timeliness_freshness",
            "domain": "timeliness",
            "scope": "column",
            "target_column": None,
            "raw_value": round(float(fresh_days), 2) if fresh_days is not None else None,
            "unit": "days",
            "threshold": thresholds.get("freshness_days", 30.0),
            "status": "PASS" if (fresh_days or 0) <= thresholds.get("freshness_days", 30.0) else "WARNING",
            "metric_score": time_ds.metric_scores.get("freshness_score", 100.0),
            "contributes_to_score": True,
            "effective_weight": round(eff_w_time * 0.50, 4),
            "calculation_method": "Elapsed days between max timestamp and reference time",
            "reason": None,
        })
    else:
        rows.append({
            "dimension": QualityDimension.TIMELINESS.value,
            "metric_name": "timeliness_freshness",
            "domain": "timeliness",
            "scope": "column",
            "target_column": None,
            "raw_value": None,
            "unit": "days",
            "threshold": thresholds.get("freshness_days", 30.0),
            "status": "NOT_APPLICABLE",
            "metric_score": None,
            "contributes_to_score": False,
            "effective_weight": 0.0,
            "calculation_method": "Elapsed days between max timestamp and reference time",
            "reason": time_ds.reason if time_ds else "No datetime columns detected",
        })

    # 6. STABILITY METRICS
    eff_w_stab = stab_ds.effective_weight if stab_ds and stab_ds.applicable else 0.0
    if stab_ds and stab_ds.applicable:
        for _, meta_row in meta_df.iterrows():
            col = meta_row.get("column_name")
            psi_val = meta_row.get("psi_divergence")
            if pd.notna(psi_val):
                p_thresh = thresholds.get("psi_divergence", 0.25)
                p_score = calculate_metric_score(float(psi_val), p_thresh, score_mode=config.score_mode, k=3.0) * 100.0
                rows.append({
                    "dimension": QualityDimension.STABILITY.value,
                    "metric_name": "stability_psi",
                    "domain": "drift_analysis",
                    "scope": "column",
                    "target_column": col,
                    "raw_value": round(float(psi_val), 4),
                    "unit": "PSI",
                    "threshold": p_thresh,
                    "status": "PASS" if float(psi_val) < 0.10 else ("WARNING" if float(psi_val) < p_thresh else "CRITICAL"),
                    "metric_score": round(p_score, 2),
                    "contributes_to_score": True,
                    "effective_weight": round(eff_w_stab * 0.50 / max(1, len(meta_df)), 4),
                    "calculation_method": "Population Stability Index against reference baseline",
                    "reason": None,
                })
    else:
        rows.append({
            "dimension": QualityDimension.STABILITY.value,
            "metric_name": "stability_psi",
            "domain": "drift_analysis",
            "scope": "column",
            "target_column": None,
            "raw_value": None,
            "unit": "PSI",
            "threshold": thresholds.get("psi_divergence", 0.25),
            "status": "NOT_APPLICABLE",
            "metric_score": None,
            "contributes_to_score": False,
            "effective_weight": 0.0,
            "calculation_method": "Population Stability Index against reference baseline",
            "reason": stab_ds.reason if stab_ds else "No baseline reference dataset provided",
        })

    inv_df = pd.DataFrame(rows)
    return inv_df

