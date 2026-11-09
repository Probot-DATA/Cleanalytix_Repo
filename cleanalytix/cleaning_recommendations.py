from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union, Sequence
import pandas as pd
import numpy as np

from cleanalytix.types import (
    QualityDimension,
    RecommendationItem,
    InsightItem,
    InsightSeverity,
    ProfileContext,
)


def _profile_context_to_rec_rows(
    ctx_or_list: Union[ProfileContext, Sequence[ProfileContext]]
) -> List[Dict[str, Any]]:
    """Extract per-column recommendation attributes directly from ProfileContext objects."""
    contexts = [ctx_or_list] if isinstance(ctx_or_list, ProfileContext) else list(ctx_or_list)
    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        dataset_name = ctx.dataset.dataset_name
        for col in ctx.column_profiles:
            role = str(col.role)
            dtype_obj = col.dtype_object if col.dtype_object is not None else col.dtype
            comp = col.completeness
            uniq = col.uniqueness
            val = col.validity
            stat = col.statistical
            dist = col.distribution
            temp = col.temporal

            non_null_count = int(comp.non_null_count) if comp is not None else 0
            missing_count = int(comp.null_count) if comp is not None else 0
            missing_percent = float(comp.null_percentage) if comp is not None else 0.0
            unique_count = int(uniq.distinct_count) if uniq is not None else 0
            type_errors = int(val.type_error_count) if val is not None else 0
            rule_errors = int(val.rule_error_count) if val is not None else 0

            is_num_branch = (
                role in {"continuous", "discrete"}
                and stat is not None
                and pd.api.types.is_numeric_dtype(dtype_obj)
                and not pd.api.types.is_bool_dtype(dtype_obj)
            )
            is_dt_branch = (
                role == "datetime"
                or pd.api.types.is_datetime64_any_dtype(dtype_obj)
            )

            skewness_level = None
            outliers_val = None
            rare_category_percent = None
            whitespace_issues = None
            date_range_days = None

            if is_num_branch:
                if non_null_count > 0 and stat is not None:
                    if stat.skewness is not None:
                        abs_s = abs(stat.skewness)
                        skewness_level = (
                            "low" if abs_s < 0.5 else ("moderate" if abs_s < 1.0 else "high")
                        )
                    outliers_val = int(stat.outlier_count)
            elif role in {"binary", "identifier"}:
                outliers_val = 0
                rare_category_percent = (
                    float(dist.rare_categories_percentage) if dist is not None else 0.0
                )
                whitespace_issues = int(val.whitespace_count) if val is not None else 0
            elif is_dt_branch:
                if temp is not None:
                    date_range_days = temp.date_range_days
            else:
                rare_category_percent = (
                    float(dist.rare_categories_percentage) if dist is not None else 0.0
                )
                whitespace_issues = int(val.whitespace_count) if val is not None else 0

            rows.append({
                "dataset_name": dataset_name,
                "column_name": col.name,
                "column_role": role,
                "dtype": dtype_obj,
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_percent": missing_percent,
                "unique_count": unique_count,
                "skewness_level": skewness_level,
                "outlier_count": outliers_val,
                "rare_category_percent": rare_category_percent,
                "whitespace_issues": whitespace_issues,
                "date_range_days": date_range_days,
                "type_errors": type_errors,
                "rule_errors": rule_errors,
            })
    return rows


def cleaning_recommendations(
    meta_data_combined: Union[ProfileContext, Sequence[ProfileContext], pd.DataFrame]
) -> pd.DataFrame:
    """Convert metadata rows into cleaning recommendations.

    Parameters
    ----------
    meta_data_combined : pandas.DataFrame
        Metadata table returned by :func:`generate_meta`. Required columns
        include ``dataset_name``, ``column_name``, ``dtype``,
        ``missing_percent``, ``type_errors``, ``whitespace_issues``,
        ``rule_errors``, ``outlier_count``, ``rare_category_percent``,
        ``date_range_days``, ``unique_count``, and ``skewness_level``.
        ``column_role`` is used when present to avoid treating identifiers and
        binary flags as continuous numeric columns.

    Returns
    -------
    pandas.DataFrame
        Recommendation table with one row per metadata row and the following
        columns:

        ``dataset_name``
            Dataset label.
        ``column_name``
            Column receiving recommendations.
        ``rule_ids``
            Ordered list of integer rule IDs. Current IDs are:
            ``1`` excessive missing values, ``2`` missing values, ``3``
            outliers, ``4`` whitespace issues, ``5`` rare categories, ``6``
            collapsed or invalid date range, ``7`` type errors, ``8`` business
            rule violations, ``9`` high skewness, and ``10`` no issues
            detected.
        ``inferred_type``
            Recommendation-time type bucket: ``numeric``, ``datetime``, or
            ``categorical``.
        ``issue_names``
            Human-readable names aligned with ``rule_ids``.
        ``recommended_fix_order``
            Numeric sort keys showing the order in which fixes should be
            attempted.

    Raises
    ------
    KeyError
        If required metadata columns are missing.

    Notes
    -----
    Recommendations are deterministic and rule-based. They are intended to be
    reviewable evidence for cleaning decisions, not a hidden model.

    Examples
    --------
    >>> meta = generate_meta(["customers"], [customers_df])
    >>> recs = cleaning_recommendations(meta)
    >>> recs[["column_name", "rule_ids", "issue_names"]]

    See Also
    --------
    generate_meta : Creates the metadata consumed by this function.
    get_cleaned_data : Applies recommendations to DataFrames.
    Run_DQ_Pipeline : Calls this function in the end-to-end workflow.
    """

    recs = []

    def infer_type(row):
        # Prefer explicit column role when available
        has_role = (
            ("column_role" in row and pd.notna(row.get("column_role")))
            if isinstance(row, dict)
            else ("column_role" in row.index and pd.notna(row.get("column_role")))
        )
        if has_role:
            role = str(row["column_role"]).lower()
            if role == "continuous":
                return "numeric"
            if role == "datetime":
                return "datetime"
            if role in {"binary", "identifier", "categorical"}:
                return "categorical"

        # Fall back to dtype string with conservative heuristics for mixed/object types.
        dtype = str(row.get("dtype", "")).lower()

        # Numeric dtype explicit
        if any(x in dtype for x in ["int", "float", "number"]):
            return "numeric"

        # Explicit datetime dtype
        if "datetime" in dtype:
            return "datetime"

        # Heuristic: object/string columns that look mostly numeric (few type errors)
        # should be treated as numeric for recommendation purposes.
        try:
            non_null = int(row.get("non_null_count") or 0)
            type_errors = int(row.get("type_errors") or 0)
        except Exception:
            non_null = 0
            type_errors = 0

        if dtype in {"object", "string", "mixed"}:
            # If we have non-null counts and a small fraction of type errors,
            # consider the column numeric-like.
            if non_null > 0 and type_errors / non_null < 0.05:
                return "numeric"

            # If a date-range was detected in profiling, prefer datetime
            if pd.notna(row.get("date_range_days")):
                return "datetime"

            return "categorical"

        # Default conservative fallback
        return "categorical"

    RULE_PRIORITY = {
        7: 1,   # type fixing FIRST
        4: 2,   # whitespace
        2: 3,   # missing
        3: 4,   # outliers
        5: 5,   # categorical cleanup
        9: 6    # transformations LAST
    }

    # NEW: Add human-readable metadata
    RULE_INFO = {
        1: "Excessive Missing Values",
        2: "Missing Values",
        3: "Outliers",
        4: "Whitespace Issues",
        5: "Rare Categories",
        6: "Collapsed Date Range",
        7: "Type Errors",
        8: "Business Rule Violations",
        9: "High Skewness",
        10: "No Issues Detected"
    }

    if isinstance(meta_data_combined, ProfileContext) or (
        isinstance(meta_data_combined, Sequence) and not isinstance(meta_data_combined, pd.DataFrame)
    ):
        row_iter = _profile_context_to_rec_rows(meta_data_combined)
    else:
        row_iter = [row for _, row in meta_data_combined.iterrows()]

    for row in row_iter:
        suggestions = []
        col_type = infer_type(row)

        if row['missing_percent'] > 40:
            suggestions.append(1)

        else:
            if pd.notna(row['type_errors']) and row['type_errors'] > 0:
                suggestions.append(7)

            if col_type == "categorical":
                if pd.notna(row['whitespace_issues']) and row['whitespace_issues'] > 0:
                    suggestions.append(4)

            if pd.notna(row['rule_errors']) and row['rule_errors'] > 0:
                suggestions.append(8)

            if 0 < row['missing_percent']:
                suggestions.append(2)

            if col_type == "numeric":
                if pd.notna(row['outlier_count']) and row['outlier_count'] > 0:
                    suggestions.append(3)

            if col_type == "categorical":
                if pd.notna(row['rare_category_percent']) and row['rare_category_percent'] > 0:
                    suggestions.append(5)

            if col_type == "datetime":
                if pd.notna(row['date_range_days']) and row['unique_count'] <= 1:
                    suggestions.append(6)

            if col_type == "numeric":
                if row['skewness_level'] == 'high':
                    suggestions.append(9)

            if not suggestions:
                suggestions.append(10)

        suggestions = list(set(suggestions))
        suggestions = sorted(
            suggestions,
            key=lambda x: RULE_PRIORITY.get(x, 100)
        )

        recs.append({
            "dataset_name": row["dataset_name"],
            "column_name": row["column_name"],

            "rule_ids": suggestions,
            "inferred_type": col_type,

            "issue_names": [
                RULE_INFO.get(rule_id, "Unknown Issue")
                for rule_id in suggestions
            ],

            "recommended_fix_order": [
                RULE_PRIORITY.get(rule_id, 100)
                for rule_id in suggestions
            ]
        })

    return pd.DataFrame(recs)


def normalize_severity(severity: Any) -> str:
    """Normalize severity to uppercase standard string (CRITICAL, WARNING, INFO, OPTIMIZATION)."""
    if isinstance(severity, InsightSeverity):
        return severity.name
    if hasattr(severity, "value"):
        return str(severity.value).strip().upper()
    return str(severity).strip().upper()


def generate_recommendations(
    meta_df: Optional[Union[ProfileContext, Sequence[ProfileContext], pd.DataFrame]],
    insights: Optional[List[InsightItem]] = None,
) -> Tuple[List[RecommendationItem], pd.DataFrame]:
    """Generate structured, actionable recommendations across all detected DQ issues.

    Parameters
    ----------
    meta_df : ProfileContext or pd.DataFrame
        ``ProfileContext`` (v2.0 native) or profiling metadata from ``generate_meta()``.
    insights : list of InsightItem, optional
        Pre-computed dataset and column insights.

    Returns
    -------
    tuple of (list of RecommendationItem, pd.DataFrame)
        Structured recommendation items and tabular recommendations DataFrame.
    """
    raw_items: List[Dict[str, Any]] = []

    is_context_input = isinstance(meta_df, ProfileContext) or (
        isinstance(meta_df, Sequence) and not isinstance(meta_df, pd.DataFrame)
    )
    if is_context_input:
        col_rec_rows = _profile_context_to_rec_rows(meta_df)  # type: ignore[arg-type]
        has_meta_rows = len(col_rec_rows) > 0
    else:
        has_meta_rows = meta_df is not None and not meta_df.empty
        col_rec_rows = [row for _, row in meta_df.iterrows()] if has_meta_rows else []  # type: ignore[union-attr]

    # 1. Dataset-level and multidimensional items from insights
    if insights:
        for ins in insights:
            sev_str = normalize_severity(ins.severity)
            if sev_str in {"CRITICAL", "WARNING"}:
                # Avoid duplicating column-level completeness/validity recommendations already generated from meta_df
                if has_meta_rows:
                    if ins.dimension in {QualityDimension.COMPLETENESS, QualityDimension.VALIDITY} and ins.column is not None:
                        continue

                # Map insight metric to rule_id if applicable
                r_id = 8 if "rule" in ins.metric_name or "consistency" in ins.metric_name else (
                    11 if "duplicate" in ins.metric_name else (
                        6 if "date" in ins.metric_name or "freshness" in ins.metric_name else (
                            12 if "psi" in ins.metric_name or "drift" in ins.metric_name else 0
                        )
                    )
                )
                raw_items.append({
                    "rule_id": r_id,
                    "dimension": ins.dimension,
                    "severity": sev_str,
                    "affected_column": ins.column,
                    "issue_name": ins.message.split(":")[0] if ":" in ins.message else ins.metric_name.replace("_", " ").title(),
                    "evidence": ins.message,
                    "rationale": f"Quality anomaly violates {ins.dimension.value if hasattr(ins.dimension, 'value') else ins.dimension} standards.",
                    "suggested_action": ins.recommendation,
                    "details": ins.details or {"metric_name": ins.metric_name, "metric_value": ins.metric_value, "threshold": ins.threshold},
                })

    # 2. Column-level items from meta_df / ProfileContext
    if has_meta_rows:
        for row in col_rec_rows:
            col = str(row.get("column_name", ""))
            role = str(row.get("column_role", "")).lower()
            dtype = str(row.get("dtype", "")).lower()
            n_non_null = int(row.get("non_null_count", 0) or 0)
            n_missing = int(row.get("missing_count", 0) or 0)
            n_total = max(1, n_non_null + n_missing)

            # Rule 7: Type errors
            type_err = row.get("type_errors", 0)
            if pd.notna(type_err) and int(type_err) > 0:
                t_cnt = int(type_err)
                t_pct = (t_cnt / max(1, n_non_null)) * 100.0 if n_non_null > 0 else 0.0
                raw_items.append({
                    "rule_id": 7,
                    "dimension": QualityDimension.VALIDITY,
                    "severity": "CRITICAL" if t_pct >= 5.0 else "WARNING",
                    "affected_column": col,
                    "issue_name": "Type Errors",
                    "evidence": f"{t_cnt} values ({t_pct:.2f}%) violate inferred column type.",
                    "rationale": "Mixed types cause silent downstream computation failures and schema rejection.",
                    "suggested_action": f"Coerce '{col}' to the expected type using pd.to_numeric() or pd.to_datetime(errors='coerce').",
                    "details": {"type_errors": t_cnt, "error_percent": t_pct},
                })

            # Rule 4: Whitespace issues
            ws_err = row.get("whitespace_issues", 0)
            if pd.notna(ws_err) and int(ws_err) > 0:
                w_cnt = int(ws_err)
                w_pct = (w_cnt / max(1, n_non_null)) * 100.0 if n_non_null > 0 else 0.0
                raw_items.append({
                    "rule_id": 4,
                    "dimension": QualityDimension.VALIDITY,
                    "severity": "INFO",
                    "affected_column": col,
                    "issue_name": "Whitespace Issues",
                    "evidence": f"{w_cnt} values ({w_pct:.2f}%) have unstripped leading/trailing whitespace.",
                    "rationale": "Surrounding whitespace creates spurious distinct categories and join mismatches.",
                    "suggested_action": f"Apply df['{col}'] = df['{col}'].astype(str).str.strip() to normalize categories.",
                    "details": {"whitespace_issues": w_cnt, "whitespace_percent": w_pct},
                })

            # Rule 1: Excessive Missing Values
            miss_pct = float(row.get("missing_percent", 0.0) or 0.0)
            if miss_pct > 40.0:
                raw_items.append({
                    "rule_id": 1,
                    "dimension": QualityDimension.COMPLETENESS,
                    "severity": "CRITICAL",
                    "affected_column": col,
                    "issue_name": "Excessive Missing Values",
                    "evidence": f"{miss_pct:.2f}% null values ({n_missing} missing records).",
                    "rationale": "Severe missingness (>40%) limits feature utility and may degrade analytical validity.",
                    "suggested_action": f"Evaluate whether '{col}' should be dropped, retained with an explicit missing indicator, or collected upstream.",
                    "details": {"missing_percent": miss_pct, "missing_count": n_missing},
                })
            elif miss_pct > 0.0:
                # Rule 2: Missing Values
                raw_items.append({
                    "rule_id": 2,
                    "dimension": QualityDimension.COMPLETENESS,
                    "severity": "WARNING" if miss_pct >= 10.0 else "INFO",
                    "affected_column": col,
                    "issue_name": "Missing Values",
                    "evidence": f"{miss_pct:.2f}% null values ({n_missing} missing records).",
                    "rationale": "Missing values reduce effective sample size and introduce estimation bias.",
                    "suggested_action": f"Impute '{col}' using domain-appropriate strategy (median for continuous, mode for categorical).",
                    "details": {"missing_percent": miss_pct, "missing_count": n_missing},
                })

            # Rule 3: Outliers
            outliers = row.get("outlier_count", 0)
            if pd.notna(outliers) and int(outliers) > 0 and role in {"continuous", "discrete", "numeric"}:
                o_cnt = int(outliers)
                o_pct = (o_cnt / max(1, n_non_null)) * 100.0
                raw_items.append({
                    "rule_id": 3,
                    "dimension": QualityDimension.VALIDITY,
                    "severity": "WARNING" if o_pct >= 5.0 else "INFO",
                    "affected_column": col,
                    "issue_name": "Outliers",
                    "evidence": f"{o_cnt} statistical outliers ({o_pct:.2f}% of non-null records).",
                    "rationale": "Extreme observations distort parametric statistics (mean, variance) and linear models.",
                    "suggested_action": f"Verify whether outliers in '{col}' are valid extremes or errors; apply winsorization or robust scaling.",
                    "details": {"outlier_count": o_cnt, "outlier_percent": o_pct},
                })

            # Rule 5: Rare Categories
            rare_pct = float(row.get("rare_category_percent", 0.0) or 0.0)
            if rare_pct > 0.0 and role in {"categorical", "binary"}:
                raw_items.append({
                    "rule_id": 5,
                    "dimension": QualityDimension.VALIDITY,
                    "severity": "INFO",
                    "affected_column": col,
                    "issue_name": "Rare Categories",
                    "evidence": f"Infrequent categories (<1% frequency) constitute {rare_pct:.2f}% of entries.",
                    "rationale": "High-cardinality long tails can lead to model overfitting and cross-validation sparsity.",
                    "suggested_action": f"Group rare categories in '{col}' into an 'Other' bucket prior to modeling.",
                    "details": {"rare_category_percent": rare_pct},
                })

            # Rule 6: Collapsed Date Range
            if "datetime" in dtype or role == "datetime":
                u_cnt = int(row.get("unique_count", 0) or 0)
                d_days = row.get("date_range_days", 0)
                if u_cnt <= 1 or (pd.notna(d_days) and float(d_days) == 0.0):
                    raw_items.append({
                        "rule_id": 6,
                        "dimension": QualityDimension.TIMELINESS,
                        "severity": "WARNING",
                        "affected_column": col,
                        "issue_name": "Collapsed Date Range",
                        "evidence": f"Date span is {d_days} days with {u_cnt} unique timestamp(s).",
                        "rationale": "A single constant timestamp precludes temporal ordering, trend detection, and time-based splitting.",
                        "suggested_action": f"Inspect timestamp ingestion for '{col}' to verify date parsing and time-zone configuration.",
                        "details": {"unique_count": u_cnt, "date_range_days": d_days},
                    })

            # Rule 8: Rule errors
            r_err = row.get("rule_errors", 0)
            if pd.notna(r_err) and int(r_err) > 0:
                re_cnt = int(r_err)
                re_pct = (re_cnt / n_total) * 100.0
                raw_items.append({
                    "rule_id": 8,
                    "dimension": QualityDimension.VALIDITY,
                    "severity": "CRITICAL" if re_pct >= 5.0 else "WARNING",
                    "affected_column": col,
                    "issue_name": "Business Rule Violations",
                    "evidence": f"{re_cnt} values ({re_pct:.2f}%) violate custom domain rules.",
                    "rationale": "Records violating domain invariants indicate pipeline or upstream source data corruption.",
                    "suggested_action": f"Audit invalid records in '{col}' and apply domain-specified remediation.",
                    "details": {"rule_errors": re_cnt, "rule_error_percent": re_pct},
                })

            # Rule 9: High Skewness
            if str(row.get("skewness_level", "")).lower() == "high":
                raw_items.append({
                    "rule_id": 9,
                    "dimension": QualityDimension.VALIDITY,
                    "severity": "OPTIMIZATION",
                    "affected_column": col,
                    "issue_name": "High Skewness",
                    "evidence": "Distribution is heavily skewed (|skewness| > 1.0).",
                    "rationale": "High skewness violates normality assumptions in regression and variance-based algorithms.",
                    "suggested_action": f"Consider transforming '{col}' with np.log1p() or PowerTransformer.",
                    "details": {"skewness_level": "high"},
                })

    # Deduplicate items by (rule_id, affected_column)
    seen = set()
    unique_items = []
    for it in raw_items:
        key = (it["rule_id"], it["affected_column"])
        if key not in seen:
            seen.add(key)
            unique_items.append(it)

    # Sort by severity priority: CRITICAL (0), WARNING (1), INFO (2), OPTIMIZATION (3)
    sev_rank = {"CRITICAL": 0, "WARNING": 1, "INFO": 2, "OPTIMIZATION": 3}
    unique_items.sort(key=lambda x: sev_rank.get(x["severity"], 99))

    # Construct final RecommendationItem list and DataFrame with 1-based priority_rank
    items: List[RecommendationItem] = []
    table_rows: List[Dict[str, Any]] = []

    for rank, it in enumerate(unique_items, start=1):
        rec_obj = RecommendationItem(
            priority_rank=rank,
            rule_id=it["rule_id"],
            dimension=it["dimension"],
            severity=it["severity"],
            affected_column=it["affected_column"],
            issue_name=it["issue_name"],
            evidence=it["evidence"],
            rationale=it["rationale"],
            suggested_action=it["suggested_action"],
            details=it.get("details", {}),
        )
        items.append(rec_obj)
        table_rows.append({
            "priority_rank": rank,
            "rule_id": it["rule_id"],
            "dimension": it["dimension"].value if hasattr(it["dimension"], "value") else str(it["dimension"]),
            "severity": it["severity"],
            "affected_column": it["affected_column"],
            "column_name": it["affected_column"],
            "issue_name": it["issue_name"],
            "evidence": it["evidence"],
            "rationale": it["rationale"],
            "suggested_action": it["suggested_action"],
        })

    if not table_rows:
        table_df = pd.DataFrame(columns=[
            "priority_rank", "rule_id", "dimension", "severity", "affected_column",
            "column_name", "issue_name", "evidence", "rationale", "suggested_action"
        ])
    else:
        table_df = pd.DataFrame(table_rows)

    return items, table_df


generate_cleaning_recommendations_v2 = generate_recommendations

