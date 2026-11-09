"""Scoring table preparation module for Cleanalytix.

Converts metadata counts into percentage-based metric tables for legacy scoring workflows.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd

from cleanalytix.types import ProfileContext


def get_table_for_DQ_computation(
    meta_table: Union[ProfileContext, Sequence[ProfileContext], pd.DataFrame],
    dataset_list: Optional[Sequence[pd.DataFrame]] = None,
    dataset_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build the scoring-ready metric table used by ``Compute_DQ_Score``.

    Parameters
    ----------
    meta_table : ProfileContext, sequence of ProfileContext, or pd.DataFrame
        ``ProfileContext`` (v2.0 native, using canonical ``N_non_null``
        validity/outlier denominators per Architecture Decision B) or legacy
        metadata table returned by :func:`generate_meta` (preserving historical
        ``N_rows`` denominator behavior for legacy callers).
    dataset_list : sequence of pandas.DataFrame, optional
        Datasets whose row counts are used to convert raw counts into
        percentages in the legacy ``meta_table`` DataFrame path.
    dataset_names : sequence of str, optional
        Dataset labels aligned with ``dataset_list`` and with
        ``meta_table["dataset_name"]``.

    Returns
    -------
    pd.DataFrame
        Scoring-ready table with one row per dataset-column pair.
    """
    if isinstance(meta_table, ProfileContext) or (
        isinstance(meta_table, Sequence) and not isinstance(meta_table, pd.DataFrame)
    ):
        contexts = [meta_table] if isinstance(meta_table, ProfileContext) else list(meta_table)
        rows = []
        for ctx in contexts:
            ds_name = ctx.dataset.dataset_name
            total_rows = ctx.dataset.total_rows
            dup_row_pct = (
                float((ctx.dataset.duplicate_rows_count / total_rows) * 100.0)
                if total_rows > 0
                else float(ctx.dataset.duplicate_rows_percent)
            )
            for col in ctx.column_profiles:
                role = str(col.role)
                dtype_obj = col.dtype_object if col.dtype_object is not None else col.dtype
                comp = col.completeness
                val = col.validity
                stat = col.statistical
                drift = col.drift

                non_null = int(comp.non_null_count) if comp is not None else 0
                miss_pct = float(comp.null_percentage) if comp is not None else 0.0

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

                skew_val = None
                skew_lvl = None
                out_pct: float = np.nan
                ws_pct: float = np.nan

                if is_num_branch:
                    if non_null > 0 and stat is not None:
                        skew_val = stat.skewness
                        if skew_val is not None:
                            abs_s = abs(skew_val)
                            skew_lvl = (
                                "low" if abs_s < 0.5 else ("moderate" if abs_s < 1.0 else "high")
                            )
                        out_pct = float(stat.outlier_percentage)
                elif role in {"binary", "identifier"}:
                    out_pct = 0.0
                    ws_pct = float(val.whitespace_percentage) if (val is not None and non_null > 0) else 0.0
                elif not is_dt_branch:
                    ws_pct = float(val.whitespace_percentage) if (val is not None and non_null > 0) else 0.0

                rule_pct = float(val.rule_error_percentage) if (val is not None and non_null > 0) else 0.0
                type_pct = float(val.type_error_percentage) if (val is not None and non_null > 0) else 0.0

                rows.append({
                    "dataset_name": ds_name,
                    "column_name": col.name,
                    "missing_percent": miss_pct,
                    "skewness": skew_val,
                    "skewness_level": skew_lvl,
                    "duplicate_rows_percent": dup_row_pct,
                    "reverse_kl_divergence": drift.reverse_kl_divergence if drift is not None else None,
                    "js_divergence": drift.js_divergence if drift is not None else None,
                    "psi_divergence": drift.psi if drift is not None else None,
                    "outlier_percent": out_pct,
                    "whitespace_percent": ws_pct,
                    "rule_errors_percent": rule_pct,
                    "type_errors_percent": type_pct,
                })
        expected_cols = [
            "dataset_name",
            "column_name",
            "missing_percent",
            "skewness",
            "skewness_level",
            "duplicate_rows_percent",
            "reverse_kl_divergence",
            "js_divergence",
            "psi_divergence",
            "outlier_percent",
            "whitespace_percent",
            "rule_errors_percent",
            "type_errors_percent",
        ]
        return pd.DataFrame(rows, columns=expected_cols)

    dataset_list = dataset_list or []
    dataset_names = dataset_names or []
    drop_cols = [
        'column_role', 'dtype', 'non_null_count', 'missing_count', 'unique_count',
        'mode', 'mode_percent', 'mean', 'std', 'min', 'max', 'median',
        'num_categories', 'min_date', 'max_date', 'date_range_days', 'Entropy',
    ]
    # Drop only existing columns defensively
    existing_drop = [c for c in drop_cols if c in meta_table.columns]
    summarized_meta = meta_table.drop(columns=existing_drop).copy()

    table_lengths = {
        dataset_names[i]: len(data) for i, data in enumerate(dataset_list)
    }

    outlier_percent = []
    whitespace_percent = []
    rule_errors_percent = []
    type_errors_percent = []

    for _, val in summarized_meta.iterrows():
        n_non_null = val.get('non_null_count')
        if pd.notna(n_non_null):
            n_safe = max(1, int(n_non_null))
            has_present = int(n_non_null) > 0
        else:
            n = table_lengths.get(val['dataset_name'], 1)
            n_safe = max(1, n)
            has_present = True

        # Handle None / NaN gracefully without ZeroDivisionError
        out_cnt = val.get('outlier_count')
        ws_cnt = val.get('whitespace_issues')
        rule_cnt = val.get('rule_errors')
        type_cnt = val.get('type_errors')

        outlier_percent.append((float(out_cnt) / n_safe * 100.0) if (pd.notna(out_cnt) and has_present) else (np.nan if pd.isna(out_cnt) else 0.0))
        whitespace_percent.append((float(ws_cnt) / n_safe * 100.0) if (pd.notna(ws_cnt) and has_present) else (np.nan if pd.isna(ws_cnt) else 0.0))
        rule_errors_percent.append((float(rule_cnt) / n_safe * 100.0) if (pd.notna(rule_cnt) and has_present) else 0.0)
        type_errors_percent.append((float(type_cnt) / n_safe * 100.0) if (pd.notna(type_cnt) and has_present) else 0.0)

    summarized_meta['outlier_percent'] = outlier_percent
    summarized_meta['whitespace_percent'] = whitespace_percent
    summarized_meta['rule_errors_percent'] = rule_errors_percent
    summarized_meta['type_errors_percent'] = type_errors_percent

    cleanup_cols = [
        'outlier_count', 'whitespace_issues', 'rule_errors', 'type_errors',
        'unique_percent', 'rare_category_percent', 'kl_divergence',
    ]
    existing_cleanup = [c for c in cleanup_cols if c in summarized_meta.columns]
    summarized_meta.drop(columns=existing_cleanup, inplace=True)
    return summarized_meta
