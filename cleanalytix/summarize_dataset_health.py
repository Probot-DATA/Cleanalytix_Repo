"""Dataset health summary aggregation module for Cleanalytix."""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np
import pandas as pd

from cleanalytix.types import ProfileContext


def summarize_dataset_health(
    meta_table: Union[ProfileContext, Sequence[ProfileContext], pd.DataFrame]
) -> pd.DataFrame:
    """Summarize column-level metadata into dataset-level health metrics.

    Parameters
    ----------
    meta_table : ProfileContext, sequence of ProfileContext, or pandas.DataFrame
        ``ProfileContext`` (v2.0 native) or metadata table returned by :func:`generate_meta`.

    Returns
    -------
    pandas.DataFrame
        One row per dataset with health summary aggregates.
    """
    expected_cols = [
        "dataset_name",
        "avg_missing_percent",
        "avg_duplicate_percent",
        "total_outliers",
        "total_whitespace_issues",
    ]

    if isinstance(meta_table, ProfileContext) or (
        isinstance(meta_table, Sequence) and not isinstance(meta_table, pd.DataFrame)
    ):
        contexts = [meta_table] if isinstance(meta_table, ProfileContext) else list(meta_table)
        contexts = [c for c in contexts if len(c.columns) > 0]
        if not contexts:
            return pd.DataFrame(columns=expected_cols)

        summary_rows = []
        for ctx in contexts:
            ds_name = ctx.dataset.dataset_name
            total_rows = ctx.dataset.total_rows
            dup_pct = (
                float((ctx.dataset.duplicate_rows_count / total_rows) * 100.0)
                if total_rows > 0
                else float(ctx.dataset.duplicate_rows_percent)
            )
            miss_pcts = [
                float(c.completeness.null_percentage) if c.completeness else 0.0
                for c in ctx.column_profiles
            ]
            avg_miss = float(np.mean(miss_pcts)) if miss_pcts else 0.0

            tot_outliers = 0
            tot_ws = 0
            for c in ctx.column_profiles:
                role = str(c.role)
                dtype_obj = c.dtype_object if c.dtype_object is not None else c.dtype
                is_num_branch = (
                    role in {"continuous", "discrete"}
                    and c.statistical is not None
                    and pd.api.types.is_numeric_dtype(dtype_obj)
                    and not pd.api.types.is_bool_dtype(dtype_obj)
                )
                is_dt_branch = (
                    role == "datetime"
                    or pd.api.types.is_datetime64_any_dtype(dtype_obj)
                )
                if is_num_branch and c.completeness and c.completeness.non_null_count > 0 and c.statistical:
                    tot_outliers += int(c.statistical.outlier_count)
                elif not is_num_branch and not is_dt_branch and c.validity:
                    tot_ws += int(c.validity.whitespace_count)

            summary_rows.append({
                "dataset_name": ds_name,
                "avg_missing_percent": avg_miss,
                "avg_duplicate_percent": dup_pct,
                "total_outliers": tot_outliers,
                "total_whitespace_issues": tot_ws,
            })
        return pd.DataFrame(summary_rows)

    if meta_table.empty:
        return pd.DataFrame(columns=expected_cols)

    agg_dict = {
        "missing_percent": "mean",
        "duplicate_rows_percent": "mean",
        "outlier_count": "sum",
        "whitespace_issues": "sum",
    }
    # Keep only columns present in meta_table
    safe_agg = {k: v for k, v in agg_dict.items() if k in meta_table.columns}

    summary = meta_table.groupby(
        "dataset_name",
        as_index=False,
    ).agg(safe_agg)

    rename_map = {
        "missing_percent": "avg_missing_percent",
        "duplicate_rows_percent": "avg_duplicate_percent",
        "outlier_count": "total_outliers",
        "whitespace_issues": "total_whitespace_issues",
    }
    summary.rename(columns=rename_map, inplace=True)

    return summary
