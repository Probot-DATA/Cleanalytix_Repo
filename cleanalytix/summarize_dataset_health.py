"""Dataset health summary aggregation module for Cleanalytix."""

from __future__ import annotations
import pandas as pd


def summarize_dataset_health(meta_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize column-level metadata into dataset-level health metrics.

    Parameters
    ----------
    meta_table : pandas.DataFrame
        Metadata table returned by :func:`generate_meta`.

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
