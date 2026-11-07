"""Completeness analytics for tabular datasets."""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def compute_completeness_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute comprehensive completeness diagnostics across rows and columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset to analyze. Never mutated.

    Returns
    -------
    dict[str, Any]
        Dictionary of completeness metrics:
        - ``column_missing_count``: Mapping of column name to null count
        - ``column_missing_percent``: Mapping of column name to null percentage
        - ``row_completeness_mean``: Average percentage of non-null fields per row
        - ``complete_rows_count``: Number of rows with zero missing values
        - ``complete_rows_percent``: Percentage of completely filled rows
        - ``empty_rows_count``: Number of rows where all fields are null
        - ``empty_rows_percent``: Percentage of completely empty rows
        - ``missingness_cooccurrence``: Top pairs of columns missing together (Jaccard)
        - ``missingness_correlations``: Correlation between missingness indicators
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    if n_rows == 0 or n_cols == 0:
        return {
            "column_missing_count": {c: 0 for c in df.columns},
            "column_missing_percent": {c: 0.0 for c in df.columns},
            "row_completeness_mean": 100.0,
            "complete_rows_count": 0,
            "complete_rows_percent": 0.0,
            "empty_rows_count": 0,
            "empty_rows_percent": 0.0,
            "missingness_cooccurrence": [],
            "missingness_correlations": {},
        }

    null_mask = df.isna()
    col_null_counts = null_mask.sum().to_dict()
    col_null_pcts = {c: float((cnt / n_rows) * 100.0) for c, cnt in col_null_counts.items()}

    # Row completeness
    row_non_null = n_cols - null_mask.sum(axis=1)
    row_completeness = (row_non_null / n_cols) * 100.0
    complete_rows = int((null_mask.sum(axis=1) == 0).sum())
    empty_rows = int((null_mask.sum(axis=1) == n_cols).sum())

    # Missingness co-occurrence (Jaccard index of missingness)
    cols_with_nulls = [c for c, cnt in col_null_counts.items() if cnt > 0]
    cooccurrences = []

    for i in range(len(cols_with_nulls)):
        for j in range(i + 1, len(cols_with_nulls)):
            c1 = cols_with_nulls[i]
            c2 = cols_with_nulls[j]
            m1 = null_mask[c1]
            m2 = null_mask[c2]
            both_null = int((m1 & m2).sum())
            either_null = int((m1 | m2).sum())
            if either_null > 0:
                jaccard = both_null / either_null
                if jaccard >= 0.20:
                    cooccurrences.append({
                        "column_a": c1,
                        "column_b": c2,
                        "both_missing_count": both_null,
                        "jaccard_similarity": float(round(jaccard, 4)),
                    })

    cooccurrences.sort(key=lambda x: x["jaccard_similarity"], reverse=True)

    # Missingness correlation between indicator columns
    correlations = {}
    if len(cols_with_nulls) >= 2:
        try:
            null_float = null_mask[cols_with_nulls].astype(float)
            corr_mat = null_float.corr()
            for c1 in cols_with_nulls:
                for c2 in cols_with_nulls:
                    if c1 < c2:
                        val = corr_mat.loc[c1, c2]
                        if pd.notna(val) and abs(val) >= 0.30:
                            correlations[f"{c1}__{c2}"] = float(round(val, 4))
        except Exception:
            correlations = {}

    return {
        "column_missing_count": {c: int(v) for c, v in col_null_counts.items()},
        "column_missing_percent": col_null_pcts,
        "row_completeness_mean": float(round(float(row_completeness.mean()), 2)),
        "complete_rows_count": complete_rows,
        "complete_rows_percent": float(round((complete_rows / n_rows) * 100.0, 2)),
        "empty_rows_count": empty_rows,
        "empty_rows_percent": float(round((empty_rows / n_rows) * 100.0, 2)),
        "missingness_cooccurrence": cooccurrences[:10],
        "missingness_correlations": correlations,
    }


# Convenience alias
compute_missingness_metrics = compute_completeness_metrics

