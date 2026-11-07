"""Uniqueness and key discovery metrics for Cleanalytix.

Computes distinct counts, uniqueness ratios, constant column identification,
duplicate row detection, and single/composite primary key candidate discovery.
"""

from typing import Dict, Any, List, Optional, Tuple
import itertools
import pandas as pd
import numpy as np


def compute_column_uniqueness(series: pd.Series) -> Dict[str, Any]:
    """Compute comprehensive uniqueness metrics for a single column.

    Parameters
    ----------
    series : pd.Series
        The column series to inspect.

    Returns
    -------
    dict
        Dictionary containing distinct counts, ratios, constant flags, etc.
    """
    total_count = len(series)
    if total_count == 0:
        return {
            "distinct_count": 0,
            "distinct_ratio": 0.0,
            "non_null_distinct_ratio": 0.0,
            "is_unique": False,
            "is_constant": True,
            "is_near_constant": True,
            "top_frequency_ratio": 0.0,
            "top_value": None,
        }

    valid_series = series.dropna()
    non_null_count = len(valid_series)

    try:
        distinct_count = int(series.nunique(dropna=True))
    except TypeError:
        # Handles unhashable types (e.g. lists, dicts) safely
        distinct_count = int(series.astype(str).nunique(dropna=True))

    distinct_ratio = float(distinct_count / total_count) if total_count > 0 else 0.0
    non_null_distinct_ratio = (
        float(distinct_count / non_null_count) if non_null_count > 0 else 0.0
    )

    is_unique = (distinct_count == total_count) and (non_null_count == total_count)
    is_constant = distinct_count <= 1

    top_frequency_ratio = 0.0
    top_value = None
    if non_null_count > 0:
        try:
            val_counts = valid_series.value_counts()
            if not val_counts.empty:
                top_freq = int(val_counts.iloc[0])
                top_value = val_counts.index[0]
                # Convert numpy types to native python if applicable
                if hasattr(top_value, "item"):
                    try:
                        top_value = top_value.item()
                    except (ValueError, TypeError):
                        pass
                top_frequency_ratio = float(top_freq / non_null_count)
        except Exception:
            top_frequency_ratio = 1.0 if is_constant else 0.0

    is_near_constant = top_frequency_ratio >= 0.95

    return {
        "distinct_count": distinct_count,
        "distinct_ratio": distinct_ratio,
        "non_null_distinct_ratio": non_null_distinct_ratio,
        "is_unique": is_unique,
        "is_constant": is_constant,
        "is_near_constant": is_near_constant,
        "top_frequency_ratio": top_frequency_ratio,
        "top_value": top_value,
    }


def compute_duplicate_row_metrics(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute duplicate row metrics for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    subset : list of str, optional
        Specific subset of columns to evaluate. If None, uses all columns.

    Returns
    -------
    dict
        Duplicate row counts and ratios.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "duplicate_row_count": 0,
            "duplicate_row_ratio": 0.0,
            "unique_row_count": 0,
        }

    try:
        dup_mask = df.duplicated(subset=subset, keep="first")
        duplicate_row_count = int(dup_mask.sum())
    except TypeError:
        # If columns contain unhashable types
        str_df = df[subset] if subset else df
        dup_mask = str_df.astype(str).duplicated(keep="first")
        duplicate_row_count = int(dup_mask.sum())

    duplicate_row_ratio = float(duplicate_row_count / total_rows)
    unique_row_count = total_rows - duplicate_row_count

    return {
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_ratio": duplicate_row_ratio,
        "unique_row_count": unique_row_count,
    }


def detect_candidate_keys(
    df: pd.DataFrame,
    max_composite: int = 2,
    candidate_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Identify candidate single-column and composite primary keys.

    A candidate key must uniquely identify all rows and have 0 missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    max_composite : int, default 2
        Maximum number of columns in composite key search (1 or 2).
    candidate_columns : list of str, optional
        Subset of columns to consider.

    Returns
    -------
    dict
        Dictionary with 'single_keys' and 'composite_keys'.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {"single_keys": [], "composite_keys": []}

    cols = candidate_columns if candidate_columns is not None else list(df.columns)
    single_keys: List[str] = []
    composite_keys: List[Tuple[str, ...]] = []

    # First pass: check single columns
    viable_cols: List[str] = []
    for col in cols:
        s = df[col]
        # Must have no nulls
        if s.isna().any():
            continue
        try:
            if s.nunique(dropna=False) == total_rows:
                single_keys.append(col)
            else:
                viable_cols.append(col)
        except TypeError:
            if s.astype(str).nunique(dropna=False) == total_rows:
                single_keys.append(col)
            else:
                viable_cols.append(col)

    # Second pass: composite keys (if max_composite >= 2 and total_rows > 1)
    if max_composite >= 2 and total_rows > 1 and len(viable_cols) >= 2:
        # Limit search to top 20 candidate columns to avoid O(N^2) combinatorial explosion on wide tables
        search_cols = viable_cols[:20]
        for combo in itertools.combinations(search_cols, 2):
            try:
                sub = df[list(combo)]
                if not sub.duplicated().any():
                    composite_keys.append(combo)
            except Exception:
                continue

    return {
        "single_keys": single_keys,
        "composite_keys": composite_keys,
    }


def compute_identifier_uniqueness(
    df: pd.DataFrame,
    identifier_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute uniqueness and duplicate metrics specifically for primary/identifier columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    identifier_columns : list of str, optional
        Explicit identifier columns to evaluate. If None, checks for columns with
        names containing 'id', 'uuid', 'key', 'code', 'pk' or single-key candidates.

    Returns
    -------
    dict
        Per-column identifier uniqueness statistics and overall identifier score.
    """
    total_rows = len(df)
    if total_rows == 0:
        return {
            "has_identifiers": False,
            "identifier_columns": [],
            "column_metrics": {},
            "mean_duplicate_percent": 0.0,
            "identifier_uniqueness_score": 100.0,
        }

    cols = identifier_columns
    if cols is None:
        # Detect candidates heuristically
        cols = [
            c for c in df.columns
            if any(k in str(c).lower() for k in ["_id", "id", "uuid", "guid", "pk", "key", "code"])
        ]

    cols = [c for c in cols if c in df.columns]
    if not cols:
        return {
            "has_identifiers": False,
            "identifier_columns": [],
            "column_metrics": {},
            "mean_duplicate_percent": 0.0,
            "identifier_uniqueness_score": 100.0,
        }

    col_metrics: Dict[str, Dict[str, Any]] = {}
    dup_percents: List[float] = []

    for c in cols:
        ser = df[c]
        non_null_cnt = int(ser.dropna().shape[0])
        if non_null_cnt == 0:
            dup_pct = 0.0
            distinct_cnt = 0
            dup_cnt = 0
        else:
            try:
                distinct_cnt = int(ser.nunique(dropna=True))
            except TypeError:
                distinct_cnt = int(ser.astype(str).nunique(dropna=True))
            dup_cnt = max(0, non_null_cnt - distinct_cnt)
            dup_pct = float(dup_cnt / non_null_cnt) * 100.0

        dup_percents.append(dup_pct)
        col_metrics[c] = {
            "non_null_count": non_null_cnt,
            "distinct_count": distinct_cnt,
            "duplicate_count": dup_cnt,
            "duplicate_percent": round(dup_pct, 4),
            "is_unique": dup_cnt == 0 and non_null_cnt == total_rows,
        }

    mean_dup = float(np.mean(dup_percents)) if dup_percents else 0.0
    id_score = max(0.0, 100.0 - mean_dup)

    return {
        "has_identifiers": True,
        "identifier_columns": cols,
        "column_metrics": col_metrics,
        "mean_duplicate_percent": round(mean_dup, 4),
        "identifier_uniqueness_score": round(id_score, 2),
    }
