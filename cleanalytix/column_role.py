"""Semantic column-role heuristics for profiling, drift, and diagnostics."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .types import ColumnRole

_ID_NAME_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|(^|_)key$|_key$|(^|_)code$|_code$|(^|_)num$|(^|_)number$|(^|_)index$|(^|_)token$)",
    re.IGNORECASE,
)

VALID_COLUMN_ROLES = {
    "continuous",
    "discrete",
    "binary",
    "identifier",
    "categorical",
    "free_text",
    "datetime",
    "boolean",
    "unknown",
}


def _normalized_name(column_name) -> str:
    if column_name is None:
        return ""
    return str(column_name).strip().replace(" ", "_")


def name_suggests_identifier(column_name: object) -> bool:
    """Return whether a column name looks identifier-like."""
    return bool(_ID_NAME_PATTERN.search(_normalized_name(column_name)))


def is_binary_flag(series: pd.Series) -> bool:
    """Return whether a Series behaves like a binary flag."""
    if is_bool_dtype(series):
        return True

    values = series.dropna()
    if values.empty:
        return False

    if is_numeric_dtype(values) and not is_bool_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        if numeric.empty:
            return False
        unique = set(numeric.unique().tolist())
        return len(unique) <= 2 and unique.issubset({0.0, 1.0})

    as_str = values.astype(str).str.strip().str.lower()
    unique = set(as_str.unique().tolist())
    return len(unique) <= 2 and unique.issubset({"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"})


def is_identifier_column(
    series: pd.Series,
    column_name: object = None,
    n_rows: Optional[int] = None,
) -> bool:
    """Return whether a column behaves like an identifier."""
    if n_rows is None:
        n_rows = len(series)

    if n_rows == 0:
        return name_suggests_identifier(column_name)

    values = series.dropna()
    if values.empty:
        return name_suggests_identifier(column_name)

    nunique = int(values.nunique())
    unique_ratio = nunique / max(1, n_rows)
    name_hint = name_suggests_identifier(column_name)

    # 1. Strong name hint + reasonable uniqueness
    if name_hint and unique_ratio >= 0.5:
        return True

    # 2. Perfect 100% uniqueness for non-empty string/numeric columns of sufficient size
    if unique_ratio == 1.0 and n_rows >= 10 and not is_bool_dtype(series):
        # Distinguish from continuous floats (which might happen to be unique)
        if not (is_numeric_dtype(series) and not is_bool_dtype(series) and not name_hint):
            return True

    return False


def is_free_text_column(series: pd.Series) -> bool:
    """Return whether a non-numeric Series behaves like unstructured free text."""
    values = series.dropna().astype(str)
    if values.empty:
        return False

    # Check average character length and word count
    sample = values.head(200)
    word_counts = sample.str.split().str.len()
    avg_words = float(word_counts.mean()) if not word_counts.empty else 0.0
    avg_length = float(sample.str.len().mean()) if not sample.empty else 0.0

    return avg_words >= 3.5 or avg_length > 40


def infer_column_role(
    series: pd.Series,
    column_name: object = None,
    n_rows: Optional[int] = None,
) -> ColumnRole:
    """Infer the semantic role of a column."""
    if n_rows is None:
        n_rows = len(series)

    if is_bool_dtype(series):
        return "binary"

    if is_datetime64_any_dtype(series):
        return "datetime"

    if is_identifier_column(series, column_name, n_rows):
        return "identifier"

    if is_binary_flag(series):
        return "binary"

    if is_numeric_dtype(series) and not is_bool_dtype(series):
        vals = series.dropna()
        if not vals.empty and vals.nunique() <= 10 and (vals % 1 == 0).all():
            return "discrete"
        return "continuous"

    if is_free_text_column(series):
        return "free_text"

    return "categorical"


def harmonize_series_for_drift(
    reference_series: pd.Series,
    comparison_series: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align comparison values to the reference dtype for distribution comparisons."""
    ref = reference_series
    new = comparison_series

    if is_bool_dtype(ref):
        return ref.astype(str), new.astype(str)

    if is_datetime64_any_dtype(ref):
        return ref, pd.to_datetime(new, errors="coerce")

    if is_numeric_dtype(ref) and not is_bool_dtype(ref):
        return pd.to_numeric(ref, errors="coerce"), pd.to_numeric(new, errors="coerce")

    return ref.astype(str), new.astype(str)


def role_blocks_continuous_analysis(role: str) -> bool:
    """Return whether a role should skip continuous numeric analysis."""
    return role in {"binary", "identifier", "categorical", "free_text", "datetime"}
