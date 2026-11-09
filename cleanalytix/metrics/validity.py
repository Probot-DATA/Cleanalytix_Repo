"""Validity domain metrics for Cleanalytix.

Computes column-level validity diagnostics across three orthogonal sub-checks:
1. Semantic type validity (type coercion errors and infinite float values).
2. String whitespace contamination (leading or trailing spaces).
3. Custom business-rule validation (callable predicate violations and exceptions).

Denominator Semantics
---------------------
All validity percentages use ``N_non_null = int(series.notnull().sum())`` as the
denominator. Because pandas ``.notnull()`` evaluates ``+inf`` and ``-inf`` as
``True`` (non-null), infinite values in numeric columns are included in the
denominator and counted in the numerator as semantic type errors.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


ALLOWED_EXPECTED_TYPES = ("numeric", "datetime", "categorical")


def compute_type_validity_metrics(
    series: pd.Series,
    expected_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute semantic type error count and percentage for a Series.

    Preserves the exact semantic type inference and error counting behavior of
    ``generate_meta.py::get_type_error_count()``:
    - Numeric columns flag unparseable strings/objects and ``+inf`` / ``-inf``.
    - Datetime columns flag non-null values that fail datetime coercion.
    - Predominantly string categorical columns (>= 50% ``str`` instances among
      non-null values) flag raw non-string objects while treating string-encoded
      numbers (e.g., ``"404"``) or dates as valid categorical labels.

    Parameters
    ----------
    series : pd.Series
        Column data to evaluate. Never mutated.
    expected_type : str, optional
        Explicit semantic type override (``"numeric"``, ``"datetime"``, or
        ``"categorical"``). If ``None``, inferred automatically from non-null
        values. Must be one of ``"numeric"``, ``"datetime"``, ``"categorical"``,
        or ``None``.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``type_error_count`` (int): Number of non-null values violating semantic type.
        - ``type_error_percentage`` (float): Percentage relative to ``N_non_null``.
        - ``non_null_count`` (int): ``int(series.notnull().sum())``.
        - ``inferred_type`` (str): Semantic type used for evaluation.

    Raises
    ------
    ValueError
        If ``expected_type`` is not ``None`` and not one of ``"numeric"``,
        ``"datetime"``, or ``"categorical"``.
    """
    if expected_type is not None and expected_type not in ALLOWED_EXPECTED_TYPES:
        raise ValueError(
            f"Invalid expected_type '{expected_type}'. "
            f"Allowed values are {ALLOWED_EXPECTED_TYPES} or None."
        )

    col_non_null = series.dropna()
    n_non_null = int(series.notnull().sum())

    if expected_type is not None:
        inferred = expected_type
    elif n_non_null == 0:
        if is_numeric_dtype(series) and not is_bool_dtype(series):
            inferred = "numeric"
        elif is_datetime64_any_dtype(series):
            inferred = "datetime"
        else:
            inferred = "categorical"
    elif is_numeric_dtype(col_non_null) and not is_bool_dtype(col_non_null):
        inferred = "numeric"
    elif is_datetime64_any_dtype(col_non_null):
        inferred = "datetime"
    else:
        numeric_conv = pd.to_numeric(col_non_null, errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            datetime_conv = pd.to_datetime(col_non_null, errors="coerce", format="mixed")

        numeric_ratio = float(numeric_conv.notna().mean()) if n_non_null else 0.0
        datetime_ratio = float(datetime_conv.notna().mean()) if n_non_null else 0.0

        if numeric_ratio >= 0.5 and numeric_ratio >= datetime_ratio:
            inferred = "numeric"
        elif datetime_ratio >= 0.5 and datetime_ratio > numeric_ratio:
            inferred = "datetime"
        else:
            inferred = "categorical"

    if n_non_null == 0:
        return {
            "type_error_count": 0,
            "type_error_percentage": 0.0,
            "non_null_count": 0,
            "inferred_type": inferred,
        }

    if inferred == "numeric":
        converted = pd.to_numeric(series, errors="coerce")
        error_count = int(((converted.isna() & series.notna()) | np.isinf(converted)).sum())
    elif inferred == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(series, errors="coerce", format="mixed")
        error_count = int(converted.isna().sum() - series.isna().sum())
    else:
        str_count = sum(1 for v in col_non_null if isinstance(v, str))
        if str_count >= 0.5 * n_non_null:
            error_count = int(sum(1 for v in col_non_null if not isinstance(v, str)))
        else:
            error_count = 0

    error_pct = float((error_count / n_non_null) * 100.0) if n_non_null > 0 else 0.0

    return {
        "type_error_count": error_count,
        "type_error_percentage": error_pct,
        "non_null_count": n_non_null,
        "inferred_type": inferred,
    }


def compute_whitespace_metrics(series: pd.Series) -> Dict[str, Any]:
    """Compute leading and trailing whitespace contamination metrics for a Series.

    Preserves the exact whitespace detection behavior of ``generate_meta.py``:
    only leading (``startswith(" ")``) or trailing (``endswith(" ")``) spaces on
    non-null stringified values are counted as whitespace issues. Internal spaces
    (e.g., ``"New York"``) are not flagged.

    Parameters
    ----------
    series : pd.Series
        Column data to evaluate. Never mutated.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``whitespace_count`` (int): Count of non-null values with leading/trailing space.
        - ``whitespace_percentage`` (float): Percentage relative to ``N_non_null``.
        - ``non_null_count`` (int): ``int(series.notnull().sum())``.
    """
    n_non_null = int(series.notnull().sum())

    if n_non_null == 0:
        return {
            "whitespace_count": 0,
            "whitespace_percentage": 0.0,
            "non_null_count": 0,
        }

    vals = series.dropna().astype(str)
    whitespace_count = int((vals.str.startswith(" ") | vals.str.endswith(" ")).sum())
    whitespace_pct = float((whitespace_count / n_non_null) * 100.0)

    return {
        "whitespace_count": whitespace_count,
        "whitespace_percentage": whitespace_pct,
        "non_null_count": n_non_null,
    }


def compute_rule_validity_metrics(
    series: pd.Series,
    rule_fn: Optional[Callable[[Any], bool]] = None,
) -> Dict[str, Any]:
    """Evaluate a custom business-rule callable against non-null values of a Series.

    Preserves the exact evaluation semantics of ``generate_meta.py::rule_based_errors()``:
    - Null values (``pd.isna(val)``) are skipped and excluded from rule failures.
    - Non-null values for which ``rule_fn(val)`` returns a falsy value are counted.
    - Non-null values for which ``rule_fn(val)`` raises any ``Exception`` are
      safely caught and counted as rule violations.

    Parameters
    ----------
    series : pd.Series
        Column data to evaluate. Never mutated.
    rule_fn : Callable[[Any], bool], optional
        Validation predicate returning ``True`` for valid values and ``False``
        for invalid values. If ``None``, zero rule errors are reported.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - ``rule_error_count`` (int): Count of non-null values failing ``rule_fn``.
        - ``rule_error_percentage`` (float): Percentage relative to ``N_non_null``.
        - ``non_null_count`` (int): ``int(series.notnull().sum())``.
    """
    n_non_null = int(series.notnull().sum())

    if rule_fn is None or n_non_null == 0:
        return {
            "rule_error_count": 0,
            "rule_error_percentage": 0.0,
            "non_null_count": n_non_null,
        }

    errors = 0
    for val in series:
        if pd.isna(val):
            continue
        try:
            if not rule_fn(val):
                errors += 1
        except Exception:
            errors += 1

    error_pct = float((errors / n_non_null) * 100.0)

    return {
        "rule_error_count": int(errors),
        "rule_error_percentage": error_pct,
        "non_null_count": n_non_null,
    }


def compute_column_validity(
    series: pd.Series,
    rule_fn: Optional[Callable[[Any], bool]] = None,
    expected_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute all column-level validity metrics in a single consolidated dictionary.

    Parameters
    ----------
    series : pd.Series
        Column data to evaluate. Never mutated.
    rule_fn : Callable[[Any], bool], optional
        Optional business-rule validation callable.
    expected_type : str, optional
        Optional explicit semantic type override.

    Returns
    -------
    dict[str, Any]
        Combined dictionary containing ``non_null_count``, ``inferred_type``,
        ``type_error_count``, ``type_error_percentage``, ``whitespace_count``,
        ``whitespace_percentage``, ``rule_error_count``, and
        ``rule_error_percentage``.
    """
    type_res = compute_type_validity_metrics(series, expected_type=expected_type)
    ws_res = compute_whitespace_metrics(series)
    rule_res = compute_rule_validity_metrics(series, rule_fn=rule_fn)

    return {
        "non_null_count": type_res["non_null_count"],
        "inferred_type": type_res["inferred_type"],
        "type_error_count": type_res["type_error_count"],
        "type_error_percentage": type_res["type_error_percentage"],
        "whitespace_count": ws_res["whitespace_count"],
        "whitespace_percentage": ws_res["whitespace_percentage"],
        "rule_error_count": rule_res["rule_error_count"],
        "rule_error_percentage": rule_res["rule_error_percentage"],
    }
