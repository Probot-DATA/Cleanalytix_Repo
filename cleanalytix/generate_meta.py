"""Comprehensive metadata profiling and dataset characterization module for Cleanalytix.

Computes column-level profiling metrics, statistical distributions, cardinality,
type conformance, whitespace issues, custom rule errors, and distribution drift.
"""

from __future__ import annotations

import math
import re
import warnings
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from cleanalytix.column_role import (
    harmonize_series_for_drift,
    infer_column_role,
)
from cleanalytix.types import ColumnRole, Rules
from cleanalytix.metrics.drift import (
    compute_numeric_drift,
    compute_categorical_drift,
)
from cleanalytix.metrics.statistical import (
    compute_numeric_statistics,
    detect_outliers,
)
from cleanalytix.metrics.distribution import (
    compute_categorical_distribution,
)

EPS = 1e-9


def get_entropy(
    df_base: pd.DataFrame,
    column: str,
    column_role: Optional[ColumnRole] = None,
) -> float:
    """Compute normalized Shannon entropy in [0.0, 1.0] for one column.

    For continuous/discrete numeric columns, uses equal-width histogram binning
    with the Freedman-Diaconis rule (or Scott's rule fallback) over [min, max].
    This provides normalized discrete entropy H / ln(K) in [0, 1], properly
    distinguishing uniform, peaked, skewed, and constant distributions.
    For categorical columns, uses the normalized discrete Shannon entropy over
    observed category frequencies.
    """
    try:
        col_ser = df_base[column]
        if column_role is None:
            column_role = infer_column_role(col_ser, column, len(df_base))

        if column_role in {"continuous", "discrete"} and is_numeric_dtype(col_ser) and not is_bool_dtype(col_ser):
            clean = pd.to_numeric(col_ser, errors="coerce").dropna()
            clean_arr = clean[~np.isinf(clean)].to_numpy(dtype=float)
            n = len(clean_arr)
            if n < 2:
                return 0.0

            x_min, x_max = float(np.min(clean_arr)), float(np.max(clean_arr))
            if x_min == x_max:
                return 0.0

            # Freedman-Diaconis rule
            max_abs = float(np.max(np.abs(clean_arr)))
            with np.errstate(over="ignore", invalid="ignore"):
                q75, q25 = np.percentile(clean_arr, [75, 25])
                iqr = float(q75 - q25)
                if iqr > 0 and np.isfinite(iqr):
                    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
                else:
                    # Scott's rule fallback
                    if max_abs > 1e150:
                        std = float(np.std(clean_arr / max_abs, ddof=1)) * max_abs if n > 1 else 0.0
                    else:
                        std = float(np.std(clean_arr, ddof=1)) if n > 1 else 0.0
                    if std > 0 and np.isfinite(std):
                        bin_width = 3.49 * std / (n ** (1.0 / 3.0))
                    else:
                        return 0.0

            if bin_width <= 0 or not np.isfinite(bin_width):
                return 0.0

            try:
                diff = x_max - x_min
                if not np.isfinite(diff) or diff <= 0:
                    k = 10
                else:
                    k = int(np.ceil(diff / bin_width))
                    k = max(5, min(k, 50))
            except Exception:
                k = 10
            counts, _ = np.histogram(clean_arr, bins=k, range=(x_min, x_max))
            probs = counts / n
            probs = probs[probs > 0]
            if len(probs) <= 1:
                return 0.0
            h = -float(np.sum(probs * np.log(probs)))
            norm_ent = float(h / math.log(k))
            return min(1.0, max(0.0, norm_ent))

        else:
            clean = col_ser.dropna()
            n = len(clean)
            if n < 2:
                return 0.0
            try:
                val_counts = clean.value_counts()
            except TypeError:
                val_counts = clean.astype(str).value_counts()

            k = len(val_counts)
            if k <= 1:
                return 0.0
            probs = val_counts.to_numpy(dtype=float) / n
            h = -float(np.sum(probs * np.log(probs + 1e-15)))
            norm_ent = float(h / math.log(k))
            return min(1.0, max(0.0, norm_ent))

    except Exception:
        return 0.0


def get_type_error_count(df: pd.DataFrame, column: str) -> int:
    """Count values that conflict with the column's inferred semantic type."""
    col = df[column]
    col_non_null = col.dropna()
    n_non_null = len(col_non_null)

    if n_non_null == 0:
        return 0

    def is_numeric_value(v):
        if isinstance(v, bool):
            return False
        try:
            pd.to_numeric([v], errors="raise")
            return True
        except (ValueError, TypeError):
            return False

    def is_date_like(v):
        if not isinstance(v, str):
            return False
        s = v.strip()
        if re.search(r"[-/:\sT]", s):
            return True
        if s.isdigit() and len(s) == 8:
            return True
        return False

    if is_numeric_dtype(col_non_null) and not is_bool_dtype(col_non_null):
        expected = "numeric"
    else:
        numeric_conv = pd.to_numeric(col_non_null, errors="coerce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            datetime_conv = pd.to_datetime(col_non_null, errors="coerce", format="mixed")

        numeric_ratio = float(numeric_conv.notna().mean()) if n_non_null else 0.0
        datetime_ratio = float(datetime_conv.notna().mean()) if n_non_null else 0.0

        if numeric_ratio >= 0.5 and numeric_ratio >= datetime_ratio:
            expected = "numeric"
        elif datetime_ratio >= 0.5 and datetime_ratio > numeric_ratio:
            expected = "datetime"
        else:
            expected = "categorical"

    if expected == "numeric":
        converted = pd.to_numeric(col, errors="coerce")
        return int(((converted.isna() & col.notna()) | np.isinf(converted)).sum())

    if expected == "datetime":
        converted = pd.to_datetime(col, errors="coerce", format="mixed")
        return int(converted.isna().sum() - col.isna().sum())

    # In a categorical/string column, legitimate category labels frequently include
    # numeric codes ("404", "100"), date strings ("March 2024"), or range labels ("10-20").
    # String values are valid categorical labels and must not be flagged as type errors.
    # Only non-string objects (raw numeric or datetime types in a predominantly string column)
    # represent type errors.
    str_count = sum(1 for v in col_non_null if isinstance(v, str))
    if str_count >= 0.5 * n_non_null:
        return int(sum(1 for v in col_non_null if not isinstance(v, str)))

    return 0


def rule_based_errors(
    df: pd.DataFrame,
    column: str,
    rules: Optional[Rules] = None,
) -> int:
    """Count values that fail a user-defined business rule."""
    if not rules or column not in rules:
        return 0

    rule_fn = rules[column]
    errors = 0
    for val in df[column]:
        if pd.isna(val):
            continue
        try:
            if not rule_fn(val):
                errors += 1
        except Exception:
            errors += 1
    return errors


def get_probs(
    df: pd.DataFrame,
    column: str,
    bins: Sequence[Any],
    column_role: Optional[ColumnRole] = None,
) -> list[float]:
    """Compute a probability distribution for one column over fixed bins."""
    col_vals = df[column].dropna()
    if len(col_vals) == 0:
        return [1.0]

    if column_role is None:
        column_role = infer_column_role(df[column], column, len(df))

    probabilities = []

    if column_role in {"continuous", "discrete"} and is_numeric_dtype(col_vals) and not is_bool_dtype(col_vals):
        try:
            clipped = col_vals.clip(bins[0], bins[-1])
            bin_indices = pd.cut(
                clipped,
                bins=bins,
                include_lowest=True,
                labels=False,
            )
            k = len(bins) - 1
            counts = Counter(bin_indices)
            for i in range(k):
                probabilities.append(counts.get(i, 0) / max(1, len(clipped)))
        except Exception:
            k = max(1, len(bins) - 1)
            probabilities = [1.0 / k] * k
    else:
        bins = list(bins)
        counts = Counter(col_vals.astype(str))
        for b in bins:
            probabilities.append(counts.get(b, 0) / max(1, len(col_vals)))

    probs = np.array(probabilities, dtype=float) + EPS
    probs = probs / probs.sum()
    return probs.tolist()


def KL_PSI_and_JS(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    column: str,
) -> tuple[float, float, float]:
    """Compute drift divergences (KL, PSI, JS) for one baseline/production column pair."""
    try:
        column_role = infer_column_role(df_base[column], column, len(df_base))
        base_series, new_series = harmonize_series_for_drift(df_base[column], df_new[column])

        # Use specialized drift module for numeric continuous
        if column_role in {"continuous", "discrete"} and is_numeric_dtype(base_series) and not is_bool_dtype(base_series):
            drift_res = compute_numeric_drift(base_series, new_series)
            kl_val = drift_res["kl_divergence"]
            psi_val = drift_res["psi"]
            js_val = drift_res["js_divergence"]
            return float(kl_val), float(psi_val), float(js_val)

        base_clean = base_series.dropna()
        new_clean = new_series.dropna()

        if len(base_clean) == 0 or len(new_clean) == 0:
            return 0.0, 0.0, 0.0

        cat_drift = compute_categorical_drift(base_series, new_series)
        kl_val = cat_drift.get("kl_divergence", 0.0)
        psi_val = cat_drift["psi"]
        js_val = cat_drift["js_divergence"]
        return float(kl_val), float(psi_val), float(js_val)

    except Exception:
        return 0.0, 0.0, 0.0


def drift_metrics(
    dataset_name_list: Sequence[str],
    df_base_list: Sequence[pd.DataFrame],
    df_new_list: Sequence[pd.DataFrame],
) -> list[list[Any]]:
    """Compute drift metrics between aligned baseline and comparison datasets."""
    metrics = []
    for df_base, df_new, dataset_name in zip(df_base_list, df_new_list, dataset_name_list):
        df_base = pd.DataFrame(df_base)
        df_new = pd.DataFrame(df_new)

        for column in df_base.columns:
            if column in df_new.columns:
                KL_div, PSI_div, JS_div = KL_PSI_and_JS(df_base, df_new, column)
            else:
                KL_div, PSI_div, JS_div = 1.0, 1.0, 1.0

            metrics.append([
                dataset_name,
                column,
                KL_div,
                PSI_div,
                JS_div,
            ])
    return metrics


def get_metrics(
    df_list: Sequence[pd.DataFrame],
    dataset_name_list: Sequence[str],
    rules: Optional[Rules] = None,
) -> list[list[Any]]:
    """Collect raw column-level profiling metrics across datasets."""
    rules = {} if rules is None else dict(rules)
    data = []

    for df, dataset_name in zip(df_list, dataset_name_list):
        df = pd.DataFrame(df)
        n_rows = len(df)

        if n_rows == 0:
            duplicate_row_percentage = 0.0
        else:
            try:
                n_unique_rows = len(df.drop_duplicates())
                duplicate_row_percentage = ((n_rows - n_unique_rows) / n_rows) * 100.0
            except TypeError:
                # Unhashable types handling
                n_unique_rows = len(df.astype(str).drop_duplicates())
                duplicate_row_percentage = ((n_rows - n_unique_rows) / n_rows) * 100.0

        for column in df.columns:
            dtype = df[column].dtype
            column_role = infer_column_role(df[column], column, n_rows)
            non_null_count = int(df[column].notnull().sum())
            missing_count = int(df[column].isnull().sum())
            missing_percent = (missing_count / max(1, n_rows)) * 100.0

            try:
                unique_count = int(df[column].nunique(dropna=True))
            except TypeError:
                unique_count = int(df[column].astype(str).nunique(dropna=True))

            unique_percent = (unique_count / max(1, n_rows)) * 100.0

            try:
                mode_series = df[column].mode()
                mode = mode_series.iloc[0] if len(mode_series) > 0 else None
                if mode is not None and hasattr(mode, "item"):
                    try:
                        mode = mode.item()
                    except (ValueError, TypeError):
                        pass
            except Exception:
                mode = None

            if mode is not None:
                try:
                    mode_percent = float(((df[column] == mode).sum() / max(1, n_rows)) * 100.0)
                except Exception:
                    mode_percent = None
            else:
                mode_percent = None

            # Statistical defaults
            mean = None
            std = None
            minv = None
            maxv = None
            median = None
            skewness = None
            skewness_level = None
            outliers = None
            num_categories = None
            rare_category_percent = None
            whitespace_issues = None
            min_date = None
            max_date = None
            date_range_days = None

            col_vals = df[column].dropna()
            entropy = get_entropy(df, column, column_role=column_role)
            type_errors = get_type_error_count(df, column)
            rule_errors = rule_based_errors(df, column, rules)

            # Continuous and Discrete Numeric columns
            if column_role in {"continuous", "discrete"} and is_numeric_dtype(df[column]) and not is_bool_dtype(df[column]):
                if not col_vals.empty:
                    stats_dict = compute_numeric_statistics(col_vals)
                    mean = stats_dict["mean"]
                    # Bessel-corrected sample std (ddof=1)
                    std = stats_dict["std"]
                    minv = stats_dict["min"]
                    maxv = stats_dict["max"]
                    median = stats_dict["median"]
                    skewness = stats_dict["skewness"]

                    if skewness is not None:
                        abs_skew = abs(skewness)
                        if abs_skew < 0.5:
                            skewness_level = "low"
                        elif abs_skew < 1.0:
                            skewness_level = "moderate"
                        else:
                            skewness_level = "high"

                    # Outlier detection
                    out_dict = detect_outliers(col_vals)
                    outliers = out_dict["recommended_outlier_count"]

            # Binary flags and identifiers
            elif column_role in {"binary", "identifier"}:
                num_categories = unique_count
                outliers = 0
                vals = df[column].dropna().astype(str)
                if not vals.empty:
                    freq = vals.value_counts(normalize=True) * 100.0
                    rare = freq[freq < 1.0]
                    rare_category_percent = float(rare.sum()) if not rare.empty else 0.0
                    whitespace_issues = int((vals.str.startswith(" ") | vals.str.endswith(" ")).sum())
                else:
                    rare_category_percent = 0.0
                    whitespace_issues = 0

            # Datetime-like
            elif column_role == "datetime" or pd.api.types.is_datetime64_any_dtype(df[column]):
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    ser_dt = df[column]
                else:
                    ser_dt = pd.to_datetime(df[column], errors="coerce")
                if ser_dt.notna().any():
                    min_date = ser_dt.min()
                    max_date = ser_dt.max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range_days = int((max_date - min_date).days)

            # Categorical / text / boolean / other
            else:
                num_categories = unique_count
                vals = df[column].dropna().astype(str)
                if not vals.empty:
                    freq = vals.value_counts(normalize=True) * 100.0
                    rare = freq[freq < 1.0]
                    rare_category_percent = float(rare.sum()) if not rare.empty else 0.0
                    whitespace_issues = int((vals.str.startswith(" ") | vals.str.endswith(" ")).sum())
                else:
                    rare_category_percent = 0.0
                    whitespace_issues = 0

            data.append([
                dataset_name,
                column,
                column_role,
                dtype,
                non_null_count,
                missing_count,
                missing_percent,
                unique_count,
                unique_percent,
                mode,
                mode_percent,
                mean,
                std,
                skewness,
                skewness_level,
                minv,
                maxv,
                median,
                outliers,
                num_categories,
                rare_category_percent,
                whitespace_issues,
                min_date,
                max_date,
                date_range_days,
                entropy,
                type_errors,
                rule_errors,
                duplicate_row_percentage,
            ])

    return data


def generate_meta(
    dataset_name_list: Sequence[str],
    dataset_list: Sequence[pd.DataFrame],
    new_dataset_list: Optional[Sequence[pd.DataFrame]] = None,
    rules: Optional[Rules] = None,
) -> pd.DataFrame:
    """Generate the full Cleanalytix metadata table.

    Fixed and validated:
    - BUG-03: Baseline-only profiling does NOT self-split or penalize dataset against itself.
    - BUG-04: Constant-value baseline columns handle production divergence gracefully.
    - BUG-08 & BUG-09: Sample statistics use Bessel's correction (ddof=1) and Fisher excess kurtosis.
    - BUG-12: Duplicate column names in any input DataFrame are identified and rejected.
    """
    rules = {} if rules is None else dict(rules)

    # Validate column uniqueness across all input DataFrames (BUG-12 fix)
    for idx, df in enumerate(dataset_list):
        if not df.columns.is_unique:
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            name = dataset_name_list[idx] if idx < len(dataset_name_list) else f"index {idx}"
            raise ValueError(
                f"DataFrame '{name}' contains duplicate column names: {dup_cols}. "
                f"Columns must be strictly unique for data quality analysis."
            )

    if new_dataset_list is not None:
        for idx, df in enumerate(new_dataset_list):
            if not df.columns.is_unique:
                dup_cols = df.columns[df.columns.duplicated()].tolist()
                name = dataset_name_list[idx] if idx < len(dataset_name_list) else f"index {idx}"
                raise ValueError(
                    f"Production DataFrame '{name}' contains duplicate column names: {dup_cols}. "
                    f"Columns must be strictly unique for data quality analysis."
                )

    profile_list = new_dataset_list if new_dataset_list is not None else dataset_list
    data = get_metrics(profile_list, dataset_name_list, rules)

    meta_data = pd.DataFrame(
        data=data,
        columns=[
            "dataset_name",
            "column_name",
            "column_role",
            "dtype",
            "non_null_count",
            "missing_count",
            "missing_percent",
            "unique_count",
            "unique_percent",
            "mode",
            "mode_percent",
            "mean",
            "std",
            "skewness",
            "skewness_level",
            "min",
            "max",
            "median",
            "outlier_count",
            "num_categories",
            "rare_category_percent",
            "whitespace_issues",
            "min_date",
            "max_date",
            "date_range_days",
            "Entropy",
            "type_errors",
            "rule_errors",
            "duplicate_rows_percent",
        ],
    )
    meta_data.replace({pd.NaT: np.nan}, inplace=True)

    # Compute drift metrics
    if new_dataset_list is None:
        # BUG-03 FIX: Single dataset baseline mode does NOT split against itself.
        # Divergence of dataset against itself is 0.0.
        drift_rows = []
        for df, ds_name in zip(dataset_list, dataset_name_list):
            for col in df.columns:
                drift_rows.append([ds_name, col, 0.0, 0.0, 0.0])
        meta_table_adv = pd.DataFrame(
            data=drift_rows,
            columns=[
                "dataset_name",
                "column_name",
                "kl_divergence",
                "psi_divergence",
                "js_divergence",
            ],
        )
    else:
        if len(new_dataset_list) != len(dataset_list):
            raise ValueError("new_dataset_list must match dataset_list length")

        for i, (ref_df, new_df) in enumerate(zip(dataset_list, new_dataset_list)):
            if set(ref_df.columns) != set(new_df.columns):
                missing_in_new = set(ref_df.columns) - set(new_df.columns)
                extra_in_new = set(new_df.columns) - set(ref_df.columns)
                raise ValueError(
                    f"Column mismatch in dataset index {i}. "
                    f"Missing in new: {missing_in_new}. "
                    f"Extra in new: {extra_in_new}."
                )

        metrics = drift_metrics(dataset_name_list, dataset_list, new_dataset_list)
        meta_table_adv = pd.DataFrame(
            data=metrics,
            columns=[
                "dataset_name",
                "column_name",
                "kl_divergence",
                "psi_divergence",
                "js_divergence",
            ],
        )

    meta_table_combined = pd.merge(
        meta_data,
        meta_table_adv,
        on=["dataset_name", "column_name"],
        how="inner",
    )
    return meta_table_combined
