import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

EPS = 1e-9

def get_entropy(df_base, column):
    try:
        if is_numeric_dtype(df_base[column]):
            n = len(df_base[column].dropna())

            if n < 10000:
                quantiles = np.linspace(0, 1, 6)
            elif n < 50000:
                quantiles = np.linspace(0, 1, 11)
            else:
                quantiles = np.linspace(0, 1, 21)

            bins = np.unique(
                np.quantile(df_base[column].dropna(), q=quantiles)
            )
            bins = np.sort(bins)

            if len(bins) < 2:
                return 0.0
        else:
            bins = df_base[column].dropna().astype(str).unique().tolist()

    except Exception:
        return 0.0

    probs = np.array(get_probs(df_base, column, bins), dtype=float)
    probs = probs / (probs.sum() + EPS)

    entropy = -np.sum(probs * np.log(probs + EPS))

    if len(probs) <= 1:
        return 0.0

    return float(entropy / math.log(len(probs) + EPS))


def get_type_error_count(df, column):
    col = df[column]
    col_non_null = col.dropna()

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

        # obvious date patterns
        if re.search(r"[-/:\sT]", s):
            return True

        # compact numeric date like 20240131
        if s.isdigit() and len(s) == 8:
            return True

        return False

    # Step 1: conversions
    numeric_conv = pd.to_numeric(col_non_null, errors="coerce")
    datetime_conv = pd.to_datetime(col_non_null, errors="coerce", format="mixed")

    numeric_ratio = numeric_conv.notna().mean() if len(col_non_null) else 0
    datetime_ratio = datetime_conv.notna().mean() if len(col_non_null) else 0

    # Step 2: decide type
    if numeric_ratio > 0.9 and numeric_ratio >= datetime_ratio:
        expected = "numeric"
    elif datetime_ratio > 0.9 and datetime_ratio > numeric_ratio:
        expected = "datetime"
    else:
        expected = "categorical"

    # Step 3: count errors
    if expected == "numeric":
        converted = pd.to_numeric(col, errors="coerce")
        return int(converted.isna().sum() - col.isna().sum())

    if expected == "datetime":
        converted = pd.to_datetime(col, errors="coerce", format="mixed")
        return int(converted.isna().sum() - col.isna().sum())

    error_count = 0
    for v in col:
        if pd.isna(v):
            continue

        if is_numeric_value(v):
            error_count += 1
            continue

        if is_date_like(v):
            try:
                parsed = pd.to_datetime(v, errors="raise", format="mixed")
                if pd.notna(parsed):
                    error_count += 1
            except (ValueError, TypeError):
                pass

    return error_count


def rule_based_errors(df, column, rules=None):
    errors = 0
    rules = {} if rules is None else dict(rules)

    if column in rules.keys():
        rule_fn = rules[column]

        for val in df[column]:
            try:
                if not rule_fn(val):
                    errors += 1
            except Exception:
                errors += 1

        return int(errors)

    return 0


def get_metrics(df_list, dataset_name_list, rules=None):
    rules = {} if rules is None else dict(rules)
    data = []

    for df, dataset_name in zip(df_list, dataset_name_list):
        df = pd.DataFrame(df)
        n_rows = len(df)

        duplicate_row_percentage = (
            (n_rows - len(df.drop_duplicates())) / max(1, n_rows)
        ) * 100

        for column in df.columns:
            dtype = df[column].dtype
            non_null_count = int(df[column].notnull().sum())
            missing_count = int(df[column].isnull().sum())
            missing_percent = (missing_count / max(1, n_rows)) * 100

            unique_count = int(df[column].nunique(dropna=True))
            unique_percent = (unique_count / max(1, n_rows)) * 100

            mode_series = df[column].mode()
            mode = mode_series.iloc[0] if len(mode_series) > 0 else None

            if mode is not None:
                mode_percent = ((df[column] == mode).sum() / max(1, n_rows)) * 100
            else:
                mode_percent = None

            # defaults
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

            entropy = get_entropy(df, column)
            type_errors = get_type_error_count(df, column)
            rule_errors = rule_based_errors(df, column, rules)

            # numeric
            if is_numeric_dtype(df[column]):
                if not col_vals.empty:
                    mean = float(col_vals.mean())
                    std = float(col_vals.std(ddof=0)) if col_vals.size > 1 else 0.0
                    minv = float(col_vals.min())
                    maxv = float(col_vals.max())
                    median = float(col_vals.median())
                    skewness = float(col_vals.skew())

                    abs_skew = abs(skewness) if skewness is not None else 0.0

                    if abs_skew < 0.5:
                        skewness_level = "low"

                        if std and std > 0:
                            z = (col_vals - col_vals.mean()) / std
                            outliers = int((np.abs(z) >= 3).sum())
                        else:
                            outliers = 0

                    elif abs_skew < 1:
                        skewness_level = "moderate"
                        Q1 = col_vals.quantile(0.25)
                        Q3 = col_vals.quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        outliers = int(
                            ((col_vals < lower) | (col_vals > upper)).sum()
                        )

                    else:
                        skewness_level = "high"
                        median_v = col_vals.median()
                        MAD = np.median(np.abs(col_vals - median_v))

                        if MAD > 0:
                            modified_z = 0.6745 * (col_vals - median_v) / MAD
                            outliers = int((np.abs(modified_z) > 3.5).sum())
                        else:
                            outliers = 0
                else:
                    mean = 0
                    std = 0
                    minv = 0
                    maxv = 0
                    median = 0
                    skewness = 0
                    outliers = 0

            # datetime-like
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                ser_dt = pd.to_datetime(df[column].astype(str), errors="coerce")

                if ser_dt.notna().any():
                    min_date = ser_dt.min()
                    max_date = ser_dt.max()

                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range_days = int((max_date - min_date).days)

                mean = None
                std = None
                minv = None
                maxv = None
                median = None
                skewness = None
                outliers = None
                num_categories = None
                rare_category_percent = None
                whitespace_issues = None

            # categorical / text / object
            else:
                num_categories = unique_count
                vals = df[column].dropna().astype(str)

                if not vals.empty:
                    freq = vals.value_counts(normalize=True) * 100
                    rare = freq[freq <= 5]
                    rare_category_percent = float(rare.sum()) if not rare.empty else 0.0
                    # Count rows that have leading OR trailing whitespace (not both separately,
                    # which would double-count strings like " hello ").
                    whitespace_issues = int(
                        (vals.str.startswith(" ") | vals.str.endswith(" ")).sum()
                    )
                else:
                    rare_category_percent = 0.0
                    whitespace_issues = 0

                mean = None
                std = None
                minv = None
                maxv = None
                median = None
                outliers = None
                min_date = None
                max_date = None
                date_range_days = None
                skewness = None
                skewness_level = None

            data.append(
                [
                    dataset_name,
                    column,
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
                ]
            )

    return data


def get_probs(df, column, bins):
    col_vals = df[column].dropna()

    if len(col_vals) == 0:
        return [1.0]

    probabilities = []

    # numeric: bins is numeric edges
    if is_numeric_dtype(col_vals):
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
                probabilities.append(
                    counts.get(i, 0) / max(1, len(clipped))
                )

        except Exception:
            k = max(1, len(bins) - 1)
            probabilities = [1.0 / k] * k

    # categorical/text: bins is the list of categories to check
    else:
        bins = list(bins)
        counts = Counter(col_vals.astype(str))

        for b in bins:
            probabilities.append(counts.get(b, 0) / max(1, len(col_vals)))

    probs = np.array(probabilities, dtype=float)
    probs = probs + EPS
    probs = probs / probs.sum()

    return probs.tolist()


def KL_PSI_and_JS(df_base, df_new, column):
    try:
        if is_numeric_dtype(df_base[column]):
            n = len(df_base[column].dropna())

            if n < 10000:
                quantiles = np.linspace(0, 1, 6)
            elif n < 50000:
                quantiles = np.linspace(0, 1, 11)
            else:
                quantiles = np.linspace(0, 1, 21)

            bins = np.unique(
                np.quantile(df_base[column].dropna(), q=quantiles)
            )
            bins = np.sort(bins)

            if len(bins) < 2:
                return 0.0, 0.0, 0.0

        else:
            base_vals = df_base[column].dropna().astype(str).unique().tolist()
            new_vals = df_new[column].dropna().astype(str).unique().tolist()

            bins = sorted(set(base_vals) | set(new_vals))  # UNION

    except Exception:
        return 0.0, 0.0, 0.0

    probs_base = np.array(get_probs(df_base, column, bins), dtype=float)
    probs_new  = np.array(get_probs(df_new, column, bins), dtype=float)

    # 🔥 FORCE SAME LENGTH (CRASH FIX)
    max_len = max(len(probs_base), len(probs_new))

    if len(probs_base) != len(probs_new):
        probs_base = np.pad(probs_base, (0, max_len - len(probs_base)), constant_values=EPS)
        probs_new  = np.pad(probs_new,  (0, max_len - len(probs_new)),  constant_values=EPS)

    # normalize AFTER padding
    probs_base = probs_base / (probs_base.sum() + EPS)
    probs_new  = probs_new / (probs_new.sum() + EPS)

    M = 0.5 * (probs_base + probs_new)

    KL_div = float(np.sum(probs_base * np.log(probs_base / (probs_new + EPS))))
    PSI_div = float(np.sum((probs_base - probs_new) * np.log(probs_base / (probs_new + EPS))))
    JS_div = float(0.5 * (
        np.sum(probs_base * np.log(probs_base / (M + EPS))) +
        np.sum(probs_new * np.log(probs_new / (M + EPS)))
    ))

    return KL_div, PSI_div, JS_div


def drift_metrics(dataset_name_list, df_base_list, df_new_list):
    metrics = []

    for df_base, df_new, dataset_name in zip(
        df_base_list,
        df_new_list,
        dataset_name_list,
    ):
        df_base = pd.DataFrame(df_base)
        df_new = pd.DataFrame(df_new)

        for column in df_base.columns:
            KL_div, PSI_div, JS_div = KL_PSI_and_JS(
                df_base,
                df_new,
                column,
            )

            metrics.append(
                [
                    dataset_name,
                    column,
                    KL_div,
                    PSI_div,
                    JS_div,
                ]
            )

    return metrics


def generate_meta(dataset_name_list, dataset_list, new_dataset_list=None, rules=None):
    rules = {} if rules is None else dict(rules)
    data = get_metrics(dataset_list, dataset_name_list, rules)
    df_old_list = []
    df_new_list = []

    if new_dataset_list is None:
        # internal train-test split for the "new" dataset
        for df in dataset_list:
            df_old, df_new = train_test_split(
                df,
                test_size=0.3,
                random_state=1,
            )
            df_old_list.append(df_old)
            df_new_list.append(df_new)
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

            df_old_list.append(ref_df)
            df_new_list.append(new_df)

    metrics = drift_metrics(dataset_name_list, df_old_list, df_new_list)

    meta_data = pd.DataFrame(
        data=data,
        columns=[
            "dataset_name",
            "column_name",
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
