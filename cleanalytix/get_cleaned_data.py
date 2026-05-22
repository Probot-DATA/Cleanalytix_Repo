import ast
import pandas as pd
import numpy as np
from nltk import ngrams
from nltk.metrics.distance import jaccard_distance
from sklearn.preprocessing import PowerTransformer
from pandas.api.types import (
    is_numeric_dtype,
    is_integer_dtype,
    is_datetime64_any_dtype,
)

# ----------------------------
# Semantic type inference
# ----------------------------
def _safe_strip(v):
    if pd.isna(v):
        return v
    return str(v).strip()

def _safe_numeric_series(col):
    """Convert a Series to numeric without mutating non-numeric values first."""
    if is_numeric_dtype(col) and col.dtype != bool:
        return pd.to_numeric(col, errors="coerce")

    s = col.astype("string")
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.strip()
    s = s.replace("", pd.NA)
    return pd.to_numeric(s, errors="coerce")

def infer_semantic_type(col, numeric_threshold=0.95, datetime_threshold=0.95):
    """
    Return ``numeric``, ``datetime``, or ``categorical``.

    Conservative on purpose: if unsure, treat as categorical.
    """
    s = col.dropna()
    if s.empty:
        return "categorical"

    if is_datetime64_any_dtype(col):
        return "datetime"

    if is_numeric_dtype(col) and col.dtype != bool:
        return "numeric"

    num_ratio = _safe_numeric_series(s).notna().mean()
    if num_ratio >= numeric_threshold:
        return "numeric"

    dt_ratio = pd.to_datetime(s, errors="coerce", format="mixed").notna().mean()
    if dt_ratio >= datetime_threshold:
        return "datetime"

    return "categorical"

def _fill_default_by_type(series, semantic_type):
    if semantic_type == "numeric":
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().any():
            return s.fillna(s.median())
        return s.fillna(0)

    if semantic_type == "datetime":
        s = pd.to_datetime(series, errors="coerce", format="mixed")
        mode = s.mode(dropna=True)
        if not mode.empty:
            return s.fillna(mode.iloc[0])
        return s

    s = series.astype("string").map(_safe_strip)
    mode = s.mode(dropna=True)
    if not mode.empty:
        return s.fillna(mode.iloc[0])
    return s.fillna("Unknown")

def _normalize_rule_ids(rule_ids):
    if isinstance(rule_ids, str):
        try:
            rule_ids = ast.literal_eval(rule_ids)
        except Exception:
            rule_ids = [rule_ids]

    if not isinstance(rule_ids, (list, tuple, set, np.ndarray)):
        rule_ids = [rule_ids]

    normalized = []
    for rid in rule_ids:
        try:
            normalized.append(int(rid))
        except Exception:
            continue
    return normalized

def _warn_if_collapsed(before_series, after_series, dataset_name, column, clean_type):
    try:
        before_unique = before_series.nunique(dropna=True)
        after_unique = after_series.nunique(dropna=True)
        after_null_pct = after_series.isna().mean()
    except Exception:
        return

    if (before_unique > 1 and after_unique <= 1) or after_null_pct == 1.0:
        print(
            f"Column collapse detected: dataset={dataset_name}, "
            f"column={column}, rule_id={clean_type}"
        )

# ----------------------------
# Main cleaner
# ----------------------------
def clean(df, column, clean_type, replacement=None, rules=None, interactive=False, semantic_type=None):
    rules = {} if rules is None else dict(rules)
    if semantic_type is None:
        semantic_type = infer_semantic_type(df[column])

    def clean_outliers(series, replacement=None):
        col = pd.to_numeric(series, errors="coerce")

        if col.dropna().empty or col.dropna().std() == 0:
            return series

        skewness_level = "low" if col.skew() < 0.5 else "moderate" if col.skew() < 1 else "high"

        if skewness_level == "low":
            std = col.std()
            if std == 0 or np.isnan(std):
                return series
            z = ((col - col.mean()) / std).abs()
            outlier_idx = z >= 3

        elif skewness_level == "moderate":
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or np.isnan(IQR):
                return series
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_idx = (col < lower) | (col > upper)

        else:
            median = col.median()
            MAD = np.median(np.abs(col - median))
            if MAD == 0 or np.isnan(MAD):
                return series
            modified_z = 0.6745 * (col - median) / MAD
            outlier_idx = np.abs(modified_z) > 3.5

        if outlier_idx.sum() == 0:
            return series

        result = series.copy()
        if replacement is not None:
            result.loc[outlier_idx] = replacement
        else:
            if is_integer_dtype(col.dropna()):
                result.loc[outlier_idx] = int(col.median())
            else:
                result.loc[outlier_idx] = col.median()
        return result

    # 1. Drop column due to high missing values
    if clean_type == 1:
        df.drop(columns=[column], inplace=True)
        return None

    # 2. Impute missing values
    elif clean_type == 2:
        if replacement is not None:
            df[column] = df[column].fillna(replacement)
        else:
            df[column] = _fill_default_by_type(df[column], semantic_type)

    # 3. Handle outliers
    elif clean_type == 3:
        if semantic_type != "numeric":
            return df[column]
        df[column] = clean_outliers(df[column], replacement)

    # 4. Strip leading/trailing spaces
    elif clean_type == 4:
        if semantic_type == "numeric":
            return df[column]
        df[column] = df[column].apply(_safe_strip)

    # 5. Combine rare categories and resolve typos
    elif clean_type == 5:
        if semantic_type != "categorical":
            return df[column]

        df[column] = df[column].astype("string").map(_safe_strip)

        counts = df[column].value_counts(dropna=True)
        total = len(df)

        substitutes = {}
        popular_categories = counts[(counts / total * 100) > 5].index.tolist()

        for k, v in counts.items():
            if pd.isna(k):
                continue

            val_pct = (v / total) * 100
            if val_pct <= 5:
                found_target = False
                k_str = str(k).lower().strip()

                if len(k_str) >= 3:
                    k_set = set("".join(g) for g in ngrams(k_str, 3))

                    for pop_val in popular_categories:
                        pop_str = str(pop_val).lower().strip()
                        pop_set = set("".join(g) for g in ngrams(pop_str, 3))

                        if k_set or pop_set:
                            dist = jaccard_distance(k_set, pop_set)
                            if dist <= 0.4:
                                substitutes[k] = pop_val
                                found_target = True
                                break

                if not found_target:
                    substitutes[k] = "Other"

        df[column] = df[column].replace(substitutes)

    # 6. Check for incorrect/same date entries
    elif clean_type == 6:
        if semantic_type == "datetime" and df[column].nunique(dropna=True) <= 1:
            print(column, "has identical or invalid date entries not required for data analysis")

    # 7. Resolve mismatches and convert safely
    elif clean_type == 7:
        if semantic_type == "numeric":
            df[column] = _safe_numeric_series(df[column])
        elif semantic_type == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
        else:
            df[column] = df[column].astype("string").map(_safe_strip)

        if interactive:
            print(f"\nColumn: {column}")
            print(f"Column type: {df[column].dtype}")
            user_val = input(
                "Enter replacement value (or type 'default' for median in numeric columns and mode in non-numeric columns): "
            ).strip()
        else:
            user_val = "default"

        if user_val == "default":
            df[column] = _fill_default_by_type(df[column], semantic_type)
        else:
            if semantic_type == "numeric":
                user_val = pd.to_numeric(user_val, errors="coerce")
            elif semantic_type == "datetime":
                user_val = pd.to_datetime(user_val, errors="coerce", format="mixed")
            df[column] = df[column].fillna(user_val)

    # 8. Replace invalid business values
    elif clean_type == 8:
        func = rules.get(column)
        if func is None:
            return df[column]
        for i, val in enumerate(df[column]):
            if replacement is not None:
                if not func(val):
                    df.loc[i, column] = replacement
            else:
                if not func(val):
                    df.loc[i, column] = None

    # 9. Highly skewed distribution; consider power transformation
    elif clean_type == 9:
        if semantic_type != "numeric":
            return df[column]

        col = pd.to_numeric(df[column], errors="coerce")
        if col.isna().all() or col.nunique(dropna=True) <= 1:
            return df[column]

        col_nonnull = col.dropna()
        if len(col_nonnull) < 3:
            return df[column]

        if abs(col_nonnull.skew()) < 1:
            return df[column]

        pt = PowerTransformer(method="yeo-johnson")
        try:
            transformed_nonnull = pt.fit_transform(col_nonnull.values.reshape(-1, 1)).ravel()
            result = col.astype(float).copy()
            result[col.notna()] = transformed_nonnull
            return result
        except Exception:
            return df[column]

    elif clean_type == 10:
        pass

    return df[column]

# ----------------------------
# Dataset-level cleaning loop
# ----------------------------
def get_cleaned_data(dataset_name_list, dataset_list, recommendations, rules=None, interactive=False):
    rules = {} if rules is None else dict(rules)
    NEEDS_REPLACEMENT = {2, 3, 8}
    RULE_PRIORITY = {
        7: 1,
        4: 2,
        2: 3,
        3: 4,
        5: 5,
        8: 6,
        6: 7,
        9: 8,
        1: 9,
        10: 10
    }

    dataset_list_cleaned = []
    change_log = []

    for item, dataset_name in zip(dataset_list, dataset_name_list):
        item = item.copy(deep=True)

        for column in list(item.columns):
            rec = recommendations[
                (recommendations["dataset_name"] == dataset_name) &
                (recommendations["column_name"] == column)
            ]

            if rec.empty:
                continue

            rule_ids = _normalize_rule_ids(rec["rule_ids"].iloc[0])
            rule_ids = sorted(rule_ids, key=lambda x: RULE_PRIORITY.get(x, 100))

            for clean_type in rule_ids:
                if column not in item.columns:
                    break

                semantic_type = infer_semantic_type(item[column])

                # Rule compatibility guards
                if clean_type in {3, 9} and semantic_type != "numeric":
                    continue
                if clean_type == 5 and semantic_type != "categorical":
                    continue
                if clean_type == 6 and semantic_type != "datetime":
                    continue
                if clean_type == 4 and semantic_type == "numeric":
                    continue

                replacement = None
                user_val = "default"

                if interactive:
                    col_dtype = item[column].dtype
                    print(f"\nDataset: {dataset_name}")
                    print(f"Column: {column}")
                    print(f"Column type: {col_dtype}")
                    print(f"Semantic type: {semantic_type}")
                    print(f"Rule ID: {clean_type}")
                    ans = input("Apply this fix? (yes/no): ").strip().lower()
                    if ans != "yes":
                        continue

                    if clean_type in NEEDS_REPLACEMENT:
                        user_val = input(
                            "Enter replacement value (or type 'default' (median for numeric columns and mode for categorical columns)): "
                        ).strip()

                if user_val.lower() != "default":
                    try:
                        if semantic_type == "numeric":
                            replacement = float(user_val)
                        elif semantic_type == "datetime":
                            replacement = pd.to_datetime(user_val, errors="coerce", format="mixed")
                        else:
                            replacement = user_val
                    except Exception:
                        print("Invalid value. Using default.")
                        replacement = None
                else:
                    replacement = None

                before = item[column].copy()

                result = clean(
                    item,
                    column,
                    clean_type,
                    replacement,
                    rules=rules,
                    interactive=interactive,
                    semantic_type=semantic_type,
                )

                if result is not None:
                    item[column] = result

                if result is not None and len(before) == len(result):
                    try:
                        affected = (before != result).sum()
                    except Exception:
                        affected = 0
                else:
                    affected = 0

                _warn_if_collapsed(before, item[column], dataset_name, column, clean_type)

                # Rule 6 only prints an advisory message; no data is changed,
                # so skip the change_log entry to avoid misleading the user.
                if clean_type == 6:
                    continue

                change_log.append({
                    "dataset": dataset_name,
                    "column": column,
                    "rule_id": clean_type,
                    "replacement": replacement if replacement is not None else "default",
                    "affected_rows": int(affected)
                })

        # dataset-level duplicate handling
        if interactive:
            print("Do you want to remove any possible duplicates from dataset", dataset_name, "? (yes/no)")
            ans = input().strip().lower()
            if ans == "yes":
                before_len = len(item)
                item = item.drop_duplicates()
                after_len = len(item)
                dropped = before_len - after_len
                if dropped > 0:
                    change_log.append({
                        "dataset": dataset_name,
                        "column": "__ROW_LEVEL__",
                        "rule_id": 11,
                        "replacement": None,
                        "affected_rows": int(dropped),
                    })
        else:
            before_len = len(item)
            item = item.drop_duplicates()
            after_len = len(item)
            dropped = before_len - after_len
            if dropped > 0:
                change_log.append({
                    "dataset": dataset_name,
                    "column": "__ROW_LEVEL__",
                    "rule_id": 11,
                    "replacement": None,
                    "affected_rows": int(dropped),
                })

        # Convert only truly categorical columns to category
        for c in item.columns:
            if infer_semantic_type(item[c]) == "categorical" and item[c].dtype == "object":
                item[c] = item[c].astype("category")

        dataset_list_cleaned.append(item)

    return dataset_list_cleaned, change_log
