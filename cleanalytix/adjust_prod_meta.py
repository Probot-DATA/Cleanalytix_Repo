"""Production metadata adjustment using learned reference baselines.

Enables production datasets to be scored and evaluated relative to baseline
statistical distributions, outlier bounds, and category frequencies.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd

from cleanalytix.column_role import infer_column_role
from cleanalytix.types import ReferenceProfile

EPS = 1e-9


def learn_reference_profile(
    dataset_names: Sequence[str],
    dataset_list: Sequence[pd.DataFrame],
    rare_pct_threshold: float = 5.0,
) -> ReferenceProfile:
    """Learn baseline column statistics for production metadata adjustment.

    Parameters
    ----------
    dataset_names : sequence of str
        Names aligned with ``dataset_list``.
    dataset_list : sequence of pandas.DataFrame
        Baseline/reference datasets.
    rare_pct_threshold : float, default=5.0
        Threshold for categoricals.

    Returns
    -------
    dict
        Reference profile mapping dataset and column names to baseline statistics.
    """
    profile = {}
    for ds_name, df in zip(dataset_names, dataset_list):
        profile[ds_name] = {}
        n_rows = len(df)
        for col in df.columns:
            ser = df[col].dropna()
            column_role = infer_column_role(df[col], col, n_rows)
            if ser.empty:
                profile[ds_name][col] = {"type": "empty", "column_role": column_role}
                continue

            if column_role in {"continuous", "discrete"} and pd.api.types.is_numeric_dtype(ser):
                raw_skew = ser.skew() if len(ser) > 2 else 0.0
                skew = float(raw_skew) if pd.notna(raw_skew) else 0.0
                abs_skew = abs(skew)

                if abs_skew < 0.5:
                    mean = float(ser.mean())
                    # Bessel's correction ddof=1
                    std = float(ser.std(ddof=1)) if ser.size > 1 else 0.0
                    if std == 0 or np.isnan(std):
                        # Constant baseline: lower and upper bound are identical
                        lower = upper = mean
                    else:
                        lower, upper = mean - 3.0 * std, mean + 3.0 * std
                    method = "z"

                elif abs_skew < 1.0:
                    Q1 = float(ser.quantile(0.25))
                    Q3 = float(ser.quantile(0.75))
                    IQR = Q3 - Q1
                    if IQR == 0 or np.isnan(IQR):
                        lower = upper = float(ser.median())
                    else:
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    method = "iqr"

                else:
                    median = float(ser.median())
                    MAD = float(np.median(np.abs(ser - median)))
                    if MAD == 0 or np.isnan(MAD):
                        lower = upper = median
                    else:
                        factor = 3.5 / 0.6745
                        delta = factor * MAD
                        lower, upper = median - delta, median + delta
                    method = "mad"

                profile[ds_name][col] = {
                    "type": "numeric",
                    "column_role": column_role,
                    "lower": float(lower) if np.isfinite(lower) else np.nan,
                    "upper": float(upper) if np.isfinite(upper) else np.nan,
                    "method": method,
                    "skewness": float(skew),
                }
            else:
                vals = ser.astype(str)
                freq = vals.value_counts(normalize=True) * 100.0
                popular = freq[freq > rare_pct_threshold].index.tolist()
                profile[ds_name][col] = {
                    "type": "categorical",
                    "column_role": column_role,
                    "popular": popular,
                }
    return profile


def adjust_prod_meta_with_reference(
    profile: ReferenceProfile,
    reference_meta: pd.DataFrame,
    prod_meta: pd.DataFrame,
    prod_dataset_names: Sequence[str],
    prod_dataset_list: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Adjust production metadata using learned baseline reference profiles."""
    prod_meta = prod_meta.copy(deep=True)

    for ds_name, df in zip(prod_dataset_names, prod_dataset_list):
        for col in df.columns:
            mask = (prod_meta["dataset_name"] == ds_name) & (prod_meta["column_name"] == col)
            if not mask.any():
                continue

            info = profile.get(ds_name, {}).get(col, None)
            series = df[col].dropna()
            column_role = infer_column_role(df[col], col, len(df))

            if column_role in {"binary", "identifier"}:
                prod_meta.loc[mask, "outlier_count"] = 0
                continue

            # numeric outliers using reference bounds
            if info and info.get("type") == "numeric" and len(series) > 0:
                lb, ub = info.get("lower", np.nan), info.get("upper", np.nan)
                outliers = 0
                if pd.notna(lb) and pd.notna(ub):
                    try:
                        outliers = int(((series < lb) | (series > ub)).sum())
                    except Exception:
                        outliers = 0
                else:
                    outliers = int(prod_meta.loc[mask, "outlier_count"].fillna(0).values[0])

                prod_meta.loc[mask, "outlier_count"] = int(outliers)

            # categorical rare percent relative to baseline popular categories
            elif info and info.get("type") == "categorical" and len(series) > 0:
                popular = info.get("popular", [])
                vals = series.astype(str)
                if len(popular) == 0:
                    rare_percent = float(
                        prod_meta.loc[mask, "rare_category_percent"].fillna(0).values[0]
                    )
                else:
                    rare_percent = float((~vals.isin(popular)).mean() * 100.0)
                prod_meta.loc[mask, "rare_category_percent"] = rare_percent

    prod_meta.replace({pd.NaT: np.nan}, inplace=True)
    return prod_meta
