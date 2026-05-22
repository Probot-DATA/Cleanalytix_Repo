# cleanalytix/adjust_prod_meta.py
import pandas as pd
import numpy as np

EPS = 1e-9

def learn_reference_profile(dataset_names, dataset_list, rare_pct_threshold=5.0):
    """
    Learn per-column numeric bounds and training-popular categories from reference datasets.
    Returns nested dict: profile[dataset_name][column] = {type, lower, upper, popular, skewness_level}
    """
    profile = {}
    for ds_name, df in zip(dataset_names, dataset_list):
        profile[ds_name] = {}
        for col in df.columns:
            ser = df[col].dropna()
            if ser.empty:
                profile[ds_name][col] = {"type": "empty"}
                continue

            if pd.api.types.is_numeric_dtype(ser):
                skew = ser.skew() if len(ser) > 2 else 0.0
                abs_skew = abs(skew)
                if abs_skew < 0.5:
                    mean = ser.mean()
                    std = ser.std(ddof=0) if ser.size > 1 else 0.0
                    if std == 0 or np.isnan(std):
                        lower, upper = -np.inf, np.inf
                    else:
                        lower, upper = mean - 3.0 * std, mean + 3.0 * std
                    method = "z"
                elif abs_skew < 1:
                    Q1, Q3 = ser.quantile(0.25), ser.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR == 0 or np.isnan(IQR):
                        lower, upper = -np.inf, np.inf
                    else:
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    method = "iqr"
                else:
                    median = ser.median()
                    MAD = np.median(np.abs(ser - median))
                    if MAD == 0 or np.isnan(MAD):
                        lower, upper = -np.inf, np.inf
                    else:
                        factor = 3.5 / 0.6745
                        delta = factor * MAD
                        lower, upper = median - delta, median + delta
                    method = "mad"

                profile[ds_name][col] = {
                    "type": "numeric",
                    "lower": float(lower) if np.isfinite(lower) else np.nan,
                    "upper": float(upper) if np.isfinite(upper) else np.nan,
                    "method": method,
                    "skewness": float(skew)
                }
            else:
                vals = ser.astype(str)
                freq = vals.value_counts(normalize=True) * 100
                popular = freq[freq > rare_pct_threshold].index.tolist()
                profile[ds_name][col] = {
                    "type": "categorical",
                    "popular": popular
                }
    return profile


def adjust_prod_meta_with_reference(profile, reference_meta, prod_meta, prod_dataset_names, prod_dataset_list):
    """
    Adjust production meta statistics using the learned reference profile.

    Returns a deep copy of ``prod_meta`` with two fields overwritten:

    - ``outlier_count``: recomputed using the numeric bounds (mean±3σ / IQR / MAD)
      derived from the reference (training) dataset, so production outliers are
      judged against the training distribution rather than their own.
    - ``rare_category_percent``: recomputed as the fraction of production values
      that do not appear in the training dataset's popular category list.

    Parameters
    ----------
    profile : dict
        Output of ``learn_reference_profile``.
    reference_meta : pd.DataFrame
        Meta table for the reference (training) split — accepted but not
        currently consumed; reserved for future drift-metric alignment.
    prod_meta : pd.DataFrame
        Meta table for the production dataset (output of ``generate_meta``).
    prod_dataset_names : list of str
    prod_dataset_list : list of pd.DataFrame
    """
    prod_meta = prod_meta.copy(deep=True)

    for ds_name, df in zip(prod_dataset_names, prod_dataset_list):
        for col in df.columns:
            mask = (prod_meta["dataset_name"] == ds_name) & (prod_meta["column_name"] == col)
            if not mask.any():
                continue

            info = profile.get(ds_name, {}).get(col, None)
            series = df[col].dropna()

            # numeric outliers using reference bounds
            if info and info.get("type") == "numeric" and len(series) > 0:
                lb, ub = info.get("lower", np.nan), info.get("upper", np.nan)
                outliers = 0
                if pd.notna(lb) and pd.notna(ub):
                    # count finite bounds only
                    try:
                        outliers = int(((series < lb) | (series > ub)).sum())
                    except Exception:
                        outliers = 0
                else:
                    # fallback: keep existing value in prod_meta (or recompute if desired)
                    # we choose to recompute with the same logic as generate_meta fallback
                    outliers = int(prod_meta.loc[mask, "outlier_count"].fillna(0).values[0])

                prod_meta.loc[mask, "outlier_count"] = int(outliers)

            # categorical rare percent relative to training popular categories
            elif info and info.get("type") == "categorical" and len(series) > 0:
                popular = info.get("popular", [])
                vals = series.astype(str)
                if len(popular) == 0:
                    # no popular categories in training -> set rare percent to 0 (or keep existing)
                    rare_percent = float(prod_meta.loc[mask, "rare_category_percent"].fillna(0).values[0])
                else:
                    rare_percent = float((~vals.isin(popular)).mean() * 100)
                prod_meta.loc[mask, "rare_category_percent"] = rare_percent


    # ensure NaT replaced
    prod_meta.replace({pd.NaT: np.nan}, inplace=True)
    return prod_meta
