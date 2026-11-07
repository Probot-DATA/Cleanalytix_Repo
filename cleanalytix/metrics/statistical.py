"""Statistical and numerical distribution metrics for Cleanalytix.

Computes central tendency, dispersion (Bessel-corrected sample variance & std),
shape metrics (Fisher-Pearson excess kurtosis & skewness), quantile spreads,
and multi-method outlier detection (Z-score, Tukey IQR, Modified Z-score).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_numeric_statistics(series: pd.Series) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics for a numeric Series.

    Ensures mathematical rigor:
    - Sample variance and standard deviation use Bessel's correction (ddof=1).
    - Excess kurtosis uses Fisher's definition (normal distribution = 0.0).
    - Nullable types (pd.NA) are cleanly converted to native Python floats/None.
    - Sample statistics on N < 2 safely return None/NaN rather than misleading zeros.

    Parameters
    ----------
    series : pd.Series
        Numeric or coercible Series.

    Returns
    -------
    dict
        Comprehensive statistical summary.
    """
    # Coerce to numeric, ignoring non-convertible
    numeric_series = pd.to_numeric(series, errors="coerce")
    
    # Check for infinite values before filtering
    inf_mask = np.isinf(numeric_series)
    infinite_count = int(inf_mask.sum())

    # Filter out nulls and infs for statistical computation
    clean_vals = numeric_series[numeric_series.notna() & ~inf_mask]
    n = len(clean_vals)

    res: Dict[str, Any] = {
        "count": n,
        "infinite_count": infinite_count,
        "mean": None,
        "median": None,
        "std": None,
        "variance": None,
        "min": None,
        "max": None,
        "p01": None,
        "p05": None,
        "q25": None,
        "q75": None,
        "p95": None,
        "p99": None,
        "iqr": None,
        "mad": None,
        "skewness": None,
        "kurtosis": None,
        "coefficient_of_variation": None,
        "zero_count": 0,
        "zero_ratio": 0.0,
        "negative_count": 0,
        "negative_ratio": 0.0,
    }

    if n == 0:
        return res

    arr = clean_vals.to_numpy(dtype=float)

    # Basic counts
    res["zero_count"] = int((arr == 0.0).sum())
    res["zero_ratio"] = float(res["zero_count"] / n)
    res["negative_count"] = int((arr < 0.0).sum())
    res["negative_ratio"] = float(res["negative_count"] / n)

    res["min"] = float(np.min(arr))
    res["max"] = float(np.max(arr))
    mean_val = float(np.mean(arr))
    res["mean"] = mean_val
    median_val = float(np.median(arr))
    res["median"] = median_val

    # Percentiles
    if n >= 1:
        res["p01"] = float(np.percentile(arr, 1))
        res["p05"] = float(np.percentile(arr, 5))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        res["q25"] = q25
        res["q75"] = q75
        res["p95"] = float(np.percentile(arr, 95))
        res["p99"] = float(np.percentile(arr, 99))
        res["iqr"] = float(q75 - q25)

        # Median Absolute Deviation (MAD)
        mad_val = float(np.median(np.abs(arr - median_val)))
        res["mad"] = mad_val

    # N >= 2 required for sample variance and std (ddof=1)
    if n >= 2:
        var_val = float(np.var(arr, ddof=1))
        std_val = float(np.std(arr, ddof=1))
        res["variance"] = var_val
        res["std"] = std_val

        if abs(mean_val) > 1e-12:
            res["coefficient_of_variation"] = float(std_val / abs(mean_val))

    # N >= 3 required for skewness, N >= 4 for kurtosis
    if n >= 3:
        raw_skew = clean_vals.skew()
        if pd.notna(raw_skew):
            res["skewness"] = float(raw_skew)

    if n >= 4:
        raw_kurt = clean_vals.kurt()  # pandas kurt() computes Fisher's excess kurtosis
        if pd.notna(raw_kurt):
            res["kurtosis"] = float(raw_kurt)

    return res


def detect_outliers(
    series: pd.Series,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    modified_z_threshold: float = 3.5,
) -> Dict[str, Any]:
    """Detect outliers using Z-score, Tukey's IQR, and Modified Z-score methods.

    Automatically selects the recommended method based on distribution skewness:
    - Low skew (|skew| <= 1.0): Z-score or Modified Z-score recommended.
    - High skew (|skew| > 1.0): Tukey's IQR or Modified Z-score recommended.

    Parameters
    ----------
    series : pd.Series
        Numeric column data.
    z_threshold : float, default 3.0
        Cutoff for standard Z-score.
    iqr_multiplier : float, default 1.5
        Multiplier for Tukey's fences (1.5 for outliers, 3.0 for extreme outliers).
    modified_z_threshold : float, default 3.5
        Cutoff for Boris Iglewicz & David Hoaglin's Modified Z-score.

    Returns
    -------
    dict
        Outlier counts, ratios, upper/lower bounds, and recommended method result.
    """
    numeric_series = pd.to_numeric(series, errors="coerce")
    clean = numeric_series[numeric_series.notna() & ~np.isinf(numeric_series)]
    n = len(clean)

    empty_res = {
        "zscore": {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "lower_bound": None,
            "upper_bound": None,
        },
        "iqr": {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "lower_bound": None,
            "upper_bound": None,
        },
        "modified_zscore": {
            "outlier_count": 0,
            "outlier_ratio": 0.0,
            "threshold": modified_z_threshold,
        },
        "recommended_method": "iqr",
        "recommended_outlier_count": 0,
        "recommended_outlier_ratio": 0.0,
    }

    if n < 3:
        return empty_res

    arr = clean.to_numpy(dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    median_val = float(np.median(arr))

    # 1. Z-score
    z_count = 0
    z_lower = None
    z_upper = None
    if std_val > 1e-12:
        z_scores = np.abs((arr - mean_val) / std_val)
        z_count = int((z_scores > z_threshold).sum())
        z_lower = float(mean_val - z_threshold * std_val)
        z_upper = float(mean_val + z_threshold * std_val)

    # 2. Tukey's IQR
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr_val = q75 - q25
    iqr_lower = float(q25 - iqr_multiplier * iqr_val)
    iqr_upper = float(q75 + iqr_multiplier * iqr_val)
    iqr_count = int(((arr < iqr_lower) | (arr > iqr_upper)).sum())

    # 3. Modified Z-score (Iglewicz & Hoaglin: M_i = 0.6745 * |x_i - median| / MAD)
    mad_val = float(np.median(np.abs(arr - median_val)))
    mod_z_count = 0
    if mad_val > 1e-12:
        mod_z_scores = 0.6745 * np.abs(arr - median_val) / mad_val
        mod_z_count = int((mod_z_scores > modified_z_threshold).sum())
    else:
        # Fallback when MAD is zero (e.g. >50% identical values)
        # Use mean absolute deviation
        mean_ad = float(np.mean(np.abs(arr - median_val)))
        if mean_ad > 1e-12:
            mod_z_scores = 0.6745 * np.abs(arr - median_val) / mean_ad
            mod_z_count = int((mod_z_scores > modified_z_threshold).sum())

    # Determine recommended method based on skewness
    skew_val = clean.skew()
    is_skewed = False
    if pd.notna(skew_val) and abs(float(skew_val)) > 1.0:
        is_skewed = True

    if is_skewed or std_val <= 1e-12:
        rec_method = "iqr"
        rec_count = iqr_count
    else:
        rec_method = "zscore"
        rec_count = z_count

    return {
        "zscore": {
            "outlier_count": z_count,
            "outlier_ratio": float(z_count / n),
            "lower_bound": z_lower,
            "upper_bound": z_upper,
        },
        "iqr": {
            "outlier_count": iqr_count,
            "outlier_ratio": float(iqr_count / n),
            "lower_bound": iqr_lower,
            "upper_bound": iqr_upper,
        },
        "modified_zscore": {
            "outlier_count": mod_z_count,
            "outlier_ratio": float(mod_z_count / n),
            "threshold": modified_z_threshold,
        },
        "recommended_method": rec_method,
        "recommended_outlier_count": rec_count,
        "recommended_outlier_ratio": float(rec_count / n),
    }
