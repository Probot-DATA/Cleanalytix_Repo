"""Data drift and distribution divergence metrics for Cleanalytix.

Computes Kolmogorov-Smirnov (KS) test, Wasserstein distance, Population Stability
Index (PSI), Jensen-Shannon Divergence (JSD), unseen category detection, and
robust handling of constant-value baselines.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy import stats


def compute_numeric_drift(
    base_series: pd.Series,
    prod_series: pd.Series,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Compute numerical distribution drift between baseline and production series.

    Parameters
    ----------
    base_series : pd.Series
        Reference / baseline series.
    prod_series : pd.Series
        Current / production series.
    num_bins : int, default 10
        Number of bins for Population Stability Index (PSI).
    epsilon : float, default 1e-4
        Zero-smoothing constant for PSI calculation.

    Returns
    -------
    dict
        Dictionary containing KS statistic, p-value, Wasserstein distance,
        PSI, drift status, and baseline shift metrics.

    Notes
    -----
    When the baseline is constant (zero variance), continuous Jeffreys divergence
    is formally undefined (infinite). Cleanalytix employs a discrete 3-bin surrogate
    approximation (partitioning into intervals (-inf, c - delta), [c - delta, c + delta],
    (c + delta, inf)) with Laplace smoothing to provide a smooth, continuous drift metric.
    """
    b_clean = pd.to_numeric(base_series, errors="coerce").dropna()
    p_clean = pd.to_numeric(prod_series, errors="coerce").dropna()

    b_arr = b_clean[~np.isinf(b_clean)].to_numpy(dtype=float)
    p_arr = p_clean[~np.isinf(p_clean)].to_numpy(dtype=float)

    n_b = len(b_arr)
    n_p = len(p_arr)

    default_result = {
        "ks_statistic": 0.0,
        "ks_p_value": 1.0,
        "wasserstein_distance": 0.0,
        "normalized_wasserstein": 0.0,
        "psi": 0.0,
        "kl_divergence": 0.0,
        "reverse_kl_divergence": 0.0,
        "js_divergence": 0.0,
        "drift_detected": False,
        "drift_severity": "none",
        "sample_size_base": n_b,
        "sample_size_prod": n_p,
    }

    if n_b == 0 or n_p == 0:
        return default_result

    # Check for constant baseline
    b_min, b_max = float(np.min(b_arr)), float(np.max(b_arr))
    p_min, p_max = float(np.min(p_arr)), float(np.max(p_arr))

    if b_min == b_max and p_min == p_max and b_min == p_min:
        # Both constant and identical: zero drift
        return default_result

    # 1. Kolmogorov-Smirnov 2-sample test (exact or asymptotic via scipy)
    ks_method = "asymp" if (n_b > 100 or n_p > 100) else "auto"
    ks_res = stats.ks_2samp(b_arr, p_arr, method=ks_method)
    ks_stat = float(ks_res.statistic)
    ks_p = float(ks_res.pvalue)

    # 2. Wasserstein distance
    w_dist = float(stats.wasserstein_distance(b_arr, p_arr))
    # Normalized Wasserstein by pooled standard deviation
    b_std = float(np.std(b_arr, ddof=1)) if n_b > 1 else 0.0
    p_std = float(np.std(p_arr, ddof=1)) if n_p > 1 else 0.0
    pooled_std = math.sqrt(0.5 * (b_std**2 + p_std**2)) if (b_std + p_std) > 0 else 1.0
    norm_w = float(w_dist / pooled_std) if pooled_std > 1e-12 else 0.0

    # 3. Discretization for Binned PSI, KL, and JS Divergence
    if b_min == b_max:
        # Constant baseline: discrete 3-bin surrogate approximation (partitioning into before, at, and after constant value)
        delta = 1e-7 * (abs(b_min) + 1.0)
        bin_edges = np.array([-np.inf, b_min - delta, b_min + delta, np.inf])
    else:
        # 10-quantile binning over baseline distribution
        quantiles = np.linspace(0, 100, num_bins + 1)
        bin_edges = np.percentile(b_arr, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 1:
            delta = 1e-7 * (abs(b_min) + 1.0)
            bin_edges = np.array([-np.inf, b_min - delta, b_min + delta, np.inf])
        else:
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

    b_counts, _ = np.histogram(b_arr, bins=bin_edges)
    p_counts, _ = np.histogram(p_arr, bins=bin_edges)
    k = len(b_counts)

    # Laplace smoothing across bins
    p_b = (b_counts + epsilon) / (n_b + epsilon * k)
    p_p = (p_counts + epsilon) / (n_p + epsilon * k)

    # Discrete Kullback-Leibler divergences over bins
    kl_bp = float(np.sum(p_b * np.log(p_b / p_p)))
    kl_pb = float(np.sum(p_p * np.log(p_p / p_b)))

    # Binned Population Stability Index (discrete symmetric Jeffreys divergence: KL(P||Q) + KL(Q||P))
    psi_val = float(np.sum((p_p - p_b) * np.log(p_p / p_b)))
    psi_val = max(0.0, psi_val)

    # Discrete Jensen-Shannon Divergence over bins: 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5 * (P + Q)
    m = 0.5 * (p_b + p_p)
    jsd_val = float(0.5 * np.sum(p_b * np.log(p_b / m)) + 0.5 * np.sum(p_p * np.log(p_p / m)))
    jsd_val = max(0.0, jsd_val)

    # Classify drift severity based on PSI and KS test
    if psi_val >= 0.25 or (ks_stat > 0.3 and ks_p < 0.01):
        severity = "critical"
        drift_detected = True
    elif psi_val >= 0.10 or (ks_stat > 0.15 and ks_p < 0.05):
        severity = "moderate"
        drift_detected = True
    else:
        severity = "none"
        drift_detected = False

    return {
        "ks_statistic": ks_stat,
        "ks_p_value": ks_p,
        "wasserstein_distance": w_dist,
        "normalized_wasserstein": norm_w,
        "psi": psi_val,
        "kl_divergence": kl_bp,
        "reverse_kl_divergence": kl_pb,
        "js_divergence": jsd_val,
        "drift_detected": drift_detected,
        "drift_severity": severity,
        "sample_size_base": n_b,
        "sample_size_prod": n_p,
    }


def compute_categorical_drift(
    base_series: pd.Series,
    prod_series: pd.Series,
    epsilon: float = 1e-4,
) -> Dict[str, Any]:
    """Compute categorical distribution drift, unseen categories, and JS divergence.

    Parameters
    ----------
    base_series : pd.Series
        Reference / baseline categorical series.
    prod_series : pd.Series
        Current / production categorical series.
    epsilon : float, default 1e-4
        Smoothing parameter for frequency distributions.

    Returns
    -------
    dict
        Categorical PSI, Jensen-Shannon divergence, unseen categories, and drift severity.
    """
    b_clean = base_series.dropna()
    p_clean = prod_series.dropna()

    n_b = len(b_clean)
    n_p = len(p_clean)

    default_result = {
        "psi": 0.0,
        "kl_divergence": 0.0,
        "reverse_kl_divergence": 0.0,
        "js_divergence": 0.0,
        "unseen_categories": [],
        "unseen_categories_count": 0,
        "unseen_categories_ratio": 0.0,
        "drift_detected": False,
        "drift_severity": "none",
        "sample_size_base": n_b,
        "sample_size_prod": n_p,
    }

    if n_b == 0 or n_p == 0:
        return default_result

    try:
        b_counts = b_clean.value_counts()
        p_counts = p_clean.value_counts()
    except TypeError:
        b_counts = b_clean.astype(str).value_counts()
        p_counts = p_clean.astype(str).value_counts()

    b_cats = set(b_counts.index)
    p_cats = set(p_counts.index)

    # Unseen categories in production
    unseen = list(p_cats - b_cats)
    unseen_count = len(unseen)
    unseen_row_count = sum(p_counts[c] for c in unseen) if unseen else 0
    unseen_ratio = float(unseen_row_count / n_p) if n_p > 0 else 0.0

    all_cats = list(b_cats.union(p_cats))
    k = len(all_cats)

    b_arr = np.array([b_counts.get(c, 0) for c in all_cats], dtype=float)
    p_arr = np.array([p_counts.get(c, 0) for c in all_cats], dtype=float)

    # Normalized frequencies with Laplace/epsilon smoothing
    p_b = (b_arr + epsilon) / (n_b + epsilon * k)
    p_p = (p_arr + epsilon) / (n_p + epsilon * k)

    # Exact KL divergences
    kl_bp = float(np.sum(p_b * np.log(p_b / p_p)))
    kl_pb = float(np.sum(p_p * np.log(p_p / p_b)))

    # Categorical PSI
    psi_val = float(np.sum((p_p - p_b) * np.log(p_p / p_b)))
    psi_val = max(0.0, psi_val)

    # Jensen-Shannon Divergence: JSD(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    # where M = 0.5 * (P + Q)
    m = 0.5 * (p_b + p_p)
    kl_bm = np.sum(p_b * np.log(p_b / m))
    kl_pm = np.sum(p_p * np.log(p_p / m))
    jsd_val = float(0.5 * kl_bm + 0.5 * kl_pm)
    jsd_val = max(0.0, jsd_val)

    if psi_val >= 0.25 or unseen_ratio > 0.10:
        severity = "critical"
        drift_detected = True
    elif psi_val >= 0.10 or unseen_ratio > 0.02:
        severity = "moderate"
        drift_detected = True
    else:
        severity = "none"
        drift_detected = False

    return {
        "psi": psi_val,
        "kl_divergence": kl_bp,
        "reverse_kl_divergence": kl_pb,
        "js_divergence": jsd_val,
        "unseen_categories": unseen[:20],  # cap at 20 for JSON readability
        "unseen_categories_count": unseen_count,
        "unseen_categories_ratio": unseen_ratio,
        "drift_detected": drift_detected,
        "drift_severity": severity,
        "sample_size_base": n_b,
        "sample_size_prod": n_p,
    }
