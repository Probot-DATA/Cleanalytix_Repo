"""Distribution and categorical diversity metrics for Cleanalytix.

Computes Shannon entropy (raw and normalized), Herfindahl-Hirschman Index (HHI),
Gini-Simpson diversity index, category frequency tables, and rare category detection.
"""

from typing import Dict, Any, List, Optional
import math
import pandas as pd
import numpy as np


def compute_categorical_distribution(
    series: pd.Series, top_n: int = 10, rare_threshold: float = 0.01
) -> Dict[str, Any]:
    """Compute distribution diversity, entropy, and concentration for categorical data.

    Parameters
    ----------
    series : pd.Series
        Categorical or discrete Series.
    top_n : int, default 10
        Number of top categories to return in summary.
    rare_threshold : float, default 0.01
        Proportion threshold below which a category is classified as 'rare'.

    Returns
    -------
    dict
        Dictionary containing entropy, HHI, dominance, top frequencies, and rare category stats.
    """
    valid = series.dropna()
    n = len(valid)

    empty_res = {
        "unique_categories": 0,
        "shannon_entropy": 0.0,
        "normalized_entropy": 0.0,
        "hhi": 0.0,
        "gini_simpson_diversity": 0.0,
        "top_category": None,
        "top_category_ratio": 0.0,
        "is_dominated": False,
        "rare_categories_count": 0,
        "rare_categories_ratio": 0.0,
        "top_frequencies": {},
    }

    if n == 0:
        return empty_res

    try:
        val_counts = valid.value_counts()
    except TypeError:
        val_counts = valid.astype(str).value_counts()

    k = len(val_counts)
    if k == 0:
        return empty_res

    probs = val_counts.to_numpy(dtype=float) / n

    # Shannon entropy: H = -sum(p * ln(p))
    entropy = -float(np.sum(probs * np.log(probs + 1e-15)))
    entropy = max(0.0, entropy)

    # Normalized entropy: H / ln(K)
    if k > 1:
        max_entropy = math.log(k)
        normalized_entropy = float(entropy / max_entropy)
        normalized_entropy = min(1.0, max(0.0, normalized_entropy))
    else:
        normalized_entropy = 0.0

    # Herfindahl-Hirschman Index (HHI): sum(p^2)
    hhi = float(np.sum(probs**2))

    # Gini-Simpson diversity: 1 - HHI
    gini_simpson = float(1.0 - hhi)

    top_cat = val_counts.index[0]
    if hasattr(top_cat, "item"):
        try:
            top_cat = top_cat.item()
        except (ValueError, TypeError):
            pass

    top_ratio = float(probs[0])
    is_dominated = top_ratio >= 0.90

    # Rare categories (proportion < rare_threshold)
    rare_mask = probs < rare_threshold
    rare_count = int(np.sum(rare_mask))
    rare_ratio = float(np.sum(probs[rare_mask]))

    # Top N frequencies formatted cleanly
    top_freq_dict = {}
    for idx, (val, cnt) in enumerate(val_counts.head(top_n).items()):
        val_key = str(val) if not isinstance(val, (str, int, float, bool)) else val
        top_freq_dict[val_key] = {
            "count": int(cnt),
            "ratio": float(cnt / n),
        }

    return {
        "unique_categories": k,
        "shannon_entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "hhi": hhi,
        "gini_simpson_diversity": gini_simpson,
        "top_category": top_cat,
        "top_category_ratio": top_ratio,
        "is_dominated": is_dominated,
        "rare_categories_count": rare_count,
        "rare_categories_ratio": rare_ratio,
        "top_frequencies": top_freq_dict,
    }


def compute_continuous_distribution(
    series: pd.Series,
    num_bins: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute distribution metrics for continuous numeric data using equal-width histogram binning.

    Calculates Shannon entropy on the discretized histogram via Freedman-Diaconis rule
    (or Scott's rule fallback), normalized entropy in [0, 1], Herfindahl-Hirschman Index (HHI),
    and distribution dispersion statistics.

    Parameters
    ----------
    series : pd.Series
        Numeric series to analyze.
    num_bins : int, optional
        Explicit number of histogram bins. If None, chosen adaptively via Freedman-Diaconis.

    Returns
    -------
    dict
        Dictionary containing entropy, normalized_entropy, hhi, and bin metadata.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean_arr = clean[~np.isinf(clean)].to_numpy(dtype=float)
    n = len(clean_arr)

    empty_res = {
        "sample_size": n,
        "num_bins": 0,
        "shannon_entropy": 0.0,
        "normalized_entropy": 0.0,
        "hhi": 0.0,
        "gini_simpson_diversity": 0.0,
        "bin_width": 0.0,
        "is_constant": True,
    }

    if n < 2:
        return empty_res

    x_min, x_max = float(np.min(clean_arr)), float(np.max(clean_arr))
    if x_min == x_max:
        return empty_res

    if num_bins is not None and num_bins >= 2:
        k = num_bins
        bin_width = (x_max - x_min) / k
    else:
        q75, q25 = np.percentile(clean_arr, [75, 25])
        iqr = float(q75 - q25)
        if iqr > 0:
            bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
        else:
            std = float(np.std(clean_arr, ddof=1)) if n > 1 else 0.0
            if std > 0:
                bin_width = 3.49 * std / (n ** (1.0 / 3.0))
            else:
                return empty_res

        if bin_width <= 0:
            return empty_res

        k = int(np.ceil((x_max - x_min) / bin_width))
        k = max(5, min(k, 50))
        bin_width = (x_max - x_min) / k

    counts, _ = np.histogram(clean_arr, bins=k, range=(x_min, x_max))
    probs = counts / n
    active_probs = probs[probs > 0]

    h = -float(np.sum(active_probs * np.log(active_probs)))
    norm_h = float(h / math.log(k)) if k > 1 else 0.0
    hhi = float(np.sum(probs**2))
    gini_simpson = float(1.0 - hhi)

    return {
        "sample_size": n,
        "num_bins": k,
        "shannon_entropy": max(0.0, h),
        "normalized_entropy": min(1.0, max(0.0, norm_h)),
        "hhi": hhi,
        "gini_simpson_diversity": gini_simpson,
        "bin_width": float(bin_width),
        "is_constant": False,
    }
