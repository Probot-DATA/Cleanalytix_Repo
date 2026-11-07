"""Feature relationship and cross-column correlation metrics for Cleanalytix.

Computes Pearson and Spearman correlation matrices, Cramér's V for categorical
associations, multicollinearity / high correlation detection, duplicate columns,
and near-duplicate columns.
"""

from typing import Dict, Any, List, Tuple, Optional
import math
import itertools
import numpy as np
import pandas as pd
from scipy import stats


def compute_numeric_correlations(
    df: pd.DataFrame,
    threshold: float = 0.85,
) -> Dict[str, Any]:
    """Compute Pearson and Spearman correlation matrices and identify highly correlated pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default 0.85
        Absolute correlation threshold for flagging multicollinear pairs.

    Returns
    -------
    dict
        Pearson matrix, Spearman matrix, and flagged high-correlation pairs.
    """
    # Select numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return {
            "pearson": {},
            "spearman": {},
            "high_correlation_pairs": [],
        }

    # Only inspect columns with some variance
    viable_cols = [c for c in num_cols if df[c].dropna().nunique() > 1]
    if len(viable_cols) < 2:
        return {
            "pearson": {},
            "spearman": {},
            "high_correlation_pairs": [],
        }

    sub_df = df[viable_cols]
    pearson_df = sub_df.corr(method="pearson")
    spearman_df = sub_df.corr(method="spearman")

    high_corr_pairs = []
    seen = set()

    for c1, c2 in itertools.combinations(viable_cols, 2):
        p_val = pearson_df.loc[c1, c2]
        s_val = spearman_df.loc[c1, c2]

        # Use maximum absolute correlation
        p_abs = abs(p_val) if pd.notna(p_val) else 0.0
        s_abs = abs(s_val) if pd.notna(s_val) else 0.0
        max_abs = max(p_abs, s_abs)

        if max_abs >= threshold:
            high_corr_pairs.append({
                "column_1": c1,
                "column_2": c2,
                "pearson": float(p_val) if pd.notna(p_val) else None,
                "spearman": float(s_val) if pd.notna(s_val) else None,
                "max_abs_correlation": float(max_abs),
            })

    # Sort descending by correlation strength
    high_corr_pairs.sort(key=lambda x: x["max_abs_correlation"], reverse=True)

    return {
        "pearson": pearson_df.round(4).to_dict(),
        "spearman": spearman_df.round(4).to_dict(),
        "high_correlation_pairs": high_corr_pairs,
    }


def compute_cramers_v(series1: pd.Series, series2: pd.Series) -> float:
    """Compute Cramér's V statistic for association between two categorical series.

    Applies bias-corrected formulation (Bergsma 2013).

    Parameters
    ----------
    series1 : pd.Series
        First categorical variable.
    series2 : pd.Series
        Second categorical variable.

    Returns
    -------
    float
        Cramér's V value between 0.0 (no association) and 1.0 (perfect association).
    """
    # Drop rows where either is NA
    valid_mask = series1.notna() & series2.notna()
    s1 = series1[valid_mask]
    s2 = series2[valid_mask]
    n = len(s1)

    if n == 0:
        return 0.0

    confusion_matrix = pd.crosstab(s1, s2)
    r, k = confusion_matrix.shape

    if r <= 1 or k <= 1:
        return 0.0

    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    phi2 = chi2 / n

    # Bias correction
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1)) if n > 1 else 0.0
    r_corr = r - ((r - 1) ** 2) / (n - 1) if n > 1 else r
    k_corr = k - ((k - 1) ** 2) / (n - 1) if n > 1 else k
    denom = min((k_corr - 1), (r_corr - 1))

    if denom <= 0:
        return 0.0

    v = math.sqrt(phi2_corr / denom)
    return float(min(1.0, max(0.0, v)))


def detect_duplicate_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Detect exactly identical and near-duplicate columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    dict
        List of duplicate column pairs and near-duplicate column pairs.
    """
    cols = list(df.columns)
    n_cols = len(cols)
    if n_cols < 2:
        return {"duplicate_columns": [], "near_duplicate_columns": []}

    exact_duplicates: List[Tuple[str, str]] = []
    near_duplicates: List[Dict[str, Any]] = []

    # Limit search to avoid combinatorial explosion if thousands of columns
    search_cols = cols[:100]

    for c1, c2 in itertools.combinations(search_cols, 2):
        s1 = df[c1]
        s2 = df[c2]

        try:
            # Check exact equality (including null alignment)
            if s1.equals(s2):
                exact_duplicates.append((c1, c2))
                continue
        except Exception:
            pass

        # Check near duplicate (same non-null match ratio >= 0.99)
        try:
            both_valid = s1.notna() & s2.notna()
            n_both = int(both_valid.sum())
            if n_both > 10:
                match_cnt = int((s1[both_valid] == s2[both_valid]).sum())
                match_ratio = match_cnt / n_both
                if match_ratio >= 0.99:
                    near_duplicates.append({
                        "column_1": c1,
                        "column_2": c2,
                        "match_ratio": float(match_ratio),
                    })
        except Exception:
            continue

    return {
        "duplicate_columns": exact_duplicates,
        "near_duplicate_columns": near_duplicates,
    }
