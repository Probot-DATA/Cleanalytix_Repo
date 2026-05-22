import math
import warnings
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Default thresholds for metrics (values from design assumptions).
# Keys are in LOWERCASE for easier matching.
# Note: 'skewness' is NOT scored (see _metric_columns); kept only for reference.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "missing_percent": 20.0,
    "outlier_percent": 10.0,
    "whitespace_percent": 5.0,
    "type_errors_percent": 2.0,
    "rule_errors_percent": 2.0,
    "psi_divergence": 0.25,
    "js_divergence": 0.10,
}

DEFAULT_SCORE_MODE = "linear"
VALID_SCORE_MODES = {"linear", "exponential"}


def resolve_thresholds(user_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Merge user-provided thresholds into DEFAULT_THRESHOLDS.
    Keys are case-insensitive; unknown keys are reported and ignored.
    """
    final = {k.lower(): float(v) for k, v in DEFAULT_THRESHOLDS.items()}
    if not user_thresholds:
        return final

    for key, val in user_thresholds.items():
        key_lower = key.lower()
        if key_lower in final:
            try:
                final[key_lower] = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"Threshold for '{key}' must be numeric")
        else:
            warnings.warn(
                f"Unknown threshold key '{key}' — ignoring it.",
                UserWarning,
                stacklevel=3,
            )

    return final


def _metric_columns(summarized_meta: pd.DataFrame) -> List[str]:
    """
    Return the list of metric column names from summarized_meta,
    excluding metadata columns.
    """
    drop_cols = {"skewness", "skewness_level", "dataset_name", "column_name"}
    metrics = [col for col in summarized_meta.columns if col not in drop_cols]
    return metrics


def _validate_score_mode(score_mode: str) -> str:
    """
    Normalize and validate scoring mode.
    """
    if not isinstance(score_mode, str):
        raise TypeError("score_mode must be a string: 'linear' or 'exponential'")

    mode = score_mode.strip().lower()
    if mode not in VALID_SCORE_MODES:
        raise ValueError(
            f"Invalid score_mode '{score_mode}'. "
            f"Expected one of: {sorted(VALID_SCORE_MODES)}"
        )
    return mode


def get_weights(weights, summarized_meta):
    """
    Validate or create weights for metrics. Returns a list of floats that sum to 1.
    Raises TypeError/ValueError on invalid input.
    """
    metrics = _metric_columns(summarized_meta)
    n = len(metrics)

    if weights is not None:
        if not isinstance(weights, (list, tuple, np.ndarray, pd.Series)):
            raise TypeError("weights must be a list/tuple/ndarray/Series or None")
        weights_list = list(weights)
        if len(weights_list) != n:
            raise ValueError(f"Weights length ({len(weights_list)}) does not match number of metrics ({n})")
        total = float(np.sum(weights_list))
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"Weights must sum to 1.0 (got {total})")
        return [float(w) for w in weights_list]

    if n > 0:
        return [1.0 / n] * n
    return []


def _linear_score(value: float, threshold: float) -> float:
    """
    Linear decay score in [0,1].
    0 -> 1.0, threshold -> 0.0, above threshold -> 0.0
    """
    if threshold is None or threshold <= 0 or not np.isfinite(threshold):
        threshold = 1.0
    v = max(0.0, float(value))
    score = 1.0 - (v / threshold)
    return float(max(0.0, min(1.0, score)))


def _exp_score(value: float, threshold: float, k: float = 2.0) -> float:
    """
    Apply exponential decay to a positive metric value.
    Returns a score in (0,1], with larger value -> lower score.
    """
    if threshold is None or threshold <= 0 or not np.isfinite(threshold):
        threshold = 1.0
    v = max(0.0, float(value))
    severity = v / threshold
    return float(np.exp(-k * severity))


def _score_value(
    value: float,
    threshold: float,
    score_mode: str = DEFAULT_SCORE_MODE,
    k: float = 2.0
) -> float:
    """
    Convert a raw metric value into a [0,1] score using the selected mode.
    """
    mode = _validate_score_mode(score_mode)

    if mode == "linear":
        return _linear_score(value, threshold)

    return _exp_score(value, threshold, k=k)


def get_new_values(
    summarized_meta: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    score_mode: str = DEFAULT_SCORE_MODE
) -> pd.DataFrame:
    """
    Return a new DataFrame where each metric column is replaced by a score in [0,1].
    The score mode can be 'linear' or 'exponential'.
    Missing values are scored as 1.0.
    """
    thr_map = resolve_thresholds(thresholds)
    metrics = _metric_columns(summarized_meta)
    temp = summarized_meta.copy(deep=True)
    # Validate score_mode once — not inside the per-row loop
    mode = _validate_score_mode(score_mode)

    for metric in metrics:
        metric_key = metric.lower()
        scores = []

        for _, row in temp.iterrows():
            v = row.get(metric, np.nan)
            if pd.isna(v):
                scores.append(1.0)
                continue

            if metric_key in {"psi_divergence", "js_divergence"}:
                # Divergence metrics use steeper decay (k=3)
                thr = thr_map.get(metric_key, 1.0)
                scores.append(_score_value(float(v), float(thr), score_mode=mode, k=3.0))

            else:
                thr = thr_map.get(metric_key, 1.0)
                scores.append(_score_value(float(v), float(thr), score_mode=mode, k=2.0))

        temp[metric] = scores

    return temp


def Compute_DQ_Score(
    dataset_name_list: List[str],
    dataset_list: List[pd.DataFrame],
    summarized_meta: pd.DataFrame,
    given_weights: Optional[List[float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
    score_mode: str = DEFAULT_SCORE_MODE
) -> Dict[str, float]:
    """
    Compute DQ score for each dataset in dataset_list.

    Returns:
        Dict[str, float]: {dataset_name: score}

    score_mode:
        - 'linear'
        - 'exponential'
    """
    thr_map = resolve_thresholds(thresholds)
    metrics = _metric_columns(summarized_meta)
    scored_meta = get_new_values(summarized_meta, thresholds=thr_map, score_mode=score_mode)
    results: Dict[str, float] = {}

    if len(dataset_name_list) != len(dataset_list):
        warnings.warn(
        "dataset_name_list and dataset_list lengths differ; "
        "ignoring extras (zip behavior).",
        UserWarning,
        stacklevel=2
)   

    weights = get_weights(given_weights, summarized_meta)

    for ds_name, df in zip(dataset_name_list, dataset_list):
        per_column_scores = []
        df = pd.DataFrame(df)

        for col in df.columns:
            col_score = 0.0

            for j, metric in enumerate(metrics):
                key_match = (
                    (scored_meta["dataset_name"] == ds_name) &
                    (scored_meta["column_name"] == col)
                )
                matched = scored_meta.loc[key_match, metric].values
                score = float(matched[0]) if len(matched) > 0 and not pd.isna(matched[0]) else 1.0
                col_score += score * weights[j]

            per_column_scores.append(col_score)

        avg_score = float(np.mean(per_column_scores)) * 100.0 if per_column_scores else 0.0
        results[ds_name] = avg_score

    return results