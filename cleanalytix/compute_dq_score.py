"""Legacy data quality scoring module for Cleanalytix.

Provides backward-compatible interfaces to Compute_DQ_Score and get_new_values.
"""

from __future__ import annotations

import math
import warnings
from typing import Final, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

from cleanalytix.types import MetricWeights, ScoreMode, Thresholds
from cleanalytix.scoring import (
    DEFAULT_THRESHOLDS,
    resolve_thresholds,
    calculate_metric_score,
)

WeightsInput = Optional[Union[MetricWeights, np.ndarray, pd.Series]]
DEFAULT_SCORE_MODE: Final[ScoreMode] = "linear"
VALID_SCORE_MODES: Final[set[ScoreMode]] = {"linear", "exponential"}


def _metric_columns(summarized_meta: pd.DataFrame) -> list[str]:
    """Return metric column names from summarized_meta, excluding metadata columns."""
    drop_cols = {"skewness", "skewness_level", "dataset_name", "column_name"}
    metrics = [col for col in summarized_meta.columns if col not in drop_cols]
    return metrics


def _validate_score_mode(score_mode: str) -> ScoreMode:
    """Normalize and validate scoring mode."""
    if not isinstance(score_mode, str):
        raise TypeError("score_mode must be a string: 'linear' or 'exponential'")

    mode = score_mode.strip().lower()
    if mode not in VALID_SCORE_MODES:
        raise ValueError(
            f"Invalid score_mode '{score_mode}'. "
            f"Expected one of: {sorted(VALID_SCORE_MODES)}"
        )
    return cast(ScoreMode, mode)


def get_weights(weights: WeightsInput, summarized_meta: pd.DataFrame) -> list[float]:
    """Validate or create metric weights for data-quality scoring."""
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


def get_new_values(
    summarized_meta: pd.DataFrame,
    thresholds: Optional[Thresholds] = None,
    score_mode: ScoreMode = DEFAULT_SCORE_MODE,
) -> pd.DataFrame:
    """Convert scoring-ready metric values into per-metric scores in [0.0, 1.0]."""
    thr_map = resolve_thresholds(thresholds)
    metrics = _metric_columns(summarized_meta)
    temp = summarized_meta.copy(deep=True)
    mode = _validate_score_mode(score_mode)

    for metric in metrics:
        metric_key = metric.lower()
        scores = []

        for _, row in temp.iterrows():
            v = row.get(metric, np.nan)
            if pd.isna(v):
                scores.append(np.nan)
                continue

            k_val = 3.0 if metric_key in {"psi_divergence", "js_divergence"} else 2.0
            thr = thr_map.get(metric_key, 1.0)
            scores.append(calculate_metric_score(float(v), float(thr), score_mode=mode, k=k_val))

        temp[metric] = scores

    return temp


def Compute_DQ_Score(
    dataset_name_list: Sequence[str],
    dataset_list: Sequence[pd.DataFrame],
    summarized_meta: pd.DataFrame,
    given_weights: WeightsInput = None,
    thresholds: Optional[Thresholds] = None,
    score_mode: ScoreMode = DEFAULT_SCORE_MODE,
) -> dict[str, float]:
    """Compute one data-quality score per dataset with dynamic applicability re-weighting."""
    metrics = _metric_columns(summarized_meta)
    scored_meta = get_new_values(summarized_meta, thresholds=thresholds, score_mode=score_mode)
    results: dict[str, float] = {}

    if len(dataset_name_list) != len(dataset_list):
        warnings.warn(
            "dataset_name_list and dataset_list lengths differ; ignoring extras (zip behavior).",
            UserWarning,
            stacklevel=2,
        )

    base_weights = get_weights(given_weights, summarized_meta)
    weight_map = dict(zip(metrics, base_weights))

    for ds_name, df in zip(dataset_name_list, dataset_list):
        per_column_scores = []
        df = pd.DataFrame(df)

        if len(df) == 0:
            results[ds_name] = 0.0
            continue

        for col in df.columns:
            key_match = (
                (scored_meta["dataset_name"] == ds_name) &
                (scored_meta["column_name"] == col)
            )
            col_rows = scored_meta[key_match]

            applicable_scores = []
            applicable_weights = []

            for metric in metrics:
                if col_rows.empty:
                    continue
                val = col_rows[metric].iloc[0]
                if pd.notna(val):
                    applicable_scores.append(float(val))
                    applicable_weights.append(weight_map.get(metric, 1.0))

            if applicable_scores:
                tot_w = sum(applicable_weights)
                col_score = sum(s * (w / tot_w) for s, w in zip(applicable_scores, applicable_weights))
                per_column_scores.append(col_score)
            else:
                per_column_scores.append(1.0)

        avg_score = float(np.mean(per_column_scores)) * 100.0 if per_column_scores else 0.0
        results[ds_name] = round(avg_score, 2)

    return results
