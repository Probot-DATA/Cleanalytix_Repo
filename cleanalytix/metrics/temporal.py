"""Temporal consistency, freshness, and datetime quality metrics for Cleanalytix.

Computes datetime parse rates, freshness, temporal span, monotonicity,
gap irregularity, and future date anomaly detection.
"""

from typing import Dict, Any, Optional
import datetime
import pandas as pd
import numpy as np


def compute_temporal_metrics(
    series: pd.Series,
    reference_time: Optional[datetime.datetime] = None,
) -> Dict[str, Any]:
    """Compute temporal quality, freshness, and consistency metrics for a datetime Series.

    Parameters
    ----------
    series : pd.Series
        Datetime or string series to evaluate.
    reference_time : datetime.datetime, optional
        Reference cutoff for future date detection and freshness. Defaults to utcnow.

    Returns
    -------
    dict
        Temporal metrics including span, freshness, gaps, and monotonicity.
    """
    total_count = len(series)
    empty_res = {
        "is_temporal": False,
        "parse_rate": 0.0,
        "min_date": None,
        "max_date": None,
        "time_span_days": None,
        "is_monotonic_increasing": False,
        "future_date_count": 0,
        "future_date_ratio": 0.0,
        "median_step_seconds": None,
        "max_gap_seconds": None,
        "freshness_days": None,
    }

    if total_count == 0:
        return empty_res

    # Convert to datetime safely
    if pd.api.types.is_datetime64_any_dtype(series):
        dt_series = series.dropna()
        parse_rate = float(len(dt_series) / total_count)
    else:
        # Attempt conversion
        dt_series = pd.to_datetime(series, errors="coerce").dropna()
        parse_rate = float(len(dt_series) / total_count)

    # Only treat as temporal if at least 50% parsed successfully and non-empty
    if len(dt_series) == 0 or parse_rate < 0.5:
        return empty_res

    min_dt = dt_series.min()
    max_dt = dt_series.max()
    time_span = (max_dt - min_dt).total_seconds() / 86400.0  # in days

    # Future dates check
    if reference_time is None:
        ref_dt = pd.Timestamp.now(tz=getattr(max_dt, "tz", None))
    else:
        ref_dt = pd.Timestamp(reference_time)

    try:
        # Align timezones if needed
        if getattr(ref_dt, "tz", None) is not None and getattr(dt_series.iloc[0], "tz", None) is None:
            ref_dt = ref_dt.tz_localize(None)
        elif getattr(ref_dt, "tz", None) is None and getattr(dt_series.iloc[0], "tz", None) is not None:
            ref_dt = ref_dt.tz_localize(dt_series.iloc[0].tz)

        future_mask = dt_series > ref_dt
        future_count = int(future_mask.sum())
    except Exception:
        future_count = 0

    future_ratio = float(future_count / len(dt_series)) if len(dt_series) > 0 else 0.0

    # Freshness
    freshness_days = None
    try:
        freshness_days = float((ref_dt - max_dt).total_seconds() / 86400.0)
    except Exception:
        pass

    # Monotonicity & gaps
    is_monotonic = bool(dt_series.is_monotonic_increasing)
    median_step = None
    max_gap = None

    if len(dt_series) >= 2:
        try:
            sorted_dt = dt_series.sort_values()
            diffs = (sorted_dt.diff().dropna()).dt.total_seconds()
            if not diffs.empty:
                median_step = float(diffs.median())
                max_gap = float(diffs.max())
        except Exception:
            pass

    return {
        "is_temporal": True,
        "parse_rate": parse_rate,
        "min_date": str(min_dt),
        "max_date": str(max_dt),
        "time_span_days": time_span,
        "is_monotonic_increasing": is_monotonic,
        "future_date_count": future_count,
        "future_date_ratio": future_ratio,
        "median_step_seconds": median_step,
        "max_gap_seconds": max_gap,
        "freshness_days": freshness_days,
    }
