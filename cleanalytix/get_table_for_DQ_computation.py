"""Scoring table preparation module for Cleanalytix.

Converts metadata counts into percentage-based metric tables for legacy scoring workflows.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def get_table_for_DQ_computation(
    meta_table: pd.DataFrame,
    dataset_list: Sequence[pd.DataFrame],
    dataset_names: Sequence[str],
) -> pd.DataFrame:
    """Build the scoring-ready metric table used by ``Compute_DQ_Score``.

    Parameters
    ----------
    meta_table : pd.DataFrame
        Full metadata table returned by :func:`generate_meta`.
    dataset_list : sequence of pandas.DataFrame
        Datasets whose row counts are used to convert raw counts into
        percentages. The sequence must align with ``dataset_names``.
    dataset_names : sequence of str
        Dataset labels aligned with ``dataset_list`` and with
        ``meta_table["dataset_name"]``.

    Returns
    -------
    pd.DataFrame
        Scoring-ready table with one row per dataset-column pair.
    """
    drop_cols = [
        'column_role', 'dtype', 'non_null_count', 'missing_count', 'unique_count',
        'mode', 'mode_percent', 'mean', 'std', 'min', 'max', 'median',
        'num_categories', 'min_date', 'max_date', 'date_range_days', 'Entropy',
    ]
    # Drop only existing columns defensively
    existing_drop = [c for c in drop_cols if c in meta_table.columns]
    summarized_meta = meta_table.drop(columns=existing_drop).copy()

    table_lengths = {
        dataset_names[i]: len(data) for i, data in enumerate(dataset_list)
    }

    outlier_percent = []
    whitespace_percent = []
    rule_errors_percent = []
    type_errors_percent = []

    for _, val in summarized_meta.iterrows():
        n_non_null = val.get('non_null_count')
        if pd.notna(n_non_null):
            n_safe = max(1, int(n_non_null))
            has_present = int(n_non_null) > 0
        else:
            n = table_lengths.get(val['dataset_name'], 1)
            n_safe = max(1, n)
            has_present = True

        # Handle None / NaN gracefully without ZeroDivisionError
        out_cnt = val.get('outlier_count')
        ws_cnt = val.get('whitespace_issues')
        rule_cnt = val.get('rule_errors')
        type_cnt = val.get('type_errors')

        outlier_percent.append((float(out_cnt) / n_safe * 100.0) if (pd.notna(out_cnt) and has_present) else (np.nan if pd.isna(out_cnt) else 0.0))
        whitespace_percent.append((float(ws_cnt) / n_safe * 100.0) if (pd.notna(ws_cnt) and has_present) else (np.nan if pd.isna(ws_cnt) else 0.0))
        rule_errors_percent.append((float(rule_cnt) / n_safe * 100.0) if (pd.notna(rule_cnt) and has_present) else 0.0)
        type_errors_percent.append((float(type_cnt) / n_safe * 100.0) if (pd.notna(type_cnt) and has_present) else 0.0)

    summarized_meta['outlier_percent'] = outlier_percent
    summarized_meta['whitespace_percent'] = whitespace_percent
    summarized_meta['rule_errors_percent'] = rule_errors_percent
    summarized_meta['type_errors_percent'] = type_errors_percent

    cleanup_cols = [
        'outlier_count', 'whitespace_issues', 'rule_errors', 'type_errors',
        'unique_percent', 'rare_category_percent', 'kl_divergence',
    ]
    existing_cleanup = [c for c in cleanup_cols if c in summarized_meta.columns]
    summarized_meta.drop(columns=existing_cleanup, inplace=True)
    return summarized_meta
