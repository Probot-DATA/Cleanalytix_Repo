import pandas as pd


def get_table_for_DQ_computation(
    meta_table: pd.DataFrame,
    dataset_list: list,
    dataset_names: list,
) -> pd.DataFrame:
    """
    Convert raw meta counts into per-row percentages ready for DQ scoring.

    Drops informational columns that are not used by the scorer and converts
    raw counts (outlier_count, whitespace_issues, type_errors, rule_errors)
    into percentages relative to the row count of each dataset.

    Parameters
    ----------
    meta_table : pd.DataFrame
        Output of generate_meta().
    dataset_list : list of pd.DataFrame
        The datasets whose row counts are used for percentage computation.
    dataset_names : list of str
        Dataset names that map ``dataset_list`` entries to rows in ``meta_table``.

    Returns
    -------
    pd.DataFrame
        Summarised meta with percentage-based metric columns.
    """
    drop_cols = [
        'dtype', 'non_null_count', 'missing_count', 'unique_count',
        'mode', 'mode_percent', 'mean', 'std', 'min', 'max', 'median',
        'num_categories', 'min_date', 'max_date', 'date_range_days', 'Entropy',
    ]
    summarized_meta = meta_table.drop(columns=drop_cols).copy()

    table_lengths = {
        dataset_names[i]: len(data) for i, data in enumerate(dataset_list)
    }

    outlier_percent = []
    whitespace_percent = []
    rule_errors_percent = []
    type_errors_percent = []

    for _, val in summarized_meta.iterrows():
        n = table_lengths[val['dataset_name']]
        outlier_percent.append((float(val['outlier_count'] or 0) / n) * 100)
        whitespace_percent.append((float(val['whitespace_issues'] or 0) / n) * 100)
        rule_errors_percent.append((float(val['rule_errors'] or 0) / n) * 100)
        type_errors_percent.append((float(val['type_errors'] or 0) / n) * 100)

    summarized_meta['outlier_percent'] = outlier_percent
    summarized_meta['whitespace_percent'] = whitespace_percent
    summarized_meta['rule_errors_percent'] = rule_errors_percent
    summarized_meta['type_errors_percent'] = type_errors_percent

    summarized_meta.drop(
        columns=[
            'outlier_count', 'whitespace_issues', 'rule_errors', 'type_errors',
            'unique_percent', 'rare_category_percent', 'kl_divergence',
        ],
        inplace=True,
    )
    return summarized_meta
