def summarize_dataset_health(meta_table):
    summary = meta_table.groupby(
        "dataset_name",
        as_index=False
    ).agg({
        "missing_percent": "mean",
        "duplicate_rows_percent": "mean",
        "outlier_count": "sum",
        "whitespace_issues": "sum"
    })

    summary.rename(columns={
        "missing_percent": "avg_missing_percent",
        "duplicate_rows_percent": "avg_duplicate_percent",
        "outlier_count": "total_outliers",
        "whitespace_issues": "total_whitespace_issues"
    }, inplace=True)

    return summary
