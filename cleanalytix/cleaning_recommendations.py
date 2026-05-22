import pandas as pd


def cleaning_recommendations(meta_data_combined):
    """Convert column-level profiling metrics into ordered cleaning rule ids."""

    recs = []

    def infer_type(row):
        dtype = str(row["dtype"]).lower()

        if any(x in dtype for x in ["int", "float"]):
            return "numeric"

        elif "datetime" in dtype:
            return "datetime"

        return "categorical"

    RULE_PRIORITY = {
        7: 1,   # type fixing FIRST
        4: 2,   # whitespace
        2: 3,   # missing
        3: 4,   # outliers
        5: 5,   # categorical cleanup
        9: 6    # transformations LAST
    }

    # NEW: Add human-readable metadata
    RULE_INFO = {
        1: "Excessive Missing Values",
        2: "Missing Values",
        3: "Outliers",
        4: "Whitespace Issues",
        5: "Rare Categories",
        6: "Collapsed Date Range",
        7: "Type Errors",
        8: "Business Rule Violations",
        9: "High Skewness",
        10: "No Issues Detected"
    }

    for i, row in meta_data_combined.iterrows():
        suggestions = []
        col_type = infer_type(row)

        if row['missing_percent'] > 40:
            suggestions.append(1)

        else:
            if pd.notna(row['type_errors']) and row['type_errors'] > 0:
                suggestions.append(7)

            if col_type == "categorical":
                if pd.notna(row['whitespace_issues']) and row['whitespace_issues'] > 0:
                    suggestions.append(4)

            if pd.notna(row['rule_errors']) and row['rule_errors'] > 0:
                suggestions.append(8)

            if 0 < row['missing_percent']:
                suggestions.append(2)

            if col_type == "numeric":
                if pd.notna(row['outlier_count']) and row['outlier_count'] > 0:
                    suggestions.append(3)

            if col_type == "categorical":
                if pd.notna(row['rare_category_percent']) and row['rare_category_percent'] > 0:
                    suggestions.append(5)

            if col_type == "datetime":
                if pd.notna(row['date_range_days']) and row['unique_count'] <= 1:
                    suggestions.append(6)

            if col_type == "numeric":
                if row['skewness_level'] == 'high':
                    suggestions.append(9)

            if not suggestions:
                suggestions.append(10)

        suggestions = list(set(suggestions))
        suggestions = sorted(
            suggestions,
            key=lambda x: RULE_PRIORITY.get(x, 100)
        )

        recs.append({
            "dataset_name": row["dataset_name"],
            "column_name": row["column_name"],

            "rule_ids": suggestions,
            "inferred_type": col_type,

            "issue_names": [
                RULE_INFO.get(rule_id, "Unknown Issue")
                for rule_id in suggestions
            ],

            "recommended_fix_order": [
                RULE_PRIORITY.get(rule_id, 100)
                for rule_id in suggestions
            ]
        })

    return pd.DataFrame(recs)