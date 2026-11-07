from __future__ import annotations

import pandas as pd


def cleaning_recommendations(meta_data_combined: pd.DataFrame) -> pd.DataFrame:
    """Convert metadata rows into cleaning recommendations.

    Parameters
    ----------
    meta_data_combined : pandas.DataFrame
        Metadata table returned by :func:`generate_meta`. Required columns
        include ``dataset_name``, ``column_name``, ``dtype``,
        ``missing_percent``, ``type_errors``, ``whitespace_issues``,
        ``rule_errors``, ``outlier_count``, ``rare_category_percent``,
        ``date_range_days``, ``unique_count``, and ``skewness_level``.
        ``column_role`` is used when present to avoid treating identifiers and
        binary flags as continuous numeric columns.

    Returns
    -------
    pandas.DataFrame
        Recommendation table with one row per metadata row and the following
        columns:

        ``dataset_name``
            Dataset label.
        ``column_name``
            Column receiving recommendations.
        ``rule_ids``
            Ordered list of integer rule IDs. Current IDs are:
            ``1`` excessive missing values, ``2`` missing values, ``3``
            outliers, ``4`` whitespace issues, ``5`` rare categories, ``6``
            collapsed or invalid date range, ``7`` type errors, ``8`` business
            rule violations, ``9`` high skewness, and ``10`` no issues
            detected.
        ``inferred_type``
            Recommendation-time type bucket: ``numeric``, ``datetime``, or
            ``categorical``.
        ``issue_names``
            Human-readable names aligned with ``rule_ids``.
        ``recommended_fix_order``
            Numeric sort keys showing the order in which fixes should be
            attempted.

    Raises
    ------
    KeyError
        If required metadata columns are missing.

    Notes
    -----
    Recommendations are deterministic and rule-based. They are intended to be
    reviewable evidence for cleaning decisions, not a hidden model.

    Examples
    --------
    >>> meta = generate_meta(["customers"], [customers_df])
    >>> recs = cleaning_recommendations(meta)
    >>> recs[["column_name", "rule_ids", "issue_names"]]

    See Also
    --------
    generate_meta : Creates the metadata consumed by this function.
    get_cleaned_data : Applies recommendations to DataFrames.
    Run_DQ_Pipeline : Calls this function in the end-to-end workflow.
    """

    recs = []

    def infer_type(row):
        # Prefer explicit column role when available
        if "column_role" in row.index and pd.notna(row.get("column_role")):
            role = str(row["column_role"]).lower()
            if role == "continuous":
                return "numeric"
            if role == "datetime":
                return "datetime"
            if role in {"binary", "identifier", "categorical"}:
                return "categorical"

        # Fall back to dtype string with conservative heuristics for mixed/object types.
        dtype = str(row.get("dtype", "")).lower()

        # Numeric dtype explicit
        if any(x in dtype for x in ["int", "float", "number"]):
            return "numeric"

        # Explicit datetime dtype
        if "datetime" in dtype:
            return "datetime"

        # Heuristic: object/string columns that look mostly numeric (few type errors)
        # should be treated as numeric for recommendation purposes.
        try:
            non_null = int(row.get("non_null_count") or 0)
            type_errors = int(row.get("type_errors") or 0)
        except Exception:
            non_null = 0
            type_errors = 0

        if dtype in {"object", "string", "mixed"}:
            # If we have non-null counts and a small fraction of type errors,
            # consider the column numeric-like.
            if non_null > 0 and type_errors / non_null < 0.05:
                return "numeric"

            # If a date-range was detected in profiling, prefer datetime
            if pd.notna(row.get("date_range_days")):
                return "datetime"

            return "categorical"

        # Default conservative fallback
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
