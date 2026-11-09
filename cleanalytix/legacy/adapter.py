"""One-way LegacyMetaAdapter projecting ProfileContext into the 32-column legacy meta_df.

Architectural Contract
----------------------
- Strictly one-way: ``ProfileContext -> pd.DataFrame``. No reverse adapter exists.
- Zero recomputation: Reads exclusively from pre-computed domain objects on
  ``ProfileContext`` without touching raw DataFrames or re-running metric functions.
- Preserves exact 32-column schema, column ordering, dtypes, and ``pd.NaT -> np.nan``
  conventions of legacy ``generate_meta()``.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Union
import numpy as np
import pandas as pd

from cleanalytix.types import ProfileContext


LEGACY_BASE_META_COLUMNS: List[str] = [
    "dataset_name",
    "column_name",
    "column_role",
    "dtype",
    "non_null_count",
    "missing_count",
    "missing_percent",
    "unique_count",
    "unique_percent",
    "mode",
    "mode_percent",
    "mean",
    "std",
    "skewness",
    "skewness_level",
    "min",
    "max",
    "median",
    "outlier_count",
    "num_categories",
    "rare_category_percent",
    "whitespace_issues",
    "min_date",
    "max_date",
    "date_range_days",
    "Entropy",
    "type_errors",
    "rule_errors",
    "duplicate_rows_percent",
]

LEGACY_DRIFT_META_COLUMNS: List[str] = [
    "kl_divergence",
    "psi_divergence",
    "js_divergence",
]

LEGACY_META_COLUMNS: List[str] = LEGACY_BASE_META_COLUMNS + LEGACY_DRIFT_META_COLUMNS


class LegacyMetaAdapter:
    """One-way adapter converting ``ProfileContext`` into the 32-column legacy ``meta_df``."""

    @staticmethod
    def context_to_meta_df(
        context: Union[ProfileContext, Sequence[ProfileContext]],
    ) -> pd.DataFrame:
        """Project one or more ``ProfileContext`` instances into the 32-column legacy ``meta_df``.

        Parameters
        ----------
        context : ProfileContext or Sequence[ProfileContext]
            Populated ``ProfileContext`` (or sequence of ``ProfileContext`` objects
            for multi-dataset legacy calls).

        Returns
        -------
        pd.DataFrame
            32-column metadata DataFrame matching the legacy ``generate_meta()`` schema.

        Raises
        ------
        TypeError
            If ``context`` is not a ``ProfileContext`` or sequence of ``ProfileContext``.
        """
        if isinstance(context, ProfileContext):
            contexts: List[ProfileContext] = [context]
        elif isinstance(context, Sequence) and not isinstance(context, (str, bytes)):
            contexts = list(context)
            for item in contexts:
                if not isinstance(item, ProfileContext):
                    raise TypeError(
                        f"Expected ProfileContext in sequence, got {type(item).__name__}."
                    )
        else:
            raise TypeError(
                f"Expected ProfileContext or Sequence[ProfileContext], got {type(context).__name__}."
            )

        base_rows: List[List[Any]] = []
        drift_rows: List[List[Any]] = []

        for ctx in contexts:
            dataset_name = ctx.dataset.dataset_name
            total_rows = ctx.dataset.total_rows
            if total_rows > 0:
                dup_row_pct = float((ctx.dataset.duplicate_rows_count / total_rows) * 100.0)
            else:
                dup_row_pct = float(ctx.dataset.duplicate_rows_percent)

            for col in ctx.column_profiles:
                role = str(col.role)
                if getattr(col, "dtype_object", None) is not None:
                    dtype_obj = col.dtype_object
                else:
                    try:
                        dtype_obj = pd.api.types.pandas_dtype(col.dtype)
                    except Exception:
                        dtype_obj = col.dtype

                comp = col.completeness
                non_null_count = int(comp.non_null_count) if comp is not None else 0
                missing_count = int(comp.null_count) if comp is not None else 0
                missing_percent = float(comp.null_percentage) if comp is not None else 0.0

                uniq = col.uniqueness
                unique_count = int(uniq.distinct_count) if uniq is not None else 0
                unique_percent = float(uniq.unique_percentage) if uniq is not None else 0.0

                if uniq is not None and uniq.mode is not None:
                    mode_val = uniq.mode
                elif uniq is not None:
                    mode_val = uniq.top_value
                else:
                    mode_val = None

                if uniq is not None and uniq.mode_percentage is not None:
                    mode_pct = float(uniq.mode_percentage)
                elif uniq is not None and mode_val is not None:
                    mode_pct = float(
                        (uniq.top_frequency_ratio * non_null_count / max(1, total_rows)) * 100.0
                    )
                else:
                    mode_pct = None

                val = col.validity
                type_errors = int(val.type_error_count) if val is not None else 0
                rule_errors = int(val.rule_error_count) if val is not None else 0

                stat = col.statistical
                dist = col.distribution
                temp = col.temporal
                drift = col.drift

                mean_val = None
                std_val = None
                min_val = None
                max_val = None
                median_val = None
                skewness_val = None
                skewness_level = None
                outliers_val = None
                num_categories = None
                rare_category_percent = None
                whitespace_issues = None
                min_date = None
                max_date = None
                date_range_days = None

                is_num_branch = (
                    role in {"continuous", "discrete"}
                    and stat is not None
                    and pd.api.types.is_numeric_dtype(dtype_obj)
                    and not pd.api.types.is_bool_dtype(dtype_obj)
                )
                is_dt_branch = (
                    role == "datetime"
                    or pd.api.types.is_datetime64_any_dtype(dtype_obj)
                )

                if is_num_branch:
                    if non_null_count > 0:
                        mean_val = stat.mean
                        std_val = stat.std
                        min_val = stat.min
                        max_val = stat.max
                        median_val = stat.median
                        skewness_val = stat.skewness
                        if skewness_val is not None:
                            abs_skew = abs(skewness_val)
                            if abs_skew < 0.5:
                                skewness_level = "low"
                            elif abs_skew < 1.0:
                                skewness_level = "moderate"
                            else:
                                skewness_level = "high"
                        outliers_val = int(stat.outlier_count)
                elif role in {"binary", "identifier"}:
                    num_categories = unique_count
                    outliers_val = 0
                    rare_category_percent = (
                        float(dist.rare_categories_percentage) if dist is not None else 0.0
                    )
                    whitespace_issues = int(val.whitespace_count) if val is not None else 0
                elif is_dt_branch:
                    if temp is not None:
                        min_date = (
                            temp.min_timestamp
                            if temp.min_timestamp is not None
                            else temp.min_date
                        )
                        max_date = (
                            temp.max_timestamp
                            if temp.max_timestamp is not None
                            else temp.max_date
                        )
                        date_range_days = temp.date_range_days
                else:
                    num_categories = unique_count
                    rare_category_percent = (
                        float(dist.rare_categories_percentage) if dist is not None else 0.0
                    )
                    whitespace_issues = int(val.whitespace_count) if val is not None else 0

                entropy_val = float(col.entropy) if col.entropy is not None else 0.0

                base_rows.append([
                    dataset_name,
                    col.name,
                    role,
                    dtype_obj,
                    non_null_count,
                    missing_count,
                    missing_percent,
                    unique_count,
                    unique_percent,
                    mode_val,
                    mode_pct,
                    mean_val,
                    std_val,
                    skewness_val,
                    skewness_level,
                    min_val,
                    max_val,
                    median_val,
                    outliers_val,
                    num_categories,
                    rare_category_percent,
                    whitespace_issues,
                    min_date,
                    max_date,
                    date_range_days,
                    entropy_val,
                    type_errors,
                    rule_errors,
                    dup_row_pct,
                ])

                kl_val = float(drift.kl_divergence) if drift is not None else 0.0
                psi_val = float(drift.psi) if drift is not None else 0.0
                js_val = float(drift.js_divergence) if drift is not None else 0.0
                drift_rows.append([dataset_name, col.name, kl_val, psi_val, js_val])

        meta_data = pd.DataFrame(data=base_rows, columns=LEGACY_BASE_META_COLUMNS)
        meta_data.replace({pd.NaT: np.nan}, inplace=True)

        meta_table_adv = pd.DataFrame(
            data=drift_rows,
            columns=["dataset_name", "column_name"] + LEGACY_DRIFT_META_COLUMNS,
        )

        if len(base_rows) == 0:
            return pd.DataFrame(columns=LEGACY_META_COLUMNS)

        meta_data["kl_divergence"] = meta_table_adv["kl_divergence"].values
        meta_data["psi_divergence"] = meta_table_adv["psi_divergence"].values
        meta_data["js_divergence"] = meta_table_adv["js_divergence"].values
        return meta_data
