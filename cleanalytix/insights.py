"""Deterministic insight and actionable recommendation engine for Cleanalytix.

Generates structured, advisory data quality insights and recommendations across
the 6 quality dimensions without any destructive data modification.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from cleanalytix.types import (
    InsightItem,
    InsightSeverity,
    QualityDimension,
    DimensionScore,
)


def generate_dataset_insights(
    meta_df: pd.DataFrame,
    thresholds: Dict[str, float],
    has_baseline: bool = False,
    drift_summary: Optional[Dict[str, Any]] = None,
    quality_dimensions: Optional[Dict[str, DimensionScore]] = None,
) -> List[InsightItem]:
    """Generate structured data quality insights and actionable recommendations.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Column metadata table containing profiling statistics.
    thresholds : dict
        Configured quality thresholds.
    has_baseline : bool, default False
        Whether baseline drift comparison was conducted.
    drift_summary : dict, optional
        Additional drift summary metrics.
    quality_dimensions : dict, optional
        Evaluated QualityDimension scores and metrics.

    Returns
    -------
    list of InsightItem
        Deterministic, structured list of insights sorted by severity.
    """
    insights: List[InsightItem] = []

    if meta_df.empty:
        return insights

    # 1. Dataset-level Uniqueness / Duplication check
    dup_percent = 0.0
    if "duplicate_rows_percent" in meta_df.columns:
        first_dup = meta_df["duplicate_rows_percent"].iloc[0]
        if pd.notna(first_dup):
            dup_percent = float(first_dup)

    dup_thresh = thresholds.get("duplicate_rows_percent", 1.0)
    if dup_percent > 0:
        if dup_percent >= 10.0:
            severity = InsightSeverity.CRITICAL
        elif dup_percent > dup_thresh:
            severity = InsightSeverity.WARNING
        else:
            severity = InsightSeverity.INFO

        insights.append(
            InsightItem(
                dimension=QualityDimension.UNIQUENESS,
                severity=severity,
                column=None,
                message=f"Dataset contains {dup_percent:.2f}% duplicate rows.",
                metric_name="duplicate_rows_percent",
                metric_value=dup_percent,
                threshold=dup_thresh,
                recommendation=(
                    "Verify upstream extraction and ingestion logic to eliminate duplicate records. "
                    "Confirm primary key uniqueness requirements."
                ),
            )
        )

    # 2. Check quality dimensions for Timeliness, Identifier Uniqueness, and Consistency Rules
    if quality_dimensions:
        # Identifier Uniqueness
        uniq_ds = quality_dimensions.get(QualityDimension.UNIQUENESS.value)
        if uniq_ds and uniq_ds.applicable:
            id_dup = uniq_ds.metrics.get("identifier_duplicate_percent", 0.0)
            if id_dup > 0:
                id_cols = uniq_ds.details.get("identifier_columns", [])
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.UNIQUENESS,
                        severity=InsightSeverity.CRITICAL if id_dup > 5.0 else InsightSeverity.WARNING,
                        column=", ".join(id_cols) if id_cols else None,
                        message=f"Identifier columns contain {id_dup:.2f}% duplicate key values.",
                        metric_name="identifier_duplicate_percent",
                        metric_value=id_dup,
                        threshold=dup_thresh,
                        recommendation=(
                            "Primary/identifier columns must be strictly unique. Review duplicate records "
                            "and check ingestion deduplication rules."
                        ),
                    )
                )

        # Consistency Rules
        cons_ds = quality_dimensions.get(QualityDimension.CONSISTENCY.value)
        if cons_ds and cons_ds.applicable:
            rule_details = cons_ds.details.get("rule_details", {})
            for r_name, r_info in rule_details.items():
                v_pct = r_info.get("violation_percent", 0.0)
                v_cnt = r_info.get("violations", 0)
                if v_cnt > 0:
                    insights.append(
                        InsightItem(
                            dimension=QualityDimension.CONSISTENCY,
                            severity=InsightSeverity.CRITICAL if v_pct >= 5.0 else InsightSeverity.WARNING,
                            column=None,
                            message=f"Cross-column consistency rule '{r_name}' violated by {v_cnt} rows ({v_pct:.2f}%).",
                            metric_name="consistency_rule_violations",
                            metric_value=v_pct,
                            threshold=thresholds.get("rule_errors_percent", 2.0),
                            recommendation=(
                                f"Investigate relational inconsistency violating rule '{r_name}'. "
                                "Check upstream join logic or business rule dependencies."
                            ),
                            details=r_info,
                        )
                    )

        # Timeliness
        time_ds = quality_dimensions.get(QualityDimension.TIMELINESS.value)
        if time_ds and time_ds.applicable:
            col_details = time_ds.details.get("columns", {})
            for c_name, c_info in col_details.items():
                fut_pct = c_info.get("future_dates_percent", 0.0)
                if fut_pct > 0:
                    insights.append(
                        InsightItem(
                            dimension=QualityDimension.TIMELINESS,
                            severity=InsightSeverity.CRITICAL if fut_pct >= 1.0 else InsightSeverity.WARNING,
                            column=c_name,
                            message=f"Future timestamps detected: column '{c_name}' has {fut_pct:.2f}% future dates.",
                            metric_name="future_dates_percent",
                            metric_value=fut_pct,
                            threshold=thresholds.get("future_dates_percent", 0.1),
                            recommendation=(
                                f"Verify timezone offsets or system clock synchronization generating future timestamps in '{c_name}'."
                            ),
                        )
                    )
                fresh_days = c_info.get("freshness_days", 0.0)
                fresh_thresh = thresholds.get("freshness_days", 30.0)
                if fresh_days > fresh_thresh:
                    insights.append(
                        InsightItem(
                            dimension=QualityDimension.TIMELINESS,
                            severity=InsightSeverity.WARNING,
                            column=c_name,
                            message=f"Data staleness: column '{c_name}' is {fresh_days:.1f} days behind reference time (threshold: {fresh_thresh} days).",
                            metric_name="freshness_days",
                            metric_value=fresh_days,
                            threshold=fresh_thresh,
                            recommendation=(
                                f"Ingestion pipeline for '{c_name}' appears delayed or halted. Ensure scheduled updates are executing."
                            ),
                        )
                    )

    # 2. Column-level evaluations
    for _, row in meta_df.iterrows():
        col = str(row.get("column_name", ""))
        role = str(row.get("column_role", ""))

        # Completeness: Missingness
        miss_pct = row.get("missing_percent", 0.0)
        miss_thresh = thresholds.get("missing_percent", 20.0)
        if pd.notna(miss_pct) and float(miss_pct) > 0.0:
            val = float(miss_pct)
            if val >= 50.0:
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.COMPLETENESS,
                        severity=InsightSeverity.CRITICAL,
                        column=col,
                        message=f"High missingness: column '{col}' has {val:.2f}% null values.",
                        metric_name="missing_percent",
                        metric_value=val,
                        threshold=miss_thresh,
                        recommendation=(
                            f"Column '{col}' has majority missing values. Assess whether this feature is "
                            "optional in business logic or indicates an upstream collection failure."
                        ),
                    )
                )
            elif val > miss_thresh:
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.COMPLETENESS,
                        severity=InsightSeverity.WARNING,
                        column=col,
                        message=f"Elevated missingness: column '{col}' has {val:.2f}% null values (threshold: {miss_thresh}%).",
                        metric_name="missing_percent",
                        metric_value=val,
                        threshold=miss_thresh,
                        recommendation=(
                            f"Investigate cause of missingness in '{col}'. If modeling, consider domain-aware "
                            "indicator encoding or targeted imputation."
                        ),
                    )
                )

        # Validity: Type errors
        type_err = row.get("type_errors", 0)
        if pd.notna(type_err) and int(type_err) > 0:
            n_non_null = int(row.get("non_null_count", 0))
            err_pct = (int(type_err) / max(1, n_non_null)) * 100.0 if n_non_null > 0 else 0.0
            type_thresh = thresholds.get("type_errors_percent", 2.0)
            if err_pct > type_thresh:
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.VALIDITY,
                        severity=InsightSeverity.WARNING,
                        column=col,
                        message=f"Semantic type inconsistency: {int(type_err)} values ({err_pct:.2f}%) in '{col}' violate the inferred column type.",
                        metric_name="type_errors_percent",
                        metric_value=err_pct,
                        threshold=type_thresh,
                        recommendation=(
                            f"Standardize type representations in '{col}'. Mixed string and numeric/datetime "
                            "representations lead to silent downstream analysis failures."
                        ),
                    )
                )

        # Validity: Whitespace issues (strings/categorical)
        ws_err = row.get("whitespace_issues", 0)
        if pd.notna(ws_err) and int(ws_err) > 0:
            n_non_null = int(row.get("non_null_count", 0))
            ws_pct = (int(ws_err) / max(1, n_non_null)) * 100.0 if n_non_null > 0 else 0.0
            ws_thresh = thresholds.get("whitespace_percent", 5.0)
            if ws_pct > ws_thresh:
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.VALIDITY,
                        severity=InsightSeverity.INFO,
                        column=col,
                        message=f"Unstripped whitespace: {int(ws_err)} values ({ws_pct:.2f}%) in '{col}' contain leading/trailing whitespace.",
                        metric_name="whitespace_percent",
                        metric_value=ws_pct,
                        threshold=ws_thresh,
                        recommendation=(
                            f"Apply trimming (`str.strip()`) to column '{col}' to prevent duplicate category keys."
                        ),
                    )
                )

        # Validity: Rule errors
        rule_err = row.get("rule_errors", 0)
        if pd.notna(rule_err) and int(rule_err) > 0:
            n_non_null = int(row.get("non_null_count", 0))
            rule_pct = (int(rule_err) / max(1, n_non_null)) * 100.0 if n_non_null > 0 else 0.0
            rule_thresh = thresholds.get("rule_errors_percent", 2.0)
            if rule_pct > rule_thresh:
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.VALIDITY,
                        severity=InsightSeverity.CRITICAL if rule_pct >= 10.0 else InsightSeverity.WARNING,
                        column=col,
                        message=f"Business rule violation: {int(rule_err)} values ({rule_pct:.2f}%) in '{col}' failed domain validation.",
                        metric_name="rule_errors_percent",
                        metric_value=rule_pct,
                        threshold=rule_thresh,
                        recommendation=(
                            f"Review records violating business rules in '{col}' with domain stakeholders."
                        ),
                    )
                )

        # Validity: Numeric outliers / distributional plausibility
        outliers = row.get("outlier_count", 0)
        if pd.notna(outliers) and int(outliers) > 0 and role in {"continuous", "discrete"}:
            n_non_null = int(row.get("non_null_count", 0))
            out_pct = (int(outliers) / max(1, n_non_null)) * 100.0
            out_thresh = thresholds.get("outlier_percent", 10.0)
            if out_pct > out_thresh:
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.VALIDITY,
                        severity=InsightSeverity.WARNING,
                        column=col,
                        message=f"Statistical outliers: {int(outliers)} values ({out_pct:.2f}%) in '{col}' are statistical outliers.",
                        metric_name="outlier_percent",
                        metric_value=out_pct,
                        threshold=out_thresh,
                        recommendation=(
                            f"Inspect the distribution of '{col}'. Verify whether outliers represent measurement errors "
                            "or genuine extreme events."
                        ),
                    )
                )

        # Stability: Distribution Drift (only if baseline present)
        if has_baseline:
            psi_val = row.get("psi_divergence", 0.0)
            psi_thresh = thresholds.get("psi_divergence", 0.25)
            if pd.notna(psi_val) and float(psi_val) >= 0.10:
                p_val = float(psi_val)
                severity = InsightSeverity.CRITICAL if p_val >= psi_thresh else InsightSeverity.WARNING
                insights.append(
                    InsightItem(
                        dimension=QualityDimension.STABILITY,
                        severity=severity,
                        column=col,
                        message=f"Distribution drift: PSI for '{col}' is {p_val:.4f} (threshold: {psi_thresh}).",
                        metric_name="psi_divergence",
                        metric_value=p_val,
                        threshold=psi_thresh,
                        recommendation=(
                            f"Significant distribution shift detected in '{col}'. Retrain models or update baseline "
                            "expectations if the population has shifted permanently."
                        ),
                    )
                )

    # Sort: CRITICAL first, then WARNING, then INFO, then OPTIMIZATION
    severity_order = {
        InsightSeverity.CRITICAL: 0,
        InsightSeverity.WARNING: 1,
        InsightSeverity.INFO: 2,
        InsightSeverity.OPTIMIZATION: 3,
    }
    insights.sort(key=lambda item: severity_order.get(item.severity, 99))

    return insights


def build_anomalies_dataframe(insights: List[InsightItem]) -> pd.DataFrame:
    """Format structured insight findings into a tabular anomaly DataFrame.

    Parameters
    ----------
    insights : list of InsightItem
        Detected quality anomalies and findings.

    Returns
    -------
    pd.DataFrame
        Tabular anomaly diagnostics with severity, dimension, column,
        metric, threshold, message, and recommendation.
    """
    rows = []
    for item in (insights or []):
        dim_str = item.dimension.value if isinstance(item.dimension, QualityDimension) else str(item.dimension)
        sev_str = item.severity.value if isinstance(item.severity, InsightSeverity) else str(item.severity)
        rows.append({
            "severity": sev_str,
            "dimension": dim_str,
            "column": item.column,
            "column_name": item.column,
            "metric_name": item.metric_name,
            "metric_value": item.metric_value,
            "threshold": item.threshold,
            "message": item.message,
            "recommendation": item.recommendation,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "severity", "dimension", "column", "column_name", "metric_name", "metric_value",
            "threshold", "message", "recommendation"
        ])
    return pd.DataFrame(rows)

