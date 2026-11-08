"""Pipeline orchestration module for Cleanalytix.

Provides:
1. `analyze(...)`: Modern, unified entry point returning structured `CleanalytixResult`.
2. `Run_DQ_Pipeline(...)`: Fully backward-compatible pipeline for existing workflows.

Strictly non-mutating: user data is NEVER modified, transformed, or imputed.
"""

from __future__ import annotations

from typing import Optional, Sequence, Dict, Any, List
import pandas as pd
import numpy as np

from cleanalytix.generate_meta import generate_meta
from cleanalytix.cleaning_recommendations import cleaning_recommendations, generate_recommendations
from cleanalytix.get_table_for_DQ_computation import get_table_for_DQ_computation
from cleanalytix.summarize_dataset_health import summarize_dataset_health
from cleanalytix.compute_dq_score import Compute_DQ_Score, DEFAULT_SCORE_MODE
from cleanalytix.adjust_prod_meta import learn_reference_profile, adjust_prod_meta_with_reference
from cleanalytix.preprocess_types import infer_and_fix_types
from cleanalytix.scoring import score_dataset, resolve_thresholds
from cleanalytix.insights import generate_dataset_insights, build_anomalies_dataframe
from cleanalytix.metrics.uniqueness import detect_candidate_keys, compute_duplicate_row_metrics
from cleanalytix.metrics.relationships import compute_numeric_correlations
from cleanalytix.profiling import (
    build_dataset_profile,
    build_column_profile,
    build_statistical_profile,
    build_relationship_profile,
    build_temporal_profile,
    build_drift_profile,
)
from cleanalytix.types import (
    DQPipelineResult,
    MetricWeights,
    PipelineBlock,
    Rules,
    ScoreMode,
    Thresholds,
    CleanalytixConfig,
    CleanalytixResult,
    AnalysisLevel,
    FeatureRetiredError,
    InsightSeverity,
)


def _empty_scores() -> pd.DataFrame:
    return pd.DataFrame(columns=["Dataset_name", "DQ_Score"])


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_base_block() -> PipelineBlock:
    return {
        "dirty_scores": _empty_scores(),
        "cleaned_scores": _empty_scores(),
        "cleaned_datasets": [],
        "meta_before_cleaning": _empty_df(),
        "meta_after_cleaning": _empty_df(),
        "recommendations": _empty_df(),
        "change_log": _empty_df(),
        "summarized_before": _empty_df(),
        "summarized_after": _empty_df(),
        "main_metrics_before": _empty_df(),
        "main_metrics_after": _empty_df(),
    }


def analyze(
    df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    config: Optional[CleanalytixConfig] = None,
    rules: Optional[Rules] = None,
    level: AnalysisLevel = AnalysisLevel.STANDARD,
    *,
    baseline: Optional[pd.DataFrame] = None,
) -> CleanalytixResult:
    """Analyze a dataset and produce comprehensive Data Quality Intelligence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to analyze. Input is treated as strictly read-only and never mutated.
    baseline_df : pd.DataFrame, optional
        Reference baseline dataset to evaluate against for distribution drift.
    config : CleanalytixConfig, optional
        Custom scoring thresholds, weights, and configuration.
    rules : dict of str to callable, optional
        Custom domain business rules.
    level : AnalysisLevel, default AnalysisLevel.STANDARD
        Depth of analysis: BASIC, STANDARD, or DEEP.

    Returns
    -------
    CleanalytixResult
        Structured analytical result with dimensional scores, insights, recommendations,
        and profiling metadata. Supports both object attribute access and dictionary key indexing.
    """
    if baseline_df is None and baseline is not None:
        baseline_df = baseline

    if config is None:
        config = CleanalytixConfig(rules=rules)
    elif rules is not None:
        from dataclasses import replace
        config = replace(config, rules=rules)

    dataset_name = "dataset"
    has_baseline = baseline_df is not None

    # Step 1: Generate metadata profiling & reference adjustments
    if has_baseline:
        # 1a. Profile baseline independently
        base_meta = generate_meta(
            [dataset_name],
            [baseline_df],
            rules=config.rules,
        )
        base_score, _, _, _, _, _, _ = score_dataset(
            baseline_df,
            base_meta,
            config=config,
            has_baseline=False,
            return_details=True,
        )
        base_health = summarize_dataset_health(base_meta)
        base_metrics = get_table_for_DQ_computation(base_meta, [baseline_df], [dataset_name])
        base_recs = cleaning_recommendations(base_meta)

        base_data = _empty_base_block()
        base_data.update({
            "dirty_scores": pd.DataFrame([{"Dataset_name": dataset_name, "DQ_Score": base_score}]),
            "meta_before_cleaning": base_meta,
            "recommendations": base_recs,
            "summarized_before": base_health,
            "main_metrics_before": base_metrics,
        })

        # 1b. Learn reference profile from baseline and profile production with drift
        ref_profile = learn_reference_profile([dataset_name], [baseline_df])
        prod_meta_raw = generate_meta(
            [dataset_name],
            [baseline_df],
            new_dataset_list=[df],
            rules=config.rules,
        )
        meta_df = adjust_prod_meta_with_reference(
            ref_profile,
            base_meta,
            prod_meta_raw,
            [dataset_name],
            [df],
        )
    else:
        meta_df = generate_meta(
            [dataset_name],
            [df],
            rules=config.rules,
        )
        base_data = None

    # Step 2: Scoring across Quality Dimensions with full audit ledger
    (
        overall_score,
        dim_scores,
        col_scores,
        scoring_matrix_df,
        dimension_summary_df,
        metric_inventory_df,
        scoring_audit_trail_str,
    ) = score_dataset(
        df, meta_df, config=config, has_baseline=has_baseline, return_details=True
    )

    # Step 3: Insights & Anomaly Diagnostics
    thresholds = resolve_thresholds(config.thresholds)
    insights = generate_dataset_insights(
        meta_df, thresholds, has_baseline=has_baseline, quality_dimensions=dim_scores
    )
    anomalies = build_anomalies_dataframe(insights)

    # Step 4: Structured Recommendations Engine
    recs, recs_table = generate_recommendations(meta_df, insights=insights)

    critical_count = sum(
        1 for it in insights
        if getattr(it, "severity", None) in {InsightSeverity.CRITICAL, "CRITICAL"}
    )
    warning_count = sum(
        1 for it in insights
        if getattr(it, "severity", None) in {InsightSeverity.WARNING, "WARNING"}
    )
    recommendations_count = len(recs)

    # Step 5: Domain Profiling across 10 Analytical Domains
    dataset_profile = build_dataset_profile(
        df,
        meta_df,
        overall_score=overall_score,
        dim_scores=dim_scores,
        critical_count=critical_count,
        warning_count=warning_count,
        recommendations_count=recommendations_count,
        baseline_df=baseline_df,
        dataset_name=dataset_name,
    )
    column_profile = build_column_profile(df, meta_df, col_scores)
    statistical_profile = build_statistical_profile(df, meta_df)
    relationships = build_relationship_profile(df, config=config)
    temporal_profile = build_temporal_profile(df, meta_df, config=config)
    drift_profile = build_drift_profile(df, baseline_df, meta_df)

    # Step 6: Standardized Metrics Table & Health Summary
    metrics_table = get_table_for_DQ_computation(meta_df, [df], [dataset_name])
    health_summary = summarize_dataset_health(meta_df)

    # Step 7: Legacy Pipeline Block Compatibility
    dup_row_info = compute_duplicate_row_metrics(df)
    key_info = relationships.candidate_keys
    rel_info = compute_numeric_correlations(df) if level in {AnalysisLevel.STANDARD, AnalysisLevel.DEEP} else {}

    drift_summary = {
        "has_baseline": has_baseline,
        "candidate_keys": key_info,
        "duplicate_rows": dup_row_info,
        "relationships": rel_info,
    }

    if not has_baseline:
        base_data = _empty_base_block()
        base_data.update({
            "dirty_scores": pd.DataFrame([{"Dataset_name": dataset_name, "DQ_Score": overall_score}]),
            "meta_before_cleaning": meta_df,
            "recommendations": cleaning_recommendations(meta_df),
            "summarized_before": health_summary,
            "main_metrics_before": metrics_table,
        })
        prod_data = {}
    else:
        prod_data = _empty_base_block()
        prod_data.update({
            "dirty_scores": pd.DataFrame([{"Dataset_name": dataset_name, "DQ_Score": overall_score}]),
            "meta_before_cleaning": meta_df,
            "recommendations": cleaning_recommendations(meta_df),
            "summarized_before": health_summary,
            "main_metrics_before": metrics_table,
        })

    # Step 8: Build Comprehensive CleanalytixResult
    result = CleanalytixResult(
        overall_score=overall_score,
        quality_dimensions=dim_scores,
        column_scores=col_scores,
        dataset_profile=dataset_profile,
        column_profile=column_profile,
        statistical_profile=statistical_profile,
        scoring_matrix=scoring_matrix_df,
        dimension_summary=dimension_summary_df,
        metrics_table=metrics_table,
        metric_inventory=metric_inventory_df,
        relationships=relationships,
        temporal_profile=temporal_profile,
        drift_profile=drift_profile,
        anomalies=anomalies,
        recommendations_table=recs_table,
        recommendations=recs,
        insights=insights,
        scoring_audit_trail=scoring_audit_trail_str,
        health_summary=health_summary,
        metadata=meta_df,
        drift_summary=drift_summary,
        base_data=base_data,
        prod_data=prod_data,
    )
    return result


def Run_DQ_Pipeline(
    dataset_names: Sequence[str],
    dataset_list: Sequence[pd.DataFrame],
    new_dataset_list: Optional[Sequence[pd.DataFrame]] = None,
    rules: Optional[Rules] = None,
    interactive: bool = False,
    cleaning: bool = False,
    weights: Optional[MetricWeights] = None,
    thresholds: Optional[Thresholds] = None,
    infer_types: bool = False,
    infer_types_for_new: bool = False,
    score_mode: ScoreMode = DEFAULT_SCORE_MODE,
) -> DQPipelineResult:
    """Run the Cleanalytix data-quality profiling, scoring, and analysis pipeline.

    Parameters
    ----------
    dataset_names : sequence of str
        Dataset names.
    dataset_list : sequence of pandas.DataFrame
        Baseline/reference datasets.
    new_dataset_list : sequence of pandas.DataFrame, optional
        Production or comparison datasets.
    rules : mapping of str to callable, optional
        Custom business validation rules.
    interactive : bool, default=False
        Unused in non-mutating library.
    cleaning : bool, default=False
        If True, raises `FeatureRetiredError` because automated mutation has been retired.
    weights : sequence of float, optional
        Scoring metric weights.
    thresholds : mapping of str to float, optional
        Metric thresholds for scoring.
    infer_types : bool, default=False
        If True, passes a copy through type diagnosis/inference for profiling.
    infer_types_for_new : bool, default=False
        If True, passes production copy through type inference.
    score_mode : {"linear", "exponential"}, default="linear"
        Scoring mode.

    Returns
    -------
    DQPipelineResult
        Pipeline result dictionary with `base_data` and optional `prod_data`.
    """
    if cleaning:
        raise FeatureRetiredError(
            "Automatic data cleaning (cleaning=True) has been retired in Cleanalytix 0.2.0. "
            "Cleanalytix is now a strictly non-mutating Data Quality Intelligence library. "
            "To inspect quality issues and remediation recommendations, run the pipeline with "
            "cleaning=False or use the modern cleanalytix.analyze() API."
        )

    rules = {} if rules is None else dict(rules)

    if len(dataset_names) != len(dataset_list):
        raise ValueError(
            f"dataset_names has {len(dataset_names)} entries but dataset_list has "
            f"{len(dataset_list)}. They must be the same length."
        )

    if new_dataset_list is not None and len(new_dataset_list) != len(dataset_list):
        raise ValueError(
            f"new_dataset_list has {len(new_dataset_list)} entries but dataset_list has "
            f"{len(dataset_list)}. They must be the same length."
        )

    # Non-mutating type copies if requested
    eval_dataset_list = dataset_list
    if infer_types:
        eval_dataset_list = [infer_and_fix_types(df) for df in dataset_list]

    eval_new_dataset_list = new_dataset_list
    if new_dataset_list is not None and infer_types and infer_types_for_new:
        eval_new_dataset_list = [infer_and_fix_types(df) for df in new_dataset_list]

    # ================= BASE BEFORE =================
    base_meta_before = generate_meta(dataset_names, eval_dataset_list, rules=rules)
    base_recommendations = cleaning_recommendations(base_meta_before)
    base_main_metrics_before = get_table_for_DQ_computation(
        base_meta_before, eval_dataset_list, dataset_names
    )
    base_summarized_before = summarize_dataset_health(base_meta_before)
    base_dirty_scores = pd.DataFrame(
        Compute_DQ_Score(
            dataset_names,
            eval_dataset_list,
            base_main_metrics_before,
            weights,
            thresholds=thresholds,
            score_mode=score_mode,
        ).items(),
        columns=["Dataset_name", "DQ_Score"],
    )

    base_data = _empty_base_block()
    base_data.update({
        "dirty_scores": base_dirty_scores,
        "meta_before_cleaning": base_meta_before,
        "recommendations": base_recommendations,
        "summarized_before": base_summarized_before,
        "main_metrics_before": base_main_metrics_before,
    })

    # ================= NO PROD =================
    if new_dataset_list is None:
        return {"base_data": base_data, "prod_data": {}}

    # ================= PROD BEFORE =================
    reference_profile = learn_reference_profile(dataset_names, eval_dataset_list)

    prod_meta_before = generate_meta(
        dataset_names, eval_dataset_list, eval_new_dataset_list, rules=rules
    )
    prod_meta_before = adjust_prod_meta_with_reference(
        reference_profile,
        base_meta_before,
        prod_meta_before,
        dataset_names,
        eval_new_dataset_list,
    )

    prod_recommendations = cleaning_recommendations(prod_meta_before)
    prod_main_metrics_before = get_table_for_DQ_computation(
        prod_meta_before, eval_new_dataset_list, dataset_names
    )
    prod_summarized_before = summarize_dataset_health(prod_meta_before)
    prod_dirty_scores = pd.DataFrame(
        Compute_DQ_Score(
            dataset_names,
            eval_new_dataset_list,
            prod_main_metrics_before,
            weights,
            thresholds=thresholds,
            score_mode=score_mode,
        ).items(),
        columns=["Dataset_name", "DQ_Score"],
    )

    prod_data = _empty_base_block()
    prod_data.update({
        "dirty_scores": prod_dirty_scores,
        "meta_before_cleaning": prod_meta_before,
        "recommendations": prod_recommendations,
        "summarized_before": prod_summarized_before,
        "main_metrics_before": prod_main_metrics_before,
    })

    return {"base_data": base_data, "prod_data": prod_data}
