import pandas as pd
from .generate_meta import generate_meta
from .cleaning_recommendations import cleaning_recommendations
from .get_cleaned_data import get_cleaned_data
from .get_table_for_DQ_computation import get_table_for_DQ_computation
from .summarize_dataset_health import summarize_dataset_health
from .compute_dq_score import Compute_DQ_Score, DEFAULT_SCORE_MODE
from .adjust_prod_meta import learn_reference_profile, adjust_prod_meta_with_reference
from .preprocess_types import infer_and_fix_types

# ---------- helpers ----------
def _empty_scores():
    return pd.DataFrame(columns=["Dataset_name", "DQ_Score"])

def _empty_df():
    return pd.DataFrame()

def _empty_base_block():
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


def Run_DQ_Pipeline(
    dataset_names,
    dataset_list,
    new_dataset_list=None,
    rules=None,
    interactive=False,
    cleaning=False,
    weights=None,
    thresholds=None,
    infer_types_for_new=True,
    score_mode=DEFAULT_SCORE_MODE,
):
    """
    Run the data-quality pipeline for one or more baseline datasets and,
    optionally, a matching set of production datasets.

    Parameters are intentionally kept close to the original notebook-era API.
    The return structure is stable and always contains ``base_data`` and
    ``prod_data`` top-level keys.
    """
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

    dataset_list = [infer_and_fix_types(df) for df in dataset_list]

    if new_dataset_list is not None and infer_types_for_new:
        new_dataset_list = [infer_and_fix_types(df) for df in new_dataset_list]
    # ================= BASE BEFORE =================
    base_meta_before = generate_meta(dataset_names, dataset_list, rules=rules)
    base_recommendations = cleaning_recommendations(base_meta_before)
    base_main_metrics_before = get_table_for_DQ_computation(base_meta_before, dataset_list, dataset_names)
    base_summarized_before = summarize_dataset_health(base_meta_before)
    base_dirty_scores = pd.DataFrame(
        Compute_DQ_Score(
            dataset_names,
            dataset_list,
            base_main_metrics_before,
            weights,
            thresholds=thresholds,
            score_mode=score_mode,
        ).items(),
        columns=["Dataset_name", "DQ_Score"]
    )

    base_data = _empty_base_block()
    base_data.update({
        "dirty_scores": base_dirty_scores,
        "meta_before_cleaning": base_meta_before,
        "recommendations": base_recommendations,
        "summarized_before": base_summarized_before,
        "main_metrics_before": base_main_metrics_before
    })

    # ================= CLEAN BASE (optional) =================
    if cleaning:
        base_cleaned_list, base_change_log = get_cleaned_data(dataset_names, dataset_list, base_recommendations, rules=rules, interactive=interactive)
        base_meta_after = generate_meta(dataset_names, base_cleaned_list, rules=rules)
        base_main_metrics_after = get_table_for_DQ_computation(base_meta_after, base_cleaned_list, dataset_names)
        base_summarized_after = summarize_dataset_health(base_meta_after)
        base_cleaned_scores = pd.DataFrame(
            Compute_DQ_Score(
                dataset_names,
                base_cleaned_list,
                base_main_metrics_after,
                weights,
                thresholds=thresholds,
                score_mode=score_mode,
            ).items(),
            columns=["Dataset_name", "DQ_Score"]
        )

        base_data.update({
            "cleaned_scores": base_cleaned_scores,
            "cleaned_datasets": base_cleaned_list,
            "meta_after_cleaning": base_meta_after,
            "change_log": pd.DataFrame(base_change_log),
            "summarized_after": base_summarized_after,
            "main_metrics_after": base_main_metrics_after
        })

    # ================= NO PROD =================
    if new_dataset_list is None:
        return {"base_data": base_data, "prod_data": {}}

    # ================= PROD BEFORE =================
    reference_profile = learn_reference_profile(dataset_names, dataset_list)

    prod_meta_before = generate_meta(dataset_names,dataset_list,new_dataset_list, rules=rules)
    prod_meta_before = adjust_prod_meta_with_reference(reference_profile, base_meta_before, prod_meta_before, dataset_names, new_dataset_list)

    prod_recommendations = cleaning_recommendations(prod_meta_before)
    prod_main_metrics_before = get_table_for_DQ_computation(prod_meta_before, new_dataset_list, dataset_names)
    prod_summarized_before = summarize_dataset_health(prod_meta_before)
    prod_dirty_scores = pd.DataFrame(
        Compute_DQ_Score(
            dataset_names,
            new_dataset_list,
            prod_main_metrics_before,
            weights,
            thresholds=thresholds,
            score_mode=score_mode,
        ).items(),
        columns=["Dataset_name", "DQ_Score"]
    )

    prod_data = _empty_base_block()
    prod_data.update({
        "dirty_scores": prod_dirty_scores,
        "meta_before_cleaning": prod_meta_before,
        "recommendations": prod_recommendations,
        "summarized_before": prod_summarized_before,
        "main_metrics_before": prod_main_metrics_before
    })

    # ================= CLEAN PROD (optional) =================
    if cleaning:
        prod_cleaned_list, prod_change_log = get_cleaned_data(dataset_names, new_dataset_list, prod_recommendations, rules=rules, interactive=interactive)
        prod_meta_after = generate_meta(dataset_names,dataset_list, prod_cleaned_list, rules=rules)
        prod_meta_after = adjust_prod_meta_with_reference(reference_profile, base_meta_before, prod_meta_after, dataset_names, prod_cleaned_list)

        prod_main_metrics_after = get_table_for_DQ_computation(prod_meta_after, prod_cleaned_list, dataset_names)
        prod_summarized_after = summarize_dataset_health(prod_meta_after)
        prod_cleaned_scores = pd.DataFrame(
            Compute_DQ_Score(
                dataset_names,
                prod_cleaned_list,
                prod_main_metrics_after,
                weights,
                thresholds=thresholds,
                score_mode=score_mode,
            ).items(),
            columns=["Dataset_name", "DQ_Score"]
        )

        prod_data.update({
            "cleaned_scores": prod_cleaned_scores,
            "cleaned_datasets": prod_cleaned_list,
            "meta_after_cleaning": prod_meta_after,
            "change_log": pd.DataFrame(prod_change_log),
            "summarized_after": prod_summarized_after,
            "main_metrics_after": prod_main_metrics_after
        })

    return {"base_data": base_data, "prod_data": prod_data}
