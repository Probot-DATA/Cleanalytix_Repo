"""
Cleanalytix - Data Quality Framework.

Quick start
-----------
>>> from cleanalytix import Run_DQ_Pipeline
>>> result = Run_DQ_Pipeline(["my_dataset"], [df])
>>> print(result["base_data"]["dirty_scores"])
"""

from .version import __author__, __version__
from .pipeline import Run_DQ_Pipeline
from .compute_dq_score import Compute_DQ_Score, DEFAULT_THRESHOLDS
from .generate_meta import generate_meta
from .cleaning_recommendations import cleaning_recommendations
from .get_cleaned_data import get_cleaned_data
from .get_table_for_DQ_computation import get_table_for_DQ_computation
from .summarize_dataset_health import summarize_dataset_health
from .adjust_prod_meta import learn_reference_profile, adjust_prod_meta_with_reference
from .preprocess_types import infer_and_fix_types

run_dq_pipeline = Run_DQ_Pipeline

__all__ = [
    "Run_DQ_Pipeline",
    "run_dq_pipeline",
    "Compute_DQ_Score",
    "DEFAULT_THRESHOLDS",
    "__author__",
    "__version__",
    "generate_meta",
    "cleaning_recommendations",
    "get_cleaned_data",
    "get_table_for_DQ_computation",
    "summarize_dataset_health",
    "learn_reference_profile",
    "adjust_prod_meta_with_reference",
    "infer_and_fix_types",
]
