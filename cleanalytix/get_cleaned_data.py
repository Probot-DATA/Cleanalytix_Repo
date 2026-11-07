"""Retired automatic cleaning subsystem for Cleanalytix.

NOTICE: The automatic data modification and cleaning engine was retired in
Cleanalytix 0.2.0. Cleanalytix is now a strictly non-mutating Data Quality
Intelligence and Advanced Analytics library.

Cleanalytix analyzes datasets, computes dimensional scores, detects anomalies,
and provides evidence-based recommendations, but it NEVER modifies, imputes,
deduplicates, or deletes the user's data.

To remediate data, inspect ``result['insights']`` or ``result['recommendations']``
and apply domain-appropriate transformations under explicit user control.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import pandas as pd

from .types import ChangeLogEntry, FeatureRetiredError, Rules, SemanticType


def clean(
    df: pd.DataFrame,
    column: str,
    clean_type: int,
    replacement: Any = None,
    rules: Optional[Rules] = None,
    interactive: bool = False,
    semantic_type: Optional[SemanticType] = None,
) -> Optional[pd.Series]:
    """Retired in Cleanalytix 0.2.0.

    Raises
    ------
    FeatureRetiredError
        Always raised. Automatic data mutation has been retired.
    """
    raise FeatureRetiredError(
        "cleanalytix.clean() was retired in version 0.2.0. "
        "Cleanalytix is now a strictly non-mutating Data Quality Intelligence library. "
        "To inspect anomalies and recommendations without modifying data, use "
        "cleanalytix.Run_DQ_Pipeline() or cleanalytix.analyze() and inspect "
        "result['insights'] or result['recommendations']."
    )


def get_cleaned_data(
    dataset_name_list: Sequence[str],
    dataset_list: Sequence[pd.DataFrame],
    recommendations: pd.DataFrame,
    rules: Optional[Rules] = None,
    interactive: bool = False,
) -> tuple[list[pd.DataFrame], list[ChangeLogEntry]]:
    """Retired in Cleanalytix 0.2.0.

    Raises
    ------
    FeatureRetiredError
        Always raised. Automatic data mutation has been retired.
    """
    raise FeatureRetiredError(
        "cleanalytix.get_cleaned_data() was retired in version 0.2.0. "
        "Cleanalytix is now a strictly non-mutating Data Quality Intelligence library. "
        "Automatic row deletion, imputation, and column drops are no longer supported. "
        "Use cleanalytix.Run_DQ_Pipeline() to obtain non-mutating profiling and insights."
    )


def infer_semantic_type(
    col: pd.Series,
    numeric_threshold: float = 0.95,
    datetime_threshold: float = 0.95,
    column_name: Optional[str] = None,
) -> SemanticType:
    """Diagnostic semantic type helper retained for backwards compatibility."""
    from .column_role import infer_column_role

    role = infer_column_role(col, column_name, len(col))
    if role in {"continuous", "discrete"}:
        return "numeric"
    if role == "datetime":
        return "datetime"
    if role == "binary":
        return "binary"
    if role == "identifier":
        return "identifier"
    if role == "free_text":
        return "text"
    return "categorical"
