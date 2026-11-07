"""Semantic type inference and coercive dtype stabilization for Cleanalytix.

This module separates semantic inference from physical pandas conversion.

The inference engine combines multiple column-level signals instead of
treating a single parseability threshold as proof of a semantic type. It is
conservative during identification but aggressive during conversion: once a
semantic type is confidently identified, values that cannot be parsed as that
type are intentionally coerced to NaN/NaT/pd.NA rather than rejecting the
entire conversion.

This is HARD/COERCIVE type inference. Ambiguous cases (e.g., ambiguous
numeric dates) are still left unchanged because the intended interpretation
cannot be determined. But for confidently identified types, invalid values
become missing rather than blocking the conversion.

The public API remains ``infer_and_fix_types(df) -> DataFrame`` so the existing
Cleanalytix pipeline does not need to change. When ``infer_types=False`` in the
pipeline, this function is not called and normal Pandas dtypes remain.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

# Import dateparser for hard datetime normalization. Missing dateparser raises
# explicitly when datetime conversion is needed.
try:
    import dateparser as _dateparser
    HAS_DATEPARSER = True
except ImportError:
    HAS_DATEPARSER = False
    _dateparser = None



# ---------------------------------------------------------------------------
# Internal vocabulary
# ---------------------------------------------------------------------------


_STRONG_ID_TOKENS = {
    "id",
    "identifier",
    "uuid",
    "guid",
}

_ID_SUFFIXES = {
    "id",
    "uuid",
    "guid",
}

# Only explicit date-bearing tokens are strong date-name evidence.
# Generic words such as "start", "end", and "time" are intentionally excluded.
_DATE_TOKENS = {
    "date",
    "datetime",
    "timestamp",
    "dob",
    "birthdate",
    "created_date",
    "updated_date",
    "modified_date",
    "expiry_date",
    "expiration_date",
}

_NUMERIC_TOKENS = {
    "age",
    "amount",
    "price",
    "cost",
    "income",
    "salary",
    "revenue",
    "quantity",
    "count",
    "score",
    "rate",
    "ratio",
    "percentage",
    "percent",
    "weight",
    "height",
    "length",
    "distance",
    "duration",
    "latitude",
    "longitude",
}

_BOOLEAN_TOKENS = {
    "active",
    "enabled",
    "valid",
    "verified",
    "deleted",
    "available",
    "approved",
    "completed",
}

_BOOLEAN_PREFIXES = {
    "is",
    "has",
    "was",
    "can",
    "should",
}

_CATEGORICAL_TOKENS = {
    "category",
    "type",
    "class",
    "status",
    "country",
    "state",
    "city",
    "gender",
    "segment",
    "group",
    "department",
    "region",
    "level",
}

_TRUE_VALUES = {
    "true",
    "yes",
    "y",
    "t",
    "on",
    "1",
}

_FALSE_VALUES = {
    "false",
    "no",
    "n",
    "f",
    "off",
    "0",
}

# Exact boolean-like textual domains.
_BOOLEAN_VALUES = _TRUE_VALUES | _FALSE_VALUES

# Date shapes are evidence only. They do not establish the intended
# day/month ordering when a representation is ambiguous.
#
# Numeric dates are split into:
#   1. year-first forms: YYYY-MM-DD (unambiguous ordering)
#   2. day/month/year forms: DD-MM-YYYY or MM-DD-YYYY, which are ambiguous
#      whenever both first and second components are <= 12.
# Matches a numeric date either by itself or at the start of a valid
# timestamp representation. The captured date components remain groups 1-4.
#
# Examples:
#   01/02/2024
#   01/02/2024 10:30:00
#   01/02/2024T10:30:00
#   13-02-2024 10:00
#   2024-01-02T10:30:00Z
# Matches a numeric date either by itself or at the start of a supported
# timestamp representation. This intentionally covers only the timestamp
# forms handled by this module; it does not claim to recognize every valid
# timestamp syntax. The captured date components remain groups 1-4.
_NUMERIC_DATE_PATTERN = re.compile(
    r"^"
    r"(\d{1,4})([-/.])(\d{1,2})\2(\d{1,4})"
    r"(?:"
    r"$"
    r"|"
    r"[Tt\s]+"
    r"\d{1,2}:\d{2}"
    r"(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?"
    r"$"
    r")"
)

# Textual dates require a four-digit year so automatic conversion does not
# silently choose a century for values such as ``01 Feb 24``.
_TEXT_DATE_PATTERN = re.compile(
    r"^(?:"
    r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"
    r"|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
    r")"
    r"(?:[Tt\s]+\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?)?$"
)

_ID_PATTERN = re.compile(
    r"^(?:"
    r"[A-Za-z]{1,8}[-_]?\d{2,}"
    r"|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,}"
    r")$"
)

_CURRENCY_PREFIX = re.compile(r"^[₹$€£]\s*")


_VALID_THOUSANDS_PATTERN = re.compile(
    r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$"
)

_PLAIN_NUMERIC_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
)


_LEADING_ZERO_INTEGER_PATTERN = re.compile(
    r"^[+-]?0\d+$"
)


def _has_meaningful_leading_zero(series: pd.Series) -> bool:
    """Return True if any non-null textual value has meaningful leading zeros.

    The check uses the same validated numeric-format grammar as numeric
    conversion. Thus ``00123``, ``01,234``, and ``000,123`` are protected,
    while an ordinary thousands-formatted number such as ``1,234`` is not.
    The check examines the full source column.
    """
    if len(series) == 0:
        return False

    if (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        return False

    values = series.dropna().astype("string").str.strip()

    if len(values) == 0:
        return False

    # Reuse the numeric normalization grammar rather than checking only the
    # raw token. This catches comma-formatted leading-zero representations
    # without weakening malformed-comma validation.
    normalized = _normalise_numeric_strings(values)

    return bool(
        normalized.str.fullmatch(
            _LEADING_ZERO_INTEGER_PATTERN,
            na=False,
        ).any()
    )


def _normalise_numeric_strings(values: pd.Series) -> pd.Series:
    """Normalize numeric formatting without accepting malformed commas.

    Currency symbols may be removed as representation-level formatting.
    Commas are removed only when they form valid Western thousands
    separators. Indian-style grouping such as ``1,23,456`` is intentionally
    unsupported without an explicit locale policy.
    Tokens such as ``12,34`` or ``123,45,6`` remain non-numeric.
    """
    strings = values.astype("string").str.strip()

    currency_cleaned = strings.str.replace(
        _CURRENCY_PREFIX,
        "",
        regex=True,
    )

    has_comma = currency_cleaned.str.contains(
        ",",
        regex=False,
        na=False,
    )

    valid_thousands = currency_cleaned.str.fullmatch(
        _VALID_THOUSANDS_PATTERN,
        na=False,
    )

    plain_numeric = currency_cleaned.str.fullmatch(
        _PLAIN_NUMERIC_PATTERN,
        na=False,
    )

    valid_numeric = (
        valid_thousands
        | (
            ~has_comma
            & plain_numeric
        )
    )

    cleaned = currency_cleaned.where(
        valid_numeric,
        other=pd.NA,
    )

    return cleaned.str.replace(
        ",",
        "",
        regex=False,
    )



# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _DateEvidence:
    parse_ratio: float
    pattern_ratio: float
    ambiguous_ratio: float
    unambiguous_pattern_ratio: float
    format_consistency: float
    dmy_ratio: float
    mdy_ratio: float
    mixed_numeric_conventions: bool


@dataclass(frozen=True)
class _ColumnEvidence:
    name: str
    tokens: tuple[str, ...]
    original_dtype: str

    n_rows: int
    n_non_null: int
    n_unique: int
    unique_ratio: float

    numeric_ratio: float
    datetime_ratio: float
    boolean_ratio: float

    integer_like_ratio: float
    string_ratio: float

    id_name_score: float
    date_name_score: float
    numeric_name_score: float
    categorical_name_score: float
    boolean_name_score: float

    id_pattern_ratio: float
    date_pattern_ratio: float

    date: _DateEvidence

    is_monotonic: bool
    is_unique: bool
    has_two_distinct_values: bool

    is_known_binary_numeric: bool
    is_potential_binary_numeric: bool


@dataclass(frozen=True)
class _Inference:
    """Internal semantic decision.

    ``decision_strength`` is a relative evidence measure, not a probability.
    """

    role: str
    decision_strength: float
    scores: dict[str, float]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _normalise_column_name(name: Any) -> list[str]:
    text = str(name).strip()

    # Split camelCase before normalizing separators.
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower().replace("%", " percent ")

    tokens = re.split(r"[^a-z0-9]+", text)
    return [token for token in tokens if token]


def _safe_ratio(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())


def _sample_values(series: pd.Series, max_values: int = 5000) -> pd.Series:
    """Return a deterministic sample using positions, not index labels.

    The sample combines head, tail, and evenly spaced observations. This
    avoids using a potentially non-unique DataFrame index to deduplicate
    legitimate observations.
    """
    values = series.dropna()

    if len(values) <= max_values:
        return values

    head_n = min(500, max_values // 5)
    tail_n = min(500, max_values // 5)
    middle_n = max_values - head_n - tail_n

    positions: list[int] = []

    positions.extend(range(head_n))
    positions.extend(range(len(values) - tail_n, len(values)))

    if middle_n > 0:
        middle_positions = np.linspace(
            head_n,
            len(values) - tail_n - 1,
            middle_n,
            dtype=int,
        )
        positions.extend(middle_positions.tolist())

    positions = sorted(set(
        position
        for position in positions
        if 0 <= position < len(values)
    ))

    # iloc is positional, so duplicate DataFrame index labels are irrelevant.
    return values.iloc[positions]


def _token_score(tokens: list[str], vocabulary: set[str]) -> float:
    if not tokens:
        return 0.0

    matches = sum(token in vocabulary for token in tokens)
    return min(1.0, matches / len(tokens))


def _has_id_suffix(tokens: list[str]) -> bool:
    return bool(tokens) and tokens[-1] in _ID_SUFFIXES


def _has_boolean_prefix(tokens: list[str]) -> bool:
    return bool(tokens) and tokens[0] in _BOOLEAN_PREFIXES


# ---------------------------------------------------------------------------
# Numeric evidence
# ---------------------------------------------------------------------------

def _numeric_conversion_candidate(
    series: pd.Series,
) -> tuple[pd.Series, float]:
    """Parse numeric representations without silently changing malformed data."""
    values = _sample_values(series)

    if len(values) == 0:
        return pd.Series(dtype="float64"), 0.0

    if (
        pd.api.types.is_numeric_dtype(values)
        and not pd.api.types.is_bool_dtype(values)
    ):
        converted = pd.to_numeric(values, errors="coerce")
        return converted, _safe_ratio(converted.notna())

    # This representation-changing decision uses the full source column so a
    # rare leading-zero code outside the inference sample cannot slip through.
    if _has_meaningful_leading_zero(series):
        return pd.Series(
            pd.NA,
            index=values.index,
            dtype="object",
        ), 0.0

    cleaned = _normalise_numeric_strings(values)

    converted = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    return converted, _safe_ratio(converted.notna())


def _numeric_parse_ratio(series: pd.Series) -> float:
    _, ratio = _numeric_conversion_candidate(series)
    return ratio


def _integer_like_ratio(series: pd.Series) -> float:
    """Estimate integer-like numeric structure from the supplied sample.

    This is an inference-only metric. The supplied Series is already the
    inference sample, so it performs sample-local numeric parsing directly
    rather than invoking a conversion safety path intended for full columns.
    """
    values = _sample_values(series)

    if len(values) == 0:
        return 0.0

    if pd.api.types.is_integer_dtype(values):
        return _safe_ratio(values.notna())

    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
    else:
        cleaned = _normalise_numeric_strings(values)
        numeric = pd.to_numeric(
            cleaned,
            errors="coerce",
        )

    valid = numeric.notna()
    if not valid.any():
        return 0.0

    integer_like = valid & numeric.mod(1).eq(0)

    return float(integer_like.mean())


def _boolean_parse_ratio(series: pd.Series) -> float:
    values = _sample_values(series)

    if len(values) == 0:
        return 0.0

    if pd.api.types.is_bool_dtype(values):
        return 1.0

    if pd.api.types.is_numeric_dtype(values):
        unique_values = set(values.dropna().tolist())

        if unique_values and unique_values.issubset({0, 1}):
            return 1.0

        return 0.0

    normalised = values.astype("string").str.strip().str.lower()

    return float(normalised.isin(_BOOLEAN_VALUES).mean())


def _is_known_binary_numeric(series: pd.Series) -> bool:
    """Return True only when the FULL column is contained in {0, 1}."""
    if len(series) == 0:
        return False

    if not pd.api.types.is_numeric_dtype(series):
        return False

    if pd.api.types.is_bool_dtype(series):
        return True

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    # This intentionally scans the full non-null column. Binary semantics are
    # too consequential to infer from a sample when a rare third value could
    # exist outside that sample.
    return bool(non_null.isin([0, 1]).all())


def _is_potential_binary_numeric(series: pd.Series) -> bool:
    """Identify two-state numeric data without declaring it semantically binary."""
    values = _sample_values(series)

    if len(values) == 0:
        return False

    if not pd.api.types.is_numeric_dtype(values):
        return False

    if pd.api.types.is_bool_dtype(values):
        return False

    return len(set(values.dropna().tolist())) == 2


# ---------------------------------------------------------------------------
# Datetime evidence
# ---------------------------------------------------------------------------


def _extract_numeric_date_components(
    series: pd.Series,
) -> pd.DataFrame:
    """Extract numeric date components from dates or valid timestamps."""
    values = series.dropna().astype("string").str.strip()

    if len(values) == 0:
        return pd.DataFrame(
            columns=[0, 1, 2, 3],
            index=values.index,
        )

    return values.str.extract(
        _NUMERIC_DATE_PATTERN,
        expand=True,
    )


def _has_short_numeric_date_year(series: pd.Series) -> bool:
    """Return True if any full-column non-year-first numeric date has a
    shortened final year.

    Non-year-first numeric dates require exactly four digits in the final
    component. A one-, two-, or three-digit final component lacks an explicit
    century policy and is therefore unsafe for automatic datetime conversion.
    Year-first values such as ``2024-02-01`` remain allowed because their
    first component is the four-digit year.
    """
    if len(series) == 0:
        return False

    if (
        pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_numeric_dtype(series)
    ):
        return False

    matches = _extract_numeric_date_components(series)
    numeric_shape = matches[0].notna()

    if not numeric_shape.any():
        return False

    first_text = matches.loc[numeric_shape, 0].astype("string")
    final_text = matches.loc[numeric_shape, 3].astype("string")

    first_is_year = (
        first_text.str.len().eq(4)
        & first_text.str.match(r"^\d{4}$", na=False)
    )

    final_is_four_digit = (
        final_text.str.len().eq(4)
        & final_text.str.match(r"^\d{4}$", na=False)
    )

    return bool((~first_is_year & ~final_is_four_digit).any())




def _has_ambiguous_numeric_date(series: pd.Series) -> bool:
    """Return True if ANY full-column date/timestamp has ambiguous ordering.

    The check recognizes numeric dates embedded in the supported timestamp
    representation, e.g. ``01/02/2024 10:30:00`` and
    ``01/02/2024T10:30:00``.
    """
    if len(series) == 0:
        return False

    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    if pd.api.types.is_numeric_dtype(series):
        return False

    matches = _extract_numeric_date_components(series)

    numeric_shape = matches[0].notna()

    if not numeric_shape.any():
        return False

    first = pd.to_numeric(
        matches.loc[numeric_shape, 0],
        errors="coerce",
    )
    second = pd.to_numeric(
        matches.loc[numeric_shape, 2],
        errors="coerce",
    )
    last = pd.to_numeric(
        matches.loc[numeric_shape, 3],
        errors="coerce",
    )

    # YYYY-MM-DD is unambiguous because the year is first.
    year_first = first.between(1000, 9999)

    # Automatic DMY/MDY reasoning requires an explicit four-digit year.
    four_digit_year = last.between(1000, 9999)

    ambiguous = (
        ~year_first
        & four_digit_year
        & first.between(1, 12)
        & second.between(1, 12)
    )

    return bool(ambiguous.any())


def _is_supported_absolute_datetime_value(value: Any) -> bool:
    """Return True only for absolute date/time representations supported by inference.

    This deliberately rejects values that dateparser can understand but that do
    not contain an explicit calendar year, such as ``today``, ``tomorrow``,
    ``March``, ``Jan 5``, ``2024``, and relative expressions. The hard parser may
    be permissive *within* an identified datetime column, but it must not invent
    missing temporal precision or temporal meaning.
    """
    if pd.isna(value):
        return False

    text = str(value).strip()
    if not text:
        return False

    return bool(
        _NUMERIC_DATE_PATTERN.fullmatch(text)
        or _TEXT_DATE_PATTERN.fullmatch(text)
    )


def _date_pattern_details(series: pd.Series) -> _DateEvidence:
    values = _sample_values(series)

    if len(values) == 0:
        return _DateEvidence(
            parse_ratio=0.0,
            pattern_ratio=0.0,
            ambiguous_ratio=0.0,
            unambiguous_pattern_ratio=0.0,
            format_consistency=0.0,
            dmy_ratio=0.0,
            mdy_ratio=0.0,
            mixed_numeric_conventions=False,
        )

    if pd.api.types.is_datetime64_any_dtype(values):
        return _DateEvidence(
            parse_ratio=1.0,
            pattern_ratio=1.0,
            ambiguous_ratio=0.0,
            unambiguous_pattern_ratio=1.0,
            format_consistency=1.0,
            dmy_ratio=0.0,
            mdy_ratio=0.0,
            mixed_numeric_conventions=False,
        )

    # Numeric columns are not treated as dates automatically. Epoch semantics
    # require explicit domain knowledge and are intentionally outside this
    # automatic conversion layer.
    if pd.api.types.is_numeric_dtype(values):
        return _DateEvidence(
            parse_ratio=0.0,
            pattern_ratio=0.0,
            ambiguous_ratio=0.0,
            unambiguous_pattern_ratio=0.0,
            format_consistency=0.0,
            dmy_ratio=0.0,
            mdy_ratio=0.0,
            mixed_numeric_conventions=False,
        )

    strings = values.astype("string").str.strip()

    parsed = pd.to_datetime(
        strings,
        errors="coerce",
        format="mixed",
    )

    parse_ratio = _safe_ratio(parsed.notna())

    numeric_matches = strings.str.extract(
        _NUMERIC_DATE_PATTERN,
        expand=True,
    )

    numeric_shape = numeric_matches[0].notna()

    # These masks are indexed exactly like `strings`.
    ambiguous_mask = pd.Series(False, index=strings.index, dtype=bool)
    unambiguous_numeric_mask = pd.Series(False, index=strings.index, dtype=bool)

    if numeric_shape.any():
        # The regex groups are:
        #   group 1 = first component
        #   group 2 = separator
        #   group 3 = second component
        #   group 4 = final component
        first_component = pd.to_numeric(
            numeric_matches.loc[numeric_shape, 0],
            errors="coerce",
        )
        second = pd.to_numeric(
            numeric_matches.loc[numeric_shape, 2],
            errors="coerce",
        )
        fourth_component = pd.to_numeric(
            numeric_matches.loc[numeric_shape, 3],
            errors="coerce",
        )

        # Automatic numeric DMY/MDY reasoning requires an explicit four-digit
        # year. Two-digit years need an external century policy.
        four_digit_year = fourth_component.between(1000, 9999)

        # YYYY-MM-DD is unambiguous because the year is first.
        year_first = (
            first_component.between(1000, 9999)
            & second.between(1, 12)
            & fourth_component.between(1, 31)
        )

        # DD-MM-YYYY / MM-DD-YYYY are ambiguous when both first and second
        # components can legally represent a month/day.
        day_month_ambiguous = (
            four_digit_year
            & first_component.between(1, 12)
            & second.between(1, 12)
        )

        # If either component is > 12, ordering is constrained:
        # 13-02-2024 must be DD-MM-YYYY;
        # 02-13-2024 must be MM-DD-YYYY.
        numeric_unambiguous = (
            (~year_first)
            & (
                first_component.gt(12)
                | second.gt(12)
            )
        )

        ambiguous_mask.loc[numeric_shape] = (
            day_month_ambiguous
            & ~year_first
        ).astype(bool)

        unambiguous_numeric_mask.loc[numeric_shape] = (
            year_first | numeric_unambiguous
        ).astype(bool)

    text_mask = strings.str.match(
        _TEXT_DATE_PATTERN,
        na=False,
    )

    pattern_mask = (
        numeric_shape
        | text_mask
    )

    unambiguous_mask = (
        unambiguous_numeric_mask
        | text_mask
    )

    pattern_ratio = _safe_ratio(pattern_mask)
    ambiguous_ratio = _safe_ratio(ambiguous_mask)
    unambiguous_pattern_ratio = _safe_ratio(unambiguous_mask)

    # Track numeric DMY and MDY separately, including timestamp representations. Values with a day/month
    # component > 12 are individually unambiguous and reveal the convention:
    #
    #   13-02-2024 -> DMY
    #   03-14-2024 -> MDY
    #
    # Values where both components are <= 12 remain ambiguous and therefore
    # do not contribute to either convention.
    dmy_numeric_mask = pd.Series(False, index=strings.index, dtype=bool)
    mdy_numeric_mask = pd.Series(False, index=strings.index, dtype=bool)

    if numeric_shape.any():
        first_component = pd.to_numeric(
            numeric_matches.loc[numeric_shape, 0],
            errors="coerce",
        )
        second_component = pd.to_numeric(
            numeric_matches.loc[numeric_shape, 2],
            errors="coerce",
        )
        fourth_component = pd.to_numeric(
            numeric_matches.loc[numeric_shape, 3],
            errors="coerce",
        )

        year_first = first_component.between(1000, 9999)

        four_digit_year = fourth_component.between(1000, 9999)

        dmy_numeric_mask.loc[numeric_shape] = (
            ~year_first
            & four_digit_year
            & first_component.gt(12)
            & second_component.between(1, 12)
        ).astype(bool)

        mdy_numeric_mask.loc[numeric_shape] = (
            ~year_first
            & four_digit_year
            & first_component.between(1, 12)
            & second_component.gt(12)
        ).astype(bool)

    # Broad representation families. A mixture of families lowers confidence
    # even when individual values are parseable.
    format_family = pd.Series(
        "other",
        index=strings.index,
        dtype="string",
    )

    format_family.loc[
        numeric_shape
        & numeric_matches[0].str.len().eq(4)
    ] = "ymd_numeric"

    format_family.loc[dmy_numeric_mask] = "numeric_dmy"
    format_family.loc[mdy_numeric_mask] = "numeric_mdy"

    format_family.loc[
        text_mask
        & strings.str.match(
            r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$",
            na=False,
        )
    ] = "text_dmy"

    format_family.loc[
        text_mask
        & strings.str.match(
            r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$",
            na=False,
        )
    ] = "text_mdy"

    dmy_ratio = _safe_ratio(
        dmy_numeric_mask
        | (
            text_mask
            & strings.str.match(
                r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$",
                na=False,
            )
        )
    )

    mdy_ratio = _safe_ratio(
        mdy_numeric_mask
        | (
            text_mask
            & strings.str.match(
                r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}$",
                na=False,
            )
        )
    )

    mixed_numeric_conventions = bool(
        dmy_numeric_mask.any()
        and mdy_numeric_mask.any()
    )

    valid_families = format_family[
        format_family != "other"
    ]

    if len(valid_families):
        counts = valid_families.value_counts()
        format_consistency = float(
            counts.iloc[0] / len(valid_families)
        )
    else:
        format_consistency = 0.0

    return _DateEvidence(
        parse_ratio=parse_ratio,
        pattern_ratio=pattern_ratio,
        ambiguous_ratio=ambiguous_ratio,
        unambiguous_pattern_ratio=unambiguous_pattern_ratio,
        format_consistency=format_consistency,
        dmy_ratio=dmy_ratio,
        mdy_ratio=mdy_ratio,
        mixed_numeric_conventions=mixed_numeric_conventions,
    )



def _has_mixed_numeric_date_conventions(series: pd.Series) -> bool:
    """Return True if ANY full-column values establish both DMY and MDY.

    Numeric dates may be plain dates or embedded in supported timestamp
    representations.
    """
    if len(series) == 0:
        return False

    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    if pd.api.types.is_numeric_dtype(series):
        return False

    matches = _extract_numeric_date_components(series)

    numeric_shape = matches[0].notna()

    if not numeric_shape.any():
        return False

    first = pd.to_numeric(
        matches.loc[numeric_shape, 0],
        errors="coerce",
    )
    second = pd.to_numeric(
        matches.loc[numeric_shape, 2],
        errors="coerce",
    )
    last = pd.to_numeric(
        matches.loc[numeric_shape, 3],
        errors="coerce",
    )

    year_first = first.between(1000, 9999)
    four_digit_year = last.between(1000, 9999)

    dmy = (
        ~year_first
        & four_digit_year
        & first.gt(12)
        & second.between(1, 12)
    )

    mdy = (
        ~year_first
        & four_digit_year
        & first.between(1, 12)
        & second.gt(12)
    )

    return bool(dmy.any() and mdy.any())


def _datetime_parse_ratio(series: pd.Series) -> float:
    return _date_pattern_details(series).parse_ratio


def _date_pattern_ratio(series: pd.Series) -> float:
    return _date_pattern_details(series).pattern_ratio


# ---------------------------------------------------------------------------
# Identifier evidence
# ---------------------------------------------------------------------------

def _identifier_pattern_ratio(series: pd.Series) -> float:
    values = _sample_values(series)

    if len(values) == 0:
        return 0.0

    strings = values.astype("string").str.strip()

    return float(
        strings.str.match(_ID_PATTERN, na=False).mean()
    )


# ---------------------------------------------------------------------------
# Name evidence
# ---------------------------------------------------------------------------

def _name_scores(
    tokens: list[str],
) -> tuple[float, float, float, float, float]:
    """Return conservative semantic signals from the column name."""
    joined = "_".join(tokens)

    id_score = 0.0

    if _has_id_suffix(tokens):
        id_score = 1.0
    elif any(token in _STRONG_ID_TOKENS for token in tokens):
        id_score = 0.80

    # "code", "ref", "reference", and "key" are deliberately weak.
    if any(
        token in {"code", "ref", "reference", "key"}
        for token in tokens
    ):
        id_score = max(id_score, 0.20)

    date_score = 0.0

    if joined in _DATE_TOKENS:
        date_score = 1.0
    elif any(
        token in {"date", "datetime", "timestamp", "dob"}
        for token in tokens
    ):
        date_score = 0.85

    numeric_score = _token_score(
        tokens,
        _NUMERIC_TOKENS,
    )

    boolean_score = _token_score(
        tokens,
        _BOOLEAN_TOKENS,
    )

    if _has_boolean_prefix(tokens):
        boolean_score = max(
            boolean_score,
            0.85,
        )

    categorical_score = _token_score(
        tokens,
        _CATEGORICAL_TOKENS,
    )

    return (
        id_score,
        date_score,
        numeric_score,
        categorical_score,
        boolean_score,
    )


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

def _collect_evidence(
    series: pd.Series,
    column_name: Any,
    n_rows: int,
) -> _ColumnEvidence:
    values = _sample_values(series)

    n_non_null = int(series.notna().sum())
    n_unique = int(series.nunique(dropna=True))

    unique_ratio = (
        n_unique / n_non_null
        if n_non_null
        else 0.0
    )

    tokens = _normalise_column_name(column_name)

    (
        id_name_score,
        date_name_score,
        numeric_name_score,
        categorical_name_score,
        boolean_name_score,
    ) = _name_scores(tokens)

    date_evidence = _date_pattern_details(series)

    if len(series):
        string_ratio = float(
            series.map(
                lambda value: isinstance(value, str)
            ).mean()
        )
    else:
        string_ratio = 0.0

    is_monotonic = False

    try:
        non_null = series.dropna()

        if len(non_null) > 1:
            is_monotonic = bool(
                non_null.is_monotonic_increasing
                or non_null.is_monotonic_decreasing
            )
    except (TypeError, ValueError):
        pass

    return _ColumnEvidence(
        name=str(column_name),
        tokens=tuple(tokens),
        original_dtype=str(series.dtype),

        n_rows=n_rows,
        n_non_null=n_non_null,
        n_unique=n_unique,
        unique_ratio=unique_ratio,

        numeric_ratio=_numeric_parse_ratio(series),
        datetime_ratio=date_evidence.parse_ratio,
        boolean_ratio=_boolean_parse_ratio(series),

        integer_like_ratio=_integer_like_ratio(series),
        string_ratio=string_ratio,

        id_name_score=id_name_score,
        date_name_score=date_name_score,
        numeric_name_score=numeric_name_score,
        categorical_name_score=categorical_name_score,
        boolean_name_score=boolean_name_score,

        id_pattern_ratio=_identifier_pattern_ratio(series),
        date_pattern_ratio=date_evidence.pattern_ratio,

        date=date_evidence,

        is_monotonic=is_monotonic,
        is_unique=(
            n_non_null > 0
            and n_unique == n_non_null
        ),
        has_two_distinct_values=n_unique == 2,

        is_known_binary_numeric=_is_known_binary_numeric(series),
        is_potential_binary_numeric=_is_potential_binary_numeric(series),
    )


# ---------------------------------------------------------------------------
# Semantic classification
# ---------------------------------------------------------------------------

def _infer_role(evidence: _ColumnEvidence) -> _Inference:
    """Infer a semantic role from competing evidence.

    The scores are relative evidence strengths, not calibrated probabilities.
    """
    scores = {
        "continuous": 0.0,
        "binary": 0.0,
        "identifier": 0.0,
        "categorical": 0.0,
        "datetime": 0.0,
    }

    # ------------------------------------------------------------------
    # DATETIME
    # ------------------------------------------------------------------

    scores["datetime"] += 0.40 * evidence.datetime_ratio
    scores["datetime"] += 0.20 * evidence.date_name_score
    scores["datetime"] += 0.20 * evidence.date.unambiguous_pattern_ratio
    scores["datetime"] += 0.10 * evidence.date.format_consistency

    # Parser success is NOT enough for ambiguous numeric dates.
    if evidence.date.ambiguous_ratio > 0:
        scores["datetime"] *= max(
            0.20,
            1.0 - 0.80 * evidence.date.ambiguous_ratio,
        )

    # Strong date-name evidence can identify a date-like column even when
    # some values are missing or malformed, but it still must have substantial
    # date evidence before conversion.
    if (
        evidence.date_name_score >= 0.85
        and evidence.datetime_ratio >= 0.90
        and evidence.date.format_consistency >= 0.80
    ):
        scores["datetime"] += 0.20

    # ------------------------------------------------------------------
    # BINARY
    # ------------------------------------------------------------------

    # Boolean dtype and explicit textual boolean domains are strong evidence.
    # `boolean_ratio` already captures native bool columns as 1.0.
    if evidence.boolean_ratio >= 0.95:
        scores["binary"] += 0.70

    if evidence.boolean_name_score >= 0.85:
        scores["binary"] += 0.25

    # Canonical numeric binary domain {0, 1} is strong evidence. This
    # decision is based on the full column, not the inference sample.
    if evidence.is_known_binary_numeric:
        scores["binary"] += 0.80

    # Any other two-valued numeric domain is only weak evidence.
    # {10,20}, {100,200}, {1,5}, etc. remain ambiguous unless other semantic
    # evidence supports binary meaning.
    if evidence.is_potential_binary_numeric:
        scores["binary"] += 0.08

    # A constant column has no binary evidence.
    if evidence.n_unique <= 1:
        scores["binary"] = 0.0

    # ------------------------------------------------------------------
    # IDENTIFIER
    # ------------------------------------------------------------------

    scores["identifier"] += 0.35 * evidence.id_name_score
    scores["identifier"] += 0.25 * evidence.id_pattern_ratio

    # Uniqueness is supporting evidence, not proof.
    if (
        evidence.unique_ratio >= 0.98
        and evidence.n_unique >= 10
    ):
        scores["identifier"] += 0.18

    if (
        evidence.is_monotonic
        and evidence.integer_like_ratio >= 0.95
        and evidence.unique_ratio >= 0.98
    ):
        scores["identifier"] += 0.12

    # Strong identifier names/patterns can make a high-cardinality numeric
    # column identifier-like without making uniqueness alone decisive.
    if (
        evidence.numeric_ratio >= 0.95
        and evidence.unique_ratio >= 0.995
        and (
            evidence.id_name_score >= 0.75
            or evidence.id_pattern_ratio >= 0.80
        )
    ):
        scores["identifier"] += 0.25

    # ------------------------------------------------------------------
    # CONTINUOUS
    # ------------------------------------------------------------------

    scores["continuous"] += 0.65 * evidence.numeric_ratio
    scores["continuous"] += 0.20 * evidence.numeric_name_score

    if evidence.n_unique > 10:
        scores["continuous"] += 0.10

    if evidence.integer_like_ratio < 0.95:
        scores["continuous"] += 0.10

    # Canonical binary numeric data should not be treated as continuous.
    if evidence.is_known_binary_numeric:
        scores["continuous"] *= 0.20

    # A high-cardinality numeric identifier remains possible, but only strong
    # identifier evidence should suppress continuous semantics.
    if (
        evidence.unique_ratio >= 0.995
        and evidence.n_unique >= 20
        and (
            evidence.id_name_score >= 0.75
            or evidence.id_pattern_ratio >= 0.80
        )
    ):
        scores["continuous"] *= 0.65

    # ------------------------------------------------------------------
    # CATEGORICAL
    # ------------------------------------------------------------------

    scores["categorical"] += (
        0.30 * evidence.categorical_name_score
    )

    if evidence.string_ratio >= 0.80:
        scores["categorical"] += 0.25

    if evidence.unique_ratio <= 0.20:
        scores["categorical"] += 0.20

    if 2 <= evidence.n_unique <= 50:
        scores["categorical"] += 0.15

    # A two-state non-boolean textual domain is naturally categorical.
    if (
        evidence.has_two_distinct_values
        and evidence.boolean_ratio < 0.95
        and not evidence.is_known_binary_numeric
    ):
        scores["categorical"] += 0.15

    # Numeric low-cardinality domains may be categorical, but cardinality
    # alone does not establish binary semantics.
    if (
        evidence.numeric_ratio >= 0.95
        and 2 < evidence.n_unique <= 20
    ):
        scores["categorical"] += 0.10

    # ------------------------------------------------------------------
    # Hard semantic precedence rules
    # ------------------------------------------------------------------

    # Only canonical {0,1} numeric domains receive a strong binary floor.
    if evidence.is_known_binary_numeric:
        scores["binary"] = max(
            scores["binary"],
            0.90,
        )

    # Explicit boolean text gets a strong binary floor.
    if evidence.boolean_ratio >= 0.98:
        scores["binary"] = max(
            scores["binary"],
            0.90,
        )

    # Ambiguous numeric dates do NOT receive a hard datetime floor. An explicit
    # date name can support datetime semantics, but the intended day/month
    # ordering remains unresolved.
    if (
        evidence.datetime_ratio >= 0.98
        and evidence.date.unambiguous_pattern_ratio >= 0.90
        and evidence.date.ambiguous_ratio == 0
        and not evidence.date.mixed_numeric_conventions
        and evidence.date.format_consistency >= 0.90
    ):
        scores["datetime"] = max(
            scores["datetime"],
            0.90,
        )

    # Strong identifier suffix/name + structural evidence can dominate
    # ordinary numeric semantics.
    if (
        evidence.id_name_score >= 0.75
        and evidence.unique_ratio >= 0.95
        and (
            evidence.id_pattern_ratio >= 0.50
            or evidence.numeric_ratio >= 0.95
        )
    ):
        scores["identifier"] = max(
            scores["identifier"],
            0.85,
        )

    scores = {
        role: max(0.0, float(score))
        for role, score in scores.items()
    }

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    winner, winner_score = ranked[0]
    second_score = ranked[1][1]

    total = sum(scores.values())

    relative_strength = (
        winner_score / total
        if total > 0
        else 0.0
    )

    margin = (
        winner_score - second_score
    ) / max(winner_score, 1e-12)

    decision_strength = (
        0.70 * relative_strength
        + 0.30 * max(0.0, min(1.0, margin))
    )

    return _Inference(
        role=winner,
        decision_strength=float(decision_strength),
        scores=scores,
    )


# ---------------------------------------------------------------------------
# Safe physical conversions
# ---------------------------------------------------------------------------

def _convert_numeric(series: pd.Series) -> pd.Series:
    """Convert numeric representations, coercing invalid values to NaN.
    
    Hard mode: When the semantic inference confidently identifies a column
    as continuous/numeric, this function normalizes it and intentionally
    converts unparseable values to NaN. This is the hard/aggressive version
    that does NOT reject the conversion because some values are invalid.
    
    Protections:
    - Meaningful leading zeros are preserved (identifiers like "00123" are not converted)
    - Currency symbols and Western thousands separators are normalized
    - Invalid numeric strings become NaN (not converted back to original representation)
    """
    if len(series) == 0:
        return series.copy()

    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):
        # Already numeric; coerce any remaining non-numeric to NaN
        converted = pd.to_numeric(series, errors="coerce")
        return converted
    
    # String/object dtype: check for meaningful leading zeros before normalizing
    # Leading zeros on identifiers like "00123" should prevent numeric conversion
    if _has_meaningful_leading_zero(series):
        return series.copy()

    # Normalize numeric strings: strip currency, handle thousands separators
    cleaned = _normalise_numeric_strings(series)

    # Convert to numeric; invalid values become NaN (intentional in hard mode)
    converted = pd.to_numeric(
        cleaned,
        errors="coerce",
    )

    return converted


def _convert_datetime(series: pd.Series) -> pd.Series:
    """Convert supported absolute date/time values to pandas datetime.

    Hard mode is coercive after semantic identification: invalid values become
    ``NaT``. However, ``dateparser`` is intentionally not allowed to expand the
    semantic contract by interpreting relative or incomplete expressions such
    as ``today``, ``tomorrow``, ``March``, ``Jan 5``, or bare ``2024``.

    Timezone policy:
    - all-naive parsed values remain timezone-naive ``datetime64[ns]``;
    - if any parsed value is timezone-aware, all parsed values are normalized
      to UTC, with naive values in that mixed column treated as UTC so the
      result has one deterministic timezone-aware dtype;
    - timezone information is never silently discarded.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.copy()

    # Ambiguous dates must still refuse conversion - hard mode does not guess.
    if (
        _has_ambiguous_numeric_date(series)
        or _has_mixed_numeric_date_conventions(series)
        or _has_short_numeric_date_year(series)
    ):
        return series.copy()

    if not HAS_DATEPARSER:
        # The hard datetime normalizer relies on dateparser. Keep the failure
        # explicit rather than silently changing parsing semantics.
        raise ImportError(
            "dateparser is required for hard datetime normalization. "
            "Install it with `pip install dateparser`."
        )

    # Keep parsed values positional rather than index-aligned.
    #
    # This is important because pandas DataFrames are allowed to have
    # duplicate index labels. Using the original index while constructing
    # intermediate Series can cause reindexing/alignment errors. The parsed
    # list therefore always has exactly the same length and row order as
    # the source Series, with NaT representing values that are missing,
    # unsupported, or unparseable.
    parsed_values: list[Any] = []

    for value in series:
        if pd.isna(value):
            parsed_values.append(pd.NaT)
            continue

        # Critical safety boundary: dateparser must not turn a value that the
        # inference grammar does not recognize as an absolute date/time into a
        # datetime merely because it happens to understand the expression.
        if not _is_supported_absolute_datetime_value(value):
            parsed_values.append(pd.NaT)
            continue

        try:
            parsed = _dateparser.parse(str(value).strip())
        except Exception:
            parsed = None

        parsed_values.append(
            parsed if parsed is not None else pd.NaT
        )

    # Detect whether dateparser returned any timezone-aware values. A timezone
    # aware dtype cannot safely be forced into naive datetime64[ns].
    has_aware = any(
        getattr(value, "tzinfo", None) is not None
        and value.utcoffset() is not None
        for value in parsed_values
        if pd.notna(value)
    )

    try:
        # IMPORTANT:
        # The intermediate Series deliberately uses a fresh RangeIndex.
        # Alignment with the caller's index happens only once, at the end,
        # by position. This remains safe when series.index contains duplicates.
        parsed_series = pd.Series(parsed_values)

        if has_aware:
            # Deterministic policy: normalize the complete parsed column to UTC.
            # pandas treats naive values as UTC when utc=True, which is the
            # explicit policy for mixed naive/aware source columns here.
            normalized = pd.to_datetime(
                parsed_series,
                errors="coerce",
                utc=True,
            )

            # Reattach the original index positionally. Using .array preserves
            # the timezone-aware pandas dtype.
            return pd.Series(
                normalized.array,
                index=series.index,
                dtype="datetime64[ns, UTC]",
            )

        normalized = pd.to_datetime(
            parsed_series,
            errors="coerce",
        )

        # Reattach the original index positionally. No label-based .loc
        # assignment is used, so duplicate index labels are safe.
        return pd.Series(
            normalized.array,
            index=series.index,
            dtype="datetime64[ns]",
        )

    except (TypeError, ValueError, OverflowError):
        # Extremely unusual parser outputs should still obey hard-mode
        # semantics: values that cannot be represented become NaT.
        fallback_values: list[Any] = []

        for value in parsed_values:
            try:
                if pd.isna(value):
                    fallback_values.append(pd.NaT)
                    continue

                if has_aware:
                    result_value = pd.to_datetime(
                        value,
                        errors="coerce",
                        utc=True,
                    )
                else:
                    result_value = pd.to_datetime(
                        value,
                        errors="coerce",
                    )

                fallback_values.append(
                    result_value if pd.notna(result_value) else pd.NaT
                )

            except Exception:
                fallback_values.append(pd.NaT)

        if has_aware:
            try:
                fallback_normalized = pd.to_datetime(
                    pd.Series(fallback_values),
                    errors="coerce",
                    utc=True,
                )

                return pd.Series(
                    fallback_normalized.array,
                    index=series.index,
                    dtype="datetime64[ns, UTC]",
                )
            except Exception:
                return pd.Series(
                    pd.NaT,
                    index=series.index,
                    dtype="datetime64[ns, UTC]",
                )

        try:
            fallback_normalized = pd.to_datetime(
                pd.Series(fallback_values),
                errors="coerce",
            )

            return pd.Series(
                fallback_normalized.array,
                index=series.index,
                dtype="datetime64[ns]",
            )
        except Exception:
            return pd.Series(
                pd.NaT,
                index=series.index,
                dtype="datetime64[ns]",
            )

def _convert_binary(series: pd.Series) -> pd.Series:
    """Convert binary/boolean representations, coercing invalid values to pd.NA.
    
    Hard mode: When semantic inference confidently identifies a column as
    binary/boolean, this function converts it to nullable boolean dtype and
    intentionally converts unrecognized values to pd.NA.
    
    Recognized boolean representations:
    - Native bool dtype
    - Numeric 0/1
    - Text: "true"/"false", "yes"/"no", "y"/"n", "t"/"f", "on"/"off", "1"/"0"
    
    Unrecognized non-null values become pd.NA (hard mode intentional behavior).
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        unique_values = set(series.dropna().tolist())

        if unique_values and unique_values.issubset({0, 1}):
            return series.astype("boolean")

        # Never coerce arbitrary two-valued numeric measurements as binary.
        # The inference engine should have avoided classifying {10, 20} as binary.
        return series.copy()

    # String/object dtype: map recognized boolean values
    values = series.astype("string").str.strip().str.lower()

    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
        "on": True,
        "off": False,
        "1": True,
        "0": False,
    }

    converted = values.map(mapping)

    # Hard mode: unrecognized values become pd.NA (not reverted to original)
    return converted.astype("boolean")


def _convert_categorical(series: pd.Series) -> pd.Series:
    return series.astype("category")


def _convert_identifier(series: pd.Series) -> pd.Series:
    """Preserve identifier values; do not silently alter their representation."""
    return series.copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_and_fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Infer semantic column roles and apply coercive dtype normalization.

    The inference engine considers column names, original pandas dtype,
    cardinality, uniqueness, numeric parseability, datetime parseability,
    boolean patterns, identifier patterns, date patterns, and value
    distribution signals to confidently identify semantic types.

    Once a type is confidently identified (decision_strength >= 0.55), this
    function applies COERCIVE conversion: invalid values are converted to
    NaN/NaT/pd.NA rather than rejecting the entire conversion.

    Semantic role and physical pandas dtype are deliberately treated as
    separate concepts. In particular, identifiers are preserved rather than
    silently converted to categorical dtype.

    Conservative decisions during inference:
    - Only high-confidence semantic identifications trigger conversion (decision_strength >= 0.55)
    - Ambiguous numeric dates (01/02/2024) are left unchanged
    - Mixed DMY/MDY conventions are left unchanged
    - Numeric dates with shortened years are left unchanged
    - Ambiguity cannot be resolved, so no conversion is attempted

    Aggressive behavior during conversion (hard mode):
    - Valid numeric strings → float64 (invalid → NaN)
    - Valid date strings → datetime64[ns] (invalid → NaT)
    - Recognized boolean representations → boolean (unrecognized → pd.NA)
    - Low-cardinality strings → category (no coercion)
    - Identifiers → preserved as-is (no coercion)

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame. The original object is never mutated.

    Returns
    -------
    pandas.DataFrame
        A copy with confidently identified semantic conversions applied
        aggressively. Invalid values for the target type become missing.

    Notes
    -----
    This hard/coercive inference is appropriate for baseline datasets where
    you want to normalize obvious types and accept that malformed values
    become missing. For production monitoring where preserving original
    representations is more important, use infer_types_for_new=False.

    Ambiguous numeric dates and mixed conventions are not resolved because
    the intended interpretation cannot be determined. For example, "01/02/2024"
    cannot safely become either "1 Feb 2024" or "2 Jan 2024".

    Numeric strings with meaningful leading zeros (like "00123") are
    preserved as identifiers and not converted to numeric.

    The dateparser library is required for hard datetime normalization.
    Timezone-aware values are normalized to UTC.

    Examples
    --------
    >>> fixed = infer_and_fix_types(raw_df)
    >>> fixed.dtypes

    Pipeline usage:

    >>> result = Run_DQ_Pipeline(
    ...     ["customers"],
    ...     [raw_df],
    ...     infer_types=True,  # Enable hard type inference
    ... )

    See Also
    --------
    Run_DQ_Pipeline : Offers pipeline-level type inference.
    generate_meta : Profiles the stabilized DataFrame.
    """
    result = df.copy()
    n_rows = len(result)

    for column_name in result.columns:
        series = result[column_name]

        evidence = _collect_evidence(
            series,
            column_name,
            n_rows,
        )

        inference = _infer_role(evidence)

        # Explicitly detected unsafe numeric date representations are treated
        # as unsafe/unknown. Do not allow them to fall through to
        # generic categorical conversion: the module promises ambiguous data
        # remains unchanged.
        if (
            _has_ambiguous_numeric_date(series)
            or _has_mixed_numeric_date_conventions(series)
            or _has_short_numeric_date_year(series)
        ):
            continue

        # Weak or ambiguous decisions should not alter the user's data.
        if inference.decision_strength < 0.55:
            continue

        if inference.role == "datetime":
            result[column_name] = _convert_datetime(series)

        elif inference.role == "continuous":
            result[column_name] = _convert_numeric(series)

        elif inference.role == "binary":
            result[column_name] = _convert_binary(series)

        elif inference.role == "categorical":
            result[column_name] = _convert_categorical(series)

        elif inference.role == "identifier":
            result[column_name] = _convert_identifier(series)

    return result


def diagnose_column_types(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Perform non-mutating diagnostic semantic type inference on all columns.

    This function never modifies the input DataFrame or changes any values.
    It returns diagnostic information for every column, including:
    - physical_dtype: the current pandas dtype
    - suggested_semantic_type: inferred role ('numeric', 'datetime', 'binary', 'identifier', etc.)
    - confidence: float score between 0.0 and 1.0
    - convertible_count: count of values consistent with suggested type
    - non_convertible_count: count of conflicting values
    - conversion_rate: float ratio in [0, 1]
    - sample_conflicts: sample list of conflicting values
    """
    diagnostics = {}
    n_rows = len(df)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        n_non_null = len(non_null)

        evidence = _collect_evidence(series, col, n_rows)
        inference = _infer_role(evidence)

        # Check conversion stats
        suggested = inference.role
        conflicts = []
        convertible = n_non_null

        if suggested == "continuous" and not pd.api.types.is_numeric_dtype(series):
            parsed = pd.to_numeric(non_null, errors="coerce")
            convertible = int(parsed.notna().sum())
            conflicts = non_null[parsed.isna()].head(5).tolist()

        elif suggested == "datetime" and not pd.api.types.is_datetime64_any_dtype(series):
            parsed = pd.to_datetime(non_null.astype(str), errors="coerce")
            convertible = int(parsed.notna().sum())
            conflicts = non_null[parsed.isna()].head(5).tolist()

        elif suggested == "binary" and not pd.api.types.is_bool_dtype(series):
            as_str = non_null.astype(str).str.strip().str.lower()
            valid_bin = as_str.isin({"0", "1", "true", "false", "yes", "no", "y", "n", "t", "f"})
            convertible = int(valid_bin.sum())
            conflicts = non_null[~valid_bin].head(5).tolist()

        non_convertible = max(0, n_non_null - convertible)
        rate = (convertible / n_non_null) if n_non_null > 0 else 1.0

        diagnostics[str(col)] = {
            "physical_dtype": str(series.dtype),
            "suggested_semantic_type": suggested,
            "confidence": float(round(inference.decision_strength, 3)),
            "convertible_count": convertible,
            "non_convertible_count": non_convertible,
            "conversion_rate": float(round(rate, 4)),
            "sample_conflicts": conflicts,
        }

    return diagnostics
