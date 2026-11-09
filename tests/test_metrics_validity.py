"""Unit and regression parity tests for cleanalytix.metrics.validity (Phase 1)."""

import numpy as np
import pandas as pd
import pytest

from cleanalytix.generate_meta import (
    generate_meta,
    get_type_error_count,
    rule_based_errors,
)
from cleanalytix.metrics.validity import (
    compute_column_validity,
    compute_rule_validity_metrics,
    compute_type_validity_metrics,
    compute_whitespace_metrics,
)


class TestTypeValidityMetrics:
    """Tests for semantic type error detection and N_non_null denominator semantics."""

    def test_valid_numeric_series(self):
        """1. Valid numeric series -> 0 type errors, 0.0%."""
        s = pd.Series([10.0, 20.5, 30.2, 40.0, 50.1])
        res = compute_type_validity_metrics(s)
        assert res["type_error_count"] == 0
        assert res["type_error_percentage"] == 0.0
        assert res["non_null_count"] == 5
        assert res["inferred_type"] == "numeric"

    def test_numeric_series_with_invalid_strings(self):
        """2. Numeric series with invalid strings ('abc', 'N/A') -> expected count and %."""
        s = pd.Series(["10", "20", "abc", "N/A", "30"])
        res = compute_type_validity_metrics(s)
        assert res["non_null_count"] == 5
        assert res["inferred_type"] == "numeric"
        assert res["type_error_count"] == 2
        assert res["type_error_percentage"] == pytest.approx(40.0)

    def test_numeric_series_with_pos_and_neg_infinity(self):
        """3. Numeric series with +inf and -inf -> counted as type errors and in N_non_null."""
        s = pd.Series([1.0, 2.0, np.inf, -np.inf, 3.0])
        res = compute_type_validity_metrics(s)
        assert res["non_null_count"] == 5
        assert res["type_error_count"] == 2
        assert res["type_error_percentage"] == pytest.approx((2 / 5) * 100.0)

    def test_numeric_series_with_mixed_nulls_and_invalid_values(self):
        """4. Numeric series with mixed nulls and invalid values -> nulls excluded from denominator."""
        s = pd.Series(["1.0", "2.0", "bad", None, np.nan, "4.0"])
        res = compute_type_validity_metrics(s)
        assert res["non_null_count"] == 4
        assert res["type_error_count"] == 1
        assert res["type_error_percentage"] == pytest.approx(25.0)

    def test_all_null_series(self):
        """5. All-null series -> 0 errors, 0.0%."""
        s = pd.Series([None, np.nan, None], dtype=object)
        res = compute_type_validity_metrics(s)
        assert res["non_null_count"] == 0
        assert res["type_error_count"] == 0
        assert res["type_error_percentage"] == 0.0

    def test_empty_series(self):
        """6. Empty series -> 0 errors, 0.0%."""
        s = pd.Series([], dtype=float)
        res = compute_type_validity_metrics(s)
        assert res["non_null_count"] == 0
        assert res["type_error_count"] == 0
        assert res["type_error_percentage"] == 0.0
        assert res["inferred_type"] == "numeric"

    def test_datetime_and_categorical_type_validity(self):
        """Verify datetime coercion errors and categorical string label preservation."""
        dt_s = pd.Series(["2024-01-01", "2024-01-02", "2024-01-03", "not_a_date", None])
        dt_res = compute_type_validity_metrics(dt_s)
        assert dt_res["inferred_type"] == "datetime"
        assert dt_res["non_null_count"] == 4
        assert dt_res["type_error_count"] == 1
        assert dt_res["type_error_percentage"] == pytest.approx(25.0)

        # Categorical column with string-encoded numbers ("404") must have 0 errors
        cat_s = pd.Series(["apple", "banana", "cherry", "404", "March 2024"])
        cat_res = compute_type_validity_metrics(cat_s)
        assert cat_res["inferred_type"] == "categorical"
        assert cat_res["type_error_count"] == 0
        assert cat_res["type_error_percentage"] == 0.0

        # Predominantly string column with raw non-string object
        mixed_cat = pd.Series(["alpha", "beta", "gamma", 999])
        mixed_res = compute_type_validity_metrics(mixed_cat)
        assert mixed_res["inferred_type"] == "categorical"
        assert mixed_res["type_error_count"] == 1
        assert mixed_res["type_error_percentage"] == pytest.approx(25.0)

    def test_invalid_expected_type_raises_value_error(self):
        """Invalid expected_type must raise ValueError and not silently fall back."""
        s = pd.Series([1, 2, 3])
        with pytest.raises(ValueError, match="Invalid expected_type"):
            compute_type_validity_metrics(s, expected_type="boolean")

        with pytest.raises(ValueError, match="Invalid expected_type"):
            compute_type_validity_metrics(s, expected_type="unknown_dtype")


class TestWhitespaceMetrics:
    """Tests for leading and trailing whitespace contamination detection."""

    def test_leading_whitespace_counted(self):
        """7. Leading whitespace (' abc') -> counted."""
        s = pd.Series([" abc", "def", " ghi", None])
        res = compute_whitespace_metrics(s)
        assert res["non_null_count"] == 3
        assert res["whitespace_count"] == 2
        assert res["whitespace_percentage"] == pytest.approx((2 / 3) * 100.0)

    def test_trailing_whitespace_counted(self):
        """8. Trailing whitespace ('abc ') -> counted."""
        s = pd.Series(["abc ", "def ", "ghi", "jkl"])
        res = compute_whitespace_metrics(s)
        assert res["non_null_count"] == 4
        assert res["whitespace_count"] == 2
        assert res["whitespace_percentage"] == pytest.approx(50.0)

    def test_internal_whitespace_only_not_counted(self):
        """9. Internal whitespace only ('a b c') -> NOT counted as leading/trailing issue."""
        s = pd.Series(["a b c", "New York", "San Francisco", "hello   world"])
        res = compute_whitespace_metrics(s)
        assert res["non_null_count"] == 4
        assert res["whitespace_count"] == 0
        assert res["whitespace_percentage"] == 0.0

    def test_clean_strings_zero_whitespace_issues(self):
        """10. Clean strings -> 0 whitespace issues."""
        s = pd.Series(["alpha", "beta", "gamma"])
        res = compute_whitespace_metrics(s)
        assert res["non_null_count"] == 3
        assert res["whitespace_count"] == 0
        assert res["whitespace_percentage"] == 0.0


class TestRuleValidityMetrics:
    """Tests for custom business-rule validation."""

    def test_custom_rule_passing_all_values(self):
        """11. Custom rule passing all values -> 0 rule errors."""
        s = pd.Series([10, 20, 30, 40])
        res = compute_rule_validity_metrics(s, rule_fn=lambda x: x > 0)
        assert res["non_null_count"] == 4
        assert res["rule_error_count"] == 0
        assert res["rule_error_percentage"] == 0.0

    def test_custom_rule_failing_subset_of_values(self):
        """12. Custom rule failing subset of values -> exact count and percentage."""
        s = pd.Series([10, -5, 20, -1, 30])
        res = compute_rule_validity_metrics(s, rule_fn=lambda x: x > 0)
        assert res["non_null_count"] == 5
        assert res["rule_error_count"] == 2
        assert res["rule_error_percentage"] == pytest.approx(40.0)

    def test_custom_rule_with_null_values(self):
        """13. Custom rule with null values -> nulls skipped and excluded from denominator."""
        s = pd.Series([10.0, -5.0, np.nan, None, 20.0, -2.0])
        res = compute_rule_validity_metrics(s, rule_fn=lambda x: x > 0)
        assert res["non_null_count"] == 4
        assert res["rule_error_count"] == 2
        assert res["rule_error_percentage"] == pytest.approx(50.0)

    def test_custom_rule_raising_exception(self):
        """14. Custom rule raising an exception (ValueError, TypeError) -> counted as rule failure."""
        s = pd.Series(["10", "20", "invalid_int", None])

        def strict_positive_int(val):
            return int(val) > 0

        res = compute_rule_validity_metrics(s, rule_fn=strict_positive_int)
        assert res["non_null_count"] == 3
        assert res["rule_error_count"] == 1
        assert res["rule_error_percentage"] == pytest.approx((1 / 3) * 100.0)

    def test_zero_denominator_safety_across_all_validity_functions(self):
        """15. Zero-denominator safety (N_non_null == 0) across all validity functions."""
        for s in [pd.Series([], dtype=object), pd.Series([None, np.nan, None])]:
            t_res = compute_type_validity_metrics(s)
            assert t_res["non_null_count"] == 0
            assert t_res["type_error_count"] == 0
            assert t_res["type_error_percentage"] == 0.0

            w_res = compute_whitespace_metrics(s)
            assert w_res["non_null_count"] == 0
            assert w_res["whitespace_count"] == 0
            assert w_res["whitespace_percentage"] == 0.0

            r_res = compute_rule_validity_metrics(s, rule_fn=lambda x: False)
            assert r_res["non_null_count"] == 0
            assert r_res["rule_error_count"] == 0
            assert r_res["rule_error_percentage"] == 0.0

            col_res = compute_column_validity(s, rule_fn=lambda x: False)
            assert col_res["non_null_count"] == 0
            assert col_res["type_error_percentage"] == 0.0
            assert col_res["whitespace_percentage"] == 0.0
            assert col_res["rule_error_percentage"] == 0.0


class TestValidityLegacyParity:
    """Direct regression comparison against generate_meta.py functions."""

    def test_parity_with_generate_meta(self):
        df = pd.DataFrame({
            "num_with_inf": [1.0, 2.0, np.inf, -np.inf, 3.0, np.nan],
            "num_coercible": ["10", "20", "abc", "30", "40", None],
            "cat_ws": [" apple", "banana ", "cherry pie", "date", None, " fig "],
            "dates": ["2023-01-01", "2023-01-02", "2023-01-03", "bad_date", "2023-01-05", None],
        })
        rules = {
            "num_with_inf": lambda x: np.isfinite(x) and x > 1.5,
            "cat_ws": lambda x: len(str(x).strip()) >= 5,
        }
        meta_df = generate_meta(["test_ds"], [df], rules=rules)

        for col in df.columns:
            row = meta_df[meta_df["column_name"] == col].iloc[0]
            rule_fn = rules.get(col)
            col_val = compute_column_validity(df[col], rule_fn=rule_fn)
            n_non_null = int(row["non_null_count"])

            assert col_val["non_null_count"] == n_non_null
            assert col_val["type_error_count"] == get_type_error_count(df, col)
            assert col_val["type_error_count"] == int(row["type_errors"])
            assert col_val["type_error_percentage"] == pytest.approx(
                (int(row["type_errors"]) / n_non_null) * 100.0
            )

            if pd.notna(row["whitespace_issues"]):
                assert col_val["whitespace_count"] == int(row["whitespace_issues"])
                assert col_val["whitespace_percentage"] == pytest.approx(
                    (int(row["whitespace_issues"]) / n_non_null) * 100.0
                )

            assert col_val["rule_error_count"] == rule_based_errors(df, col, rules=rules)
            assert col_val["rule_error_count"] == int(row["rule_errors"])
            assert col_val["rule_error_percentage"] == pytest.approx(
                (int(row["rule_errors"]) / n_non_null) * 100.0
            )
