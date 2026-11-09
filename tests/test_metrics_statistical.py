"""Unit and regression parity tests for continuous entropy in cleanalytix.metrics.statistical (Phase 1)."""

import numpy as np
import pandas as pd
import pytest

from cleanalytix.generate_meta import get_entropy
from cleanalytix.metrics.statistical import (
    compute_continuous_entropy,
    compute_numeric_statistics,
)


class TestContinuousEntropy:
    """Tests for compute_continuous_entropy and integration into compute_numeric_statistics."""

    def test_continuous_numeric_distribution_normal_and_uniform(self):
        """1. Continuous numeric distribution (normal / uniform) -> entropy in (0.0, 1.0]."""
        rng = np.random.default_rng(42)
        normal_s = pd.Series(rng.normal(loc=50.0, scale=10.0, size=500))
        uniform_s = pd.Series(np.linspace(0.0, 100.0, 500))

        ent_normal = compute_continuous_entropy(normal_s)
        ent_uniform = compute_continuous_entropy(uniform_s)

        assert 0.0 < ent_normal <= 1.0
        assert 0.0 < ent_uniform <= 1.0
        # Uniform distribution over equal-width bins has maximal entropy (~1.0)
        assert ent_uniform == pytest.approx(1.0, abs=1e-3)
        assert ent_uniform > ent_normal

    def test_skewed_continuous_distribution(self):
        """2. Skewed continuous distribution -> valid finite entropy in [0.0, 1.0]."""
        rng = np.random.default_rng(123)
        skewed_s = pd.Series(rng.exponential(scale=2.0, size=400))
        ent = compute_continuous_entropy(skewed_s)
        assert np.isfinite(ent)
        assert 0.0 < ent <= 1.0

    def test_constant_numeric_series(self):
        """3. Constant numeric series ([5, 5, 5, 5]) -> 0.0."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0])
        assert compute_continuous_entropy(s) == 0.0
        stats = compute_numeric_statistics(s)
        assert stats["continuous_entropy"] == 0.0

    def test_empty_series(self):
        """4. Empty series -> 0.0."""
        s = pd.Series([], dtype=float)
        assert compute_continuous_entropy(s) == 0.0
        stats = compute_numeric_statistics(s)
        assert stats["continuous_entropy"] == 0.0

    def test_all_null_series(self):
        """5. All-null series -> 0.0."""
        s = pd.Series([np.nan, None, np.nan], dtype=float)
        assert compute_continuous_entropy(s) == 0.0
        stats = compute_numeric_statistics(s)
        assert stats["continuous_entropy"] == 0.0

    def test_series_containing_pos_and_neg_infinity(self):
        """6. Series containing +-inf -> infinities excluded from N_finite; finite entropy returned."""
        finite_base = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
        with_inf = pd.Series([10.0, 20.0, np.inf, 30.0, 40.0, -np.inf, 50.0, 60.0, 70.0, 80.0])

        ent_base = compute_continuous_entropy(finite_base)
        ent_inf = compute_continuous_entropy(with_inf)

        assert np.isfinite(ent_inf)
        assert 0.0 < ent_inf <= 1.0
        assert ent_inf == pytest.approx(ent_base)

        # If only 1 finite value remains alongside infinities, N_finite < 2 -> 0.0
        single_finite = pd.Series([10.0, np.inf, -np.inf])
        assert compute_continuous_entropy(single_finite) == 0.0

    def test_small_sample_size_under_two(self):
        """7. Small sample size (N_finite < 2) -> 0.0."""
        s_one = pd.Series([42.0])
        s_one_with_nulls = pd.Series([42.0, np.nan, None])
        assert compute_continuous_entropy(s_one) == 0.0
        assert compute_continuous_entropy(s_one_with_nulls) == 0.0
        assert compute_numeric_statistics(s_one)["continuous_entropy"] == 0.0

    def test_highly_repeated_values_zero_iqr_scotts_rule_fallback(self):
        """8. Highly repeated values where IQR == 0 -> Scott's rule fallback works cleanly."""
        # 80% zeros so Q25 == Q75 == 0 (IQR == 0), but std > 0 from tail values
        vals = [0.0] * 40 + [10.0, 20.0, 30.0, 40.0, 50.0]
        s = pd.Series(vals)
        ent = compute_continuous_entropy(s)
        assert np.isfinite(ent)
        assert 0.0 < ent <= 1.0

    def test_extreme_numeric_magnitudes_above_1e150(self):
        """9. Extreme numeric magnitudes (> 1e150) -> no overflow or crash."""
        s_spread = pd.Series([1e160, 2e160, 3e160, 4e160, 5e160, 6e160, 7e160, 8e160])
        ent_spread = compute_continuous_entropy(s_spread)
        assert np.isfinite(ent_spread)
        assert 0.0 < ent_spread <= 1.0

        # Also test IQR == 0 with extreme magnitudes (> 1e150) triggering scaled Scott's rule
        s_iqr_zero_extreme = pd.Series([1e160] * 20 + [2e160, 3e160, 4e160])
        ent_iqr_zero = compute_continuous_entropy(s_iqr_zero_extreme)
        assert np.isfinite(ent_iqr_zero)
        assert 0.0 <= ent_iqr_zero <= 1.0

    def test_regression_parity_against_generate_meta_get_entropy(self):
        """10. Direct regression comparison against generate_meta.get_entropy() on numeric columns."""
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "normal": rng.normal(0.0, 1.0, 200),
            "uniform": rng.uniform(-50.0, 50.0, 200),
            "exponential": rng.exponential(5.0, 200),
            "discrete_num": rng.integers(0, 25, size=200).astype(float),
            "constant": np.full(200, 7.5),
            "zero_iqr": [1.0] * 170 + list(np.linspace(2.0, 31.0, 30)),
            "with_nulls_and_infs": [
                np.nan if i % 10 == 0 else (np.inf if i % 15 == 0 else float(i))
                for i in range(200)
            ],
            "extreme_scale": np.linspace(1e155, 9e155, 200),
        })

        for col in df.columns:
            expected_ent = get_entropy(df, col, column_role="continuous")
            actual_ent = compute_continuous_entropy(df[col])
            stats_ent = compute_numeric_statistics(df[col])["continuous_entropy"]

            assert actual_ent == pytest.approx(expected_ent, abs=1e-12), f"Mismatch on {col}"
            assert stats_ent == pytest.approx(expected_ent, abs=1e-12), f"Stats mismatch on {col}"
