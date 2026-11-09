"""Phase 3 tests: Native ProfileContext orchestration and LegacyMetaAdapter parity.

Verifies:
1. Native ``build_profile_context()`` construction across all 8 analytical domains.
2. Correct domain assignment (statistical vs distribution, temporal, drift, relationships).
3. Correct metric ownership and denominator semantics (N_non_null vs N_finite, +/-inf).
4. Continuous vs categorical entropy resolution via ``ColumnProfile.entropy``.
5. Empty and edge-case datasets (0 rows, 0 columns, N=1, all-null, unhashable, duplicate cols).
6. ``LegacyMetaAdapter.context_to_meta_df()`` 32-column schema and value parity.
7. ``ProfileContext``-driven profile table builders vs ``(df, meta_df)`` parity.
8. Lazy ``CleanalytixResult.metadata`` materialization via ``LegacyMetaAdapter``.
9. Single-pass execution (no duplicate metric calls; adapter computes zero metrics).
10. Input DataFrame immutability and non-recursive ``generate_meta()`` facade execution.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import cleanalytix as ca
from cleanalytix.generate_meta import (
    drift_metrics,
    generate_meta,
    get_entropy,
    get_metrics,
    get_type_error_count,
    rule_based_errors,
)
from cleanalytix.legacy import LEGACY_META_COLUMNS, LegacyMetaAdapter
from cleanalytix.profiling import (
    build_column_profile,
    build_dataset_profile,
    build_drift_profile,
    build_profile_context,
    build_relationship_profile,
    build_statistical_profile,
    build_temporal_profile,
)
from cleanalytix.types import (
    CleanalytixConfig,
    CleanalytixResult,
    ColumnCompleteness,
    ColumnDistribution,
    ColumnDrift,
    ColumnProfile,
    ColumnStatistical,
    ColumnTemporal,
    ColumnUniqueness,
    ColumnValidity,
    DatasetProfile,
    ProfileContext,
    RelationshipProfile,
)


def _build_reference_legacy_meta_df(
    dataset_name_list,
    dataset_list,
    new_dataset_list=None,
    rules=None,
) -> pd.DataFrame:
    """Reference legacy meta_df builder using raw get_metrics + drift_metrics."""
    rules = {} if rules is None else dict(rules)
    profile_list = new_dataset_list if new_dataset_list is not None else dataset_list
    data = get_metrics(profile_list, dataset_name_list, rules)
    meta_data = pd.DataFrame(data=data, columns=list(LEGACY_META_COLUMNS[:29]))
    meta_data.replace({pd.NaT: np.nan}, inplace=True)

    if new_dataset_list is None:
        drift_rows = []
        for df, ds_name in zip(dataset_list, dataset_name_list):
            for col in df.columns:
                drift_rows.append([ds_name, col, 0.0, 0.0, 0.0])
        meta_table_adv = pd.DataFrame(
            data=drift_rows,
            columns=[
                "dataset_name",
                "column_name",
                "kl_divergence",
                "psi_divergence",
                "js_divergence",
            ],
        )
    else:
        metrics = drift_metrics(dataset_name_list, dataset_list, new_dataset_list)
        meta_table_adv = pd.DataFrame(
            data=metrics,
            columns=[
                "dataset_name",
                "column_name",
                "kl_divergence",
                "psi_divergence",
                "js_divergence",
            ],
        )

    return pd.merge(
        meta_data,
        meta_table_adv,
        on=["dataset_name", "column_name"],
        how="inner",
    )


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """Representative mixed-role DataFrame covering all column roles."""
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "amount": np.concatenate([rng.normal(100.0, 15.0, n - 3), [np.nan, 950.0, -10.0]]),
        "rating": rng.choice([1, 2, 3, 4, 5], size=n),
        "is_active": rng.choice([0, 1], size=n),
        "flag_bool": rng.choice([True, False], size=n),
        "category": ["  alpha "] * 20 + ["beta"] * 25 + ["gamma"] * 13 + ["rare_x", None],
        "notes": [
            f"Customer submitted feedback message number {i} with detailed review text"
            for i in range(n)
        ],
        "created_at": pd.date_range("2025-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def mixed_prod_df(mixed_df: pd.DataFrame) -> pd.DataFrame:
    """Shifted production DataFrame aligned with mixed_df."""
    rng = np.random.default_rng(99)
    n = len(mixed_df)
    return pd.DataFrame({
        "customer_id": [f"CUST_{i + 100:04d}" for i in range(n)],
        "amount": rng.normal(135.0, 25.0, n),
        "rating": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.4, 0.3, 0.1, 0.1, 0.1]),
        "is_active": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
        "flag_bool": rng.choice([True, False], size=n),
        "category": ["alpha"] * 10 + ["beta"] * 30 + ["delta_new"] * 18 + [None, "rare_y"],
        "notes": [
            f"Updated production review text entry with multiple words {i}"
            for i in range(n)
        ],
        "created_at": pd.date_range("2025-03-01", periods=n, freq="D"),
    })


class TestNativeBuildProfileContext:
    """1-4. Native build_profile_context() construction, domain assignment, and ownership."""

    def test_build_profile_context_populates_all_eight_domains(
        self, mixed_df: pd.DataFrame, mixed_prod_df: pd.DataFrame
    ) -> None:
        rules = {"amount": lambda x: x >= 0}
        ctx = build_profile_context(
            mixed_prod_df,
            baseline_df=mixed_df,
            rules=rules,
            dataset_name="orders",
        )

        assert isinstance(ctx, ProfileContext)
        assert isinstance(ctx.dataset, DatasetProfile)
        assert ctx.dataset.dataset_name == "orders"
        assert ctx.dataset.total_rows == len(mixed_prod_df)
        assert ctx.dataset.total_columns == len(mixed_prod_df.columns)
        assert len(ctx.columns) == len(mixed_prod_df.columns)

        # Relationships domain (Domain 8) populated at dataset level
        assert isinstance(ctx.relationships, RelationshipProfile)
        assert ctx.dataset.relationships is ctx.relationships

        # Verify column-level domains (Domains 1-7)
        for col_name, col_prof in ctx.columns.items():
            assert isinstance(col_prof, ColumnProfile)
            assert isinstance(col_prof.completeness, ColumnCompleteness)
            assert isinstance(col_prof.uniqueness, ColumnUniqueness)
            assert isinstance(col_prof.validity, ColumnValidity)
            assert isinstance(col_prof.drift, ColumnDrift)

        # Continuous/discrete numeric -> statistical populated, distribution None
        assert ctx.columns["amount"].role == "continuous"
        assert isinstance(ctx.columns["amount"].statistical, ColumnStatistical)
        assert ctx.columns["amount"].distribution is None

        assert ctx.columns["rating"].role == "discrete"
        assert isinstance(ctx.columns["rating"].statistical, ColumnStatistical)
        assert ctx.columns["rating"].distribution is None

        # Categorical / binary / identifier / free_text -> distribution populated, statistical None
        for non_cont_col in ("customer_id", "is_active", "flag_bool", "category", "notes"):
            assert ctx.columns[non_cont_col].statistical is None
            assert isinstance(ctx.columns[non_cont_col].distribution, ColumnDistribution)

        # Temporal domain -> populated only for datetime column
        assert isinstance(ctx.columns["created_at"].temporal, ColumnTemporal)
        assert ctx.columns["created_at"].temporal.is_temporal is True
        for non_dt_col in ("customer_id", "amount", "rating", "is_active", "category"):
            assert ctx.columns[non_dt_col].temporal is None

    def test_baseline_only_context_has_no_drift_objects(self, mixed_df: pd.DataFrame) -> None:
        ctx = build_profile_context(mixed_df, baseline_df=None)
        for col_prof in ctx.column_profiles:
            assert col_prof.drift is None
        assert build_drift_profile(ctx) is None

    def test_include_relationships_false_skips_relationship_domain(
        self, mixed_df: pd.DataFrame
    ) -> None:
        ctx = build_profile_context(mixed_df, include_relationships=False)
        assert ctx.relationships is None
        assert ctx.dataset.relationships is None

    def test_infinity_and_missing_ownership_and_denominators(self) -> None:
        df = pd.DataFrame({
            "val": [10.0, 20.0, 30.0, np.inf, -np.inf, np.nan],
        })
        ctx = build_profile_context(df, rules={"val": lambda x: x > 0})
        col = ctx.columns["val"]

        # Completeness: N_total = 6, N_non_null = 5, null = 1
        assert col.completeness.non_null_count == 5
        assert col.completeness.null_count == 1

        # Validity: +/-inf are type errors in numeric columns; denominator is N_non_null = 5
        assert col.validity.type_error_count == 2
        assert col.validity.type_error_percentage == pytest.approx(40.0)
        # Rule errors: -inf fails x > 0; denominator is N_non_null = 5
        assert col.validity.rule_error_count == 1
        assert col.validity.rule_error_percentage == pytest.approx(20.0)

        # Statistical: count is N_finite = 3; infinite_count = 2
        assert col.statistical is not None
        assert col.statistical.count == 3
        assert col.statistical.infinite_count == 2
        assert col.statistical.mean == pytest.approx(20.0)

    def test_continuous_vs_categorical_entropy_ownership(self, mixed_df: pd.DataFrame) -> None:
        ctx = build_profile_context(mixed_df)

        # Continuous column gets entropy from statistical.continuous_entropy
        amount_prof = ctx.columns["amount"]
        assert amount_prof.statistical.continuous_entropy is not None
        assert amount_prof.entropy == amount_prof.statistical.continuous_entropy
        assert amount_prof.entropy == pytest.approx(
            get_entropy(mixed_df, "amount", column_role="continuous")
        )

        # Categorical column gets entropy from distribution.normalized_entropy
        cat_prof = ctx.columns["category"]
        assert cat_prof.distribution.normalized_entropy is not None
        assert cat_prof.entropy == cat_prof.distribution.normalized_entropy
        assert cat_prof.entropy == pytest.approx(
            get_entropy(mixed_df, "category", column_role="categorical")
        )


class TestEdgeCaseDatasets:
    """5. Empty and edge-case dataset handling."""

    def test_zero_row_dataframe(self) -> None:
        df = pd.DataFrame({"a": pd.Series(dtype=float), "b": pd.Series(dtype=object)})
        ctx = build_profile_context(df, dataset_name="empty_rows")
        assert ctx.dataset.total_rows == 0
        assert ctx.dataset.total_columns == 2
        meta = LegacyMetaAdapter.context_to_meta_df(ctx)
        assert list(meta.columns) == list(LEGACY_META_COLUMNS)
        assert len(meta) == 2

    def test_zero_column_dataframe(self) -> None:
        df = pd.DataFrame()
        ctx = build_profile_context(df, dataset_name="empty_cols")
        assert ctx.dataset.total_rows == 0
        assert ctx.dataset.total_columns == 0
        meta = LegacyMetaAdapter.context_to_meta_df(ctx)
        assert list(meta.columns) == list(LEGACY_META_COLUMNS)
        assert len(meta) == 0

    def test_single_row_dataframe(self) -> None:
        df = pd.DataFrame({"x": [42.0], "label": ["only_one"]})
        ctx = build_profile_context(df)
        assert ctx.columns["x"].statistical.mean == 42.0
        assert ctx.columns["x"].statistical.std is None
        meta = LegacyMetaAdapter.context_to_meta_df(ctx)
        assert meta.loc[meta["column_name"] == "x", "mean"].iloc[0] == 42.0
        assert pd.isna(meta.loc[meta["column_name"] == "x", "std"].iloc[0])

    def test_all_null_column_and_unhashable_column(self) -> None:
        df = pd.DataFrame({
            "all_null": [None, None, None],
            "unhashable": [[1, 2], [3, 4], [1, 2]],
        })
        ctx = build_profile_context(df)
        assert ctx.columns["all_null"].completeness.null_count == 3
        assert ctx.columns["unhashable"].uniqueness.distinct_count == 2
        meta = LegacyMetaAdapter.context_to_meta_df(ctx)
        assert len(meta) == 2

    def test_duplicate_columns_and_invalid_inputs_rejected(self) -> None:
        dup_df = pd.DataFrame([[1, 2]], columns=["dup", "dup"])
        with pytest.raises(ValueError, match="duplicate column names"):
            build_profile_context(dup_df)

        with pytest.raises(TypeError, match="pandas DataFrame"):
            build_profile_context([1, 2, 3])  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="pandas DataFrame"):
            build_profile_context(pd.DataFrame({"a": [1]}), baseline_df=[1])  # type: ignore[arg-type]


class TestLegacyMetaAdapterParity:
    """6. LegacyMetaAdapter 32-column schema and exact value parity."""

    def test_single_dataset_adapter_parity_with_reference(self, mixed_df: pd.DataFrame) -> None:
        rules = {
            "amount": lambda x: x > 0,
            "rating": lambda x: 1 <= x <= 5,
        }
        ctx = build_profile_context(
            mixed_df,
            rules=rules,
            dataset_name="mixed_ds",
            include_relationships=False,
        )
        adapted_meta = LegacyMetaAdapter.context_to_meta_df(ctx)
        ref_meta = _build_reference_legacy_meta_df(["mixed_ds"], [mixed_df], rules=rules)

        assert list(adapted_meta.columns) == list(LEGACY_META_COLUMNS)
        assert len(adapted_meta.columns) == 32
        pd.testing.assert_frame_equal(adapted_meta, ref_meta, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_drift_dataset_adapter_parity_with_reference(
        self, mixed_df: pd.DataFrame, mixed_prod_df: pd.DataFrame
    ) -> None:
        rules = {"amount": lambda x: x >= 0}
        ctx = build_profile_context(
            mixed_prod_df,
            baseline_df=mixed_df,
            rules=rules,
            dataset_name="prod_ds",
            include_relationships=False,
        )
        adapted_meta = LegacyMetaAdapter.context_to_meta_df(ctx)
        ref_meta = _build_reference_legacy_meta_df(
            ["prod_ds"], [mixed_df], new_dataset_list=[mixed_prod_df], rules=rules
        )

        assert list(adapted_meta.columns) == list(LEGACY_META_COLUMNS)
        pd.testing.assert_frame_equal(adapted_meta, ref_meta, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_multi_dataset_generate_meta_parity(
        self, mixed_df: pd.DataFrame, mixed_prod_df: pd.DataFrame
    ) -> None:
        df2 = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "z"]})
        df2_prod = pd.DataFrame({"a": [2, 3, 4, 5], "b": ["x", "w", "x", "z"]})

        meta_facade = generate_meta(
            ["ds1", "ds2"],
            [mixed_df, df2],
            new_dataset_list=[mixed_prod_df, df2_prod],
        )
        meta_ref = _build_reference_legacy_meta_df(
            ["ds1", "ds2"],
            [mixed_df, df2],
            new_dataset_list=[mixed_prod_df, df2_prod],
        )
        pd.testing.assert_frame_equal(meta_facade, meta_ref, check_exact=False, rtol=1e-12, atol=1e-12)

    def test_adapter_rejects_invalid_inputs(self) -> None:
        with pytest.raises(TypeError, match="ProfileContext"):
            LegacyMetaAdapter.context_to_meta_df("not_a_context")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="ProfileContext"):
            LegacyMetaAdapter.context_to_meta_df(["not_a_context"])  # type: ignore[list-item]


class TestProfileBuildersWithProfileContext:
    """7. ProfileContext-driven profile table builders match (df, meta_df) builders."""

    def test_all_profile_builders_match_legacy_path(
        self, mixed_df: pd.DataFrame, mixed_prod_df: pd.DataFrame
    ) -> None:
        cfg = CleanalytixConfig()
        ctx = build_profile_context(
            mixed_prod_df,
            baseline_df=mixed_df,
            config=cfg,
            dataset_name="prod",
        )
        meta_df = LegacyMetaAdapter.context_to_meta_df(ctx)

        col_from_ctx = build_column_profile(ctx)
        col_from_meta = build_column_profile(mixed_prod_df, meta_df)
        pd.testing.assert_frame_equal(col_from_ctx, col_from_meta)

        stat_from_ctx = build_statistical_profile(ctx)
        stat_from_meta = build_statistical_profile(mixed_prod_df, meta_df)
        pd.testing.assert_frame_equal(stat_from_ctx, stat_from_meta)

        rel_from_ctx = build_relationship_profile(ctx)
        rel_from_meta = build_relationship_profile(mixed_prod_df, config=cfg)
        pd.testing.assert_frame_equal(
            rel_from_ctx.pearson_correlations, rel_from_meta.pearson_correlations
        )
        assert rel_from_ctx.candidate_keys == rel_from_meta.candidate_keys

        temp_from_ctx = build_temporal_profile(ctx, config=cfg)
        temp_from_meta = build_temporal_profile(mixed_prod_df, meta_df, config=cfg)
        pd.testing.assert_frame_equal(temp_from_ctx, temp_from_meta)

        drift_from_ctx = build_drift_profile(ctx)
        drift_from_meta = build_drift_profile(mixed_prod_df, mixed_df, meta_df)
        assert drift_from_ctx is not None and drift_from_meta is not None
        pd.testing.assert_frame_equal(drift_from_ctx, drift_from_meta)

        dp_from_ctx = build_dataset_profile(mixed_prod_df, ctx, overall_score=88.5)
        assert dp_from_ctx.overall_score == 88.5
        assert dp_from_ctx.total_rows == len(mixed_prod_df)


class TestLazyMetadataAndSinglePassGuarantees:
    """8-11. Lazy CleanalytixResult.metadata, single-pass execution, immutability, non-recursion."""

    def test_cleanalytix_result_lazy_metadata_materialization(
        self, mixed_df: pd.DataFrame
    ) -> None:
        ctx = build_profile_context(mixed_df, dataset_name="lazy_ds")
        res = CleanalytixResult(
            overall_score=95.0,
            metadata=None,
            context=ctx,
        )

        # Before accessing .metadata, raw dict entry is None and _lazy_metadata_pending is True
        assert dict.get(res, "metadata") is None
        assert res._lazy_metadata_pending is True
        assert res.context is ctx

        # Accessing .metadata triggers LegacyMetaAdapter.context_to_meta_df(ctx)
        meta = res.metadata
        assert isinstance(meta, pd.DataFrame)
        assert list(meta.columns) == list(LEGACY_META_COLUMNS)
        assert res._lazy_metadata_pending is False
        assert res["metadata"] is meta

    def test_adapter_performs_zero_metric_computations(self, mixed_df: pd.DataFrame) -> None:
        ctx = build_profile_context(mixed_df)
        with (
            patch("cleanalytix.metrics.statistical.compute_numeric_statistics") as m_stat,
            patch("cleanalytix.metrics.statistical.detect_outliers") as m_out,
            patch("cleanalytix.metrics.statistical.compute_continuous_entropy") as m_ent,
            patch("cleanalytix.metrics.distribution.compute_categorical_distribution") as m_cat,
            patch("cleanalytix.metrics.validity.compute_column_validity") as m_val,
        ):
            meta = LegacyMetaAdapter.context_to_meta_df(ctx)
            assert len(meta) == len(mixed_df.columns)
            m_stat.assert_not_called()
            m_out.assert_not_called()
            m_ent.assert_not_called()
            m_cat.assert_not_called()
            m_val.assert_not_called()

    def test_input_dataframes_are_not_mutated(
        self, mixed_df: pd.DataFrame, mixed_prod_df: pd.DataFrame
    ) -> None:
        base_copy = mixed_df.copy(deep=True)
        prod_copy = mixed_prod_df.copy(deep=True)

        _ = build_profile_context(mixed_prod_df, baseline_df=mixed_df)
        _ = generate_meta(["ds"], [mixed_df], new_dataset_list=[mixed_prod_df])

        pd.testing.assert_frame_equal(mixed_df, base_copy)
        pd.testing.assert_frame_equal(mixed_prod_df, prod_copy)

    def test_generate_meta_is_non_recursive_and_never_calls_analyze(
        self, mixed_df: pd.DataFrame
    ) -> None:
        with patch("cleanalytix.pipeline.analyze", wraps=ca.analyze) as spy_analyze:
            meta = generate_meta(["ds"], [mixed_df])
            assert len(meta) == len(mixed_df.columns)
            spy_analyze.assert_not_called()

        # Also verify helpers delegate cleanly
        for col in mixed_df.columns:
            assert isinstance(get_entropy(mixed_df, col), float)
            assert isinstance(get_type_error_count(mixed_df, col), int)
            assert isinstance(rule_based_errors(mixed_df, col, {"amount": lambda x: x > 0}), int)


class TestPhase3Remediation:
    """Remediation tests addressing acceptance audit findings."""

    def test_single_call_to_categorical_drift(
        self, mixed_df: pd.DataFrame, mixed_prod_df: pd.DataFrame
    ) -> None:
        """Finding 1: Verify compute_categorical_drift is called exactly once per non-numeric column."""
        import cleanalytix.profiling

        with patch(
            "cleanalytix.profiling.compute_categorical_drift",
            wraps=cleanalytix.profiling.compute_categorical_drift,
        ) as spy:
            ctx = build_profile_context(mixed_prod_df, baseline_df=mixed_df)

            # Count non-numeric columns in prod that exist in base
            non_num_cols = [
                c
                for c in mixed_prod_df.columns
                if c in mixed_df.columns
                and (
                    not pd.api.types.is_numeric_dtype(mixed_df[c])
                    or pd.api.types.is_bool_dtype(mixed_df[c])
                    or ctx.get_column(c).role not in {"continuous", "discrete"}
                )
            ]
            assert spy.call_count == len(non_num_cols)
            called_names = [call[0][0].name for call in spy.call_args_list]
            assert sorted(called_names) == sorted(non_num_cols)

    def test_categorical_dtype_preservation(self) -> None:
        """Finding 3: Verify CategoricalDtype categories, order, and unused categories survive."""
        cat_dtype = pd.CategoricalDtype(
            categories=["bronze", "silver", "gold", "platinum"], ordered=True
        )
        df = pd.DataFrame({
            "status": pd.Series(["bronze", "gold", "bronze"], dtype=cat_dtype),
            "val": [10.0, 20.0, 30.0],
        })

        ctx = build_profile_context(df)
        col_prof = ctx.get_column("status")
        assert col_prof is not None
        assert col_prof.dtype == "category"
        assert col_prof.dtype_object == cat_dtype
        assert list(col_prof.dtype_object.categories) == ["bronze", "silver", "gold", "platinum"]
        assert col_prof.dtype_object.ordered is True

        # Reconstructed via LegacyMetaAdapter
        meta_df = LegacyMetaAdapter.context_to_meta_df(ctx)
        status_row = meta_df.loc[meta_df["column_name"] == "status"]
        recovered_dtype = status_row["dtype"].iloc[0]

        assert recovered_dtype == cat_dtype
        assert list(recovered_dtype.categories) == ["bronze", "silver", "gold", "platinum"]
        assert recovered_dtype.ordered is True

    def test_missing_value_categorical_drift_semantics(self) -> None:
        """Finding 2: Null values excluded from drift calculations, tracked in completeness."""
        base = pd.Series(["A", "A", "B", "B", None, None])
        prod = pd.Series(["A", "B", "B", "B", None, None])

        # build_profile_context drops nulls before computing categorical drift
        df_base = pd.DataFrame({"col": base})
        df_prod = pd.DataFrame({"col": prod})

        ctx = build_profile_context(df_prod, baseline_df=df_base)
        col = ctx.get_column("col")
        assert col is not None

        # Completeness strictly owns the missing values
        assert col.completeness.null_count == 2
        assert np.isclose(col.completeness.null_percentage, 33.3333, atol=0.01)

        # Drift operates on the non-null categories ("A", "B")
        assert col.drift is not None
        assert col.drift.sample_size_base == 4
        assert col.drift.sample_size_prod == 4
        assert "None" not in col.drift.unseen_categories
        assert "nan" not in col.drift.unseen_categories
        assert col.drift.unseen_categories_count == 0


class TestGenuineLegacyV020Fixtures:
    """Finding 4: Regression tests against frozen reference fixtures from commit 4b16c49."""

    @pytest.fixture(scope="class")
    def v020_fixtures(self):
        fixture_path = Path(__file__).parent / "sample_data" / "v020_4b16c49_reference_fixtures.pkl"
        with open(fixture_path, "rb") as f:
            return pickle.load(f)

    def test_v020_single_dataset_parity(self, v020_fixtures) -> None:
        """Verify identical 32-column metadata generation against 4b16c49 frozen fixtures."""
        ds = v020_fixtures["datasets"]
        refs = v020_fixtures["metadata_4b16c49"]

        # 1. Mixed single dataset (with rules)
        rules = {"num_continuous": lambda x: x > 0, "num_discrete": lambda x: x % 2 == 0}
        res_mixed = generate_meta(["mixed_base"], [ds["df_mixed_base"]], rules=rules)
        pd.testing.assert_frame_equal(res_mixed, refs["meta_mixed_single"])

        # 2. Empty dataset
        res_empty = generate_meta(["empty"], [ds["df_empty"]])
        pd.testing.assert_frame_equal(res_empty, refs["meta_empty"])

        # 3. Single-row dataset
        res_single = generate_meta(["single"], [ds["df_single"]])
        pd.testing.assert_frame_equal(res_single, refs["meta_single"])

        # 4. Null and infinite values dataset
        res_null_inf = generate_meta(
            ["null_inf"], [ds["df_null_inf"]], rules={"num": lambda x: x > 0}
        )
        pd.testing.assert_frame_equal(res_null_inf, refs["meta_null_inf"])

    def test_v020_drift_dataset_parity_and_intentional_drift_difference(
        self, v020_fixtures
    ) -> None:
        """Verify drift numerical parity and document intentional missing-category difference."""
        ds = v020_fixtures["datasets"]
        refs = v020_fixtures["metadata_4b16c49"]

        # Mixed drift dataset
        rules = {"num_continuous": lambda x: x > 0, "num_discrete": lambda x: x % 2 == 0}
        res_drift = generate_meta(
            ["mixed_drift"],
            [ds["df_mixed_base"]],
            new_dataset_list=[ds["df_mixed_prod"]],
            rules=rules,
        )
        ref_drift = refs["meta_mixed_drift"]

        # Non-drift columns match exactly
        for col in LEGACY_META_COLUMNS[:29]:
            pd.testing.assert_series_equal(res_drift[col], ref_drift[col])

        # Drift columns match within floating-point tolerance (machine epsilon <= 1e-12)
        for col in ["kl_divergence", "psi_divergence", "js_divergence"]:
            np.testing.assert_allclose(
                res_drift[col].values,
                ref_drift[col].values,
                atol=1e-12,
                err_msg=f"Drift metric {col} mismatch against 4b16c49 reference",
            )

        # Categorical drift on dataset with nulls: intentional semantic correction in v2.0
        # In 4b16c49, nulls were stringified into "None" and treated as an observed category.
        # In v2.0, nulls are excluded from categorical drift calculations and owned by completeness.
        res_cat_null = generate_meta(
            ["cat_null_drift"],
            [ds["df_cat_null_base"]],
            new_dataset_list=[ds["df_cat_null_prod"]],
        )
        ref_cat_null = refs["meta_cat_null_drift"]

        # Non-drift columns match exactly
        for col in LEGACY_META_COLUMNS[:29]:
            pd.testing.assert_series_equal(res_cat_null[col], ref_cat_null[col])

        # Intentional drift semantic difference documented:
        # In 4b16c49 with stringified "None": PSI ~ 0.201488, KL ~ 0.106907
        # In v2.0 with dropped nulls: PSI ~ 0.196346, KL ~ 0.101469
        assert not np.isclose(
            res_cat_null["psi_divergence"].iloc[0],
            ref_cat_null["psi_divergence"].iloc[0],
            atol=1e-4,
        )
        assert np.isclose(res_cat_null["psi_divergence"].iloc[0], 0.196346, atol=1e-4)
        assert np.isclose(ref_cat_null["psi_divergence"].iloc[0], 0.201488, atol=1e-4)
