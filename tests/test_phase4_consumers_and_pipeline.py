"""Phase 4 regression tests: Downstream consumer migration and pipeline inversion.

Verifies:
1. Native ``ProfileContext`` support and exact parity across all downstream consumers:
   - ``score_dataset`` / ``score_dataset_v2`` and ``build_metric_inventory``
   - ``generate_dataset_insights`` / ``generate_insights``
   - ``generate_recommendations`` / ``generate_cleaning_recommendations_v2`` and ``cleaning_recommendations``
   - ``summarize_dataset_health``
   - ``get_table_for_DQ_computation`` (including Architecture Decision B: canonical ``N_non_null``
     for ``ProfileContext`` vs historical ``N_rows`` for legacy ``meta_df`` DataFrame callers)
2. Main pipeline inversion in ``ca.analyze()``:
   - Builds ``ProfileContext`` first via ``build_profile_context()``
   - Never calls ``generate_meta()`` internally
   - Never calls ``LegacyMetaAdapter.context_to_meta_df()`` unless ``.metadata`` or
     ``meta_before_cleaning`` is explicitly accessed
3. Single-pass metric computation across profiling and scoring:
   - ``ca.analyze()`` does not re-run ``compute_duplicate_row_metrics``,
     ``compute_identifier_uniqueness``, ``compute_temporal_metrics``,
     ``build_relationship_profile``, or ``compute_numeric_correlations`` after
     ``build_profile_context()``
4. Preservation of all 44 ``METRIC_CATALOG`` metrics and strict DataFrame non-mutation.
"""

from __future__ import annotations

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

import cleanalytix as ca
from cleanalytix.cleaning_recommendations import (
    cleaning_recommendations,
    generate_cleaning_recommendations_v2,
    generate_recommendations,
)
from cleanalytix.get_table_for_DQ_computation import get_table_for_DQ_computation
from cleanalytix.insights import generate_dataset_insights, generate_insights
from cleanalytix.legacy import LEGACY_META_COLUMNS, LegacyMetaAdapter
from cleanalytix.profiling import build_profile_context
from cleanalytix.registry import METRIC_CATALOG
from cleanalytix.scoring import (
    build_metric_inventory,
    resolve_thresholds,
    score_dataset,
    score_dataset_v2,
)
from cleanalytix.summarize_dataset_health import summarize_dataset_health
from cleanalytix.types import (
    CleanalytixConfig,
    ConsistencyRule,
    ProfileContext,
    TimelinessConfig,
)


@pytest.fixture
def phase4_base_df() -> pd.DataFrame:
    """Rich multi-role baseline dataset for Phase 4 verification."""
    rng = np.random.default_rng(2026)
    n = 50
    return pd.DataFrame({
        "order_id": [f"ORD_{i:04d}" for i in range(n)],
        "amount": np.concatenate([rng.normal(200.0, 25.0, n - 4), [np.nan, np.nan, 1800.0, -5.0]]),
        "quantity": rng.choice([1, 2, 3, 4, 5], size=n),
        "is_priority": rng.choice([0, 1], size=n),
        "region": ["  NORTH "] * 15 + ["SOUTH"] * 20 + ["EAST"] * 12 + ["RARE_REG", None, "WEST"],
        "notes": [
            f"Customer order shipment processing note entry number {i} with details"
            for i in range(n)
        ],
        "order_date": pd.date_range("2025-01-01", periods=n, freq="D"),
    })


@pytest.fixture
def phase4_prod_df(phase4_base_df: pd.DataFrame) -> pd.DataFrame:
    """Shifted production dataset with duplicates, missingness, rules, and drift."""
    rng = np.random.default_rng(2027)
    n = len(phase4_base_df)
    df = pd.DataFrame({
        "order_id": [f"ORD_{i:04d}" for i in range(n - 1)] + ["ORD_0000"],
        "amount": np.concatenate([rng.normal(260.0, 40.0, n - 5), [np.nan, np.nan, -20.0, 2500.0, 100.0]]),
        "quantity": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.5, 0.2, 0.1, 0.1, 0.1]),
        "is_priority": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
        "region": ["  NORTH "] * 10 + ["SOUTH"] * 20 + ["NEW_REGION"] * 17 + ["RARE_X", None, None],
        "notes": [
            f"Updated production order delivery comment record {i} for audit"
            for i in range(n)
        ],
        "order_date": pd.date_range("2025-02-15", periods=n, freq="D"),
    })
    # Add an exact duplicate row
    df.loc[n - 1] = df.loc[0]
    return df


class TestConsumerMigrationParity:
    """1. Verify all migrated consumers accept ProfileContext natively and match legacy meta_df."""

    def test_score_dataset_and_metric_inventory_parity(
        self, phase4_base_df: pd.DataFrame, phase4_prod_df: pd.DataFrame
    ) -> None:
        cfg = CleanalytixConfig(
            rules={"amount": lambda x: x > 0},
            consistency_rules=[
                ConsistencyRule(
                    name="positive_total",
                    condition=lambda d: d["amount"].fillna(1.0) > 0,
                )
            ],
            timeliness_config=TimelinessConfig(
                reference_time="2025-04-15",
                freshness_days=30.0,
            ),
        )
        ctx = build_profile_context(
            phase4_prod_df,
            baseline_df=phase4_base_df,
            config=cfg,
            rules=cfg.rules,
            dataset_name="orders_prod",
        )
        meta_df = LegacyMetaAdapter.context_to_meta_df(ctx)

        res_ctx = score_dataset(
            phase4_prod_df, ctx, config=cfg, has_baseline=True, return_details=True
        )
        res_v2 = score_dataset_v2(
            phase4_prod_df, ctx, config=cfg, has_baseline=True, return_details=True
        )
        res_meta = score_dataset(
            phase4_prod_df, meta_df, config=cfg, has_baseline=True, return_details=True
        )

        # Overall score and column scores identical
        assert res_ctx[0] == res_meta[0]
        assert res_v2[0] == res_meta[0]
        assert res_ctx[2] == res_meta[2]

        # Dimension scores identical across all 6 dimensions
        for dim_name in res_meta[1]:
            assert res_ctx[1][dim_name].to_dict() == res_meta[1][dim_name].to_dict()

        # Scoring matrix, dimension summary, metric inventory, and audit trail identical
        pd.testing.assert_frame_equal(res_ctx[3], res_meta[3])
        pd.testing.assert_frame_equal(res_ctx[4], res_meta[4])
        pd.testing.assert_frame_equal(res_ctx[5], res_meta[5])
        assert res_ctx[6] == res_meta[6]

    def test_score_dataset_accepts_profile_context_as_first_argument(
        self, phase4_base_df: pd.DataFrame
    ) -> None:
        cfg = CleanalytixConfig(
            timeliness_config=TimelinessConfig(reference_time="2025-03-01")
        )
        ctx = build_profile_context(phase4_base_df, config=cfg)
        s_ctx_only, dims_ctx_only, cols_ctx_only = score_dataset(ctx, config=cfg)
        s_with_df, dims_with_df, cols_with_df = score_dataset(phase4_base_df, ctx, config=cfg)

        assert s_ctx_only == s_with_df
        assert cols_ctx_only == cols_with_df
        for k in dims_with_df:
            assert dims_ctx_only[k].to_dict() == dims_with_df[k].to_dict()

    def test_insights_and_recommendations_parity(
        self, phase4_base_df: pd.DataFrame, phase4_prod_df: pd.DataFrame
    ) -> None:
        cfg = CleanalytixConfig(rules={"amount": lambda x: x > 0})
        ctx = build_profile_context(
            phase4_prod_df,
            baseline_df=phase4_base_df,
            config=cfg,
            rules=cfg.rules,
            dataset_name="orders_prod",
        )
        meta_df = LegacyMetaAdapter.context_to_meta_df(ctx)
        _, dim_scores, _ = score_dataset(
            phase4_prod_df, ctx, config=cfg, has_baseline=True
        )
        thresholds = resolve_thresholds(cfg.thresholds)

        ins_ctx = generate_dataset_insights(
            ctx, thresholds, has_baseline=True, quality_dimensions=dim_scores
        )
        ins_alias = generate_insights(
            ctx, thresholds, has_baseline=True, quality_dimensions=dim_scores
        )
        ins_meta = generate_dataset_insights(
            meta_df, thresholds, has_baseline=True, quality_dimensions=dim_scores
        )

        assert [i.to_dict() for i in ins_ctx] == [i.to_dict() for i in ins_meta]
        assert [i.to_dict() for i in ins_alias] == [i.to_dict() for i in ins_meta]

        recs_ctx, recs_df_ctx = generate_recommendations(ctx, insights=ins_ctx)
        recs_v2, recs_df_v2 = generate_cleaning_recommendations_v2(ctx, insights=ins_ctx)
        recs_meta, recs_df_meta = generate_recommendations(meta_df, insights=ins_meta)

        assert [r.to_dict() for r in recs_ctx] == [r.to_dict() for r in recs_meta]
        assert [r.to_dict() for r in recs_v2] == [r.to_dict() for r in recs_meta]
        pd.testing.assert_frame_equal(recs_df_ctx, recs_df_meta)
        pd.testing.assert_frame_equal(recs_df_v2, recs_df_meta)

        clean_recs_ctx = cleaning_recommendations(ctx)
        clean_recs_meta = cleaning_recommendations(meta_df)
        pd.testing.assert_frame_equal(clean_recs_ctx, clean_recs_meta)

    def test_summarize_dataset_health_and_dq_table_decision_b(
        self, phase4_prod_df: pd.DataFrame
    ) -> None:
        rules = {"amount": lambda x: x > 0}
        ctx = build_profile_context(
            phase4_prod_df,
            rules=rules,
            dataset_name="prod_ds",
        )
        meta_df = LegacyMetaAdapter.context_to_meta_df(ctx)

        health_ctx = summarize_dataset_health(ctx)
        health_meta = summarize_dataset_health(meta_df)
        pd.testing.assert_frame_equal(health_ctx, health_meta, check_dtype=False)

        # Architecture Decision B:
        # 1) Native v2.0 ProfileContext path uses canonical N_non_null denominators
        dq_ctx = get_table_for_DQ_computation(ctx)
        amount_cp = ctx.get_column("amount")
        assert amount_cp is not None and amount_cp.completeness is not None
        assert amount_cp.completeness.null_count > 0  # Has missing values so N_non_null < N_rows

        amount_row_ctx = dq_ctx[dq_ctx["column_name"] == "amount"].iloc[0]
        expected_rule_pct_non_null = (
            amount_cp.validity.rule_error_count / amount_cp.completeness.non_null_count
        ) * 100.0
        assert np.isclose(amount_row_ctx["rule_errors_percent"], expected_rule_pct_non_null)

        # 2) Legacy meta_df DataFrame path preserves historical N_rows denominator
        dq_legacy = get_table_for_DQ_computation(meta_df, [phase4_prod_df], ["prod_ds"])
        amount_row_legacy = dq_legacy[dq_legacy["column_name"] == "amount"].iloc[0]
        expected_rule_pct_rows = (
            amount_cp.validity.rule_error_count / len(phase4_prod_df)
        ) * 100.0
        assert np.isclose(amount_row_legacy["rule_errors_percent"], expected_rule_pct_rows)
        assert amount_row_ctx["rule_errors_percent"] > amount_row_legacy["rule_errors_percent"]


class TestPipelineInversionAndLazyMetadata:
    """2 & 3. Verify ca.analyze() pipeline inversion, lazy metadata, and zero duplicate work."""

    def test_analyze_never_calls_generate_meta_or_eager_adapter(
        self, phase4_base_df: pd.DataFrame, phase4_prod_df: pd.DataFrame
    ) -> None:
        with (
            patch("cleanalytix.pipeline.generate_meta") as mock_gen_meta,
            patch.object(
                LegacyMetaAdapter,
                "context_to_meta_df",
                wraps=LegacyMetaAdapter.context_to_meta_df,
            ) as mock_adapter,
        ):
            res_single = ca.analyze(phase4_base_df)
            res_drift = ca.analyze(phase4_prod_df, baseline_df=phase4_base_df)

            # Neither generate_meta nor LegacyMetaAdapter.context_to_meta_df was called during analyze()
            mock_gen_meta.assert_not_called()
            mock_adapter.assert_not_called()

            # Accessing native v2.0 attributes and tables still does NOT trigger adapter
            _ = res_single.overall_score
            _ = res_single.quality_dimensions
            _ = res_single.column_scores
            _ = res_single.dataset_profile
            _ = res_single.column_profile
            _ = res_single.statistical_profile
            _ = res_single.scoring_matrix
            _ = res_single.dimension_summary
            _ = res_single.metric_inventory
            _ = res_single.relationships
            _ = res_single.temporal_profile
            _ = res_single.anomalies
            _ = res_single.recommendations
            _ = res_single.recommendations_table
            _ = res_single.insights
            _ = res_single.health_summary
            _ = res_single.metrics_table
            _ = res_drift.drift_profile
            assert isinstance(res_single.context, ProfileContext)
            assert isinstance(res_drift.context, ProfileContext)
            mock_adapter.assert_not_called()

            # Accessing .metadata lazily invokes LegacyMetaAdapter.context_to_meta_df once
            meta_single = res_single.metadata
            assert mock_adapter.call_count == 1
            assert list(meta_single.columns) == list(LEGACY_META_COLUMNS)
            assert len(meta_single) == len(phase4_base_df.columns)

            # Second access uses cached DataFrame
            _ = res_single.metadata
            _ = res_single.base_data["meta_before_cleaning"]
            assert mock_adapter.call_count == 1

            # Accessing baseline and prod meta_before_cleaning on drift result materializes on demand
            base_meta_drift = res_drift.base_data["meta_before_cleaning"]
            assert mock_adapter.call_count == 2
            assert list(base_meta_drift.columns) == list(LEGACY_META_COLUMNS)

            prod_meta_drift = res_drift.prod_data["meta_before_cleaning"]
            assert mock_adapter.call_count == 3
            assert prod_meta_drift is res_drift.metadata
            assert mock_adapter.call_count == 3

    def test_analyze_performs_zero_duplicate_metric_computations(
        self, phase4_base_df: pd.DataFrame
    ) -> None:
        import cleanalytix.metrics.completeness as comp_mod
        import cleanalytix.metrics.temporal as temp_mod
        import cleanalytix.metrics.uniqueness as uniq_mod
        import cleanalytix.profiling as prof_mod

        with (
            patch(
                "cleanalytix.profiling.compute_completeness_metrics",
                wraps=comp_mod.compute_completeness_metrics,
            ) as m_comp,
            patch(
                "cleanalytix.profiling.compute_duplicate_row_metrics",
                wraps=uniq_mod.compute_duplicate_row_metrics,
            ) as m_dup_prof,
            patch(
                "cleanalytix.scoring.compute_duplicate_row_metrics",
                wraps=uniq_mod.compute_duplicate_row_metrics,
            ) as m_dup_score,
            patch(
                "cleanalytix.scoring.compute_identifier_uniqueness",
                wraps=uniq_mod.compute_identifier_uniqueness,
            ) as m_id_score,
            patch(
                "cleanalytix.profiling.compute_temporal_metrics",
                wraps=temp_mod.compute_temporal_metrics,
            ) as m_temp_prof,
            patch(
                "cleanalytix.scoring.compute_temporal_metrics",
                wraps=temp_mod.compute_temporal_metrics,
            ) as m_temp_score,
            patch(
                "cleanalytix.profiling.build_relationship_profile",
                wraps=prof_mod.build_relationship_profile,
            ) as m_rel,
        ):
            res = ca.analyze(phase4_base_df)
            assert res.overall_score > 0.0

            # Dataset-level metrics computed exactly once during build_profile_context
            assert m_comp.call_count == 1
            assert m_dup_prof.call_count == 1
            assert m_rel.call_count == 1
            # Temporal metrics computed once for the single datetime column ("order_date")
            assert m_temp_prof.call_count == 1

            # Downstream scoring never recomputes duplicate rows, identifier uniqueness, or temporal metrics
            m_dup_score.assert_not_called()
            m_id_score.assert_not_called()
            m_temp_score.assert_not_called()

    def test_metric_catalog_and_dataframe_immutability_preserved(
        self, phase4_base_df: pd.DataFrame, phase4_prod_df: pd.DataFrame
    ) -> None:
        assert len(METRIC_CATALOG) == 44

        base_copy = phase4_base_df.copy(deep=True)
        prod_copy = phase4_prod_df.copy(deep=True)

        _ = ca.analyze(phase4_prod_df, baseline_df=phase4_base_df)

        pd.testing.assert_frame_equal(phase4_base_df, base_copy)
        pd.testing.assert_frame_equal(phase4_prod_df, prod_copy)
