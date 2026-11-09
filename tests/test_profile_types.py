"""Focused unit tests for Cleanalytix v2.0 Phase 2 Typed Contracts and ProfileContext."""

import importlib
import sys
import pandas as pd
import pytest

import cleanalytix
import cleanalytix.types as ca_types
from cleanalytix.types import (
    AnalysisLevel,
    CleanalytixConfig,
    CleanalytixResult,
    ColumnCompleteness,
    ColumnDistribution,
    ColumnDrift,
    ColumnProfile,
    ColumnRole,
    ColumnStatistical,
    ColumnTemporal,
    ColumnUniqueness,
    ColumnValidity,
    DatasetProfile,
    DimensionScore,
    FeatureRetiredError,
    InsightItem,
    InsightSeverity,
    ProfileContext,
    QualityDimension,
    RecommendationItem,
    RelationshipProfile,
    ScoreMode,
    SemanticType,
)


# ---------------------------------------------------------------------------
# 1. Domain Contract Tests
# ---------------------------------------------------------------------------


def test_instantiate_all_domain_structures_with_defaults():
    """All 8 domain structures instantiate cleanly with safe default values."""
    comp = ColumnCompleteness()
    uniq = ColumnUniqueness()
    val = ColumnValidity()
    stat = ColumnStatistical()
    dist = ColumnDistribution()
    temp = ColumnTemporal()
    drift = ColumnDrift()
    rel = RelationshipProfile()

    # Completeness defaults
    assert comp.total_count == 0
    assert comp.non_null_count == 0
    assert comp.null_count == 0
    assert comp.null_percentage == 0.0
    assert comp.completeness_ratio == 1.0
    assert comp.empty_string_count == 0
    assert comp.missing_count == 0
    assert comp.missing_percent == 0.0

    # Uniqueness defaults
    assert uniq.distinct_count == 0
    assert uniq.distinct_ratio == 0.0
    assert uniq.non_null_distinct_ratio == 0.0
    assert uniq.unique_percentage == 0.0
    assert uniq.duplicate_count == 0
    assert uniq.duplicate_percentage == 0.0
    assert uniq.is_unique is False
    assert uniq.is_constant is False
    assert uniq.is_near_constant is False
    assert uniq.is_primary_key_candidate is False
    assert uniq.top_frequency_ratio == 0.0
    assert uniq.top_value is None
    assert uniq.uniqueness_ratio == 0.0
    assert uniq.unique_count == 0
    assert uniq.unique_percent == 0.0

    # Validity defaults
    assert val.inferred_type == "categorical"
    assert val.type_error_count == 0
    assert val.type_error_percentage == 0.0
    assert val.whitespace_count == 0
    assert val.whitespace_percentage == 0.0
    assert val.rule_error_count == 0
    assert val.rule_error_percentage == 0.0

    # Statistical defaults
    assert stat.count == 0
    assert stat.infinite_count == 0
    assert stat.mean is None
    assert stat.median is None
    assert stat.std is None
    assert stat.variance is None
    assert stat.min is None
    assert stat.max is None
    assert stat.p01 is None
    assert stat.p05 is None
    assert stat.q25 is None
    assert stat.q75 is None
    assert stat.p95 is None
    assert stat.p99 is None
    assert stat.iqr is None
    assert stat.mad is None
    assert stat.skewness is None
    assert stat.skewness_class is None
    assert stat.kurtosis is None
    assert stat.kurtosis_class is None
    assert stat.coefficient_of_variation is None
    assert stat.zero_count == 0
    assert stat.zero_ratio == 0.0
    assert stat.negative_count == 0
    assert stat.negative_ratio == 0.0
    assert stat.continuous_entropy == 0.0
    assert stat.outlier_method == "iqr"
    assert stat.outlier_count == 0
    assert stat.outlier_ratio == 0.0
    assert stat.outlier_percentage == 0.0
    assert stat.outlier_lower_bound is None
    assert stat.outlier_upper_bound is None
    assert stat.zscore_outliers == {}
    assert stat.iqr_outliers == {}
    assert stat.modified_zscore_outliers == {}
    assert stat.min_value is None
    assert stat.max_value is None
    assert stat.kurtosis_fisher is None

    # Distribution defaults
    assert dist.unique_categories == 0
    assert dist.shannon_entropy == 0.0
    assert dist.normalized_entropy == 0.0
    assert dist.hhi == 0.0
    assert dist.gini_simpson_diversity == 0.0
    assert dist.top_category is None
    assert dist.top_category_ratio == 0.0
    assert dist.top_category_percentage == 0.0
    assert dist.is_dominated is False
    assert dist.rare_categories_count == 0
    assert dist.rare_categories_ratio == 0.0
    assert dist.rare_categories_percentage == 0.0
    assert dist.top_frequencies == {}
    assert dist.mode is None
    assert dist.mode_percentage == 0.0

    # Temporal defaults
    assert temp.is_temporal is False
    assert temp.parse_rate == 0.0
    assert temp.min_date is None
    assert temp.max_date is None
    assert temp.time_span_days is None
    assert temp.is_monotonic_increasing is False
    assert temp.future_date_count == 0
    assert temp.future_date_ratio == 0.0
    assert temp.median_step_seconds is None
    assert temp.max_gap_seconds is None
    assert temp.freshness_days is None
    assert temp.inferred_cadence is None
    assert temp.freshness_status is None

    # Drift defaults
    assert drift.psi == 0.0
    assert drift.kl_divergence == 0.0
    assert drift.reverse_kl_divergence == 0.0
    assert drift.js_divergence == 0.0
    assert drift.ks_statistic is None
    assert drift.ks_p_value is None
    assert drift.wasserstein_distance is None
    assert drift.normalized_wasserstein is None
    assert drift.unseen_categories == []
    assert drift.unseen_categories_count == 0
    assert drift.unseen_categories_ratio == 0.0
    assert drift.drift_detected is False
    assert drift.drift_severity == "none"
    assert drift.sample_size_base == 0
    assert drift.sample_size_prod == 0

    # RelationshipProfile defaults
    assert isinstance(rel.pearson_correlations, pd.DataFrame)
    assert isinstance(rel.spearman_correlations, pd.DataFrame)
    assert isinstance(rel.cramers_v, pd.DataFrame)
    assert isinstance(rel.high_correlations, pd.DataFrame)
    assert isinstance(rel.redundant_columns, pd.DataFrame)
    assert isinstance(rel.missingness_cooccurrence, pd.DataFrame)
    assert rel.candidate_keys == {}
    assert rel.duplicate_columns == []
    assert rel.near_duplicate_columns == []
    assert rel.missingness_correlations == {}


def test_domain_structures_accept_populated_fields_and_aliases():
    """Domain structures store explicit metric values and expose consistent read-only aliases."""
    comp = ColumnCompleteness(
        total_count=100,
        non_null_count=92,
        null_count=8,
        null_percentage=8.0,
        completeness_ratio=0.92,
        empty_string_count=3,
    )
    assert comp.missing_count == 8
    assert comp.missing_percent == 8.0

    uniq = ColumnUniqueness(
        distinct_count=80,
        distinct_ratio=0.80,
        non_null_distinct_ratio=80 / 92,
        unique_percentage=86.9565,
        duplicate_count=12,
        duplicate_percentage=13.0435,
        is_unique=False,
        is_constant=False,
        is_near_constant=False,
        is_primary_key_candidate=False,
        top_frequency_ratio=0.05,
        top_value="A",
    )
    assert uniq.uniqueness_ratio == pytest.approx(80 / 92)
    assert uniq.unique_count == 80
    assert uniq.unique_percent == 86.9565

    val = ColumnValidity(
        inferred_type="numeric",
        type_error_count=2,
        type_error_percentage=2.1739,
        whitespace_count=0,
        whitespace_percentage=0.0,
        rule_error_count=1,
        rule_error_percentage=1.087,
    )
    assert val.inferred_type == "numeric"
    assert val.rule_error_count == 1
    assert val.rule_error_percentage == 1.087

    stat = ColumnStatistical(
        count=90,
        infinite_count=2,
        mean=50.5,
        median=50.0,
        std=12.3,
        variance=151.29,
        min=10.0,
        max=99.0,
        p01=11.0,
        p05=15.0,
        q25=35.0,
        q75=65.0,
        p95=85.0,
        p99=97.0,
        iqr=30.0,
        mad=10.5,
        skewness=0.12,
        skewness_class="symmetric",
        kurtosis=-0.45,
        kurtosis_class="platykurtic",
        coefficient_of_variation=0.2435,
        zero_count=1,
        zero_ratio=1 / 90,
        negative_count=0,
        negative_ratio=0.0,
        continuous_entropy=3.14159,
        outlier_method="iqr",
        outlier_count=3,
        outlier_ratio=3 / 90,
        outlier_percentage=3.3333,
        outlier_lower_bound=-10.0,
        outlier_upper_bound=110.0,
        iqr_outliers={"outlier_count": 3, "outlier_ratio": 3 / 90},
    )
    assert stat.min_value == 10.0
    assert stat.max_value == 99.0
    assert stat.kurtosis_fisher == -0.45
    assert stat.infinite_count == 2

    dist = ColumnDistribution(
        unique_categories=5,
        shannon_entropy=1.85,
        normalized_entropy=0.797,
        hhi=0.28,
        gini_simpson_diversity=0.72,
        top_category="North",
        top_category_ratio=0.42,
        top_category_percentage=42.0,
        is_dominated=False,
        rare_categories_count=1,
        rare_categories_ratio=0.20,
        rare_categories_percentage=20.0,
        top_frequencies={"North": {"count": 42, "ratio": 0.42}},
    )
    assert dist.mode == "North"
    assert dist.mode_percentage == 42.0


def test_mutable_defaults_not_shared_across_instances():
    """Mutable collection fields use default_factory and are strictly isolated across instances."""
    s1 = ColumnStatistical()
    s2 = ColumnStatistical()
    s1.iqr_outliers["outlier_count"] = 5
    s1.zscore_outliers["outlier_count"] = 2
    s1.modified_zscore_outliers["outlier_count"] = 3
    assert s2.iqr_outliers == {}
    assert s2.zscore_outliers == {}
    assert s2.modified_zscore_outliers == {}

    d1 = ColumnDistribution()
    d2 = ColumnDistribution()
    d1.top_frequencies["cat_a"] = {"count": 10}
    assert d2.top_frequencies == {}

    dr1 = ColumnDrift()
    dr2 = ColumnDrift()
    dr1.unseen_categories.append("NEW_VAL")
    assert dr2.unseen_categories == []

    r1 = RelationshipProfile()
    r2 = RelationshipProfile()
    r1.candidate_keys["single"] = ["id"]
    r1.duplicate_columns.append(("a", "b"))
    r1.near_duplicate_columns.append({"col_1": "a", "col_2": "b", "match_ratio": 0.995})
    r1.missingness_correlations["a__b"] = 0.85
    assert r2.candidate_keys == {}
    assert r2.duplicate_columns == []
    assert r2.near_duplicate_columns == []
    assert r2.missingness_correlations == {}
    assert r1.pearson_correlations is not r2.pearson_correlations
    assert r1.high_correlations is not r2.high_correlations
    assert r1.redundant_columns is not r2.redundant_columns


def test_domain_ownership_no_cross_domain_duplicate_fields():
    """Verify single authoritative ownership of metrics across domain dataclasses."""
    completeness_fields = set(ColumnCompleteness.__dataclass_fields__.keys())
    uniqueness_fields = set(ColumnUniqueness.__dataclass_fields__.keys())
    validity_fields = set(ColumnValidity.__dataclass_fields__.keys())
    statistical_fields = set(ColumnStatistical.__dataclass_fields__.keys())
    distribution_fields = set(ColumnDistribution.__dataclass_fields__.keys())

    # Completeness owns null/non-null counts; Statistical owns finite count & infinite_count
    assert "infinite_count" not in completeness_fields
    assert "infinite_count" in statistical_fields
    assert "null_count" not in statistical_fields
    assert "null_count" not in validity_fields
    assert "non_null_count" not in validity_fields

    # Continuous entropy belongs only to ColumnStatistical; Shannon entropy only to ColumnDistribution
    assert "continuous_entropy" in statistical_fields
    assert "continuous_entropy" not in distribution_fields
    assert "shannon_entropy" in distribution_fields
    assert "shannon_entropy" not in statistical_fields

    # Outliers belong to ColumnStatistical, not ColumnValidity
    assert "outlier_count" in statistical_fields
    assert "outlier_count" not in validity_fields

    # Dataset-level row duplicates belong to DatasetProfile, not ColumnUniqueness
    assert "duplicate_rows_count" not in uniqueness_fields


# ---------------------------------------------------------------------------
# 2. Composition Tests
# ---------------------------------------------------------------------------


def test_column_profile_composes_multiple_domains_and_entropy_property():
    """ColumnProfile composes all 7 column-level domains and resolves .entropy by role."""
    num_col = ColumnProfile(
        name="amount",
        dtype="float64",
        role="continuous",
        semantic_type="numeric",
        completeness=ColumnCompleteness(total_count=100, non_null_count=95, null_count=5),
        uniqueness=ColumnUniqueness(distinct_count=90),
        validity=ColumnValidity(inferred_type="numeric"),
        statistical=ColumnStatistical(count=95, continuous_entropy=4.12),
        distribution=ColumnDistribution(shannon_entropy=6.30, normalized_entropy=0.95),
    )
    assert num_col.entropy == 4.12

    cat_col = ColumnProfile(
        name="region",
        dtype="object",
        role="categorical",
        semantic_type="categorical",
        completeness=ColumnCompleteness(total_count=100, non_null_count=100, null_count=0),
        uniqueness=ColumnUniqueness(distinct_count=4),
        validity=ColumnValidity(inferred_type="categorical"),
        distribution=ColumnDistribution(shannon_entropy=1.95, normalized_entropy=0.975),
    )
    assert cat_col.statistical is None
    assert cat_col.temporal is None
    assert cat_col.drift is None
    assert cat_col.entropy == 0.975


def test_column_profile_partial_domains_remain_none():
    """Inapplicable domains on ColumnProfile remain None rather than fabricated zero objects."""
    col = ColumnProfile(
        name="created_at",
        dtype="datetime64[ns]",
        role="datetime",
        semantic_type="datetime",
        completeness=ColumnCompleteness(total_count=50, non_null_count=50),
        temporal=ColumnTemporal(is_temporal=True, parse_rate=1.0),
    )
    assert col.completeness is not None
    assert col.temporal is not None
    assert col.statistical is None
    assert col.distribution is None
    assert col.drift is None
    assert col.entropy is None


def test_dataset_profile_composition_and_backward_compatibility():
    """DatasetProfile supports existing v0.2.0 fields/methods and v2.0 column/relationship composition."""
    c1 = ColumnProfile(name="id", dtype="int64", role="identifier")
    c2 = ColumnProfile(name="price", dtype="float64", role="continuous")
    rel = RelationshipProfile(candidate_keys={"single_column": ["id"]})

    # Instantiate with sequence of ColumnProfiles
    ds = DatasetProfile(
        dataset_name="orders",
        total_rows=1000,
        total_columns=2,
        total_cells=2000,
        memory_bytes=52428,
        memory_mb=0.05,
        duplicate_rows_count=0,
        duplicate_rows_percent=0.0,
        complete_rows_count=1000,
        complete_rows_percent=100.0,
        columns=[c1, c2],
        relationships=rel,
    )

    assert ds.row_count == 1000
    assert ds.column_count == 2
    assert ds.cell_count == 2000
    assert ds.memory_usage_bytes == 52428
    assert ds.duplicate_row_count == 0
    assert ds.duplicate_row_percentage == 0.0
    assert ds.column_order == ["id", "price"]
    assert list(ds.columns.keys()) == ["id", "price"]
    assert ds.column_profiles == [c1, c2]
    assert ds.get_column("id") is c1
    assert ds.get_column("missing") is None
    assert ds.relationships is rel

    # Backward-compatible methods must remain intact
    d_dict = ds.to_dict()
    assert d_dict["total_rows"] == 1000
    assert d_dict["total_columns"] == 2
    assert "columns" not in d_dict  # scalar summary dict remains clean for DataFrame construction

    df_summary = ds.to_dataframe()
    assert isinstance(df_summary, pd.DataFrame)
    assert df_summary.shape[0] == 1
    assert df_summary.loc[0, "total_rows"] == 1000


def test_profile_context_composition_and_inheritance_from_dataset():
    """ProfileContext composes DatasetProfile, ColumnProfiles, RelationshipProfile, and reference_profile."""
    c1 = ColumnProfile(name="user_id", dtype="int64", role="identifier")
    c2 = ColumnProfile(name="country", dtype="object", role="categorical")
    rel = RelationshipProfile(candidate_keys={"single_column": ["user_id"]})

    ds = DatasetProfile(
        dataset_name="users_prod",
        total_rows=500,
        total_columns=2,
        columns=[c1, c2],
        relationships=rel,
    )

    ref_ctx = ProfileContext(
        dataset=DatasetProfile(dataset_name="users_base", total_rows=450, total_columns=2)
    )

    # ProfileContext inherits columns and relationships from DatasetProfile when not duplicated
    ctx = ProfileContext(dataset=ds, reference_profile=ref_ctx)

    assert ctx.dataset is ds
    assert ctx.dataset_name == "users_prod"
    assert ctx.column_order == ["user_id", "country"]
    assert ctx.column_profiles == [c1, c2]
    assert ctx.get_column("country") is c2
    assert ctx.relationships is rel
    assert ctx.reference_profile is ref_ctx


def test_profile_context_preserves_column_ordering_and_rejects_duplicates():
    """ProfileContext and DatasetProfile preserve exact column order and reject duplicate names."""
    col_z = ColumnProfile(name="z_col", dtype="int64")
    col_a = ColumnProfile(name="a_col", dtype="float64")
    col_m = ColumnProfile(name="m_col", dtype="object")

    ctx = ProfileContext(columns=[col_z, col_a, col_m])
    assert ctx.column_order == ["z_col", "a_col", "m_col"]
    assert [c.name for c in ctx.column_profiles] == ["z_col", "a_col", "m_col"]

    # Duplicate column names in sequence must raise ValueError rather than silently overwriting
    col_z_dup = ColumnProfile(name="z_col", dtype="object")
    with pytest.raises(ValueError, match="Duplicate column name 'z_col'"):
        ProfileContext(columns=[col_z, col_a, col_z_dup])

    with pytest.raises(ValueError, match="Duplicate column name 'z_col'"):
        DatasetProfile(columns=[col_z, col_z_dup])

    # Mapping key mismatch must raise ValueError
    with pytest.raises(ValueError, match="does not match ColumnProfile.name"):
        ProfileContext(columns={"wrong_key": col_z})


def test_empty_and_minimally_populated_profile_context():
    """Empty and minimally populated ProfileContext instances can be created safely."""
    empty_ctx = ProfileContext()
    assert isinstance(empty_ctx.dataset, DatasetProfile)
    assert empty_ctx.dataset.row_count == 0
    assert empty_ctx.columns == {}
    assert empty_ctx.column_order == []
    assert empty_ctx.column_profiles == []
    assert empty_ctx.relationships is None
    assert empty_ctx.reference_profile is None

    minimal_ctx = ProfileContext(
        columns={"only_col": ColumnProfile(name="only_col", dtype="int64")}
    )
    assert minimal_ctx.column_order == ["only_col"]
    assert minimal_ctx.get_column("only_col") is not None
    assert minimal_ctx.dataset.columns["only_col"].name == "only_col"


# ---------------------------------------------------------------------------
# 3. Import and Compatibility Tests
# ---------------------------------------------------------------------------


def test_existing_and_new_imports_from_cleanalytix_types():
    """All legacy types, enums, and new v2.0 contracts are importable without circularity."""
    # Legacy enums
    assert QualityDimension.COMPLETENESS.value == "completeness"
    assert InsightSeverity.CRITICAL.value == "critical"
    assert AnalysisLevel.STANDARD.value == "standard"

    # Legacy dataclasses & exception
    insight = InsightItem(
        dimension=QualityDimension.VALIDITY,
        severity=InsightSeverity.WARNING,
        column="age",
        message="Negative values found",
        metric_name="negative_count",
        metric_value=3.0,
        threshold=0.0,
        recommendation="Filter or impute negative ages",
    )
    assert insight.to_dict()["column"] == "age"

    rec = RecommendationItem(
        priority_rank=1,
        rule_id=101,
        dimension=QualityDimension.VALIDITY,
        severity="critical",
        affected_column="age",
        issue_name="Type errors",
        evidence="2 type errors detected",
        rationale="High error rate",
        suggested_action="Cast to int",
    )
    assert rec.to_dict()["priority_rank"] == 1

    dim_score = DimensionScore(dimension=QualityDimension.VALIDITY, score=92.5, weight=0.2)
    assert dim_score.score == 92.5
    assert dim_score.effective_weight == 0.2

    cfg = CleanalytixConfig()
    assert cfg.score_mode == "linear"

    res = CleanalytixResult()
    assert isinstance(res.metadata, pd.DataFrame)

    with pytest.raises(FeatureRetiredError):
        raise FeatureRetiredError("legacy_feature")

    # Verify top-level cleanalytix package imports cleanly without circular import issues
    imported = importlib.import_module("cleanalytix.types")
    assert hasattr(imported, "ProfileContext")
    assert hasattr(cleanalytix, "analyze")
    assert hasattr(cleanalytix, "CleanalytixResult")
    assert "cleanalytix.types" in sys.modules
