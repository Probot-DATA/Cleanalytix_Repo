# INDEPENDENT POST-IMPLEMENTATION VERIFICATION AUDIT: CLEANALYTIX v0.2.0

**Audit Date:** November 2026  
**Auditor:** Principal Software Architect, Senior QA Lead, and Statistical Computing Specialist  
**Repository Location:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\cleanalytix`  
**Scope:** Independent verification of claims made in `REDEVELOPMENT_REPORT.md` against actual repository source code, Git working tree, mathematical formulations, API contracts, test suites, and packaging configurations.  
**Mode:** Inspection-Only (Strictly zero code modifications, file deletions, or destructive Git operations).

---

## EXECUTIVE AUDIT SUMMARY

An independent post-implementation verification audit was conducted on the Cleanalytix repository following the v0.2.0 redevelopment. The audit independently evaluated:
1. Repository change inventory and Git working tree integrity.
2. Mathematical and statistical correctness of all quality dimensions, metrics, and divergence algorithms.
3. API compatibility, breaking changes, and primary entry point behavior.
4. Non-mutation guarantees across all public entry points.
5. Test suite rigor, genuine mathematical assertions vs. smoke tests, and missing edge cases.
6. Package architecture, module coupling, duplicate logic, and cleaning retirement.
7. Documentation, dependency declarations, and packaging buildability.

### Overall Verification Scorecard

| Area | Status | Critical Findings Summary |
| :--- | :---: | :--- |
| **1. Repository Change Inventory** | **VERIFIED** | 15 modified files, 7 untracked library files/packages, 6 audit reports. No tracked files deleted. |
| **2. Non-Mutation Guarantee** | **VERIFIED** | 100% verified across all 14 public entry points; zero DataFrame mutation under all scenarios. |
| **3. Cleaning Retirement** | **VERIFIED** | Destructive transformation engine completely purged; `clean()` and `get_cleaned_data()` raise `FeatureRetiredError`. |
| **4. Core Statistical Metrics** | **VERIFIED** | Bessel's correction (`ddof=1`), Fisher excess kurtosis, MAD, IQR, and Cramér's V verified mathematically. |
| **5. Quality Dimension Framework** | **INCORRECT** | **Critical Defect:** Timeliness is completely omitted from default weights and scoring aggregation (only 5 dimensions scored). Consistency is equated solely with numeric outliers. |
| **6. Divergence Implementations** | **INCORRECT** | **Critical Defect:** In `generate_meta.py`, numeric KL divergence is faked as `0.5 * PSI` and JS divergence is faked as `min(1.0, 0.25 * PSI)`. Constant-baseline KS is hardcoded to 1.0. |
| **7. Entropy Formulation** | **MATHEMATICALLY FLAWED** | Continuous entropy in `generate_meta.py` uses empirical quantile binning, forcing normalized entropy to $\approx 1.0$ for almost all continuous data. |
| **8. API Compatibility** | **VERIFIED** | `analyze()`, `CleanalytixResult`, and `Run_DQ_Pipeline` adhere to stated contracts; breaking changes are properly gated. |
| **9. Test Suite Depth** | **PARTIALLY VERIFIED** | All 30 tests pass, but 10 tests are shallow smoke tests and one contains a tautology assertion. |
| **10. Packaging & Dependencies** | **PARTIALLY VERIFIED** | Package builds cleanly as wheel (`cleanalytix-0.2.0-py3-none-any.whl`), but **`scipy` is missing from `pyproject.toml` and `requirements.txt`**. |

---

## SECTION 1: REPOSITORY CHANGE INVENTORY

### 1.1 Git Status & Tracked Changes
The working branch is `main` (behind `origin/main` by 1 commit). A total of 15 tracked files have uncommitted modifications:

| File | Status | Description of Modifications |
| :--- | :---: | :--- |
| `README.md` | Modified | Updated to document Data Quality Intelligence, zero mutation, and `analyze()` API. |
| `cleanalytix/__init__.py` | Modified | Export modern symbols (`analyze`, `QualityDimension`, `CleanalytixResult`), subpackage `metrics`. |
| `cleanalytix/adjust_prod_meta.py` | Modified | Added Bessel's correction (`ddof=1`), constant-baseline handling, and nullable type safety. |
| `cleanalytix/cleaning_recommendations.py` | Modified | Minor formatting and alignment with non-mutating recommendation schemas. |
| `cleanalytix/compute_dq_score.py` | Modified | Dynamic applicability re-weighting, delegates to `scoring.py`, handles NaN/null safely. |
| `cleanalytix/generate_meta.py` | Modified | Eliminated 70/30 baseline self-split, added column uniqueness check, Bessel correction, nullable handling. |
| `cleanalytix/get_cleaned_data.py` | Modified | **Purged 500+ lines of destructive mutation rules**; raises `FeatureRetiredError`. |
| `cleanalytix/get_table_for_DQ_computation.py` | Modified | Guards against `ZeroDivisionError` on empty tables; preserves NaNs for non-applicable metrics. |
| `cleanalytix/pipeline.py` | Modified | Added `analyze()`, non-mutating `Run_DQ_Pipeline`, gated `cleaning=True` with `FeatureRetiredError`. |
| `cleanalytix/preprocess_types.py` | Modified | Fixed boolean Series indexing FutureWarning (BUG-17); added `diagnose_column_types()`. |
| `cleanalytix/summarize_dataset_health.py` | Modified | Defensive aggregation handling empty metadata DataFrames. |
| `cleanalytix/version.py` | Modified | Bumped `__version__ = "0.2.0"`. |
| `pyproject.toml` | Modified | Version 0.2.0, updated description and metadata. |
| `requirements.txt` | Modified | Updated dependency requirements. |
| `tests/test_pipeline.py` | Modified | Added tests for `analyze()`, `CleanalytixResult`, and `FeatureRetiredError`. |

### 1.2 Newly Created Untracked Files

| Path | Category | Purpose |
| :--- | :--- | :--- |
| `cleanalytix/column_role.py` | Production Source | Inferred column roles: continuous, discrete, binary, identifier, categorical, free_text, datetime, boolean, unknown. |
| `cleanalytix/insights.py` | Production Source | Deterministic rule-based advisory insight and recommendation generator. |
| `cleanalytix/scoring.py` | Production Source | Two-level dimensional scoring engine with dynamic metric applicability re-weighting. |
| `cleanalytix/types.py` | Production Source | Public enums (`QualityDimension`, `InsightSeverity`), dataclasses (`DimensionScore`), `CleanalytixResult`. |
| `cleanalytix/metrics/__init__.py` | Metrics Subpackage | Module exports for modular analytics. |
| `cleanalytix/metrics/completeness.py` | Metrics Subpackage | Null density, complete rows, Jaccard co-occurrence, missingness correlation. |
| `cleanalytix/metrics/distribution.py` | Metrics Subpackage | Normalized Shannon entropy, HHI, Gini-Simpson diversity, rare category ratios. |
| `cleanalytix/metrics/drift.py` | Metrics Subpackage | KS 2-sample test, Wasserstein distance, decile PSI, categorical PSI, JSD. |
| `cleanalytix/metrics/relationships.py` | Metrics Subpackage | Pearson, Spearman, Cramér's V, duplicate/near-duplicate column discovery. |
| `cleanalytix/metrics/statistical.py` | Metrics Subpackage | Central tendency, dispersion (`ddof=1`), Fisher excess kurtosis, outlier routing (Z-score/IQR/ModZ). |
| `cleanalytix/metrics/temporal.py` | Metrics Subpackage | Freshness in days, datetime parsing rates, monotonicity, step gaps, future dates. |
| `cleanalytix/metrics/uniqueness.py` | Metrics Subpackage | Distinct ratios, constant/near-constant columns, duplicate rows, candidate primary keys. |
| `tests/test_fixes.py` | Test Suite | 22 unit and regression tests covering P0–P4 bugs and modular metrics. |
| `audit_reports/*.md` (6 files) | Documentation | Audit reports (`AUDIT_REPORT.md`, `ARCHITECTURE_MAP.md`, etc.). |

### 1.3 Deleted Files
- **Zero tracked files have been deleted from Git.**

### 1.4 Unintended Temporary / Generated Files
During buildability testing, the following generated artifacts were created and are currently untracked:
1. `cleanalytix/tests/outputs/cleanalytix-0.2.0-py3-none-any.whl` (64,631 bytes): Wheel generated during non-destructive build verification.
2. `cleanalytix/__pycache__/`: Python bytecode cache files.
3. `cleanalytix/metrics/__pycache__/`: Python bytecode cache files.
4. `tests/__pycache__/`: Pytest cache files.

> **Audit Recommendation:** Add `cleanalytix/tests/` to `.gitignore` or clean up `tests/outputs/` prior to release commits.

---

## SECTION 2: MATHEMATICAL & STATISTICAL AUDIT

### 2.1 Quality Dimension Framework & Dimension Weights
* **Classification:** **INCORRECT**
* **Verification Detail:**
  In `REDEVELOPMENT_REPORT.md` (Sections 1, 4, 8) and `cleanalytix/__init__.py`, the library explicitly advertises a **6-Dimension Quality Framework**:
  1. Completeness
  2. Validity
  3. Uniqueness
  4. Consistency
  5. Timeliness
  6. Stability

  However, inspection of `cleanalytix/scoring.py` reveals:
  ```python
  # cleanalytix/scoring.py, lines 44-57
  DEFAULT_DIMENSION_WEIGHTS: Dict[QualityDimension, float] = {
      QualityDimension.COMPLETENESS: 0.35,
      QualityDimension.VALIDITY: 0.30,
      QualityDimension.UNIQUENESS: 0.20,
      QualityDimension.CONSISTENCY: 0.15,
  }

  DEFAULT_DIMENSION_WEIGHTS_WITH_BASELINE: Dict[QualityDimension, float] = {
      QualityDimension.COMPLETENESS: 0.25,
      QualityDimension.VALIDITY: 0.25,
      QualityDimension.UNIQUENESS: 0.15,
      QualityDimension.CONSISTENCY: 0.15,
      QualityDimension.STABILITY: 0.20,
  }
  ```
  **Findings:**
  1. `QualityDimension.TIMELINESS` is **COMPLETELY ABSENT** from `DEFAULT_DIMENSION_WEIGHTS` and `DEFAULT_DIMENSION_WEIGHTS_WITH_BASELINE`.
  2. Inside `score_dataset()` (`scoring.py:160-355`), there is **NO code calculating a Timeliness dimension score**.
  3. `dimension_scores` dictionary returned by `score_dataset()` contains only `completeness`, `validity`, `uniqueness`, `consistency`, and optionally `stability`. It **never contains `timeliness`**.
  4. In `DEFAULT_THRESHOLDS` (`scoring.py:41`), `"freshness_days": 30.0` is defined, but it is **never accessed or evaluated anywhere in `score_dataset()`**.
  5. **Discrepancy in Report Diagram:** In `REDEVELOPMENT_REPORT.md` Section 4, the ASCII architecture diagram depicts exactly **5 dimension boxes** beneath `OVERALL DQ SCORE`, even though the prose claims 6 dimensions. This proves that Timeliness was omitted from the scoring engine during implementation.

### 2.2 Equating Consistency with Numeric Outlier Containment
* **Classification:** **INCORRECT DOMAIN MODELING**
* **Verification Detail:**
  Inspection of `cleanalytix/scoring.py` lines 246-264:
  ```python
  # 4. Consistency Dimension (numeric outliers)
  consistency_scores = []
  for _, row in meta_df.iterrows():
      role = row.get("column_role", "")
      if role in {"continuous", "discrete"}:
          n_non_null = int(row.get("non_null_count", 0))
          if n_non_null >= 3:
              out_cnt = row.get("outlier_count", 0)
              out_pct = (float(out_cnt or 0) / max(1, n_non_null)) * 100.0
              o_score = calculate_metric_score(
                  out_pct, thresholds.get("outlier_percent", 10.0), score_mode=score_mode
              )
              consistency_scores.append(o_score)
  consistency_score = float(np.mean(consistency_scores) * 100.0) if consistency_scores else 100.0
  ```
  **Findings:**
  1. The entire Consistency dimension is computed **exclusively from `outlier_count` on numeric columns**.
  2. In data quality standards (DAMA-DMBOK, ISO/IEC 25012), **Consistency** evaluates whether data values across multiple columns, records, or systems do not conflict with each other (e.g. referential integrity, cross-field logical rules like `DischargeDate >= AdmissionDate`, absence of contradictory records).
  3. Outlier containment is a measure of statistical distribution normality or range plausibility, not consistency.
  4. While `cleanalytix.metrics.relationships` computes Pearson/Spearman multicollinearity, Cramér's V, and duplicate columns, **none of these relational consistency metrics are incorporated into the Consistency score**.

### 2.3 Uniqueness Dimension Formulation
* **Classification:** **PARTIALLY VERIFIED**
* **Verification Detail:**
  Inspection of `cleanalytix/scoring.py` lines 160-168:
  ```python
  dup_percent = 0.0
  if "duplicate_rows_percent" in meta_df.columns:
      first_dup = meta_df["duplicate_rows_percent"].iloc[0]
      if pd.notna(first_dup):
          dup_percent = float(first_dup)

  dup_thresh = thresholds.get("duplicate_rows_percent", 1.0)
  uniqueness_raw_score = calculate_metric_score(dup_percent, dup_thresh, score_mode=score_mode)
  uniqueness_score = uniqueness_raw_score * 100.0
  ```
  **Findings:**
  1. The Uniqueness dimension score is computed solely from dataset-level `duplicate_rows_percent`.
  2. Column-level uniqueness metrics (such as `distinct_ratio`, `is_unique`, and primary key candidate discovery in `cleanalytix.metrics.uniqueness`) are **completely ignored** in the Uniqueness dimension score.
  3. If a column is inferred as an `"identifier"` (e.g. `customer_id` or `ssn`) and contains duplicates, this domain failure is never reflected in the Uniqueness score.
  4. Furthermore, Uniqueness does not contribute to per-column scores (`final_col_scores`); it exists only as a dataset-level scalar.

### 2.4 Numeric Drift: Faked KL and JS Divergences in `generate_meta.py`
* **Classification:** **INCORRECT / ARBITRARY SHORTCUT**
* **Verification Detail:**
  Inspection of `cleanalytix/generate_meta.py` lines 226-231:
  ```python
  if column_role in {"continuous", "discrete"} and is_numeric_dtype(base_series) and not is_bool_dtype(base_series):
      drift_res = compute_numeric_drift(base_series, new_series)
      psi_val = drift_res["psi"]
      # Approximate KL and JS consistently
      return float(psi_val * 0.5), float(psi_val), float(min(1.0, psi_val * 0.25))
  ```
  **Findings:**
  1. For numeric columns, `KL_div` is **faked** as `0.5 * psi_val`.
  2. For numeric columns, `JS_div` is **faked** as `min(1.0, 0.25 * psi_val)`.
  3. Mathematical definition of KL divergence:
     $$D_{KL}(P \parallel Q) = \sum P(i) \ln\left(\frac{P(i)}{Q(i)}\right)$$
     KL divergence is strictly asymmetric ($D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$).
  4. Mathematical definition of PSI:
     $$\text{PSI} = \sum (P(i) - Q(i)) \ln\left(\frac{P(i)}{Q(i)}\right) = D_{KL}(P \parallel Q) + D_{KL}(Q \parallel P)$$
     Assuming $D_{KL}(P \parallel Q) = 0.5 \cdot \text{PSI}$ mathematically asserts that the baseline-to-production divergence is identical to the production-to-baseline divergence, which is false for skewed shifts.
  5. Mathematical definition of Jensen-Shannon divergence:
     $$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)$$
     $JSD$ is bounded by $\ln(2) \approx 0.6931$ (or 1.0 in base 2). Scaling PSI by `0.25` is an arbitrary heuristic without mathematical justification.
  6. While `compute_numeric_drift()` in `cleanalytix.metrics.drift` computes Wasserstein distance and KS statistics, it never computes KL or JS divergence, leading `generate_meta.py` to resort to these shortcuts.

### 2.5 Constant-Baseline Drift Handling
* **Classification:** **PARTIALLY VERIFIED with MATHEMATICALLY QUESTIONABLE SHORTCUT**
* **Verification Detail:**
  Inspection of `cleanalytix/metrics/drift.py` lines 68-86:
  ```python
  if b_min == b_max:
      if p_min == p_max and p_min == b_min:
          return default_result
      else:
          return {
              "ks_statistic": 1.0,
              "ks_p_value": 0.0,
              "wasserstein_distance": float(np.mean(np.abs(p_arr - b_min))),
              "normalized_wasserstein": 1.0,
              "psi": 1.0,
              "drift_detected": True,
              "drift_severity": "critical",
              "sample_size_base": n_b,
              "sample_size_prod": n_p,
          }
  ```
  **Findings:**
  1. If baseline is constant ($b = [5, 5, \dots]$) and production contains even one slightly different value out of 10,000 records ($p = [5, 5, \dots, 5.01]$), the code hardcodes `ks_statistic = 1.0`, `ks_p_value = 0.0`, and `psi = 1.0`.
  2. In reality, the 2-sample Kolmogorov-Smirnov statistic is the maximum vertical distance between empirical CDFs:
     $$D_{n, m} = \sup_x |F_1(x) - F_2(x)|$$
     For 1 differing value in 10,000 rows, the true KS statistic is $1 / 10000 = 0.0001$.
  3. `scipy.stats.ks_2samp(b_arr, p_arr)` handles degenerate constant baselines natively and returns the true exact statistic (verified experimentally).
  4. Hardcoding `ks_statistic = 1.0` and `psi = 1.0` is an arbitrary step-function jump rather than a continuous metric.

### 2.6 Normalized Shannon Entropy for Continuous Variables
* **Classification:** **MATHEMATICALLY FLAWED**
* **Verification Detail:**
  Inspection of `cleanalytix/generate_meta.py` lines 49-65:
  ```python
  if column_role in {"continuous", "discrete"} and is_numeric_dtype(col_ser) and not is_bool_dtype(col_ser):
      quantiles = np.linspace(0, 1, 6) # for n < 10000
      bins = np.unique(np.quantile(clean, q=quantiles))
  ...
  probs = np.array(get_probs(df_base, column, bins, ...))
  norm_ent = float(entropy / math.log(len(probs) + EPS))
  ```
  **Findings:**
  1. By definition, empirical quantile binning constructs intervals such that an equal fraction ($1/K$) of data points falls into each bin.
  2. Because each quantile bin contains approximately $1/K$ of the observations, the empirical discrete probability distribution over these quantile bins is approximately uniform:
     $$P(i) \approx \frac{1}{K} \implies H = -\sum \frac{1}{K} \ln\left(\frac{1}{K}\right) = \ln(K)$$
  3. Consequently, $H_{\text{norm}} = \frac{H}{\ln(K)} \approx 1.0$ for **any** continuous distribution regardless of its underlying shape (e.g. Normal returns $0.999999996$, Exponential returns $0.999999996$).
  4. Conversely, if a bimodal dataset with large duplicate ties collapses the quantile edges, `len(bins) - 1 == 1`, and line 78 aborts with `return 0.0`.
  5. Continuous entropy estimation requires equal-width histogram binning (or Scott's rule / Freedman-Diaconis rule), not quantile binning.

### 2.7 Verified Mathematical Formulations
The following statistical algorithms were independently verified as mathematically sound:
1. **Sample Variance and Standard Deviation (`cleanalytix.metrics.statistical:104-108`):**
   Uses Bessel's correction (`ddof=1`):
   $$s^2 = \frac{1}{N - 1} \sum_{i=1}^N (x_i - \bar{x})^2$$
   For $N < 2$, variance and std return `None` / `NaN` instead of misleading zeros. **[VERIFIED]**
2. **Fisher's Excess Kurtosis (`cleanalytix.metrics.statistical:120-123`):**
   Uses pandas `Series.kurt()` which evaluates Fisher's definition ($Kurt_{\text{excess}} = Kurt_{\text{Pearson}} - 3$). Normal distribution yields $0.0$. **[VERIFIED]**
3. **Median Absolute Deviation (`cleanalytix.metrics.statistical:100-101`):**
   $$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$
   Correctly scaled by $0.6745$ for Boris Iglewicz & David Hoaglin's Modified Z-score ($M_i = \frac{0.6745 \cdot |x_i - \text{median}|}{\text{MAD}}$). **[VERIFIED]**
4. **Wasserstein Distance (`cleanalytix.metrics.drift:94-99`):**
   Computes exact 1-Wasserstein / Earth Mover's Distance via `scipy.stats.wasserstein_distance()`, normalized by pooled standard deviation:
   $$s_{\text{pooled}} = \sqrt{\frac{s_{\text{base}}^2 + s_{\text{prod}}^2}{2}}$$
   **[VERIFIED]**
5. **Cramér's V with Bias Correction (`cleanalytix.metrics.relationships:87-133`):**
   Implements Bergsma (2013) bias correction:
   $$\tilde{\phi}^2 = \max\left(0, \phi^2 - \frac{(k - 1)(r - 1)}{n - 1}\right), \quad V = \sqrt{\frac{\tilde{\phi}^2}{\min(\tilde{r} - 1, \tilde{k} - 1)}}$$
   **[VERIFIED]**
6. **Categorical PSI & Jensen-Shannon Divergence (`cleanalytix.metrics.drift:209-224`):**
   Full discrete calculation with Laplace smoothing ($P_i = \frac{c_i + \epsilon}{N + \epsilon K}$), symmetric PSI, and exact $M = \frac{1}{2}(P + Q)$ mixture JSD. **[VERIFIED]**

---

## SECTION 3: API COMPATIBILITY AUDIT

### 3.1 Public Surface & Signature Comparison

| Symbol | Original v0.1.0 Contract | Current v0.2.0 Contract | Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| `Run_DQ_Pipeline` | Multi-dataset pipeline | Multi-dataset pipeline | **VERIFIED** | All 11 parameters preserved; identical default values. |
| `run_dq_pipeline` | Function alias | Function alias | **VERIFIED** | `run_dq_pipeline = Run_DQ_Pipeline`. |
| `analyze` | *Did not exist* | Primary single-dataset API | **VERIFIED** | Returns `CleanalytixResult`. |
| `clean` | Rule-based mutation | Raises `FeatureRetiredError` | **BREAKING** | Documented intentional breaking change. |
| `get_cleaned_data` | Rule-based mutation | Raises `FeatureRetiredError` | **BREAKING** | Documented intentional breaking change. |
| `Compute_DQ_Score` | Scored metric table | Scored metric table | **VERIFIED** | Preserves signature; supports dynamic applicability. |
| `generate_meta` | Full profiling table | Full profiling table | **VERIFIED** | Output schema identical; self-drift removed. |
| `cleaning_recommendations` | Output recommendation DF | Output recommendation DF | **VERIFIED** | Same schema (`rule_ids`, `inferred_type`, etc.). |
| `get_table_for_DQ_computation` | Scoring table | Scoring table | **VERIFIED** | Same output schema (`outlier_percent`, etc.). |
| `summarize_dataset_health` | Dataset summary DF | Dataset summary DF | **VERIFIED** | Same schema (`avg_missing_percent`, etc.). |
| `learn_reference_profile` | Learns baseline stats | Learns baseline stats | **VERIFIED** | Same dictionary profile structure. |
| `adjust_prod_meta_with_reference` | Adjusts prod meta | Adjusts prod meta | **VERIFIED** | Same output schema. |
| `infer_and_fix_types` | Hard type inference | Hard type inference | **VERIFIED** | Returns modified copy; does not mutate input. |
| `diagnose_column_types` | *Did not exist* | Non-mutating type diagnostics | **VERIFIED** | Newly added modern utility. |

### 3.2 Output Schema Consistency for `Run_DQ_Pipeline`
When called with `cleaning=False` (default), the returned dictionary has the exact top-level keys:
- `base_data`
- `prod_data` (empty dict `{}` when `new_dataset_list=None`)

Inside `base_data`, all 11 documented keys exist:
`dirty_scores`, `cleaned_scores`, `cleaned_datasets`, `meta_before_cleaning`, `meta_after_cleaning`, `recommendations`, `change_log`, `summarized_before`, `summarized_after`, `main_metrics_before`, `main_metrics_after`.
The `cleaned_*` and `*_after` artifacts are returned as empty DataFrames / empty lists, preserving downstream dictionary lookups without raising `KeyError`.

### 3.3 Modern API: `analyze()` and `CleanalytixResult`
* **Classification:** **VERIFIED**
* **Verification Detail:**
  Tested `ca.analyze(df)`:
  1. Returns `CleanalytixResult` which subclasses `dict`.
  2. Dict access works: `res["overall_score"]`, `res["base_data"]`, `res["quality_dimensions"]`.
  3. Attribute access works: `res.overall_score`, `res.quality_dimensions`, `res.recommendations`, `res.insights`, `res.metadata`.
  4. `res.summary()` produces a compact JSON-serializable health dictionary containing `overall_score`, `dimensions`, `critical_insights_count`, `warning_insights_count`, and `recommendations_count`.
  5. Tested production comparison: `ca.analyze(prod_df, baseline_df=base_df)` successfully populates `res.quality_dimensions["stability"]` and evaluates drift.

---

## SECTION 4: NON-MUTATION AUDIT

* **Classification:** **VERIFIED (100% Strict Compliance)**
* **Verification Detail:**
  Every public analysis entry point was subjected to strict deep-copy equality checks before and after execution:
  - Input DataFrame: 5 rows with mixed types (float with outlier, categorical, string with leading/trailing whitespace, datetime, null values) and a non-standard index `[10, 20, 30, 40, 50]`.
  - Assertions: `pd.testing.assert_frame_equal(orig, copy, check_dtype=True, check_index_type=True, check_column_type=True, check_frame_type=True)`.

### Entry Point Non-Mutation Verification Matrix

| Entry Point | Mutation Status | Evidence |
| :--- | :---: | :--- |
| `ca.analyze(df)` | **IMMUTABLE** | Input DataFrame strictly equals copy; index, dtypes, and values unchanged. |
| `ca.analyze(prod, baseline_df=base)` | **IMMUTABLE** | Both production and baseline DataFrames strictly equal copies. |
| `ca.Run_DQ_Pipeline(names, [df])` | **IMMUTABLE** | Baseline DataFrame in list is untouched. |
| `ca.Run_DQ_Pipeline(names, [df], [prod])` | **IMMUTABLE** | Both baseline and production DataFrames are untouched. |
| `ca.generate_meta(names, [df])` | **IMMUTABLE** | Input DataFrame strictly equals copy. |
| `ca.generate_meta(names, [df], [prod])` | **IMMUTABLE** | Both baseline and production DataFrames are untouched. |
| `ca.cleaning_recommendations(meta)` | **IMMUTABLE** | Input metadata table is untouched. |
| `ca.get_table_for_DQ_computation(meta, [df], names)` | **IMMUTABLE** | Metadata and input DataFrames are untouched. |
| `ca.Compute_DQ_Score(names, [df], metrics)` | **IMMUTABLE** | Scoring metrics and input DataFrames are untouched. |
| `ca.summarize_dataset_health(meta)` | **IMMUTABLE** | Input metadata table is untouched. |
| `ca.learn_reference_profile(names, [df])` | **IMMUTABLE** | Input DataFrames are untouched. |
| `ca.adjust_prod_meta_with_reference(...)` | **IMMUTABLE** | Takes `prod_meta.copy(deep=True)`; original metadata and DataFrames untouched. |
| `ca.diagnose_column_types(df)` | **IMMUTABLE** | Pure read-only inspection. |
| `ca.infer_and_fix_types(df)` | **IMMUTABLE** | Takes `df.copy(deep=True)` at entry; returns new DataFrame without touching caller's object. |

---

## SECTION 5: TEST SUITE AUDIT

### 5.1 Test Inventory & Execution
* **Total Collected Tests:** 30 tests (22 in `tests/test_fixes.py`, 8 in `tests/test_pipeline.py`).
* **Execution Status:** 30 PASSED, 0 FAILED, 0 ERRORS in 2.71 seconds.
* **Test Isolation:** Executed with `-p no:cacheprovider` and `PYTHONDONTWRITEBYTECODE=1`.

### 5.2 Critical Categorization of Test Rigor

```
Total Tests (30)
├── Genuine Mathematical / Algorithmic Tests: 8 tests (26.7%)
├── API Contract & Type / Exception Tests:    12 tests (40.0%)
└── Shallow Smoke / Existence Tests:          10 tests (33.3%)
```

#### Category A: Genuine Mathematical & Algorithmic Tests (8 tests)
These tests assert specific numerical results derived from mathematical formulas:
1. `test_sample_variance_uses_bessel_correction` (`test_fixes.py:122`): Asserts $\text{std}([2.0, 4.0]) == \sqrt{2.0} \approx 1.4142$ (`ddof=1`).
2. `test_production_meta_uses_production_missingness` (`test_fixes.py:14`): Asserts `missing_percent == approx(66.6666, rel=1e-3)`.
3. `test_duplicate_threshold_is_documented_default` (`test_fixes.py:67`): Asserts exact 50.0% duplication on repeating pairs.
4. `test_zero_self_drift_in_baseline_only_mode` (`test_fixes.py:100`): Asserts exact 0.0 PSI, JS, and KL divergence on single datasets.
5. `test_constant_baseline_drift_detection` (`test_fixes.py:109`): Asserts 0.0 drift for identical constant data, and $> 0.0$ for shifted data.
6. `test_nullable_types_handling` (`test_fixes.py:131`): Asserts exact mean 23.333 and 2.833 on nullable `Int64` and `Float64` with `pd.NA`.
7. `test_metrics_relationships_module` (`test_fixes.py:228`): Asserts Cramér's V $== 1.0 \pm 0.01$ for perfectly associated categories.
8. `test_mostly_numeric_mixed_column_counts_text_as_error` (`test_fixes.py:46`): Asserts exact count of 1 type error.

#### Category B: API Contract, Type Handling & Retirement Tests (12 tests)
These tests verify that expected functions raise appropriate errors or preserve types:
1. `test_retired_cleaning_raises_feature_retired_error` (`test_fixes.py:24`): Asserts `clean()` and `get_cleaned_data()` raise `FeatureRetiredError`.
2. `test_pipeline_cleaning_raises_feature_retired_error` (`test_pipeline.py:166`): Asserts `Run_DQ_Pipeline(cleaning=True)` raises `FeatureRetiredError`.
3. `test_duplicate_column_names_rejected` (`test_fixes.py:142`): Asserts `ValueError` on duplicate column names.
4. `test_infer_types_for_new_false_preserves_production_dtype` (`test_fixes.py:80`): Asserts prod dtype remains object.
5. `test_infer_types_false_does_not_call_preprocessor_for_base_or_prod` (`test_pipeline.py:104`): Monkeypatch verification.
6. `test_hard_type_inference_current_semantics` (`test_pipeline.py:122`): Asserts coercions to boolean, UTC datetime, and numeric.
7. `test_binary_flag_not_treated_as_continuous` (`test_fixes.py:51`): Asserts role is `"binary"`.
8. `test_identifier_column_skips_continuous_profiling` (`test_fixes.py:59`): Asserts role is `"identifier"`.
9. `test_generate_meta_handles_boolean_columns` (`test_fixes.py:96`): Asserts role is `"binary"` and numeric mean is `None`.
10. `test_single_row_dataset_handling` (`test_fixes.py:254`): Asserts sample std is `NaN`/`None` on $N=1$.
11. `test_public_import_surface` (`test_pipeline.py:65`): Asserts `__version__` and callables.
12. `test_zero_row_dataframe_handling` (`test_fixes.py:150`): Asserts empty DataFrame creates 2-row metadata without crashing.

#### Category C: Shallow Smoke Tests & Tautology Assertions (10 tests)
These tests merely check dictionary keys, non-empty DataFrames, or wide ranges:
1. `test_pipeline_base_only_runs_without_errors` (`test_pipeline.py:70`): Calls helper `_assert_pipeline_block()` which asserts `not df.empty`. **Never checks whether any score or metric is correct.**
2. `test_pipeline_with_production_options_runs_without_errors` (`test_pipeline.py:82`): Only asserts keys exist and DataFrames are not empty.
3. `test_analyze_api_contract` (`test_pipeline.py:177`): Asserts `0.0 <= result.overall_score <= 100.0`. A score of 0.01 or 99.99 passes equally.
4. `test_analyze_with_baseline` (`test_pipeline.py:192`): Asserts `"stability" in result.quality_dimensions` and `0.0 <= overall_score <= 100.0`.
5. **Tautology in `test_metrics_statistical_module` (`test_fixes.py:196`):**
   ```python
   # tests/test_fixes.py, line 196
   assert stats["outliers"] if "outliers" in stats else True
   ```
   Inspection of `compute_numeric_statistics()` shows that the returned `stats` dictionary has keys: `count`, `mean`, `median`, `std`, `variance`, `min`, `max`, `iqr`, `mad`, `skewness`, `kurtosis`, etc. It **does NOT contain an `"outliers"` key** (outliers are in `detect_outliers()`).
   Therefore, `"outliers" in stats` is **always `False`**, and the statement evaluates to `assert True` without testing anything!
6. `test_metrics_distribution_module` (`test_fixes.py:204`): Only asserts `0.0 < normalized_entropy < 1.0` and `hhi > 0.5`.
7. `test_metrics_drift_module` (`test_fixes.py:214`): Runs drift on unseeded random numbers (`np.random.normal`) and only checks `drift_detected is True` and `psi > 0.25`.
8. `test_metrics_completeness_module` (`test_fixes.py:160`): Asserts `len(res["missingness_cooccurrence"]) >= 1`.
9. `test_metrics_uniqueness_module` (`test_fixes.py:173`): Asserts `is_constant is True` on an all-9 column.
10. `test_metrics_temporal_module` (`test_fixes.py:244`): Asserts `time_span_days == 9.0`.

### 5.3 Missing Test Coverage
The test suite currently lacks coverage for:
1. **Dimensional Score Accuracy:** No test verifies the exact numerical overall score or dimensional scores against hand-calculated benchmarks.
2. **Exponential Score Decay Mode:** No test verifies that `score_mode="exponential"` produces the expected $e^{-k \cdot v / T}$ decay values.
3. **Dynamic Applicability Re-Weighting:** No test verifies that columns with non-applicable metrics (e.g. strings with NaN outliers) have their weights re-normalized rather than inflated.
4. **Composite Candidate Keys:** `detect_candidate_keys` is only tested on single-column keys (`"id"`), not 2-column composite keys.
5. **Wasserstein Distance:** No test verifies Wasserstein distance output against `scipy.stats.wasserstein_distance`.

---

## SECTION 6: ARCHITECTURE AUDIT

### 6.1 Package Structure vs. Report Claims
* **Classification:** **VERIFIED**
* **Verification Detail:**
  The actual package hierarchy exactly matches the structure described in `ARCHITECTURE_MAP.md` and `REDEVELOPMENT_REPORT.md`:
  ```
  cleanalytix/
  ├── __init__.py
  ├── adjust_prod_meta.py
  ├── cleaning_recommendations.py
  ├── column_role.py
  ├── compute_dq_score.py
  ├── generate_meta.py
  ├── get_cleaned_data.py
  ├── get_table_for_DQ_computation.py
  ├── insights.py
  ├── pipeline.py
  ├── preprocess_types.py
  ├── scoring.py
  ├── summarize_dataset_health.py
  ├── types.py
  ├── version.py
  └── metrics/
      ├── __init__.py
      ├── completeness.py
      ├── distribution.py
      ├── drift.py
      ├── relationships.py
      ├── statistical.py
      ├── temporal.py
      └── uniqueness.py
  ```

### 6.2 Duplicated Logic and Legacy Coupling
While the new modular architecture in `cleanalytix.metrics` is well-separated, several legacy modules retain duplicate implementations:
1. **Scoring Decay Logic:** `cleanalytix/compute_dq_score.py` retains `_linear_score()` and `_exp_score()` (lines 184-206) which duplicate `calculate_metric_score()` in `cleanalytix/scoring.py`.
2. **Categorical Entropy:** `get_entropy()` in `generate_meta.py` calculates entropy using quantile/Counter bins, while `cleanalytix/metrics/distribution.py` calculates normalized Shannon entropy independently.
3. **Parallel Recommendation Engines:**
   - `cleanalytix/cleaning_recommendations.py`: Generates the legacy Rule 1–10 recommendation DataFrame.
   - `cleanalytix/insights.py`: Generates the modern structured `InsightItem` objects.
   These two systems operate independently with different rule structures and different threshold triggers.

### 6.3 Cleaning Engine Retirement Verification
* **Classification:** **VERIFIED**
* **Verification Detail:**
  The claim that the cleaning engine was completely retired rather than simply masked was verified:
  - `cleanalytix/get_cleaned_data.py` lines 23-68: All former rule functions (`rule_1_drop_excessive_missing`, `rule_2_impute_missing`, `rule_3_handle_outliers`, `rule_4_trim_whitespace`, `rule_5_rare_categories`, `rule_6_dates`, `rule_7_type_errors`, `rule_8_business_rules`, `rule_9_power_transform`, `rule_10_no_op`, `rule_11_deduplicate`) were **physically removed from the file**.
  - `clean()` unconditionally raises `FeatureRetiredError`.
  - `get_cleaned_data()` unconditionally raises `FeatureRetiredError`.
  - Calling `Run_DQ_Pipeline(cleaning=True)` immediately aborts with `FeatureRetiredError`.

---

## SECTION 7: DOCUMENTATION & PACKAGING AUDIT

### 7.1 Version Synchronization
* **`cleanalytix/version.py`:** `__version__ = "0.2.0"` **[VERIFIED]**
* **`pyproject.toml`:** `version = "0.2.0"` **[VERIFIED]**
* **`cleanalytix/__init__.py`:** Exposes `__version__` matching `version.py`. **[VERIFIED]**

### 7.2 README Accuracy
* **Classification:** **PARTIALLY VERIFIED**
* **Verification Detail:**
  The `README.md` accurately describes the non-mutation philosophy, the `analyze()` API, the `CleanalytixResult` structure, and the modular metrics subpackage.
  However, it perpetuates the same discrepancy as the redevelopment report by listing **Timeliness** as an active dimension of the quality score (under "Multidimensional Data Quality Scoring", point 5), even though Timeliness is omitted from `score_dataset()`.

### 7.3 Packaging and Build Verification
* **Wheel Build:** Executed `python -m pip wheel --no-deps .`.
  - Successfully produced `cleanalytix-0.2.0-py3-none-any.whl` (64,631 bytes).
  - Package discovery finds `['cleanalytix', 'cleanalytix.metrics']`.
* **CRITICAL DEPENDENCY DEFECT:**
  Inspection of `pyproject.toml`:
  ```toml
  # pyproject.toml, lines 36-42
  dependencies = [
      "pandas>=1.5.0",
      "numpy>=1.23.0",
      "scikit-learn>=1.1.0",
      "nltk>=3.8.0",
      "dateparser>=1.2.0",
  ]
  ```
  And `requirements.txt`:
  ```text
  pandas>=1.5.0
  numpy>=1.23.0
  scikit-learn>=1.1.0
  nltk>=3.8.0
  dateparser>=1.2.0
  ```
  **Findings:**
  - `cleanalytix/metrics/drift.py` imports `from scipy import stats` (`ks_2samp`, `wasserstein_distance`).
  - `cleanalytix/metrics/relationships.py` imports `from scipy import stats` (`chi2_contingency`).
  - **`scipy` is completely missing from both `pyproject.toml` and `requirements.txt`!**
  - If a user installs `cleanalytix` in a clean environment using `pip install cleanalytix`, importing `cleanalytix` or running `analyze()` will immediately crash with:
    `ModuleNotFoundError: No module named 'scipy'`.

---

## SECTION 8: COMPREHENSIVE FINDINGS MATRIX

| Finding ID | Classification | Location | Description |
| :---: | :---: | :--- | :--- |
| **F-01** | **INCORRECT** | `cleanalytix/scoring.py:44-57, 323-355` | **Timeliness Dimension Missing from Scoring:** Timeliness is omitted from default dimension weights and is never computed or added to `dimension_scores` or `overall_score`. |
| **F-02** | **INCORRECT** | `cleanalytix/scoring.py:246-264` | **Consistency Equated Solely to Outliers:** Consistency dimension is calculated 100% from numeric outlier counts. Relational consistency and cross-column metrics are ignored. |
| **F-03** | **INCORRECT** | `cleanalytix/generate_meta.py:228-230` | **Faked Numeric Divergences:** `KL_div` is faked as `0.5 * PSI` and `JS_div` is faked as `min(1.0, 0.25 * PSI)` instead of computing true divergence formulas for numeric continuous data. |
| **F-04** | **MATHEMATICALLY FLAWED** | `cleanalytix/generate_meta.py:55-82` | **Quantile Binned Entropy Invariant:** Using quantile binning forces equal frequency per bin, making normalized continuous entropy mathematically invariant ($\approx 1.0$) across all distributions. |
| **F-05** | **PARTIALLY VERIFIED** | `cleanalytix/metrics/drift.py:76-86` | **Constant-Baseline Drift Heuristic:** KS statistic and PSI are hardcoded to 1.0 when a baseline is constant, even for infinitesimal shifts where true KS is close to 0.0. |
| **F-06** | **PARTIALLY VERIFIED** | `cleanalytix/scoring.py:160-168` | **Uniqueness Ignores Column Cardinality:** Uniqueness is calculated solely from dataset-level duplicate rows. Identifier column duplicate rates are not factored in. |
| **F-07** | **PARTIALLY VERIFIED** | `pyproject.toml:36-42`, `requirements.txt` | **Missing Runtime Dependency:** `scipy` is required by `drift.py` and `relationships.py` but is missing from package dependencies. |
| **F-08** | **PARTIALLY VERIFIED** | `tests/test_fixes.py:196` | **Tautology Assertion in Tests:** `assert stats["outliers"] if "outliers" in stats else True` always passes because `"outliers"` is not in `stats`. |
| **F-09** | **PARTIALLY VERIFIED** | `cleanalytix/scoring.py:213, 222, 262` | **Arbitrary Hardcoded Scoring Constants:** Validity weights (0.5 for type, 0.25 for whitespace) and column outlier weight (0.8) are hardcoded arbitrary scalars. |
| **F-10** | **VERIFIED** | Entire repository | **Strict Zero-Mutation Verified:** 100% confirmed across all 14 public entry points; original DataFrames are strictly immutable. |
| **F-11** | **VERIFIED** | `cleanalytix/get_cleaned_data.py` | **Cleaning Engine Fully Retired:** All mutation rules purged; `FeatureRetiredError` raised as documented. |
| **F-12** | **VERIFIED** | `cleanalytix/pipeline.py:59` | **Modern API & Result Container:** `ca.analyze()` and `CleanalytixResult` function as specified with dual attribute/dict access and `.summary()`. |
| **F-13** | **VERIFIED** | `cleanalytix/metrics/statistical.py` | **Bessel's Correction & Excess Kurtosis:** Sample variance uses `ddof=1` and Fisher excess kurtosis is correctly computed. |
| **F-14** | **VERIFIED** | `cleanalytix/generate_meta.py:545` | **Zero Baseline Self-Drift:** Baseline-only profiling eliminates holdout self-split and yields exactly 0.0 self-drift. |

---

## SECTION 9: PRIORITIZED RECOMMENDATIONS FOR NEXT ITERATION

Although this audit was strictly inspection-only, the following remediation steps are recommended to bring Cleanalytix to full mathematical and architectural excellence:

1. **Wire Timeliness into the Scoring Engine (Priority: HIGH):**
   - In `cleanalytix/scoring.py`, add `QualityDimension.TIMELINESS` to `DEFAULT_DIMENSION_WEIGHTS` (e.g. Completeness: 0.30, Validity: 0.25, Uniqueness: 0.20, Consistency: 0.15, Timeliness: 0.10).
   - In `score_dataset()`, evaluate datetime columns using `compute_temporal_metrics()` and calculate a genuine timeliness score.
2. **Declare `scipy` as a Core Dependency (Priority: CRITICAL):**
   - Add `"scipy>=1.10.0"` to `dependencies` in `pyproject.toml` and `requirements.txt` to prevent runtime `ModuleNotFoundError` on fresh installs.
3. **Implement True Numeric Divergence in `generate_meta.py` (Priority: HIGH):**
   - Replace the `psi_val * 0.5` and `min(1.0, psi_val * 0.25)` shortcuts in `generate_meta.py:230` with genuine discrete KL and JS divergence computed over fixed histogram bins (or continuous KDE).
4. **Fix Continuous Entropy Binning (Priority: MEDIUM):**
   - Replace quantile binning in `get_entropy()` with equal-width histogram binning or Scott's rule so that continuous entropy reflects actual distribution dispersion.
5. **Broaden Consistency Dimension (Priority: MEDIUM):**
   - Incorporate cross-column multicollinearity pairs and Cramér's V associations from `cleanalytix.metrics.relationships` into the Consistency dimension alongside outliers.
6. **Strengthen Test Suite (Priority: MEDIUM):**
   - Replace shallow smoke tests in `tests/test_pipeline.py` with exact numerical assertions on known toy datasets.
   - Fix the tautological assertion in `tests/test_fixes.py:196`.
