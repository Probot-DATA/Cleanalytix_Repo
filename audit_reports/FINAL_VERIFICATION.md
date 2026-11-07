# CLEANALYTIX v0.2.0 — POST-REMEDIATION FINAL VERIFICATION REPORT

**Inspection Date:** November 2026  
**Auditor:** Independent Post-Remediation Verification  
**Repository Inspected:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\cleanalytix`  
**Reference Document:** `audit_reports/MATHEMATICAL_SCORING_REMEDIATION_REPORT.md`  
**Mode:** Inspection-Only (No code modification, refactoring, or API alterations)

---

## Executive Summary

A targeted inspection was conducted across the five specified focus areas of the latest Cleanalytix v0.2.0 remediation. The remediation successfully resolved the primary mathematical shortcuts ($0.5 \times \text{PSI}$, $0.25 \times \text{PSI}$), restored the 6-dimension scoring architecture, fixed continuous entropy collapse via the Freedman-Diaconis rule, corrected constant-baseline KS step discontinuities, and added SciPy to package dependencies.

However, detailed inspection of the source code, mathematical representations, and test suite identified **four genuine issues**:
1. **Mathematical Terminology:** Constant-baseline PSI is inaccurately documented in the report as "exact continuous 3-bin PSI" when it is mathematically a discrete 3-bin surrogate approximation.
2. **Timeliness Monotonicity Penalty:** Unsorted timestamps in general tabular datasets receive an arbitrary 15% sequence penalty ($1.5\%$ on the dimension) regardless of temporal validity or freshness.
3. **Consistency Dimension Scope:** Statistical outliers in numeric columns continue to be treated as `Consistency` violations (constituting 100% of the Consistency score when no rules are configured).
4. **Test Suite Coverage Gap:** Custom intra-dimension `metric_weights` have zero automated regression tests in `tests/test_remediation.py`.

---

## Detailed Findings Across the 5 Focus Areas

### 1. Custom Weights & Applicability Governance
**Status: VERIFIED**

- **Default Weights:** `DEFAULT_DIMENSION_WEIGHTS` in `cleanalytix/scoring.py` defines all 6 standard quality dimensions:
  $$\text{Completeness: } 0.20, \text{ Validity: } 0.20, \text{ Uniqueness: } 0.15, \text{ Consistency: } 0.15, \text{ Timeliness: } 0.15, \text{ Stability: } 0.15$$
  The sum strictly equals $1.0000$.
- **Validation:** `resolve_dimension_weights()` strictly enforces:
  - Non-negative weights (raises `ValueError` on negative inputs).
  - Non-zero total allocation (raises `ValueError` if all weights sum to $\le 0$).
  - Valid dimension names against `QualityDimension` enum.
- **Overrides:**
  - Full overrides (all 6 dimensions provided) are normalized to sum to 1.0.
  - Partial overrides explicitly summing to 1.0 allocate 0.0 to omitted dimensions.
  - Partial overrides not summing to 1.0 assign user values to specified dimensions, inherit defaults for unspecified dimensions, and normalize the complete set to 1.0.
- **Applicability-Aware Normalization:**
  - Non-applicable dimensions (e.g., Stability without baseline, Timeliness without datetime) receive `effective_weight = 0.0`.
  - Configured weights of applicable dimensions are dynamically re-normalized such that $\sum_{d \in \mathcal{A}} W_{\text{effective}}(d) = 1.00$.
  - Each `DimensionScore` records `configured_weight`, `effective_weight`, `applicable`, and diagnostic `reason`.
- **Metric-Level Custom Weights:** `resolve_metric_weights()` properly merges user dictionary overrides with `DEFAULT_METRIC_WEIGHTS`, and internal dimension scorers normalize these weights before score aggregation.

---

### 2. Mathematics (KL, JS, PSI, Entropy, Constant Baselines)
**Status: ISSUE FOUND (Report Description Nuance)**

- **Numeric KL Divergence:** **VERIFIED.** Computed via exact closed-form formulation:
  $$D_{KL}(P \parallel Q) = \sum_{i=1}^K p_i \ln\left(\frac{p_i}{q_i}\right)$$
  Verified to be strictly asymmetric ($D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$).
- **Numeric JS Divergence:** **VERIFIED.** Computed via exact mixture formulation $M = 0.5(P + Q)$:
  $$\text{JSD}(P \parallel Q) = 0.5 D_{KL}(P \parallel M) + 0.5 D_{KL}(Q \parallel M)$$
  Verified to be strictly bounded in $[0, \ln(2)]$.
- **Population Stability Index (PSI):** **VERIFIED.** Satisfies the symmetric Jeffreys divergence identity:
  $$\text{PSI}(P, Q) = D_{KL}(P \parallel Q) + D_{KL}(Q \parallel P)$$
- **Continuous Shannon Entropy:** **VERIFIED.** Replaced quantile binning with equal-width histogram binning via the **Freedman-Diaconis rule** ($\Delta = 2 \cdot \text{IQR} / N^{1/3}$) with Scott's rule fallback ($\Delta = 3.49 s / N^{1/3}$). Normalized discrete entropy $H / \ln(K)$ properly orders distributions: Uniform ($0.9996$) > Normal ($0.8531$) > Exponential ($0.6822$) > Constant ($0.0000$).
- **Constant-Baseline Drift:**
  - **KS Statistic:** **VERIFIED.** Uses `scipy.stats.ks_2samp(b_arr, p_arr, method="asymp" if N > 100 else "auto")`. Correctly measures the empirical CDF supremum distance ($D = 1/N$ for 1 shifted observation, avoiding the previous 1.0 step discontinuity).
  - **Constant-Baseline PSI:** **ISSUE FOUND.**
    - *The Issue:* `audit_reports/MATHEMATICAL_SCORING_REMEDIATION_REPORT.md` (Table 2 and Section 3.4) describes this metric as "exact continuous 3-bin PSI".
    - *Mathematical Reality:* In continuous measure theory, a constant baseline represents a Dirac delta measure $\delta(x - c)$. The true continuous KL/Jeffreys divergence against a continuous distribution is infinite ($\infty$).
    - *Actual Code:* `cleanalytix/metrics/drift.py` partitions the real line into 3 bins:
      $$[-\infty, c - \delta), \quad [c - \delta, c + \delta], \quad (c + \delta, +\infty)$$
      with $\delta = 10^{-7} \times (|c| + 1)$ and Laplace smoothing $\epsilon$.
    - *Assessment:* The Python implementation is an effective, smooth **discrete 3-bin heuristic approximation**. Describing it as "exact continuous PSI" is mathematically inaccurate.

---

### 3. Timeliness and Consistency Dimensions
**Status: ISSUE FOUND (Two Conceptual & Implementation Defects)**

#### 3.1. Timeliness: Unsorted Timestamps Penalized
- **Code Location:** `cleanalytix/scoring.py` lines 270–273:
  ```python
  parse_rate = float(t_res.get("parse_rate", 1.0))
  is_mono = 1.0 if t_res.get("is_monotonic_increasing", False) else 0.85
  s_seq = 0.5 * parse_rate + 0.5 * is_mono
  col_score = (w_fresh * s_fresh + w_future * s_future + w_seq * s_seq) * 100.0
  ```
- **The Issue:** `is_monotonic_increasing` checks whether the rows of the DataFrame appear in strictly non-decreasing chronological order. In standard relational tables and tabular analytics, row order is technically arbitrary (e.g., sorted by `user_id`, partitioned by region, or returned without an `ORDER BY` clause).
- **Impact:** Any perfectly valid, completely fresh datetime column whose rows are not pre-sorted in ascending order receives an automatic 15% discount on `s_seq`, causing the column's Timeliness score to be capped at $98.50\%$ instead of $100.00\%$. Unsorted row order is conflated with sequence degradation.

#### 3.2. Consistency: Equated to Statistical Outliers When No Rules Configured
- **Code Location:** `cleanalytix/scoring.py` lines 504–508 and `cleanalytix/insights.py` lines 189–210:
  ```python
  elif has_outlier_cols:
      consistency_score = plausibility_score_val * 100.0
      consistency_applicable = True
      consistency_reason = None
  ```
- **The Issue:**
  1. When a user defines no `consistency_rules` (the standard out-of-the-box scenario), the `Consistency` dimension score ($w=1.00$) is derived exclusively from univariate statistical outlier counts (`plausibility_score_val`).
  2. In `insights.py`, univariate numeric outliers exceeding 10% are classified with `dimension=QualityDimension.CONSISTENCY` and message `"Statistical outliers: X values in 'col' are statistical outliers."`
- **Assessment:** In standard data quality frameworks (DAMA-DMBOK, ISO 8000), **Consistency** evaluates whether values across multiple columns, tables, or systems adhere to relational dependencies, logic rules, and referential integrity (e.g. `End_Date >= Start_Date`). Statistical outliers within a single continuous distribution measure dispersion or extreme values—they are NOT consistency defects. Treating outliers as consistency violations when no rules exist retains the conceptual defect from v0.2.0.

---

### 4. Testing Rigor & Custom Weights Verification
**Status: ISSUE FOUND (Test Coverage Gap)**

- **Existing Suite:** 42 tests in `tests/test_fixes.py`, `tests/test_pipeline.py`, and `tests/test_remediation.py` all execute cleanly with 0 failures and 0 warnings.
- **Genuine Mathematical Tests:**
  - `test_hand_calculated_deterministic_score`: Genuinely hand-calculates each component and asserts the exact expected score ($81.43$).
  - `test_exact_numeric_divergence_math`: Confirms asymmetry of KL ($D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$), the Jeffreys identity, and JSD bounds.
  - `test_constant_baseline_drift_continuity`: Confirms KS and PSI continuity on $N=1000$.
  - `test_continuous_entropy_fd_scott_binning`: Confirms strict entropy ordering across distributions.
  - `test_strict_non_mutation_across_entry_points`: Deep-copies input and baseline DataFrames and asserts `assert_frame_equal` before and after all entry points.
- **The Issue (Coverage Gap):**
  - While dimension-level custom weights are tested (`test_dimension_weight_validation`), there is **no test in the entire repository** that configures custom intra-dimension `metric_weights` via `CleanalytixConfig(metric_weights=...)` to verify that custom metric weights alter the resulting dimensional scores.
  - While manual execution verifies the feature works, it lacks regression test coverage in CI.

---

### 5. Packaging & Environment Portability
**Status: VERIFIED**

- **Dependency Declaration:**
  - `pyproject.toml`: `"scipy>=1.10.0"` is present under `dependencies`.
  - `requirements.txt`: `scipy>=1.10.0` is present.
- **Package Build:** Wheel `cleanalytix-0.2.0-py3-none-any.whl` (72 KB) builds successfully via standard setuptools/pip backend.
- **Wheel Metadata:** Wheel archive contains `cleanalytix-0.2.0.dist-info/METADATA` with verified directive:
  `Requires-Dist: scipy>=1.10.0`
- **Clean Import:** Verified in Python 3.10 with no circular imports or missing modules.

---

## Issue Summary Matrix

| # | Inspection Area | Finding | Classification | Recommendation |
| :- | :--- | :--- | :--- | :--- |
| **1** | Custom Weights | Default 6-dimension weights sum to 1.0; validation rejects negative and zero allocations; partial overrides re-normalize correctly; non-applicable dimensions receive weight 0.0. | **VERIFIED** | Ready for use. |
| **2** | Mathematics | Numeric KL, JS, PSI, and Freedman-Diaconis entropy are mathematically sound. Constant-baseline PSI is a discrete 3-bin surrogate with Laplace smoothing, but documented in report as "exact continuous PSI". | **ISSUE FOUND** | Update documentation to clarify that constant-baseline PSI is a discrete 3-bin approximation. |
| **3** | Timeliness | Datetime columns that are valid and fresh but not sorted in ascending row order receive an automatic 15% sequence penalty ($1.5\%$ on dimension). | **ISSUE FOUND** | Only penalize monotonicity if configured as an ordered time series, or sort before evaluating sequence integrity. |
| **4** | Consistency | When no consistency rules are defined, the Consistency dimension is 100% computed from statistical outliers, and outliers are reported as Consistency insights. | **ISSUE FOUND** | Classify statistical outliers under Validity / Plausibility; mark Consistency as non-applicable if no cross-column rules are configured. |
| **5** | Testing | 42 deterministic tests pass with 0 warnings. Verified exact hand-calculations. However, custom `metric_weights` has no dedicated regression test. | **ISSUE FOUND** | Add an automated unit test asserting custom `metric_weights` behavior in `tests/test_remediation.py`. |
| **6** | Packaging | SciPy is declared in `pyproject.toml` and `requirements.txt`. Wheel builds cleanly with valid metadata and installs/imports in a clean environment. | **VERIFIED** | Ready for distribution. |

---

## Production Readiness Verdict

**VERDICT: CONDITIONALLY QUALIFIED (Not Production-Ready Without Policy Alignment)**

While the core mathematical calculations (KL, JS, PSI, entropy, KS) and non-mutating architecture are verified and sound, **production deployment should be held** until the engineering team aligns on:
1. Whether unsorted row ordering should penalize Timeliness in general relational tables.
2. Whether statistical outliers should continue to serve as a surrogate for the Consistency dimension when no relational rules are provided.
3. Adding regression tests for custom intra-dimension metric weights.
4. Correcting the documentation to accurately describe constant-baseline PSI as a discrete 3-bin approximation.
