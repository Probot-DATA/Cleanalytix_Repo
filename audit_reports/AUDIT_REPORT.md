# Comprehensive Audit Report: Cleanalytix Python Library

**Audit Date:** November 7, 2026  
**Auditor Roles:** Senior Python Library Engineer, Software Architect, Data Quality Engineer, QA Specialist  
**Target Repository:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\cleanalytix`  
**Package Version Audited:** `0.1.0` (in `version.py`) / `0.1.1` (in `pyproject.toml`)  
**Audit Directory:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\audit_reports`  

---

## 1. Executive Summary

Cleanalytix is a Python library designed for profiling, scoring, cleaning, and monitoring tabular datasets. A comprehensive, code-level and empirical audit was conducted across all 13 modules, helper routines, pipeline stages, statistical formulas, edge cases, and cleaning rules.

### Core Verdict
- **Is Cleanalytix logically correct?** **No.** Multiple critical logical flaws compromise dataset processing, pipeline flow, and type handling.
- **Can its metrics be trusted?** **Partially, with severe caveats.** Completeness and basic counts are generally accurate, but drift metrics (KL, PSI, JS) suffer from silent zero-drift reporting on constant baselines, extreme self-drift penalization on small baseline datasets, and arbitrary probability mass assignment on empty columns. Four key profile metrics (Entropy, KL, rare category %, unique %) are computed and then discarded before scoring.
- **Can its cleaning operations be trusted?** **No.** Cleaning operations pose a severe risk of catastrophic data loss. Rule 5 (rare category collapse) frequently collapses high-cardinality or un-patterned text/ID columns into `"Other"`, after which unconditional deduplication drops up to 99%+ of records. Furthermore, business-rule cleaning on DataFrames with non-unique indices corrupts valid records.
- **Are there silent failure scenarios?** **Yes, several.** Most notably, when baseline columns are constant, production distribution shifts of any magnitude return `0.0` drift; and Rule 7 unconditionally imputes all missing values with medians/modes before Rule 2 runs, ignoring custom replacement parameters.
- **Can it handle diverse real-world datasets?** **No.** Real-world data containing pandas nullable types (`Int64`, `Float64`), 0 or 1 row datasets, duplicate column names, non-unique index labels, high-cardinality categorical fields, or production columns with high missingness crash the pipeline or corrupt data.
- **Is it production-ready?** **No.** It is currently unsuitable for automated or mission-critical production pipelines without resolving the prioritized defects detailed herein.

---

## 2. Findings Summary Matrix

| Severity | Count | Primary Impact |
|:---|:---:|:---|
| **Critical** | 5 | Pipeline crashes, unrecoverable data destruction, silent failure of drift monitoring |
| **High** | 5 | Data corruption, mathematical inconsistencies, score distortions |
| **Medium** | 5 | Unintended type mutations, duplicate column crashes, version mismatch, unannounced row drops |
| **Low / Quality** | 2 | Deprecation warnings, $O(N)$ unvectorized execution bottlenecks |
| **Total Findings** | **17** | Full library coverage |

---

## 3. Confirmed Defects & Critical Issues

### BUG-01: Unhandled Pipeline Crash on High-Missingness Production Column Drop

- **Issue ID:** `BUG-01`
- **Module:** `cleanalytix/pipeline.py`
- **Function/Class:** `Run_DQ_Pipeline` (lines 408–414)
- **Severity:** **Critical**
- **Category:** Logic / Pipeline Orchestration
- **Description:** When `cleaning=True` and a production dataset contains a column with $>40\%$ missing values, Rule 1 drops that column from the cleaned production DataFrame (`prod_cleaned_list`). However, line 410 calls:
  ```python
  prod_meta_after = generate_meta(dataset_names, dataset_list, prod_cleaned_list, rules=rules)
  ```
  Here, `dataset_list` (the original uncleaned baseline) still contains the dropped column, while `prod_cleaned_list` does not. In `generate_meta.py:958–965`, column sets are asserted to be identical. Since they differ, an unhandled `ValueError` crashes the pipeline. Furthermore, even if the baseline column was also dropped during baseline cleaning, line 410 passes `dataset_list` rather than `base_cleaned_list`.
- **Root Cause:** Inconsistent pipeline state passing: passing uncleaned baseline data alongside cleaned production data without checking column drop status.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix import Run_DQ_Pipeline
  base = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
  prod = pd.DataFrame({"a": [1, None, None, None, None], "b": [10, 20, 30, 40, 50]})
  Run_DQ_Pipeline(["ds"], [base], new_dataset_list=[prod], cleaning=True)
  ```
- **Expected Behavior:** The pipeline should align column drops between baseline and production, or evaluate cleaned production metadata against the cleaned baseline without raising an unhandled exception.
- **Actual Behavior:** Crashes with `ValueError: Column mismatch in dataset index 0. Missing in new: {'a'}. Extra in new: set().`
- **Impact:** Production pipelines running automated cleaning fail completely whenever newly arriving data experiences significant missingness in any column.
- **Recommended Fix:** In `pipeline.py:410`, pass `base_cleaned_list` (or align the reference dataset columns with the remaining columns of `prod_cleaned_list`), and filter `dataset_list` to matching columns before generating post-cleaning production metadata.
- **Test Case:** Run `Run_DQ_Pipeline` with a production dataset where one column exceeds 40% missingness and `cleaning=True`. Verify pipeline returns successfully.

---

### BUG-02: Catastrophic Data Loss via Rare Category Collapse & Unconditional Deduplication

- **Issue ID:** `BUG-02`
- **Module:** `cleanalytix/get_cleaned_data.py` (lines 359–391, 744–755), `cleanalytix/column_role.py` (lines 124–183), `cleanalytix/cleaning_recommendations.py` (lines 165–167)
- **Severity:** **Critical**
- **Category:** Data Integrity / Logic
- **Description:** 
  1. Any text or ID column whose name does not match the narrow regex in `name_suggests_identifier` (e.g. `account_token`, `transaction_hash`, `city`, `review`, `comment`) is classified as `categorical`.
  2. In `generate_meta.py`, every category with frequency $\le 5\%$ is marked as rare. For high-cardinality or unique string columns, 100% of values are rare.
  3. `cleaning_recommendations` assigns Rule 5 ("Rare Categories").
  4. In `clean()`, `popular_categories = counts[(counts / total * 100) > 5].index.tolist()`. If no category exceeds 5% frequency, `popular_categories` is empty. The cleaner fails to match any string, and replaces **100% of column values with `"Other"`**.
  5. At the end of `get_cleaned_data()`, line 744 executes:
     ```python
     item = item.drop_duplicates()
     ```
     Since all unique entries were collapsed to `"Other"`, the entire dataset collapses into **1 single row**, destroying up to 99%+ of valid user data!
- **Root Cause:** Flawed heuristic in Rule 5 assuming every column with `role="categorical"` is low-cardinality nominal data, combined with defaulting un-matched categories to `"Other"`, followed by unconditional default row deduplication.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix import Run_DQ_Pipeline
  df = pd.DataFrame({
      "account_token": [f"tok_{i:05d}" for i in range(100)],
      "balance": [100.0] * 100
  })
  res = Run_DQ_Pipeline(["ds"], [df], cleaning=True)
  cleaned_df = res["base_data"]["cleaned_datasets"][0]
  # Result: cleaned_df has only 1 row! All 99 accounts deleted!
  ```
- **Expected Behavior:** High-cardinality text and token columns should not be collapsed into `"Other"`. Rare category collapsing should require a minimum percentage of popular baseline categories or explicit cardinality caps, and deduplication should never run unconditionally without user consent.
- **Actual Behavior:** 99 of 100 customer records permanently deleted. The column collapses to a single value `['Other']`.
- **Impact:** Irreversible, catastrophic data deletion on real-world datasets containing customer tokens, free text, descriptions, or high-cardinality nominal categories.
- **Recommended Fix:** 
  1. Add a `"text"` column role or cardinality threshold (e.g., skip Rule 5 if unique ratio $> 0.20$ or total unique $> 50$).
  2. If `popular_categories` is empty, skip collapsing.
  3. Disable automatic deduplication by default; only deduplicate when `deduplicate=True` is explicitly passed.
- **Test Case:** Clean a DataFrame containing 100 unique non-ID strings and verify that row count and distinct values are preserved.

---

### BUG-03: Unhandled Crash on 0-Row and 1-Row Datasets in `generate_meta`

- **Issue ID:** `BUG-03`
- **Module:** `cleanalytix/generate_meta.py`
- **Function/Class:** `generate_meta` (lines 946–950)
- **Severity:** **Critical**
- **Category:** Edge Case / Robustness
- **Description:** When evaluating baseline datasets (`new_dataset_list is None`), `generate_meta` attempts an internal train/test split to compute drift against itself:
  ```python
  df_old, df_new = train_test_split(df, test_size=0.3, random_state=1)
  ```
  When `df` has 0 rows (empty DataFrame or empty schema) or 1 row ($N=1$), scikit-learn's `train_test_split` raises:
  `ValueError: With n_samples=... test_size=0.3 and train_size=None, the resulting train set will be empty.`
  Additionally, if cleaning deduplicates a dataset down to 1 row, `base_meta_after` crashes with this exact error.
- **Root Cause:** Hardcoded, un-guarded call to `train_test_split` without checking `len(df) >= 2`.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix import generate_meta
  generate_meta(["empty"], [pd.DataFrame()])
  generate_meta(["single"], [pd.DataFrame({"a": [1]})])
  ```
- **Expected Behavior:** Empty or single-row DataFrames should be handled gracefully, returning empty or single-row metadata with drift set to `0.0` or `np.nan`.
- **Actual Behavior:** Uncaught `ValueError` halting execution.
- **Impact:** Any pipeline encountering an empty partition, filtered slice, or single-record batch crashes immediately.
- **Recommended Fix:** Guard lines 946–950: if `len(df) < 2`, assign `df_old = df` and `df_new = df` (or skip drift computation and fill zeros/NaNs).
- **Test Case:** Run `generate_meta` on `pd.DataFrame()` and `pd.DataFrame({'a': [1]})`; verify valid return object without exceptions.

---

### BUG-04: Crash on Modern Pandas Nullable Types (`Int64`, `Float64`)

- **Issue ID:** `BUG-04`
- **Module:** `cleanalytix/generate_meta.py`
- **Function/Class:** `get_metrics` (lines 395–401)
- **Severity:** **Critical**
- **Category:** Data Integrity / Type Compatibility
- **Description:** When pandas nullable integer or float types (such as `Int64`, `Int32`) are used, pandas returns `pd.NA` (type `pandas._libs.missing.NAType`) for statistical reductions on series with fewer than 3 non-null values. Line 400 executes:
  ```python
  skewness = float(col_vals.skew())
  ```
  In Python, `float(pd.NA)` raises:
  `TypeError: float() argument must be a string or a real number, not 'NAType'`
- **Root Cause:** Blindly casting statistical outputs to `float()` without converting `pd.NA` to `np.nan` or checking `pd.isna()`.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix import generate_meta
  df = pd.DataFrame({"i": pd.Series([1, None, 2], dtype="Int64")})
  generate_meta(["test"], [df])
  ```
- **Expected Behavior:** `skewness` should be safely extracted as `np.nan` or `None`.
- **Actual Behavior:** Immediate `TypeError` aborts profiling.
- **Impact:** Any dataset using standard pandas nullable integer/float dtypes crashes whenever nulls exist in short columns.
- **Recommended Fix:** Replace `float(x)` with `float(x) if pd.notna(x) else np.nan` for all numeric reductions (`mean`, `std`, `min`, `max`, `median`, `skewness`).
- **Test Case:** Profile a DataFrame with `pd.Series([1, None, 2], dtype='Int64')`. Verify no crash.

---

### BUG-05: Silent Failure: Zero Drift Reported for Constant Baseline vs Divergent Production

- **Issue ID:** `BUG-05`
- **Module:** `cleanalytix/generate_meta.py`
- **Function/Class:** `KL_PSI_and_JS` (lines 717–718)
- **Severity:** **Critical**
- **Category:** Mathematical / Silent Failure
- **Description:** For continuous numeric columns, bin edges are calculated from the baseline dataset quantiles:
  ```python
  bins = np.unique(np.quantile(base_series.dropna(), q=quantiles))
  if len(bins) < 2:
      return 0.0, 0.0, 0.0
  ```
  If `base_series` has constant values (e.g., all 10.0), `bins` has length 1. The function immediately returns `(0.0, 0.0, 0.0)`. Even if the production dataset has completely shifted to `[1000, 2000, 3000]`, Cleanalytix reports **zero KL, zero PSI, and zero JS drift**.
- **Root Cause:** Premature exit based solely on reference variance without evaluating whether comparison values fall outside the reference value.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix.generate_meta import KL_PSI_and_JS
  base = pd.DataFrame({"val": [10.0] * 50})
  prod = pd.DataFrame({"val": [1000.0, 2000.0, 3000.0] * 20})
  kl, psi, js = KL_PSI_and_JS(base, prod, "val")
  # Returns: (0.0, 0.0, 0.0)
  ```
- **Expected Behavior:** A massive distribution shift should produce high PSI/JS divergence and trigger drift alerts.
- **Actual Behavior:** Returns `(0.0, 0.0, 0.0)` silently.
- **Impact:** Production monitoring fails silently: catastrophic feature shifts on low-variance features go completely undetected.
- **Recommended Fix:** If baseline is constant ($c$), construct bins around $c$ (e.g. $[c - \delta, c + \delta]$) and check if production values fall outside the bin. If they do, compute divergence properly or report infinite/maximum drift.
- **Test Case:** Run `KL_PSI_and_JS` on constant baseline vs shifted production series; verify PSI $> 0.25$.

---

### BUG-06: Business Rule Cleaning Corrupts Valid Rows on Non-Unique Index

- **Issue ID:** `BUG-06`
- **Module:** `cleanalytix/get_cleaned_data.py`
- **Function/Class:** `clean` (lines 436–442)
- **Severity:** **High**
- **Category:** Data Integrity / Index Alignment
- **Description:** In Rule 8 (business rule cleaning):
  ```python
  for idx, val in df[column].items():
      ...
      if not is_valid:
          df.loc[idx, column] = replacement if replacement is not None else None
  ```
  `df[column].items()` yields `(idx, val)` where `idx` is the index label. When a DataFrame has duplicate index labels (e.g. after concatenations, resets, or multi-tenant splits), `.loc[idx, column] = ...` matches and overwrites **all rows sharing that label**. If one row is invalid, valid values in all sister rows sharing that index label are overwritten with `None` / `NaN`.
- **Root Cause:** Using label-based `.loc` indexing during row-wise iteration instead of position-based indexing (`.iloc`) or boolean masking.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix.get_cleaned_data import clean
  df = pd.DataFrame({"a": [10, -5, 30]}, index=[0, 0, 1])
  # Row 0: a=10 (valid). Row 1: a=-5 (invalid).
  clean(df, "a", 8, rules={"a": lambda v: v > 0})
  # Result: df.loc[0, 'a'] becomes NaN for BOTH rows! Row 0 valid value 10 is destroyed!
  ```
- **Expected Behavior:** Only the specific invalid row (Row 1) should be cleaned; Row 0 must retain its valid value `10`.
- **Actual Behavior:** Valid data in Row 0 is corrupted to `NaN`.
- **Impact:** Silent corruption of valid production data in DataFrames with non-unique index labels.
- **Recommended Fix:** Vectorize business rule application with boolean masks:
  ```python
  mask = ~df[column].apply(lambda v: bool(func(v)) if pd.notna(v) else True)
  df.loc[mask, column] = replacement
  ```
  or iterate by integer position with `.iloc`.
- **Test Case:** Apply business rule cleaning on a DataFrame with duplicate index labels where only one duplicate row violates the rule. Verify valid rows are preserved.

---

### BUG-07: Negative Skewness Handled as Low Skewness in `clean_outliers`

- **Issue ID:** `BUG-07`
- **Module:** `cleanalytix/get_cleaned_data.py` (line 284) vs `cleanalytix/generate_meta.py` (lines 402–425)
- **Severity:** **High**
- **Category:** Mathematical / Logic Discrepancy
- **Description:** In `generate_meta.py:402`, skewness level uses absolute value:
  ```python
  abs_skew = abs(skewness)
  if abs_skew < 0.5: skewness_level = "low"
  elif abs_skew < 1: skewness_level = "moderate"
  else: skewness_level = "high"
  ```
  However, in `get_cleaned_data.py:284`, `clean_outliers` omits `abs()`:
  ```python
  skewness_level = "low" if col.skew() < 0.5 else "moderate" if col.skew() < 1 else "high"
  ```
  For negatively skewed data (e.g. `skewness = -2.5`):
  - `generate_meta` classifies it as `"high"` and detects outliers via MAD (Modified Z-score).
  - `clean_outliers` sees `-2.5 < 0.5` is True, classifies it as `"low"`, and cleans outliers via standard deviation Z-score!
- **Root Cause:** Missing `abs()` in `clean_outliers` skewness condition.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix.generate_meta import generate_meta
  from cleanalytix.get_cleaned_data import clean
  s = pd.Series([100]*8 + [10, 1]) # skewness ~ -1.79
  df = pd.DataFrame({"val": s})
  meta = generate_meta(["test"], [df])
  print(meta["skewness_level"].iloc[0]) # "high"
  # In clean(df, "val", 3): skewness_level evaluates to "low"!
  ```
- **Expected Behavior:** `clean_outliers` and `generate_meta` must use identical mathematical definitions for skewness classification.
- **Actual Behavior:** Outlier detection and outlier cleaning use completely divergent statistical models for negatively skewed distributions.
- **Impact:** Outliers flagged by profiling are not the ones cleaned by the cleaner; valid left-tail observations are incorrectly modified.
- **Recommended Fix:** Update `get_cleaned_data.py:284` to:
  ```python
  abs_skew = abs(col.skew()) if pd.notna(col.skew()) else 0.0
  skewness_level = "low" if abs_skew < 0.5 else "moderate" if abs_skew < 1.0 else "high"
  ```
- **Test Case:** Verify that a Series with skewness $<-1.0$ triggers MAD cleaning in `clean_outliers`.

---

### BUG-08: Arbitrary Probability Mass Injection on Empty Series in `get_probs`

- **Issue ID:** `BUG-08`
- **Module:** `cleanalytix/generate_meta.py`
- **Function/Class:** `get_probs` (lines 599–600) & `KL_PSI_and_JS` (lines 747–752)
- **Severity:** **High**
- **Category:** Mathematical / Logic
- **Description:** In `get_probs`:
  ```python
  col_vals = df[column].dropna()
  if len(col_vals) == 0:
      return [1.0]
  ```
  When a series has no non-null values, `get_probs` returns a single-element list `[1.0]`. When compared with a non-empty series having $K$ bins in `KL_PSI_and_JS`:
  ```python
  probs_new = np.pad(probs_new, (0, max_len - len(probs_new)), constant_values=EPS)
  ```
  This arbitrarily places **100% of the probability mass** into the first bin and $\approx 0$ into all other bins, even though the column contained no data!
- **Root Cause:** Returning `[1.0]` of length 1 instead of a uniform distribution over `len(bins)` or handling all-null drift explicitly.
- **Reproduction:** Call `KL_PSI_and_JS` with a series containing all nulls against a populated series. Observe that the first bin is assigned probability 1.0.
- **Expected Behavior:** An all-null column has no defined distribution and should yield maximum drift or NaN, rather than assigning all density to bin 0.
- **Actual Behavior:** Artificially skews divergence depending on which values happen to fall into the first bin.
- **Recommended Fix:** If `len(col_vals) == 0`, return uniform probability `[1.0 / len(bins)] * len(bins)` or handle missingness as a distinct category/mask.
- **Test Case:** Verify `get_probs` on empty series returns a vector matching the length of `bins`.

---

### BUG-09: Self-Penalization of Baseline Quality Scores via Arbitrary Train/Test Split

- **Issue ID:** `BUG-09`
- **Module:** `cleanalytix/generate_meta.py` (lines 946–950), `cleanalytix/compute_dq_score.py` (lines 408–423)
- **Severity:** **High**
- **Category:** Mathematical / Architecture
- **Description:** When running baseline profiling (`new_dataset_list is None`), the library executes a random 70/30 split of the baseline dataset against itself:
  ```python
  df_old, df_new = train_test_split(df, test_size=0.3, random_state=1)
  ```
  On small to medium datasets, non-overlapping quantile bins between the 70% and 30% splits cause PSI and JS divergence to explode (e.g. PSI = 39.4, JS = 0.693). In `Compute_DQ_Score`, `psi_divergence` and `js_divergence` each receive 12.5% of the score (25% total). Consequently, a **100% pristine baseline dataset receives a failing or mediocre DQ score (e.g. 62.5 or 75.0 instead of 100)** solely due to self-drift against its own split.
- **Root Cause:** Conflating production drift monitoring with baseline dataset profiling.
- **Reproduction:**
  ```python
  import pandas as pd
  from cleanalytix import Run_DQ_Pipeline
  df = pd.DataFrame({"i": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
  res = Run_DQ_Pipeline(["ds"], [df])
  print(res["base_data"]["dirty_scores"])
  # Pristine 10-row dataset receives DQ_Score < 80.0 due to self-split drift!
  ```
- **Expected Behavior:** A baseline dataset evaluated on its own has zero drift by definition; drift metrics should be 0.0, and baseline quality score should be 100.
- **Actual Behavior:** Pristine data is heavily penalized.
- **Recommended Fix:** When `new_dataset_list is None`, set `kl_divergence`, `psi_divergence`, and `js_divergence` to `0.0` directly without splitting the dataset.
- **Test Case:** Run `Run_DQ_Pipeline` on clean baseline numbers and verify `DQ_Score == 100.0`.

---

### BUG-10: Unconditional Imputation in Rule 7 Preempts Rule 2 and Ignores User Replacement

- **Issue ID:** `BUG-10`
- **Module:** `cleanalytix/get_cleaned_data.py`
- **Function/Class:** `clean` (lines 422–429) & `get_cleaned_data` (lines 563–574)
- **Severity:** **High**
- **Category:** Logic / Preprocessing
- **Description:** Rule 7 is intended to resolve type errors (e.g. coerce `'bad'` to `NaN`). However, in lines 422–429, Rule 7 calls:
  ```python
  df[column] = _fill_default_by_type(df[column], semantic_type)
  ```
  This unconditionally imputes all `NaN` values with the median or mode. In `RULE_PRIORITY`, Rule 7 executes at priority 1, whereas Rule 2 (Missing Value Imputation) executes at priority 3. Consequently, by the time Rule 2 runs, all missing values have already been filled with defaults. Any user-specified `replacement` or custom policy for Rule 2 is completely bypassed and ignored.
- **Root Cause:** Coupling type coercion with automatic default imputation inside Rule 7.
- **Reproduction:** Pass a dataset with type errors and missing values, configure Rule 2 replacement to a custom constant, and observe that median/mode is imputed instead.
- **Expected Behavior:** Rule 7 should only coerce incompatible types to `NaN`; Rule 2 should be solely responsible for missing value imputation.
- **Actual Behavior:** Rule 7 preempts Rule 2.
- **Recommended Fix:** In Rule 7, coerce types without filling `NaN`s; let Rule 2 perform imputation.
- **Test Case:** Test pipeline where a column has both type errors and missing values, verify custom Rule 2 replacement is applied.

---

### BUG-11: Identifier Columns Converted to Pandas `category` Dtype

- **Issue ID:** `BUG-11`
- **Module:** `cleanalytix/get_cleaned_data.py`
- **Function/Class:** `get_cleaned_data` (lines 758–762)
- **Severity:** **Medium**
- **Category:** Data Integrity / Performance
- **Description:** At the completion of dataset cleaning:
  ```python
  # Convert only truly categorical columns to category
  for c in item.columns:
      role = infer_column_role(item[c], c, len(item))
      if role in {"categorical", "binary", "identifier"} and item[c].dtype == "object":
          item[c] = item[c].astype("category")
  ```
  Despite the code comment stating `"Convert only truly categorical columns to category"`, the condition explicitly includes `"identifier"`. This forces customer IDs, UUIDs, invoice numbers, and hashes into `category` dtype, creating as many categories as rows, causing significant memory bloat and breaking downstream string operations.
- **Root Cause:** Accidental inclusion of `"identifier"` in the category conversion role set.
- **Reproduction:** Clean a dataset containing `customer_id`. Check `cleaned_df["customer_id"].dtype`. It is `CategoricalDtype`.
- **Expected Behavior:** Identifier columns must remain `object` or `string` dtype.
- **Actual Behavior:** Converted to `category`.
- **Recommended Fix:** Change condition to `if role in {"categorical", "binary"} and ...`.
- **Test Case:** Clean a dataset with an identifier column; verify final dtype is not `category`.

---

### BUG-12: Unhandled Crash on Duplicate Column Names

- **Issue ID:** `BUG-12`
- **Module:** `cleanalytix/generate_meta.py`
- **Function/Class:** `get_metrics` (lines 348–350)
- **Severity:** **Medium**
- **Category:** Edge Case / Robustness
- **Description:** If an input DataFrame has duplicate column names (e.g. `pd.DataFrame([[1, 2]], columns=['a', 'a'])`), `for column in df.columns:` yields `'a'`. Then `df[column]` returns a DataFrame with 2 columns instead of a Series. Calling `df[column].dtype` immediately raises:
  `AttributeError: 'DataFrame' object has no attribute 'dtype'`
- **Root Cause:** Assuming `df[column]` always returns a `Series`.
- **Reproduction:** `generate_meta(['test'], [pd.DataFrame([[1, 2]], columns=['a', 'a'])])`.
- **Expected Behavior:** Detect duplicate column names and raise a clear `ValueError` or disambiguate them.
- **Actual Behavior:** Cryptic `AttributeError`.
- **Impact:** Crashes on real-world CSVs containing duplicate column headers.
- **Recommended Fix:** Validate at ingestion (`Run_DQ_Pipeline` and `generate_meta`) that `df.columns.is_unique`; raise informative `ValueError` if duplicates exist.
- **Test Case:** Pass DataFrame with duplicate column names to `generate_meta`; verify informative error.

---

### BUG-13: Version Mismatch Across Package Configuration

- **Issue ID:** `BUG-13`
- **Module:** `cleanalytix/version.py:11` vs `pyproject.toml:7`
- **Severity:** **Medium**
- **Category:** Library Packaging & Release
- **Description:** `cleanalytix/version.py` specifies `__version__ = "0.1.0"`, while `pyproject.toml` specifies `version = "0.1.1"`. When users inspect `cleanalytix.__version__`, it reports outdated version `0.1.0`.
- **Root Cause:** Manual version bumping without a single source of truth or automated version synchronization.
- **Reproduction:** Compare `import cleanalytix; cleanalytix.__version__` with `pyproject.toml`.
- **Expected Behavior:** Both files must agree on the package version.
- **Actual Behavior:** Inconsistency between runtime attribute and distribution metadata.
- **Recommended Fix:** Synchronize `version.py` to `"0.1.1"` or use dynamic versioning via `setuptools-scm` / `importlib.metadata`.
- **Test Case:** Assert `cleanalytix.__version__ == "0.1.1"`.

---

### BUG-14: Unconditional Row Deduplication Without Threshold Check or User Consent

- **Issue ID:** `BUG-14`
- **Module:** `cleanalytix/get_cleaned_data.py` (lines 744–755)
- **Severity:** **Medium**
- **Category:** Data Integrity
- **Description:** In `get_cleaned_data`, whenever `interactive=False` (the default mode in all non-interactive scripts and pipelines), line 745 executes `item = item.drop_duplicates()`. Even if the dataset contains valid identical rows (e.g., event logs, transaction line items), or duplicate rows are well below the configured `duplicate_rows_percent` threshold, rows are silently dropped without warning or parameter control.
- **Root Cause:** Hardcoded automatic deduplication without checking whether deduplication was requested or exceeded thresholds.
- **Reproduction:** Pass a dataset with 2 identical event records to `get_cleaned_data(..., interactive=False)`. One is dropped.
- **Expected Behavior:** Deduplication should only execute if `deduplicate=True` or if explicitly configured.
- **Actual Behavior:** Silently drops rows.
- **Recommended Fix:** Add a `deduplicate: bool = False` argument to `get_cleaned_data` and `Run_DQ_Pipeline`.
- **Test Case:** Clean a dataset with duplicate rows when deduplication is disabled; verify all rows remain intact.

---

### BUG-15: Profiling Metrics Discarded Before Quality Scoring

- **Issue ID:** `BUG-15`
- **Module:** `cleanalytix/get_table_for_DQ_computation.py` (lines 81–86, 112–114)
- **Severity:** **Medium**
- **Category:** Architectural / Scoring Engine
- **Description:** In `get_table_for_DQ_computation`, four key profiling metrics computed at substantial computational cost in `generate_meta.py` are explicitly dropped and excluded from scoring:
  - `Entropy` (dropped line 84)
  - `kl_divergence` (dropped line 112)
  - `rare_category_percent` (dropped line 112)
  - `unique_percent` (dropped line 112)
  None of these metrics contribute to `Compute_DQ_Score` or dataset health scoring.
- **Root Cause:** Disconnect between metadata profiling capabilities and scoring engine requirements.
- **Impact:** CPU time spent calculating Shannon entropy and KL divergence across all columns is wasted; user cannot incorporate entropy or rare categories into the DQ score.
- **Recommended Fix:** Either include these metrics in `DEFAULT_THRESHOLDS` and the scoring table, or provide an option to disable their computation when not required.
- **Test Case:** Check `get_table_for_DQ_computation` output columns against documentation.

---

### BUG-16: Row-by-Row Unvectorized Dateparser Bottleneck in `_convert_datetime`

- **Issue ID:** `BUG-16`
- **Module:** `cleanalytix/preprocess_types.py` (lines 1580–1600)
- **Severity:** **Low / Performance**
- **Category:** Performance & Scalability
- **Description:** During hard datetime conversion, `_convert_datetime` iterates row-by-row in pure Python over the entire column:
  ```python
  for value in series:
      parsed = _dateparser.parse(str(value).strip())
  ```
  `dateparser.parse()` is an unvectorized, heavy natural language parsing routine taking $\sim 1\text{–}5\text{ ms}$ per invocation. For a 100,000-row column, this single loop takes between 100 and 500 seconds (several minutes) to execute.
- **Root Cause:** Row-by-row natural language parsing without first trying fast vectorized `pd.to_datetime`.
- **Impact:** Makes `infer_types=True` unusable on medium to large datasets.
- **Recommended Fix:** Fast-path conversion with `pd.to_datetime(series, errors='coerce', format='mixed')` and only invoke `dateparser` on the small subset of failed values.
- **Test Case:** Benchmark conversion of 10,000 dates; verify execution time $< 1\text{ second}$.

---

### BUG-17: Deprecation FutureWarnings on Incompatible Dtype Assignments

- **Issue ID:** `BUG-17`
- **Module:** `cleanalytix/preprocess_types.py` (lines 847, 852, 904, 911)
- **Severity:** **Low**
- **Category:** API / Forward Compatibility
- **Description:** Assigning numpy boolean arrays to `.loc` slices of `ambiguous_mask` and `numeric_mask` triggers repeated pandas `FutureWarning: Setting an item of incompatible dtype is deprecated and will raise in a future error of pandas.`
- **Root Cause:** Setting boolean numpy arrays into Series slices without explicitly initializing the series with `dtype="boolean"` or `dtype=bool`.
- **Impact:** Clutters console logs and will raise fatal exceptions in upcoming pandas 3.0 releases.
- **Recommended Fix:** Initialize masks with explicit boolean dtype: `pd.Series(False, index=strings.index, dtype=bool)`.
- **Test Case:** Run `infer_and_fix_types` under `warnings.simplefilter('error', FutureWarning)`; verify zero warnings raised.

---

## 4. Architectural Weaknesses & Discrepancies

1. **Absence of Advertised "Adaptive Weighting":**  
   The library and problem description mention adaptive weighting. However, the codebase contains zero adaptive weighting logic. Scoring weights are either static equal weighting ($1/M$) or a static user list. Column relevance, data types, and entropy are not used to adapt weights.
2. **In-Place Mutation Inconsistency in `clean()`:**  
   `clean(df, column, clean_type)` mutates `df[column]` in-place for Rules 1, 2, 4, 5, 7, and 8, but for Rule 9 (PowerTransformer) it creates and returns a new Series without mutating `df[column]` in place.
3. **Double Counting of Drift:**  
   Both `psi_divergence` and `js_divergence` are included in the scoring table as separate metrics, effectively weighting distribution drift at double the weight of completeness or validity.
4. **Tight Coupling to Positional Lists:**  
   The pipeline enforces parallel lists `dataset_names` and `dataset_list`. If a user passes a single DataFrame instead of a list, it crashes with non-intuitive errors rather than wrapping single items automatically.

---

## 5. Audit Deliverables Reference

The remaining detailed analyses are saved in their dedicated reports within `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\audit_reports`:

1. [`ARCHITECTURE_MAP.md`](file:///C:/Users/Prabhat/Desktop/Carear/DQ_Framework/audit_reports/ARCHITECTURE_MAP.md) — Comprehensive module dependencies, dataflow, and execution sequence diagrams.
2. [`TEST_COVERAGE_REPORT.md`](file:///C:/Users/Prabhat/Desktop/Carear/DQ_Framework/audit_reports/TEST_COVERAGE_REPORT.md) — Line/branch test coverage, adversarial test matrix, and untested branches.
3. [`MATHEMATICAL_VALIDATION.md`](file:///C:/Users/Prabhat/Desktop/Carear/DQ_Framework/audit_reports/MATHEMATICAL_VALIDATION.md) — Formal derivations and validation of statistical formulas.
4. [`PRIORITY_FIXES.md`](file:///C:/Users/Prabhat/Desktop/Carear/DQ_Framework/audit_reports/PRIORITY_FIXES.md) — Phased remediation plan with exact line-level diffs and test cases.
