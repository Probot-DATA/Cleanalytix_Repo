# Priority Fixes & Remediation Roadmap: Cleanalytix

**Library Location:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\cleanalytix`  
**Audit Directory:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\audit_reports`  

---

## 1. Prioritization Framework

Issues are classified based on operational risk, data safety, and mathematical integrity:

| Priority | Class | Criteria |
|:---:|:---|:---|
| **P0** | **Blocker** | Unhandled pipeline crashes, fatal exceptions on common input types, permanent data loss. |
| **P1** | **Critical** | Silent monitoring failure, data corruption on non-unique indices, score distortion. |
| **P2** | **High** | Preempted cleaning rules, unintended dtype conversions, unannounced row drops. |
| **P3** | **Medium** | Cryptic crashes on duplicate column headers, package version mismatch, discarded metrics. |
| **P4** | **Low** | Performance bottlenecks, future deprecation warnings. |

---

## 2. P0: Blocker Fixes (Immediate Remediation Required)

### Fix 1: Resolve Pipeline Crash on Production Column Drop (BUG-01)
- **Target File:** `cleanalytix/pipeline.py` (lines 408–414)
- **Problem:** When cleaning drops a high-missingness column from production data, `generate_meta` asserts that column sets between reference and production match, crashing with `ValueError`.
- **Root Cause:** Passing raw `dataset_list` (which still has the dropped column) instead of matching cleaned schemas.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/pipeline.py
  +++ b/cleanalytix/pipeline.py
  @@ -407,7 +407,15 @@ def Run_DQ_Pipeline(
       if cleaning:
           prod_cleaned_list, prod_change_log = get_cleaned_data(dataset_names, new_dataset_list, prod_recommendations, rules=rules, interactive=interactive)
           prod_cleaned_list = [pd.DataFrame(df) for df in prod_cleaned_list]
  -        prod_meta_after = generate_meta(dataset_names,dataset_list, prod_cleaned_list, rules=rules)
  +        # Align baseline reference columns with remaining cleaned production columns
  +        aligned_ref_list = []
  +        for ref_df, p_df in zip(dataset_list, prod_cleaned_list):
  +            common_cols = [c for c in p_df.columns if c in ref_df.columns]
  +            aligned_ref_list.append(ref_df[common_cols].copy())
  +        prod_meta_after = generate_meta(dataset_names, aligned_ref_list, prod_cleaned_list, rules=rules)
           prod_meta_after = adjust_prod_meta_with_reference(reference_profile, base_meta_before, prod_meta_after, dataset_names, prod_cleaned_list)
  ```
- **Verification:** Run `Run_DQ_Pipeline` with production dataset where one column exceeds 40% missingness and `cleaning=True`. Ensure pipeline returns without exception.

---

### Fix 2: Prevent Rare Category Collapse & Catastrophic Data Loss (BUG-02)
- **Target File:** `cleanalytix/get_cleaned_data.py` (lines 359–391), `cleanalytix/column_role.py` (lines 166–185)
- **Problem:** Any string column without an identifier name pattern is classified as categorical. If no category exceeds 5% frequency, Rule 5 replaces 100% of values with `"Other"`, after which default deduplication drops 99%+ of records.
- **Root Cause:** Missing cardinality check and falling back to `"Other"` when `popular_categories` is empty.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/get_cleaned_data.py
  +++ b/cleanalytix/get_cleaned_data.py
  @@ -361,6 +361,10 @@ def clean(...):
           substitutes = {}
           popular_categories = counts[(counts / total * 100) > 5].index.tolist()
   
  +        # If no category is popular (>5%), the column is high-cardinality or free text; do not collapse
  +        if not popular_categories:
  +            return df[column]
  +
           for k, v in counts.items():
               if pd.isna(k):
                   continue
  ```
- **Verification:** Clean a DataFrame with 100 unique non-ID strings; verify all 100 distinct rows are preserved.

---

### Fix 3: Handle 0-Row and 1-Row Datasets Gracefully (BUG-03)
- **Target File:** `cleanalytix/generate_meta.py` (lines 943–953)
- **Problem:** `train_test_split` raises `ValueError` on DataFrames with fewer than 2 rows.
- **Root Cause:** Calling `train_test_split` without guarding for sample size.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/generate_meta.py
  +++ b/cleanalytix/generate_meta.py
  @@ -944,12 +944,16 @@ def generate_meta(...):
       if new_dataset_list is None:
           # internal train-test split for the "new" dataset
           for df in dataset_list:
  -            df_old, df_new = train_test_split(
  -                df,
  -                test_size=0.3,
  -                random_state=1,
  -            )
  -            df_old_list.append(df_old)
  -            df_new_list.append(df_new)
  +            if len(df) < 2:
  +                df_old_list.append(df.copy())
  +                df_new_list.append(df.copy())
  +            else:
  +                df_old, df_new = train_test_split(
  +                    df,
  +                    test_size=0.3,
  +                    random_state=1,
  +                )
  +                df_old_list.append(df_old)
  +                df_new_list.append(df_new)
  ```
- **Verification:** Run `generate_meta` on `pd.DataFrame()` and `pd.DataFrame({'a': [1]})`; verify valid return table without errors.

---

### Fix 4: Support Pandas Nullable Dtypes (`Int64`, `Float64`) (BUG-04)
- **Target File:** `cleanalytix/generate_meta.py` (lines 395–401)
- **Problem:** `col_vals.skew()` on nullable integer series with $<3$ values returns `pd.NA`. Calling `float(pd.NA)` raises `TypeError`.
- **Root Cause:** Casting `pd.NA` directly to `float()`.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/generate_meta.py
  +++ b/cleanalytix/generate_meta.py
  @@ -395,7 +395,8 @@ def get_metrics(...):
                       mean = float(col_vals.mean()) if pd.notna(col_vals.mean()) else np.nan
                       std = float(col_vals.std(ddof=0)) if col_vals.size > 1 and pd.notna(col_vals.std(ddof=0)) else 0.0
                       minv = float(col_vals.min()) if pd.notna(col_vals.min()) else np.nan
                       maxv = float(col_vals.max()) if pd.notna(col_vals.max()) else np.nan
                       median = float(col_vals.median()) if pd.notna(col_vals.median()) else np.nan
  -                    skewness = float(col_vals.skew())
  +                    raw_skew = col_vals.skew()
  +                    skewness = float(raw_skew) if pd.notna(raw_skew) else np.nan
  ```
- **Verification:** Profile a DataFrame containing `pd.Series([1, None, 2], dtype='Int64')`; verify successful profiling.

---

## 3. P1: Critical Fixes

### Fix 5: Detect Production Drift on Constant Baselines (BUG-05)
- **Target File:** `cleanalytix/generate_meta.py` (lines 717–719)
- **Problem:** If baseline has constant values, `len(bins) < 2` exits early and reports `(0.0, 0.0, 0.0)` drift, even if production has completely diverged.
- **Root Cause:** Assuming absence of baseline variance implies absence of drift.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/generate_meta.py
  +++ b/cleanalytix/generate_meta.py
  @@ -717,3 +717,7 @@ def KL_PSI_and_JS(...):
               if len(bins) < 2:
  -                return 0.0, 0.0, 0.0
  +                # If baseline is constant, check whether production matches that constant
  +                base_val = base_series.dropna().iloc[0] if len(base_series.dropna()) else 0.0
  +                prod_diff = (new_series.dropna() != base_val).mean()
  +                if prod_diff == 0.0:
  +                    return 0.0, 0.0, 0.0
  +                # Extreme drift: production differs from constant baseline
  +                return 10.0, 10.0, float(np.log(2))
  ```
- **Verification:** Test `KL_PSI_and_JS` with constant baseline `[10]*50` vs shifted production `[1000]*50`. Verify non-zero drift returned.

---

### Fix 6: Prevent Business Rule Corruption on Non-Unique Indices (BUG-06)
- **Target File:** `cleanalytix/get_cleaned_data.py` (lines 436–443)
- **Problem:** `df.loc[idx, column] = None` overwrites all rows sharing `idx`, destroying valid data in sister rows.
- **Root Cause:** Using label-based `.loc` during row iteration.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/get_cleaned_data.py
  +++ b/cleanalytix/get_cleaned_data.py
  @@ -436,8 +436,13 @@ def clean(...):
           func = rules.get(column)
           if func is None:
               return df[column]
  -        for idx, val in df[column].items():
  -            try:
  -                is_valid = func(val)
  -            except Exception:
  -                is_valid = False
  -            if not is_valid:
  -                df.loc[idx, column] = replacement if replacement is not None else None
  +        # Position-based cleaning to safeguard non-unique index labels
  +        col_vals = df[column].tolist()
  +        for pos in range(len(col_vals)):
  +            val = col_vals[pos]
  +            try:
  +                is_valid = func(val)
  +            except Exception:
  +                is_valid = False
  +            if not is_valid:
  +                df.iat[pos, df.columns.get_loc(column)] = replacement if replacement is not None else None
  ```
- **Verification:** Clean a DataFrame with duplicate index labels where only one row violates business rule. Verify valid duplicate row is unchanged.

---

### Fix 7: Fix Skewness Absolute Value Discrepancy in `clean_outliers` (BUG-07)
- **Target File:** `cleanalytix/get_cleaned_data.py` (line 284)
- **Problem:** Missing `abs()` causes negative skewness to be treated as low skewness.
- **Root Cause:** `skewness_level = "low" if col.skew() < 0.5 else ...`
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/get_cleaned_data.py
  +++ b/cleanalytix/get_cleaned_data.py
  @@ -284,1 +284,3 @@ def clean(...):
  -        skewness_level = "low" if col.skew() < 0.5 else "moderate" if col.skew() < 1 else "high"
  +        s_val = col.skew()
  +        abs_skew = abs(s_val) if pd.notna(s_val) else 0.0
  +        skewness_level = "low" if abs_skew < 0.5 else "moderate" if abs_skew < 1.0 else "high"
  ```
- **Verification:** Verify that a Series with skewness $<-1.5$ is classified as `"high"` and cleaned with MAD.

---

### Fix 8: Eliminate Self-Penalization on Baseline Data (BUG-09)
- **Target File:** `cleanalytix/generate_meta.py` (lines 943–953, 970)
- **Problem:** Baseline datasets are split 70/30 against themselves to compute drift, artificially reducing clean baseline DQ scores by up to 25 points.
- **Root Cause:** Running drift calculations on baseline data without a production comparison dataset.
- **Recommended Code Change:**
  ```diff
  --- a/cleanalytix/generate_meta.py
  +++ b/cleanalytix/generate_meta.py
  @@ -943,12 +943,6 @@ def generate_meta(...):
       if new_dataset_list is None:
  -        # internal train-test split for the "new" dataset
  -        for df in dataset_list:
  -            df_old, df_new = train_test_split(...)
  -        metrics = drift_metrics(dataset_name_list, df_old_list, df_new_list)
  +        # Baseline data has no drift against itself by definition
  +        metrics = [[ds, col, 0.0, 0.0, 0.0] for ds, df in zip(dataset_name_list, dataset_list) for col in df.columns]
  ```
- **Verification:** Run `Run_DQ_Pipeline` on clean baseline numbers and verify `dirty_scores["DQ_Score"] == 100.0`.

---

## 4. P2: High Priority Fixes

### Fix 9: Stop Premature Imputation in Rule 7 (BUG-10)
- **Target File:** `cleanalytix/get_cleaned_data.py` (lines 422–429)
- **Problem:** Rule 7 (Type Errors) calls `_fill_default_by_type`, filling all missing values with median/mode before Rule 2 (Missing Values) can execute.
- **Recommended Fix:** In Rule 7, coerce types to numeric/datetime/string; do not call `_fill_default_by_type`. Let Rule 2 perform imputation.

### Fix 10: Remove Identifier from Category Conversion (BUG-11)
- **Target File:** `cleanalytix/get_cleaned_data.py` (lines 758–762)
- **Problem:** Identifier columns are converted to pandas `category` dtype, bloating memory.
- **Recommended Code Change:**
  ```python
  # Change line 760:
  if role in {"categorical", "binary"} and item[c].dtype == "object":
      item[c] = item[c].astype("category")
  ```

### Fix 11: Make Row Deduplication Opt-In (BUG-14)
- **Target File:** `cleanalytix/get_cleaned_data.py` (lines 744–755), `cleanalytix/pipeline.py`
- **Problem:** When `interactive=False`, duplicate rows are dropped unconditionally without checking thresholds.
- **Recommended Fix:** Add a `deduplicate: bool = False` argument to `get_cleaned_data` and `Run_DQ_Pipeline`. Only drop duplicates when explicitly requested.

---

## 5. P3: Medium Priority Fixes

### Fix 12: Validate Unique Column Names on Ingestion (BUG-12)
- **Target File:** `cleanalytix/generate_meta.py` (line 341), `cleanalytix/pipeline.py`
- **Problem:** Duplicate column names cause `AttributeError: 'DataFrame' object has no attribute 'dtype'`.
- **Recommended Fix:** Add validation at function entry:
  ```python
  if not df.columns.is_unique:
      duplicates = df.columns[df.columns.duplicated()].tolist()
      raise ValueError(f"DataFrame has duplicate column names: {duplicates}")
  ```

### Fix 13: Synchronize Package Version (BUG-13)
- **Target File:** `cleanalytix/version.py:11`
- **Problem:** `version.py` states `__version__ = "0.1.0"` while `pyproject.toml` states `0.1.1`.
- **Recommended Fix:** Update `version.py` to `"0.1.1"`.

---

## 6. P4: Low Priority & Optimization Fixes

### Fix 14: Fast-Path Vectorized Datetime Parsing (BUG-16)
- **Target File:** `cleanalytix/preprocess_types.py` (lines 1580–1600)
- **Problem:** Row-by-row `dateparser.parse()` takes minutes on large datasets.
- **Recommended Fix:** Run `pd.to_datetime(series, errors='coerce', format='mixed')` first. Only pass unparsed values through `dateparser`.

### Fix 15: Fix Pandas FutureWarning Masks (BUG-17)
- **Target File:** `cleanalytix/preprocess_types.py` (lines 795, 847, 852, 904, 911)
- **Problem:** Incompatible dtype setting warnings on boolean masks.
- **Recommended Fix:** Initialize series explicitly: `pd.Series(False, index=strings.index, dtype=bool)`.
