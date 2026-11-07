# Test Coverage & Verification Report: Cleanalytix

**Library Location:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\cleanalytix`  
**Audit Directory:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\audit_reports`  
**Test Suite Executed:** `pytest` on `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\tests`  

---

## 1. Existing Test Suite Audit

The pre-existing test suite contains only two test files located in `tests/`:

- `tests/test_fixes.py`: 11 test functions (131 lines)
- `tests/test_pipeline.py`: 6 test functions (171 lines)
- **Total Test Cases:** **17**

### 1.1 Complete Test Case Inventory

| Test Identifier | File | Focus | Assessment |
|:---|:---|:---|:---|
| `test_production_meta_uses_production_missingness` | `test_fixes.py:14` | Verifies production metadata uses prod missing % | Narrow: tests single 3-row DataFrame. |
| `test_high_missing_column_drop_does_not_crash` | `test_fixes.py:24` | Verifies baseline column drop does not raise `KeyError` | Superficial: tests single column drop on baseline only; misses production crash (BUG-01). |
| `test_business_rule_cleaning_respects_non_default_index` | `test_fixes.py:33` | Verifies business rule cleaning works with custom index | Flawed assumption: tests unique custom index `[10, 20, 30]`; fails to test non-unique indices (BUG-06). |
| `test_mostly_numeric_mixed_column_counts_text_as_error` | `test_fixes.py:46` | Verifies `get_type_error_count` on `['1', '2', 'bad']` | Trivial: 5 string values only. |
| `test_binary_flag_not_treated_as_continuous` | `test_fixes.py:51` | Verifies `0/1` column gets `binary` role and 0 outliers | Valid for 0/1 integers; misses mixed string boolean types. |
| `test_identifier_column_skips_continuous_profiling` | `test_fixes.py:59` | Verifies `customer_id` name gets `identifier` role | Relies entirely on `customer_id` name hint; fails on unhinted IDs. |
| `test_duplicate_threshold_is_documented_default` | `test_fixes.py:67` | Verifies default threshold mapping and scoring | Checks single threshold constant. |
| `test_infer_types_for_new_false_preserves_production_dtype`| `test_fixes.py:80` | Verifies production dtype preservation when option is False | Good test for inference flag propagation. |
| `test_generate_meta_handles_boolean_columns` | `test_fixes.py:96` | Verifies boolean dtype gets `binary` role and no mean | Checks single 3-row series. |
| `test_change_log_skips_no_op_rules` | `test_fixes.py:103` | Verifies Rule 10 does not append to `change_log` | Unit test on mock recs DataFrame. |
| `test_existing_pipeline_smoke_still_passes` | `test_fixes.py:119` | End-to-end smoke test on sample CSVs | Uses 5-row toy data; overflows warnings emitted during run. |
| `test_public_import_surface` | `test_pipeline.py:65` | Verifies `__version__` and `Run_DQ_Pipeline` importable | Passes, but missed `version.py` mismatch (`0.1.0` vs `0.1.1`). |
| `test_pipeline_base_only_runs_without_errors` | `test_pipeline.py:70` | End-to-end baseline run on sample data | Verifies dictionary keys on 5-row dataset. |
| `test_pipeline_with_production_options_runs_without_errors`| `test_pipeline.py:82` | Full options test (weights, thresholds, prod, rules) | Broad smoke test; misses boundary values and calculation errors. |
| `test_infer_types_false_does_not_call_preprocessor...` | `test_pipeline.py:104`| Mocks preprocessor to ensure it is not called when False | Good unit test for monkeypatching. |
| `test_hard_type_inference_current_semantics` | `test_pipeline.py:122`| Tests type inference on synthetic DataFrame | Good semantic test, but emits multiple pandas FutureWarnings. |
| `test_pipeline_with_cleaning_and_interactive_branch_runs` | `test_pipeline.py:156`| Tests interactive prompt bypass with monkeypatched input | Verifies interactive answering loop. |

---

## 2. Test Coverage Analysis by Module

| Module | Total Lines | Estimated Line Coverage | Untested Functions & Branches |
|:---|:---:|:---:|:---|
| `cleanalytix/pipeline.py` | 436 | ~70% | Production column drop cleanup (BUG-01), custom score mode edge cases, empty branch outputs. |
| `cleanalytix/generate_meta.py` | 1,027 | ~55% | Datasets with $N < 2$, constant baseline drift, empty probability vectors, pandas nullable types (`Int64`), datetime interval bounds. |
| `cleanalytix/get_cleaned_data.py` | 768 | ~45% | Non-unique index business rules (BUG-06), rare category collapse on high-cardinality, negative skew outlier cleaning (BUG-07), Rule 6, Rule 9 errors. |
| `cleanalytix/compute_dq_score.py` | 425 | ~65% | Threshold error validation, weight length mismatch, weights sum validation, exponential decay with divergence, zero-row scoring. |
| `cleanalytix/adjust_prod_meta.py` | 239 | ~40% | Zero variance bounds, infinite limits, empty profiles, non-numeric profile fallbacks. |
| `cleanalytix/column_role.py` | 408 | ~75% | Harmonize series for datetime/bool drift, role blocking predicates. |
| `cleanalytix/preprocess_types.py` | 1,900 | ~35% | Currency normalization, unicode characters, mixed numeric date patterns, error handling in dateparser, thousands separators. |
| `cleanalytix/get_table_for_DQ_computation.py` | 116 | ~80% | Zero-length table division by zero, NaN fallback evaluation. |
| `cleanalytix/cleaning_recommendations.py`| 204 | ~75% | Fallback heuristics when `column_role` is absent, Rule 6 recommendation. |
| `cleanalytix/summarize_dataset_health.py` | 73 | ~85% | Empty metadata tables. |
| `cleanalytix/types.py` | 101 | 100% | Pure type annotations. |
| `cleanalytix/version.py` | 12 | 100% | Static attribute definition. |
| `cleanalytix/__init__.py` | 311 | 100% | Symbol re-exports. |

**Overall Estimated Codebase Coverage:** **~48%** (Substantial logic paths, especially defensive fallbacks and complex edge cases, remain completely unexecuted).

---

## 3. Adversarial & Edge-Case Testing Matrix

A battery of 20 adversarial edge cases was executed against the library using in-memory synthetic datasets to expose logical, mathematical, and data integrity failures.

| Case # | Scenario | Input Description | `generate_meta` | Pipeline (No Clean) | Pipeline (`cleaning=True`) | Failure Cause / Bug ID |
|:---:|:---|:---|:---:|:---:|:---:|:---|
| 1 | Empty DataFrame | `pd.DataFrame()` (0x0) | **CRASH** | **CRASH** | **CRASH** | `ValueError: n_samples=0` in `train_test_split` (BUG-03) |
| 2 | Zero Columns | `pd.DataFrame(index=[0, 1, 2])` | Pass | Pass | Pass | Returns 0 rows, score 0.0 |
| 3 | Zero Rows | `pd.DataFrame(columns=['a', 'b'])` | **CRASH** | **CRASH** | **CRASH** | `ValueError: n_samples=0` in `train_test_split` (BUG-03) |
| 4 | All-Null Column | `pd.DataFrame({'a': [None, None]})` | Pass | Pass | Pass | Missing % = 100%, score = 0.0 |
| 5 | Entirely Unique Nominal | `pd.DataFrame({'a': ['id_' + str(i) for i in range(100)]})` | Pass | Pass | **CRASH** | Rule 5 collapsed all to "Other"; dedup reduced to 1 row; `train_test_split` crashed (BUG-02, BUG-03) |
| 6 | Constant Column | `pd.DataFrame({'a': [42] * 20})` | Pass | Pass | **CRASH** | Dedup reduced to 1 row; post-clean `train_test_split` crashed (BUG-03) |
| 7 | Duplicate Column Names | `pd.DataFrame([[1, 2]], columns=['a', 'a'])` | **CRASH** | **CRASH** | **CRASH** | `AttributeError: 'DataFrame' object has no attribute 'dtype'` (BUG-12) |
| 8 | Mixed Numeric & String | `pd.DataFrame({'a': [1, 'two', 3.0, 'four']})` | Pass | Pass | Pass | Evaluated as categorical with type errors |
| 9 | Extreme Values / Overflow | `pd.DataFrame({'a': [1e308, -1e308, 1e200]})` | Pass | Pass | **CRASH** | Skewness overflowed to NaN; Rule 9 PowerTransform produced NaNs; dedup collapsed to 1 row (BUG-03) |
| 10 | Negative Numbers | `pd.DataFrame({'a': [-10, -20, -30]})` | Pass | Pass | Pass | Skewness handled properly |
| 11 | Infinite Values | `pd.DataFrame({'a': [1.0, np.inf, -np.inf, 2.0]})` | Pass | Pass | Pass | Infinite values treated as non-null |
| 12 | High Cardinality Cat | `pd.DataFrame({'cat': [f'cat_{i}' for i in range(500)]})` | Pass | Pass | **CRASH** | Collapsed to "Other", deduplicated to 1 row, crashed (BUG-02, BUG-03) |
| 13 | Whitespace Issues | `pd.DataFrame({'txt': [' hello', 'world ', ' both ']})` | Pass | Pass | Pass | Correctly flagged and stripped |
| 14 | Unicode Characters | `pd.DataFrame({'txt': ['café', '你好', '🌟', '\x00null']})` | Pass | Pass | Pass | Preserved and counted without encoding crashes |
| 15 | Inconsistent Datetimes | `pd.DataFrame({'dt': ['2024-01-01', '01/02/2024', 'bad']})` | Pass | Pass | Pass | Coerced or flagged properly |
| 16 | Boolean Columns | `pd.DataFrame({'flag': [True, False, True]})` | Pass | Pass | Pass | Binary role assigned |
| 17 | Nullable Pandas Types | `pd.DataFrame({'i': pd.Series([1, None, 2], dtype='Int64')})` | **CRASH** | **CRASH** | **CRASH** | `TypeError: float() argument must be real number, not 'NAType'` (BUG-04) |
| 18 | Single-Row Dataset | `pd.DataFrame({'a': [1], 'b': ['x']})` (N=1) | **CRASH** | **CRASH** | **CRASH** | `ValueError: n_samples=1` in `train_test_split` (BUG-03) |
| 19 | Two-Row Dataset | `pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})` (N=2) | Pass | Pass | Pass | Minimum viable sample size for split |
| 20 | Imbalanced Categorical | `pd.DataFrame({'cat': ['A'] * 99 + ['B']})` | Pass | Pass | Pass | Rare category identified (1%) |

---

## 4. Key Takeaways & Coverage Gaps

1. **Extreme Sensitivity to Sample Size:**  
   Datasets with $N < 2$ fail 100% of the time on baseline profiling due to un-guarded `train_test_split`.
2. **Cascading Failures in Cleaning Engine:**  
   When cleaning collapses columns (via Rule 5 or Rule 9) and deduplicates the DataFrame down to 1 row, post-cleaning profiling crashes on `train_test_split`.
3. **Absence of Real-World Pandas Type Testing:**  
   Nullable types (`Int64`, `Float64`, `boolean`, `string`) are completely untested in the existing test suite, concealing critical `TypeError` exceptions.
4. **Mocked / Toy Data Reliance:**  
   Existing tests rely almost exclusively on two 5-row CSV files (`base_customers.csv` and `prod_customers.csv`). No multi-thousand-row, high-cardinality, or adversarial datasets are included in the repository test suite.
