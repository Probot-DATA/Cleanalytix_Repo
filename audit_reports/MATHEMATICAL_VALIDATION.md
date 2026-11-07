# Mathematical & Statistical Validation Report: Cleanalytix

**Library Location:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\cleanalytix`  
**Audit Directory:** `C:\Users\Prabhat\Desktop\Carear\DQ_Framework\audit_reports`  

---

## 1. Overview & Verification Methodology

Every mathematical formula, statistical metric, divergence calculation, score transformation, and aggregation mechanism implemented in Cleanalytix was independently audited. Each implementation was compared against established statistical theory, tested for numerical stability, inspected for edge-case boundaries, and verified for aggregation consistency.

---

## 2. Descriptive & Summary Statistics

### 2.1 Completeness & Cardinality
- **Missing Percentage:**  
  $$\text{missing\_percent} = \frac{\text{missing\_count}}{\max(1, N)} \times 100$$  
  *Implementation:* `(missing_count / max(1, n_rows)) * 100`.  
  *Status:* **Mathematically Sound.** Safely guards against division-by-zero for $N=0$.
- **Unique Percentage:**  
  $$\text{unique\_percent} = \frac{\text{unique\_count}}{\max(1, N)} \times 100$$  
  *Implementation:* `(unique_count / max(1, n_rows)) * 100` where `unique_count = df[column].nunique(dropna=True)`.  
  *Status:* **Mathematically Sound.** Drops `NA` before counting distinct values.
- **Mode Percentage:**  
  $$\text{mode\_percent} = \frac{\sum_{i=1}^N \mathbb{I}(x_i = \text{mode})}{\max(1, N)} \times 100$$  
  *Implementation:* `((df[column] == mode).sum() / max(1, n_rows)) * 100`.  
  *Status:* **Minor Risk.** For mixed-type columns, `df[column] == mode` can emit pandas comparison warnings or fail to match NaN modes when all values are null.

### 2.2 Standard Deviation & Variance Degrees of Freedom
- **Standard Deviation (`std`):**  
  In `generate_meta.py:396`:  
  ```python
  std = float(col_vals.std(ddof=0)) if col_vals.size > 1 else 0.0
  ```
  *Analysis:* The code sets `ddof=0`, which computes the **population standard deviation** ($\sigma = \sqrt{\frac{1}{N}\sum(x_i - \mu)^2}$) rather than the unbiased **sample standard deviation** ($s = \sqrt{\frac{1}{N-1}\sum(x_i - \bar{x})^2}$).  
  *Impact:* For small sample sizes (e.g. $N=5$), `ddof=0` underestimates standard error by $\sqrt{4/5} \approx 10.5\%$, making standard deviation smaller and Z-scores artificially larger.

### 2.3 Skewness Computation & Boundary Breakdown
- **Implementation:** `skewness = float(col_vals.skew())`.
- **Mathematical Risk:** Sample skewness requires at least 3 distinct observations ($N \ge 3$) and non-zero variance.
  - When $N < 3$ or variance is 0, pandas returns `np.nan` (or `pd.NA` on nullable integers).
  - In `generate_meta.py:402`:
    ```python
    abs_skew = abs(skewness) if skewness is not None else 0.0
    ```
    If `skewness` is `np.nan`, `abs(np.nan)` is `nan`.
    In Python:
    `nan < 0.5` $\to$ `False`  
    `nan < 1.0` $\to$ `False`  
    The control flow falls through to `else: skewness_level = "high"`.
  - **Critical Flaw:** Any column with undefined or uncomputable skewness is classified as **"high skewness"**, triggering Modified Z-score outlier detection and Yeo-Johnson power transformation recommendations!

---

## 3. Outlier Detection Mechanisms

Cleanalytix selects one of three outlier models based on skewness level:

### 3.1 Low Skewness ($|\text{skew}| < 0.5$): Standard Z-Score
- **Mathematical Formula:**  
  $$Z_i = \frac{x_i - \bar{x}}{\sigma}, \quad \text{Outlier if } |Z_i| \ge 3.0$$
- *Implementation:* `z = (col_vals - col_vals.mean()) / std`, outlier if `np.abs(z) >= 3`.
- *Status:* **Mathematically Sound.** Follows the standard 3-sigma rule for near-normal data.

### 3.2 Moderate Skewness ($0.5 \le |\text{skew}| < 1.0$): Tukey's Interquartile Range (IQR)
- **Mathematical Formula:**  
  $$\text{IQR} = Q_3 - Q_1, \quad \text{Bounds} = [Q_1 - 1.5 \cdot \text{IQR}, \; Q_3 + 1.5 \cdot \text{IQR}]$$
  $$\text{Outlier if } x_i < \text{Lower} \quad \text{or} \quad x_i > \text{Upper}$$
- *Implementation:* `Q1 = col_vals.quantile(0.25)`, `Q3 = col_vals.quantile(0.75)`, `IQR = Q3 - Q1`.
- *Status:* **Mathematically Sound.** Strictly implements Tukey's fences.

### 3.3 High Skewness ($|\text{skew}| \ge 1.0$): Median Absolute Deviation (MAD)
- **Mathematical Formula (Iglewicz & Hoaglin, 1993):**  
  $$\text{MAD} = \text{median}(|x_i - \tilde{x}|), \quad \text{Modified } Z_i = \frac{0.6745 \cdot (x_i - \tilde{x})}{\text{MAD}}$$
  $$\text{Outlier if } |\text{Modified } Z_i| > 3.5$$
- *Implementation:*  
  ```python
  median_v = col_vals.median()
  MAD = np.median(np.abs(col_vals - median_v))
  modified_z = 0.6745 * (col_vals - median_v) / MAD
  outliers = int((np.abs(modified_z) > 3.5).sum())
  ```
- *Status:* **Mathematically Sound.** Factor $0.6745$ is the reciprocal of $1.4826$ ($\approx \Phi^{-1}(0.75)$), properly normalizing MAD to estimate standard deviation for normal distributions.

### 3.4 Logic Discrepancy (BUG-07)
As identified in BUG-07, `clean_outliers` in `get_cleaned_data.py:284` omits `abs()`. For negative skewness ($s < -1.0$), `clean_outliers` erroneously applies Z-score instead of MAD.

---

## 4. Information Theory & Divergence Metrics

### 4.1 Shannon Entropy (Normalized)
- **Mathematical Formula:**  
  $$H(X) = -\sum_{i=1}^K p_i \ln(p_i), \quad H_{\text{norm}}(X) = \frac{H(X)}{\ln(K)}$$
- **Implementation in `generate_meta.py:95–103`:**
  ```python
  probs = np.array(get_probs(df_base, column, bins), dtype=float)
  probs = probs / (probs.sum() + EPS)
  entropy = -np.sum(probs * np.log(probs + EPS))
  return float(entropy / math.log(len(probs) + EPS))
  ```
- **Validation Findings:**
  1. **Base Logarithm Consistency:** Both `np.log` and `math.log` use natural log ($\ln$). The theoretical maximum entropy for $K$ discrete bins is $\ln(K)$. Dividing by $\ln(K)$ properly normalizes entropy to $[0.0, 1.0]$.
  2. **Epsilon Perturbation:** `EPS = 1e-9` is added in `get_probs` and again in `get_entropy`. For very large $K$ ($K > 10,000$), accumulated epsilon slightly elevates baseline entropy, but the effect on 5–20 bins is negligible ($< 10^{-7}$).
  3. **Quantile Binning:** Continuous columns use quantiles (5 bins for $N < 10,000$, 10 for $N < 50,000$, 20 for larger). For a continuous uniform distribution, each quantile contains equal mass ($p_i \approx 1/K$), yielding normalized entropy $\approx 1.0$.

---

### 4.2 Kullback-Leibler (KL) Divergence
- **Mathematical Formula:**  
  $$D_{\text{KL}}(P \parallel Q) = \sum_{i=1}^K P_i \ln\left(\frac{P_i}{Q_i}\right)$$
- **Implementation in `generate_meta.py:756`:**
  ```python
  KL_div = float(np.sum(probs_base * np.log(probs_base / (probs_new + EPS))))
  ```
- **Validation Findings:**
  - $D_{\text{KL}}(P \parallel Q) \ge 0$ (Gibbs' inequality).
  - **Asymmetric Support Vulnerability:** If $P_i > 0$ and $Q_i = 0$, $D_{\text{KL}}$ diverges to $+\infty$. Cleanalytix adds `EPS` ($10^{-9}$) to the denominator:
    $$\ln\left(\frac{P_i}{10^{-9}}\right) = \ln(P_i) + 9\ln(10) \approx \ln(P_i) + 20.72$$
    This caps single-bin divergence at $\approx 20.72 \cdot P_i$, preventing `inf` while signaling extreme mismatch.
  - **Numerator Instability:** In `probs_base / (probs_new + EPS)`, epsilon is added only to the denominator. If $P_i = 0$ (or near zero), $P_i \ln(P_i / Q_i)$ approaches 0 mathematically ($\lim_{x \to 0} x \ln x = 0$). In code:
    If `probs_base` has values $\le 0$, `np.log` can raise warnings. In practice, `get_probs` adds `EPS` to all bins, keeping $P_i \ge 10^{-9}$.

---

### 4.3 Population Stability Index (PSI)
- **Mathematical Formula:**  
  $$\text{PSI} = \sum_{i=1}^K (P_i - Q_i) \cdot \ln\left(\frac{P_i}{Q_i}\right)$$
- **Implementation in `generate_meta.py:757`:**
  ```python
  PSI_div = float(np.sum((probs_base - probs_new) * np.log(probs_base / (probs_new + EPS))))
  ```
- **Validation Findings:**
  1. **Symmetry:** Notice that $(P_i - Q_i) \ln(P_i / Q_i) = (Q_i - P_i) \ln(Q_i / P_i) \ge 0$ for all $i$.
  2. **Asymmetric Regularization Flaw:** Because `EPS` is added only to `probs_new` inside `np.log`, the formula is not strictly symmetric:
     - When $P_i \approx 0$ and $Q_i = 0.2$: $(10^{-9} - 0.2) \cdot \ln(10^{-9} / 0.2) \approx (-0.2) \cdot (-19.1) = +3.82$.
     - When $P_i = 0.2$ and $Q_i \approx 0$: $(0.2 - 10^{-9}) \cdot \ln(0.2 / 2 \cdot 10^{-9}) \approx (0.2) \cdot (18.4) = +3.68$.
     The asymmetry is slight ($\sim 3.8\%$), but mathematically inconsistent with the formal symmetric definition.

---

### 4.4 Jensen-Shannon (JS) Divergence
- **Mathematical Formula:**  
  $$M = \frac{1}{2}(P + Q)$$
  $$\text{JS}(P \parallel Q) = \frac{1}{2} D_{\text{KL}}(P \parallel M) + \frac{1}{2} D_{\text{KL}}(Q \parallel M)$$
- **Implementation in `generate_meta.py:754–761`:**
  ```python
  M = 0.5 * (probs_base + probs_new)
  JS_div = float(0.5 * (
      np.sum(probs_base * np.log(probs_base / (M + EPS))) +
      np.sum(probs_new * np.log(probs_new / (M + EPS)))
  ))
  ```
- **Validation Findings:**
  - **Boundedness:** Unlike KL divergence, JS divergence is strictly bounded:
    $$0 \le \text{JS}(P \parallel Q) \le \ln(2) \approx 0.693147$$
  - When distributions are disjoint ($P \cap Q = \emptyset$), $\text{JS} = \ln(2) \approx 0.693147$.
  - Cleanalytix correctly computes JS divergence.

---

## 5. Scoring Transformations & Multi-Level Aggregation

### 5.1 Score Transformation Models

1. **Linear Decay (`_linear_score`):**
   $$S_{\text{linear}}(v, T) = \max\left(0.0, \; \min\left(1.0, \; 1.0 - \frac{v}{T}\right)\right)$$
   - When $v = 0$: $S = 1.0$ (perfect).
   - When $v = T$: $S = 0.0$ (threshold reached).
   - When $v > T$: $S = 0.0$ (clipped).
   - *Limitation:* Discontinuous derivative at $v = T$; provides no discrimination between a column at $20\%$ missingness vs $100\%$ missingness when threshold is $20\%$.

2. **Exponential Decay (`_exp_score`):**
   $$S_{\text{exp}}(v, T, k) = \exp\left(-k \cdot \frac{v}{T}\right)$$
   - When $v = 0$: $S = 1.0$.
   - When $v = T$: $S = \exp(-k)$ ($0.1353$ for $k=2$; $0.0498$ for $k=3$).
   - As $v \to \infty$: $S \to 0.0$ asymptotically.
   - *Advantage:* Differentiable, strictly monotonic, preserves severity ordering above threshold.

---

### 5.2 Aggregation Architecture & Level Inconsistency

The dataset DQ Score is computed in `compute_dq_score.py:408–423`:

$$\text{Column Score: } S_i = \sum_{j=1}^M w_j \cdot s_{i, j}$$
$$\text{Dataset DQ Score: } \text{DQ\_Score} = \frac{100}{N_{\text{cols}}} \sum_{i=1}^{N_{\text{cols}}} S_i$$

#### Aggregation Flaws Identified:

1. **Dataset-Level Metric Repeated at Column Level:**  
   `duplicate_rows_percent` is a global dataset property. In Cleanalytix, this value is duplicated across all $N$ columns and scored repeatedly in each column's score. While the mathematical average equals the dataset score, treating a dataset-level property as a column attribute conflates column quality with table structure.
2. **Missing Value Scoring Bias:**  
   When a metric is not applicable to a column (e.g. `whitespace_percent` on continuous numeric columns, or `outlier_percent` on categorical columns), `summarized_meta` contains `NaN`. In `get_new_values:291`:
   ```python
   if pd.isna(v):
       scores.append(1.0)
   ```
   Unapplicable metrics are awarded a **perfect score of 1.0**.
   - A numeric column always receives free 1.0 scores for `whitespace_percent` and `type_errors_percent`.
   - A categorical column always receives a free 1.0 score for `outlier_percent`.
   This systematically inflates data quality scores, making a severely defective column appear cleaner because irrelevant metrics grant unearned points.
3. **Double Counting of Drift:**  
   `psi_divergence` (threshold 0.25) and `js_divergence` (threshold 0.10) both measure distribution drift. With default weights ($1/8 = 12.5\%$ each), drift accounts for $25\%$ of the total score, giving drift twice the weight of missingness or outliers.
4. **Absence of Adaptive Weighting:**  
   No dynamic weight reallocation occurs based on column roles or applicability. If adaptive weighting were implemented, weights for non-applicable metrics (such as outliers on categorical columns) would be redistributed among valid metrics. Cleanalytix does not implement this.
