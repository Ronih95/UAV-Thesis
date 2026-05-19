# UAV Propeller Fault Detection — Machine Learning Pipeline

This project is a comprehensive machine learning pipeline developed to detect and classify propeller faults in the **Parrot Bebop 2** UAV (Unmanned Aerial Vehicle) using IMU (Inertial Measurement Unit) sensor data.

---

## Table of Contents

1. [Project Purpose](#project-purpose)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation and Dependencies](#installation-and-dependencies)
5. [Part 0 — Setup and Helper Functions](#part-0--setup-and-helper-functions)
6. [Part 1 — Time Series Analysis](#part-1--time-series-analysis)
7. [Part 2 — Signal Processing and Feature Extraction](#part-2--signal-processing-and-feature-extraction)
8. [Part 3 — Fault Classification](#part-3--fault-classification)
9. [Part 4 — Anomaly Detection](#part-4--anomaly-detection)
10. [Results and Findings](#results-and-findings)
11. [Generated Outputs](#generated-outputs)
12. [How to Run](#how-to-run)

---

## Project Purpose

Detecting mechanical faults in any propeller of a quadcopter (four-propeller UAV) **in real time** during flight is a critical safety requirement. This project:

- Visualizes fault signatures by analyzing raw accelerometer and gyroscope signals.
- Extracts features using signal processing techniques.
- Compares supervised learning (Random Forest, SVM) and unsupervised learning (Isolation Forest) methods.
- Classifies each propeller's condition — healthy / chipped / bent — with high accuracy.

---

## Dataset

### Source
**PADRE — Public UAV Measurement Dataset**

### Platform
Parrot Bebop 2 quadcopter

### Sampling Rate
500 Hz (500 measurements per second)

### Propeller Configuration

```
   C   A
    \ /
    / \
   D   B
```

Each propeller can independently be in one of 3 states:

| Code | State | Description |
|------|-------|-------------|
| `0` | Healthy | No fault |
| `1` | Chipped | Damage on propeller edge |
| `2` | Bent | Propeller tip is bent |

### Filename Encoding

The 4 digits in the filename represent the state of propellers `ABCD`.

Example: `Bebop2_16g_1kdps_normalized_1022.csv`
- Propeller A = `1` → Chipped
- Propeller B = `0` → Healthy
- Propeller C = `2` → Bent
- Propeller D = `2` → Bent

### Data Size

| Property | Value |
|----------|-------|
| Total CSV files | 20 |
| Rows per file | ~86,016 |
| Total rows | ~1,720,320 |
| Flight duration (per file) | ~172 seconds |
| Sensor channels | 24 (4 propellers × 6 axes) |

6 channels per propeller: 3-axis accelerometer (ax, ay, az) + 3-axis gyroscope (gx, gy, gz).

---

## Project Structure

```
Hakan_Hoca_Proje/
├── Untitled-1.ipynb                  ← Main analysis file (Jupyter Notebook)
├── README.md                      ← This file
├── UAV_measurement_data/
│   ├── README.md                     ← Dataset documentation
│   ├── Parrot_Bebop_2/
│   │   ├── Normalized_data/          ← 20 CSV files used in analysis
│   │   ├── Raw_data/
│   │   ├── FFT_data/
│   │   └── Range_data/
│   └── 3DR_Solo/                     ← Alternative platform (not analyzed)
├── fig_1_2_accel_timeseries.png
├── fig_1_3_gyro_timeseries.png
├── fig_1_4_accel_fault_overlay.png
├── fig_1_5_gyro_fault_overlay.png
├── fig_1_6_rolling_rms.png
├── fig_2_1_bandpass.png
├── fig_2_3_feature_correlation.png
├── fig_2_4_psd.png
├── fig_3_2_rf_confusion.png
├── fig_3_3_classifier_comparison.png
├── fig_3_4_feature_importance.png
├── fig_4_1_isolation_forest.png
├── fig_4_2_zscore.png
├── fig_4_3_reconstruction_error.png
└── fig_4_4_roc_curves.png
```

---

## Installation and Dependencies

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

| Library | Purpose |
|---------|---------|
| `numpy` | Numerical arrays and mathematical operations |
| `pandas` | CSV reading, DataFrame manipulation |
| `scipy.signal` | Butterworth filter, zero-phase filtering, Welch PSD |
| `scipy.stats` | Skewness, kurtosis, Spearman correlation |
| `sklearn.ensemble` | Random Forest, Isolation Forest models |
| `sklearn.svm` | Support Vector Machine (SVM) model |
| `sklearn.preprocessing` | Feature normalization with StandardScaler |
| `sklearn.model_selection` | Train/test data splitting |
| `sklearn.metrics` | Confusion matrix, F1, ROC, AUC computation |
| `matplotlib` | Base plotting library |
| `seaborn` | Statistical visualizations (heatmaps, box plots) |

---

## Part 0 — Setup and Helper Functions

### Constants

```python
SAMPLE_RATE  = 500       # Hz — sensor sampling rate
WINDOW_SIZE  = 500       # samples — each window equals 1 second
PROPELLERS   = ['A','B','C','D']
FAULT_MAP    = {0: 'Healthy', 1: 'Chipped', 2: 'Bent'}
```

**Why a 500-sample window?**
At a 500 Hz sampling rate, 500 samples correspond to exactly **1 second**. This is sufficient to capture the typical frequency components of propeller fault signals (10–200 Hz range), while remaining an acceptable latency for real-time applications.

### Helper Functions

```python
accel_cols(prop)   # Returns accelerometer column names for a propeller (ax, ay, az)
gyro_cols(prop)    # Returns gyroscope column names for a propeller (gx, gy, gz)
imu_cols(prop)     # Union of the above — all 6 channels

load_file(path)    # Loads a CSV file, adds a time column, assigns fault labels
```

**Why is `load_file` so important?**
The 4-digit code (`ABCD`) in the filename is parsed via regex, generating independent labels (`fault_A`, `fault_B`, `fault_C`, `fault_D`) for each propeller. This gives 4 separate classification targets from a single flight file.

---

## Part 1 — Time Series Analysis

The goal of this section is to visually inspect the raw sensor signals and build an intuitive understanding of how faults manifest in the signal.

### §1.1 — Dataset Overview

A table is created showing the fault state of each propeller across all 20 files. Basic statistics (mean, std, min, max) are computed for healthy-flight sensors.

**Why?** To understand data balance and which fault combinations are present.

### §1.2 — Accelerometer Time Series (Healthy Flight)

**Output:** `fig_1_2_accel_timeseries.png`

4 propellers × 3 axes = 12 subplots. The raw acceleration for the first 5 seconds is shown along with a 50-sample rolling mean overlay.

**Why?** To establish a reference profile for the healthy condition. The rolling mean suppresses sudden noise and reveals the general trend.

### §1.3 — Gyroscope Time Series (Healthy Flight)

**Output:** `fig_1_3_gyro_timeseries.png`

Same layout as §1.2, applied to angular velocity data.

**Why?** Acceleration and angular velocity contain different physical information. A propeller fault affects both sensors; they are therefore examined separately.

### §1.4 — Fault Overlay — Accelerometer

**Output:** `fig_1_4_accel_fault_overlay.png`

Using single-fault files, **Healthy vs Chipped vs Bent** signals are overlaid.

**Why?** To directly visualize what kinds of changes a fault introduces to the signal. Different fault types produce different vibration amplitudes and frequency content.

### §1.5 — Fault Overlay — Gyroscope

**Output:** `fig_1_5_gyro_fault_overlay.png`

Same logic as §1.4, applied to gyroscope signals.

### §1.6 — Rolling RMS — Vibration Intensity

**Output:** `fig_1_6_rolling_rms.png`

The change in acceleration magnitude (RMS) over the full ~172-second flight. Window: 500 samples, stride: 50 samples.

**Why?** RMS (Root Mean Square) is the most natural measure of vibration energy. If a propeller fault changes the energy level, it becomes visible in this plot. Energy profiles across different fault conditions are compared over time.

---

## Part 2 — Signal Processing and Feature Extraction

Numerical features that machine learning algorithms can process are derived from the raw sensor signals.

### §2.1 — Butterworth Bandpass Filter

**Output:** `fig_2_1_bandpass.png`

```python
nyq = SAMPLE_RATE / 2          # Nyquist: 250 Hz
b, a = butter(4, [5/nyq, 200/nyq], btype='band')
filtered = filtfilt(b, a, signal)
```

**Why Butterworth?**
- 4th-order filter → sufficient sharpness, computationally efficient
- Band: **5–200 Hz** → DC drift (0–5 Hz) and high-frequency electronic noise (>200 Hz) are removed; mechanical vibrations representing propeller faults are preserved
- `filtfilt` → **zero-phase filtering** (the signal does not shift forward or backward in time)

### §2.2 — Windowed Feature Extraction

Each file is divided into non-overlapping 500-sample windows:

- Total number of windows: **3,440**
- Features per window: **240** (4 propellers × 6 channels × 10 statistics)

**10 statistical features computed per window:**

| Feature | Formula / Description |
|---------|----------------------|
| `mean` | Average value |
| `std` | Standard deviation |
| `min` | Minimum value |
| `max` | Maximum value |
| `ptp` | Peak-to-peak amplitude (max − min) |
| `rms` | Root mean square — vibration energy |
| `skew` | Skewness — asymmetry of the distribution |
| `kurt` | Kurtosis — tail heaviness (impulsive faults produce high kurtosis) |
| `energy` | Sum of squares / length — normalized energy |
| `zcr` | Zero-crossing rate — indicator of frequency content |

**Why these features?**
Together, these 10 statistics cover the amplitude, energy, frequency, and shape characteristics of the signal. Using these features instead of the raw signal:
1. Dramatically reduces dimensionality (86,016 → 10 numbers per channel)
2. Focuses models on periodic statistics rather than temporal patterns
3. Produces a lightweight structure suitable for real-time applications

### §2.3 — Feature–Fault Correlation

**Output:** `fig_2_3_feature_correlation.png`

**Spearman rank correlation** between each feature and the fault label is computed; the 20 features with the highest absolute correlation are visualized.

**Why Spearman?** Pearson correlation measures linear relationships. The fault–feature relationship may be monotonic but non-linear; Spearman captures such relationships as well.

### §2.4 — Power Spectral Density (PSD)

**Output:** `fig_2_4_psd.png`

```python
freqs, psd = welch(signal, fs=500, nperseg=1024)
```

Frequency content between 0–250 Hz is examined using Welch's method. PSDs for healthy, chipped, and bent conditions are compared.

**Why Welch?** Welch's method divides the signal into overlapping segments, computes the periodogram of each, and averages them. This provides a **more noise-robust** frequency estimate compared to plain FFT.

---

## Part 3 — Fault Classification

One of 4 propeller labels (0/1/2) is predicted for each window.

### Data Preparation

- **Train/test split:** 80% / 20%, stratified
- **Feature scaling:** `StandardScaler` fitted on the training set is also applied to the test set
- **Input:** All 240 features (all propellers)
- **Target:** Independent label for each propeller

**Why use all 240 features?**
Only Propeller D's own 60 features are insufficient for classification. Data from other propellers provides context: the model can compare D's readings against A/B/C to distinguish D-specific fault signals from cross-talk noise.

### §3.2 — Random Forest Classifier

**Output:** `fig_3_2_rf_confusion.png`

```python
RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    class_weight='balanced', # Automatically balances imbalanced classes
    random_state=42
)
```

**Why `class_weight='balanced'`?**
Some fault combinations may have few examples in the dataset. The `balanced` option automatically adjusts each class's weight, preventing the model from ignoring rare classes.

**Results:**

| Propeller | Accuracy |
|-----------|----------|
| A | 99% |
| B | 99% |
| C | 99% |
| D | 96% |

Propeller D's lower accuracy stems from the small vibration change produced by the bent fault, which overlaps with cross-talk noise from other propellers.

### §3.3 — SVM Comparison

**Output:** `fig_3_3_classifier_comparison.png`

```python
SVC(kernel='rbf', C=10, class_weight='balanced', random_state=42)
```

Accuracy and Weighted F1 scores for Random Forest and SVM are compared in a bar chart.

**Why RBF kernel?** The feature space is non-linear; the RBF (Radial Basis Function) kernel can learn non-linear decision boundaries.

**Why C=10?** Medium-high regularization; prevents overfitting while maintaining boundary flexibility.

### §3.4 — Feature Importance Analysis

**Output:** `fig_3_4_feature_importance.png`

The top 20 features most used by Random Forest for Propeller D are ranked by mean decrease in impurity. Color coding shows which propeller each feature belongs to.

**Why important?** Which sensors and statistics does the model rely on? This plot provides interpretability and shows which physical quantities are linked to faults.

### §3.5 — Classification Reports

Precision, recall, F1 score, and support values are printed for each propeller.

---

## Part 4 — Anomaly Detection

Going beyond supervised learning: methods that detect abnormal sensor readings without label information are examined.

### §4.1 — Isolation Forest (Unsupervised)

**Output:** `fig_4_1_isolation_forest.png`

```python
IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
```

**How does it work?**
Isolation Forest tries to isolate data points by randomly partitioning them. Abnormal points are "rare" or "outliers" and can be isolated with fewer partitions → shorter tree path → higher anomaly score.

**Training:** Only healthy windows are used.
**Testing:** All windows are scored; windows exceeding the 90th percentile threshold are flagged as anomalies.

**Why unsupervised?** In the real world, labeled data for every fault type may not be available. Isolation Forest requires no labels; it only detects "deviation from normal."

### §4.2 — Z-Score Anomaly Detection

**Output:** `fig_4_2_zscore.png`

```python
z = (value - rolling_mean) / rolling_std
# Window: 200 samples | Threshold: |z| > 3
```

**Why rolling Z-score?** Using a global mean and standard deviation could misclassify changing conditions throughout a flight (acceleration, deceleration) as anomalies. A local rolling Z-score solves this problem.

**Threshold |z| > 3:** Under the Gaussian distribution assumption, 99.7% of values fall within this bound. Values exceeding it are very likely genuine anomaly signals.

### §4.3 — Reconstruction Error

**Output:** `fig_4_3_reconstruction_error.png`

Deviation from the healthy baseline is computed for each window:

```
error = mean(|window − healthy_mean| / healthy_std)
```

**Left plot:** Mean error per file, colored by number of faulty propellers.
**Right plot:** Error distribution box plot by fault type.

**Key finding:** Error increases monotonically with the number of faulty propellers. Multiple faults produce higher error than a single fault.

### §4.4 — ROC Curves — Detection Performance

**Output:** `fig_4_4_roc_curves.png`

**Task:** Binary classification (Healthy vs Any Fault)
**Classifier:** Isolation Forest anomaly scores
**Metric:** AUC (Area Under the Curve)

4 subplots (one per propeller) draw the True Positive Rate vs False Positive Rate curve. The diagonal line (AUC=0.5) represents random guessing; the higher the model's curve, the better the performance.

---

## Results and Findings

### Supervised Learning

| Method | Propeller A | Propeller B | Propeller C | Propeller D |
|--------|-------------|-------------|-------------|-------------|
| Random Forest | 99% | 99% | 99% | 96% |
| SVM (RBF) | Below RF | Below RF | Below RF | Below RF |

**Random Forest** outperformed SVM on this dataset.

### The Propeller D Challenge

Propeller D's bent fault produces a small acceleration change (22% increase in mean) that overlaps with neighboring propeller noise. Cohen's d = 0.55 → 62% of bent windows fall within the healthy range. Solution: Using all 240 features provides cross-propeller context.

### Most Discriminative Features

- **RMS** and **energy** metrics show the highest correlation with fault status.
- **Skewness** and **kurtosis** capture the non-Gaussian character of impulsive fault signals.
- **Zero-crossing rate** provides indirect information about frequency content.

### Frequency Domain

Distinct PSD signatures for different fault types are found in the 50, 100, 150, and 200 Hz bands.

### Unsupervised Detection

- Isolation Forest achieves high AUC on the healthy vs any-fault task.
- Z-score method is effective for real-time spike detection.
- Reconstruction error increases consistently with the number of faulty propellers.

---

## Generated Outputs

| File | Content |
|------|---------|
| `fig_1_2_accel_timeseries.png` | Raw accelerometer signals — healthy flight |
| `fig_1_3_gyro_timeseries.png` | Raw gyroscope signals — healthy flight |
| `fig_1_4_accel_fault_overlay.png` | Acceleration comparison: Healthy vs Chipped vs Bent |
| `fig_1_5_gyro_fault_overlay.png` | Angular velocity comparison: three fault conditions |
| `fig_1_6_rolling_rms.png` | Vibration energy change over 172-second flight |
| `fig_2_1_bandpass.png` | Butterworth bandpass filter demonstration |
| `fig_2_3_feature_correlation.png` | Feature–fault Spearman correlation heatmap |
| `fig_2_4_psd.png` | Power spectral density comparison |
| `fig_3_2_rf_confusion.png` | Random Forest confusion matrix (4 propellers) |
| `fig_3_3_classifier_comparison.png` | RF vs SVM accuracy/F1 bar chart |
| `fig_3_4_feature_importance.png` | Top 20 feature importances for Propeller D |
| `fig_4_1_isolation_forest.png` | Isolation Forest anomaly scores |
| `fig_4_2_zscore.png` | Rolling Z-score spike detection |
| `fig_4_3_reconstruction_error.png` | Reconstruction error — by file and fault type |
| `fig_4_4_roc_curves.png` | ROC curves (4 propellers) |

---

## How to Run

### Requirements

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Steps

```bash
# 1. Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter

# 2. Start the notebook
jupyter notebook

# 3. Open Untitled-1.ipynb and run all cells in order
#    Kernel → Restart & Run All
```

### Data Path

The notebook expects data at the following relative path:
```
./UAV_measurement_data/Parrot_Bebop_2/Normalized_data/*.csv
```

### Reproducibility

All random operations are fixed with `random_state=42`. The same results are obtained with the same data.

---

## Data Flow Summary

```
Raw CSV Files
      ↓  load_file() — label assignment via regex
DataFrame (24 sensor channels + labels)
      ↓  Butterworth bandpass (5–200 Hz)
Filtered Signal
      ↓  500-sample windows
Window Array (3,440 windows)
      ↓  extract_window_features()
Feature Matrix (3,440 × 240)
      ↓
  ┌───┴──────┬──────────────┬──────────────┐
  ↓          ↓              ↓              ↓
Visual    Correlation    RF / SVM      Anomaly
Analysis  + PSD          Models        Detection
(§1)      (§2)           (§3)          (§4)
```
