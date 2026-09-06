# ✈️ Aircraft Engine Remaining Useful Life Prediction

A machine learning project for **Remaining Useful Life (RUL) estimation** of aircraft turbofan engines using the **NASA C-MAPSS Turbofan Engine Degradation Dataset**.

The main goal is to estimate how many operational cycles remain before an engine reaches failure, using sensor measurements and operating-condition data.

---

## 📌 Project Overview

Predictive maintenance plays an important role in aviation by allowing maintenance activities to be scheduled based on the predicted health condition of an engine rather than relying only on fixed maintenance intervals.

In this project, machine learning models are developed to learn degradation patterns from historical turbofan engine trajectories and estimate their **Remaining Useful Life (RUL)**.

The workflow includes:

```text
Raw C-MAPSS Data
       │
       ▼
Data Cleaning & EDA
       │
       ▼
RUL Label Generation
       │
       ▼
Sensor Selection
       │
       ▼
Feature Engineering
       │
       ├── Rolling Mean
       ├── Rolling Standard Deviation
       ├── Rolling Slope
       ├── Sensor Difference
       ├── Deviation from Healthy Baseline
       └── Engine Cycle Information
       │
       ▼
Engine-Level Cross Validation
       │
       ▼
Hyperparameter Optimization
       │
       ├── Random Forest
       ├── Extra Trees
       └── Histogram Gradient Boosting
       │
       ▼
RUL Prediction
       │
       ▼
Model Evaluation & Feature Importance
```

---

## 🗂️ Dataset

The project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan engine degradation dataset.

The dataset contains simulated degradation trajectories for multiple aircraft engines.

### Files

| File            | Description                                                                   |
| --------------- | ----------------------------------------------------------------------------- |
| `Training.txt`  | Complete degradation trajectories of training engines                         |
| `Testing.txt`   | Truncated trajectories of test engines                                        |
| `Safe-life.txt` | True RUL values corresponding to the final observed cycle of each test engine |

The dataset contains:

* Engine ID
* Cycle number
* 3 operating settings
* 21 sensor measurements

In total, each observation contains **26 columns**.

---

## 🎯 Objective

The primary objective is to develop a predictive maintenance model capable of estimating:

> **How many operating cycles remain before engine failure?**

For the training data, RUL is obtained from the final operating cycle of each engine:

$$
RUL = Cycle_{max} - Cycle_{current}
$$

The model then learns the relationship between engine condition and remaining lifetime.

---

# 🔍 Exploratory Data Analysis

The dataset is first investigated to understand its structure and degradation characteristics.

### Missing Values

The dataset does not contain missing sensor measurements, so no imputation strategy is required.

### Sensor Variability

Near-zero variance sensors are removed because they provide little useful information for predicting degradation.

The following sensors were identified as low-variance in the analyzed dataset:

```text
s1
s5
s6
s10
s16
s18
s19
```

### Degradation Trends

Several sensors exhibit systematic trends as the engine approaches failure.

Examples include:

```text
Increasing:
s2, s3, s4, s8, s11, s13

Decreasing:
s9, s14, s15
```

These trends provide useful degradation signatures for RUL estimation.

### Important Sensors

Correlation analysis identified several sensors with strong relationships to RUL, particularly:

```text
s11
s12
s15
s9
s14
```

---

# ⚙️ Feature Engineering

A major part of the project focuses on transforming raw sensor signals into features that better represent engine health and degradation.

## Rolling Statistics

For each active sensor, rolling statistics are calculated using multiple time windows:

```text
10 cycles
20 cycles
30 cycles
```

The following quantities are extracted:

* Rolling mean
* Rolling standard deviation
* Rolling slope

---

## 📈 Degradation Slope

The slope of a sensor signal is particularly useful because degradation is not represented only by the sensor's current value, but also by **how rapidly it is changing**.

For example:

```text
Current sensor value
        +
Rate of change
        ↓
Degradation indication
```

This is especially relevant for sensors such as `s11`, which show an increasing trend before failure.

---

## 🧠 Healthy Baseline Deviation

For each engine, an initial healthy operating condition is estimated from its early trajectory.

The current sensor state is then compared with that baseline:

$$
Deviation = Sensor_{current} - Sensor_{healthy}
$$

This provides an additional representation of the engine's degradation state.

---

## 🔄 First Difference Features

First differences are also calculated:

$$
\Delta x_t = x_t - x_{t-1}
$$

These features capture short-term changes in sensor measurements.

---

# 🧪 Data Leakage Prevention

A major consideration in this project is avoiding leakage between engines.

Instead of randomly splitting individual rows, the training data is divided using **engine-level GroupKFold cross-validation**.

This ensures that observations belonging to the same engine remain in the same fold.

```text
Engine 1 ───────► Training

Engine 2 ───────► Training

Engine 3 ───────► Validation
```

rather than:

```text
Engine 3 - Cycle 1 ──► Training
Engine 3 - Cycle 2 ──► Validation
```

This provides a more realistic estimate of how the model performs on completely unseen engines.

---

# 🤖 Machine Learning Models

The project evaluates several nonlinear machine learning approaches.

## 🌲 Random Forest

Random Forest is used to model nonlinear relationships and interactions between engine sensors and degradation indicators.

Key hyperparameters are optimized using **Optuna**.

---

## 🌳 Extra Trees

Extra Trees is included as a highly randomized tree ensemble and provides an additional nonlinear model for comparison.

---

## ⚡ Histogram Gradient Boosting

Histogram Gradient Boosting is used as a boosting-based alternative to the tree ensemble models.

Its hyperparameters are also optimized with Optuna.

---

# 🔧 Hyperparameter Optimization

The project uses **Optuna** for automated hyperparameter optimization.

The optimization process searches for parameters that minimize the validation RMSE.

Examples include:

```text
n_estimators
max_depth
min_samples_leaf
max_features
learning_rate
max_iter
max_leaf_nodes
regularization
```

The optimization process is evaluated using **GroupKFold cross-validation** rather than ordinary random K-fold splitting.

---

# 📊 Evaluation Metrics

Three primary metrics are used.

### MAE

Mean Absolute Error:

$$
MAE = \frac{1}{n}\sum |y-\hat{y}|
$$

Measures the average prediction error in engine cycles.

### RMSE

Root Mean Squared Error:

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum
(y-\hat{y})^2
}
$$

RMSE penalizes large prediction errors more strongly.

### R²

Coefficient of determination:

$$
R^2 =
1-
\frac{
\sum(y-\hat{y})^2
}{
\sum(y-\bar{y})^2
}
$$

A value close to 1 indicates strong agreement between predictions and actual RUL values.

---

# 🔬 Feature Importance

To understand which variables contribute most strongly to the predictions, **permutation importance** is calculated after model evaluation.

This allows the project to investigate whether the model relies mainly on:

```text
Raw sensor values
       ↓
Rolling statistics
       ↓
Degradation slopes
       ↓
Healthy-baseline deviations
       ↓
Engine age
```

rather than treating the model as a complete black box.

The project also aggregates feature importance into higher-level groups such as:

* Raw Sensors
* Rolling Mean
* Rolling Standard Deviation
* Rolling Slope
* Baseline Deviation
* Normalized Deviation
* First Difference
* Engine Age
* Operational Settings

---

# 🚨 Failure Onset Detection

In addition to RUL prediction, the project investigates early degradation detection.

Sensor `s11` is used as a representative health indicator. A rolling mean is compared against an initial baseline to identify significant deviations.

The analysis of representative engines indicates that detectable degradation appears relatively late in the engine lifetime, with the investigated examples showing onset at approximately **65–80% of total operating life**.

This provides an additional condition-monitoring signal that can complement RUL prediction.

---

# 🛠️ Predictive Maintenance Strategy

The predicted RUL can be translated into maintenance decisions.

### 🟡 Warning Level

```text
RUL < 50 cycles
```

Possible actions:

* Schedule maintenance
* Prepare spare parts
* Allocate maintenance resources

### 🔴 Critical Level

```text
RUL < 20 cycles
```

Possible actions:

* Remove engine from service
* Perform detailed inspection
* Carry out required maintenance

These thresholds are used as a practical engineering interpretation of the RUL prediction pipeline.

---

# 💻 Technologies

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Main programming language      |
| NumPy        | Numerical computation          |
| Pandas       | Data manipulation              |
| Matplotlib   | Visualization                  |
| Seaborn      | Statistical visualization      |
| Scikit-learn | Machine learning               |
| Optuna       | Hyperparameter optimization    |
| TensorFlow   | Neural-network experimentation |

The project was developed in a Jupyter Notebook environment.

---

# 📁 Project Structure

```text
.
├── Training.txt
├── Testing.txt
├── Safe-life.txt
├── Diren_Gürgül_Project_fixed.ipynb
└── README.md
```

---

# 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn optuna tensorflow jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Diren_Gürgül_Project_fixed.ipynb
```

and run the cells sequentially.

---

# 📌 Key Engineering Takeaways

This project demonstrates that RUL prediction benefits from combining raw sensor measurements with temporal degradation information.

The most important concepts explored are:

```text
Sensor state
    +
Sensor trend
    +
Sensor deviation from healthy condition
    +
Engine age
    ↓
Engine health estimation
    ↓
Remaining Useful Life prediction
```

The project also highlights the importance of **engine-level validation**, since randomly splitting individual time-series observations can give overly optimistic estimates of generalization.

---

# 🔮 Future Improvements

Possible extensions include:

* LSTM / GRU sequence models
* Temporal Convolutional Networks
* Transformer-based RUL prediction
* Sensor selection using model-based importance
* Uncertainty-aware RUL prediction
* Physics-informed machine learning
* Hybrid physics + machine learning degradation models
* Remaining-life confidence intervals
* Real-time predictive maintenance dashboard

---

# 👨‍💻 Author

**Diren Gürgül**

Mechanical Engineering Student
Interested in:

```text
Machine Learning
Computational Fluid Dynamics
Turbomachinery
Compressible Flows
Predictive Maintenance
Data-Driven Engineering
```

---

## ⭐ Project Goal

> **Turning engine sensor data into actionable maintenance decisions before failure occurs.**

---

