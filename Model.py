# ============================================================
# C-MAPSS RUL PREDICTION
# ============================================================


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import optuna

RANDOM_STATE = 42

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
# Summary: global constants that control the RUL cap, the
# rolling-window sizes used for feature engineering, the
# baseline window used to define "healthy" sensor behavior,
# and the max RUL value kept in the training set.

RUL_CAP = 125

ROLLING_WINDOWS = [10, 20, 30]

BASELINE_WINDOW = 30

# Only train on degradation region
# instead of giving the model thousands of RUL=125 samples
TRAIN_MAX_RUL = 125


# ============================================================
# 1. LOAD DATA
# ============================================================
# Summary: reads the raw training set, test set, and the
# "safe-life" file (true RUL values for the test engines) from
# whitespace-separated text files into pandas DataFrames.

COLS = [
    "machine",
    "cycle",
    "setting1",
    "setting2",
    "setting3",
    *[f"s{i}" for i in range(1, 22)]
]

train = pd.read_csv(
    "Training.txt",
    sep=r"\s+",
    header=None,
    names=COLS
)

test = pd.read_csv(
    "Testing.txt",
    sep=r"\s+",
    header=None,
    names=COLS
)

safe = pd.read_csv(
    "Safe-life.txt",
    sep=r"\s+",
    header=None,
    names=["rul"]
)

safe["machine"] = np.arange(1, 101)

print("Train:", train.shape)
print("Test :", test.shape)


# ============================================================
# 2. GENERATE TRAINING RUL
# ============================================================
# Summary: for each engine, the max observed cycle is treated
# as the failure point. RUL at every row = (max_cycle - cycle).
# The raw RUL is then clipped at RUL_CAP (piecewise-linear RUL
# target, a standard trick for C-MAPSS: early life is assumed
# "healthy" so RUL doesn't need to grow without bound).

max_cycles = (
    train.groupby("machine")["cycle"]
    .max()
    .rename("max_cycle")
)

train = train.join(
    max_cycles,
    on="machine"
)

train["rul_raw"] = (
    train["max_cycle"] - train["cycle"]
)

train["rul"] = np.minimum(
    train["rul_raw"],
    RUL_CAP
)

train.drop(
    columns=["max_cycle"],
    inplace=True
)

print("\nRaw RUL:")
print(train["rul_raw"].describe())

print("\nClipped RUL:")
print(train["rul"].describe())


# ============================================================
# 3. REMOVE CONSTANT / LOW-VARIANCE SENSORS
# ============================================================
# Summary: sensors with almost no variance carry no useful
# signal (and can break scaling steps), so they are dropped.
# Everything else becomes the "active_sensors" list used
# throughout feature engineering.

sensor_cols = [
    f"s{i}" for i in range(1, 22)
]

sensor_std = (
    train[sensor_cols]
    .std()
    .sort_values()
)

low_var_sensors = sensor_std[
    sensor_std < 0.01
].index.tolist()

active_sensors = [
    s
    for s in sensor_cols
    if s not in low_var_sensors
]

print("\nRemoved sensors:")
print(low_var_sensors)

print("\nActive sensors:")
print(active_sensors)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
# Summary: this is the core feature-building block. For every
# active sensor it computes:
#   - a per-engine "healthy baseline" (mean of first N cycles)
#   - raw deviation from baseline
#   - z-scored deviation from baseline (normalized by baseline std)
#   - first difference (cycle-to-cycle change)
#   - rolling mean / std / slope over several window sizes
# It also builds two global health indicators (average absolute
# deviation, average absolute degradation slope) and simple
# age-related features (cycle, cycle squared).

def rolling_slope(x):

    n = len(x)

    if n < 2:
        return 0.0

    xx = np.arange(n)

    x_mean = x.mean()

    denominator = np.sum(
        (xx - xx.mean()) ** 2
    )

    if denominator == 0:
        return 0.0

    numerator = np.sum(
        (xx - xx.mean())
        * (x - x_mean)
    )

    return numerator / denominator


def create_features(df):

    df = df.copy()

    df = (
        df.sort_values(
            ["machine", "cycle"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Initial healthy baseline
    # --------------------------------------------------------
    # Summary: average sensor readings over the first
    # BASELINE_WINDOW cycles of each engine's life, used as a
    # reference point for "how much has this sensor drifted".

    baseline = (
        df.groupby("machine")[active_sensors]
        .apply(
            lambda x:
            x.iloc[:BASELINE_WINDOW].mean()
        )
    )

    baseline.index.name = "machine"

    baseline.columns = [
        f"{s}_baseline"
        for s in active_sensors
    ]

    df = df.merge(
        baseline,
        on="machine",
        how="left"
    )

    # --------------------------------------------------------
    # Normalized sensor deviation
    # --------------------------------------------------------
    # Summary: raw distance of the current reading from the
    # healthy baseline.

    for s in active_sensors:

        base = f"{s}_baseline"

        df[f"{s}_dev"] = (
            df[s] - df[base]
        )

    # --------------------------------------------------------
    # Absolute deviation
    # --------------------------------------------------------
    # Summary: same deviation as above, but scaled by the
    # baseline's own standard deviation, so different sensors
    # (different units/scales) become comparable (a z-score-like
    # feature).

    for s in active_sensors:

        base = f"{s}_baseline"

        scale = (
            df.groupby("machine")[s]
            .transform(
                lambda x:
                x.iloc[:BASELINE_WINDOW].std()
            )
        )

        scale = scale.replace(
            0,
            1e-6
        )

        df[f"{s}_zdev"] = (
            df[s] - df[base]
        ) / scale

    # --------------------------------------------------------
    # FIRST DIFFERENCE / TREND
    # --------------------------------------------------------
    # Summary: cycle-over-cycle change per engine, a simple
    # short-term trend indicator.

    for s in active_sensors:

        df[f"{s}_delta"] = (
            df.groupby("machine")[s]
            .diff()
            .fillna(0)
        )

    # --------------------------------------------------------
    # ROLLING FEATURES
    # --------------------------------------------------------
    # Summary: for each rolling window size (10/20/30 cycles),
    # compute the rolling mean, rolling std, and rolling slope
    # (linear trend) of each sensor. These capture both the
    # recent level and the recent trend of degradation.

    for s in active_sensors:

        grouped = df.groupby("machine")[s]

        for w in ROLLING_WINDOWS:

            # Rolling mean
            df[
                f"{s}_mean_{w}"
            ] = (
                grouped
                .transform(
                    lambda x:
                    x.rolling(
                        w,
                        min_periods=1
                    ).mean()
                )
            )

            # Rolling standard deviation
            df[
                f"{s}_std_{w}"
            ] = (
                grouped
                .transform(
                    lambda x:
                    x.rolling(
                        w,
                        min_periods=2
                    ).std()
                )
                .fillna(0)
            )

            # Rolling slope
            df[
                f"{s}_slope_{w}"
            ] = (
                grouped
                .transform(
                    lambda x:
                    x.rolling(
                        w,
                        min_periods=2
                    )
                    .apply(
                        rolling_slope,
                        raw=False
                    )
                )
                .fillna(0)
            )

    # --------------------------------------------------------
    # GLOBAL HEALTH SCORE
    # --------------------------------------------------------
    # Summary: two single-number summaries of overall engine
    # health — average absolute z-deviation across all sensors,
    # and average absolute 30-cycle slope across all sensors.

    zdev_cols = [
        f"{s}_zdev"
        for s in active_sensors
    ]

    slope30_cols = [
        f"{s}_slope_30"
        for s in active_sensors
    ]

    df["health_deviation"] = (
        df[zdev_cols]
        .abs()
        .mean(axis=1)
    )

    df["mean_degradation_slope"] = (
        df[slope30_cols]
        .abs()
        .mean(axis=1)
    )

    # --------------------------------------------------------
    # AGE FEATURES
    # --------------------------------------------------------
    # Summary: absolute cycle number is safe to use directly
    # (it doesn't leak the engine's max/failure cycle). A
    # squared term is added to let the model capture non-linear
    # aging effects. Deliberately NOT using cycle / max_observed
    # cycle, since that ratio isn't available at prediction time
    # for the test set.

    df["cycle_sq"] = (
        df["cycle"] ** 2
    )

    # --------------------------------------------------------
    # Remove helper baseline columns
    # --------------------------------------------------------
    # Summary: the raw "_baseline" columns were only needed to
    # compute the deviation features above; they're dropped so
    # they don't get treated as model inputs.

    baseline_cols = [
        f"{s}_baseline"
        for s in active_sensors
    ]

    df.drop(
        columns=baseline_cols,
        inplace=True
    )

    return df


print("\nCreating features...")

train_fe = create_features(train)

test_fe = create_features(test)

print(
    "Train engineered:",
    train_fe.shape
)

print(
    "Test engineered:",
    test_fe.shape
)


# ============================================================
# 5. FEATURE LIST
# ============================================================
# Summary: assembles the final list of column names that will
# be fed into the models — age/settings/health features, raw
# sensors, and every engineered variant (dev, zdev, delta,
# rolling mean/std/slope) for each active sensor.

FEATURES = [
    "cycle",
    "cycle_sq",
    "setting1",
    "setting2",
    "health_deviation",
    "mean_degradation_slope"
]

# Raw sensors
FEATURES += active_sensors

# Deviation
FEATURES += [
    f"{s}_dev"
    for s in active_sensors
]

# Z-deviation
FEATURES += [
    f"{s}_zdev"
    for s in active_sensors
]

# First difference
FEATURES += [
    f"{s}_delta"
    for s in active_sensors
]

# Rolling features
for s in active_sensors:

    for w in ROLLING_WINDOWS:

        FEATURES.append(
            f"{s}_mean_{w}"
        )

        FEATURES.append(
            f"{s}_std_{w}"
        )

        FEATURES.append(
            f"{s}_slope_{w}"
        )


print(
    "\nNumber of features:",
    len(FEATURES)
)


# ============================================================
# 6. TRAINING DATA
# ============================================================
# Summary: filters the engineered training rows to only the
# degradation region (raw RUL <= TRAIN_MAX_RUL), so the model
# isn't dominated by thousands of "healthy, RUL=125" samples.
# Builds X_train, y_train, and the per-row engine id ("groups")
# needed for GroupKFold.

# IMPORTANT:
# Only train on RUL <= 125.
#
# This prevents the huge early-life region from dominating
# the regression problem.

train_model = train_fe[
    train_fe["rul_raw"] <= TRAIN_MAX_RUL
].copy()

print(
    "\nTraining rows before filtering:",
    len(train_fe)
)

print(
    "Training rows after filtering :",
    len(train_model)
)

print(
    "Removed early-life rows       :",
    len(train_fe) - len(train_model)
)


X_train = train_model[
    FEATURES
].values

y_train = train_model[
    "rul"
].values

groups = train_model[
    "machine"
].values


# ============================================================
# 7. TEST SET
# ============================================================
# Summary: for the test set, only the LAST recorded cycle per
# engine matters (that's the point at which RUL is predicted).
# y_test_raw comes from the safe-life file (ground truth);
# y_test_cap is the same value clipped at RUL_CAP for a second,
# more lenient evaluation.

test_last = (
    test_fe
    .sort_values(
        ["machine", "cycle"]
    )
    .groupby("machine")
    .tail(1)
    .sort_values("machine")
    .reset_index(drop=True)
)

X_test = test_last[
    FEATURES
].values

y_test_raw = (
    safe.sort_values("machine")
    ["rul"]
    .values
)

# A clipped evaluation is also reported
y_test_cap = np.minimum(
    y_test_raw,
    RUL_CAP
)

print(
    "\nTest RUL raw:"
)

print(
    pd.Series(y_test_raw)
    .describe()
)


# ============================================================
# 8. GROUP K-FOLD
# ============================================================
# Summary: 5-fold cross-validation grouped by engine id, so
# rows from the same engine never appear in both the train and
# validation split of a fold (prevents data leakage).

gkf = GroupKFold(
    n_splits=5
)


# ============================================================
# 9. EVALUATION FUNCTION
# ============================================================
# Summary: small helper that prints and returns MAE, RMSE and
# R² for a given set of predictions.

def evaluate(
    name,
    y_true,
    y_pred
):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    print(
        f"{name:<25}"
        f" MAE={mae:8.3f}"
        f" RMSE={rmse:8.3f}"
        f" R²={r2:8.4f}"
    )

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 10. BASELINE MODEL
# ============================================================
# Summary: a naive "always predict the training mean" model,
# used as a sanity-check reference point for the real models.

baseline_prediction = np.full(
    len(y_test_raw),
    np.mean(y_train)
)

print("\nBaseline:")
evaluate(
    "Mean baseline",
    y_test_raw,
    baseline_prediction
)


# ============================================================
# 11. RANDOM FOREST OPTUNA
# ============================================================
# Summary: Optuna hyperparameter search for RandomForest.
# Each trial runs 5-fold GroupKFold CV and returns the average
# validation RMSE, which Optuna tries to minimize over 20 trials.

print(
    "\n" + "=" * 60
)

print(
    "RANDOM FOREST"
)

print(
    "=" * 60
)


def rf_objective(trial):

    params = {

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                200,
                500,
                step=100
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                8,
                30
            ),

        "min_samples_leaf":
            trial.suggest_int(
                "min_samples_leaf",
                1,
                6
            ),

        "max_features":
            trial.suggest_categorical(
                "max_features",
                [
                    0.5,
                    0.8,
                    1.0,
                    "sqrt"
                ]
            )
    }

    scores = []

    for tr_idx, val_idx in gkf.split(
        X_train,
        y_train,
        groups
    ):

        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        model = RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_tr,
            y_tr
        )

        pred = model.predict(
            X_val
        )

        score = np.sqrt(
            mean_squared_error(
                y_val,
                pred
            )
        )

        scores.append(score)

    return np.mean(scores)


study_rf = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )
)

study_rf.optimize(
    rf_objective,
    n_trials=20
)

print(
    "\nBest RF:",
    study_rf.best_params
)

print(
    "CV RMSE:",
    study_rf.best_value
)


# ============================================================
# 12. FINAL RANDOM FOREST
# ============================================================
# Summary: refits RandomForest on the full training set using
# the best hyperparameters found above, predicts on the test
# set, and clips predictions to a physically valid RUL range.

rf = RandomForestRegressor(
    **study_rf.best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

pred_rf = rf.predict(
    X_test
)

# Keep prediction physically reasonable
pred_rf = np.clip(
    pred_rf,
    0,
    RUL_CAP
)


# ============================================================
# 13. EXTRA TREES
# ============================================================
# Summary: same Optuna tuning process as Random Forest, but
# for the ExtraTreesRegressor model.

print(
    "\n" + "=" * 60
)

print(
    "EXTRA TREES"
)

print(
    "=" * 60
)


def extra_objective(trial):

    params = {

        "n_estimators":
            trial.suggest_int(
                "n_estimators",
                200,
                500,
                step=100
            ),

        "max_depth":
            trial.suggest_int(
                "max_depth",
                8,
                35
            ),

        "min_samples_leaf":
            trial.suggest_int(
                "min_samples_leaf",
                1,
                6
            ),

        "max_features":
            trial.suggest_categorical(
                "max_features",
                [
                    0.5,
                    0.8,
                    1.0,
                    "sqrt"
                ]
            )
    }

    scores = []

    for tr_idx, val_idx in gkf.split(
        X_train,
        y_train,
        groups
    ):

        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        model = ExtraTreesRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        model.fit(
            X_tr,
            y_tr
        )

        pred = model.predict(
            X_val
        )

        score = np.sqrt(
            mean_squared_error(
                y_val,
                pred
            )
        )

        scores.append(score)

    return np.mean(scores)


study_extra = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )
)

study_extra.optimize(
    extra_objective,
    n_trials=20
)

print(
    "\nBest ExtraTrees:",
    study_extra.best_params
)

print(
    "CV RMSE:",
    study_extra.best_value
)


# ============================================================
# 14. FINAL EXTRA TREES
# ============================================================
# Summary: refits ExtraTrees on the full training set with the
# best found hyperparameters and predicts + clips on the test set.

extra = ExtraTreesRegressor(
    **study_extra.best_params,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

extra.fit(
    X_train,
    y_train
)

pred_extra = extra.predict(
    X_test
)

pred_extra = np.clip(
    pred_extra,
    0,
    RUL_CAP
)


# ============================================================
# 15. HISTOGRAM GRADIENT BOOSTING
# ============================================================
# Summary: same Optuna tuning process again, this time for
# HistGradientBoostingRegressor (tuning max_iter, learning_rate,
# max_leaf_nodes, l2_regularization).

print(
    "\n" + "=" * 60
)

print(
    "HISTOGRAM GRADIENT BOOSTING"
)

print(
    "=" * 60
)


def hgb_objective(trial):

    params = {

        "max_iter":
            trial.suggest_int(
                "max_iter",
                100,
                400,
                step=50
            ),

        "learning_rate":
            trial.suggest_float(
                "learning_rate",
                0.02,
                0.15,
                log=True
            ),

        "max_leaf_nodes":
            trial.suggest_int(
                "max_leaf_nodes",
                15,
                63
            ),

        "l2_regularization":
            trial.suggest_float(
                "l2_regularization",
                1e-4,
                10,
                log=True
            )
    }

    scores = []

    for tr_idx, val_idx in gkf.split(
        X_train,
        y_train,
        groups
    ):

        X_tr = X_train[tr_idx]
        X_val = X_train[val_idx]

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        model = HistGradientBoostingRegressor(
            **params,
            random_state=RANDOM_STATE
        )

        model.fit(
            X_tr,
            y_tr
        )

        pred = model.predict(
            X_val
        )

        score = np.sqrt(
            mean_squared_error(
                y_val,
                pred
            )
        )

        scores.append(score)

    return np.mean(scores)


study_hgb = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(
        seed=RANDOM_STATE
    )
)

study_hgb.optimize(
    hgb_objective,
    n_trials=20
)

print(
    "\nBest HGB:",
    study_hgb.best_params
)

print(
    "CV RMSE:",
    study_hgb.best_value
)


# ============================================================
# 16. FINAL HGB
# ============================================================
# Summary: refits HistGradientBoosting with the best found
# hyperparameters and predicts + clips on the test set.

hgb = HistGradientBoostingRegressor(
    **study_hgb.best_params,
    random_state=RANDOM_STATE
)

hgb.fit(
    X_train,
    y_train
)

pred_hgb = hgb.predict(
    X_test
)

pred_hgb = np.clip(
    pred_hgb,
    0,
    RUL_CAP
)


# ============================================================
# 17. FINAL COMPARISON
# ============================================================
# Summary: evaluates all three tuned models against the raw
# (unclipped) true RUL and ranks them by RMSE.

print(
    "\n" + "=" * 70
)

print(
    "FINAL MODEL COMPARISON"
)

print(
    "=" * 70
)

results_raw = []

results_raw.append(
    evaluate(
        "Random Forest",
        y_test_raw,
        pred_rf
    )
)

results_raw.append(
    evaluate(
        "Extra Trees",
        y_test_raw,
        pred_extra
    )
)

results_raw.append(
    evaluate(
        "HistGradientBoosting",
        y_test_raw,
        pred_hgb
    )
)

df_results = (
    pd.DataFrame(results_raw)
    .sort_values("RMSE")
    .reset_index(drop=True)
)

print(
    "\nRAW TEST RUL:"
)

print(
    df_results.to_string(
        index=False
    )
)


# ============================================================
# 18. CLIPPED TEST EVALUATION
# ============================================================
# Summary: same comparison, but against the clipped (capped at
# RUL_CAP) ground truth — a more lenient, standard-benchmark
# style evaluation.

print(
    "\n" + "=" * 70
)

print(
    "CLIPPED TEST RUL COMPARISON"
)

print(
    "=" * 70
)

evaluate(
    "Random Forest",
    y_test_cap,
    pred_rf
)

evaluate(
    "Extra Trees",
    y_test_cap,
    pred_extra
)

evaluate(
    "HistGradientBoosting",
    y_test_cap,
    pred_hgb
)


# ============================================================
# 19. SELECT BEST MODEL
# ============================================================
# Summary: picks whichever model had the lowest RMSE in the
# raw-RUL comparison and keeps a reference to its predictions.

best_name = (
    df_results.iloc[0]["model"]
)

prediction_map = {

    "Random Forest":
        pred_rf,

    "Extra Trees":
        pred_extra,

    "HistGradientBoosting":
        pred_hgb
}

best_pred = prediction_map[
    best_name
]

print(
    "\nBEST MODEL:",
    best_name
)


# ============================================================
# 20. TRUE VS PREDICTED
# ============================================================
# Summary: scatter plot of true RUL vs. predicted RUL for the
# best model, with a diagonal reference line for "perfect
# prediction".

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_test_raw,
    best_pred,
    alpha=0.75,
    edgecolors="k"
)

max_value = max(
    y_test_raw.max(),
    best_pred.max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    "r--",
    label="Perfect prediction"
)

plt.xlabel(
    "True RUL (cycles)"
)

plt.ylabel(
    "Predicted RUL (cycles)"
)

plt.title(
    f"{best_name}: True vs Predicted RUL"
)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 21. RESIDUAL PLOT
# ============================================================
# Summary: plots prediction error (predicted - true) against
# true RUL, to visually check for bias or heteroscedasticity.

residuals = (
    best_pred - y_test_raw
)

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    y_test_raw,
    residuals,
    alpha=0.75,
    edgecolors="k"
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel(
    "True RUL"
)

plt.ylabel(
    "Residual"
)

plt.title(
    f"{best_name}: Residual Plot"
)

plt.tight_layout()

plt.show()


# ============================================================
# 22. PREDICTION TABLE
# ============================================================
# Summary: builds a per-engine table with true RUL, predicted
# RUL, and absolute error, for quick manual inspection.

prediction_table = pd.DataFrame({

    "Engine":
        test_last["machine"],

    "True_RUL":
        y_test_raw,

    "Predicted_RUL":
        best_pred,

    "Absolute_Error":
        np.abs(
            best_pred - y_test_raw
        )
})

print(
    "\nPrediction examples:"
)

print(
    prediction_table.head(20)
)



# ============================================================
# FEATURE IMPORTANCE - PERMUTATION IMPORTANCE
# ============================================================
# Summary: computes permutation importance (how much RMSE
# degrades when a feature is shuffled) for the best model, on
# the test set.

from sklearn.inspection import permutation_importance

print("\nCalculating permutation importance...")

# ------------------------------------------------------------
# Determine the correct model and test data
# ------------------------------------------------------------

if best_name == "Random Forest":

    model_for_importance = rf
    X_importance = X_test

elif best_name == "Extra Trees":

    model_for_importance = extra
    X_importance = X_test

elif best_name == "HistGradientBoosting":

    model_for_importance = hgb
    X_importance = X_test

else:
    raise ValueError(
        f"Unknown model: {best_name}"
    )


# ------------------------------------------------------------
# Permutation importance
# ------------------------------------------------------------

perm = permutation_importance(
    model_for_importance,
    X_importance,
    y_test_raw,
    n_repeats=20,
    random_state=42,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
})

importance_df = (
    importance_df
    .sort_values(
        "importance_mean",
        ascending=False
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Print top 25
# ------------------------------------------------------------

print("\n" + "=" * 70)
print(f"TOP 25 FEATURES - {best_name}")
print("=" * 70)

print(
    importance_df.head(25).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Plot top 20
# ------------------------------------------------------------

top20 = (
    importance_df
    .head(20)
    .sort_values(
        "importance_mean"
    )
)

plt.figure(
    figsize=(10, 8)
)

plt.barh(
    top20["feature"],
    top20["importance_mean"],
    xerr=top20["importance_std"],
    capsize=3
)

plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    f"Top 20 Feature Importance - {best_name}"
)

plt.tight_layout()

plt.show()


# ============================================================
# GROUPED IMPORTANCE
# ============================================================
# Summary: buckets every feature into a human-readable category
# (Engine Age, Operating Settings, Rolling Mean, etc.) and sums
# the permutation importance within each category, to see which
# *type* of feature engineering mattered most overall.

def get_feature_group(feature):

    if feature in [
        "cycle",
        "cycle_sq"
    ]:
        return "Engine Age"

    if feature in [
        "setting1",
        "setting2"
    ]:
        return "Operating Settings"

    if feature in [
        "health_deviation",
        "mean_degradation_slope"
    ]:
        return "Global Health Indicators"

    if feature.endswith("_dev"):
        return "Baseline Deviation"

    if feature.endswith("_zdev"):
        return "Normalized Deviation"

    if feature.endswith("_delta"):
        return "First Difference"

    if "_mean_" in feature:
        return "Rolling Mean"

    if "_std_" in feature:
        return "Rolling Std"

    if "_slope_" in feature:
        return "Rolling Slope"

    if feature.startswith("s"):
        return "Raw Sensors"

    return "Other"


importance_df["group"] = (
    importance_df["feature"]
    .apply(get_feature_group)
)


group_importance = (
    importance_df
    .groupby("group")["importance_mean"]
    .sum()
    .sort_values(
        ascending=False
    )
)

print("\n" + "=" * 70)
print("FEATURE GROUP IMPORTANCE")
print("=" * 70)

print(
    group_importance.to_string()
)


plt.figure(
    figsize=(10, 6)
)

group_importance.sort_values().plot(
    kind="barh"
)

plt.xlabel(
    "Total Permutation Importance"
)

plt.ylabel(
    "Feature Group"
)

plt.title(
    f"Feature Group Importance - {best_name}"
)

plt.tight_layout()

plt.show()

# ============================================================
# 24. PREDICTION DISTRIBUTION DIAGNOSTIC
# ============================================================
# Summary: final sanity check — compares the distribution
# (min/max/mean/quartiles) of predicted RUL vs. true RUL, then
# prints the final chosen model's MAE/RMSE/R² one more time.

print(
    "\nPrediction statistics:"
)

print(
    pd.Series(best_pred)
    .describe()
)

print(
    "\nTrue RUL statistics:"
)

print(
    pd.Series(y_test_raw)
    .describe()
)

print(
    "\nMean true RUL:",
    y_test_raw.mean()
)

print(
    "Mean predicted RUL:",
    best_pred.mean()
)

print(
    "\nFinal result:"
)

best_result = df_results.iloc[0]

print(
    f"Model: {best_result['model']}"
)

print(
    f"MAE  : {best_result['MAE']:.3f}"
)

print(
    f"RMSE : {best_result['RMSE']:.3f}"
)

print(
    f"R²   : {best_result['R2']:.4f}"
)