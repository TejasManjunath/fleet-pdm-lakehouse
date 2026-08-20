# Databricks notebook source
import numpy as np

COST_FALSE_NEGATIVE = 500
COST_FALSE_POSITIVE = 10

def total_cost(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)

assert total_cost([0,0,1,1], [0,0,1,1]) == 0
assert total_cost([1], [0]) == 500
assert total_cost([0], [1]) == 10
print("✓ works")

# COMMAND ----------

train_spark = spark.table("fleet_pdm.gold.aps_train")
test_spark = spark.table("fleet_pdm.gold.aps_test")

# Convert to pandas — safe at this size (60k rows x 233 features easily fits in memory)
train_pd = train_spark.toPandas()
test_pd = test_spark.toPandas()

print(f"Train: {train_pd.shape}")
print(f"Test: {test_pd.shape}")
train_pd.head()

# COMMAND ----------

X_train = train_pd.drop(columns=["label"])
y_train = train_pd["label"]

X_test = test_pd.drop(columns=["label"])
y_test = test_pd["label"]

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Class balance in y_train:\n{y_train.value_counts()}")

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# class_weight='balanced' — because the data is 98.3% negative / 1.7% positive.
# Without this, the model would just predict "no failure" for everything and be 98% "accurate" while useless.
lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

# Stratified 5-fold CV — keeps the same class ratio in every fold,
# critical here since positives are only 1.7% of the data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_predict trains on 4 folds, predicts on the 5th, rotates through all 5 —
# gives you out-of-fold predictions for the ENTIRE training set without ever
# letting the model see the row it's predicting
y_proba_cv = cross_val_predict(lr_model, X_train, y_train, cv=skf, method="predict_proba")[:, 1]

print("✓ 5-fold CV complete")
print(f"Predicted probabilities shape: {y_proba_cv.shape}")

# COMMAND ----------

def find_optimal_threshold(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)
    results = []
    for t in thresholds:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        n = len(y_true)
        y_true_arr = np.asarray(y_true)
        fn = np.sum((y_true_arr == 1) & (y_pred == 0))
        fp = np.sum((y_true_arr == 0) & (y_pred == 1))
        cost = ((fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)) * 1000 / n
        results.append({"threshold": t, "cost_per_1000": cost})
    best = min(results, key=lambda r: r["cost_per_1000"])
    return best, results

best_lr, sweep_lr = find_optimal_threshold(y_train, y_proba_cv)
print(f"Best threshold: {best_lr['threshold']:.2f}")
print(f"Best cost per 1000 trucks: €{best_lr['cost_per_1000']:,.2f}")

# Compare to naive baselines
naive_05_pred = (y_proba_cv >= 0.5).astype(int)
inspect_all_pred = np.ones(len(y_train))
inspect_none_pred = np.zeros(len(y_train))

def cost_of(y_true, y_pred):
    y_true_arr = np.asarray(y_true)
    n = len(y_true_arr)
    fn = np.sum((y_true_arr == 1) & (y_pred == 0))
    fp = np.sum((y_true_arr == 0) & (y_pred == 1))
    return ((fn * COST_FALSE_NEGATIVE) + (fp * COST_FALSE_POSITIVE)) * 1000 / n

print(f"\nBaseline comparison (cost per 1000 trucks):")
print(f"  Threshold=0.5:     €{cost_of(y_train, naive_05_pred):,.2f}")
print(f"  Inspect everyone:  €{cost_of(y_train, inspect_all_pred):,.2f}")
print(f"  Inspect no one:    €{cost_of(y_train, inspect_none_pred):,.2f}")
print(f"  Optimized (t={best_lr['threshold']:.2f}):  €{best_lr['cost_per_1000']:,.2f}")

# COMMAND ----------

import mlflow
import mlflow.sklearn

mlflow.set_experiment("/Shared/fleet_pdm_experiments")

with mlflow.start_run(run_name="logistic_regression_baseline"):
    lr_model.fit(X_train, y_train)
    
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("best_threshold", best_lr["threshold"])
    
    mlflow.log_metric("cost_per_1000_cv", best_lr["cost_per_1000"])
    mlflow.log_metric("cost_per_1000_naive_05", cost_of(y_train, naive_05_pred))
    mlflow.log_metric("cost_per_1000_inspect_all", cost_of(y_train, inspect_all_pred))
    mlflow.log_metric("cost_per_1000_inspect_none", cost_of(y_train, inspect_none_pred))
    
    mlflow.sklearn.log_model(lr_model, "model")
    
    print("✓ Logged to MLflow")

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42, n_jobs=-1)
y_proba_cv_rf = cross_val_predict(rf_model, X_train, y_train, cv=skf, method="predict_proba")[:, 1]

best_rf, sweep_rf = find_optimal_threshold(y_train, y_proba_cv_rf)
print(f"RF best threshold: {best_rf['threshold']:.2f}")
print(f"RF best cost per 1000: €{best_rf['cost_per_1000']:,.2f}")

with mlflow.start_run(run_name="random_forest"):
    rf_model.fit(X_train, y_train)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("best_threshold", best_rf["threshold"])
    mlflow.log_metric("cost_per_1000_cv", best_rf["cost_per_1000"])
    mlflow.sklearn.log_model(rf_model, "model")
    print("✓ RF logged to MLflow")

# COMMAND ----------

from xgboost import XGBClassifier
import numpy as np

# Train + CV predict
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=200,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)
y_proba_cv_xgb = cross_val_predict(xgb_model, X_train, y_train, cv=skf, method="predict_proba")[:, 1]

# Log-scale threshold sweep (the correct one — replaces all earlier sweep attempts)
log_thresholds = np.logspace(-5, 0, 100)
best_xgb_log, sweep_xgb_log = find_optimal_threshold(y_train, y_proba_cv_xgb, thresholds=log_thresholds)
print(f"XGBoost best threshold: {best_xgb_log['threshold']:.6f}")
print(f"XGBoost best cost per 1000: €{best_xgb_log['cost_per_1000']:,.2f}")

# Fit final model + log to MLflow — same cell, same run, no cross-cell dependency
with mlflow.start_run(run_name="xgboost_final"):
    xgb_model.fit(X_train, y_train)
    mlflow.log_param("model_type", "XGBoost")
    mlflow.log_param("scale_pos_weight", scale_pos_weight)
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("best_threshold", best_xgb_log["threshold"])
    mlflow.log_metric("cost_per_1000_cv", best_xgb_log["cost_per_1000"])
    mlflow.sklearn.log_model(xgb_model, "model")
    print("✓ XGBoost logged to MLflow")

# COMMAND ----------

import numpy as np

print("XGBoost predicted probability percentiles:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(y_proba_cv_xgb, p):.6f}")

log_thresholds = np.logspace(-5, 0, 100)
best_xgb_log, sweep_xgb_log = find_optimal_threshold(y_train, y_proba_cv_xgb, thresholds=log_thresholds)
print(f"\nLog-scale sweep — best threshold: {best_xgb_log['threshold']:.6f}")
print(f"Log-scale sweep — best cost per 1000: €{best_xgb_log['cost_per_1000']:,.2f}")

# COMMAND ----------

np.random.seed(42)
n_bootstrap = 1000
n_rows = len(y_train)

rf_threshold = best_rf["threshold"]
xgb_threshold = best_xgb_log["threshold"]

y_train_arr = np.asarray(y_train)
rf_proba_arr = np.asarray(y_proba_cv_rf)
xgb_proba_arr = np.asarray(y_proba_cv_xgb)

cost_diffs = []

for i in range(n_bootstrap):
    # Resample row indices with replacement — same indices used for BOTH models,
    # so the comparison stays paired (fair) on each resample
    idx = np.random.choice(n_rows, size=n_rows, replace=True)
    
    y_true_sample = y_train_arr[idx]
    rf_proba_sample = rf_proba_arr[idx]
    xgb_proba_sample = xgb_proba_arr[idx]
    
    rf_pred_sample = (rf_proba_sample >= rf_threshold).astype(int)
    xgb_pred_sample = (xgb_proba_sample >= xgb_threshold).astype(int)
    
    rf_cost = cost_of(y_true_sample, rf_pred_sample)
    xgb_cost = cost_of(y_true_sample, xgb_pred_sample)
    
    cost_diffs.append(rf_cost - xgb_cost)  # positive = XGBoost is cheaper (better)

cost_diffs = np.array(cost_diffs)

ci_lower = np.percentile(cost_diffs, 2.5)
ci_upper = np.percentile(cost_diffs, 97.5)
mean_diff = np.mean(cost_diffs)

print(f"Mean cost advantage of XGBoost over RF: €{mean_diff:,.2f} per 1000 trucks")
print(f"95% CI: [€{ci_lower:,.2f}, €{ci_upper:,.2f}]")
print(f"XGBoost wins in {(cost_diffs > 0).mean() * 100:.1f}% of bootstrap resamples")

# COMMAND ----------

# Both rf_model and xgb_model were already .fit() on the FULL training set
# inside their MLflow logging cells — no retraining needed, just predict on test

rf_test_proba = rf_model.predict_proba(X_test)[:, 1]
xgb_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# Apply the thresholds we already selected via CV — NOT re-optimized on test
rf_test_pred = (rf_test_proba >= best_rf["threshold"]).astype(int)
xgb_test_pred = (xgb_test_proba >= best_xgb_log["threshold"]).astype(int)

rf_test_cost = cost_of(y_test, rf_test_pred)
xgb_test_cost = cost_of(y_test, xgb_test_pred)

print("=== FINAL HELD-OUT TEST SET RESULTS ===")
print(f"Random Forest  (threshold={best_rf['threshold']:.2f}):  €{rf_test_cost:,.2f} per 1000 trucks")
print(f"XGBoost        (threshold={best_xgb_log['threshold']:.6f}):  €{xgb_test_cost:,.2f} per 1000 trucks")
print(f"\nDifference: XGBoost is €{rf_test_cost - xgb_test_cost:,.2f} cheaper per 1000 trucks on the test set")

# Compare to naive baselines on test set too, for the full picture
inspect_all_test = np.ones(len(y_test))
inspect_none_test = np.zeros(len(y_test))
print(f"\nBaseline comparison on test set:")
print(f"  Inspect everyone: €{cost_of(y_test, inspect_all_test):,.2f}")
print(f"  Inspect no one:   €{cost_of(y_test, inspect_none_test):,.2f}")
print(f"  Random Forest:    €{rf_test_cost:,.2f}")
print(f"  XGBoost:          €{xgb_test_cost:,.2f}")

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name="random_forest_final"):
    rf_model.fit(X_train, y_train)
    
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("best_threshold", best_rf["threshold"])
    mlflow.log_metric("cost_per_1000_test", rf_test_cost)
    
    mlflow.sklearn.log_model(
        rf_model,
        "model",
        input_example=X_train.iloc[:5],
        registered_model_name="fleet_pdm.gold.aps_failure_rf"
    )
    print("✓ Random Forest registered in Unity Catalog")

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://dbc-807047c6-74fb.cloud.databricks.com/serving-endpoints/fleet-pdm-rf-endpoint/invocations'
    headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()

# COMMAND ----------

# Get an auth token automatically from the notebook's own session
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = "https://dbc-807047c6-74fb.cloud.databricks.com/serving-endpoints/fleet-pdm-rf-endpoint/invocations"

def score_model(dataset):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method="POST", headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()

# Actually CALL it, with real data, and PRINT the result
sample = X_test.iloc[:5]
result = score_model(sample)

print("Endpoint response:")
print(result)
print(f"\nActual labels for these 5 trucks: {y_test.iloc[:5].tolist()}")

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

healthy_sample = test_pd[test_pd["label"] == 0].sample(n=20, random_state=42)
failure_sample = test_pd[test_pd["label"] == 1].sample(n=10, random_state=42)

sample_trucks = pd.concat([healthy_sample, failure_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

sample_trucks.to_csv("/Volumes/fleet_pdm/gold/raw/sample_trucks.csv", index=False)
print(f"✓ Sample exported — {sum(sample_trucks['label']==1)} failure cases, {sum(sample_trucks['label']==0)} healthy")