
# ============================================================
# CUSTOMER CHURN PREDICTION PROJECT
# ============================================================

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

import warnings
warnings.filterwarnings("ignore")

# DATABASE CONNECTION
con = duckdb.connect("file.db")

# LOAD DATASETS
con.sql("CREATE OR REPLACE TABLE cease AS SELECT * FROM 'cease.csv'")
con.sql("CREATE OR REPLACE TABLE customer_info AS SELECT * FROM 'customer_info.csv'")
con.sql("CREATE OR REPLACE TABLE calls AS SELECT * FROM 'calls.csv'")
con.sql("CREATE OR REPLACE TABLE usage AS SELECT * FROM 'usage.csv'")

# TARGET VARIABLE
target_df = con.sql(
    """
    SELECT DISTINCT
        unique_customer_identifier,
        1 AS target
    FROM cease
    """
).df()

# CUSTOMER FEATURES
customer_df = con.sql(
    """
    SELECT
        unique_customer_identifier,
        contract_status,
        contract_dd_cancels,
        dd_cancel_60_day,
        ooc_days,
        technology,
        speed,
        line_speed,
        sales_channel,
        crm_package_name,
        tenure_days
    FROM customer_info
    """
).df()

# CALL FEATURES
call_df = con.sql(
    """
    SELECT
        unique_customer_identifier,
        COUNT(*) AS total_calls,
        SUM(talk_time_seconds) AS total_talk_time,
        SUM(hold_time_seconds) AS total_hold_time
    FROM calls
    GROUP BY unique_customer_identifier
    """
).df()

# USAGE FEATURES
usage_df = con.sql(
    """
    SELECT
        unique_customer_identifier,
        AVG(CAST(usage_download_mbs AS DOUBLE)) AS avg_download_usage,
        AVG(CAST(usage_upload_mbs AS DOUBLE)) AS avg_upload_usage
    FROM usage
    GROUP BY unique_customer_identifier
    """
).df()

# MERGE DATASETS
df = customer_df.merge(
    target_df,
    on="unique_customer_identifier",
    how="left"
)

df = df.merge(
    call_df,
    on="unique_customer_identifier",
    how="left"
)

df = df.merge(
    usage_df,
    on="unique_customer_identifier",
    how="left"
)

df["target"] = df["target"].fillna(0)

# CLEANING
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(0)

categorical_cols = df.select_dtypes(include=["object"]).columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# TYPE CONVERSION
df["ooc_days"] = pd.to_numeric(df["ooc_days"], errors="coerce")
df["dd_cancel_60_day"] = pd.to_numeric(df["dd_cancel_60_day"], errors="coerce")
df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
df["line_speed"] = pd.to_numeric(df["line_speed"], errors="coerce")

# FEATURE ENGINEERING
df["is_out_of_contract"] = np.where(df["ooc_days"] > 0, 1, 0)
df["high_dd_cancel"] = np.where(df["dd_cancel_60_day"] > 1, 1, 0)
df["speed_gap"] = df["speed"] - df["line_speed"]

# ENCODE CATEGORICAL VARIABLES
encoder = LabelEncoder()

for col in categorical_cols:
    if col != "unique_customer_identifier":
        df[col] = encoder.fit_transform(df[col])

# FEATURES & TARGET
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# LOGISTIC REGRESSION
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
log_prob = log_model.predict_proba(X_test)[:, 1]

print("\nLOGISTIC REGRESSION RESULTS")
print(classification_report(y_test, log_pred))
print("ROC-AUC:", roc_auc_score(y_test, log_prob))

# RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=50,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nRANDOM FOREST RESULTS")
print(classification_report(y_test, y_pred))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))

print("\nROC-AUC SCORE")
print(roc_auc_score(y_test, y_prob))

# FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTOP FEATURES")
print(importance_df.head(10))

# EXPORT PREDICTIONS
df["churn_probability"] = model.predict_proba(X)[:, 1]

df["churn_risk"] = np.where(
    df["churn_probability"] >= 0.70,
    "High Risk",
    np.where(
        df["churn_probability"] >= 0.40,
        "Medium Risk",
        "Low Risk"
    )
)

df[[
    "unique_customer_identifier",
    "churn_probability",
    "churn_risk"
]].to_csv("customer_churn_predictions.csv", index=False)

print("\nPrediction file exported successfully.")
