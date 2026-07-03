import warnings
warnings.filterwarnings("ignore")
import sys

print("=" * 60)
print("Python executable :", sys.executable)
print("Python version    :", sys.version)
print("=" * 60)

import os
import json
import joblib
import mlflow
import mlflow.sklearn

import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

mlflow.sklearn.autolog(disable=True)

# =====================================================
# MLflow Configuration
# =====================================================

# mlflow.set_tracking_uri("http://127.0.0.1:5000/")
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_tracking_uri("file:./mlruns")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(tracking_uri)

# Hanya set experiment jika script dijalankan langsung
if os.getenv("MLFLOW_RUN_ID") is None:
    mlflow.set_experiment("RandomForest_Hyperparameter_Tuning")

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("dataset/preprocessed_diabetes_data.csv")

# Nama kolom target
TARGET = "Diabetes"

# Pisahkan fitur dan target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Split dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Jumlah data latih : {X_train.shape}")
print(f"Jumlah data uji   : {X_test.shape}")

print(df["Diabetes"].value_counts())
print(df["Diabetes"].value_counts(normalize=True))

# =====================================================
# Hyperparameter
# =====================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini", "entropy"]
}

# =====================================================
# MLflow Run
# =====================================================

with mlflow.start_run(nested=False):
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"],
        "max_features":["sqrt","log2",None]
    }

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, oob_score=True),
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=1,
        verbose=2
    )

    search.fit(X_train,y_train)
    
    best_model = search.best_estimator_
    prediction = best_model.predict(X_test)
    probability = best_model.predict_proba(X_test)[:,1]
    print(best_model.oob_score_)

    accuracy = accuracy_score(y_test,prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted"
    )

    roc = roc_auc_score(
        y_test,
        probability
    )

    # ==========================================
    # Manual Logging Parameter
    # ==========================================

    mlflow.log_params(search.best_params_)
    mlflow.log_param("cv",3)
    mlflow.log_param("scoring","roc_auc")

    # ==========================================
    # Manual Logging Metrics
    # ==========================================

    mlflow.log_metric("oob_score", best_model.oob_score_)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1_score",f1)
    mlflow.log_metric("roc_auc",roc)
    mlflow.log_metric(
        "best_cv_score",
        search.best_score_
    )

    # ==========================================
    # Save Artifact
    # ==========================================

    os.makedirs("artifacts",exist_ok=True)
    cm = confusion_matrix(
        y_test,
        prediction
    )
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax)
    fig.tight_layout()
    fig.savefig("artifacts/confusion_matrix.png")
    plt.close(fig)

    with open(
        "artifacts/classification_report.txt",
        "w"
    ) as f:
        f.write(
            classification_report(
                y_test,
                prediction
            )
        )

    metrics = {
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1,
        "roc_auc":roc,
        "best_cv_score":search.best_score_
    }

    with open(
        "artifacts/metrics.json",
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    joblib.dump(
        best_model,
        "artifacts/random_forest_model.pkl"
    )

    # ==========================================
    # Manual Artifact Logging
    # ==========================================

    mlflow.log_artifact(
        "artifacts/confusion_matrix.png"
    )

    mlflow.log_artifact(
        "artifacts/classification_report.txt"
    )

    mlflow.log_artifact(
        "artifacts/metrics.json"
    )

    # ==========================================
    # Save Model
    # ==========================================

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model"
    )

    print("="*50)
    print("Best Parameter")
    print(search.best_params_)
    print("="*50)
    print("Accuracy :",accuracy)
    print("Precision:",precision)
    print("Recall   :",recall)
    print("F1 Score :",f1)
    print("ROC AUC  :",roc)
    print("="*50)

    plt.close("all")