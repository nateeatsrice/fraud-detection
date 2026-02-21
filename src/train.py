"""
Enhanced model training script for the multivariate fraud‑detection project.

This script loads the synthetic dataset, splits it into training and test
subsets, trains several machine‑learning models for each of the target variables
(`fraud_label`, `chargeback_label`, `takeover_label`, `anomaly_score`),
evaluates them on a held‑out test set, and logs the experiments using
MLflow.  In addition to the original RandomForest and LogisticRegression
classifiers, it supports gradient boosting via **XGBoost** for both
classification and regression tasks.

Usage:
    python train.py --data data/transactions.csv

You can customise the test split, random seed and list of models via
command‑line arguments.

MLflow Configuration:
    - Tracking store (metadata): Local SQLite database (mlflow.db) synced
      to S3 before and after each run for persistence across environments.
    - Artifact store (models, plots): S3 bucket shared across all projects,
      namespaced by project prefix.
    - Each model type gets its own MLflow experiment for clean metric comparison.

    Override via environment variables:
        MLFLOW_TRACKING_URI   – e.g. sqlite:///mlflow.db (default)
        MLFLOW_S3_BUCKET      – e.g. nateeatsrice-mlflow (default)
        MLFLOW_S3_PREFIX      – e.g. fraud-detection (default)
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List

import boto3
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
)
import xgboost as xgb

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------
# Two storage concerns:
#   1. Tracking store  → WHERE metadata lives (params, metrics, run info)
#   2. Artifact store  → WHERE heavy files live (serialized models, plots)
#
# Tracking store: SQLite downloaded from S3 and reuploaded each run.
#                 Cheap and simple — survives Codespace resets.
# Artifact store: S3 bucket shared across all projects, namespaced by prefix.
#
# The S3 bucket is NOT managed by Terraform — it is long-lived foundation
# infrastructure that survives `terraform destroy`.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

# --- Tracking store (metadata) ---
TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{MLFLOW_DB.absolute()}",
)
mlflow.set_tracking_uri(TRACKING_URI)

# --- Artifact store (models, plots) ---
# S3 bucket name — shared across ALL your GitHub projects.
# Each project gets its own prefix (subfolder) inside the bucket.
MLFLOW_S3_BUCKET = os.environ.get(
    "MLFLOW_S3_BUCKET",
    "nateeatsrice-mlflow",
)

# Project-specific prefix — keeps fraud-detection artifacts separate from
# future projects in the same bucket.
MLFLOW_S3_PREFIX = os.environ.get(
    "MLFLOW_S3_PREFIX",
    "fraud-detection",
)

# Construct the S3 artifact URI.
# Final path example: s3://nateeatsrice-mlflow/fraud-detection/mlflow-artifacts
ARTIFACT_LOCATION = f"s3://{MLFLOW_S3_BUCKET}/{MLFLOW_S3_PREFIX}/mlflow-artifacts"


# ---------------------------------------------------------------------------
# S3 sync helpers — infrastructure concern, kept separate from training logic
# ---------------------------------------------------------------------------

def _sync_db_from_s3() -> None:
    """Pull mlflow.db from S3 before the run so we accumulate history.

    On first run (no DB in S3 yet) this is a no-op and a fresh DB will be
    created locally by MLflow, then pushed to S3 at the end of the run.
    """
    s3 = boto3.client("s3")
    s3_key = f"{MLFLOW_S3_PREFIX}/mlflow.db"
    try:
        s3.download_file(MLFLOW_S3_BUCKET, s3_key, str(MLFLOW_DB))
        print(f"[mlflow] Synced mlflow.db from s3://{MLFLOW_S3_BUCKET}/{s3_key}")
    except s3.exceptions.NoSuchKey:
        print("[mlflow] No existing mlflow.db found in S3 — starting fresh.")
    except Exception as e:
        print(f"[mlflow] WARNING: Could not sync mlflow.db from S3: {e}")
        print("[mlflow] Proceeding with local DB (history may be incomplete).")


def _push_db_to_s3() -> None:
    """Push mlflow.db back to S3 after the run so history is persisted.

    Called at the end of main() after all runs have been logged and closed.
    """
    if not MLFLOW_DB.exists():
        print("[mlflow] WARNING: mlflow.db not found locally — nothing to push.")
        return
    s3 = boto3.client("s3")
    s3_key = f"{MLFLOW_S3_PREFIX}/mlflow.db"
    s3.upload_file(str(MLFLOW_DB), MLFLOW_S3_BUCKET, s3_key)
    print(f"[mlflow] Pushed mlflow.db to s3://{MLFLOW_S3_BUCKET}/{s3_key}")


# ---------------------------------------------------------------------------
# MLflow experiment helpers
# ---------------------------------------------------------------------------

def _get_or_create_experiment(experiment_name: str) -> str:
    """Get an existing MLflow experiment or create one with S3 artifact storage.

    mlflow.set_experiment() alone does NOT update the artifact_location of an
    existing experiment — it is baked in at creation time.  This helper uses
    create_experiment() on first creation to lock in the S3 artifact root.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment.

    Returns
    -------
    str
        The experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id

    # First time seeing this experiment — create with the S3 artifact location
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=ARTIFACT_LOCATION,
    )
    print(f"[mlflow] Created experiment '{experiment_name}' → artifacts at {ARTIFACT_LOCATION}")
    return experiment_id


# ---------------------------------------------------------------------------
# Model factory functions
# ---------------------------------------------------------------------------

def get_classifier(model_name: str, random_state: int) -> object:
    """Factory function to instantiate a classifier based on name.

    Parameters
    ----------
    model_name : str
        One of 'RandomForest', 'LogisticRegression', or 'XGBoost'.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    object
        An instantiated classifier ready to fit.
    """
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
        )
    if model_name == "LogisticRegression":
        return LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=500,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def get_regressor(model_name: str, random_state: int) -> object:
    """Factory function to instantiate a regressor based on name.

    Parameters
    ----------
    model_name : str
        One of 'RandomForest' or 'XGBoost'.  LogisticRegression is excluded
        because it is not appropriate for regression tasks.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    object
        An instantiated regressor ready to fit.
    """
    if model_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
        )
    if model_name == "XGBoost":
        return xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

def train_and_log_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    target_name: str,
    random_state: int = 42,
) -> None:
    """Train a classifier and log the results to its own MLflow experiment.

    Each model type gets a dedicated experiment (e.g. fraud_detection_RandomForest)
    so metrics across different model families are never mixed in the same view.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices for training and testing.
    y_train, y_test : np.ndarray
        Target vectors for training and testing.
    model_name : str
        Name of the model to train ('RandomForest', 'LogisticRegression',
        or 'XGBoost').
    target_name : str
        Name of the target variable — used as the run name within the experiment.
    random_state : int
        Random seed for reproducibility.
    """
    model = get_classifier(model_name, random_state)

    # Each model type lives in its own experiment
    experiment_id = _get_or_create_experiment(f"fraud_detection_{model_name}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=target_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target_name)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        try:
            probas = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, probas[:, 1]) if probas.shape[1] > 1 else float("nan")
        except Exception:
            roc_auc = float("nan")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        if not np.isnan(roc_auc):
            mlflow.log_metric("roc_auc", roc_auc)

        mlflow.sklearn.log_model(model, name=f"model_{target_name}_{model_name}")


def train_and_log_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    target_name: str,
    random_state: int = 42,
) -> None:
    """Train a regressor and log the results to its own MLflow experiment.

    Each model type gets a dedicated experiment (e.g. fraud_detection_RandomForest)
    so metrics across different model families are never mixed in the same view.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices for training and testing.
    y_train, y_test : np.ndarray
        Target vectors for training and testing.
    model_name : str
        Name of the model to train ('RandomForest' or 'XGBoost').
    target_name : str
        Name of the target variable — used as the run name within the experiment.
    random_state : int
        Random seed for reproducibility.
    """
    model = get_regressor(model_name, random_state)

    # Each model type lives in its own experiment
    experiment_id = _get_or_create_experiment(f"fraud_detection_{model_name}")

    with mlflow.start_run(experiment_id=experiment_id, run_name=target_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target_name)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, name=f"model_{target_name}_{model_name}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_models_arg(models_str: str) -> List[str]:
    """Parse a comma‑separated list of model names from the command line.

    If the argument is empty or 'all', return all supported models.

    Supported classification models: RandomForest, LogisticRegression, XGBoost
    Supported regression models: RandomForest, XGBoost

    Parameters
    ----------
    models_str : str
        Comma‑separated list of model identifiers or the special word 'all'.

    Returns
    -------
    list of str
        Normalised model names.
    """
    default_models = ["RandomForest", "LogisticRegression", "XGBoost"]
    if not models_str or models_str.lower() == "all":
        return default_models
    models = [m.strip() for m in models_str.split(",") if m.strip()]
    normalised = []
    for m in models:
        if m.lower() in {"randomforest", "rf"}:
            normalised.append("RandomForest")
        elif m.lower() in {"logisticregression", "logreg", "lr"}:
            normalised.append("LogisticRegression")
        elif m.lower() in {"xgboost", "xgb"}:
            normalised.append("XGBoost")
        else:
            raise ValueError(f"Unknown model specified: {m}")
    return normalised


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train models for fraud detection and log them using MLflow."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/transactions.csv",
        help="Path to the CSV file containing the synthetic dataset.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Comma‑separated list of models to train.  Choices: RandomForest, "
            "LogisticRegression, XGBoost.  Use 'all' to train all models."
        ),
    )
    args = parser.parse_args()

    # Pull mlflow.db from S3 before any MLflow calls so we accumulate history
    _sync_db_from_s3()

    # Load data
    df = pd.read_csv(args.data)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X = df[feature_cols].values

    # Targets and task types
    targets = {
        "fraud_label": "classification",
        "chargeback_label": "classification",
        "takeover_label": "classification",
        "anomaly_score": "regression",
    }

    # Train/test split — indices reused across all targets for consistency
    all_indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )
    X_train = X[train_indices]
    X_test = X[test_indices]

    # Determine which models to train
    models_to_train = parse_models_arg(args.models)

    # Train and log — each model type routes to its own experiment internally
    for target_name, task_type in targets.items():
        y = df[target_name].values
        y_train = y[train_indices]
        y_test = y[test_indices]

        if task_type == "classification":
            for model_name in models_to_train:
                train_and_log_classification(
                    X_train, X_test, y_train, y_test,
                    model_name=model_name,
                    target_name=target_name,
                    random_state=args.random_state,
                )
        else:
            for model_name in models_to_train:
                if model_name == "LogisticRegression":
                    continue  # Not valid for regression tasks
                train_and_log_regression(
                    X_train, X_test, y_train, y_test,
                    model_name=model_name,
                    target_name=target_name,
                    random_state=args.random_state,
                )

    # Push updated mlflow.db back to S3 so history is persisted remotely
    _push_db_to_s3()

    print("Training complete.  Runs logged to MLflow.")
    print(f"Artifacts stored at: {ARTIFACT_LOCATION}")


if __name__ == "__main__":
    main()