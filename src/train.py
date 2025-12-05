"""
Model training script for the multivariate fraud‑detection project.

This script loads the synthetic dataset, splits it into training and test
subsets, trains machine‑learning models for each of the target variables
(`fraud_label`, `chargeback_label`, `takeover_label`, `anomaly_score`),
evaluates them on a held‑out test set, and logs the experiments using
MLflow.  Each model is logged as an MLflow artifact along with its
performance metrics.

Usage:
    python train.py --data data/transactions.csv

You can customise the test split and random seed with command‑line
arguments.  All outputs (artifacts and metrics) are stored in the
`mlruns` directory in the project root by default.
"""

import argparse
from pathlib import Path

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


def train_and_log_classification(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    target_name: str,
    random_state: int = 42,
) -> None:
    """Train a classifier and log the results to MLflow.

    Parameters
    ----------
    X_train, X_test : array-like
        Feature matrices for training and testing.
    y_train, y_test : array-like
        Target vectors for training and testing.
    model_name : str
        Name of the model to train ('RandomForest' or 'LogisticRegression').
    target_name : str
        Name of the target variable (used for run naming).
    random_state : int
        Random seed for reproducibility.
    """
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=random_state)
    elif model_name == "LogisticRegression":
        # Use saga solver for large datasets; increase max_iter to ensure convergence
        model = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=500,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    with mlflow.start_run(run_name=f"{target_name}_{model_name}"):
        # Log basic parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target_name)
        # Fit model
        model.fit(X_train, y_train)
        # Predict
        preds = model.predict(X_test)
        # Compute metrics
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except Exception:
            roc_auc = float("nan")
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        if not np.isnan(roc_auc):
            mlflow.log_metric("roc_auc", roc_auc)
        # Log model artifact
        mlflow.sklearn.log_model(model, f"model_{target_name}")


def train_and_log_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    target_name: str,
    random_state: int = 42,
) -> None:
    """Train a regressor and log the results to MLflow.

    Parameters
    ----------
    X_train, X_test : array-like
        Feature matrices for training and testing.
    y_train, y_test : array-like
        Target vectors for training and testing.
    model_name : str
        Name of the model to train ('RandomForest').
    target_name : str
        Name of the target variable (used for run naming).
    random_state : int
        Random seed for reproducibility.
    """
    if model_name == "RandomForest":
        model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=random_state)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    with mlflow.start_run(run_name=f"{target_name}_{model_name}"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target_name)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, f"model_{target_name}")


def main():
    parser = argparse.ArgumentParser(description="Train models for fraud detection and log them using MLflow.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/transactions.csv",
        help="Path to the CSV file containing the synthetic dataset.",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    X = df[feature_cols].values

    # Targets
    targets = {
        "fraud_label": "classification",
        "chargeback_label": "classification",
        "takeover_label": "classification",
        "anomaly_score": "regression",
    }

    # Create consistent train/test indices for all targets
    num_samples = len(df)
    all_indices = np.arange(num_samples)
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )
    X_train = X[train_indices]
    X_test = X[test_indices]

    # Start (or create) the MLflow experiment
    mlflow.set_experiment("fraud_detection_multivariate")

    # Train and log each target separately using the same splits
    for target_name, task_type in targets.items():
        y = df[target_name].values
        y_train = y[train_indices]
        y_test = y[test_indices]
        if task_type == "classification":
            # Train both logistic regression and random forest to illustrate different models
            train_and_log_classification(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name="RandomForest",
                target_name=target_name,
                random_state=args.random_state,
            )
            train_and_log_classification(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name="LogisticRegression",
                target_name=target_name,
                random_state=args.random_state,
            )
        else:
            train_and_log_regression(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name="RandomForest",
                target_name=target_name,
                random_state=args.random_state,
            )

    print("Training complete.  Runs logged to the 'mlruns' directory.")


if __name__ == "__main__":
    main()
