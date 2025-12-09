"""
Tests for the training module.

These tests verify that the training functions run without raising exceptions
for all supported model types on a tiny synthetic dataset.  They do not
assert specific metrics but ensure that the MLflow logging and model
training pipelines execute end‑to‑end.
"""

import numpy as np

from data.generate_data import generate_dataset
from src.train import (
    train_and_log_classification,
    train_and_log_regression,
    parse_models_arg,
)


def test_training_functions_smoke(tmp_path):
    # Generate a small dataset
    df = generate_dataset(n_samples=200, random_state=123)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values
    # Use a fixed split
    X_train = X[:150]
    X_test = X[150:]

    targets = {
        "fraud_label": "classification",
        "chargeback_label": "classification",
        "takeover_label": "classification",
        "anomaly_score": "regression",
    }
    models = parse_models_arg("all")
    for target, task in targets.items():
        y = df[target].values
        y_train = y[:150]
        y_test = y[150:]
        if task == "classification":
            for model_name in models:
                train_and_log_classification(
                    X_train, X_test, y_train, y_test,
                    model_name=model_name,
                    target_name=target,
                    random_state=123,
                )
        else:
            for model_name in models:
                if model_name == "LogisticRegression":
                    continue
                train_and_log_regression(
                    X_train, X_test, y_train, y_test,
                    model_name=model_name,
                    target_name=target,
                    random_state=123,
                )