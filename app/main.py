"""
FastAPI application for serving fraud detection models.

This module exposes a REST API with endpoints for performing
predictions on the synthetic fraud detection dataset.  At startup the
application generates a small dataset and trains baseline models for
each target variable.  When a request arrives the appropriate model is
selected (or trained on demand) and used to produce a prediction.

Endpoints are provided for generic prediction as well as chat and voice
channels to illustrate multi‑channel support.  All channels share the
same underlying prediction logic.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel, Field, validator

import numpy as np
import pandas as pd

from data.generate_data import generate_dataset
from src.train import (
    get_classifier,
    get_regressor,
)


class PredictionRequest(BaseModel):
    """Schema for prediction requests."""

    features: List[float] = Field(..., description="List of 20 numerical feature values")
    model_name: str = Field(
        "RandomForest",
        description="Name of the model to use (RandomForest, LogisticRegression, XGBoost)",
    )

    @validator("features")
    def check_length(cls, v: List[float]) -> List[float]:  # noqa: B902
        if len(v) != 20:
            raise ValueError("Exactly 20 feature values are required")
        return v


class PredictionResponse(BaseModel):
    """Schema for prediction responses."""

    target: str
    model_name: str
    prediction: float


def load_training_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate or load a small training dataset for warming up models."""
    # In a real system you might load a persisted training set or connect to a
    # feature store.  Here we call generate_dataset to create synthetic data.
    return generate_dataset(n_samples=n_samples, random_state=random_state)


class ModelRepository:
    """Simple in‑memory model registry for serving predictions."""

    def __init__(self) -> None:
        self.models: Dict[str, Dict[str, object]] = {}
        self.training_data = load_training_data()
        self.feature_cols = [col for col in self.training_data.columns if col.startswith("feature_")]
        # Determine which targets are classification vs regression
        self.task_types: Dict[str, str] = {
            "fraud_label": "classification",
            "chargeback_label": "classification",
            "takeover_label": "classification",
            "anomaly_score": "regression",
        }
        # Pre‑train default RandomForest models to reduce latency on first request
        for target, task in self.task_types.items():
            self.models[target] = {}
            y = self.training_data[target].values
            X = self.training_data[self.feature_cols].values
            if task == "classification":
                clf = get_classifier("RandomForest", random_state=42)
                clf.fit(X, y)
                self.models[target]["RandomForest"] = clf
            else:
                reg = get_regressor("RandomForest", random_state=42)
                reg.fit(X, y)
                self.models[target]["RandomForest"] = reg

    def get_or_train(self, target: str, model_name: str) -> object:
        """Retrieve an existing model or train a new one on demand."""
        if target not in self.task_types:
            raise KeyError(f"Unknown target: {target}")
        # return if present
        if model_name in self.models.get(target, {}):
            return self.models[target][model_name]
        # train model lazily
        y = self.training_data[target].values
        X = self.training_data[self.feature_cols].values
        if self.task_types[target] == "classification":
            model = get_classifier(model_name, random_state=42)
        else:
            if model_name == "LogisticRegression":
                raise ValueError("LogisticRegression is not valid for regression targets")
            model = get_regressor(model_name, random_state=42)
        model.fit(X, y)
        # store and return
        self.models[target][model_name] = model
        return model


@lru_cache(maxsize=1)
def get_model_repo() -> ModelRepository:
    """Cached instance of ModelRepository."""
    return ModelRepository()


app = FastAPI(title="Fraud Detection API", version="1.0")


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Fraud detection service is running"}


async def predict_internal(target: str, request: PredictionRequest) -> PredictionResponse:
    repo = get_model_repo()
    model_name = request.model_name
    try:
        model = repo.get_or_train(target, model_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown target {target}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    features = np.array(request.features).reshape(1, -1)
    pred = float(model.predict(features).ravel()[0])
    return PredictionResponse(target=target, model_name=model_name, prediction=pred)


@app.post("/predict/{target}", response_model=PredictionResponse)
async def predict(
    target: str = Path(..., description="Name of the target variable to predict"),
    request: PredictionRequest = ...,
) -> PredictionResponse:
    """Predict the specified target using the given model and features."""
    return await predict_internal(target, request)


@app.post("/chat/{target}", response_model=PredictionResponse)
async def chat_predict(
    target: str = Path(..., description="Name of the target variable to predict"),
    request: PredictionRequest = ...,
) -> PredictionResponse:
    """Chat channel for predictions.  Behaves like `/predict/{target}`."""
    return await predict_internal(target, request)


@app.post("/voice/{target}", response_model=PredictionResponse)
async def voice_predict(
    target: str = Path(..., description="Name of the target variable to predict"),
    request: PredictionRequest = ...,
) -> PredictionResponse:
    """Voice channel for predictions.  Behaves like `/predict/{target}`."""
    return await predict_internal(target, request)