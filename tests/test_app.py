"""
Tests for the FastAPI application.

These tests perform basic sanity checks on the prediction endpoints.  They
ensure that the API responds successfully and returns a prediction when
provided with a list of 20 feature values.
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _make_request(endpoint: str) -> None:
    payload = {
        "features": [0.1 * ((-1) ** i) for i in range(20)],
        "model_name": "RandomForest",
    }
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200, f"Endpoint {endpoint} returned {response.status_code}"
    data = response.json()
    assert "prediction" in data
    assert data["target"] in {"fraud_label", "chargeback_label", "takeover_label", "anomaly_score"}


def test_predict_endpoint():
    _make_request("/predict/fraud_label")


def test_chat_endpoint():
    _make_request("/chat/takeover_label")


def test_voice_endpoint():
    _make_request("/voice/anomaly_score")