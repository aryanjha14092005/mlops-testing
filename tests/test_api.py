# Import TestClient so we can send simulated HTTP requests
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import main


class FakeIrisModel:
    def predict(self, features):
        return [0]


def fake_model_loader():
    return FakeIrisModel()


def fake_version_loader():
    return "test-version"


@pytest.fixture
def client(monkeypatch):
    def fail_if_mlflow_is_used(*_args, **_kwargs):
        raise AssertionError("Injected test models must not access MLflow")

    monkeypatch.setattr(main.mlflow.sklearn, "load_model", fail_if_mlflow_is_used)
    monkeypatch.setattr(main.mlflow, "MlflowClient", fail_if_mlflow_is_used)
    app = main.create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader,
    )
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_name": "iris-classifier",
        "model_version": "test-version",
    }


def test_valid_iris_prediction(client):
    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 0, "class_name": "setosa"}


def test_injected_model_without_version_loader_avoids_mlflow(monkeypatch):
    def fail_if_mlflow_is_used(*_args, **_kwargs):
        raise AssertionError("Injected test models must not access MLflow")

    monkeypatch.setattr(main.mlflow.sklearn, "load_model", fail_if_mlflow_is_used)
    monkeypatch.setattr(main.mlflow, "MlflowClient", fail_if_mlflow_is_used)
    app = main.create_app(model_loader=fake_model_loader)

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_version"] == "injected"


def test_default_loaders_use_registered_champion(monkeypatch):
    calls = {}

    def fake_load_model(model_uri):
        calls["model_uri"] = model_uri
        return FakeIrisModel()

    class FakeMlflowClient:
        def __init__(self, tracking_uri):
            calls["tracking_uri"] = tracking_uri

        def get_model_version_by_alias(self, name, alias):
            calls["alias"] = (name, alias)
            return SimpleNamespace(version="7")

    monkeypatch.setattr(main.mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(main.mlflow.sklearn, "load_model", fake_load_model)
    monkeypatch.setattr(main.mlflow, "MlflowClient", FakeMlflowClient)
    app = main.create_app()

    with TestClient(app) as test_client:
        response = test_client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_version"] == "7"
    assert calls["model_uri"] == "models:/iris-classifier@champion"
    assert calls["alias"] == ("iris-classifier", "champion")


def test_mlflow_startup_error_is_actionable(monkeypatch):
    def fail_to_load_model(_model_uri):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(main.mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(main.mlflow.sklearn, "load_model", fail_to_load_model)

    with pytest.raises(
        RuntimeError,
        match="Failed to load registered MLflow model.*connection refused",
    ):
        with TestClient(main.create_app()):
            pass