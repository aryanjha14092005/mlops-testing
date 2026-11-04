import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import mlflow
import mlflow.sklearn

# Configure a local default while allowing deployments to point at their own
# tracking server through the environment.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"
)
MODEL_NAME = "iris-classifier"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


def load_mlflow_model() -> Any:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def load_mlflow_model_version() -> str:
    client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    champion = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
    )
    return str(champion.version)


class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


# Application Factory
def create_app(
    model_loader: Callable[[], Any] | None = None,
    version_loader: Callable[[], str] | None = None,
) -> FastAPI:
    uses_mlflow = model_loader is None
    if version_loader is None:
        version_loader = (
            load_mlflow_model_version if uses_mlflow else lambda: "injected"
        )
    if model_loader is None:
        model_loader = load_mlflow_model

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.model = model_loader()
        except Exception as exc:
            if uses_mlflow:
                raise RuntimeError(
                    f"Failed to load registered MLflow model {MODEL_URI!r} "
                    f"from tracking URI {MLFLOW_TRACKING_URI!r}: {exc}"
                ) from exc
            raise RuntimeError(f"Failed to load injected model: {exc}") from exc

        try:
            app.state.model_version = version_loader()
        except Exception as exc:
            if uses_mlflow:
                raise RuntimeError(
                    f"Failed to resolve alias {MODEL_ALIAS!r} for model "
                    f"{MODEL_NAME!r} from tracking URI "
                    f"{MLFLOW_TRACKING_URI!r}: {exc}"
                ) from exc
            raise RuntimeError(
                f"Failed to load model version from the injected version loader: {exc}"
            ) from exc

        print(f"Loaded {MODEL_NAME} version {app.state.model_version}")
        yield

    app = FastAPI(
        title="Iris Classification API",
        description="Iris prediction API powered by MLFlow",
        version="3.0.0",
        lifespan=lifespan
    )

    @app.get("/")
    def root():
        return {"message": "API is ready to serve..."}

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "model_name": MODEL_NAME,
            "model_version": app.state.model_version,
        }

    @app.post("/predict")
    def predict(data: IrisRequest, request: Request):
        model = request.app.state.model

        features = [[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width,
        ]]

        prediction = int(model.predict(features)[0])

        return {
            "prediction": prediction,
            "class_name": ["setosa", "versicolor", "virginica"][prediction],
        }

    return app


app = create_app()