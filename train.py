import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Configure MLFlow Tracking
# The tracking server stores experiment metadata, metrics, and model records.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:6001"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Group related training runs under a single experiment.
mlflow.set_experiment("iris-classification")

# Load the Iris dataset and separate its features from the target classes.
iris = load_iris()
X = iris.data
y = iris.target

# Hold out data to measure performance on examples the model did not train on.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
)

# Define the classifier configuration.
n_estimators = 10
random_state = 42

# Log the model configuration so runs can be compared later.
model_parameters = {
    "model_type": "RandomForestClassifier",
    "n_estimators": n_estimators,
    "random_state": random_state,
    "test_size": 0.20,
}

# Start a tracked run to associate parameters, metrics, and artifacts.
with mlflow.start_run() as run:
    mlflow.log_params(model_parameters)

    # Create and train the classifier with the selected configuration.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    # Evaluate predictions against the held-out test labels.
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Reject models below the minimum acceptable test accuracy.
    MIN_TEST_ACCURACY = 0.90
    if accuracy < MIN_TEST_ACCURACY:
        raise ValueError(
            f"Model rejected: test accuracy: {accuracy:.2f} "
            f"is below {MIN_TEST_ACCURACY}"
        )

    print("Model Quality Gate Passed. Model is eligible for registration")

    # Log evaluation metrics so model quality can be compared across runs.
    mlflow.log_metrics(
        {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
        }
    )

    print("MLFlow run ID:", run.info.run_id)
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)

    # Save a local copy for workflows that use a file-based model.
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "iris_model.joblib")
    print(f"Model saved to {model_dir / 'iris_model.joblib'}")

    # Log the model to MLflow and register it in the model registry.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="iris-model",
        registered_model_name="iris-classifier",
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    # Find the registered version created by this training run.
    client = MlflowClient()
    current_run_id = mlflow.active_run().info.run_id
    registered_versions = client.search_model_versions(
        "name='iris-classifier'"
    )
    current_model_version = next(
        (
            version
            for version in registered_versions
            if version.run_id == current_run_id
        ),
        None,
    )

    if current_model_version is None:
        raise RuntimeError(
            "Model was logged but its registered version could not be found"
        )

    # Point the champion alias at the newly registered, quality-checked version.
    client.set_registered_model_alias(
        name="iris-classifier",
        alias="champion",
        version=current_model_version.version,
    )

    print(
        f"Champion alias assigned to model version "
        f"{current_model_version.version}"
    )
    print("Registered Model: iris-classifier")