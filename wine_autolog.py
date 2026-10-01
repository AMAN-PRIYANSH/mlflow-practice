"""
wine_autolog.py - the MLflow quickstart from the second MLflow document:
    1. autologging   : one line makes MLflow record parameters, metrics and the model
    2. load the model: read the saved model back from MLflow and predict with it

    python wine_autolog.py
"""
import matplotlib
matplotlib.use("Agg")             # draw autolog's charts to files, no window needed
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("MLflow_Quickstart")
mlflow.sklearn.autolog()          # MLflow now records everything by itself

with mlflow.start_run(run_name="logistic_regression_autolog") as run:
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=1000))
    model.fit(X_train, y_train)
    print("Model and metrics logged automatically!")

# ---- load the model back from MLflow and use it (inference)
model_uri = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.sklearn.load_model(model_uri)
predictions = loaded_model.predict(X_test)
accuracy = (predictions == y_test).mean()
print(f"Loaded {model_uri}")
print(f"First 10 predictions: {predictions[:10].tolist()}")
print(f"Accuracy of the loaded model on the test set: {accuracy:.4f}")
