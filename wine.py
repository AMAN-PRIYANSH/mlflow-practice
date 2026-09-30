"""
wine.py - the MLflow tutorial task: "write any machine learning program,
save it as wine.py, then run `mlflow ui` and see the results".

Dataset : Wine (178 wines, 13 chemical measurements, 3 grape cultivars)
Model   : Random Forest
MLflow  : for every run it records
            - parameters  (n_estimators, max_depth)
            - metrics     (accuracy, precision, recall, F1, ROC-AUC)
            - artifacts   (confusion matrix picture, classification report)
            - the trained model itself

It trains 3 runs with different settings, so the MLflow UI has something to compare.

    python wine.py
    mlflow ui          -> open http://127.0.0.1:5000
"""
import inspect

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

# three settings to compare: (n_estimators, max_depth)
SETTINGS = [(10, 2), (100, 3), (200, None)]

data = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=42, stratify=data.target)

mlflow.set_experiment("Wine_RandomForest_Experiment")

for n_estimators, max_depth in SETTINGS:
    with mlflow.start_run(run_name=f"rf_{n_estimators}_trees_depth_{max_depth}"):
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=42)
        model.fit(X_train, y_train)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            # 3 classes -> "macro" = average of the per-class scores
            "precision": precision_score(y_test, y_pred, average="macro"),
            "recall": recall_score(y_test, y_pred, average="macro"),
            "f1": f1_score(y_test, y_pred, average="macro"),
            # one-vs-rest ROC-AUC for 3 classes
            "roc_auc": roc_auc_score(y_test, y_prob, multi_class="ovr"),
        }
        mlflow.log_metrics(metrics)

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=data.target_names, cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix ({n_estimators} trees, depth {max_depth})")
        fig.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        mlflow.log_text(classification_report(y_test, y_pred, target_names=data.target_names),
                        "classification_report.txt")

        # Newer MLflow saves models with "skops" and asks which object types to trust.
        # A random forest is made of sklearn "Tree" objects, and we trained it ourselves.
        extra = {}
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            extra["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
        mlflow.sklearn.log_model(model, name="random_forest_model", **extra)

        print(f"{n_estimators:>3} trees, depth {str(max_depth):>4}: "
              + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

print("\nRun  mlflow ui  and open http://127.0.0.1:5000 to compare the runs.")
