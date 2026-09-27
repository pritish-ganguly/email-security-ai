"""
Supervised baseline models for the Email Security AI project.

Models:
    - Multinomial Naive Bayes
    - Logistic Regression
    - Linear SVM

These models operate on sparse TF-IDF features.
"""

from dataclasses import dataclass

from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


@dataclass
class ModelResult:
    """Store evaluation results for a model."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    confusion_matrix: list


def create_models() -> dict[str, ClassifierMixin]:
    """
    Create the supervised baseline models.
    """

    return {
        "naive_bayes": MultinomialNB(),

        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),

        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate_model(
    model: ClassifierMixin,
    name: str,
    x_data,
    y_data,
) -> ModelResult:
    """
    Evaluate a trained classifier.
    """

    predictions = model.predict(x_data)

    accuracy = accuracy_score(
        y_data,
        predictions,
    )

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_data,
        predictions,
    )

    roc_auc = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            x_data
        )[:, 1]

        roc_auc = roc_auc_score(
            y_data,
            probabilities,
        )

    elif hasattr(model, "decision_function"):

        scores = model.decision_function(
            x_data
        )

        roc_auc = roc_auc_score(
            y_data,
            scores,
        )

    print("\n" + "=" * 70)
    print(f"MODEL: {name.upper()}")
    print("=" * 70)

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    if roc_auc is not None:
        print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion)

    print("\nClassification Report:")

    print(
        classification_report(
            y_data,
            predictions,
            target_names=[
                "ham",
                "spam",
            ],
            zero_division=0,
        )
    )

    return ModelResult(
        name=name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=confusion.tolist(),
    )


def train_models(
    models: dict[str, ClassifierMixin],
    x_train,
    y_train,
) -> dict[str, ClassifierMixin]:
    """
    Train all supervised models.
    """

    trained_models = {}

    for name, model in models.items():

        print(
            f"\nTraining {name}..."
        )

        model.fit(
            x_train,
            y_train,
        )

        trained_models[name] = model

        print(
            f"{name} training complete."
        )

    return trained_models
