"""
Model evaluation utilities for the Email Security AI project.

This module evaluates trained classification models on the validation
dataset and provides threshold analysis for models that expose a
decision score or probability.

The test dataset must not be used here. Final test evaluation is
performed only after model selection is complete.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Store the main evaluation metrics for a model."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    false_positives: int
    false_negatives: int


def _get_scores(
    model,
    features,
) -> np.ndarray:
    """
    Return continuous scores for a trained classifier.

    Decision scores are preferred when available. Otherwise,
    class probabilities are used.
    """

    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)

        return np.asarray(scores)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)

        return np.asarray(probabilities[:, 1])

    raise ValueError(
        "Model does not provide decision_function "
        "or predict_proba."
    )


def evaluate_model(
    model,
    name: str,
    features,
    labels,
) -> EvaluationResult:
    """
    Evaluate a trained model using the supplied validation data.
    """

    predictions = model.predict(features)

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0,
    )

    try:
        scores = _get_scores(
            model,
            features,
        )

        roc_auc = roc_auc_score(
            labels,
            scores,
        )

    except ValueError:
        roc_auc = None

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    false_positives = int(matrix[0, 1])
    false_negatives = int(matrix[1, 0])

    print("\n" + "=" * 70)
    print(f"MODEL: {name.upper()}")
    print("=" * 70)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    if roc_auc is not None:
        print(
            f"ROC-AUC  : {roc_auc:.4f}"
        )
    else:
        print(
            "ROC-AUC  : N/A"
        )

    print("\nConfusion Matrix:")

    print(matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            labels,
            predictions,
            target_names=["ham", "spam"],
            zero_division=0,
        )
    )

    print(
        f"False positives: {false_positives}"
    )

    print(
        f"False negatives: {false_negatives}"
    )

    return EvaluationResult(
        name=name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def analyze_thresholds(
    model,
    features,
    labels,
    thresholds: int = 101,
) -> pd.DataFrame:
    """
    Analyze precision, recall and F1 across decision thresholds.

    This analysis uses validation data only.

    For LinearSVC, the decision function is used. For models
    exposing probabilities, the positive-class probability is used.
    """

    scores = _get_scores(
        model,
        features,
    )

    minimum = float(scores.min())
    maximum = float(scores.max())

    threshold_values = np.linspace(
        minimum,
        maximum,
        thresholds,
    )

    records = []

    labels_array = np.asarray(labels)

    for threshold in threshold_values:

        predictions = (
            scores >= threshold
        ).astype(int)

        precision = precision_score(
            labels_array,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            labels_array,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            labels_array,
            predictions,
            zero_division=0,
        )

        matrix = confusion_matrix(
            labels_array,
            predictions,
            labels=[0, 1],
        )

        records.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": int(matrix[0, 1]),
                "false_negatives": int(matrix[1, 0]),
            }
        )

    results = pd.DataFrame(
        records
    )

    return results


def print_threshold_summary(
    threshold_results: pd.DataFrame,
) -> None:
    """
    Print useful threshold operating points.

    The function does not select a production threshold automatically.
    It only reports the validation results so that the threshold can
    be selected deliberately.
    """

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    best_f1_index = (
        threshold_results["f1"]
        .idxmax()
    )

    best_f1 = threshold_results.loc[
        best_f1_index
    ]

    print("\nHighest validation F1 point:")

    print(
        f"Threshold       : "
        f"{best_f1['threshold']:.4f}"
    )

    print(
        f"Precision       : "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1              : "
        f"{best_f1['f1']:.4f}"
    )

    print(
        f"False positives : "
        f"{int(best_f1['false_positives'])}"
    )

    print(
        f"False negatives : "
        f"{int(best_f1['false_negatives'])}"
    )

    print("\nThreshold samples:")

    sample_indices = np.linspace(
        0,
        len(threshold_results) - 1,
        10,
        dtype=int,
    )

    sample_indices = np.unique(
        sample_indices
    )

    columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
    ]

    print(
        threshold_results.iloc[
            sample_indices
        ][columns]
        .to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def compare_results(
    results: list[EvaluationResult],
) -> None:
    """
    Print a compact comparison of evaluated models.
    """

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Model':<25}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'ROC-AUC':<12}"
        f"{'FP':<8}"
        f"{'FN':<8}"
    )

    print("-" * 105)

    for result in results:

        roc_auc = (
            f"{result.roc_auc:.4f}"
            if result.roc_auc is not None
            else "N/A"
        )

        print(
            f"{result.name:<25}"
            f"{result.accuracy:<12.4f}"
            f"{result.precision:<12.4f}"
            f"{result.recall:<12.4f}"
            f"{result.f1:<12.4f}"
            f"{roc_auc:<12}"
            f"{result.false_positives:<8}"
            f"{result.false_negatives:<8}"
        )
