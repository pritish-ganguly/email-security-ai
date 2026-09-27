"""
Final evaluation for the Email Security AI project.

The model and classification threshold are selected using the
validation dataset. This module evaluates that frozen configuration
once on the untouched test dataset.

The test set must not influence model or threshold selection.
"""

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.dataset_splitter import split_dataset
from src.data.spamassassin_loader import load_spamassassin
from src.features.text_features import (
    fit_text_pipeline,
    transform_text,
)
from src.models.supervised_baseline import (
    create_models,
    train_models,
)
from src.preprocessing.cleaner import clean_dataset


SELECTED_MODEL = "linear_svm"

SELECTED_THRESHOLD = -0.0975


@dataclass
class FinalEvaluation:
    """Container for final test-set evaluation results."""

    model_name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_positives: int
    false_negatives: int
    confusion_matrix: list


def evaluate_final_model(
    model,
    x_test,
    y_test,
    threshold: float,
) -> FinalEvaluation:
    """
    Evaluate the frozen model configuration on the test set.

    For LinearSVC, decision_function() returns a continuous score.
    The frozen validation threshold is applied to those scores.
    """

    scores = model.decision_function(x_test)

    predictions = (
        scores >= threshold
    ).astype(int)

    confusion = confusion_matrix(
        y_test,
        predictions,
    )

    false_positives = int(
        confusion[0, 1]
    )

    false_negatives = int(
        confusion[1, 0]
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        scores,
    )

    return FinalEvaluation(
        model_name=SELECTED_MODEL,
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        false_positives=false_positives,
        false_negatives=false_negatives,
        confusion_matrix=confusion.tolist(),
    )


def print_final_evaluation(
    result: FinalEvaluation,
    y_test,
    model,
    x_test,
) -> None:
    """Print the final test-set evaluation report."""

    predictions = (
        model.decision_function(x_test)
        >= result.threshold
    ).astype(int)

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — FINAL TEST EVALUATION")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
        "\nThe model and threshold were selected using "
        "validation data only."
        "\nThe test set is being used for final evaluation."
    )

    print("\n" + "-" * 70)

    print(f"\nModel       : {result.model_name}")
    print(f"Threshold   : {result.threshold:.4f}")

    print(f"\nAccuracy    : {result.accuracy:.4f}")
    print(f"Precision   : {result.precision:.4f}")
    print(f"Recall      : {result.recall:.4f}")
    print(f"F1 Score    : {result.f1:.4f}")
    print(f"ROC-AUC     : {result.roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        result.confusion_matrix
    )

    print(
        "\nFalse positives:",
        result.false_positives,
    )

    print(
        "False negatives:",
        result.false_negatives,
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "ham",
                "spam",
            ],
            zero_division=0,
        )
    )


def main():
    """Run the complete final evaluation pipeline."""


    print("\nLoading SpamAssassin dataset...")

    dataframe = load_spamassassin()


    print("\nCleaning dataset...")

    dataframe, _ = clean_dataset(
        dataframe
    )


    print(
        "\nCreating leakage-safe dataset split..."
    )

    split = split_dataset(
        dataframe
    )

    train_data = split.train
    test_data = split.test


    x_train_text = train_data[
        "email_text"
    ]

    x_test_text = test_data[
        "email_text"
    ]

    y_train = train_data[
        "label"
    ]

    y_test = test_data[
        "label"
    ]


    print(
        "\nFitting NLP feature pipeline "
        "on training data only..."
    )

    text_pipeline = fit_text_pipeline(
        x_train_text
    )


    print(
        "\nTransforming training data..."
    )

    x_train = transform_text(
        text_pipeline,
        x_train_text,
    )

    print(
        "Transforming test data..."
    )

    x_test = transform_text(
        text_pipeline,
        x_test_text,
    )


    print(
        "\nCreating supervised models..."
    )

    models = create_models()


    print(
        "\nTraining supervised models..."
    )

    trained_models = train_models(
        models,
        x_train,
        y_train,
    )


    selected_model = trained_models[
        SELECTED_MODEL
    ]

    print("\n" + "=" * 70)
    print("FROZEN MODEL CONFIGURATION")
    print("=" * 70)

    print(
        f"\nModel     : {SELECTED_MODEL}"
    )

    print(
        f"Threshold : {SELECTED_THRESHOLD}"
    )

    print(
        "\nNo model or threshold selection "
        "is performed using the test set."
    )


    result = evaluate_final_model(
        selected_model,
        x_test,
        y_test,
        SELECTED_THRESHOLD,
    )


    print_final_evaluation(
        result,
        y_test,
        selected_model,
        x_test,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "\nTest records:",
        len(y_test),
    )


if __name__ == "__main__":
    main()
