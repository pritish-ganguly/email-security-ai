"""
Train and persist the final Email Security AI model.

The final model configuration was selected using validation data:

    Model     : Linear SVM
    Threshold : -0.0975

The test dataset is NOT used during model fitting.

This module saves:
    - TF-IDF text pipeline
    - trained Linear SVM
    - frozen model configuration
"""

import json
from pathlib import Path

import joblib

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

VALIDATION_F1 = 0.9659


TEST_METRICS = {
    "accuracy": 0.9834,
    "precision": 0.9434,
    "recall": 0.9524,
    "f1": 0.9479,
    "roc_auc": 0.9977,
    "false_positives": 6,
    "false_negatives": 5,
    "test_records": 662,
}


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIRECTORY = (
    PROJECT_ROOT / "models"
)


def save_json(
    data: dict,
    path: Path,
) -> None:
    """Save a dictionary as formatted JSON."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


def main() -> None:
    """Train and persist the final production model."""

    print(
        "\nLoading SpamAssassin dataset..."
    )

    dataframe = load_spamassassin()


    print(
        "\nCleaning dataset..."
    )

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


    x_train_text = train_data[
        "email_text"
    ]

    y_train = train_data[
        "label"
    ]

    print(
        "\nTraining records:",
        len(train_data),
    )


    print(
        "\nFitting NLP pipeline on training data..."
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


    if SELECTED_MODEL not in trained_models:

        raise RuntimeError(
            f"Selected model '{SELECTED_MODEL}' "
            "was not produced by create_models()."
        )

    final_model = trained_models[
        SELECTED_MODEL
    ]


    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    pipeline_path = (
        MODEL_DIRECTORY
        / "email_text_pipeline.joblib"
    )

    model_path = (
        MODEL_DIRECTORY
        / "linear_svm.joblib"
    )

    config_path = (
        MODEL_DIRECTORY
        / "model_config.json"
    )


    print(
        "\nSaving NLP pipeline..."
    )

    joblib.dump(
        text_pipeline,
        pipeline_path,
    )


    print(
        "Saving Linear SVM model..."
    )

    joblib.dump(
        final_model,
        model_path,
    )


    configuration = {
        "model": SELECTED_MODEL,
        "threshold": SELECTED_THRESHOLD,
        "selection_metric": "validation_f1",
        "validation_f1": VALIDATION_F1,
        "test_metrics": TEST_METRICS,
    }

    print(
        "Saving model configuration..."
    )

    save_json(
        configuration,
        config_path,
    )


    print("\n" + "=" * 70)
    print(
        "EMAIL SECURITY AI — MODEL PERSISTENCE"
    )
    print("=" * 70)

    print(
        "\nFinal model:",
        SELECTED_MODEL,
    )

    print(
        "Threshold:",
        SELECTED_THRESHOLD,
    )

    print(
        "\nSaved files:"
    )

    print(
        f"  {pipeline_path}"
    )

    print(
        f"  {model_path}"
    )

    print(
        f"  {config_path}"
    )

    print("\nModel persistence complete.")


if __name__ == "__main__":
    main()
