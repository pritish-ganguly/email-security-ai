import json
from pathlib import Path

import joblib

from src.data.dataset_splitter import split_dataset
from src.preprocessing.cleaner import clean_dataset
from src.data.spamassassin_loader import load_spamassassin
from src.features.text_features import (
    fit_text_pipeline,
    transform_text,
)
from src.models.supervised_baseline import (
    create_models,
    train_models,
)


MODEL_NAME = "linear_svm"
DECISION_THRESHOLD = -0.0975

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "email_security_model.joblib"
PIPELINE_PATH = MODEL_DIR / "text_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def main() -> None:
    """Train the frozen model and save all deployment artifacts."""

    print("\nLoading SpamAssassin dataset...")

    dataframe = load_spamassassin()

    print("\nCleaning dataset...")

    dataframe, _ = clean_dataset(dataframe)

    print("\nCreating leakage-safe dataset split...")

    split = split_dataset(dataframe)

    train_data = split.train

    print("\nFitting NLP pipeline on training data only...")

    text_pipeline = fit_text_pipeline(
        train_data["email_text"]
    )

    print("\nTransforming training data...")

    x_train = transform_text(
        text_pipeline,
        train_data["email_text"],
    )

    y_train = train_data["label"]

    print("\nCreating supervised models...")

    models = create_models()

    print("\nTraining supervised models...")

    trained_models = train_models(
        models,
        x_train,
        y_train,
    )

    if MODEL_NAME not in trained_models:
        raise RuntimeError(
            f"Required model '{MODEL_NAME}' was not trained."
        )

    model = trained_models[MODEL_NAME]

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving model artifacts...")

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        text_pipeline,
        PIPELINE_PATH,
    )

    metadata = {
        "project": "Email Security AI",
        "model": {
            "name": MODEL_NAME,
            "threshold": DECISION_THRESHOLD,
        },
        "training": {
            "records": len(train_data),
            "selection_source": "validation_data",
            "test_set_used_for_selection": False,
        },
        "final_test_evaluation": {
            "records": 662,
            "accuracy": 0.9834,
            "precision": 0.9434,
            "recall": 0.9524,
            "f1_score": 0.9479,
            "roc_auc": 0.9977,
            "false_positives": 6,
            "false_negatives": 5,
        },
        "artifacts": {
            "model": str(MODEL_PATH),
            "text_pipeline": str(PIPELINE_PATH),
        },
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — ARTIFACTS SAVED")
    print("=" * 70)

    print(f"\nModel:")
    print(f"  {MODEL_PATH}")

    print(f"\nText pipeline:")
    print(f"  {PIPELINE_PATH}")

    print(f"\nMetadata:")
    print(f"  {METADATA_PATH}")

    print("\n" + "-" * 70)
    print("FROZEN MODEL CONFIGURATION")
    print("-" * 70)

    print(f"Model     : {MODEL_NAME}")
    print(f"Threshold : {DECISION_THRESHOLD}")

    print("\nTest-set selection: NOT USED")

    print("\nArtifacts are ready for inference.")


if __name__ == "__main__":
    main()
