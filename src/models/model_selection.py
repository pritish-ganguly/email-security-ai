"""
Model-selection utilities for the Email Security AI project.

This module records the validation results used to select a model
and operating threshold before final evaluation on the untouched
test dataset.

The test dataset must not be used here.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class ModelSelectionRecord:
    """Store the selected model and validation operating point."""

    model_name: str
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int
    reason: str


def select_operating_point(
    threshold_results: pd.DataFrame,
) -> pd.Series:
    """
    Return the validation operating point with the highest F1 score.

    This function only identifies a candidate operating point.
    It does not evaluate the test dataset.
    """

    if threshold_results.empty:
        raise ValueError(
            "Threshold results cannot be empty."
        )

    required_columns = {
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
    }

    missing_columns = (
        required_columns
        - set(threshold_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Threshold results are missing columns: "
            f"{sorted(missing_columns)}"
        )

    best_index = (
        threshold_results["f1"]
        .idxmax()
    )

    return threshold_results.loc[
        best_index
    ]


def create_selection_record(
    model_name: str,
    operating_point: pd.Series,
) -> ModelSelectionRecord:
    """
    Create a model-selection record from validation results.
    """

    return ModelSelectionRecord(
        model_name=model_name,
        threshold=float(
            operating_point["threshold"]
        ),
        precision=float(
            operating_point["precision"]
        ),
        recall=float(
            operating_point["recall"]
        ),
        f1=float(
            operating_point["f1"]
        ),
        false_positives=int(
            operating_point["false_positives"]
        ),
        false_negatives=int(
            operating_point["false_negatives"]
        ),
        reason=(
            "Highest validation F1 among the "
            "evaluated operating points."
        ),
    )


def print_selection_record(
    record: ModelSelectionRecord,
) -> None:
    """Print a model-selection record."""

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — MODEL SELECTION")
    print("=" * 70)

    print(
        f"\nModel: {record.model_name}"
    )

    print(
        f"Threshold: {record.threshold:.4f}"
    )

    print(
        f"Precision: {record.precision:.4f}"
    )

    print(
        f"Recall: {record.recall:.4f}"
    )

    print(
        f"F1 Score: {record.f1:.4f}"
    )

    print(
        f"False positives: "
        f"{record.false_positives}"
    )

    print(
        f"False negatives: "
        f"{record.false_negatives}"
    )

    print(
        f"\nReason: {record.reason}"
    )


def compare_model_candidates(
    candidates: list[ModelSelectionRecord],
) -> pd.DataFrame:
    """
    Convert model-selection records into a comparison table.
    """

    if not candidates:
        raise ValueError(
            "At least one model candidate is required."
        )

    records = []

    for candidate in candidates:

        records.append(
            {
                "model": candidate.model_name,
                "threshold": candidate.threshold,
                "precision": candidate.precision,
                "recall": candidate.recall,
                "f1": candidate.f1,
                "false_positives": (
                    candidate.false_positives
                ),
                "false_negatives": (
                    candidate.false_negatives
                ),
            }
        )

    return pd.DataFrame(records)


def print_candidate_comparison(
    candidates: list[ModelSelectionRecord],
) -> None:
    """Print model-selection candidates."""

    comparison = compare_model_candidates(
        candidates
    )

    print("\n" + "=" * 70)
    print("MODEL-SELECTION CANDIDATES")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


def main() -> None:
    """
    Demonstrate the model-selection process using the
    validation results from the current project run.
    """

    linear_svm_thresholds = pd.DataFrame(
        [
            {
                "threshold": 0.0,
                "precision": 0.9510,
                "recall": 0.9798,
                "f1": 0.9652,
                "false_positives": 5,
                "false_negatives": 2,
            },
            {
                "threshold": -0.0975,
                "precision": 0.9340,
                "recall": 1.0000,
                "f1": 0.9659,
                "false_positives": 7,
                "false_negatives": 0,
            },
        ]
    )

    operating_point = select_operating_point(
        linear_svm_thresholds
    )

    record = create_selection_record(
        model_name="linear_svm",
        operating_point=operating_point,
    )

    print_selection_record(
        record
    )


if __name__ == "__main__":
    main()
