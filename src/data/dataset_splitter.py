"""
Dataset splitting utilities for the Email Security AI project.

Creates separate training, validation, and test datasets while
preventing identical email content from appearing across splits.

The test dataset remains untouched until final evaluation.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    """Container for train, validation and test datasets."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _normalize_text(value: object) -> str:
    """Normalize text before creating duplicate-content groups."""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _create_content_key(dataframe: pd.DataFrame) -> pd.Series:
    """
    Create a normalized key from subject and body.

    Emails with identical normalized subject/body content receive
    the same key and therefore remain in the same dataset split.
    """

    subject = dataframe["subject"].map(
        _normalize_text
    )

    body = dataframe["body"].map(
        _normalize_text
    )

    return subject + "\n" + body


def _validate_input(dataframe: pd.DataFrame) -> None:
    """Validate the dataframe before splitting."""

    required_columns = {
        "subject",
        "body",
        "label",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(
            "Cannot split an empty dataset."
        )

    invalid_labels = set(
        dataframe["label"]
        .dropna()
        .unique()
    ) - {0, 1}

    if invalid_labels:
        raise ValueError(
            "Dataset contains invalid labels: "
            f"{sorted(invalid_labels)}"
        )


def _create_group_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one record per unique content group.

    Each group receives its dominant label. Identical content should
    normally have the same label; conflicting labels are rejected
    because they require separate investigation before training.
    """

    working = dataframe.copy()

    working["_content_key"] = (
        _create_content_key(working)
    )

    label_counts = (
        working
        .groupby("_content_key")["label"]
        .nunique()
    )

    conflicting_groups = label_counts[
        label_counts > 1
    ]

    if not conflicting_groups.empty:
        raise ValueError(
            "Conflicting labels found for "
            f"{len(conflicting_groups)} identical "
            "subject/body content groups. "
            "Resolve these records before creating "
            "a leakage-safe split."
        )

    groups = (
        working
        .groupby("_content_key", as_index=False)
        .agg(
            label=("label", "first"),
            row_count=("label", "size"),
        )
    )

    return groups


def _split_groups(
    groups: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split unique content groups into two sets.

    Stratification is performed using the group labels.
    """

    train_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=random_state,
        stratify=groups["label"],
    )

    return (
        train_groups.reset_index(drop=True),
        test_groups.reset_index(drop=True),
    )


def split_dataset(
    dataframe: pd.DataFrame,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> DatasetSplit:
    """
    Split the dataset into train, validation and test sets.

    Final target proportions:

        60% train
        20% validation
        20% test

    Identical subject/body content is kept within a single split.

    The actual row counts may differ slightly from the requested
    proportions because duplicate-content groups can contain
    multiple email records.
    """

    _validate_input(dataframe)

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    if not 0 < validation_size < 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1."
        )

    groups = _create_group_dataframe(
        dataframe
    )

    train_validation_groups, test_groups = (
        _split_groups(
            groups,
            test_size=test_size,
            random_state=random_state,
        )
    )

    validation_relative_size = (
        validation_size
        / (1 - test_size)
    )

    train_groups, validation_groups = (
        _split_groups(
            train_validation_groups,
            test_size=validation_relative_size,
            random_state=random_state,
        )
    )

    train_keys = set(
        train_groups["_content_key"]
    )

    validation_keys = set(
        validation_groups["_content_key"]
    )

    test_keys = set(
        test_groups["_content_key"]
    )

    train = dataframe[
        _create_content_key(dataframe)
        .isin(train_keys)
    ].copy()

    validation = dataframe[
        _create_content_key(dataframe)
        .isin(validation_keys)
    ].copy()

    test = dataframe[
        _create_content_key(dataframe)
        .isin(test_keys)
    ].copy()

    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def print_split_summary(
    split: DatasetSplit,
) -> None:
    """Print dataset split statistics."""

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — DATASET SPLIT")
    print("=" * 70)

    datasets = {
        "TRAIN": split.train,
        "VALIDATION": split.validation,
        "TEST": split.test,
    }

    for name, dataframe in datasets.items():

        print(f"\n{name}")
        print("-" * 40)

        print(
            f"Rows: {len(dataframe)}"
        )

        label_distribution = (
            dataframe["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print(
            f"Labels: {label_distribution}"
        )

        print(
            f"Spam ratio: "
            f"{dataframe['label'].mean():.4f}"
        )


if __name__ == "__main__":

    from src.data.spamassassin_loader import (
        load_spamassassin,
    )

    from src.preprocessing.cleaner import (
        clean_dataset,
    )

    dataframe = load_spamassassin()

    dataframe, _ = clean_dataset(
        dataframe
    )

    split = split_dataset(
        dataframe
    )

    print_split_summary(
        split
    )
