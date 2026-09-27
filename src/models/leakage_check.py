"""
Dataset leakage checks for Email Security AI V2.

Checks whether identical email content appears in more than
one dataset split.
"""

from typing import Iterable

import pandas as pd

from src.data.dataset_splitter import (
    DatasetSplit,
    split_dataset,
)
from src.data.spamassassin_loader import (
    load_spamassassin,
)
from src.preprocessing.cleaner import (
    clean_dataset,
)


def normalize_text(value: object) -> str:
    """Normalize text for duplicate-content comparison."""

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def build_content_keys(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Build normalized subject/body keys for each email.

    Subject and body are deliberately kept together because
    identical content is the leakage signal we are checking.
    """

    subject = dataframe["subject"].map(
        normalize_text
    )

    body = dataframe["body"].map(
        normalize_text
    )

    return subject + "\n" + body


def get_content_keys(
    dataframe: pd.DataFrame,
) -> set[str]:
    """Return unique normalized content keys."""

    return set(
        build_content_keys(dataframe)
    )


def find_overlap(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> set[str]:
    """Return content keys appearing in both datasets."""

    first_keys = get_content_keys(first)
    second_keys = get_content_keys(second)

    return first_keys.intersection(
        second_keys
    )


def print_overlap_summary(
    split: DatasetSplit,
) -> None:
    """Print content overlap between every dataset pair."""

    comparisons: Iterable[
        tuple[str, pd.DataFrame, str, pd.DataFrame]
    ] = (
        (
            "TRAIN",
            split.train,
            "VALIDATION",
            split.validation,
        ),
        (
            "TRAIN",
            split.train,
            "TEST",
            split.test,
        ),
        (
            "VALIDATION",
            split.validation,
            "TEST",
            split.test,
        ),
    )

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI V2 — CONTENT LEAKAGE CHECK")
    print("=" * 70)

    leakage_found = False

    for (
        first_name,
        first_dataframe,
        second_name,
        second_dataframe,
    ) in comparisons:

        overlap = find_overlap(
            first_dataframe,
            second_dataframe,
        )

        print(
            f"\n{first_name} <-> {second_name}"
        )

        print(
            f"Overlapping content groups: "
            f"{len(overlap)}"
        )

        if overlap:
            leakage_found = True

    print("\n" + "-" * 70)

    if leakage_found:
        print(
            "RESULT: CONTENT OVERLAP DETECTED"
        )
        print(
            "The current random split should not "
            "be used for final model evaluation."
        )
    else:
        print(
            "RESULT: NO CONTENT OVERLAP DETECTED"
        )
        print(
            "The current split passed the "
            "duplicate-content leakage check."
        )


def main() -> None:
    """Load, clean, split and validate the dataset."""

    print(
        "Loading SpamAssassin dataset..."
    )

    dataframe = load_spamassassin()

    print(
        "\nCleaning dataset..."
    )

    dataframe, _ = clean_dataset(
        dataframe
    )

    print(
        "\nCreating dataset split..."
    )

    split = split_dataset(
        dataframe
    )

    print_overlap_summary(
        split
    )


if __name__ == "__main__":
    main()
