"""
Data quality checks for the Email Security AI project.

This module validates the standardized email dataset before
it enters the machine-learning pipeline.
"""

import pandas as pd


REQUIRED_COLUMNS = {
    "email_id",
    "source",
    "source_category",
    "sender",
    "receiver",
    "subject",
    "date",
    "body",
    "html_body",
    "attachments",
    "label",
}


def validate_schema(dataframe: pd.DataFrame) -> None:
    """
    Verify that all required columns are present.
    """

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def check_missing_values(dataframe: pd.DataFrame) -> pd.Series:
    """
    Return the number of missing values in each column.
    """

    return dataframe.isna().sum()


def check_empty_content(dataframe: pd.DataFrame) -> pd.Series:
    """
    Count records with empty subject or body.
    """

    subject_empty = (
        dataframe["subject"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    body_empty = (
        dataframe["body"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    return pd.Series(
        {
            "empty_subject": int(subject_empty.sum()),
            "empty_body": int(body_empty.sum()),
        }
    )


def check_duplicates(dataframe: pd.DataFrame) -> dict:
    """
    Check for duplicate email records.
    """

    duplicate_ids = dataframe["email_id"].duplicated().sum()

    duplicate_content = dataframe.duplicated(
        subset=["subject", "body"]
    ).sum()

    return {
        "duplicate_email_ids": int(duplicate_ids),
        "duplicate_content": int(duplicate_content),
    }


def check_labels(dataframe: pd.DataFrame) -> dict:
    """
    Validate the email classification labels.

    0 = ham
    1 = spam
    """

    valid_labels = {0, 1}

    actual_labels = set(
        dataframe["label"].dropna().unique()
    )

    invalid_labels = actual_labels - valid_labels

    return {
        "valid": len(invalid_labels) == 0,
        "invalid_labels": sorted(invalid_labels),
        "distribution": dataframe["label"].value_counts().to_dict(),
    }


def generate_quality_report(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Generate a complete data-quality report.
    """

    validate_schema(dataframe)

    missing_values = check_missing_values(dataframe)

    empty_content = check_empty_content(dataframe)

    duplicates = check_duplicates(dataframe)

    labels = check_labels(dataframe)

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "missing_values": missing_values.to_dict(),
        "empty_content": empty_content.to_dict(),
        "duplicates": duplicates,
        "labels": labels,
    }


def print_quality_report(report: dict) -> None:
    """
    Print the data-quality report in a readable format.
    """

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — DATA QUALITY REPORT")
    print("=" * 70)

    print(f"\nRows: {report['rows']}")
    print(f"Columns: {report['columns']}")

    print("\nMissing values:")

    for column, count in report["missing_values"].items():
        print(f"  {column}: {count}")

    print("\nEmpty content:")

    for name, count in report["empty_content"].items():
        print(f"  {name}: {count}")

    print("\nDuplicates:")

    for name, count in report["duplicates"].items():
        print(f"  {name}: {count}")

    print("\nLabels:")

    print(
        f"  Valid labels: "
        f"{report['labels']['valid']}"
    )

    print(
        f"  Invalid labels: "
        f"{report['labels']['invalid_labels']}"
    )

    print(
        f"  Distribution: "
        f"{report['labels']['distribution']}"
    )


def inspect_quality_issues(dataframe: pd.DataFrame) -> None:
    """
    Display examples of records that require investigation.

    This function does not modify or delete any data.
    """

    print("\n" + "=" * 70)
    print("QUALITY ISSUE INSPECTION")
    print("=" * 70)


    empty_body_mask = (
        dataframe["body"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_body = dataframe[empty_body_mask]

    print("\nEmpty body examples:")
    print(f"Count: {len(empty_body)}")

    if not empty_body.empty:

        columns = [
            "email_id",
            "source_category",
            "sender",
            "subject",
            "html_body",
            "attachments",
            "label",
        ]

        print(
            empty_body[columns]
            .head(10)
            .to_string(index=False)
        )


    empty_subject_mask = (
        dataframe["subject"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_subject = dataframe[empty_subject_mask]

    print("\nEmpty subject examples:")
    print(f"Count: {len(empty_subject)}")

    if not empty_subject.empty:

        columns = [
            "email_id",
            "source_category",
            "sender",
            "subject",
            "body",
            "label",
        ]

        print(
            empty_subject[columns]
            .head(10)
            .to_string(index=False)
        )


    duplicate_mask = dataframe.duplicated(
        subset=["subject", "body"],
        keep=False,
    )

    duplicates = dataframe[duplicate_mask]

    print("\nDuplicate content examples:")
    print(f"Count: {len(duplicates)}")

    if not duplicates.empty:

        columns = [
            "email_id",
            "source_category",
            "subject",
            "label",
        ]

        print(
            duplicates[columns]
            .sort_values(["subject", "label"])
            .head(20)
            .to_string(index=False)
        )


def inspect_conflicting_duplicates(
    dataframe: pd.DataFrame,
) -> None:
    """
    Find identical subject/body combinations that have
    conflicting labels.

    Example:

        Same email content
        ├── label 0 = ham
        └── label 1 = spam

    These records require investigation because they can
    introduce label noise into supervised learning.
    """

    content_columns = [
        "subject",
        "body",
    ]

    grouped = (
        dataframe
        .groupby(content_columns, dropna=False)["label"]
        .nunique()
    )

    conflicting_groups = grouped[grouped > 1]

    print("\n" + "=" * 70)
    print("CONFLICTING DUPLICATE INSPECTION")
    print("=" * 70)

    print(
        f"\nConflicting content groups: "
        f"{len(conflicting_groups)}"
    )

    if conflicting_groups.empty:
        print(
            "\nNo identical subject/body combinations "
            "have conflicting labels."
        )
        return

    conflicting_index = conflicting_groups.index

    conflicting_records = dataframe.merge(
        pd.DataFrame(
            conflicting_index.tolist(),
            columns=content_columns,
        ),
        on=content_columns,
        how="inner",
    )

    print("\nExamples:")

    print(
        conflicting_records[
            [
                "email_id",
                "source_category",
                "subject",
                "label",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":

    from src.data.spamassassin_loader import (
        load_spamassassin,
    )

    dataframe = load_spamassassin()

    report = generate_quality_report(dataframe)

    print_quality_report(report)

    inspect_quality_issues(dataframe)

    inspect_conflicting_duplicates(dataframe)
