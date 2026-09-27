"""
Email dataset cleaning and normalization.

This module prepares standardized email records for
feature engineering and machine-learning pipelines.

Important:
- Original fields are preserved.
- Cleaning is deterministic.
- Records are not blindly deleted.
"""

import re

import pandas as pd


TEXT_COLUMNS = [
    "sender",
    "receiver",
    "subject",
    "body",
    "html_body",
]


def normalize_text(value) -> str:
    """
    Normalize a text field.

    Operations:
    - Convert missing values to empty string.
    - Convert to string.
    - Normalize line endings.
    - Collapse repeated whitespace.
    - Strip leading/trailing whitespace.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize all supported text columns.

    Returns a copy of the dataframe.
    """

    dataframe = dataframe.copy()

    for column in TEXT_COLUMNS:

        if column in dataframe.columns:

            dataframe[column] = (
                dataframe[column]
                .apply(normalize_text)
            )

    return dataframe


def create_email_text(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a unified text representation of each email.

    Subject and body are the primary text sources.

    HTML content is used only when the plain-text body
    is unavailable.
    """

    dataframe = dataframe.copy()

    dataframe["email_text"] = (
        dataframe["subject"].fillna("")
        + "\n"
        + dataframe["body"].fillna("")
    ).str.strip()

    empty_text = dataframe["email_text"].eq("")

    dataframe.loc[empty_text, "email_text"] = (
        dataframe.loc[empty_text, "html_body"]
    )

    return dataframe


def identify_unusable_records(
    dataframe: pd.DataFrame,
) -> pd.Series:
    """
    Identify records with no usable email content.

    A record is considered unusable only when all of the
    following are empty:
    - subject
    - body
    - html_body
    """

    no_subject = dataframe["subject"].eq("")
    no_body = dataframe["body"].eq("")
    no_html = dataframe["html_body"].eq("")

    return no_subject & no_body & no_html


def clean_dataset(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Clean and normalize the email dataset.

    Returns:
        cleaned_dataframe
        cleaning_report
    """

    original_rows = len(dataframe)

    dataframe = normalize_text_columns(dataframe)

    dataframe = create_email_text(dataframe)

    unusable_mask = identify_unusable_records(dataframe)

    unusable_count = int(unusable_mask.sum())

    cleaned_dataframe = dataframe.loc[
        ~unusable_mask
    ].copy()

    removed_rows = original_rows - len(cleaned_dataframe)

    report = {
        "original_rows": original_rows,
        "removed_rows": removed_rows,
        "final_rows": len(cleaned_dataframe),
        "unusable_records": unusable_count,
    }

    return cleaned_dataframe, report


def print_cleaning_report(report: dict) -> None:
    """
    Print cleaning statistics.
    """

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — CLEANING REPORT")
    print("=" * 70)

    print(
        f"\nOriginal rows: "
        f"{report['original_rows']}"
    )

    print(
        f"Unusable records: "
        f"{report['unusable_records']}"
    )

    print(
        f"Removed rows: "
        f"{report['removed_rows']}"
    )

    print(
        f"Final rows: "
        f"{report['final_rows']}"
    )


if __name__ == "__main__":

    from src.data.spamassassin_loader import (
        load_spamassassin,
    )

    dataframe = load_spamassassin()

    cleaned_dataframe, report = clean_dataset(
        dataframe
    )

    print_cleaning_report(report)

    print("\nColumns after cleaning:")

    print(
        cleaned_dataframe.columns.tolist()
    )

    print("\nSample unified email text:")

    print(
        cleaned_dataframe[
            [
                "subject",
                "email_text",
                "label",
            ]
        ]
        .head(5)
        .to_string(index=False)
    )
