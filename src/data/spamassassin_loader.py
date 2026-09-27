"""
Load and standardize the SpamAssassin email dataset.
"""

from pathlib import Path

import pandas as pd

from src.data.email_parser import parse_email


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPAMASSASSIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spamassassin"
)


CATEGORY_LABELS = {
    "easy_ham": 0,
    "hard_ham": 0,
    "spam": 1,
}


def load_spamassassin() -> pd.DataFrame:
    """
    Load all SpamAssassin emails into a standardized DataFrame.
    """

    records = []

    for category, label in CATEGORY_LABELS.items():

        category_dir = SPAMASSASSIN_DIR / category

        if not category_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {category_dir}"
            )

        email_files = sorted(
            file
            for file in category_dir.iterdir()
            if file.is_file()
        )

        print(
            f"Loading {category}: "
            f"{len(email_files)} emails"
        )

        for email_file in email_files:

            try:
                email_data = parse_email(email_file)

                record = {
                    "email_id": email_file.name,
                    "source": "spamassassin",
                    "source_category": category,
                    "sender": email_data["sender"],
                    "receiver": email_data["receiver"],
                    "subject": email_data["subject"],
                    "date": email_data["date"],
                    "body": email_data["body"],
                    "html_body": email_data["html_body"],
                    "attachments": email_data["attachments"],
                    "label": label,
                }

                records.append(record)

            except Exception as error:

                print(
                    f"Failed to parse "
                    f"{email_file.name}: {error}"
                )

    return pd.DataFrame(records)


if __name__ == "__main__":

    dataframe = load_spamassassin()

    print("\n" + "=" * 70)
    print("SPAMASSASSIN DATASET LOADED")
    print("=" * 70)

    print(f"Shape: {dataframe.shape}")

    print("\nColumns:")
    print(list(dataframe.columns))

    print("\nLabel distribution:")
    print(dataframe["label"].value_counts())

    print("\nSource category distribution:")
    print(dataframe["source_category"].value_counts())

    print("\nFirst record:")
    print(dataframe.iloc[0])
