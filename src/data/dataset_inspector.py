"""
Inspect the raw datasets used by the Email Security AI project.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_csv(file_path: Path) -> None:
    """Inspect a CSV dataset."""

    print("\n" + "=" * 70)
    print(f"DATASET: {file_path.name}")
    print("=" * 70)

    print(f"Path: {file_path}")
    print(f"Size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")

    df = pd.read_csv(file_path)

    print(f"\nShape: {df.shape}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nLabel distribution:")

    if "label" in df.columns:
        print(df["label"].value_counts(dropna=False))

    print("\nFirst 5 rows:")
    print(df.head())


def inspect_spamassassin() -> None:
    """Inspect the SpamAssassin raw email directories."""

    spamassassin_dir = RAW_DATA_DIR / "spamassassin"

    print("\n" + "=" * 70)
    print("SPAMASSASSIN DATASET")
    print("=" * 70)

    if not spamassassin_dir.exists():
        print("SpamAssassin directory not found.")
        return

    for directory in sorted(spamassassin_dir.iterdir()):

        if directory.is_dir():

            email_files = [
                file for file in directory.iterdir()
                if file.is_file()
            ]

            print(f"\n{directory.name}")
            print(f"Email files: {len(email_files)}")

            if email_files:
                print(f"Example file: {email_files[0].name}")


def main() -> None:
    """Run dataset inspection."""

    print("EMAIL SECURITY AI — DATASET INSPECTION")

    inspect_spamassassin()

    phishing_dir = RAW_DATA_DIR / "phishing"

    for csv_file in sorted(phishing_dir.glob("*.csv")):
        inspect_csv(csv_file)


if __name__ == "__main__":
    main()
