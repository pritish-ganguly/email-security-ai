"""
Decision-boundary error analysis for Email Security AI.

This script analyzes HAM and SPAM emails from the SpamAssassin dataset
using the frozen production Linear SVM and reports examples closest to
the decision boundary.

It does not:
- retrain the model
- change the production threshold
- modify production artifacts
- use the final test set for model selection

Run from the project root:

    python -m src.evaluation.boundary_analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor


HAM_DIRECTORIES = (
    Path("data/raw/spamassassin/easy_ham"),
    Path("data/raw/spamassassin/hard_ham"),
)

SPAM_DIRECTORY = Path(
    "data/raw/spamassassin/spam"
)

MAX_FILES_PER_CLASS = 1000
THRESHOLD = -0.0975
TOP_N = 15


@dataclass
class BoundaryExample:
    path: Path
    label: str
    score: float
    distance: float
    subject: str
    word_count: int


def get_value(
    data: Any,
    field: str,
    default: Any = "",
) -> Any:
    if isinstance(data, dict):
        return data.get(
            field,
            default,
        )

    return getattr(
        data,
        field,
        default,
    )


def build_email_text(
    data: Any,
) -> str:
    subject = str(
        get_value(
            data,
            "subject",
            "",
        )
        or ""
    )

    body = str(
        get_value(
            data,
            "body",
            "",
        )
        or ""
    )

    html_body = str(
        get_value(
            data,
            "html_body",
            "",
        )
        or ""
    )

    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    )


def collect_files(
    directories: tuple[Path, ...],
    maximum: int,
) -> list[Path]:
    files: list[Path] = []

    for directory in directories:

        if not directory.exists():
            continue

        files.extend(
            path
            for path in directory.iterdir()
            if path.is_file()
        )

    files.sort()

    return files[:maximum]


def collect_spam_files() -> list[Path]:
    if not SPAM_DIRECTORY.exists():
        return []

    files = [
        path
        for path in SPAM_DIRECTORY.iterdir()
        if path.is_file()
    ]

    files.sort()

    return files[:MAX_FILES_PER_CLASS]


def count_words(
    subject: str,
    body: str,
) -> int:
    return len(
        f"{subject}\n{body}".split()
    )


def analyze_file(
    predictor: EmailPredictor,
    path: Path,
    expected_label: str,
) -> BoundaryExample | None:

    try:

        email_data = parse_email(
            path
        )

        email_text = build_email_text(
            email_data
        )

        if not email_text.strip():
            return None

        result = predictor.predict(
            email_text
        )

        score = float(
            result["decision_score"]
        )

        subject = str(
            get_value(
                email_data,
                "subject",
                "",
            )
            or ""
        )

        body = str(
            get_value(
                email_data,
                "body",
                "",
            )
            or ""
        )

        distance = abs(
            score - THRESHOLD
        )

        return BoundaryExample(
            path=path,
            label=expected_label,
            score=score,
            distance=distance,
            subject=subject,
            word_count=count_words(
                subject,
                body,
            ),
        )

    except Exception as exc:

        print(
            f"Skipping {path}: {exc}"
        )

        return None


def analyze_class(
    predictor: EmailPredictor,
    files: list[Path],
    expected_label: str,
) -> list[BoundaryExample]:

    results: list[BoundaryExample] = []

    for path in files:

        result = analyze_file(
            predictor,
            path,
            expected_label,
        )

        if result is not None:
            results.append(result)

    results.sort(
        key=lambda item: item.distance
    )

    return results


def print_examples(
    title: str,
    examples: list[BoundaryExample],
) -> None:

    print(
        "\n" + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    for index, example in enumerate(
        examples[:TOP_N],
        start=1,
    ):

        print(
            f"\n{index}. "
            f"{example.path.name}"
        )

        print(
            f"   Expected label : "
            f"{example.label}"
        )

        print(
            f"   SVM score      : "
            f"{example.score:.4f}"
        )

        print(
            f"   Distance       : "
            f"{example.distance:.4f}"
        )

        print(
            f"   Word count     : "
            f"{example.word_count}"
        )

        print(
            f"   Subject        : "
            f"{example.subject[:120]}"
        )


def print_errors(
    examples: list[BoundaryExample],
) -> None:

    errors = []

    for example in examples:

        predicted_spam = (
            example.score >= THRESHOLD
        )

        expected_spam = (
            example.label == "spam"
        )

        if predicted_spam != expected_spam:
            errors.append(
                example
            )

    print(
        "\n" + "=" * 70
    )

    print(
        "MISCLASSIFICATIONS"
    )

    print(
        "=" * 70
    )

    if not errors:

        print(
            "\nNo misclassifications found "
            "in the analyzed sample."
        )

        return

    errors.sort(
        key=lambda item: item.distance
    )

    for index, example in enumerate(
        errors[:TOP_N],
        start=1,
    ):

        prediction = (
            "spam"
            if example.score >= THRESHOLD
            else "ham"
        )

        print(
            f"\n{index}. "
            f"{example.path.name}"
        )

        print(
            f"   Expected    : "
            f"{example.label}"
        )

        print(
            f"   Predicted   : "
            f"{prediction}"
        )

        print(
            f"   Score       : "
            f"{example.score:.4f}"
        )

        print(
            f"   Distance    : "
            f"{example.distance:.4f}"
        )

        print(
            f"   Word count  : "
            f"{example.word_count}"
        )

        print(
            f"   Subject     : "
            f"{example.subject[:120]}"
        )


def main() -> None:

    print(
        "\n" + "=" * 70
    )

    print(
        "EMAIL SECURITY AI — "
        "DECISION BOUNDARY ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    ham_files = collect_files(
        HAM_DIRECTORIES,
        MAX_FILES_PER_CLASS,
    )

    spam_files = collect_spam_files()

    if not ham_files:

        raise FileNotFoundError(
            "No HAM emails found."
        )

    if not spam_files:

        raise FileNotFoundError(
            "No SPAM emails found."
        )

    print(
        f"\nHAM emails analyzed : "
        f"{len(ham_files)}"
    )

    print(
        f"SPAM emails analyzed: "
        f"{len(spam_files)}"
    )

    print(
        f"Production threshold: "
        f"{THRESHOLD:.4f}"
    )

    print(
        "\nAnalyzing HAM decision boundary..."
    )

    ham_results = analyze_class(
        predictor,
        ham_files,
        "ham",
    )

    print(
        "Analyzing SPAM decision boundary..."
    )

    spam_results = analyze_class(
        predictor,
        spam_files,
        "spam",
    )

    print_examples(
        "HAM EXAMPLES CLOSEST TO SPAM BOUNDARY",
        ham_results,
    )

    print_examples(
        "SPAM EXAMPLES CLOSEST TO HAM BOUNDARY",
        spam_results,
    )

    print_errors(
        ham_results + spam_results
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nProduction artifacts were not modified."
    )

    print(
        "Threshold was not changed."
    )

    print(
        "No model retraining was performed."
    )


if __name__ == "__main__":
    main()
