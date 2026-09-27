"""
Short-email robustness evaluation for Email Security AI.

This script evaluates the frozen production model against short legitimate
SpamAssassin HAM emails.

It does not:
- retrain the model
- change the production threshold
- modify production model artifacts
- perform model selection

Run from the project root:

    python -m src.evaluation.short_email_evaluation
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

MAX_FILES = 1000
SHORT_WORD_LIMIT = 40
THRESHOLD = -0.0975


@dataclass
class EvaluationResult:
    total: int
    short_total: int
    short_false_positives: int
    all_false_positives: int
    scores_short: list[float]


def get_value(
    data: Any,
    field: str,
    default: Any = "",
) -> Any:
    """Safely retrieve a field from parsed email data."""

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
    """
    Build the same general text representation
    used by the production inference pipeline.
    """

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


def word_count(
    text: str,
) -> int:
    """Return whitespace-separated word count."""

    return len(
        text.split()
    )


def collect_ham_files() -> list[Path]:
    """Collect HAM email files from SpamAssassin."""

    files: list[Path] = []

    for directory in HAM_DIRECTORIES:

        if not directory.exists():
            continue

        files.extend(
            path
            for path in directory.iterdir()
            if path.is_file()
        )

    files.sort()

    return files[:MAX_FILES]


def evaluate_model(
    predictor: EmailPredictor,
    files: list[Path],
) -> EvaluationResult:
    """
    Evaluate the frozen production model
    against legitimate HAM emails.
    """

    total = 0
    short_total = 0
    short_false_positives = 0
    all_false_positives = 0

    scores_short: list[float] = []

    for path in files:

        try:

            email_data = parse_email(
                path
            )

            email_text = build_email_text(
                email_data
            )

            if not email_text.strip():
                continue

            result = predictor.predict(
                email_text
            )

            total += 1

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

            content = (
                f"{subject}\n{body}"
            )

            is_short = (
                word_count(content)
                <= SHORT_WORD_LIMIT
            )

            if result["spam"]:
                all_false_positives += 1

            if is_short:

                short_total += 1

                scores_short.append(
                    score
                )

                if result["spam"]:
                    short_false_positives += 1

        except Exception as exc:

            print(
                f"Skipping {path}: {exc}"
            )

    return EvaluationResult(
        total=total,
        short_total=short_total,
        short_false_positives=(
            short_false_positives
        ),
        all_false_positives=(
            all_false_positives
        ),
        scores_short=scores_short,
    )


def run_sanity_checks(
    predictor: EmailPredictor,
) -> None:
    """
    Run inference against a few legitimate
    short-message examples.
    """

    examples = {
        "short_plain": """
hi see you soon.
regards
pritish
""",
        "short_meeting": """
Hi Pritish,

Let's meet tomorrow at 10 AM.

Regards,
John
""",
        "short_reply": """
Thanks for the update.
Regards,
Pritish
""",
    }

    print(
        "\n" + "=" * 70
    )

    print(
        "LEGITIMATE SHORT-MESSAGE SANITY CHECK"
    )

    print(
        "=" * 70
    )

    for name, text in examples.items():

        result = predictor.predict(
            text
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  Prediction : "
            f"{result['label']}"
        )

        print(
            f"  Spam       : "
            f"{result['spam']}"
        )

        print(
            f"  Score      : "
            f"{result['decision_score']:.4f}"
        )

        print(
            f"  Threshold  : "
            f"{result['threshold']:.4f}"
        )


def main() -> None:
    """Run short-email robustness evaluation."""

    print(
        "\n" + "=" * 70
    )

    print(
        "EMAIL SECURITY AI — "
        "SHORT EMAIL ROBUSTNESS EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    files = collect_ham_files()

    if not files:

        raise FileNotFoundError(
            "No HAM emails found in the expected "
            "SpamAssassin directories."
        )

    print(
        f"\nHAM emails discovered: "
        f"{len(files)}"
    )

    print(
        f"Short-email definition: "
        f"<= {SHORT_WORD_LIMIT} words"
    )

    print(
        f"Frozen threshold: "
        f"{THRESHOLD:.4f}"
    )

    print(
        "\nEvaluating frozen production model..."
    )

    result = evaluate_model(
        predictor,
        files,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nHAM emails evaluated       : "
        f"{result.total}"
    )

    print(
        f"Short HAM emails           : "
        f"{result.short_total}"
    )

    print(
        f"All HAM false positives    : "
        f"{result.all_false_positives}"
    )

    print(
        f"Short HAM false positives  : "
        f"{result.short_false_positives}"
    )

    if result.total:

        overall_fpr = (
            result.all_false_positives
            / result.total
            * 100
        )

    else:

        overall_fpr = 0.0

    if result.short_total:

        short_fpr = (
            result.short_false_positives
            / result.short_total
            * 100
        )

    else:

        short_fpr = 0.0

    print(
        f"Overall HAM false-positive rate : "
        f"{overall_fpr:.2f}%"
    )

    print(
        f"Short HAM false-positive rate   : "
        f"{short_fpr:.2f}%"
    )

    if result.scores_short:

        print(
            f"\nShort HAM minimum score    : "
            f"{min(result.scores_short):.4f}"
        )

        print(
            f"Short HAM maximum score    : "
            f"{max(result.scores_short):.4f}"
        )

        print(
            f"Short HAM average score    : "
            f"{sum(result.scores_short) / len(result.scores_short):.4f}"
        )

    run_sanity_checks(
        predictor
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "INTERPRETATION"
    )

    print(
        "=" * 70
    )

    if result.short_total == 0:

        print(
            "\nNo short HAM emails were "
            "available for evaluation."
        )

    elif short_fpr == 0:

        print(
            "\nThe frozen model produced "
            "no false positives within "
            "the evaluated short-HAM subset."
        )

    elif short_fpr <= overall_fpr + 5:

        print(
            "\nShort HAM false positives are "
            "present but are not dramatically "
            "higher than the overall HAM "
            "false-positive rate."
        )

        print(
            "\nThe next step should be targeted "
            "training-data analysis rather than "
            "changing the threshold."
        )

    else:

        print(
            "\nShort HAM false positives are "
            "materially higher than the overall "
            "HAM false-positive rate."
        )

        print(
            "\nThe model should be retrained with "
            "improved short-message representation "
            "and data coverage."
        )

    print(
        "\nNo model artifacts were modified."
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()
