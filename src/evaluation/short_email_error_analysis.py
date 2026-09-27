"""
Short-email error analysis for Email Security AI.

Purpose:
    Analyze short HAM and SPAM emails using the frozen production
    Linear SVM model and identify false positives, false negatives,
    and borderline predictions.

This script is diagnostic only.

It does NOT:
    - retrain the model
    - modify model artifacts
    - modify the production threshold
    - modify the NLP pipeline

Run from the project root:

    python -m src.evaluation.short_email_error_analysis
"""

from pathlib import Path
from statistics import mean, median

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)

SHORT_WORD_LIMIT = 40

MAX_EXAMPLES_PER_CATEGORY = 20

BORDERLINE_DISTANCE = 0.15


def get_value(
    email_data: object,
    field: str,
    default: object = "",
) -> object:
    """Safely retrieve a parsed email field."""

    if isinstance(email_data, dict):
        return email_data.get(
            field,
            default,
        )

    return getattr(
        email_data,
        field,
        default,
    )


def build_email_text(
    email_data: object,
) -> str:
    """Build the same text representation used by inference."""

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

    html_body = str(
        get_value(
            email_data,
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


def count_words(
    text: str,
) -> int:
    """Count whitespace-separated words."""

    return len(
        text.split()
    )


def count_urls(
    text: str,
) -> int:
    """Count basic HTTP/HTTPS/WWW URLs."""

    lowered = text.lower()

    return (
        lowered.count("http://")
        + lowered.count("https://")
        + lowered.count("www.")
    )


def get_html_length(
    email_data: object,
) -> int:
    """Return HTML body length."""

    html_body = str(
        get_value(
            email_data,
            "html_body",
            "",
        )
        or ""
    )

    return len(html_body)


def get_subject(
    email_data: object,
) -> str:
    """Return email subject."""

    return str(
        get_value(
            email_data,
            "subject",
            "",
        )
        or ""
    ).strip()


def get_body(
    email_data: object,
) -> str:
    """Return plain-text body."""

    return str(
        get_value(
            email_data,
            "body",
            "",
        )
        or ""
    ).strip()


def get_html(
    email_data: object,
) -> str:
    """Return HTML body."""

    return str(
        get_value(
            email_data,
            "html_body",
            "",
        )
        or ""
    ).strip()


def get_preview(
    email_data: object,
    maximum: int = 500,
) -> str:
    """Create a readable body preview."""

    body = get_body(
        email_data
    )

    if not body:
        html = get_html(
            email_data
        )

        if html:
            body = html

    body = " ".join(
        body.split()
    )

    if len(body) <= maximum:
        return body

    return (
        body[:maximum - 3]
        + "..."
    )


def detect_html(
    email_data: object,
) -> bool:
    """Determine whether an email contains HTML."""

    return bool(
        get_html(
            email_data
        )
    )


def classify_pattern(
    email_data: object,
) -> list[str]:
    """
    Identify broad diagnostic patterns.

    These are descriptive categories only.
    """

    subject = get_subject(
        email_data
    )

    body = get_body(
        email_data
    )

    html = get_html(
        email_data
    )

    combined = (
        subject
        + " "
        + body
        + " "
        + html
    ).lower()

    words = count_words(
        combined
    )

    urls = count_urls(
        combined
    )

    patterns: list[str] = []

    if words <= 15:
        patterns.append(
            "very-short"
        )

    if any(
        phrase in combined
        for phrase in (
            "regards",
            "thanks",
            "thank you",
            "see you",
            "talk to you",
            "best wishes",
            "cheers",
        )
    ):
        patterns.append(
            "conversational"
        )

    if any(
        phrase in combined
        for phrase in (
            "unsubscribe",
            "click here",
            "buy now",
            "limited time",
            "special offer",
            "sale",
            "free",
            "winner",
            "winning",
            "prize",
            "jackpot",
            "lottery",
        )
    ):
        patterns.append(
            "commercial-or-promotional"
        )

    if any(
        phrase in combined
        for phrase in (
            "warez",
            "serial",
            "crack",
            "software",
            "download",
        )
    ):
        patterns.append(
            "software-or-warez"
        )

    if any(
        phrase in combined
        for phrase in (
            "million",
            "powerball",
            "cash",
            "jackpot",
        )
    ):
        patterns.append(
            "lottery-or-money"
        )

    if html:
        patterns.append(
            "html"
        )

    if urls:
        patterns.append(
            "url"
        )

    if not patterns:
        patterns.append(
            "other"
        )

    return patterns


def format_patterns(
    patterns: list[str],
) -> str:
    """Format diagnostic categories."""

    return ", ".join(
        patterns
    )


def print_separator(
    character: str = "-",
    length: int = 78,
) -> None:
    """Print a separator."""

    print(
        character * length
    )


def analyze_dataset(
    predictor: EmailPredictor,
    directory: Path,
    expected_label: str,
) -> list[dict]:
    """
    Analyze all emails in a dataset directory.

    Only short emails are returned.
    """

    results: list[dict] = []

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
    )

    for path in files:

        try:
            email_data = parse_email(
                path
            )

            text = build_email_text(
                email_data
            )

            if not text.strip():
                continue

            word_count = count_words(
                text
            )

            if word_count > SHORT_WORD_LIMIT:
                continue

            result = predictor.predict(
                text
            )

            score = float(
                result["decision_score"]
            )

            threshold = float(
                result["threshold"]
            )

            predicted = (
                "spam"
                if result["spam"]
                else "ham"
            )

            distance = abs(
                score - threshold
            )

            results.append(
                {
                    "path": path,
                    "email_data": email_data,
                    "expected": expected_label,
                    "predicted": predicted,
                    "score": score,
                    "threshold": threshold,
                    "distance": distance,
                    "word_count": word_count,
                    "url_count": count_urls(
                        text
                    ),
                    "html_length": get_html_length(
                        email_data
                    ),
                    "html": detect_html(
                        email_data
                    ),
                    "subject": get_subject(
                        email_data
                    ),
                    "preview": get_preview(
                        email_data
                    ),
                    "patterns": classify_pattern(
                        email_data
                    ),
                }
            )

        except Exception as exc:
            print(
                f"Warning: failed to process "
                f"{path.name}: {exc}"
            )

    return results


def calculate_statistics(
    results: list[dict],
) -> dict:
    """Calculate dataset statistics."""

    if not results:
        return {
            "total": 0,
            "correct": 0,
            "errors": 0,
            "borderline": 0,
            "scores": [],
        }

    correct = [
        item
        for item in results
        if item["predicted"]
        == item["expected"]
    ]

    errors = [
        item
        for item in results
        if item["predicted"]
        != item["expected"]
    ]

    borderline = [
        item
        for item in results
        if item["distance"]
        <= BORDERLINE_DISTANCE
    ]

    scores = [
        item["score"]
        for item in results
    ]

    return {
        "total": len(results),
        "correct": len(correct),
        "errors": len(errors),
        "borderline": len(borderline),
        "scores": scores,
    }


def pattern_counts(
    items: list[dict],
) -> dict[str, int]:
    """Count diagnostic patterns."""

    counts: dict[str, int] = {}

    for item in items:
        for pattern in item["patterns"]:
            counts[pattern] = (
                counts.get(
                    pattern,
                    0,
                )
                + 1
            )

    return dict(
        sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )
    )


def print_error_examples(
    title: str,
    items: list[dict],
) -> None:
    """Print selected error examples."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    if not items:
        print(
            "\nNone found."
        )
        return

    ordered = sorted(
        items,
        key=lambda item: item["distance"],
    )

    for index, item in enumerate(
        ordered[
            :MAX_EXAMPLES_PER_CATEGORY
        ],
        start=1,
    ):

        print(
            f"\n{index}. "
            f"{item['path'].name}"
        )

        print(
            f"Expected       : "
            f"{item['expected'].upper()}"
        )

        print(
            f"Predicted      : "
            f"{item['predicted'].upper()}"
        )

        print(
            f"SVM score      : "
            f"{item['score']:.4f}"
        )

        print(
            f"Threshold      : "
            f"{item['threshold']:.4f}"
        )

        print(
            f"Distance       : "
            f"{item['distance']:.4f}"
        )

        print(
            f"Word count     : "
            f"{item['word_count']}"
        )

        print(
            f"URL count      : "
            f"{item['url_count']}"
        )

        print(
            f"HTML length    : "
            f"{item['html_length']}"
        )

        print(
            f"Patterns       : "
            f"{format_patterns(item['patterns'])}"
        )

        print(
            f"Subject        : "
            f"{item['subject']}"
        )

        print(
            "Preview        : "
            f"{item['preview']}"
        )


def print_pattern_summary(
    title: str,
    items: list[dict],
) -> None:
    """Print pattern distribution."""

    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)

    counts = pattern_counts(
        items
    )

    if not counts:
        print(
            "No patterns found."
        )
        return

    for pattern, count in counts.items():
        print(
            f"{pattern:<32} : {count}"
        )


def main() -> None:
    """Run short-email error analysis."""

    print(
        "\n" + "=" * 78
    )
    print(
        "EMAIL SECURITY AI — SHORT EMAIL ERROR ANALYSIS"
    )
    print(
        "=" * 78
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    print(
        "Model loaded."
    )

    if not HAM_DIR.exists():
        raise FileNotFoundError(
            f"HAM directory not found: {HAM_DIR}"
        )

    if not SPAM_DIR.exists():
        raise FileNotFoundError(
            f"SPAM directory not found: {SPAM_DIR}"
        )

    print(
        f"\nShort-email definition: "
        f"<= {SHORT_WORD_LIMIT} words"
    )

    print(
        f"Production threshold: "
        f"{predictor.threshold:.4f}"
    )

    print(
        "\nAnalyzing HAM emails..."
    )

    ham_results = analyze_dataset(
        predictor=predictor,
        directory=HAM_DIR,
        expected_label="ham",
    )

    print(
        f"HAM short emails found: "
        f"{len(ham_results)}"
    )

    print(
        "\nAnalyzing SPAM emails..."
    )

    spam_results = analyze_dataset(
        predictor=predictor,
        directory=SPAM_DIR,
        expected_label="spam",
    )

    print(
        f"SPAM short emails found: "
        f"{len(spam_results)}"
    )


    ham_stats = calculate_statistics(
        ham_results
    )

    spam_stats = calculate_statistics(
        spam_results
    )

    ham_false_positives = [
        item
        for item in ham_results
        if item["predicted"] == "spam"
    ]

    spam_false_negatives = [
        item
        for item in spam_results
        if item["predicted"] == "ham"
    ]

    ham_borderline = [
        item
        for item in ham_results
        if item["distance"]
        <= BORDERLINE_DISTANCE
    ]

    spam_borderline = [
        item
        for item in spam_results
        if item["distance"]
        <= BORDERLINE_DISTANCE
    ]


    print(
        "\n" + "=" * 78
    )
    print(
        "RESULTS"
    )
    print(
        "=" * 78
    )

    print(
        "\nSHORT HAM"
    )

    print(
        f"Total evaluated       : "
        f"{ham_stats['total']}"
    )

    print(
        f"Correct HAM           : "
        f"{ham_stats['correct']}"
    )

    print(
        f"False positives       : "
        f"{len(ham_false_positives)}"
    )

    print(
        f"Borderline            : "
        f"{len(ham_borderline)}"
    )

    if ham_stats["total"]:
        ham_fpr = (
            len(ham_false_positives)
            / ham_stats["total"]
            * 100
        )

        print(
            f"False-positive rate   : "
            f"{ham_fpr:.2f}%"
        )

    print(
        "\nSHORT SPAM"
    )

    print(
        f"Total evaluated       : "
        f"{spam_stats['total']}"
    )

    print(
        f"Correct SPAM          : "
        f"{spam_stats['correct']}"
    )

    print(
        f"False negatives       : "
        f"{len(spam_false_negatives)}"
    )

    print(
        f"Borderline            : "
        f"{len(spam_borderline)}"
    )

    if spam_stats["total"]:
        spam_fnr = (
            len(spam_false_negatives)
            / spam_stats["total"]
            * 100
        )

        print(
            f"False-negative rate   : "
            f"{spam_fnr:.2f}%"
        )


    print(
        "\n" + "-" * 78
    )
    print(
        "SHORT EMAIL SCORE DISTRIBUTION"
    )
    print(
        "-" * 78
    )

    if ham_stats["scores"]:
        print(
            "\nHAM:"
        )

        print(
            f"Minimum : "
            f"{min(ham_stats['scores']):.4f}"
        )

        print(
            f"Median  : "
            f"{median(ham_stats['scores']):.4f}"
        )

        print(
            f"Average : "
            f"{mean(ham_stats['scores']):.4f}"
        )

        print(
            f"Maximum : "
            f"{max(ham_stats['scores']):.4f}"
        )

    if spam_stats["scores"]:
        print(
            "\nSPAM:"
        )

        print(
            f"Minimum : "
            f"{min(spam_stats['scores']):.4f}"
        )

        print(
            f"Median  : "
            f"{median(spam_stats['scores']):.4f}"
        )

        print(
            f"Average : "
            f"{mean(spam_stats['scores']):.4f}"
        )

        print(
            f"Maximum : "
            f"{max(spam_stats['scores']):.4f}"
        )


    print_pattern_summary(
        "SHORT HAM PATTERNS",
        ham_results,
    )

    print_pattern_summary(
        "SHORT HAM FALSE-POSITIVE PATTERNS",
        ham_false_positives,
    )

    print_pattern_summary(
        "SHORT SPAM PATTERNS",
        spam_results,
    )

    print_pattern_summary(
        "SHORT SPAM FALSE-NEGATIVE PATTERNS",
        spam_false_negatives,
    )


    print_error_examples(
        "SHORT HAM FALSE POSITIVES",
        ham_false_positives,
    )

    print_error_examples(
        "SHORT SPAM FALSE NEGATIVES",
        spam_false_negatives,
    )


    print_error_examples(
        "SHORT HAM BORDERLINE CASES",
        ham_borderline,
    )

    print_error_examples(
        "SHORT SPAM BORDERLINE CASES",
        spam_borderline,
    )


    print(
        "\n" + "=" * 78
    )
    print(
        "INTERPRETATION"
    )
    print(
        "=" * 78
    )

    if ham_false_positives:
        print(
            "\nShort HAM false positives are present."
        )
    else:
        print(
            "\nNo short HAM false positives were found."
        )

    if spam_false_negatives:
        print(
            "Short SPAM false negatives are present."
        )
    else:
        print(
            "No short SPAM false negatives were found."
        )

    print(
        "\nThis analysis is diagnostic only."
    )

    print(
        "No model artifacts were modified."
    )

    print(
        "No threshold was changed."
    )

    print(
        "No retraining was performed."
    )

    print(
        "\n" + "=" * 78
    )
    print(
        "SHORT EMAIL ERROR ANALYSIS COMPLETE"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
