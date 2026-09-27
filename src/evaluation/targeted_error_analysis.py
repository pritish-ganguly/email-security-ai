"""
Targeted error analysis for Email Security AI.

This script analyzes selected borderline and misclassified
SpamAssassin emails using the frozen production model.

It does not retrain the model.
It does not modify model artifacts.
It does not modify the production threshold.
"""

from pathlib import Path
import re
from typing import Any

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor


THRESHOLD = -0.0975
BODY_PREVIEW_LENGTH = 700

TARGETS = [
    (
        "HAM borderline",
        "data/raw/spamassassin/easy_ham/"
        "0242.a02f8a0ce9077130c10d33db2a16ec36",
        "ham",
    ),
    (
        "HAM borderline",
        "data/raw/spamassassin/easy_ham/"
        "0481.3a50c09230e74743965ddec5ad754fa6",
        "ham",
    ),
    (
        "HAM false positive",
        "data/raw/spamassassin/easy_ham/"
        "0068.f1c604a78739e4f966253d762c972dde",
        "ham",
    ),
    (
        "SPAM borderline",
        "data/raw/spamassassin/spam/"
        "0484.cd802b94da9c80db5e4432bb661effd1",
        "spam",
    ),
    (
        "SPAM borderline",
        "data/raw/spamassassin/spam/"
        "0436.97d3a7bc4377152052dd717581387f36",
        "spam",
    ),
    (
        "SPAM false negative",
        "data/raw/spamassassin/spam/"
        "0299.9d0b292172cb787eb2ed9e8855222edd",
        "spam",
    ),
    (
        "SPAM false negative",
        "data/raw/spamassassin/spam/"
        "0228.23fc5aadfceb81d121d77dfe37f6929a",
        "spam",
    ),
    (
        "SPAM false negative",
        "data/raw/spamassassin/spam/"
        "0281.7e8c08897b61b9b008238efec9ca8d15",
        "spam",
    ),
    (
        "SPAM false negative",
        "data/raw/spamassassin/spam/"
        "0402.1290489e7e62ac9bb500677606540e5d",
        "spam",
    ),
    (
        "SPAM false negative",
        "data/raw/spamassassin/spam/"
        "0095.e1db2d3556c2863ef7355faf49160219",
        "spam",
    ),
]


SUSPICIOUS_TERMS = (
    "free",
    "winner",
    "win",
    "prize",
    "money",
    "cash",
    "offer",
    "click",
    "buy",
    "sale",
    "deal",
    "credit",
    "loan",
    "casino",
    "powerball",
    "warez",
    "green card",
    "viagra",
    "investment",
    "million",
    "jackpot",
    "discount",
    "limited",
    "urgent",
)


def get_value(
    data: Any,
    field: str,
    default: Any = "",
) -> Any:
    if isinstance(data, dict):
        return data.get(field, default)

    return getattr(data, field, default)


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


def count_words(
    text: str,
) -> int:
    return len(text.split())


def count_urls(
    text: str,
) -> int:
    return len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE,
        )
    )


def find_suspicious_terms(
    text: str,
) -> list[str]:
    lowered = text.lower()

    return [
        term
        for term in SUSPICIOUS_TERMS
        if term in lowered
    ]


def preview_body(
    body: str,
) -> str:
    cleaned = " ".join(
        body.split()
    )

    if not cleaned:
        return "[empty]"

    if len(cleaned) <= BODY_PREVIEW_LENGTH:
        return cleaned

    return (
        cleaned[:BODY_PREVIEW_LENGTH]
        + "..."
    )


def classify_relationship(
    expected: str,
    score: float,
) -> str:
    predicted = (
        "spam"
        if score >= THRESHOLD
        else "ham"
    )

    if predicted != expected:
        if expected == "ham":
            return "FALSE POSITIVE — HAM -> SPAM"

        return "FALSE NEGATIVE — SPAM -> HAM"

    if abs(score - THRESHOLD) <= 0.10:
        return "CORRECT — BORDERLINE"

    return "CORRECT"


def analyze_target(
    predictor: EmailPredictor,
    category: str,
    path: Path,
    expected: str,
) -> None:
    print()
    print("=" * 78)
    print(f"{category}: {path.name}")
    print("=" * 78)

    if not path.exists():
        print()
        print("STATUS: FILE NOT FOUND")
        print(f"Expected path: {path}")
        return

    try:
        email_data = parse_email(path)
    except Exception as exc:
        print()
        print("STATUS: PARSE ERROR")
        print(exc)
        return

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

    email_text = build_email_text(
        email_data
    )

    try:
        result = predictor.predict(
            email_text
        )
    except Exception as exc:
        print()
        print("STATUS: PREDICTION ERROR")
        print(exc)
        return

    score = float(
        result["decision_score"]
    )

    predicted = (
        "spam"
        if score >= THRESHOLD
        else "ham"
    )

    combined_text = (
        f"{subject}\n"
        f"{body}\n"
        f"{html_body}"
    )

    suspicious_terms = (
        find_suspicious_terms(
            combined_text
        )
    )

    url_count = count_urls(
        combined_text
    )

    word_count = count_words(
        combined_text
    )

    distance = abs(
        score - THRESHOLD
    )

    print()
    print(f"Expected label : {expected.upper()}")
    print(f"Predicted      : {predicted.upper()}")
    print(f"SVM score      : {score:.4f}")
    print(f"Threshold      : {THRESHOLD:.4f}")
    print(f"Distance       : {distance:.4f}")
    print(
        "Assessment     : "
        f"{classify_relationship(expected, score)}"
    )

    print()
    print("Email features")
    print("-" * 78)

    print(
        f"Subject length : {len(subject)}"
    )

    print(
        f"Word count     : {word_count}"
    )

    print(
        f"URL count      : {url_count}"
    )

    print(
        "HTML present   : "
        f"{bool(html_body.strip())}"
    )

    print(
        "Suspicious terms: "
        + (
            ", ".join(suspicious_terms)
            if suspicious_terms
            else "none"
        )
    )

    print()
    print("Subject")
    print("-" * 78)
    print(
        subject
        if subject
        else "[empty]"
    )

    print()
    print("Body preview")
    print("-" * 78)
    print(
        preview_body(body)
    )


def main() -> None:
    print()
    print("=" * 78)
    print(
        "EMAIL SECURITY AI — "
        "TARGETED ERROR ANALYSIS"
    )
    print("=" * 78)

    print()
    print(
        "Loading frozen production model..."
    )

    predictor = EmailPredictor()

    print(
        "Model loaded."
    )

    for category, file_path, expected in TARGETS:
        analyze_target(
            predictor=predictor,
            category=category,
            path=Path(file_path),
            expected=expected,
        )

    print()
    print("=" * 78)
    print(
        "TARGETED ERROR ANALYSIS COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "Production model: Linear SVM"
    )

    print(
        f"Production threshold: {THRESHOLD:.4f}"
    )

    print()
    print(
        "Diagnostic only."
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


if __name__ == "__main__":
    main()
