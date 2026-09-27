"""
Run Email Security AI inference on a real .eml file.

Pipeline:

    .eml file
        ↓
    Email parser
        ↓
    Email text construction
        ↓
    Frozen ML model
        ↓
    Security indicator analysis
        ↓
    Risk assessment
        ↓
    Decision engine
        ↓
    Final security decision
"""

from pathlib import Path

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor

from src.security.security_analyzer import (
    analyze_email,
    print_security_analysis,
)

from src.security.risk_engine import (
    assess_email_risk,
    print_risk_assessment,
)

from src.security.decision_engine import (
    make_decision,
    print_decision,
)


SAMPLE_EMAIL = Path(
    "data/raw/spamassassin/easy_ham/"
    "0001.ea7e79d3153e7469e7a9c3e0af6a357e"
)


def _get_email_value(
    email_data: object,
    field: str,
    default: object = "",
) -> object:

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


def _build_email_text(
    email_data: object,
) -> str:

    subject = str(
        _get_email_value(
            email_data,
            "subject",
            "",
        )
        or ""
    )

    body = str(
        _get_email_value(
            email_data,
            "body",
            "",
        )
        or ""
    )

    html_body = str(
        _get_email_value(
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


def main() -> None:

    if not SAMPLE_EMAIL.exists():
        raise FileNotFoundError(
            f"Email file not found: {SAMPLE_EMAIL}"
        )

    print(
        "\nLoading Email Security AI model..."
    )

    predictor = EmailPredictor()

    print(
        "Parsing email file..."
    )

    email_data = parse_email(
        SAMPLE_EMAIL
    )

    email_text = _build_email_text(
        email_data
    )

    if not email_text.strip():
        raise ValueError(
            "Parsed email does not contain usable text."
        )

    print(
        "Running ML prediction..."
    )

    ml_result = predictor.predict(
        email_text
    )

    subject = str(
        _get_email_value(
            email_data,
            "subject",
            "",
        )
        or ""
    )

    body = str(
        _get_email_value(
            email_data,
            "body",
            "",
        )
        or ""
    )

    html_body = str(
        _get_email_value(
            email_data,
            "html_body",
            "",
        )
        or ""
    )

    attachments = _get_email_value(
        email_data,
        "attachments",
        [],
    )

    print(
        "Running security analysis..."
    )

    security_analysis = analyze_email(
        subject=subject,
        body=body,
        html_body=html_body,
        attachments=attachments,
    )

    print(
        "Running risk assessment..."
    )

    risk_assessment = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_analysis,
    )

    print(
        "Running decision engine..."
    )

    decision = make_decision(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    prediction = ml_result.get(
        "label",
        "unknown",
    )

    is_spam = bool(
        ml_result.get(
            "spam",
            ml_result.get(
                "is_spam",
                False,
            ),
        )
    )

    score = float(
        ml_result.get(
            "decision_score",
            ml_result.get(
                "score",
                0.0,
            ),
        )
    )

    threshold = float(
        ml_result.get(
            "threshold",
            0.0,
        )
    )

    print("\n" + "=" * 70)
    print(
        "EMAIL SECURITY AI — .EML SECURITY ANALYSIS"
    )
    print("=" * 70)

    print("\nEmail information")
    print("-" * 70)

    print(
        f"\nFile:\n"
        f"  {SAMPLE_EMAIL}"
    )

    print(
        f"\nSender:\n"
        f"  {_get_email_value(email_data, 'sender', '')}"
    )

    print(
        f"\nReceiver:\n"
        f"  {_get_email_value(email_data, 'receiver', '')}"
    )

    print(
        f"\nSubject:\n"
        f"  {subject}"
    )

    print(
        f"\nDate:\n"
        f"  {_get_email_value(email_data, 'date', '')}"
    )

    print(
        f"\nAttachments:\n"
        f"  {attachments}"
    )

    print("\n" + "-" * 70)
    print(
        "MACHINE LEARNING RESULT"
    )
    print("-" * 70)

    print(
        f"\nPrediction : {prediction}"
    )

    print(
        f"Spam       : {is_spam}"
    )

    print(
        f"Score      : {score:.4f}"
    )

    print(
        f"Threshold  : {threshold:.4f}"
    )

    print_security_analysis(
        security_analysis
    )

    print_risk_assessment(
        risk_assessment
    )

    print_decision(
        decision
    )

    print("\n" + "=" * 70)
    print(
        "FINAL SECURITY DECISION"
    )
    print("=" * 70)

    print(
        f"\nClassification : "
        f"{decision.classification}"
    )

    print(
        f"Action         : "
        f"{decision.action}"
    )

    print(
        f"Confidence     : "
        f"{decision.confidence}"
    )

    print(
        f"Risk score     : "
        f"{decision.risk_score}/100"
    )

    print(
        f"Risk level     : "
        f"{decision.risk_level}"
    )

    print(
        "\nReason:"
    )

    print(
        f"  {decision.reason}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
