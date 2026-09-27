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
    Decision Engine V3.1
        ↓
    Final security decision
"""

from dataclasses import asdict, is_dataclass
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

from src.security.decision_engine_v31 import (
    make_decision_v31,
)


SAMPLE_EMAIL = Path(
    "data/raw/spamassassin/spam/"
    "0000.7b1b73cf36cf9dbc3d64e3f2ee2b91f1"
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


def _to_dict(
    value: object,
) -> dict:
    """
    Convert dataclass/object results into dictionaries.

    Decision Engine V3.1 expects dictionary inputs.
    """

    if isinstance(value, dict):
        return value

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "to_dict"):
        result = value.to_dict()

        if isinstance(result, dict):
            return result

    if hasattr(value, "__dict__"):
        return vars(value)

    raise TypeError(
        f"Cannot convert {type(value).__name__} "
        "to dictionary."
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

    ml_result = _to_dict(
        ml_result
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

    # ----------------------------------------------------------
    # SECURITY ANALYSIS
    # ----------------------------------------------------------

    print(
        "Running security analysis..."
    )

    security_analysis = analyze_email(
        subject=subject,
        body=body,
        html_body=html_body,
        attachments=attachments,
    )

    # Keep original object for reporting.
    security_analysis_dict = _to_dict(
        security_analysis
    )

    # ----------------------------------------------------------
    # RISK ASSESSMENT
    # ----------------------------------------------------------

    print(
        "Running risk assessment..."
    )

    risk_assessment = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_analysis,
    )

    # Keep original object for reporting.
    risk_assessment_dict = _to_dict(
        risk_assessment
    )

    # ----------------------------------------------------------
    # DECISION ENGINE V3.1
    # ----------------------------------------------------------

    print(
        "Running Decision Engine V3.1..."
    )

    decision = make_decision_v31(
        ml_result=ml_result,
        security_analysis=security_analysis_dict,
        risk_assessment=risk_assessment_dict,
    )

    decision = _to_dict(
        decision
    )

    # ----------------------------------------------------------
    # ML RESULT VALUES
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # MACHINE LEARNING RESULT
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # SECURITY ANALYSIS
    # ----------------------------------------------------------

    print_security_analysis(
        security_analysis
    )

    # ----------------------------------------------------------
    # RISK ASSESSMENT
    # ----------------------------------------------------------

    print_risk_assessment(
        risk_assessment
    )

    # ----------------------------------------------------------
    # DECISION ENGINE V3.1
    # ----------------------------------------------------------

    print("\n" + "-" * 70)
    print(
        "DECISION ENGINE V3.1"
    )
    print("-" * 70)

    print(
        f"\nClassification : "
        f"{decision.get('classification', '')}"
    )

    print(
        f"Action         : "
        f"{decision.get('action', '')}"
    )

    print(
        f"Confidence     : "
        f"{decision.get('confidence', '')}"
    )

    print(
        f"Risk score     : "
        f"{decision.get('risk_score', '')}/100"
    )

    print(
        f"Risk level     : "
        f"{decision.get('risk_level', '')}"
    )

    print(
        "\nReason:"
    )

    print(
        f"  {decision.get('reason', '')}"
    )

    # ----------------------------------------------------------
    # FINAL SECURITY DECISION
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "FINAL SECURITY DECISION"
    )
    print("=" * 70)

    print(
        f"\nClassification : "
        f"{decision.get('classification', '')}"
    )

    print(
        f"Action         : "
        f"{decision.get('action', '')}"
    )

    print(
        f"Confidence     : "
        f"{decision.get('confidence', '')}"
    )

    print(
        f"Risk score     : "
        f"{decision.get('risk_score', '')}/100"
    )

    print(
        f"Risk level     : "
        f"{decision.get('risk_level', '')}"
    )

    print(
        "\nReason:"
    )

    print(
        f"  {decision.get('reason', '')}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()