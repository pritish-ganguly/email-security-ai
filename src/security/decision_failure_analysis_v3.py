"""
Decision Engine V3 Failure Analysis.

Analyzes the remaining ML-vs-final-decision disagreements
after Decision Engine V3 evaluation.

This script does NOT:
    - modify the ML model
    - modify the ML threshold
    - retrain anything
    - modify the decision engine

It only investigates why the V3 decision differs from
the frozen ML prediction.
"""

from pathlib import Path

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine_v3 import make_decision_v3


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)


TARGET_FILES = [
    HAM_DIR
    / "1816.7e6c3f51ab4a45f60fbb0968d56f512c",

    SPAM_DIR
    / "0254.02daa37a4255a78f2f224f3cd2f8fa99",

    SPAM_DIR
    / "0418.89cb8cbdd1cd4424829658e11ec6a13e",
]


def get_value(
    email_data,
    field,
    default="",
):
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
    email_data,
):
    """Build the same text representation used by prediction."""

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


def print_indicator(
    name,
    value,
):
    """Print one security indicator."""

    print(
        f"{name:<28}: {value}"
    )


def analyze_file(
    email_path,
    predictor,
    expected_label,
):
    """Analyze one disagreement."""

    print("\n" + "=" * 78)

    print(
        "EMAIL SECURITY AI — V3 FAILURE ANALYSIS"
    )

    print("=" * 78)

    print(
        f"\nFile:\n  {email_path}"
    )

    print(
        f"\nExpected dataset label:\n  "
        f"{expected_label.upper()}"
    )

    if not email_path.exists():
        print(
            "\nERROR: File does not exist."
        )
        return

    print(
        "\nParsing email..."
    )

    email_data = parse_email(
        email_path
    )

    email_text = build_email_text(
        email_data
    )

    print(
        "Running frozen ML model..."
    )

    ml_result = predictor.predict(
        email_text
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

    html_body = str(
        get_value(
            email_data,
            "html_body",
            "",
        )
        or ""
    )

    attachments = get_value(
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
        "Running V3 decision engine..."
    )

    decision = make_decision_v3(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    print("\n" + "-" * 78)
    print("EMAIL INFORMATION")
    print("-" * 78)

    print(
        f"Sender       : "
        f"{get_value(email_data, 'sender', '')}"
    )

    print(
        f"Receiver     : "
        f"{get_value(email_data, 'receiver', '')}"
    )

    print(
        f"Subject      : "
        f"{subject}"
    )

    print(
        f"Date         : "
        f"{get_value(email_data, 'date', '')}"
    )

    print(
        f"Attachments  : "
        f"{attachments}"
    )

    print("\n" + "-" * 78)
    print("MACHINE LEARNING RESULT")
    print("-" * 78)

    print(
        f"ML label     : "
        f"{ml_result.get('label')}"
    )

    print(
        f"ML spam      : "
        f"{ml_result.get('spam')}"
    )

    print(
        f"ML score     : "
        f"{float(ml_result.get('decision_score', 0.0)):.4f}"
    )

    print(
        f"Threshold    : "
        f"{float(ml_result.get('threshold', 0.0)):.4f}"
    )

    print("\n" + "-" * 78)
    print("SECURITY INDICATORS")
    print("-" * 78)

    indicators = [
        (
            "Subject length",
            get_value(
                security_analysis,
                "subject_length",
                0,
            ),
        ),
        (
            "Body length",
            get_value(
                security_analysis,
                "body_length",
                0,
            ),
        ),
        (
            "Word count",
            get_value(
                security_analysis,
                "word_count",
                0,
            ),
        ),
        (
            "URL count",
            get_value(
                security_analysis,
                "url_count",
                0,
            ),
        ),
        (
            "IP-based URL count",
            get_value(
                security_analysis,
                "ip_based_url_count",
                0,
            ),
        ),
        (
            "Suspicious keywords",
            get_value(
                security_analysis,
                "suspicious_keywords",
                0,
            ),
        ),
        (
            "HTML present",
            get_value(
                security_analysis,
                "html_present",
                False,
            ),
        ),
        (
            "Script present",
            get_value(
                security_analysis,
                "script_present",
                False,
            ),
        ),
        (
            "Attachments",
            get_value(
                security_analysis,
                "attachments",
                0,
            ),
        ),
        (
            "Exclamation marks",
            get_value(
                security_analysis,
                "exclamation_marks",
                0,
            ),
        ),
        (
            "Uppercase ratio",
            get_value(
                security_analysis,
                "uppercase_ratio",
                0.0,
            ),
        ),
    ]

    for name, value in indicators:
        print_indicator(
            name,
            value,
        )

    print("\n" + "-" * 78)
    print("RISK ASSESSMENT")
    print("-" * 78)

    print(
        f"Risk score   : "
        f"{get_value(risk_assessment, 'risk_score', 0)}/100"
    )

    print(
        f"Risk level   : "
        f"{get_value(risk_assessment, 'risk_level', 'UNKNOWN')}"
    )

    reasons = get_value(
        risk_assessment,
        "reasons",
        [],
    )

    if reasons:
        print(
            "\nRisk reasons:"
        )

        for index, reason in enumerate(
            reasons,
            start=1,
        ):
            print(
                f"  {index}. {reason}"
            )

    print("\n" + "-" * 78)
    print("V3 FINAL DECISION")
    print("-" * 78)

    print(
        f"Classification : "
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

    evidence = getattr(
        decision,
        "evidence",
        "UNKNOWN",
    )

    print(
        f"Evidence       : "
        f"{evidence}"
    )

    print(
        "\nReason:"
    )

    print(
        f"  {decision.reason}"
    )

    print("\n" + "-" * 78)

    print(
        "ANALYSIS"
    )

    print("-" * 78)

    ml_label = str(
        ml_result.get(
            "label",
            "unknown",
        )
    ).lower()

    final_classification = str(
        decision.classification
    ).upper()

    expected = expected_label.upper()

    print(
        f"Expected dataset class : {expected}"
    )

    print(
        f"ML prediction          : {ml_label.upper()}"
    )

    print(
        f"V3 final classification : {final_classification}"
    )

    if (
        ml_label == "ham"
        and final_classification == "SPAM"
    ):
        print(
            "\nType: FALSE-POSITIVE ESCALATION"
        )

        print(
            "The ML model classified the email as HAM, "
            "but security evidence caused V3 to escalate it."
        )

    elif (
        ml_label == "ham"
        and final_classification == "SUSPICIOUS"
    ):
        print(
            "\nType: SECURITY ESCALATION"
        )

        print(
            "The ML model classified the email as HAM, "
            "but V3 detected elevated security risk."
        )

    elif (
        ml_label == "spam"
        and final_classification == "SPAM"
    ):
        print(
            "\nType: ML-CONFIRMED SPAM"
        )

        print(
            "Both the ML classifier and V3 security layer "
            "classified the email as SPAM."
        )

    else:
        print(
            "\nType: OTHER DECISION PATH"
        )

    print("=" * 78)


def main():
    """Run V3 failure analysis."""

    print(
        "\n" + "=" * 78
    )

    print(
        "EMAIL SECURITY AI — DECISION ENGINE V3 FAILURE ANALYSIS"
    )

    print(
        "=" * 78
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    for email_path in TARGET_FILES:

        if "easy_ham" in str(
            email_path
        ):
            expected_label = "ham"
        else:
            expected_label = "spam"

        analyze_file(
            email_path=email_path,
            predictor=predictor,
            expected_label=expected_label,
        )

    print(
        "\n" + "=" * 78
    )

    print(
        "V3 FAILURE ANALYSIS COMPLETED"
    )

    print(
        "=" * 78
    )

    print(
        "\nNo model artifacts were modified."
    )

    print(
        "No threshold was changed."
    )

    print(
        "No retraining was performed."
    )


if __name__ == "__main__":
    main()
