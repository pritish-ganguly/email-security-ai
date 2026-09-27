from pathlib import Path
import argparse

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
    print_decision_v31,
)


def get_value(
    email_data,
    field,
    default="",
):
    """
    Safely retrieve a field from either a dictionary
    or an object/dataclass.
    """

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
    """
    Build the same text representation used by
    the production ML inference pipeline.
    """

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


def print_header(
    title,
):
    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )


def scan_email(
    email_path: Path,
    predictor: EmailPredictor,
):
    """
    Run one email through the complete
    Email Security AI V3.1 pipeline.

    Pipeline:

        Email file
            ↓
        Email parser
            ↓
        Frozen ML model
            ↓
        Security analyzer
            ↓
        Risk engine
            ↓
        Decision Engine V3.1
            ↓
        Final security decision
    """


    if not email_path.exists():
        raise FileNotFoundError(
            f"Email file not found: {email_path}"
        )

    if not email_path.is_file():
        raise ValueError(
            f"Input path is not a file: {email_path}"
        )


    print(
        "Parsing email file..."
    )

    email_data = parse_email(
        email_path
    )


    email_text = build_email_text(
        email_data
    )

    if not email_text.strip():
        raise ValueError(
            "The email does not contain usable text."
        )


    print(
        "Running ML prediction..."
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
        "Running Decision Engine V3.1..."
    )

    decision = make_decision_v31(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )


    print_header(
        "EMAIL SECURITY AI — EMAIL SECURITY ANALYSIS"
    )


    print(
        "\nEmail information"
    )

    print(
        "-" * 70
    )

    print(
        f"\nFile:\n"
        f"  {email_path}"
    )

    print(
        f"\nSender:\n"
        f"  {get_value(email_data, 'sender', '')}"
    )

    print(
        f"\nReceiver:\n"
        f"  {get_value(email_data, 'receiver', '')}"
    )

    print(
        f"\nSubject:\n"
        f"  {subject}"
    )

    print(
        f"\nDate:\n"
        f"  {get_value(email_data, 'date', '')}"
    )

    print(
        f"\nAttachments:\n"
        f"  {attachments}"
    )


    print_header(
        "MACHINE LEARNING RESULT"
    )

    print(
        f"\nPrediction : "
        f"{ml_result['label']}"
    )

    print(
        f"Spam       : "
        f"{ml_result['spam']}"
    )

    print(
        f"Score      : "
        f"{ml_result['decision_score']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{ml_result['threshold']:.4f}"
    )


    print_security_analysis(
        security_analysis
    )


    print_risk_assessment(
        risk_assessment
    )


    print_decision_v31(
        decision
    )


    print_header(
        "SCAN COMPLETED"
    )

    print(
        "\nEmail Security AI V3.1 "
        "analysis completed successfully."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Email Security AI V3.1 "
            "production email scanner"
        )
    )

    parser.add_argument(
        "email_file",
        type=Path,
        help=(
            "Path to an email message file"
        ),
    )

    args = parser.parse_args()

    try:


        predictor = EmailPredictor()


        scan_email(
            email_path=args.email_file,
            predictor=predictor,
        )

    except FileNotFoundError as exc:

        print(
            f"\nERROR: {exc}"
        )

        raise SystemExit(1)

    except ValueError as exc:

        print(
            f"\nERROR: {exc}"
        )

        raise SystemExit(1)

    except Exception as exc:

        print(
            f"\nERROR: Email scanning failed: {exc}"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
