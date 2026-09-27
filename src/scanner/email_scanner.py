from pathlib import Path

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine_v31 import make_decision_v31


def get_value(email_data, field, default=""):
    if isinstance(email_data, dict):
        return email_data.get(field, default)

    return getattr(email_data, field, default)


def build_email_text(email_data):
    subject = str(get_value(email_data, "subject", "") or "")
    body = str(get_value(email_data, "body", "") or "")
    html_body = str(get_value(email_data, "html_body", "") or "")

    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    )


class EmailScanner:
    def __init__(self, predictor=None):
        self.predictor = predictor or EmailPredictor()

    def scan(self, email_path):
        email_path = Path(email_path)

        if not email_path.exists():
            raise FileNotFoundError(
                f"Email file not found: {email_path}"
            )

        if not email_path.is_file():
            raise ValueError(
                f"Input path is not a file: {email_path}"
            )

        email_data = parse_email(email_path)
        email_text = build_email_text(email_data)

        if not email_text.strip():
            raise ValueError(
                "The email does not contain usable text."
            )

        ml_result = self.predictor.predict(email_text)

        subject = str(
            get_value(email_data, "subject", "") or ""
        )
        body = str(
            get_value(email_data, "body", "") or ""
        )
        html_body = str(
            get_value(email_data, "html_body", "") or ""
        )
        attachments = get_value(
            email_data,
            "attachments",
            [],
        )

        security_analysis = analyze_email(
            subject=subject,
            body=body,
            html_body=html_body,
            attachments=attachments,
        )

        risk_assessment = assess_email_risk(
            ml_result=ml_result,
            security_analysis=security_analysis,
        )

        decision = make_decision_v31(
            ml_result=ml_result,
            security_analysis=security_analysis,
            risk_assessment=risk_assessment,
        )

        return {
            "email": {
                "file": str(email_path),
                "sender": get_value(email_data, "sender", ""),
                "receiver": get_value(email_data, "receiver", ""),
                "subject": subject,
                "date": get_value(email_data, "date", ""),
                "attachments": attachments,
            },
            "ml_result": ml_result,
            "security_analysis": security_analysis,
            "risk_assessment": risk_assessment,
            "decision": decision,
        }


def scan_email(email_path, predictor=None):
    return EmailScanner(
        predictor=predictor
    ).scan(email_path)