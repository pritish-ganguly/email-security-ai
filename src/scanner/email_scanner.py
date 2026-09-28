from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.decision_engine_v31 import make_decision_v31
from src.security.risk_engine import assess_email_risk
from src.security.security_analyzer import analyze_email


def get_value(
    email_data: object,
    field: str,
    default: Any = "",
) -> Any:
    """Safely retrieve a field from a dictionary or object."""

    if isinstance(email_data, dict):
        return email_data.get(field, default)

    return getattr(email_data, field, default)


def build_email_text(email_data: object) -> str:
    """Build the text representation used by the ML model."""

    subject = str(
        get_value(email_data, "subject", "") or ""
    )

    body = str(
        get_value(email_data, "body", "") or ""
    )

    html_body = str(
        get_value(email_data, "html_body", "") or ""
    )

    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    ).strip()


class EmailScanner:
    """
    Production email scanning pipeline.

    Pipeline:

        Email
          ↓
        Parser
          ↓
        Text construction
          ↓
        Frozen ML model
          ↓
        Security analysis
          ↓
        Risk assessment
          ↓
        Decision Engine V3.1
          ↓
        Final result
    """

    def __init__(
        self,
        predictor: EmailPredictor | None = None,
    ) -> None:
        self.predictor = predictor or EmailPredictor()

    def scan(self, email_path: str | Path) -> dict[str, Any]:
        """
        Scan an .eml file through the complete security pipeline.
        """

        path = Path(email_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Email file not found: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Input path is not a file: {path}"
            )

        # ---------------------------------------------------------
        # 1. Parse email
        # ---------------------------------------------------------

        email_data = parse_email(path)

        # ---------------------------------------------------------
        # 2. Extract email fields
        # ---------------------------------------------------------

        sender = str(
            get_value(email_data, "sender", "") or ""
        )

        receiver = str(
            get_value(email_data, "receiver", "") or ""
        )

        subject = str(
            get_value(email_data, "subject", "") or ""
        )

        date = get_value(
            email_data,
            "date",
            "",
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

        if attachments is None:
            attachments = []

        # ---------------------------------------------------------
        # 3. Build ML input
        # ---------------------------------------------------------

        email_text = build_email_text(email_data)

        if not email_text.strip():
            raise ValueError(
                "The email does not contain usable text."
            )

        # ---------------------------------------------------------
        # 4. Machine-learning prediction
        # ---------------------------------------------------------

        ml_result = self.predictor.predict(
            email_text
        )

        # ---------------------------------------------------------
        # 5. Security analysis
        # ---------------------------------------------------------

        security_result = analyze_email(
            subject=subject,
            body=body,
            html_body=html_body,
            attachments=attachments,
        )

        # Normalize SecurityAnalysis object → dictionary.
        if hasattr(security_result, "to_dict"):
            security_analysis = (
                security_result.to_dict()
            )
        elif isinstance(security_result, dict):
            security_analysis = security_result
        else:
            raise TypeError(
                "security_analysis must be a dictionary "
                "or provide a to_dict() method."
            )

        # ---------------------------------------------------------
        # 6. Risk assessment
        # ---------------------------------------------------------

        risk_result = assess_email_risk(
            ml_result=ml_result,
            security_analysis=security_analysis,
        )

        # Normalize RiskAssessment object → dictionary.
        if hasattr(risk_result, "to_dict"):
            risk_assessment = risk_result.to_dict()
        elif isinstance(risk_result, dict):
            risk_assessment = risk_result
        else:
            # RiskAssessment is currently a dataclass.
            risk_assessment = {
                "risk_score": getattr(
                    risk_result,
                    "risk_score",
                    0,
                ),
                "risk_level": getattr(
                    risk_result,
                    "risk_level",
                    "UNKNOWN",
                ),
                "reasons": getattr(
                    risk_result,
                    "reasons",
                    [],
                ),
                "recommended_action": getattr(
                    risk_result,
                    "recommended_action",
                    "",
                ),
            }

        # ---------------------------------------------------------
        # 7. Decision Engine V3.1
        # ---------------------------------------------------------

        decision = make_decision_v31(
            ml_result=ml_result,
            security_analysis=security_analysis,
            risk_assessment=risk_assessment,
        )

        # ---------------------------------------------------------
        # 8. Unified result
        # ---------------------------------------------------------

        return {
            "email": {
                "file": str(path),
                "sender": sender,
                "receiver": receiver,
                "subject": subject,
                "date": date,
                "attachments": attachments,
                "body": body,
                "html_body": html_body,
            },

            # Dashboard-friendly aliases
            "file_path": str(path),

            "email_data": {
                "file": str(path),
                "sender": sender,
                "receiver": receiver,
                "subject": subject,
                "date": date,
                "attachments": attachments,
                "body": body,
                "html_body": html_body,
            },

            "ml_result": ml_result,

            "security_analysis": security_analysis,

            "risk_assessment": risk_assessment,

            "decision": decision,
        }


def scan_email(
    email_path: str | Path,
    predictor: EmailPredictor | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around EmailScanner."""

    scanner = EmailScanner(
        predictor=predictor
    )

    return scanner.scan(email_path)