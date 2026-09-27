"""
Email Security AI — Risk Assessment Engine.

Combines machine-learning classification with security indicators
to produce an explainable email risk assessment.

The risk engine does not retrain or modify the ML model.
It provides a separate, deterministic security assessment.
"""

from dataclasses import dataclass
from typing import Any


MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 100


@dataclass(frozen=True)
class RiskAssessment:
    """Final security risk assessment for an email."""

    risk_score: int
    risk_level: str
    reasons: list[str]
    recommended_action: str


def _get_value(
    source: object,
    field: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a value from either a dictionary or an object.

    The security analyzer currently returns a SecurityAnalysis object,
    while some future components may return dictionaries.
    """

    if source is None:
        return default

    if isinstance(source, dict):
        return source.get(
            field,
            default,
        )

    return getattr(
        source,
        field,
        default,
    )


def _get_indicator(
    indicators: object,
    *names: str,
    default: Any = 0,
) -> Any:
    """
    Retrieve an indicator using multiple possible field names.

    Supports both dictionaries and dataclass/object instances.
    """

    for name in names:

        value = _get_value(
            indicators,
            name,
            None,
        )

        if value is not None:
            return value

    return default


def _clamp_score(
    score: float,
) -> int:
    """Keep the risk score within the supported range."""

    return int(
        max(
            MIN_RISK_SCORE,
            min(
                MAX_RISK_SCORE,
                round(score),
            ),
        )
    )


def calculate_risk_score(
    ml_result: object,
    security_analysis: object,
) -> tuple[int, list[str]]:
    """
    Calculate an explainable email-security risk score.

    ML classification and security indicators are treated as
    separate sources of evidence.

    Returns:
        Tuple containing:

            risk score
            reasons contributing to the score
    """

    score = 0

    reasons: list[str] = []


    is_spam = bool(
        _get_value(
            ml_result,
            "is_spam",
            _get_value(
                ml_result,
                "spam",
                False,
            ),
        )
    )

    if is_spam:

        score += 50

        reasons.append(
            "The machine-learning model classified "
            "the email as spam."
        )


    url_count = int(
        _get_indicator(
            security_analysis,
            "url_count",
            "url_count_total",
            default=0,
        )
        or 0
    )

    ip_url_count = int(
        _get_indicator(
            security_analysis,
            "ip_url_count",
            "ip_based_url_count",
            default=0,
        )
        or 0
    )

    suspicious_keyword_count = int(
        _get_indicator(
            security_analysis,
            "suspicious_keyword_count",
            "suspicious_keywords",
            default=0,
        )
        or 0
    )

    html_present = bool(
        _get_indicator(
            security_analysis,
            "html_present",
            default=False,
        )
    )

    script_present = bool(
        _get_indicator(
            security_analysis,
            "script_present",
            default=False,
        )
    )

    attachment_count = int(
        _get_indicator(
            security_analysis,
            "attachment_count",
            "attachments_count",
            default=0,
        )
        or 0
    )

    exclamation_count = int(
        _get_indicator(
            security_analysis,
            "exclamation_count",
            default=0,
        )
        or 0
    )

    uppercase_ratio = float(
        _get_indicator(
            security_analysis,
            "uppercase_ratio",
            default=0.0,
        )
        or 0.0
    )


    if url_count > 0:

        points = min(
            url_count * 3,
            15,
        )

        score += points

        reasons.append(
            f"Email contains {url_count} URL(s)."
        )


    if ip_url_count > 0:

        points = min(
            ip_url_count * 10,
            25,
        )

        score += points

        reasons.append(
            f"Email contains {ip_url_count} URL(s) "
            "using an IP address."
        )


    if suspicious_keyword_count > 0:

        points = min(
            suspicious_keyword_count * 5,
            25,
        )

        score += points

        reasons.append(
            "Suspicious security-related keywords "
            f"were detected ({suspicious_keyword_count})."
        )


    if html_present:

        score += 5

        reasons.append(
            "Email contains HTML content."
        )


    if script_present:

        score += 25

        reasons.append(
            "Script content was detected in the email."
        )


    if attachment_count > 0:

        points = min(
            attachment_count * 8,
            24,
        )

        score += points

        reasons.append(
            f"Email contains {attachment_count} attachment(s)."
        )


    if exclamation_count >= 5:

        score += 5

        reasons.append(
            "Email contains an unusually high number "
            "of exclamation marks."
        )


    if uppercase_ratio >= 0.30:

        score += 5

        reasons.append(
            "Email contains a high proportion "
            "of uppercase text."
        )


    score = _clamp_score(
        score
    )

    return score, reasons


def determine_risk_level(
    risk_score: int,
) -> str:
    """
    Convert the numeric risk score into a risk level.
    """

    if risk_score >= 75:
        return "CRITICAL"

    if risk_score >= 50:
        return "HIGH"

    if risk_score >= 25:
        return "MEDIUM"

    return "LOW"


def determine_recommended_action(
    risk_level: str,
    is_spam: bool,
) -> str:
    """
    Determine the recommended handling action.
    """

    if risk_level == "CRITICAL":

        return (
            "Quarantine the email and require security "
            "review before delivery."
        )

    if risk_level == "HIGH":

        return (
            "Treat the email as suspicious and route it "
            "for security review."
        )

    if risk_level == "MEDIUM":

        return (
            "Allow with caution and display security "
            "warnings to the recipient."
        )

    if is_spam:

        return (
            "Apply normal spam handling and monitor "
            "the message."
        )

    return (
        "No elevated security action is required based "
        "on the available indicators."
    )


def assess_email_risk(
    ml_result: object,
    security_analysis: object,
) -> RiskAssessment:
    """
    Produce the complete explainable security assessment.
    """

    risk_score, reasons = calculate_risk_score(
        ml_result,
        security_analysis,
    )

    risk_level = determine_risk_level(
        risk_score
    )

    is_spam = bool(
        _get_value(
            ml_result,
            "is_spam",
            _get_value(
                ml_result,
                "spam",
                False,
            ),
        )
    )

    recommended_action = determine_recommended_action(
        risk_level,
        is_spam,
    )

    if not reasons:

        reasons.append(
            "No elevated security indicators were detected."
        )

    return RiskAssessment(
        risk_score=risk_score,
        risk_level=risk_level,
        reasons=reasons,
        recommended_action=recommended_action,
    )


def print_risk_assessment(
    assessment: RiskAssessment,
) -> None:
    """Print the risk assessment in a readable format."""

    print("\n" + "=" * 70)

    print(
        "EMAIL SECURITY AI — RISK ASSESSMENT"
    )

    print("=" * 70)

    print(
        f"\nRisk score   : "
        f"{assessment.risk_score}/100"
    )

    print(
        f"Risk level   : "
        f"{assessment.risk_level}"
    )

    print("\nSecurity reasons:")
    print("-" * 70)

    for index, reason in enumerate(
        assessment.reasons,
        start=1,
    ):

        print(
            f"{index}. {reason}"
        )

    print("\nRecommended action:")
    print("-" * 70)

    print(
        assessment.recommended_action
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":

    example_ml_result = {
        "label": "spam",
        "spam": True,
        "is_spam": True,
        "decision_score": 0.91,
        "score": 0.91,
        "threshold": -0.0975,
    }

    example_security_analysis = {
        "url_count": 3,
        "ip_url_count": 1,
        "suspicious_keyword_count": 2,
        "html_present": True,
        "script_present": False,
        "attachment_count": 1,
        "exclamation_count": 8,
        "uppercase_ratio": 0.12,
    }

    assessment = assess_email_risk(
        ml_result=example_ml_result,
        security_analysis=example_security_analysis,
    )

    print_risk_assessment(
        assessment
    )
