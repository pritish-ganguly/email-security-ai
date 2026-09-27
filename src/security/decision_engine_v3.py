"""
Production Decision Engine V3 for Email Security AI.

Combines:
    ML classification
    +
    Security indicators
    +
    Risk assessment
    +
    Evidence strength

Produces:
    Final classification
    Final security action
    Confidence
    Explanation

IMPORTANT:
    This module must NOT import itself.
    The frozen ML model and threshold are never modified.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionResultV3:
    """Final decision returned by Decision Engine V3."""

    classification: str
    action: str
    confidence: str
    reason: str

    ml_label: str
    ml_spam: bool
    ml_score: float
    threshold: float

    risk_score: int
    risk_level: str

    evidence_level: str


def _get_value(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """Safely retrieve a value from a dictionary or object."""

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _normalize_ml_result(
    ml_result: dict,
) -> tuple[str, bool, float, float]:
    """Normalize ML prediction information."""

    label = str(
        ml_result.get(
            "label",
            "unknown",
        )
    ).lower()

    spam = bool(
        ml_result.get(
            "spam",
            label == "spam",
        )
    )

    score = float(
        ml_result.get(
            "decision_score",
            0.0,
        )
    )

    threshold = float(
        ml_result.get(
            "threshold",
            0.0,
        )
    )

    return (
        label,
        spam,
        score,
        threshold,
    )


def _get_risk_information(
    risk_assessment: Any,
) -> tuple[int, str]:
    """Extract risk score and risk level."""

    risk_score = int(
        _get_value(
            risk_assessment,
            "risk_score",
            0,
        )
    )

    risk_level = str(
        _get_value(
            risk_assessment,
            "risk_level",
            "LOW",
        )
    ).upper()

    return (
        risk_score,
        risk_level,
    )


def _get_security_indicator(
    security_analysis: Any,
    name: str,
    default: Any = 0,
) -> Any:
    """Safely retrieve a security indicator."""

    return _get_value(
        security_analysis,
        name,
        default,
    )


def _classify_evidence(
    *,
    script_present: bool,
    ip_url_count: int,
    suspicious_keywords: int,
    risk_score: int,
    url_count: int,
    attachment_count: int,
    html_present: bool,
) -> str:
    """
    Classify independent security evidence.

    STRONG:
        Clear high-risk technical indicators.

    MODERATE:
        Some security indicators exist.

    WEAK:
        Little or no independent security evidence.
    """

    strong = (
        script_present
        or ip_url_count > 0
        or suspicious_keywords >= 3
        or risk_score >= 70
    )

    if strong:
        return "STRONG"

    moderate = (
        url_count >= 3
        or suspicious_keywords >= 1
        or attachment_count > 0
        or html_present
        or risk_score >= 30
    )

    if moderate:
        return "MODERATE"

    return "WEAK"


def make_decision_v3(
    ml_result: dict,
    security_analysis: Any,
    risk_assessment: Any,
) -> DecisionResultV3:
    """
    Produce the final V3 email-security decision.

    The ML model remains the primary classification signal.

    Independent security evidence can escalate a message,
    but weak security evidence does not override a strong
    ML classification.

    No retraining occurs here.
    No ML threshold modification occurs here.
    """

    (
        ml_label,
        ml_spam,
        ml_score,
        threshold,
    ) = _normalize_ml_result(
        ml_result
    )

    (
        risk_score,
        risk_level,
    ) = _get_risk_information(
        risk_assessment
    )


    html_present = bool(
        _get_security_indicator(
            security_analysis,
            "html_present",
            False,
        )
    )

    script_present = bool(
        _get_security_indicator(
            security_analysis,
            "script_present",
            False,
        )
    )

    url_count = int(
        _get_security_indicator(
            security_analysis,
            "url_count",
            0,
        )
    )

    ip_url_count = int(
        _get_security_indicator(
            security_analysis,
            "ip_based_url_count",
            0,
        )
    )

    suspicious_keywords = int(
        _get_security_indicator(
            security_analysis,
            "suspicious_keywords",
            0,
        )
    )

    attachment_count = int(
        _get_security_indicator(
            security_analysis,
            "attachments",
            0,
        )
    )


    evidence_level = _classify_evidence(
        script_present=script_present,
        ip_url_count=ip_url_count,
        suspicious_keywords=suspicious_keywords,
        risk_score=risk_score,
        url_count=url_count,
        attachment_count=attachment_count,
        html_present=html_present,
    )


    if evidence_level == "STRONG":

        classification = "SPAM"
        action = "BLOCK / REVIEW"
        confidence = "HIGH"

        if ml_spam:
            reason = (
                "The ML model classified the message as spam "
                "and strong independent security indicators "
                "support the classification."
            )
        else:
            reason = (
                "The ML model classified the message as ham, "
                "but strong independent security indicators "
                "require security escalation."
            )


    elif ml_spam:

        classification = "SPAM"

        if evidence_level == "MODERATE":
            action = "QUARANTINE / REVIEW"
            confidence = "HIGH"

            reason = (
                "The ML model classified the message as spam "
                "and additional security indicators provide "
                "supporting evidence."
            )

        else:
            action = "REVIEW"
            confidence = "MEDIUM"

            reason = (
                "The ML model classified the message as spam, "
                "while independent security evidence is limited."
            )


    elif evidence_level == "MODERATE":

        if risk_score >= 40:

            classification = "SUSPICIOUS"
            action = "QUARANTINE / REVIEW"
            confidence = "MEDIUM"

            reason = (
                "The ML model classified the message as ham, "
                "but moderate security evidence combined with "
                "elevated risk requires review."
            )

        else:

            classification = "HAM"
            action = "ALLOW"
            confidence = "MEDIUM"

            reason = (
                "The ML model classified the message as ham. "
                "Some security indicators were detected, but "
                "the overall risk does not justify escalation."
            )


    else:

        classification = "HAM"
        action = "ALLOW"
        confidence = "HIGH"

        reason = (
            "The ML model classified the message as ham and "
            "independent security evidence is weak."
        )

    return DecisionResultV3(
        classification=classification,
        action=action,
        confidence=confidence,
        reason=reason,
        ml_label=ml_label,
        ml_spam=ml_spam,
        ml_score=ml_score,
        threshold=threshold,
        risk_score=risk_score,
        risk_level=risk_level,
        evidence_level=evidence_level,
    )


def print_decision_v3(
    decision: DecisionResultV3,
) -> None:
    """Print the V3 decision in a readable format."""

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — FINAL V3 DECISION")
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
        f"ML prediction  : "
        f"{decision.ml_label}"
    )

    print(
        f"ML score       : "
        f"{decision.ml_score:.4f}"
    )

    print(
        f"ML threshold   : "
        f"{decision.threshold:.4f}"
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
        f"Evidence       : "
        f"{decision.evidence_level}"
    )

    print("\nReason:")
    print(
        f"  {decision.reason}"
    )

    print("\n" + "=" * 70)


__all__ = [
    "DecisionResultV3",
    "make_decision_v3",
    "print_decision_v3",
]
