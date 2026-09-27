"""
Production decision layer for Email Security AI.

Combines:

    ML classification
        +
    Security indicators
        +
    Risk assessment
        ↓
    Final security decision

The frozen ML model and threshold are not modified.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionResult:
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


def _get_value(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _normalize_ml_result(
    ml_result: dict,
) -> tuple[str, bool, float, float]:

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

    return _get_value(
        security_analysis,
        name,
        default,
    )


def make_decision(
    ml_result: dict,
    security_analysis: Any,
    risk_assessment: Any,
) -> DecisionResult:

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

    body_length = int(
        _get_security_indicator(
            security_analysis,
            "body_length",
            0,
        )
    )

    subject_length = int(
        _get_security_indicator(
            security_analysis,
            "subject_length",
            0,
        )
    )

    strong_security_evidence = (
        script_present
        or ip_url_count > 0
        or suspicious_keywords >= 3
        or risk_score >= 70
    )

    moderate_security_evidence = (
        url_count >= 3
        or suspicious_keywords >= 1
        or attachment_count > 0
        or risk_score >= 40
    )

    benign_short_email = (
        body_length <= 250
        and subject_length <= 100
        and url_count == 0
        and ip_url_count == 0
        and suspicious_keywords == 0
        and not html_present
        and not script_present
        and attachment_count == 0
        and risk_score < 20
    )

    if strong_security_evidence:
        classification = "SPAM"
        action = "BLOCK / REVIEW"
        confidence = "HIGH"

        reason = (
            "Strong security indicators were detected "
            "independently of the ML classification."
        )

    elif ml_spam and benign_short_email:
        classification = "HAM"
        action = "ALLOW"
        confidence = "MEDIUM"

        reason = (
            "The ML model classified the message as spam, "
            "but the message is short and contains no "
            "independent security indicators."
        )

    elif ml_spam and moderate_security_evidence:
        classification = "SPAM"
        action = "QUARANTINE / REVIEW"
        confidence = "HIGH"

        reason = (
            "The ML model classified the message as spam "
            "and additional security indicators support "
            "the classification."
        )

    elif ml_spam:
        classification = "SPAM"
        action = "REVIEW"
        confidence = "MEDIUM"

        reason = (
            "The ML model classified the message as spam, "
            "but independent security evidence is limited."
        )

    elif moderate_security_evidence and risk_score >= 40:
        classification = "SUSPICIOUS"
        action = "QUARANTINE / REVIEW"
        confidence = "MEDIUM"

        reason = (
            "The ML model classified the message as ham, "
            "but security indicators indicate elevated risk."
        )

    else:
        classification = "HAM"
        action = "ALLOW"
        confidence = "HIGH"

        reason = (
            "The ML model classified the message as ham "
            "and no significant security indicators were detected."
        )

    return DecisionResult(
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
    )


def print_decision(
    decision: DecisionResult,
) -> None:

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — FINAL DECISION")
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

    print("\nReason:")

    print(
        f"  {decision.reason}"
    )

    print("\n" + "=" * 70)


__all__ = [
    "DecisionResult",
    "make_decision",
    "print_decision",
]
