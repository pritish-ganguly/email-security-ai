"""
Production Decision Engine V2 for Email Security AI.

Decision Engine V2 combines:

    ML classification
        +
    Security indicators
        +
    Risk assessment
        +
    Evidence consistency
        +
    ML confidence margin
        ↓
    Final security decision

Important:
    - The frozen ML model is NOT modified.
    - The frozen ML threshold is NOT modified.
    - No retraining occurs here.
    - This module only improves the decision layer.
"""

from dataclasses import dataclass
from typing import Any


HIGH_RISK_SCORE = 70
MEDIUM_RISK_SCORE = 40

STRONG_SUSPICIOUS_KEYWORDS = 3
MODERATE_SUSPICIOUS_KEYWORDS = 1

MULTIPLE_URL_THRESHOLD = 3

ML_BORDERLINE_MARGIN = 0.10


@dataclass
class DecisionResultV2:
    """Final decision returned by Decision Engine V2."""

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
    ml_margin: float


def _get_value(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a value from either:

        - dictionary
        - dataclass/object
    """

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


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


def _normalize_risk(
    risk_assessment: Any,
) -> tuple[int, str]:
    """Normalize risk assessment information."""

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


def _get_security_indicators(
    security_analysis: Any,
) -> dict:
    """Extract normalized security indicators."""

    return {
        "html_present": bool(
            _get_value(
                security_analysis,
                "html_present",
                False,
            )
        ),

        "script_present": bool(
            _get_value(
                security_analysis,
                "script_present",
                False,
            )
        ),

        "url_count": int(
            _get_value(
                security_analysis,
                "url_count",
                0,
            )
        ),

        "ip_based_url_count": int(
            _get_value(
                security_analysis,
                "ip_based_url_count",
                0,
            )
        ),

        "suspicious_keywords": int(
            _get_value(
                security_analysis,
                "suspicious_keywords",
                0,
            )
        ),

        "attachments": int(
            _get_value(
                security_analysis,
                "attachments",
                0,
            )
        ),

        "body_length": int(
            _get_value(
                security_analysis,
                "body_length",
                0,
            )
        ),

        "subject_length": int(
            _get_value(
                security_analysis,
                "subject_length",
                0,
            )
        ),
    }


def _classify_evidence(
    indicators: dict,
    risk_score: int,
) -> str:
    """
    Classify independent security evidence as:

        STRONG
        MODERATE
        WEAK
    """

    strong = (
        indicators["script_present"]
        or indicators["ip_based_url_count"] > 0
        or indicators["suspicious_keywords"]
        >= STRONG_SUSPICIOUS_KEYWORDS
        or risk_score >= HIGH_RISK_SCORE
    )

    if strong:
        return "STRONG"

    moderate = (
        indicators["url_count"]
        >= MULTIPLE_URL_THRESHOLD
        or indicators["suspicious_keywords"]
        >= MODERATE_SUSPICIOUS_KEYWORDS
        or indicators["attachments"] > 0
        or risk_score >= MEDIUM_RISK_SCORE
    )

    if moderate:
        return "MODERATE"

    return "WEAK"


def _is_benign_profile(
    indicators: dict,
) -> bool:
    """
    Determine whether the email has a clean benign profile.
    """

    return (
        indicators["body_length"] <= 250
        and indicators["subject_length"] <= 100
        and indicators["url_count"] == 0
        and indicators["ip_based_url_count"] == 0
        and indicators["suspicious_keywords"] == 0
        and not indicators["html_present"]
        and not indicators["script_present"]
        and indicators["attachments"] == 0
    )


def _calculate_ml_margin(
    score: float,
    threshold: float,
) -> float:
    """
    Calculate distance from the production threshold.
    """

    return abs(
        score - threshold
    )


def make_decision_v2(
    ml_result: dict,
    security_analysis: Any,
    risk_assessment: Any,
) -> DecisionResultV2:
    """
    Produce the final V2 security decision.

    The V2 engine does not modify the underlying ML model.

    It evaluates:

        1. ML classification
        2. Distance from ML threshold
        3. Security indicators
        4. Risk score
        5. Evidence consistency
        6. Benign-message characteristics
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
    ) = _normalize_risk(
        risk_assessment
    )

    indicators = _get_security_indicators(
        security_analysis
    )

    evidence_level = _classify_evidence(
        indicators,
        risk_score,
    )

    ml_margin = _calculate_ml_margin(
        ml_score,
        threshold,
    )

    benign_profile = _is_benign_profile(
        indicators
    )


    if evidence_level == "STRONG":

        classification = "SPAM"
        action = "BLOCK / REVIEW"
        confidence = "HIGH"

        reason = (
            "Strong independent security indicators "
            "were detected. These indicators justify "
            "escalation regardless of the ML prediction."
        )


    elif ml_spam:


        if evidence_level == "MODERATE":

            classification = "SPAM"
            action = "QUARANTINE / REVIEW"
            confidence = "HIGH"

            reason = (
                "The ML model classified the email as spam "
                "and moderate security evidence supports "
                "the classification."
            )


        elif (
            benign_profile
            and risk_score < MEDIUM_RISK_SCORE
            and ml_margin < ML_BORDERLINE_MARGIN
        ):

            classification = "SUSPICIOUS"
            action = "REVIEW"
            confidence = "MEDIUM"

            reason = (
                "The ML model classified the email as spam, "
                "but the message has a benign structural profile "
                "and the ML score is close to the production "
                "threshold."
            )


        else:

            classification = "SPAM"
            action = "REVIEW"
            confidence = "MEDIUM"

            reason = (
                "The ML model classified the email as spam, "
                "while independent security evidence is limited."
            )


    else:


        if (
            evidence_level == "MODERATE"
            and risk_score >= MEDIUM_RISK_SCORE
        ):

            classification = "SUSPICIOUS"
            action = "QUARANTINE / REVIEW"
            confidence = "MEDIUM"

            reason = (
                "The ML model classified the email as ham, "
                "but independent security indicators indicate "
                "elevated risk."
            )


        elif (
            evidence_level == "MODERATE"
            and ml_margin < ML_BORDERLINE_MARGIN
        ):

            classification = "SUSPICIOUS"
            action = "QUARANTINE / REVIEW"
            confidence = "MEDIUM"

            reason = (
                "The ML classification is close to the production "
                "threshold and additional security indicators "
                "suggest that the message requires review."
            )


        else:

            classification = "HAM"
            action = "ALLOW"
            confidence = "HIGH"

            reason = (
                "The ML model classified the email as ham "
                "and no sufficiently strong independent security "
                "evidence was detected."
            )

    return DecisionResultV2(
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
        ml_margin=ml_margin,
    )


def print_decision_v2(
    decision: DecisionResultV2,
) -> None:
    """Print a human-readable V2 decision."""

    print("\n" + "=" * 70)
    print(
        "EMAIL SECURITY AI — FINAL DECISION V2"
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
        f"ML margin      : "
        f"{decision.ml_margin:.4f}"
    )

    print(
        f"Evidence level : "
        f"{decision.evidence_level}"
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
    "DecisionResultV2",
    "make_decision_v2",
    "print_decision_v2",
]
