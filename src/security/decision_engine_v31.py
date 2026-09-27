"""
Email Security AI
Decision Engine V3.1

V3.1 goals:
    - Preserve strong ML spam decisions.
    - Allow weak-evidence spam to go to REVIEW.
    - Use security indicators to escalate genuinely risky HAM.
    - Avoid unnecessarily converting moderate-risk HAM into SPAM.
    - Keep the decision deterministic and explainable.

IMPORTANT:
    This file must NOT import itself.
"""

from typing import Any, Dict


def _get(data: Any, key: str, default=None):
    """
    Safely retrieve a value from either a dictionary or object.
    """

    if isinstance(data, dict):
        return data.get(key, default)

    return getattr(data, key, default)


def _normalise_label(label: Any) -> str:
    """
    Normalize ML classification labels.
    """

    return str(label or "").strip().lower()


def _normalise_level(level: Any) -> str:
    """
    Normalize risk levels.
    """

    return str(level or "").strip().upper()


def _security_evidence(
    security_analysis: Any,
    risk_assessment: Any,
) -> str:
    """
    Determine the strength of independent security evidence.

    STRONG:
        Multiple strong indicators or a strong combination.

    MODERATE:
        More than one moderate indicator.

    WEAK:
        Limited security evidence.

    The purpose is to prevent ordinary URLs/HTML from
    automatically overriding a strong HAM ML prediction.
    """

    url_count = int(
        _get(
            security_analysis,
            "url_count",
            0,
        )
        or 0
    )

    ip_url_count = int(
        _get(
            security_analysis,
            "ip_url_count",
            0,
        )
        or 0
    )

    suspicious_keywords = int(
        _get(
            security_analysis,
            "suspicious_keyword_count",
            0,
        )
        or 0
    )

    html_present = bool(
        _get(
            security_analysis,
            "html_present",
            False,
        )
    )

    script_present = bool(
        _get(
            security_analysis,
            "script_present",
            False,
        )
    )

    attachment_count = int(
        _get(
            security_analysis,
            "attachment_count",
            0,
        )
        or 0
    )

    risk_score = int(
        _get(
            risk_assessment,
            "risk_score",
            0,
        )
        or 0
    )

    strong_signals = 0
    moderate_signals = 0


    if script_present:
        strong_signals += 1

    if ip_url_count > 0:
        strong_signals += 1

    if suspicious_keywords >= 3:
        strong_signals += 1

    if (
        attachment_count > 0
        and (
            script_present
            or suspicious_keywords > 0
            or ip_url_count > 0
        )
    ):
        strong_signals += 1


    if url_count >= 3:
        moderate_signals += 1

    if html_present:
        moderate_signals += 1

    if suspicious_keywords > 0:
        moderate_signals += 1

    if attachment_count > 0:
        moderate_signals += 1


    if risk_score >= 60:
        strong_signals += 1

    elif risk_score >= 30:
        moderate_signals += 1


    if strong_signals >= 2:
        return "STRONG"

    if strong_signals == 1 and moderate_signals >= 1:
        return "STRONG"

    if moderate_signals >= 2:
        return "MODERATE"

    return "WEAK"


def _confidence(
    classification: str,
    ml_label: str,
    evidence: str,
    risk_level: str,
) -> str:
    """
    Determine final decision confidence.
    """

    if (
        classification == "SPAM"
        and ml_label == "spam"
        and evidence in {
            "MODERATE",
            "STRONG",
        }
    ):
        return "HIGH"

    if (
        classification == "HAM"
        and ml_label == "ham"
        and evidence == "WEAK"
        and risk_level in {
            "LOW",
            "MEDIUM",
        }
    ):
        return "HIGH"

    if evidence == "STRONG":
        return "HIGH"

    if evidence == "MODERATE":
        return "MEDIUM"

    return "MEDIUM"


def make_decision_v31(
    ml_result: Any,
    security_analysis: Any,
    risk_assessment: Any,
) -> Dict[str, Any]:
    """
    Produce the final V3.1 email-security decision.

    Decision model:

    ML SPAM:
        - STRONG evidence/high risk -> BLOCK / REVIEW
        - MODERATE evidence -> QUARANTINE / REVIEW
        - WEAK evidence -> REVIEW

    ML HAM:
        - STRONG security evidence + sufficient risk -> SPAM
        - MODERATE evidence + sufficient risk -> SUSPICIOUS
        - otherwise -> HAM / ALLOW

    This keeps ML classification important while allowing
    independent security indicators to escalate risky mail.
    """

    ml_label = _normalise_label(
        _get(
            ml_result,
            "label",
            "",
        )
    )

    ml_spam = bool(
        _get(
            ml_result,
            "spam",
            ml_label == "spam",
        )
    )

    ml_score = float(
        _get(
            ml_result,
            "decision_score",
            0.0,
        )
        or 0.0
    )

    threshold = float(
        _get(
            ml_result,
            "threshold",
            -0.0975,
        )
        or -0.0975
    )

    risk_score = int(
        _get(
            risk_assessment,
            "risk_score",
            0,
        )
        or 0
    )

    risk_level = _normalise_level(
        _get(
            risk_assessment,
            "risk_level",
            "LOW",
        )
    )

    evidence = _security_evidence(
        security_analysis,
        risk_assessment,
    )

    classification = "HAM"
    action = "ALLOW"


    if ml_spam or ml_label == "spam":

        classification = "SPAM"


        if (
            evidence == "STRONG"
            or risk_score >= 60
            or risk_level == "CRITICAL"
        ):
            action = "BLOCK / REVIEW"


        elif evidence == "MODERATE":
            action = "QUARANTINE / REVIEW"


        else:
            action = "REVIEW"


    else:

        classification = "HAM"
        action = "ALLOW"


        if (
            evidence == "STRONG"
            and (
                risk_score >= 35
                or risk_level in {
                    "HIGH",
                    "CRITICAL",
                }
            )
        ):
            classification = "SPAM"
            action = "BLOCK / REVIEW"


        elif (
            evidence == "MODERATE"
            and risk_score >= 35
        ):
            classification = "SUSPICIOUS"
            action = "QUARANTINE / REVIEW"


        elif risk_level == "CRITICAL":

            classification = "SUSPICIOUS"
            action = "QUARANTINE / REVIEW"


        elif risk_score >= 50:

            classification = "SUSPICIOUS"
            action = "QUARANTINE / REVIEW"

    confidence = _confidence(
        classification=classification,
        ml_label=ml_label,
        evidence=evidence,
        risk_level=risk_level,
    )


    if classification == "HAM":

        reason = (
            "The ML model classified the message as ham "
            "and available security evidence does not "
            "justify escalation."
        )

    elif classification == "SPAM":

        if ml_spam or ml_label == "spam":

            reason = (
                "The ML model classified the message as "
                "spam. The message is therefore escalated "
                "for security review."
            )

        else:

            reason = (
                "The ML model classified the message as "
                "ham, but strong independent security "
                "indicators require security escalation."
            )

    else:

        reason = (
            "The ML model classified the message as ham, "
            "but security indicators create sufficient "
            "uncertainty to require manual review."
        )

    return {
        "classification": classification,
        "action": action,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "evidence": evidence,
        "ml_prediction": ml_label,
        "ml_score": ml_score,
        "threshold": threshold,
        "reason": reason,
    }


def print_decision_v31(
    decision: Dict[str, Any],
):
    """
    Print the V3.1 final decision.
    """

    print()
    print("=" * 70)
    print(
        "EMAIL SECURITY AI — FINAL V3.1 DECISION"
    )
    print("=" * 70)

    print(
        f"\nClassification : "
        f"{decision['classification']}"
    )

    print(
        f"Action         : "
        f"{decision['action']}"
    )

    print(
        f"Confidence     : "
        f"{decision['confidence']}"
    )

    print(
        f"ML prediction  : "
        f"{decision['ml_prediction']}"
    )

    print(
        f"ML score       : "
        f"{decision['ml_score']:.4f}"
    )

    print(
        f"ML threshold   : "
        f"{decision['threshold']:.4f}"
    )

    print(
        f"Risk score     : "
        f"{decision['risk_score']}/100"
    )

    print(
        f"Risk level     : "
        f"{decision['risk_level']}"
    )

    print(
        f"Evidence       : "
        f"{decision['evidence']}"
    )

    print("\nReason:")

    print(
        f"  {decision['reason']}"
    )

    print("=" * 70)
