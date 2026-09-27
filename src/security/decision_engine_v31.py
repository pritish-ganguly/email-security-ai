"""
Email Security AI — Decision Engine V3.1

Combines:

    1. Machine-learning classification
    2. Security indicators
    3. Risk assessment

to produce a final security decision.

This module is the production decision layer.

Expected output:

    classification
    action
    confidence
    risk_score
    risk_level
    reason
"""


def _get_value(
    data,
    key,
    default=None,
):
    """Safely retrieve a value from a dictionary."""

    if not isinstance(data, dict):
        return default

    return data.get(
        key,
        default,
    )


def make_decision_v31(
    ml_result,
    security_analysis,
    risk_assessment,
):
    """
    Generate the final Email Security AI V3.1 decision.

    Decision hierarchy:

    1. Critical security indicators override ML HAM.
    2. High-risk malicious indicators can override ML HAM.
    3. Attachment-driven medium risk produces SUSPICIOUS.
    4. ML SPAM produces SPAM / REVIEW.
    5. Clean HAM produces HAM / ALLOW.

    Returns:
        dict containing the final security decision.
    """

    if not isinstance(ml_result, dict):
        raise TypeError(
            "ml_result must be a dictionary."
        )

    if not isinstance(security_analysis, dict):
        raise TypeError(
            "security_analysis must be a dictionary."
        )

    if not isinstance(risk_assessment, dict):
        raise TypeError(
            "risk_assessment must be a dictionary."
        )

    # --------------------------------------------------
    # Machine-learning result
    # --------------------------------------------------

    ml_label = str(
        _get_value(
            ml_result,
            "label",
            "",
        )
    ).lower()

    ml_spam = bool(
        _get_value(
            ml_result,
            "spam",
            ml_label == "spam",
        )
    )

    decision_score = float(
        _get_value(
            ml_result,
            "decision_score",
            0.0,
        )
    )

    threshold = float(
        _get_value(
            ml_result,
            "threshold",
            -0.0975,
        )
    )

    # --------------------------------------------------
    # Security analysis
    # --------------------------------------------------

    url_count = int(
        _get_value(
            security_analysis,
            "url_count",
            0,
        )
        or 0
    )

    ip_url_count = int(
        _get_value(
            security_analysis,
            "ip_url_count",
            0,
        )
        or 0
    )

    suspicious_keyword_count = int(
        _get_value(
            security_analysis,
            "suspicious_keyword_count",
            0,
        )
        or 0
    )

    html_present = bool(
        _get_value(
            security_analysis,
            "html_present",
            False,
        )
    )

    script_present = bool(
        _get_value(
            security_analysis,
            "script_present",
            False,
        )
    )

    attachment_count = int(
        _get_value(
            security_analysis,
            "attachment_count",
            0,
        )
        or 0
    )

    exclamation_count = int(
        _get_value(
            security_analysis,
            "exclamation_count",
            0,
        )
        or 0
    )

    uppercase_ratio = float(
        _get_value(
            security_analysis,
            "uppercase_ratio",
            0.0,
        )
        or 0.0
    )

    # --------------------------------------------------
    # Risk assessment
    # --------------------------------------------------

    risk_score = float(
        _get_value(
            risk_assessment,
            "risk_score",
            0,
        )
        or 0
    )

    risk_level = str(
        _get_value(
            risk_assessment,
            "risk_level",
            "LOW",
        )
    ).upper()

    # --------------------------------------------------
    # Derived security indicators
    # --------------------------------------------------

    strong_security_indicators = (
        ip_url_count >= 1
        or suspicious_keyword_count >= 4
        or script_present
        or (
            html_present
            and url_count >= 2
        )
        or (
            attachment_count >= 1
            and risk_score >= 70
        )
        or exclamation_count >= 5
        or uppercase_ratio >= 0.25
    )

    high_security_risk = (
        risk_score >= 55
        or risk_level in {
            "HIGH",
            "CRITICAL",
        }
    )

    attachment_risk = (
        attachment_count >= 1
        and risk_score >= 40
    )

    # --------------------------------------------------
    # Decision 1 — Critical / strong malicious evidence
    # --------------------------------------------------

    if (
        strong_security_indicators
        and high_security_risk
    ):
        return {
            "classification": "SPAM",
            "action": "BLOCK / REVIEW",
            "confidence": "HIGH",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": (
                "Strong security indicators combined "
                "with elevated risk."
            ),
            "ml_label": ml_label,
            "ml_spam": ml_spam,
            "decision_score": decision_score,
            "threshold": threshold,
        }

    # --------------------------------------------------
    # Decision 2 — HTML/script based attack indicators
    # --------------------------------------------------

    if (
        script_present
        and html_present
        and url_count >= 2
    ):
        return {
            "classification": "SPAM",
            "action": "BLOCK / REVIEW",
            "confidence": "HIGH",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": (
                "HTML content contains script activity "
                "and multiple URLs."
            ),
            "ml_label": ml_label,
            "ml_spam": ml_spam,
            "decision_score": decision_score,
            "threshold": threshold,
        }

    # --------------------------------------------------
    # Decision 3 — Attachment risk
    # --------------------------------------------------

    if attachment_risk:
        return {
            "classification": "SUSPICIOUS",
            "action": "QUARANTINE / REVIEW",
            "confidence": "MEDIUM",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": (
                "Email contains an attachment combined "
                "with elevated security risk."
            ),
            "ml_label": ml_label,
            "ml_spam": ml_spam,
            "decision_score": decision_score,
            "threshold": threshold,
        }

    # --------------------------------------------------
    # Decision 4 — ML classified spam
    # --------------------------------------------------

    if ml_spam or ml_label == "spam":
        return {
            "classification": "SPAM",
            "action": "REVIEW",
            "confidence": "HIGH",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": (
                "The frozen ML model classified "
                "the email as spam."
            ),
            "ml_label": ml_label,
            "ml_spam": ml_spam,
            "decision_score": decision_score,
            "threshold": threshold,
        }

    # --------------------------------------------------
    # Decision 5 — High risk despite ML HAM
    # --------------------------------------------------

    if high_security_risk:
        return {
            "classification": "SPAM",
            "action": "BLOCK / REVIEW",
            "confidence": "HIGH",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reason": (
                "The ML model classified the email as HAM, "
                "but security analysis indicates high risk."
            ),
            "ml_label": ml_label,
            "ml_spam": ml_spam,
            "decision_score": decision_score,
            "threshold": threshold,
        }

    # --------------------------------------------------
    # Decision 6 — Clean / normal HAM
    # --------------------------------------------------

    return {
        "classification": "HAM",
        "action": "ALLOW",
        "confidence": "HIGH",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reason": (
            "The email was classified as HAM and "
            "no significant security indicators were detected."
        ),
        "ml_label": ml_label,
        "ml_spam": ml_spam,
        "decision_score": decision_score,
        "threshold": threshold,
    }


def print_decision(
    decision,
):
    """Print a formatted V3.1 decision."""

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — DECISION ENGINE V3.1")
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
        f"{decision.get('risk_score', 0)}/100"
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

    print("=" * 70)