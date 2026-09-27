"""
Test suite for Email Security AI Decision Engine V2.

This test module validates the decision engine against
representative combinations of:

    - ML classification
    - security indicators
    - risk scores
    - final security actions

No model training or model artifacts are modified.
"""

from src.security.decision_engine_v2 import (
    make_decision_v2,
)


def make_security_analysis(
    *,
    html_present=False,
    script_present=False,
    url_count=0,
    ip_based_url_count=0,
    suspicious_keywords=0,
    attachments=0,
    body_length=100,
    subject_length=20,
):
    """
    Create a minimal security-analysis object compatible
    with the V2 decision engine.
    """

    return {
        "html_present": html_present,
        "script_present": script_present,
        "url_count": url_count,
        "ip_based_url_count": ip_based_url_count,
        "suspicious_keywords": suspicious_keywords,
        "attachments": attachments,
        "body_length": body_length,
        "subject_length": subject_length,
    }


def make_risk_assessment(
    risk_score,
    risk_level,
):
    """
    Create a minimal risk-assessment object.
    """

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


def make_ml_result(
    label,
    score,
    threshold=-0.0975,
):
    """
    Create a minimal ML prediction result.
    """

    return {
        "label": label,
        "spam": label == "spam",
        "decision_score": score,
        "threshold": threshold,
    }


def test_clean_ham():
    """
    A clean ham message should normally be allowed.
    """

    ml_result = make_ml_result(
        label="ham",
        score=-1.0000,
    )

    security_analysis = make_security_analysis()

    risk_assessment = make_risk_assessment(
        risk_score=5,
        risk_level="LOW",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.classification == "HAM"
    assert result.action == "ALLOW"


def test_clear_spam():
    """
    A spam prediction supported by security indicators
    should be escalated.
    """

    ml_result = make_ml_result(
        label="spam",
        score=0.5000,
    )

    security_analysis = make_security_analysis(
        url_count=5,
        suspicious_keywords=2,
        html_present=True,
    )

    risk_assessment = make_risk_assessment(
        risk_score=60,
        risk_level="HIGH",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.classification in {
        "SPAM",
        "SUSPICIOUS",
    }

    assert result.action != "ALLOW"


def test_strong_security_indicators():
    """
    Strong indicators should cause escalation even if
    the ML model predicts ham.
    """

    ml_result = make_ml_result(
        label="ham",
        score=-1.2000,
    )

    security_analysis = make_security_analysis(
        script_present=True,
        ip_based_url_count=1,
        suspicious_keywords=4,
    )

    risk_assessment = make_risk_assessment(
        risk_score=80,
        risk_level="CRITICAL",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.classification != "HAM"
    assert result.action != "ALLOW"


def test_ham_with_moderate_risk():
    """
    A ham prediction with elevated but non-critical
    security evidence should not automatically be treated
    as clean.
    """

    ml_result = make_ml_result(
        label="ham",
        score=-0.8000,
    )

    security_analysis = make_security_analysis(
        url_count=3,
        suspicious_keywords=1,
    )

    risk_assessment = make_risk_assessment(
        risk_score=45,
        risk_level="MEDIUM",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.action != "ALLOW"


def test_short_benign_email():
    """
    A short, clean ham email should remain low risk.
    """

    ml_result = make_ml_result(
        label="ham",
        score=-1.1000,
    )

    security_analysis = make_security_analysis(
        body_length=30,
        subject_length=15,
        url_count=0,
        suspicious_keywords=0,
        attachments=0,
    )

    risk_assessment = make_risk_assessment(
        risk_score=3,
        risk_level="LOW",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.classification == "HAM"
    assert result.action == "ALLOW"


def test_spam_with_weak_security_evidence():
    """
    A spam ML prediction without strong additional indicators
    should still not automatically become ham.
    """

    ml_result = make_ml_result(
        label="spam",
        score=0.1000,
    )

    security_analysis = make_security_analysis()

    risk_assessment = make_risk_assessment(
        risk_score=25,
        risk_level="MEDIUM",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.classification in {
        "SPAM",
        "SUSPICIOUS",
    }


def test_attachment_risk():
    """
    Attachments should contribute to security escalation
    when combined with elevated risk.
    """

    ml_result = make_ml_result(
        label="ham",
        score=-0.9000,
    )

    security_analysis = make_security_analysis(
        attachments=1,
    )

    risk_assessment = make_risk_assessment(
        risk_score=45,
        risk_level="MEDIUM",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.action != "ALLOW"


def test_html_script_email():
    """
    HTML combined with script content represents a strong
    security signal.
    """

    ml_result = make_ml_result(
        label="ham",
        score=-1.0000,
    )

    security_analysis = make_security_analysis(
        html_present=True,
        script_present=True,
    )

    risk_assessment = make_risk_assessment(
        risk_score=75,
        risk_level="CRITICAL",
    )

    result = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    assert result.classification != "HAM"
    assert result.action != "ALLOW"


if __name__ == "__main__":
    print("=" * 70)
    print(
        "EMAIL SECURITY AI — DECISION ENGINE V2 TESTS"
    )
    print("=" * 70)

    tests = [
        test_clean_ham,
        test_clear_spam,
        test_strong_security_indicators,
        test_ham_with_moderate_risk,
        test_short_benign_email,
        test_spam_with_weak_security_evidence,
        test_attachment_risk,
        test_html_script_email,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(
                f"[PASS] {test.__name__}"
            )
            passed += 1

        except Exception as exc:
            print(
                f"[FAIL] {test.__name__}"
            )
            print(
                f"       {exc}"
            )
            failed += 1

    print("\n" + "=" * 70)

    print(
        f"Tests passed : {passed}"
    )

    print(
        f"Tests failed : {failed}"
    )

    print("=" * 70)

    if failed:
        raise SystemExit(1)

    print(
        "\nDecision Engine V2 tests completed successfully."
    )
