"""
Email Security AI — Decision Engine V3 Tests

Tests the V3 decision engine using controlled scenarios.

Run:

    python -m src.security.test_decision_engine_v3
"""

from src.security.decision_engine_v3 import (
    make_decision_v3,
)


def create_ml_result(
    label="ham",
    spam=False,
    score=-1.0,
    threshold=-0.0975,
):
    """Create a standard ML result for testing."""

    return {
        "label": label,
        "spam": spam,
        "decision_score": score,
        "threshold": threshold,
    }


def create_security_analysis(
    html_present=False,
    script_present=False,
    url_count=0,
    ip_based_url_count=0,
    suspicious_keywords=0,
    attachments=0,
    body_length=100,
    subject_length=30,
):
    """Create a standard security-analysis object."""

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


def create_risk_assessment(
    risk_score=0,
    risk_level="LOW",
):
    """Create a standard risk assessment."""

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


def run_test(
    name,
    ml_result,
    security_analysis,
    risk_assessment,
    expected_classification,
    expected_action,
):
    """Run one V3 decision-engine test."""

    decision = make_decision_v3(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    passed = (
        decision.classification
        == expected_classification
        and decision.action
        == expected_action
    )

    if passed:
        print(f"[PASS] {name}")

    else:
        print(f"[FAIL] {name}")

        print(
            f"       Expected classification: "
            f"{expected_classification}"
        )

        print(
            f"       Actual classification: "
            f"{decision.classification}"
        )

        print(
            f"       Expected action: "
            f"{expected_action}"
        )

        print(
            f"       Actual action: "
            f"{decision.action}"
        )

    return passed


def main():
    """Run all V3 decision-engine tests."""

    print("\n" + "=" * 70)

    print(
        "EMAIL SECURITY AI — "
        "DECISION ENGINE V3 TESTS"
    )

    print("=" * 70)

    tests_passed = 0
    tests_failed = 0


    if run_test(
        name="test_clean_ham",

        ml_result=create_ml_result(
            label="ham",
            spam=False,
            score=-1.2,
        ),

        security_analysis=create_security_analysis(),

        risk_assessment=create_risk_assessment(
            risk_score=5,
            risk_level="LOW",
        ),

        expected_classification="HAM",
        expected_action="ALLOW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_clear_spam",

        ml_result=create_ml_result(
            label="spam",
            spam=True,
            score=0.8,
        ),

        security_analysis=create_security_analysis(
            suspicious_keywords=4,
            url_count=4,
        ),

        risk_assessment=create_risk_assessment(
            risk_score=75,
            risk_level="CRITICAL",
        ),

        expected_classification="SPAM",
        expected_action="BLOCK / REVIEW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_strong_security_indicators",

        ml_result=create_ml_result(
            label="ham",
            spam=False,
            score=-1.0,
        ),

        security_analysis=create_security_analysis(
            script_present=True,
            ip_based_url_count=1,
        ),

        risk_assessment=create_risk_assessment(
            risk_score=30,
            risk_level="MEDIUM",
        ),

        expected_classification="SPAM",
        expected_action="BLOCK / REVIEW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_ham_with_moderate_risk",

        ml_result=create_ml_result(
            label="ham",
            spam=False,
            score=-1.0,
        ),

        security_analysis=create_security_analysis(
            url_count=3,
        ),

        risk_assessment=create_risk_assessment(
            risk_score=45,
            risk_level="MEDIUM",
        ),

        expected_classification="SUSPICIOUS",
        expected_action="QUARANTINE / REVIEW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_short_benign_email",

        ml_result=create_ml_result(
            label="spam",
            spam=True,
            score=0.05,
        ),

        security_analysis=create_security_analysis(
            body_length=40,
            subject_length=15,
        ),

        risk_assessment=create_risk_assessment(
            risk_score=5,
            risk_level="LOW",
        ),

        expected_classification="HAM",
        expected_action="ALLOW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_spam_with_weak_security_evidence",

        ml_result=create_ml_result(
            label="spam",
            spam=True,
            score=0.3,
        ),

        security_analysis=create_security_analysis(),

        risk_assessment=create_risk_assessment(
            risk_score=15,
            risk_level="LOW",
        ),

        expected_classification="SPAM",
        expected_action="REVIEW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_attachment_risk",

        ml_result=create_ml_result(
            label="ham",
            spam=False,
            score=-0.8,
        ),

        security_analysis=create_security_analysis(
            attachments=1,
        ),

        risk_assessment=create_risk_assessment(
            risk_score=45,
            risk_level="MEDIUM",
        ),

        expected_classification="SUSPICIOUS",
        expected_action="QUARANTINE / REVIEW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    if run_test(
        name="test_html_script_email",

        ml_result=create_ml_result(
            label="ham",
            spam=False,
            score=-1.0,
        ),

        security_analysis=create_security_analysis(
            html_present=True,
            script_present=True,
        ),

        risk_assessment=create_risk_assessment(
            risk_score=80,
            risk_level="CRITICAL",
        ),

        expected_classification="SPAM",
        expected_action="BLOCK / REVIEW",
    ):
        tests_passed += 1
    else:
        tests_failed += 1


    print("\n" + "=" * 70)

    print(
        f"Tests passed : {tests_passed}"
    )

    print(
        f"Tests failed : {tests_failed}"
    )

    print("=" * 70)

    if tests_failed == 0:

        print(
            "\nDecision Engine V3 tests "
            "completed successfully."
        )

    else:

        print(
            "\nDecision Engine V3 tests "
            "completed with failures."
        )


if __name__ == "__main__":
    main()
