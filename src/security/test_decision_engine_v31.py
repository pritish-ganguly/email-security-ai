"""
Email Security AI
Decision Engine V3.1 Test Suite

Purpose:
    Validate the V3.1 decision engine against controlled scenarios.

Run from project root:

    python -m src.security.test_decision_engine_v31
"""

from src.security.decision_engine_v31 import make_decision_v31


def run_test(
    name,
    ml_result,
    security_analysis,
    risk_assessment,
    expected_classification,
    expected_action,
):
    try:
        result = make_decision_v31(
            ml_result=ml_result,
            security_analysis=security_analysis,
            risk_assessment=risk_assessment,
        )

        actual_classification = result.get(
            "classification",
            ""
        )

        actual_action = result.get(
            "action",
            ""
        )

        if (
            actual_classification == expected_classification
            and actual_action == expected_action
        ):
            print(f"[PASS] {name}")
            return True

        print(f"[FAIL] {name}")
        print(
            f"       Expected classification: "
            f"{expected_classification}"
        )
        print(
            f"       Actual classification: "
            f"{actual_classification}"
        )
        print(
            f"       Expected action: "
            f"{expected_action}"
        )
        print(
            f"       Actual action: "
            f"{actual_action}"
        )

        return False

    except Exception as exc:
        print(f"[FAIL] {name}")
        print(f"       Error: {exc}")
        return False


def test_clean_ham():
    return run_test(
        name="test_clean_ham",
        ml_result={
            "label": "ham",
            "spam": False,
            "decision_score": -1.20,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 0,
            "ip_url_count": 0,
            "suspicious_keyword_count": 0,
            "html_present": False,
            "script_present": False,
            "attachment_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0.03,
        },
        risk_assessment={
            "risk_score": 2,
            "risk_level": "LOW",
        },
        expected_classification="HAM",
        expected_action="ALLOW",
    )


def test_clear_spam():
    return run_test(
        name="test_clear_spam",
        ml_result={
            "label": "spam",
            "spam": True,
            "decision_score": -0.01,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 0,
            "ip_url_count": 0,
            "suspicious_keyword_count": 0,
            "html_present": False,
            "script_present": False,
            "attachment_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0.04,
        },
        risk_assessment={
            "risk_score": 50,
            "risk_level": "HIGH",
        },
        expected_classification="SPAM",
        expected_action="REVIEW",
    )


def test_strong_security_indicators():
    return run_test(
        name="test_strong_security_indicators",
        ml_result={
            "label": "ham",
            "spam": False,
            "decision_score": -1.10,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 3,
            "ip_url_count": 1,
            "suspicious_keyword_count": 4,
            "html_present": True,
            "script_present": True,
            "attachment_count": 1,
            "exclamation_count": 5,
            "uppercase_ratio": 0.25,
        },
        risk_assessment={
            "risk_score": 80,
            "risk_level": "CRITICAL",
        },
        expected_classification="SPAM",
        expected_action="BLOCK / REVIEW",
    )


def test_ham_with_moderate_risk():
    return run_test(
        name="test_ham_with_moderate_risk",
        ml_result={
            "label": "ham",
            "spam": False,
            "decision_score": -0.90,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 1,
            "ip_url_count": 0,
            "suspicious_keyword_count": 0,
            "html_present": False,
            "script_present": False,
            "attachment_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0.05,
        },
        risk_assessment={
            "risk_score": 25,
            "risk_level": "MEDIUM",
        },
        expected_classification="HAM",
        expected_action="ALLOW",
    )


def test_spam_with_weak_security_evidence():
    return run_test(
        name="test_spam_with_weak_security_evidence",
        ml_result={
            "label": "spam",
            "spam": True,
            "decision_score": -0.08,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 0,
            "ip_url_count": 0,
            "suspicious_keyword_count": 0,
            "html_present": False,
            "script_present": False,
            "attachment_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0.06,
        },
        risk_assessment={
            "risk_score": 20,
            "risk_level": "LOW",
        },
        expected_classification="SPAM",
        expected_action="REVIEW",
    )


def test_attachment_risk():
    return run_test(
        name="test_attachment_risk",
        ml_result={
            "label": "ham",
            "spam": False,
            "decision_score": -1.00,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 0,
            "ip_url_count": 0,
            "suspicious_keyword_count": 0,
            "html_present": False,
            "script_present": False,
            "attachment_count": 1,
            "exclamation_count": 0,
            "uppercase_ratio": 0.04,
        },
        risk_assessment={
            "risk_score": 45,
            "risk_level": "MEDIUM",
        },
        expected_classification="SUSPICIOUS",
        expected_action="QUARANTINE / REVIEW",
    )


def test_html_script_email():
    return run_test(
        name="test_html_script_email",
        ml_result={
            "label": "ham",
            "spam": False,
            "decision_score": -1.00,
            "threshold": -0.0975,
        },
        security_analysis={
            "url_count": 2,
            "ip_url_count": 0,
            "suspicious_keyword_count": 1,
            "html_present": True,
            "script_present": True,
            "attachment_count": 0,
            "exclamation_count": 0,
            "uppercase_ratio": 0.08,
        },
        risk_assessment={
            "risk_score": 55,
            "risk_level": "HIGH",
        },
        expected_classification="SPAM",
        expected_action="BLOCK / REVIEW",
    )


def main():
    print()
    print("=" * 70)
    print("EMAIL SECURITY AI — DECISION ENGINE V3.1 TESTS")
    print("=" * 70)

    tests = [
        test_clean_ham,
        test_clear_spam,
        test_strong_security_indicators,
        test_ham_with_moderate_risk,
        test_spam_with_weak_security_evidence,
        test_attachment_risk,
        test_html_script_email,
    ]

    passed = 0
    failed = 0

    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 70)
    print(f"Tests passed : {passed}")
    print(f"Tests failed : {failed}")
    print("=" * 70)

    if failed == 0:
        print()
        print(
            "Decision Engine V3.1 tests completed successfully."
        )
        print()
        return 0

    print()
    print(
        "Decision Engine V3.1 tests completed with failures."
    )
    print()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
