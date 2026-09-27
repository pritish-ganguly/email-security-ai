"""
Decision Engine Error Analysis
==============================

Analyzes the behavior of the production decision engine
without modifying:

    - ML model
    - NLP pipeline
    - decision threshold
    - training data
    - production artifacts

The analysis focuses on:

    1. HAM emails escalated by the decision engine
    2. SPAM emails allowed by the decision engine
    3. ML vs final-decision disagreements
    4. ML score vs risk-score conflicts
    5. Security indicators responsible for escalation
    6. Potential decision-engine overreach
    7. Potential decision-engine underreaction

This is diagnostic only.
No retraining is performed.
No model artifacts are modified.
"""


from pathlib import Path
from collections import Counter


from src.data.email_parser import parse_email

from src.models.predict import EmailPredictor

from src.security.security_analyzer import (
    analyze_email,
)

from src.security.risk_engine import (
    assess_email_risk,
)

from src.security.decision_engine import (
    make_decision,
)


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)

PROGRESS_INTERVAL = 250

MAX_DISPLAY = 20


def get_value(
    email_data,
    field,
    default="",
):
    """
    Safely retrieve a field from dictionary or
    object-based parser results.
    """

    if isinstance(email_data, dict):
        return email_data.get(
            field,
            default,
        )

    return getattr(
        email_data,
        field,
        default,
    )


def build_email_text(
    email_data,
):
    """
    Reconstruct the same text representation used
    during production inference.
    """

    subject = str(
        get_value(
            email_data,
            "subject",
            "",
        )
        or ""
    )

    body = str(
        get_value(
            email_data,
            "body",
            "",
        )
        or ""
    )

    html_body = str(
        get_value(
            email_data,
            "html_body",
            "",
        )
        or ""
    )

    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    )


def get_files(
    directory,
):
    """
    Return all files in a dataset directory.

    SpamAssassin files do not necessarily use the
    .eml extension, so we intentionally do not
    filter by suffix.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
        ]
    )


def get_indicator(
    security_analysis,
    name,
    default=0,
):
    """
    Safely retrieve a security-analysis indicator.
    """

    if isinstance(
        security_analysis,
        dict,
    ):
        return security_analysis.get(
            name,
            default,
        )

    return getattr(
        security_analysis,
        name,
        default,
    )


def get_security_indicators(
    security_analysis,
):
    """
    Extract important security indicators.
    """

    return {
        "url_count": int(
            get_indicator(
                security_analysis,
                "url_count",
                0,
            )
        ),

        "ip_based_url_count": int(
            get_indicator(
                security_analysis,
                "ip_based_url_count",
                0,
            )
        ),

        "suspicious_keywords": int(
            get_indicator(
                security_analysis,
                "suspicious_keywords",
                0,
            )
        ),

        "html_present": bool(
            get_indicator(
                security_analysis,
                "html_present",
                False,
            )
        ),

        "script_present": bool(
            get_indicator(
                security_analysis,
                "script_present",
                False,
            )
        ),

        "attachments": int(
            get_indicator(
                security_analysis,
                "attachments",
                0,
            )
        ),

        "body_length": int(
            get_indicator(
                security_analysis,
                "body_length",
                0,
            )
        ),

        "subject_length": int(
            get_indicator(
                security_analysis,
                "subject_length",
                0,
            )
        ),

        "word_count": int(
            get_indicator(
                security_analysis,
                "word_count",
                0,
            )
        ),
    }


def determine_triggered_rules(
    indicators,
    risk_score,
):
    """
    Identify the rule categories that could contribute
    to decision-engine escalation.

    These are diagnostic explanations only.
    """

    rules = []

    if indicators["script_present"]:
        rules.append(
            "script_present"
        )

    if indicators["ip_based_url_count"] > 0:
        rules.append(
            "ip_based_url"
        )

    if (
        indicators["suspicious_keywords"]
        >= 3
    ):
        rules.append(
            "suspicious_keywords>=3"
        )

    if risk_score >= 70:
        rules.append(
            "risk_score>=70"
        )

    if indicators["url_count"] >= 3:
        rules.append(
            "url_count>=3"
        )

    if (
        indicators["suspicious_keywords"]
        >= 1
    ):
        rules.append(
            "suspicious_keywords>=1"
        )

    if indicators["attachments"] > 0:
        rules.append(
            "attachments_present"
        )

    if risk_score >= 40:
        rules.append(
            "risk_score>=40"
        )

    return rules


def classify_conflict(
    expected,
    ml_spam,
    final_classification,
):
    """
    Categorize disagreement between expected label,
    ML prediction and final decision.
    """

    if (
        expected == "ham"
        and ml_spam is False
        and final_classification != "HAM"
    ):
        return "HAM_ESCALATION"

    if (
        expected == "spam"
        and ml_spam is False
        and final_classification == "HAM"
    ):
        return "SPAM_ML_FALSE_NEGATIVE"

    if (
        expected == "spam"
        and final_classification == "HAM"
    ):
        return "SPAM_ALLOWED"

    if (
        expected == "ham"
        and final_classification == "SPAM"
    ):
        return "HAM_BLOCKED"

    if (
        expected == "ham"
        and final_classification
        == "SUSPICIOUS"
    ):
        return "HAM_SUSPICIOUS"

    if (
        expected == "spam"
        and final_classification
        == "SUSPICIOUS"
    ):
        return "SPAM_SUSPICIOUS"

    if ml_spam != (
        final_classification == "SPAM"
    ):
        return "ML_FINAL_DISAGREEMENT"

    return "OTHER"


def analyze_email_file(
    email_path,
    expected,
    predictor,
):
    """
    Run the complete production decision pipeline
    on one dataset email.
    """

    email_data = parse_email(
        email_path
    )

    email_text = build_email_text(
        email_data
    )

    ml_result = predictor.predict(
        email_text
    )

    subject = str(
        get_value(
            email_data,
            "subject",
            "",
        )
        or ""
    )

    body = str(
        get_value(
            email_data,
            "body",
            "",
        )
        or ""
    )

    html_body = str(
        get_value(
            email_data,
            "html_body",
            "",
        )
        or ""
    )

    attachments = get_value(
        email_data,
        "attachments",
        [],
    )

    security_analysis = analyze_email(
        subject=subject,
        body=body,
        html_body=html_body,
        attachments=attachments,
    )

    risk_assessment = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_analysis,
    )

    decision = make_decision(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    risk_score = int(
        getattr(
            risk_assessment,
            "risk_score",
            0,
        )
    )

    risk_level = str(
        getattr(
            risk_assessment,
            "risk_level",
            "LOW",
        )
    ).upper()

    indicators = get_security_indicators(
        security_analysis
    )

    triggered_rules = determine_triggered_rules(
        indicators,
        risk_score,
    )

    ml_label = str(
        ml_result.get(
            "label",
            "unknown",
        )
    ).lower()

    ml_spam = bool(
        ml_result.get(
            "spam",
            ml_label == "spam",
        )
    )

    ml_score = float(
        ml_result.get(
            "decision_score",
            0.0,
        )
    )

    final_classification = str(
        decision.classification
    ).upper()

    action = str(
        decision.action
    )

    confidence = str(
        decision.confidence
    )

    conflict_type = classify_conflict(
        expected=expected,
        ml_spam=ml_spam,
        final_classification=final_classification,
    )

    return {
        "file": str(email_path),
        "expected": expected,
        "ml_label": ml_label,
        "ml_spam": ml_spam,
        "ml_score": ml_score,
        "threshold": float(
            ml_result.get(
                "threshold",
                0.0,
            )
        ),
        "final_classification":
            final_classification,
        "action": action,
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "indicators": indicators,
        "triggered_rules": triggered_rules,
        "conflict_type": conflict_type,
    }


def print_case(
    case,
):
    """
    Print a detailed diagnostic case.
    """

    indicators = case[
        "indicators"
    ]

    print(
        f"\nFile           : "
        f"{case['file']}"
    )

    print(
        f"Expected       : "
        f"{case['expected']}"
    )

    print(
        f"ML prediction  : "
        f"{case['ml_label']}"
    )

    print(
        f"ML score       : "
        f"{case['ml_score']:.4f}"
    )

    print(
        f"Final class    : "
        f"{case['final_classification']}"
    )

    print(
        f"Action         : "
        f"{case['action']}"
    )

    print(
        f"Confidence     : "
        f"{case['confidence']}"
    )

    print(
        f"Risk score     : "
        f"{case['risk_score']}/100"
    )

    print(
        f"Risk level     : "
        f"{case['risk_level']}"
    )

    print(
        "\nSecurity indicators:"
    )

    print(
        f"  URLs                 : "
        f"{indicators['url_count']}"
    )

    print(
        f"  IP-based URLs        : "
        f"{indicators['ip_based_url_count']}"
    )

    print(
        f"  Suspicious keywords  : "
        f"{indicators['suspicious_keywords']}"
    )

    print(
        f"  HTML                 : "
        f"{indicators['html_present']}"
    )

    print(
        f"  Script               : "
        f"{indicators['script_present']}"
    )

    print(
        f"  Attachments          : "
        f"{indicators['attachments']}"
    )

    print(
        f"  Body length          : "
        f"{indicators['body_length']}"
    )

    print(
        f"  Word count           : "
        f"{indicators['word_count']}"
    )

    print(
        "\nTriggered diagnostic rules:"
    )

    if case["triggered_rules"]:
        for rule in case[
            "triggered_rules"
        ]:
            print(
                f"  - {rule}"
            )
    else:
        print(
            "  None"
        )


def main():
    print(
        "\n"
        + "=" * 78
    )

    print(
        "EMAIL SECURITY AI — "
        "DECISION ENGINE ERROR ANALYSIS"
    )

    print(
        "=" * 78
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    ham_files = get_files(
        HAM_DIR
    )

    spam_files = get_files(
        SPAM_DIR
    )

    print(
        f"\nHAM emails discovered : "
        f"{len(ham_files):,}"
    )

    print(
        f"SPAM emails discovered: "
        f"{len(spam_files):,}"
    )

    all_results = []

    errors = []


    print(
        "\nEvaluating HAM emails..."
    )

    for index, email_path in enumerate(
        ham_files,
        start=1,
    ):

        try:
            result = analyze_email_file(
                email_path=email_path,
                expected="ham",
                predictor=predictor,
            )

            all_results.append(
                result
            )

        except Exception as exc:
            errors.append(
                (
                    str(email_path),
                    str(exc),
                )
            )

        if (
            index % PROGRESS_INTERVAL == 0
            or index == len(ham_files)
        ):
            print(
                f"  HAM processed: "
                f"{index:,}/"
                f"{len(ham_files):,}"
            )


    print(
        "\nEvaluating SPAM emails..."
    )

    for index, email_path in enumerate(
        spam_files,
        start=1,
    ):

        try:
            result = analyze_email_file(
                email_path=email_path,
                expected="spam",
                predictor=predictor,
            )

            all_results.append(
                result
            )

        except Exception as exc:
            errors.append(
                (
                    str(email_path),
                    str(exc),
                )
            )

        if (
            index % PROGRESS_INTERVAL == 0
            or index == len(spam_files)
        ):
            print(
                f"  SPAM processed: "
                f"{index:,}/"
                f"{len(spam_files):,}"
            )


    ham_results = [
        result
        for result in all_results
        if result["expected"] == "ham"
    ]

    spam_results = [
        result
        for result in all_results
        if result["expected"] == "spam"
    ]

    ham_escalations = [
        result
        for result in ham_results
        if result[
            "final_classification"
        ] != "HAM"
    ]

    spam_allowed = [
        result
        for result in spam_results
        if result[
            "final_classification"
        ] == "HAM"
    ]

    disagreements = [
        result
        for result in all_results
        if (
            result["ml_label"]
            != (
                "spam"
                if result[
                    "final_classification"
                ] == "SPAM"
                else "ham"
            )
        )
    ]


    escalation_rules = Counter()

    for result in ham_escalations:
        for rule in result[
            "triggered_rules"
        ]:
            escalation_rules[rule] += 1

    spam_allow_rules = Counter()

    for result in spam_allowed:
        for rule in result[
            "triggered_rules"
        ]:
            spam_allow_rules[rule] += 1

    conflict_counter = Counter(
        result["conflict_type"]
        for result in all_results
        if result["conflict_type"]
        != "OTHER"
    )


    strong_ham_escalations = []

    for result in ham_escalations:

        if (
            result["ml_label"] == "ham"
            and result["ml_score"] < -0.5
        ):
            strong_ham_escalations.append(
                result
            )

    weak_ml_spam = []

    for result in spam_results:

        if (
            result["ml_label"] == "spam"
            and result["ml_score"] < 0.2
        ):
            weak_ml_spam.append(
                result
            )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "DATASET SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        f"\nTotal evaluated : "
        f"{len(all_results):,}"
    )

    print(
        f"HAM evaluated   : "
        f"{len(ham_results):,}"
    )

    print(
        f"SPAM evaluated   : "
        f"{len(spam_results):,}"
    )

    print(
        f"Evaluation errors: "
        f"{len(errors):,}"
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "HAM ESCALATION ANALYSIS"
    )

    print(
        "=" * 78
    )

    ham_rate = (
        len(ham_escalations)
        / len(ham_results)
        * 100
        if ham_results
        else 0
    )

    print(
        f"\nHAM escalations : "
        f"{len(ham_escalations):,}"
    )

    print(
        f"HAM escalation rate : "
        f"{ham_rate:.2f}%"
    )

    print(
        "\nEscalation-triggering indicators:"
    )

    if escalation_rules:
        for rule, count in (
            escalation_rules.most_common()
        ):
            print(
                f"  {rule:<30}"
                f"{count:>6}"
            )
    else:
        print(
            "  None"
        )

    print(
        "\nHAM escalation cases:"
    )

    if ham_escalations:

        for case in ham_escalations[
            :MAX_DISPLAY
        ]:
            print_case(case)

    else:
        print(
            "  None"
        )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "STRONG HAM ML VS SECURITY-RISK CONFLICTS"
    )

    print(
        "=" * 78
    )

    print(
        "\nHAM emails with ML score < -0.5 "
        "that were escalated:"
    )

    print(
        f"Count: "
        f"{len(strong_ham_escalations):,}"
    )

    if strong_ham_escalations:

        for case in strong_ham_escalations[
            :MAX_DISPLAY
        ]:
            print_case(case)

    else:
        print(
            "  None"
        )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "SPAM ALLOWED ANALYSIS"
    )

    print(
        "=" * 78
    )

    spam_allow_rate = (
        len(spam_allowed)
        / len(spam_results)
        * 100
        if spam_results
        else 0
    )

    print(
        f"\nSPAM allowed : "
        f"{len(spam_allowed):,}"
    )

    print(
        f"SPAM allowed rate : "
        f"{spam_allow_rate:.2f}%"
    )

    print(
        "\nIndicators present in SPAM allowed cases:"
    )

    if spam_allow_rules:

        for rule, count in (
            spam_allow_rules.most_common()
        ):
            print(
                f"  {rule:<30}"
                f"{count:>6}"
            )

    else:
        print(
            "  None"
        )

    print(
        "\nSPAM allowed cases:"
    )

    if spam_allowed:

        for case in spam_allowed[
            :MAX_DISPLAY
        ]:
            print_case(case)

    else:
        print(
            "  None"
        )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "ML VS FINAL DECISION DISAGREEMENTS"
    )

    print(
        "=" * 78
    )

    print(
        f"\nTotal disagreements : "
        f"{len(disagreements):,}"
    )

    print(
        "\nConflict categories:"
    )

    for category, count in (
        conflict_counter.most_common()
    ):
        print(
            f"  {category:<35}"
            f"{count:>6}"
        )

    if disagreements:

        print(
            "\nFirst disagreement cases:"
        )

        for case in disagreements[
            :MAX_DISPLAY
        ]:
            print_case(case)


    print(
        "\n"
        + "=" * 78
    )

    print(
        "WEAK ML SPAM PREDICTIONS"
    )

    print(
        "=" * 78
    )

    print(
        "\nSPAM emails classified as SPAM by ML "
        "with score < 0.2:"
    )

    print(
        f"Count: "
        f"{len(weak_ml_spam):,}"
    )

    if weak_ml_spam:

        for case in weak_ml_spam[
            :MAX_DISPLAY
        ]:
            print_case(case)

    else:
        print(
            "  None"
        )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "DECISION ENGINE DIAGNOSTIC SUMMARY"
    )

    print(
        "=" * 78
    )

    print(
        "\n1. HAM safety"
    )

    print(
        f"   HAM escalations: "
        f"{len(ham_escalations):,}/"
        f"{len(ham_results):,}"
    )

    print(
        f"   Rate: "
        f"{ham_rate:.2f}%"
    )

    print(
        "\n2. SPAM handling"
    )

    print(
        f"   SPAM allowed: "
        f"{len(spam_allowed):,}/"
        f"{len(spam_results):,}"
    )

    print(
        f"   Rate: "
        f"{spam_allow_rate:.2f}%"
    )

    print(
        "\n3. ML vs decision layer"
    )

    print(
        f"   Disagreements: "
        f"{len(disagreements):,}"
    )

    print(
        "\n4. Strong HAM overrides"
    )

    print(
        f"   Strong HAM cases escalated: "
        f"{len(strong_ham_escalations):,}"
    )


    print(
        "\n"
        + "-" * 78
    )

    if strong_ham_escalations:

        print(
            "DIAGNOSTIC FLAG:"
        )

        print(
            "The decision engine escalated at least one "
            "HAM email despite a strongly negative ML "
            "decision score."
        )

        print(
            "\nThese cases should be manually reviewed "
            "before changing decision-engine rules."
        )

    else:

        print(
            "No strong HAM-vs-security conflict "
            "was detected."
        )

    if spam_allowed:

        print(
            "\nDIAGNOSTIC FLAG:"
        )

        print(
            "Some SPAM emails were allowed by the "
            "combined decision layer."
        )

        print(
            "These cases should be examined for "
            "missing security indicators."
        )

    else:

        print(
            "\nNo SPAM emails were allowed."
        )


    if errors:

        print(
            "\n"
            + "=" * 78
        )

        print(
            "EVALUATION ERRORS"
        )

        print(
            "=" * 78
        )

        for path, error in errors[
            :MAX_DISPLAY
        ]:
            print(
                f"\nFile: {path}"
            )

            print(
                f"Error: {error}"
            )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "DECISION ENGINE ERROR ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nDiagnostic only."
    )

    print(
        "No model artifacts were modified."
    )

    print(
        "No threshold was changed."
    )

    print(
        "No retraining was performed."
    )


if __name__ == "__main__":
    main()
