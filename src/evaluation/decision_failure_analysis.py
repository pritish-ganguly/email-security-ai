"""
Decision Failure Analysis for Email Security AI.

Purpose
-------
Analyze cases where the final decision engine differs from the
ground-truth dataset label.

This script is diagnostic only.

It does NOT:
    - retrain the ML model
    - modify model artifacts
    - modify the production threshold
    - modify the decision engine

It analyzes:
    1. False-positive final decisions
    2. False-negative final decisions
    3. ML failures corrected by the decision engine
    4. ML decisions overridden incorrectly
    5. Risk-level patterns
    6. Security-indicator patterns
"""

from pathlib import Path
from collections import Counter, defaultdict

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine import make_decision


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)

PROGRESS_INTERVAL = 250


def get_value(
    obj,
    field,
    default=None,
):
    """Safely retrieve a dictionary or object field."""

    if isinstance(obj, dict):
        return obj.get(
            field,
            default,
        )

    return getattr(
        obj,
        field,
        default,
    )


def build_email_text(
    email_data,
):
    """Build inference text using the production representation."""

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


def percentage(
    numerator,
    denominator,
):
    """Safely calculate a percentage."""

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
        * 100.0
    )


def increment(
    counter,
    key,
):
    """Increment a Counter value."""

    counter[key] += 1


def evaluate_email(
    email_path,
    expected_label,
    predictor,
):
    """
    Evaluate one email through the complete production pipeline.

    Returns a structured diagnostic dictionary.
    """

    email_data = parse_email(
        email_path
    )

    email_text = build_email_text(
        email_data
    )

    if not email_text.strip():
        raise ValueError(
            "Email contains no usable text."
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


    ml_prediction = str(
        ml_result.get(
            "label",
            "unknown",
        )
    ).lower()

    final_classification = str(
        decision.classification
    ).lower()

    expected_label = str(
        expected_label
    ).lower()

    risk_level = str(
        decision.risk_level
    ).upper()

    risk_score = int(
        decision.risk_score
    )

    action = str(
        decision.action
    )

    ml_score = float(
        ml_result.get(
            "decision_score",
            0.0,
        )
    )


    url_count = int(
        get_value(
            security_analysis,
            "url_count",
            0,
        )
    )

    ip_url_count = int(
        get_value(
            security_analysis,
            "ip_based_url_count",
            0,
        )
    )

    suspicious_keywords = int(
        get_value(
            security_analysis,
            "suspicious_keywords",
            0,
        )
    )

    attachment_count = int(
        get_value(
            security_analysis,
            "attachments",
            0,
        )
    )

    html_present = bool(
        get_value(
            security_analysis,
            "html_present",
            False,
        )
    )

    script_present = bool(
        get_value(
            security_analysis,
            "script_present",
            False,
        )
    )

    body_length = int(
        get_value(
            security_analysis,
            "body_length",
            0,
        )
    )

    subject_length = int(
        get_value(
            security_analysis,
            "subject_length",
            0,
        )
    )

    word_count = int(
        get_value(
            security_analysis,
            "word_count",
            0,
        )
    )


    final_correct = (
        (
            expected_label == "ham"
            and final_classification == "ham"
        )
        or
        (
            expected_label == "spam"
            and final_classification == "spam"
        )
    )

    ml_correct = (
        ml_prediction
        == expected_label
    )


    if final_classification == "suspicious":
        final_correct = False


    category = "NORMAL"

    if ml_correct and not final_correct:
        category = (
            "DECISION_ENGINE_HARM"
        )

    elif not ml_correct and final_correct:
        category = (
            "DECISION_ENGINE_CORRECTION"
        )

    elif not ml_correct and not final_correct:
        category = (
            "UNRESOLVED_FAILURE"
        )

    return {
        "file": str(email_path),
        "expected": expected_label,
        "ml_prediction": ml_prediction,
        "ml_score": ml_score,
        "final_classification": final_classification,
        "action": action,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "ml_correct": ml_correct,
        "final_correct": final_correct,
        "category": category,
        "url_count": url_count,
        "ip_url_count": ip_url_count,
        "suspicious_keywords": suspicious_keywords,
        "attachment_count": attachment_count,
        "html_present": html_present,
        "script_present": script_present,
        "body_length": body_length,
        "subject_length": subject_length,
        "word_count": word_count,
    }


def main():
    print("\n" + "=" * 78)
    print(
        "EMAIL SECURITY AI — DECISION FAILURE ANALYSIS"
    )
    print("=" * 78)

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()


    ham_files = sorted(
        path
        for path in HAM_DIR.iterdir()
        if path.is_file()
    )

    spam_files = sorted(
        path
        for path in SPAM_DIR.iterdir()
        if path.is_file()
    )

    print(
        f"\nHAM emails discovered : "
        f"{len(ham_files):,}"
    )

    print(
        f"SPAM emails discovered: "
        f"{len(spam_files):,}"
    )


    results = []

    evaluation_errors = []

    category_counts = Counter()

    expected_counts = Counter()

    ml_counts = Counter()

    final_counts = Counter()

    action_counts = Counter()

    risk_counts = Counter()


    datasets = [
        (
            "HAM",
            ham_files,
            "ham",
        ),
        (
            "SPAM",
            spam_files,
            "spam",
        ),
    ]

    for dataset_name, files, expected_label in datasets:

        print(
            f"\nEvaluating {dataset_name} emails..."
        )

        for index, email_path in enumerate(
            files,
            start=1,
        ):

            try:

                result = evaluate_email(
                    email_path=email_path,
                    expected_label=expected_label,
                    predictor=predictor,
                )

                results.append(
                    result
                )

                expected_counts[
                    expected_label
                ] += 1

                ml_counts[
                    result["ml_prediction"]
                ] += 1

                final_counts[
                    result["final_classification"]
                ] += 1

                action_counts[
                    result["action"]
                ] += 1

                risk_counts[
                    result["risk_level"]
                ] += 1

                category_counts[
                    result["category"]
                ] += 1

            except Exception as exc:

                evaluation_errors.append(
                    {
                        "file": str(email_path),
                        "error": str(exc),
                    }
                )

            if (
                index % PROGRESS_INTERVAL == 0
                or index == len(files)
            ):
                print(
                    f"  {dataset_name} processed: "
                    f"{index:,}/{len(files):,}"
                )


    total = len(results)

    ham_results = [
        result
        for result in results
        if result["expected"] == "ham"
    ]

    spam_results = [
        result
        for result in results
        if result["expected"] == "spam"
    ]


    ham_correct = sum(
        result["final_correct"]
        for result in ham_results
    )

    spam_correct = sum(
        result["final_correct"]
        for result in spam_results
    )

    ham_failures = [
        result
        for result in ham_results
        if not result["final_correct"]
    ]

    spam_failures = [
        result
        for result in spam_results
        if not result["final_correct"]
    ]


    corrections = [
        result
        for result in results
        if result["category"]
        == "DECISION_ENGINE_CORRECTION"
    ]

    harms = [
        result
        for result in results
        if result["category"]
        == "DECISION_ENGINE_HARM"
    ]

    unresolved = [
        result
        for result in results
        if result["category"]
        == "UNRESOLVED_FAILURE"
    ]


    print("\n" + "=" * 78)
    print(
        "EMAIL SECURITY AI — FAILURE ANALYSIS RESULTS"
    )
    print("=" * 78)

    print("\nDATASET")
    print("-" * 78)

    print(
        f"Total evaluated       : "
        f"{total:,}"
    )

    print(
        f"Evaluation errors     : "
        f"{len(evaluation_errors):,}"
    )

    print(
        f"HAM evaluated         : "
        f"{len(ham_results):,}"
    )

    print(
        f"SPAM evaluated        : "
        f"{len(spam_results):,}"
    )


    print("\n" + "=" * 78)
    print(
        "FINAL DECISION PERFORMANCE"
    )
    print("=" * 78)

    print(
        f"\nHAM correctly classified : "
        f"{ham_correct:,}"
    )

    print(
        f"HAM final failures       : "
        f"{len(ham_failures):,}"
    )

    print(
        f"HAM failure rate         : "
        f"{percentage(len(ham_failures), len(ham_results)):.2f}%"
    )

    print(
        f"\nSPAM correctly classified: "
        f"{spam_correct:,}"
    )

    print(
        f"SPAM final failures      : "
        f"{len(spam_failures):,}"
    )

    print(
        f"SPAM failure rate        : "
        f"{percentage(len(spam_failures), len(spam_results)):.2f}%"
    )


    print("\n" + "=" * 78)
    print(
        "DECISION ENGINE EFFECT"
    )
    print("=" * 78)

    print(
        f"\nML failures corrected by decision engine : "
        f"{len(corrections):,}"
    )

    print(
        f"Correct ML decisions harmed by engine    : "
        f"{len(harms):,}"
    )

    print(
        f"Failures remaining unresolved            : "
        f"{len(unresolved):,}"
    )


    print("\n" + "=" * 78)
    print(
        "MACHINE LEARNING DISTRIBUTION"
    )
    print("=" * 78)

    print(
        f"\nML HAM predictions : "
        f"{ml_counts['ham']:,}"
    )

    print(
        f"ML SPAM predictions: "
        f"{ml_counts['spam']:,}"
    )


    print("\n" + "=" * 78)
    print(
        "FINAL CLASSIFICATION DISTRIBUTION"
    )
    print("=" * 78)

    print(
        f"\nFinal HAM       : "
        f"{final_counts['ham']:,}"
    )

    print(
        f"Final SPAM      : "
        f"{final_counts['spam']:,}"
    )

    print(
        f"Final SUSPICIOUS: "
        f"{final_counts['suspicious']:,}"
    )


    print("\n" + "=" * 78)
    print(
        "ACTION DISTRIBUTION"
    )
    print("=" * 78)

    for action, count in sorted(
        action_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{action:<32}"
            f"{count:>7,}   "
            f"{percentage(count, total):>6.2f}%"
        )


    print("\n" + "=" * 78)
    print(
        "RISK LEVEL DISTRIBUTION"
    )
    print("=" * 78)

    for risk_level in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:
        count = risk_counts[
            risk_level
        ]

        print(
            f"{risk_level:<16}"
            f"{count:>7,}   "
            f"{percentage(count, total):>6.2f}%"
        )


    print("\n" + "=" * 78)
    print(
        "ML FAILURES CORRECTED BY DECISION ENGINE"
    )
    print("=" * 78)

    if not corrections:
        print("\nNone.")

    else:

        for result in corrections:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"Expected       : "
                f"{result['expected']}"
            )

            print(
                f"ML prediction  : "
                f"{result['ml_prediction']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{result['final_classification'].upper()}"
            )

            print(
                f"Action         : "
                f"{result['action']}"
            )

            print(
                f"Risk           : "
                f"{result['risk_score']}/100 "
                f"({result['risk_level']})"
            )


    print("\n" + "=" * 78)
    print(
        "CORRECT ML DECISIONS HARMED BY DECISION ENGINE"
    )
    print("=" * 78)

    if not harms:
        print("\nNone.")

    else:

        for result in harms:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"Expected       : "
                f"{result['expected']}"
            )

            print(
                f"ML prediction  : "
                f"{result['ml_prediction']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{result['final_classification'].upper()}"
            )

            print(
                f"Action         : "
                f"{result['action']}"
            )

            print(
                f"Risk           : "
                f"{result['risk_score']}/100 "
                f"({result['risk_level']})"
            )

            print(
                f"URLs           : "
                f"{result['url_count']}"
            )

            print(
                f"IP URLs        : "
                f"{result['ip_url_count']}"
            )

            print(
                f"Suspicious KW  : "
                f"{result['suspicious_keywords']}"
            )

            print(
                f"Attachments    : "
                f"{result['attachment_count']}"
            )

            print(
                f"HTML           : "
                f"{result['html_present']}"
            )

            print(
                f"Script         : "
                f"{result['script_present']}"
            )


    print("\n" + "=" * 78)
    print(
        "UNRESOLVED FAILURES"
    )
    print("=" * 78)

    if not unresolved:
        print("\nNone.")

    else:

        for result in unresolved:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"Expected       : "
                f"{result['expected']}"
            )

            print(
                f"ML prediction  : "
                f"{result['ml_prediction']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{result['final_classification'].upper()}"
            )

            print(
                f"Action         : "
                f"{result['action']}"
            )

            print(
                f"Risk           : "
                f"{result['risk_score']}/100 "
                f"({result['risk_level']})"
            )


    print("\n" + "=" * 78)
    print(
        "HAM FINAL DECISION FAILURES"
    )
    print("=" * 78)

    if not ham_failures:
        print("\nNone.")

    else:

        for result in ham_failures:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"ML prediction  : "
                f"{result['ml_prediction']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{result['final_classification'].upper()}"
            )

            print(
                f"Action         : "
                f"{result['action']}"
            )

            print(
                f"Risk           : "
                f"{result['risk_score']}/100 "
                f"({result['risk_level']})"
            )


    print("\n" + "=" * 78)
    print(
        "SPAM FINAL DECISION FAILURES"
    )
    print("=" * 78)

    if not spam_failures:
        print("\nNone.")

    else:

        for result in spam_failures:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"ML prediction  : "
                f"{result['ml_prediction']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{result['final_classification'].upper()}"
            )

            print(
                f"Action         : "
                f"{result['action']}"
            )

            print(
                f"Risk           : "
                f"{result['risk_score']}/100 "
                f"({result['risk_level']})"
            )


    if evaluation_errors:

        print("\n" + "=" * 78)
        print(
            "EVALUATION ERRORS"
        )
        print("=" * 78)

        for error in evaluation_errors:

            print(
                f"\nFile  : "
                f"{error['file']}"
            )

            print(
                f"Error : "
                f"{error['error']}"
            )


    print("\n" + "=" * 78)
    print(
        "FAILURE ANALYSIS COMPLETE"
    )
    print("=" * 78)

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

    print()


if __name__ == "__main__":
    main()
