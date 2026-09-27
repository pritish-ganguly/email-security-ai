"""
Root-cause analysis for Email Security AI decision failures.

Purpose
-------
Analyze the unresolved HAM/SPAM failures identified by the
decision-engine evaluation.

This module is diagnostic only.

It does NOT:
    - retrain the ML model
    - modify model artifacts
    - modify the production threshold
    - modify the decision engine
    - change any production behavior

The goal is to identify recurring characteristics among
incorrect final decisions so that future decision-engine
changes can be evidence-driven.
"""

from pathlib import Path
from collections import Counter, defaultdict
import re

from src.models.predict import EmailPredictor
from src.data.email_parser import parse_email
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
    default="",
):
    """
    Safely retrieve a value from either a dictionary
    or an object.
    """

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
    """
    Build the same text representation used by
    production inference.
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


def count_words(
    text,
):
    """
    Count whitespace-separated words.
    """

    return len(
        re.findall(
            r"\b\w+\b",
            text or "",
        )
    )


def classify_failure_type(
    expected,
    final_classification,
):
    """
    Determine whether the final decision is:

        false positive
        false negative
    """

    expected = expected.lower()
    final_classification = (
        final_classification.lower()
    )

    if expected == "ham":
        if final_classification != "ham":
            return "HAM_FALSE_POSITIVE"

    elif expected == "spam":
        if final_classification != "spam":
            return "SPAM_FALSE_NEGATIVE"

    return "UNKNOWN"


def classify_ml_failure(
    expected,
    ml_label,
):
    """
    Determine whether the ML model itself failed.
    """

    expected = expected.lower()
    ml_label = ml_label.lower()

    if expected == "ham":
        return ml_label != "ham"

    if expected == "spam":
        return ml_label != "spam"

    return False


def collect_indicators(
    security_analysis,
):
    """
    Extract security indicators into a normal dictionary.
    """

    return {
        "url_count": int(
            get_value(
                security_analysis,
                "url_count",
                0,
            )
        ),
        "ip_url_count": int(
            get_value(
                security_analysis,
                "ip_based_url_count",
                0,
            )
        ),
        "suspicious_keywords": int(
            get_value(
                security_analysis,
                "suspicious_keywords",
                0,
            )
        ),
        "attachments": int(
            get_value(
                security_analysis,
                "attachments",
                0,
            )
        ),
        "html": bool(
            get_value(
                security_analysis,
                "html_present",
                False,
            )
        ),
        "script": bool(
            get_value(
                security_analysis,
                "script_present",
                False,
            )
        ),
        "subject_length": int(
            get_value(
                security_analysis,
                "subject_length",
                0,
            )
        ),
        "body_length": int(
            get_value(
                security_analysis,
                "body_length",
                0,
            )
        ),
    }


def determine_pattern(
    record,
):
    """
    Identify broad recurring characteristics.

    This is diagnostic categorization only.
    """

    indicators = record["indicators"]

    patterns = []

    if record["word_count"] <= 40:
        patterns.append("short_email")

    if indicators["url_count"] == 0:
        patterns.append("no_url")

    elif indicators["url_count"] == 1:
        patterns.append("single_url")

    elif indicators["url_count"] >= 5:
        patterns.append("many_urls")

    elif indicators["url_count"] >= 2:
        patterns.append("multiple_urls")

    if indicators["ip_url_count"] > 0:
        patterns.append("ip_based_url")

    if indicators["suspicious_keywords"] > 0:
        patterns.append("suspicious_keywords")

    if indicators["attachments"] > 0:
        patterns.append("attachment")

    if indicators["html"]:
        patterns.append("html")

    if indicators["script"]:
        patterns.append("script")

    if (
        indicators["body_length"] <= 250
    ):
        patterns.append("short_body")

    if (
        indicators["body_length"] > 1000
    ):
        patterns.append("long_body")

    score = record["ml_score"]

    if score >= 0:
        patterns.append("positive_ml_score")

    elif score >= -0.1:
        patterns.append("near_threshold")

    elif score >= -0.5:
        patterns.append("moderately_negative_ml_score")

    else:
        patterns.append("strongly_negative_ml_score")

    return patterns


def analyze_single_email(
    email_path,
    expected,
    predictor,
):
    """
    Run the complete production pipeline on one email.
    """

    try:

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

        final_classification = str(
            decision.classification
        ).lower()

        failure_type = classify_failure_type(
            expected=expected,
            final_classification=final_classification,
        )

        ml_failure = classify_ml_failure(
            expected=expected,
            ml_label=ml_result["label"],
        )

        indicators = collect_indicators(
            security_analysis
        )

        full_text = (
            subject
            + "\n"
            + body
            + "\n"
            + html_body
        )

        record = {
            "file": str(email_path),
            "expected": expected,
            "ml_label": ml_result["label"],
            "ml_score": float(
                ml_result["decision_score"]
            ),
            "threshold": float(
                ml_result["threshold"]
            ),
            "final_classification":
                final_classification,
            "action": str(
                decision.action
            ),
            "confidence": str(
                decision.confidence
            ),
            "risk_score": int(
                decision.risk_score
            ),
            "risk_level": str(
                decision.risk_level
            ),
            "ml_failure": ml_failure,
            "failure_type": failure_type,
            "word_count": count_words(
                full_text
            ),
            "indicators": indicators,
        }

        record["patterns"] = determine_pattern(
            record
        )

        return record

    except Exception as exc:

        return {
            "file": str(email_path),
            "expected": expected,
            "error": str(exc),
        }


def process_dataset(
    directory,
    expected,
    predictor,
):
    """
    Process every email in a dataset directory.
    """

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
    )

    results = []

    total = len(files)

    for index, email_path in enumerate(
        files,
        start=1,
    ):

        record = analyze_single_email(
            email_path=email_path,
            expected=expected,
            predictor=predictor,
        )

        results.append(record)

        if (
            index % PROGRESS_INTERVAL == 0
            or index == total
        ):
            print(
                f"  {expected.upper()} processed: "
                f"{index:,}/{total:,}"
            )

    return results


def print_failure_summary(
    failures,
):
    """
    Print high-level failure statistics.
    """

    ham_failures = [
        record
        for record in failures
        if record["expected"] == "ham"
    ]

    spam_failures = [
        record
        for record in failures
        if record["expected"] == "spam"
    ]

    ml_failures = [
        record
        for record in failures
        if record["ml_failure"]
    ]

    engine_overrides = [
        record
        for record in failures
        if not record["ml_failure"]
    ]

    print("\n" + "=" * 78)
    print(
        "FAILURE ROOT-CAUSE SUMMARY"
    )
    print("=" * 78)

    print(
        f"\nTotal final failures : "
        f"{len(failures)}"
    )

    print(
        f"HAM final failures   : "
        f"{len(ham_failures)}"
    )

    print(
        f"SPAM final failures  : "
        f"{len(spam_failures)}"
    )

    print(
        f"ML-originated failures : "
        f"{len(ml_failures)}"
    )

    print(
        f"Decision-engine-originated failures : "
        f"{len(engine_overrides)}"
    )


def print_pattern_analysis(
    failures,
):
    """
    Print recurring characteristics across failures.
    """

    pattern_counter = Counter()

    for record in failures:

        for pattern in record["patterns"]:
            pattern_counter[pattern] += 1

    print("\n" + "=" * 78)
    print(
        "RECURRING FAILURE PATTERNS"
    )
    print("=" * 78)

    if not pattern_counter:
        print("\nNo recurring patterns identified.")

        return

    for pattern, count in (
        pattern_counter.most_common()
    ):

        print(
            f"{pattern:<35} : {count}"
        )


def print_failure_groups(
    failures,
):
    """
    Print failures grouped by root category.
    """

    groups = defaultdict(list)

    for record in failures:

        if record["ml_failure"]:
            groups["ML_FAILURE"].append(
                record
            )

        else:
            groups["DECISION_ENGINE_FAILURE"].append(
                record
            )

    print("\n" + "=" * 78)
    print(
        "FAILURES BY ORIGIN"
    )
    print("=" * 78)

    for group_name, records in groups.items():

        print(
            f"\n{group_name}"
        )

        print("-" * 78)

        for record in records:

            print(
                f"\nFile           : "
                f"{record['file']}"
            )

            print(
                f"Expected       : "
                f"{record['expected']}"
            )

            print(
                f"ML prediction  : "
                f"{record['ml_label']}"
            )

            print(
                f"ML score       : "
                f"{record['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{record['final_classification']}"
            )

            print(
                f"Action         : "
                f"{record['action']}"
            )

            print(
                f"Risk           : "
                f"{record['risk_score']}/100 "
                f"({record['risk_level']})"
            )

            print(
                f"Word count     : "
                f"{record['word_count']}"
            )

            indicators = record[
                "indicators"
            ]

            print(
                f"URLs           : "
                f"{indicators['url_count']}"
            )

            print(
                f"IP URLs        : "
                f"{indicators['ip_url_count']}"
            )

            print(
                f"Suspicious KW  : "
                f"{indicators['suspicious_keywords']}"
            )

            print(
                f"Attachments    : "
                f"{indicators['attachments']}"
            )

            print(
                f"HTML           : "
                f"{indicators['html']}"
            )

            print(
                f"Script         : "
                f"{indicators['script']}"
            )

            print(
                f"Patterns       : "
                f"{', '.join(record['patterns'])}"
            )


def print_failure_type_analysis(
    failures,
):
    """
    Analyze HAM false positives and SPAM false negatives.
    """

    false_positive_patterns = Counter()
    false_negative_patterns = Counter()

    for record in failures:

        if (
            record["failure_type"]
            == "HAM_FALSE_POSITIVE"
        ):

            for pattern in record[
                "patterns"
            ]:
                false_positive_patterns[
                    pattern
                ] += 1

        elif (
            record["failure_type"]
            == "SPAM_FALSE_NEGATIVE"
        ):

            for pattern in record[
                "patterns"
            ]:
                false_negative_patterns[
                    pattern
                ] += 1

    print("\n" + "=" * 78)
    print(
        "HAM FALSE-POSITIVE PATTERNS"
    )
    print("=" * 78)

    if false_positive_patterns:

        for pattern, count in (
            false_positive_patterns.most_common()
        ):

            print(
                f"{pattern:<35} : {count}"
            )

    else:

        print(
            "\nNo HAM false-positive patterns."
        )

    print("\n" + "=" * 78)
    print(
        "SPAM FALSE-NEGATIVE PATTERNS"
    )
    print("=" * 78)

    if false_negative_patterns:

        for pattern, count in (
            false_negative_patterns.most_common()
        ):

            print(
                f"{pattern:<35} : {count}"
            )

    else:

        print(
            "\nNo SPAM false-negative patterns."
        )


def print_score_analysis(
    failures,
):
    """
    Analyze ML score ranges among failures.
    """

    print("\n" + "=" * 78)
    print(
        "ML SCORE ANALYSIS OF FAILURES"
    )
    print("=" * 78)

    ham_failures = [
        record
        for record in failures
        if record["expected"] == "ham"
    ]

    spam_failures = [
        record
        for record in failures
        if record["expected"] == "spam"
    ]

    if ham_failures:

        scores = [
            record["ml_score"]
            for record in ham_failures
        ]

        print("\nHAM failures")

        print(
            f"Minimum : "
            f"{min(scores):.4f}"
        )

        print(
            f"Maximum : "
            f"{max(scores):.4f}"
        )

        print(
            f"Average : "
            f"{sum(scores) / len(scores):.4f}"
        )

    if spam_failures:

        scores = [
            record["ml_score"]
            for record in spam_failures
        ]

        print("\nSPAM failures")

        print(
            f"Minimum : "
            f"{min(scores):.4f}"
        )

        print(
            f"Maximum : "
            f"{max(scores):.4f}"
        )

        print(
            f"Average : "
            f"{sum(scores) / len(scores):.4f}"
        )


def main():
    print("\n" + "=" * 78)
    print(
        "EMAIL SECURITY AI — FAILURE ROOT-CAUSE ANALYSIS"
    )
    print("=" * 78)

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    print(
        f"\nHAM emails discovered : "
        f"{len(list(HAM_DIR.iterdir())):,}"
    )

    print(
        f"SPAM emails discovered: "
        f"{len(list(SPAM_DIR.iterdir())):,}"
    )

    print("\nAnalyzing HAM emails...")

    ham_results = process_dataset(
        directory=HAM_DIR,
        expected="ham",
        predictor=predictor,
    )

    print("\nAnalyzing SPAM emails...")

    spam_results = process_dataset(
        directory=SPAM_DIR,
        expected="spam",
        predictor=predictor,
    )

    all_results = (
        ham_results
        + spam_results
    )

    errors = [
        record
        for record in all_results
        if "error" in record
    ]

    failures = [
        record
        for record in all_results
        if (
            "error" not in record
            and record["failure_type"]
            in {
                "HAM_FALSE_POSITIVE",
                "SPAM_FALSE_NEGATIVE",
            }
        )
    ]

    print("\n" + "=" * 78)
    print(
        "EMAIL SECURITY AI — ROOT-CAUSE RESULTS"
    )
    print("=" * 78)

    print(
        f"\nTotal evaluated : "
        f"{len(all_results):,}"
    )

    print(
        f"Evaluation errors : "
        f"{len(errors):,}"
    )

    print(
        f"Final failures : "
        f"{len(failures):,}"
    )

    print_failure_summary(
        failures
    )

    print_pattern_analysis(
        failures
    )

    print_failure_type_analysis(
        failures
    )

    print_score_analysis(
        failures
    )

    print_failure_groups(
        failures
    )

    print("\n" + "=" * 78)
    print(
        "ROOT-CAUSE ANALYSIS COMPLETE"
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


if __name__ == "__main__":
    main()
