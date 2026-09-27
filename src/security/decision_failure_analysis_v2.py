"""
Decision Engine V2 — Failure Analysis

Analyzes the cases where Decision Engine V2 disagrees with
the expected dataset label or where the final security action
does not match the expected security outcome.

Purpose:
    1. Identify HAM emails incorrectly escalated.
    2. Identify SPAM emails incorrectly allowed.
    3. Identify ML false positives.
    4. Identify ML false negatives.
    5. Identify cases corrected by the V2 decision engine.
    6. Identify cases where the decision engine introduced
       an escalation that may require further tuning.

This script DOES NOT:
    - retrain the model
    - modify the model
    - modify the threshold
    - modify decision_engine_v2.py
    - modify any model artifacts

It is an analysis-only diagnostic tool.
"""

from pathlib import Path
from collections import Counter

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine_v2 import make_decision_v2


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)

MAX_FAILURES_TO_DISPLAY = 50


def get_value(
    email_data,
    field,
    default="",
):
    """Safely retrieve a field from parsed email data."""

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
    """Build the same text representation used during inference."""

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


def calculate_evidence_level(
    security_analysis,
    risk_assessment,
):
    """
    Determine the security evidence level.

    This mirrors the conceptual evidence categories used
    by Decision Engine V2.
    """

    risk_score = int(
        get_value(
            risk_assessment,
            "risk_score",
            0,
        )
    )

    script_present = bool(
        get_value(
            security_analysis,
            "script_present",
            False,
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

    url_count = int(
        get_value(
            security_analysis,
            "url_count",
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

    strong = (
        script_present
        or ip_url_count > 0
        or suspicious_keywords >= 3
        or risk_score >= 70
    )

    moderate = (
        url_count >= 3
        or suspicious_keywords >= 1
        or attachment_count > 0
        or risk_score >= 40
    )

    if strong:
        return "STRONG"

    if moderate:
        return "MODERATE"

    return "WEAK"


def is_spam_ml(
    ml_result,
):
    """Return whether the ML model classified the email as spam."""

    return bool(
        ml_result.get(
            "spam",
            ml_result.get(
                "label",
                "",
            ) == "spam",
        )
    )


def is_final_spam(
    decision,
):
    """Return whether V2 final classification is spam."""

    classification = str(
        get_value(
            decision,
            "classification",
            "",
        )
    ).upper()

    return classification == "SPAM"


def is_final_suspicious(
    decision,
):
    """Return whether V2 final classification is suspicious."""

    classification = str(
        get_value(
            decision,
            "classification",
            "",
        )
    ).upper()

    return classification == "SUSPICIOUS"


def is_final_allowed(
    decision,
):
    """Return whether V2 allowed the email."""

    action = str(
        get_value(
            decision,
            "action",
            "",
        )
    ).upper()

    return action == "ALLOW"


def is_final_escalated(
    decision,
):
    """Return whether V2 escalated the email."""

    action = str(
        get_value(
            decision,
            "action",
            "",
        )
    ).upper()

    return action != "ALLOW"


def evaluate_email(
    email_path,
    expected_label,
    predictor,
):
    """
    Evaluate a single email.

    Returns a structured diagnostic record.
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

    decision = make_decision_v2(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    ml_spam = is_spam_ml(
        ml_result
    )

    final_spam = is_final_spam(
        decision
    )

    final_suspicious = is_final_suspicious(
        decision
    )

    final_allowed = is_final_allowed(
        decision
    )

    final_escalated = is_final_escalated(
        decision
    )

    expected_spam = (
        expected_label == "spam"
    )

    return {
        "file": str(email_path),
        "expected": expected_label,
        "ml_prediction": (
            "spam"
            if ml_spam
            else "ham"
        ),
        "ml_score": float(
            ml_result.get(
                "decision_score",
                0.0,
            )
        ),
        "final_class": str(
            get_value(
                decision,
                "classification",
                "UNKNOWN",
            )
        ).upper(),
        "action": str(
            get_value(
                decision,
                "action",
                "UNKNOWN",
            )
        ).upper(),
        "confidence": str(
            get_value(
                decision,
                "confidence",
                "UNKNOWN",
            )
        ).upper(),
        "risk_score": int(
            get_value(
                risk_assessment,
                "risk_score",
                0,
            )
        ),
        "risk_level": str(
            get_value(
                risk_assessment,
                "risk_level",
                "UNKNOWN",
            )
        ).upper(),
        "evidence": calculate_evidence_level(
            security_analysis,
            risk_assessment,
        ),
        "ml_correct": (
            ml_spam == expected_spam
        ),
        "final_correct": (
            final_spam == expected_spam
            or (
                expected_label == "ham"
                and final_allowed
            )
            or (
                expected_label == "spam"
                and final_escalated
            )
        ),
        "ml_false_positive": (
            expected_label == "ham"
            and ml_spam
        ),
        "ml_false_negative": (
            expected_label == "spam"
            and not ml_spam
        ),
        "ham_escalation": (
            expected_label == "ham"
            and final_escalated
        ),
        "spam_allowed": (
            expected_label == "spam"
            and final_allowed
        ),
        "v2_corrected_ml_error": (
            not (
                ml_spam == expected_spam
            )
            and (
                (
                    expected_label == "spam"
                    and final_escalated
                )
                or (
                    expected_label == "ham"
                    and final_allowed
                )
            )
        ),
        "final_suspicious": final_suspicious,
    }


def print_failure(
    record,
):
    """Print one failure-analysis record."""

    print(
        f"\nFile           : {record['file']}"
    )

    print(
        f"Expected       : {record['expected']}"
    )

    print(
        f"ML prediction  : {record['ml_prediction']}"
    )

    print(
        f"ML score       : {record['ml_score']:.4f}"
    )

    print(
        f"Final class    : {record['final_class']}"
    )

    print(
        f"Action         : {record['action']}"
    )

    print(
        f"Confidence     : {record['confidence']}"
    )

    print(
        f"Risk score     : {record['risk_score']}/100"
    )

    print(
        f"Risk level     : {record['risk_level']}"
    )

    print(
        f"Evidence       : {record['evidence']}"
    )


def print_section(
    title,
):
    """Print a standard section heading."""

    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def print_subsection(
    title,
):
    """Print a standard subsection heading."""

    print("\n" + "-" * 78)
    print(title)
    print("-" * 78)


def main():
    print_section(
        "EMAIL SECURITY AI — DECISION ENGINE V2 FAILURE ANALYSIS"
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    ham_files = sorted(
        HAM_DIR.iterdir()
    )

    spam_files = sorted(
        SPAM_DIR.iterdir()
    )

    print(
        f"\nHAM emails discovered : "
        f"{len(ham_files):,}"
    )

    print(
        f"SPAM emails discovered: "
        f"{len(spam_files):,}"
    )

    records = []

    errors = []


    print(
        "\nAnalyzing HAM emails..."
    )

    for index, email_path in enumerate(
        ham_files,
        start=1,
    ):

        try:

            record = evaluate_email(
                email_path=email_path,
                expected_label="ham",
                predictor=predictor,
            )

            records.append(
                record
            )

        except Exception as exc:

            errors.append(
                (
                    str(email_path),
                    str(exc),
                )
            )

        if (
            index % 250 == 0
            or index == len(ham_files)
        ):

            print(
                f"  HAM processed: "
                f"{index:,}/{len(ham_files):,}"
            )


    print(
        "\nAnalyzing SPAM emails..."
    )

    for index, email_path in enumerate(
        spam_files,
        start=1,
    ):

        try:

            record = evaluate_email(
                email_path=email_path,
                expected_label="spam",
                predictor=predictor,
            )

            records.append(
                record
            )

        except Exception as exc:

            errors.append(
                (
                    str(email_path),
                    str(exc),
                )
            )

        if (
            index % 250 == 0
            or index == len(spam_files)
        ):

            print(
                f"  SPAM processed: "
                f"{index:,}/{len(spam_files):,}"
            )


    total = len(records)

    ml_false_positives = [
        r for r in records
        if r["ml_false_positive"]
    ]

    ml_false_negatives = [
        r for r in records
        if r["ml_false_negative"]
    ]

    ham_escalations = [
        r for r in records
        if r["ham_escalation"]
    ]

    spam_allowed = [
        r for r in records
        if r["spam_allowed"]
    ]

    corrected_errors = [
        r for r in records
        if r["v2_corrected_ml_error"]
    ]

    suspicious_records = [
        r for r in records
        if r["final_suspicious"]
    ]


    print_section(
        "FAILURE ANALYSIS SUMMARY"
    )

    print(
        f"\nTotal emails analyzed : "
        f"{total:,}"
    )

    print(
        f"Evaluation errors     : "
        f"{len(errors):,}"
    )

    print_subsection(
        "MACHINE LEARNING ERRORS"
    )

    print(
        f"\nML false positives : "
        f"{len(ml_false_positives):,}"
    )

    print(
        f"ML false negatives : "
        f"{len(ml_false_negatives):,}"
    )

    print_subsection(
        "FINAL DECISION ERRORS"
    )

    print(
        f"\nHAM escalations : "
        f"{len(ham_escalations):,}"
    )

    print(
        f"SPAM allowed    : "
        f"{len(spam_allowed):,}"
    )

    print(
        f"SUSPICIOUS      : "
        f"{len(suspicious_records):,}"
    )

    print_subsection(
        "V2 CORRECTIONS"
    )

    print(
        f"\nML errors corrected by V2 : "
        f"{len(corrected_errors):,}"
    )


    print_section(
        "ML FALSE POSITIVES"
    )

    if not ml_false_positives:

        print(
            "\nNone detected."
        )

    else:

        for record in ml_false_positives[
            :MAX_FAILURES_TO_DISPLAY
        ]:

            print_failure(
                record
            )


    print_section(
        "ML FALSE NEGATIVES"
    )

    if not ml_false_negatives:

        print(
            "\nNone detected."
        )

    else:

        for record in ml_false_negatives[
            :MAX_FAILURES_TO_DISPLAY
        ]:

            print_failure(
                record
            )


    print_section(
        "HAM ESCALATIONS"
    )

    if not ham_escalations:

        print(
            "\nNone detected."
        )

    else:

        for record in ham_escalations[
            :MAX_FAILURES_TO_DISPLAY
        ]:

            print_failure(
                record
            )


    print_section(
        "SPAM ALLOWED"
    )

    if not spam_allowed:

        print(
            "\nNone detected."
        )

    else:

        for record in spam_allowed[
            :MAX_FAILURES_TO_DISPLAY
        ]:

            print_failure(
                record
            )


    print_section(
        "V2 CORRECTED ML ERRORS"
    )

    if not corrected_errors:

        print(
            "\nNo ML errors were corrected."
        )

    else:

        for record in corrected_errors[
            :MAX_FAILURES_TO_DISPLAY
        ]:

            print_failure(
                record
            )


    print_section(
        "FAILURE DISTRIBUTION BY SECURITY EVIDENCE"
    )

    evidence_counter = Counter(
        record["evidence"]
        for record in records
    )

    for evidence in (
        "WEAK",
        "MODERATE",
        "STRONG",
    ):

        count = evidence_counter.get(
            evidence,
            0,
        )

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{evidence:<15}"
            f"{count:>8,}"
            f"{percentage:>10.2f}%"
        )


    print_section(
        "FAILURE DISTRIBUTION BY RISK LEVEL"
    )

    risk_counter = Counter(
        record["risk_level"]
        for record in records
    )

    for risk_level in (
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
    ):

        count = risk_counter.get(
            risk_level,
            0,
        )

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{risk_level:<15}"
            f"{count:>8,}"
            f"{percentage:>10.2f}%"
        )


    if errors:

        print_section(
            "EVALUATION ERRORS"
        )

        for file_path, error in errors[
            :MAX_FAILURES_TO_DISPLAY
        ]:

            print(
                f"\nFile  : {file_path}"
            )

            print(
                f"Error : {error}"
            )


    print_section(
        "FAILURE ANALYSIS STATUS"
    )

    print(
        "\nDecision Engine V2 failure analysis completed."
    )

    print(
        "\nThis analysis did not modify:"
    )

    print(
        "  - ML model artifacts"
    )

    print(
        "  - NLP pipeline"
    )

    print(
        "  - Decision threshold"
    )

    print(
        "  - Decision Engine V2 logic"
    )

    print(
        "  - Training data"
    )

    print(
        "\nNo retraining was performed."
    )


if __name__ == "__main__":
    main()
