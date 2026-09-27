"""
Email Security AI — Decision Engine V2 Evaluation

Evaluates Decision Engine V2 against the complete
SpamAssassin dataset.

Dataset:
    HAM  : 2,551
    SPAM :   501
    Total: 3,052

Important:
    - Frozen ML model is used.
    - Frozen ML threshold is used.
    - No retraining.
    - No model artifacts are modified.
    - No threshold is changed.

Purpose:
    Compare the final V2 security decision against the
    expected SpamAssassin labels and expose:

        - HAM escalations
        - SPAM allowed
        - ML vs V2 disagreements
        - final classifications
        - final actions
        - risk distribution
"""


from pathlib import Path
from collections import Counter


from src.models.predict import EmailPredictor

from src.data.email_parser import parse_email

from src.security.security_analyzer import analyze_email

from src.security.risk_engine import assess_email_risk

from src.security.decision_engine_v2 import (
    make_decision_v2,
)


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)


PROGRESS_INTERVAL = 250

MAX_DISPLAY_ITEMS = 20


def get_value(
    email_data,
    field,
    default="",
):
    """
    Safely retrieve a field from the parsed email.
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
    Construct the same general email representation
    used by the production inference pipeline.
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


def discover_emails(
    directory: Path,
):
    """
    Discover email files recursively.

    The SpamAssassin dataset files do not necessarily have
    a .eml extension, so this evaluator intentionally does
    NOT require a .eml suffix.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
    ]

    return sorted(files)


def print_progress(
    label,
    current,
    total,
):
    """
    Print evaluation progress.
    """

    if (
        current % PROGRESS_INTERVAL == 0
        or current == total
    ):
        print(
            f"  {label} processed: "
            f"{current:,}/{total:,}"
        )


def evaluate_dataset(
    predictor,
    email_files,
    expected_label,
    label,
):
    """
    Evaluate one dataset partition.

    Returns a list of evaluation records.
    """

    results = []

    total = len(email_files)

    print(
        f"\nEvaluating {label.upper()} emails..."
    )

    for index, email_path in enumerate(
        email_files,
        start=1,
    ):

        try:
            email_data = parse_email(
                email_path
            )

            email_text = build_email_text(
                email_data
            )

            if not email_text.strip():
                raise ValueError(
                    "Empty email text."
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


            record = {
                "file": str(email_path),
                "expected": expected_label,

                "ml_prediction": ml_result.get(
                    "label",
                    "unknown",
                ),

                "ml_score": float(
                    ml_result.get(
                        "decision_score",
                        0.0,
                    )
                ),

                "final_classification": (
                    decision.classification
                ),

                "action": (
                    decision.action
                ),

                "confidence": (
                    decision.confidence
                ),

                "risk_score": int(
                    decision.risk_score
                ),

                "risk_level": (
                    decision.risk_level
                ),

                "evidence_level": (
                    decision.evidence_level
                ),

                "ml_margin": float(
                    decision.ml_margin
                ),
            }

            results.append(record)

        except Exception as exc:

            results.append(
                {
                    "file": str(email_path),
                    "expected": expected_label,
                    "error": str(exc),
                }
            )

        print_progress(
            label,
            index,
            total,
        )

    return results


def print_section(
    title,
):
    """
    Print a standard report section.
    """

    print("\n" + "=" * 78)

    print(title)

    print("=" * 78)


def print_counter(
    counter,
    total,
):
    """
    Print a counter with percentage.
    """

    for key, value in sorted(
        counter.items(),
        key=lambda item: item[0],
    ):

        percentage = (
            value / total * 100
            if total
            else 0
        )

        print(
            f"{str(key):25}"
            f"{value:8,}"
            f"   {percentage:6.2f}%"
        )


def print_evaluation_report(
    results,
):
    """
    Print complete V2 evaluation report.
    """

    valid_results = [
        result
        for result in results
        if "error" not in result
    ]

    errors = [
        result
        for result in results
        if "error" in result
    ]

    total = len(valid_results)


    ham_results = [
        result
        for result in valid_results
        if result["expected"] == "ham"
    ]

    spam_results = [
        result
        for result in valid_results
        if result["expected"] == "spam"
    ]


    ml_counter = Counter(
        result["ml_prediction"]
        for result in valid_results
    )


    final_counter = Counter(
        result["final_classification"]
        for result in valid_results
    )


    action_counter = Counter(
        result["action"]
        for result in valid_results
    )


    risk_counter = Counter(
        result["risk_level"]
        for result in valid_results
    )


    evidence_counter = Counter(
        result["evidence_level"]
        for result in valid_results
    )


    ham_allowed = [
        result
        for result in ham_results
        if result["action"] == "ALLOW"
    ]

    ham_escalated = [
        result
        for result in ham_results
        if result["action"] != "ALLOW"
    ]

    ham_escalation_rate = (
        len(ham_escalated)
        / len(ham_results)
        * 100
        if ham_results
        else 0
    )


    spam_escalated = [
        result
        for result in spam_results
        if result["action"] != "ALLOW"
    ]

    spam_allowed = [
        result
        for result in spam_results
        if result["action"] == "ALLOW"
    ]

    spam_escalation_rate = (
        len(spam_escalated)
        / len(spam_results)
        * 100
        if spam_results
        else 0
    )


    disagreements = [
        result
        for result in valid_results
        if (
            (
                result["ml_prediction"] == "ham"
                and result["final_classification"]
                != "HAM"
            )
            or
            (
                result["ml_prediction"] == "spam"
                and result["final_classification"]
                == "HAM"
            )
        )
    ]


    ham_escalation_records = [
        result
        for result in ham_results
        if result["action"] != "ALLOW"
    ]


    spam_allowed_records = [
        result
        for result in spam_results
        if result["action"] == "ALLOW"
    ]


    print_section(
        "EMAIL SECURITY AI — DECISION ENGINE V2 EVALUATION"
    )

    print(
        "\nDATASET RESULTS"
    )

    print("-" * 78)

    print(
        f"Total emails evaluated : "
        f"{total:,}"
    )

    print(
        f"Evaluation errors      : "
        f"{len(errors):,}"
    )

    print(
        f"HAM evaluated          : "
        f"{len(ham_results):,}"
    )

    print(
        f"SPAM evaluated         : "
        f"{len(spam_results):,}"
    )


    print_section(
        "MACHINE LEARNING CLASSIFICATION"
    )

    print(
        f"ML HAM predictions     : "
        f"{ml_counter.get('ham', 0):,}"
    )

    print(
        f"ML SPAM predictions    : "
        f"{ml_counter.get('spam', 0):,}"
    )


    print_section(
        "V2 FINAL DECISION DISTRIBUTION"
    )

    print(
        f"Final HAM              : "
        f"{final_counter.get('HAM', 0):,}"
    )

    print(
        f"Final SPAM             : "
        f"{final_counter.get('SPAM', 0):,}"
    )

    print(
        f"Final SUSPICIOUS       : "
        f"{final_counter.get('SUSPICIOUS', 0):,}"
    )


    print_section(
        "HAM SAFETY"
    )

    print(
        f"HAM correctly allowed  : "
        f"{len(ham_allowed):,}"
    )

    print(
        f"HAM escalated          : "
        f"{len(ham_escalated):,}"
    )

    print(
        f"HAM escalation rate    : "
        f"{ham_escalation_rate:.2f}%"
    )


    print_section(
        "SPAM SECURITY"
    )

    print(
        f"SPAM escalated         : "
        f"{len(spam_escalated):,}"
    )

    print(
        f"SPAM allowed           : "
        f"{len(spam_allowed):,}"
    )

    print(
        f"SPAM escalation rate   : "
        f"{spam_escalation_rate:.2f}%"
    )


    print_section(
        "FINAL ACTION DISTRIBUTION"
    )

    print_counter(
        action_counter,
        total,
    )


    print_section(
        "RISK LEVEL DISTRIBUTION"
    )

    print_counter(
        risk_counter,
        total,
    )


    print_section(
        "SECURITY EVIDENCE DISTRIBUTION"
    )

    print_counter(
        evidence_counter,
        total,
    )


    print_section(
        "ML vs V2 FINAL DECISION DISAGREEMENTS"
    )

    print(
        f"Total disagreements : "
        f"{len(disagreements):,}"
    )

    print(
        f"\nFirst {MAX_DISPLAY_ITEMS} disagreements:"
    )

    print("-" * 78)

    for result in disagreements[
        :MAX_DISPLAY_ITEMS
    ]:

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
            f"{result['final_classification']}"
        )

        print(
            f"Action         : "
            f"{result['action']}"
        )

        print(
            f"Confidence     : "
            f"{result['confidence']}"
        )

        print(
            f"Risk score     : "
            f"{result['risk_score']}/100"
        )

        print(
            f"Risk level     : "
            f"{result['risk_level']}"
        )

        print(
            f"Evidence       : "
            f"{result['evidence_level']}"
        )


    print_section(
        "HAM ESCALATIONS"
    )

    if not ham_escalation_records:

        print(
            "No HAM emails were escalated."
        )

    else:

        for result in ham_escalation_records[
            :MAX_DISPLAY_ITEMS
        ]:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Classification : "
                f"{result['final_classification']}"
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
                f"Evidence       : "
                f"{result['evidence_level']}"
            )


    print_section(
        "SPAM ALLOWED"
    )

    if not spam_allowed_records:

        print(
            "No SPAM emails were allowed."
        )

    else:

        for result in spam_allowed_records[
            :MAX_DISPLAY_ITEMS
        ]:

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Classification : "
                f"{result['final_classification']}"
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
                f"Evidence       : "
                f"{result['evidence_level']}"
            )


    if errors:

        print_section(
            "EVALUATION ERRORS"
        )

        for result in errors[
            :MAX_DISPLAY_ITEMS
        ]:

            print(
                f"\nFile  : {result['file']}"
            )

            print(
                f"Error : {result['error']}"
            )


    print_section(
        "EVALUATION STATUS"
    )

    print(
        "Decision Engine V2 evaluation completed."
    )

    print(
        "\nNo model artifacts were modified."
    )

    print(
        "No threshold was changed."
    )

    print(
        "No retraining was performed."
    )


def main():
    """
    Run the complete Decision Engine V2 evaluation.
    """

    print("=" * 78)

    print(
        "EMAIL SECURITY AI — DECISION ENGINE V2 EVALUATION"
    )

    print("=" * 78)

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    ham_files = discover_emails(
        HAM_DIR
    )

    spam_files = discover_emails(
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

    ham_results = evaluate_dataset(
        predictor=predictor,
        email_files=ham_files,
        expected_label="ham",
        label="HAM",
    )

    spam_results = evaluate_dataset(
        predictor=predictor,
        email_files=spam_files,
        expected_label="spam",
        label="SPAM",
    )

    all_results = (
        ham_results
        + spam_results
    )

    print_evaluation_report(
        all_results
    )


if __name__ == "__main__":
    main()
