"""
Full-dataset evaluation for Email Security AI Decision Engine V3.

Evaluates the frozen production ML model together with the V3
security decision engine against the SpamAssassin dataset.

IMPORTANT:
    - No model retraining
    - No model modification
    - No threshold modification
    - No artifact modification

This script is evaluation-only.
"""

from pathlib import Path
from collections import Counter

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine_v3 import make_decision_v3


HAM_DIR = Path(
    "data/raw/spamassassin/easy_ham"
)

SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


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


def build_email_text(email_data) -> str:
    """Build the text representation used by the ML model."""

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


def evaluate_email(
    predictor: EmailPredictor,
    path: Path,
    expected_label: str,
) -> dict[str, Any]:
    """
    Run one email through the complete production pipeline.

    IMPORTANT:
        This follows the same interfaces used by the working
        Decision Engine V3 evaluator.
    """


    email_data = parse_email(path)


    subject = str(
        get_value(
            email_data,
            "subject",
            default="",
        )
        or ""
    )

    body = str(
        get_value(
            email_data,
            "body",
            default="",
        )
        or ""
    )

    html_body = str(
        get_value(
            email_data,
            "html_body",
            default="",
        )
        or ""
    )

    attachments = get_value(
        email_data,
        "attachments",
        default=[],
    )

    email_text = (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    )


    ml_result = predictor.predict(
        email_text
    )


    security_result = analyze_email(
        subject=subject,
        body=body,
        html_body=html_body,
        attachments=attachments,
    )


    risk_result = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_result,
    )


    decision_result = make_decision_v31(
        ml_result=ml_result,
        security_analysis=security_result,
        risk_assessment=risk_result,
    )


    ml_label = normalize_label(
        get_value(
            ml_result,
            "label",
            default="UNKNOWN",
        )
    )

    ml_score = get_value(
        ml_result,
        "decision_score",
        "score",
        default=None,
    )

    threshold = get_value(
        ml_result,
        "threshold",
        default=None,
    )


    final_classification = normalize_label(
        get_value(
            decision_result,
            "classification",
            "final_classification",
            "final_class",
            "label",
            default="UNKNOWN",
        )
    )

    action = normalize_label(
        get_value(
            decision_result,
            "action",
            "final_action",
            "recommended_action",
            default="UNKNOWN",
        )
    )

    confidence = normalize_label(
        get_value(
            decision_result,
            "confidence",
            "decision_confidence",
            default="UNKNOWN",
        )
    )

    evidence = normalize_label(
        get_value(
            decision_result,
            "evidence",
            "security_evidence",
            "evidence_strength",
            "security_evidence_strength",
            default="UNKNOWN",
        )
    )


    risk_score = get_value(
        risk_result,
        "risk_score",
        "score",
        default=None,
    )

    risk_level = normalize_label(
        get_value(
            risk_result,
            "risk_level",
            "level",
            default="UNKNOWN",
        )
    )

    return {
        "path": str(path),
        "expected": expected_label,

        "ml_label": ml_label,
        "ml_score": ml_score,
        "threshold": threshold,

        "final_classification": final_classification,
        "action": action,
        "confidence": confidence,
        "evidence": evidence,

        "risk_score": risk_score,
        "risk_level": risk_level,
    }


def print_progress(
    label: str,
    processed: int,
    total: int,
) -> None:
    """Print progress every 250 emails and at completion."""

    if (
        processed % 250 == 0
        or processed == total
    ):
        print(
            f"  {label} processed: "
            f"{processed:,}/{total:,}"
        )


def main() -> None:

    print_header(
        "EMAIL SECURITY AI — "
        "DECISION ENGINE V3 EVALUATION"
    )

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

    errors = []

    ml_predictions = Counter()

    final_predictions = Counter()

    actions = Counter()

    risk_levels = Counter()

    evidence_levels = Counter()


    print("\nEvaluating HAM emails...")

    for index, email_path in enumerate(
        ham_files,
        start=1,
    ):

        try:

            result = evaluate_email(
                email_path=email_path,
                expected_label="ham",
                predictor=predictor,
            )

            results.append(result)

            ml_predictions[
                result["ml_label"]
            ] += 1

            final_predictions[
                result["classification"]
            ] += 1

            actions[
                result["action"]
            ] += 1

            risk_levels[
                result["risk_level"]
            ] += 1

            evidence_levels[
                result["evidence_level"]
            ] += 1

        except Exception as exc:

            errors.append(
                {
                    "file": email_path,
                    "expected": "ham",
                    "error": str(exc),
                }
            )

        print_progress(
            "HAM",
            index,
            len(ham_files),
        )


    print("\nEvaluating SPAM emails...")

    for index, email_path in enumerate(
        spam_files,
        start=1,
    ):

        try:

            result = evaluate_email(
                email_path=email_path,
                expected_label="spam",
                predictor=predictor,
            )

            results.append(result)

            ml_predictions[
                result["ml_label"]
            ] += 1

            final_predictions[
                result["classification"]
            ] += 1

            actions[
                result["action"]
            ] += 1

            risk_levels[
                result["risk_level"]
            ] += 1

            evidence_levels[
                result["evidence_level"]
            ] += 1

        except Exception as exc:

            errors.append(
                {
                    "file": email_path,
                    "expected": "spam",
                    "error": str(exc),
                }
            )

        print_progress(
            "SPAM",
            index,
            len(spam_files),
        )


    total_evaluated = len(results)

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


    ham_correctly_allowed = sum(
        1
        for result in ham_results
        if result["action"] == "ALLOW"
    )

    ham_escalated = (
        len(ham_results)
        - ham_correctly_allowed
    )

    ham_escalation_rate = (
        ham_escalated / len(ham_results) * 100
        if ham_results
        else 0.0
    )


    spam_escalated = sum(
        1
        for result in spam_results
        if result["action"] != "ALLOW"
    )

    spam_allowed = (
        len(spam_results)
        - spam_escalated
    )

    spam_escalation_rate = (
        spam_escalated / len(spam_results) * 100
        if spam_results
        else 0.0
    )


    disagreements = []

    for result in results:

        ml_is_spam = (
            result["ml_label"] == "spam"
        )

        final_is_spam = (
            result["classification"] == "SPAM"
        )

        if ml_is_spam != final_is_spam:

            disagreements.append(
                result
            )


    print_header(
        "EMAIL SECURITY AI — "
        "DECISION ENGINE V3 EVALUATION"
    )

    print(
        "\nDATASET RESULTS"
    )

    print("-" * 78)

    print(
        f"Total emails evaluated : "
        f"{total_evaluated:,}"
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


    print_header(
        "MACHINE LEARNING CLASSIFICATION"
    )

    print(
        f"ML HAM predictions     : "
        f"{ml_predictions['ham']:,}"
    )

    print(
        f"ML SPAM predictions    : "
        f"{ml_predictions['spam']:,}"
    )


    print_header(
        "V3 FINAL DECISION DISTRIBUTION"
    )

    print(
        f"Final HAM              : "
        f"{final_predictions['HAM']:,}"
    )

    print(
        f"Final SPAM             : "
        f"{final_predictions['SPAM']:,}"
    )

    print(
        f"Final SUSPICIOUS       : "
        f"{final_predictions['SUSPICIOUS']:,}"
    )


    print_header(
        "HAM SAFETY"
    )

    print(
        f"HAM correctly allowed  : "
        f"{ham_correctly_allowed:,}"
    )

    print(
        f"HAM escalated          : "
        f"{ham_escalated:,}"
    )

    print(
        f"HAM escalation rate    : "
        f"{ham_escalation_rate:.2f}%"
    )


    print_header(
        "SPAM SECURITY"
    )

    print(
        f"SPAM escalated         : "
        f"{spam_escalated:,}"
    )

    print(
        f"SPAM allowed           : "
        f"{spam_allowed:,}"
    )

    print(
        f"SPAM escalation rate   : "
        f"{spam_escalation_rate:.2f}%"
    )


    print_header(
        "FINAL ACTION DISTRIBUTION"
    )

    for action in [
        "ALLOW",
        "BLOCK / REVIEW",
        "QUARANTINE / REVIEW",
        "REVIEW",
    ]:

        count = actions[action]

        percentage = (
            count / total_evaluated * 100
            if total_evaluated
            else 0.0
        )

        print(
            f"{action:<30}"
            f"{count:>7,}"
            f"   {percentage:>6.2f}%"
        )


    print_header(
        "RISK LEVEL DISTRIBUTION"
    )

    for level in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:

        count = risk_levels[level]

        percentage = (
            count / total_evaluated * 100
            if total_evaluated
            else 0.0
        )

        print(
            f"{level:<20}"
            f"{count:>7,}"
            f"   {percentage:>6.2f}%"
        )


    print_header(
        "SECURITY EVIDENCE DISTRIBUTION"
    )

    for level in [
        "STRONG",
        "MODERATE",
        "WEAK",
    ]:

        count = evidence_levels[level]

        percentage = (
            count / total_evaluated * 100
            if total_evaluated
            else 0.0
        )

        print(
            f"{level:<20}"
            f"{count:>7,}"
            f"   {percentage:>6.2f}%"
        )


    print_header(
        "ML vs V3 FINAL DECISION DISAGREEMENTS"
    )

    print(
        f"Total disagreements : "
        f"{len(disagreements):,}"
    )

    if disagreements:

        print(
            "\nFirst 20 disagreements:"
        )

        print("-" * 78)

        for result in disagreements[:20]:

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
                f"{result['ml_label']}"
            )

            print(
                f"ML score       : "
                f"{result['ml_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{result['classification']}"
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


    ham_escalations = [
        result
        for result in ham_results
        if result["action"] != "ALLOW"
    ]

    print_header(
        "HAM ESCALATIONS"
    )

    if not ham_escalations:

        print(
            "\nNo HAM emails were escalated."
        )

    else:

        for result in ham_escalations[:20]:

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
                f"{result['classification']}"
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


    spam_allowed_results = [
        result
        for result in spam_results
        if result["action"] == "ALLOW"
    ]

    print_header(
        "SPAM ALLOWED"
    )

    if not spam_allowed_results:

        print(
            "\nNo SPAM emails were allowed."
        )

    else:

        for result in spam_allowed_results[:20]:

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
                f"{result['classification']}"
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

        print_header(
            "EVALUATION ERRORS"
        )

        for error in errors[:20]:

            print(
                f"\nFile     : "
                f"{error['file']}"
            )

            print(
                f"Expected : "
                f"{error['expected']}"
            )

            print(
                f"Error    : "
                f"{error['error']}"
            )


    print_header(
        "EVALUATION STATUS"
    )

    print(
        "Decision Engine V3 evaluation completed."
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


if __name__ == "__main__":
    main()
