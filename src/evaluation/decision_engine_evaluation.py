"""
Email Security AI — Decision Engine Evaluation

Evaluates the production decision engine against the
SpamAssassin HAM and SPAM datasets.

This script is diagnostic only.

It does NOT:
    - retrain the ML model
    - modify the ML model
    - modify the NLP pipeline
    - change the production threshold
    - save new model artifacts

Evaluation flow:

    .eml
      ↓
    Email parser
      ↓
    Frozen NLP pipeline
      ↓
    Frozen ML model
      ↓
    Security analyzer
      ↓
    Risk engine
      ↓
    Decision engine
      ↓
    Evaluation statistics
"""

from pathlib import Path
from collections import Counter


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
    email_data,
    field,
    default="",
):
    """
    Safely retrieve a value from a parsed email.

    Supports both dictionary and object parser results.
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
) -> str:
    """
    Build the same email representation used
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
    directory: Path,
):
    """
    Return all files from a dataset directory.

    SpamAssassin messages may not have a .eml extension,
    therefore we intentionally do NOT filter by suffix here.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}"
        )

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
    )


def evaluate_email(
    email_path: Path,
    expected_label: str,
    predictor: EmailPredictor,
):
    """
    Evaluate one email through the complete
    production security pipeline.
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

    return {
        "file": email_path,
        "expected": expected_label,
        "ml": ml_result,
        "security": security_analysis,
        "risk": risk_assessment,
        "decision": decision,
    }


def main():
    print("\n" + "=" * 78)
    print(
        "EMAIL SECURITY AI — DECISION ENGINE EVALUATION"
    )
    print("=" * 78)

    print("\nLoading frozen production model...")

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


    total = 0
    errors = 0

    ml_predictions = Counter()
    final_predictions = Counter()

    risk_levels = Counter()
    actions = Counter()

    ham_correctly_allowed = 0
    ham_escalated = 0

    spam_escalated = 0
    spam_allowed = 0

    disagreements = []

    ham_escalations = []
    spam_allowed_records = []


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

            total += 1

            ml = result["ml"]
            decision = result["decision"]
            risk = result["risk"]

            ml_label = str(
                ml["label"]
            ).lower()

            final_class = str(
                decision.classification
            ).upper()

            action = str(
                decision.action
            ).upper()

            risk_level = str(
                decision.risk_level
            ).upper()

            ml_predictions[
                ml_label
            ] += 1

            final_predictions[
                final_class
            ] += 1

            actions[
                action
            ] += 1

            risk_levels[
                risk_level
            ] += 1


            if action == "ALLOW":
                ham_correctly_allowed += 1

            else:
                ham_escalated += 1

                ham_escalations.append(
                    result
                )


            if (
                final_class.lower()
                != ml_label
            ):
                disagreements.append(
                    result
                )

        except Exception as exc:

            errors += 1

            print(
                f"\nHAM evaluation error:"
                f"\n  File: {email_path}"
                f"\n  Error: {exc}"
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

            total += 1

            ml = result["ml"]
            decision = result["decision"]
            risk = result["risk"]

            ml_label = str(
                ml["label"]
            ).lower()

            final_class = str(
                decision.classification
            ).upper()

            action = str(
                decision.action
            ).upper()

            risk_level = str(
                decision.risk_level
            ).upper()

            ml_predictions[
                ml_label
            ] += 1

            final_predictions[
                final_class
            ] += 1

            actions[
                action
            ] += 1

            risk_levels[
                risk_level
            ] += 1


            if action == "ALLOW":
                spam_allowed += 1

                spam_allowed_records.append(
                    result
                )

            else:
                spam_escalated += 1


            if (
                final_class.lower()
                != ml_label
            ):
                disagreements.append(
                    result
                )

        except Exception as exc:

            errors += 1

            print(
                f"\nSPAM evaluation error:"
                f"\n  File: {email_path}"
                f"\n  Error: {exc}"
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


    ham_total = len(ham_files)

    spam_total = len(spam_files)

    ham_escalation_rate = (
        ham_escalated / ham_total * 100
        if ham_total
        else 0
    )

    spam_escalation_rate = (
        spam_escalated / spam_total * 100
        if spam_total
        else 0
    )

    def percentage(
        value,
        denominator,
    ):
        if denominator == 0:
            return 0.0

        return (
            value / denominator
        ) * 100


    print("\n" + "=" * 78)
    print(
        "EMAIL SECURITY AI — DECISION ENGINE EVALUATION"
    )
    print("=" * 78)

    print("\nDATASET RESULTS")
    print("-" * 78)

    print(
        f"Total emails evaluated : "
        f"{total:,}"
    )

    print(
        f"Evaluation errors      : "
        f"{errors:,}"
    )

    print(
        f"HAM evaluated          : "
        f"{ham_total:,}"
    )

    print(
        f"SPAM evaluated         : "
        f"{spam_total:,}"
    )


    print("\n" + "=" * 78)
    print(
        "MACHINE LEARNING CLASSIFICATION"
    )
    print("=" * 78)

    print(
        f"ML HAM predictions     : "
        f"{ml_predictions['ham']:,}"
    )

    print(
        f"ML SPAM predictions    : "
        f"{ml_predictions['spam']:,}"
    )


    print("\n" + "=" * 78)
    print(
        "FINAL DECISION DISTRIBUTION"
    )
    print("=" * 78)

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


    print("\n" + "=" * 78)
    print(
        "HAM SAFETY"
    )
    print("=" * 78)

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


    print("\n" + "=" * 78)
    print(
        "SPAM SECURITY"
    )
    print("=" * 78)

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


    print("\n" + "=" * 78)
    print(
        "FINAL ACTION DISTRIBUTION"
    )
    print("=" * 78)

    for action in [
        "ALLOW",
        "BLOCK / REVIEW",
        "QUARANTINE / REVIEW",
    ]:

        count = actions[action]

        pct = percentage(
            count,
            total,
        )

        print(
            f"{action:<32}"
            f"{count:>6,}"
            f"{pct:>8.2f}%"
        )


    print("\n" + "=" * 78)
    print(
        "RISK LEVEL DISTRIBUTION"
    )
    print("=" * 78)

    for level in [
        "CRITICAL",
        "HIGH",
        "LOW",
        "MEDIUM",
    ]:

        count = risk_levels[level]

        pct = percentage(
            count,
            total,
        )

        print(
            f"{level:<20}"
            f"{count:>6,}"
            f"{pct:>8.2f}%"
        )


    print("\n" + "=" * 78)
    print(
        "ML vs FINAL DECISION DISAGREEMENTS"
    )
    print("=" * 78)

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

            ml = result["ml"]
            decision = result["decision"]
            risk = result["risk"]

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
                f"{ml['label']}"
            )

            print(
                f"ML score       : "
                f"{ml['decision_score']:.4f}"
            )

            print(
                f"Final class    : "
                f"{decision.classification}"
            )

            print(
                f"Action         : "
                f"{decision.action}"
            )

            print(
                f"Confidence     : "
                f"{decision.confidence}"
            )

            print(
                f"Risk score     : "
                f"{decision.risk_score}/100"
            )

            print(
                f"Risk level     : "
                f"{decision.risk_level}"
            )


    print("\n" + "=" * 78)
    print(
        "HAM ESCALATIONS"
    )
    print("=" * 78)

    if not ham_escalations:

        print("\nNone.")

    else:

        for result in ham_escalations:

            ml = result["ml"]
            decision = result["decision"]

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"ML score       : "
                f"{ml['decision_score']:.4f}"
            )

            print(
                f"Classification : "
                f"{decision.classification}"
            )

            print(
                f"Action         : "
                f"{decision.action}"
            )

            print(
                f"Risk           : "
                f"{decision.risk_score}/100 "
                f"({decision.risk_level})"
            )


    print("\n" + "=" * 78)
    print(
        "SPAM ALLOWED"
    )
    print("=" * 78)

    if not spam_allowed_records:

        print("\nNone.")

    else:

        for result in spam_allowed_records:

            ml = result["ml"]
            decision = result["decision"]

            print(
                f"\nFile           : "
                f"{result['file']}"
            )

            print(
                f"ML score       : "
                f"{ml['decision_score']:.4f}"
            )

            print(
                f"Classification : "
                f"{decision.classification}"
            )

            print(
                f"Action         : "
                f"{decision.action}"
            )

            print(
                f"Risk           : "
                f"{decision.risk_score}/100 "
                f"({decision.risk_level})"
            )


    print("\n" + "=" * 78)
    print(
        "EVALUATION STATUS"
    )
    print("=" * 78)

    print(
        "\nDecision engine evaluation completed."
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

    print()


if __name__ == "__main__":
    main()
