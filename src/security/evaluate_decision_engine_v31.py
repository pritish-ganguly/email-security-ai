"""
Email Security AI
Decision Engine V3.1 — Full Evaluation

Purpose:
    Evaluate the frozen production EmailPredictor together with
    Decision Engine V3.1 across the SpamAssassin dataset.

Important:
    - Does NOT retrain the model.
    - Does NOT modify model artifacts.
    - Does NOT change ML thresholds.
    - Uses the SAME production EmailPredictor used by the application.
"""

from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Any

from src.models.predict import EmailPredictor
from src.data.email_parser import parse_email
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine_v31 import make_decision_v31


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spamassassin"
    / "easy_ham"
)

SPAM_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spamassassin"
    / "spam"
)

PROGRESS_INTERVAL = 250


def get_value(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve values from:
        - dictionaries
        - dataclasses
        - normal Python objects
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]

        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)

    return default


def normalize_text(
    value: Any,
    default: str = "UNKNOWN",
) -> str:

    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text.upper()


def load_production_model():
    """
    Load the EXISTING frozen production model.

    This is intentionally identical to the application:
        EmailPredictor()

    No new model is created.
    No training occurs.
    """

    try:
        predictor = EmailPredictor()

        print(
            "Production model loaded using "
            "src.models.predict.EmailPredictor()"
        )

        return predictor

    except Exception as exc:

        raise RuntimeError(
            "Could not load the existing production model "
            "using EmailPredictor()."
        ) from exc


def read_email_file(path: Path) -> str:
    """
    Read raw SpamAssassin email.

    The dataset contains historical emails with mixed encodings,
    therefore decoding is intentionally tolerant.
    """

    raw = path.read_bytes()

    try:
        return raw.decode("utf-8")

    except UnicodeDecodeError:
        return raw.decode(
            "latin-1",
            errors="replace",
        )


def run_ml_prediction(
    predictor: Any,
    email_text: str,
) -> dict:
    """
    Use the SAME predictor.predict() interface as the application.
    """

    result = predictor.predict(email_text)

    if isinstance(result, dict):

        label = str(
            result.get(
                "label",
                "unknown",
            )
        ).lower()

        is_spam = bool(
            result.get(
                "spam",
                result.get(
                    "is_spam",
                    label == "spam",
                ),
            )
        )

        score = result.get(
            "decision_score",
            result.get(
                "score",
                None,
            ),
        )

        threshold = result.get(
            "threshold",
            -0.0975,
        )

        return {
            "label": label,
            "is_spam": is_spam,
            "score": score,
            "decision_score": score,
            "threshold": threshold,
            "spam": is_spam,
        }


    label = get_value(
        result,
        "label",
        default="unknown",
    )

    is_spam = get_value(
        result,
        "spam",
        "is_spam",
        default=str(label).lower() == "spam",
    )

    score = get_value(
        result,
        "decision_score",
        "score",
        default=None,
    )

    threshold = get_value(
        result,
        "threshold",
        default=-0.0975,
    )

    return {
        "label": str(label).lower(),
        "is_spam": bool(is_spam),
        "score": score,
        "decision_score": score,
        "threshold": threshold,
        "spam": bool(is_spam),
    }


def run_decision_engine(
    email_text: str,
    ml_result: dict,
):
    """
    Run the same security pipeline used by the application.

    Pipeline:

        email
          ↓
        parser
          ↓
        ML prediction
          ↓
        security analysis
          ↓
        risk assessment
          ↓
        V3.1 decision engine
    """


    raise RuntimeError(
        "Internal pipeline error: "
        "run_decision_engine() should be called through "
        "run_email_pipeline()."
    )


def run_email_pipeline(
    path: Path,
    predictor: Any,
):
    """
    Run the complete production inference + security pipeline.
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

    if not email_text.strip():
        raise ValueError(
            "Email contains no usable text."
        )


    ml_result = run_ml_prediction(
        predictor,
        email_text,
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


    decision = make_decision_v31(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    return {
        "email_data": email_data,
        "ml_result": ml_result,
        "security_analysis": security_analysis,
        "risk_assessment": risk_assessment,
        "decision": decision,
    }


def extract_decision_fields(
    decision: Any,
) -> dict:
    """
    Convert DecisionResultV3.1 or dictionary into
    a consistent evaluation structure.
    """

    classification = get_value(
        decision,
        "classification",
        "final_class",
        "final_classification",
        "label",
        default="UNKNOWN",
    )

    action = get_value(
        decision,
        "action",
        "final_action",
        "recommended_action",
        default="UNKNOWN",
    )

    confidence = get_value(
        decision,
        "confidence",
        "decision_confidence",
        default="UNKNOWN",
    )

    reason = get_value(
        decision,
        "reason",
        "explanation",
        "decision_reason",
        default="",
    )

    risk_score = get_value(
        decision,
        "risk_score",
        "score",
        default=None,
    )

    risk_level = get_value(
        decision,
        "risk_level",
        "level",
        default="UNKNOWN",
    )

    evidence = get_value(
        decision,
        "evidence",
        "security_evidence",
        "evidence_strength",
        "security_evidence_strength",
        default="UNKNOWN",
    )


    if (
        evidence is not None
        and not isinstance(
            evidence,
            (str, int, float, bool),
        )
    ):

        nested = get_value(
            evidence,
            "strength",
            "level",
            "evidence",
            "classification",
            "evidence_strength",
            default=None,
        )

        if nested is not None:
            evidence = nested


    if (
        risk_score is not None
        and not isinstance(
            risk_score,
            (int, float),
        )
    ):

        nested_score = get_value(
            risk_score,
            "score",
            "risk_score",
            "value",
            default=None,
        )

        if nested_score is not None:
            risk_score = nested_score


    normalized_risk_level = normalize_text(
        risk_level
    )

    if (
        normalized_risk_level == "UNKNOWN"
        and risk_score is not None
    ):

        try:

            score = float(
                risk_score
            )

            if score >= 75:
                normalized_risk_level = "CRITICAL"

            elif score >= 50:
                normalized_risk_level = "HIGH"

            elif score >= 30:
                normalized_risk_level = "MEDIUM"

            else:
                normalized_risk_level = "LOW"

        except Exception:
            pass

    return {
        "classification": normalize_text(
            classification
        ),
        "action": normalize_text(
            action
        ),
        "confidence": normalize_text(
            confidence
        ),
        "risk_score": risk_score,
        "risk_level": normalized_risk_level,
        "evidence": normalize_text(
            evidence
        ),
        "reason": str(reason),
    }


def discover_emails(
    directory: Path,
) -> list[Path]:

    if not directory.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{directory}"
        )

    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
        ]
    )


def print_progress(
    label: str,
    current: int,
    total: int,
):

    if (
        current % PROGRESS_INTERVAL == 0
        or current == total
    ):

        print(
            f"  {label} processed: "
            f"{current:,}/{total:,}"
        )


def evaluate_dataset():

    print("=" * 78)
    print(
        "EMAIL SECURITY AI — "
        "DECISION ENGINE V3.1 EVALUATION"
    )
    print("=" * 78)
    print()


    print(
        "Loading frozen production model..."
    )
    print()

    predictor = load_production_model()

    print()

    print(
        "Loading Email Security AI model..."
    )
    print()


    ham_files = discover_emails(
        HAM_DIR
    )

    spam_files = discover_emails(
        SPAM_DIR
    )

    print(
        f"HAM emails discovered : "
        f"{len(ham_files):,}"
    )

    print(
        f"SPAM emails discovered: "
        f"{len(spam_files):,}"
    )

    print()


    total = 0
    errors = 0

    ham_evaluated = 0
    spam_evaluated = 0

    ml_counts = Counter()
    final_counts = Counter()
    action_counts = Counter()
    risk_counts = Counter()
    evidence_counts = Counter()

    ham_correctly_allowed = 0
    ham_escalated = 0

    spam_escalated = 0
    spam_allowed = 0

    disagreements = []

    ham_escalations = []
    spam_allowed_records = []

    error_records = []


    def process_file(
        path: Path,
        expected: str,
        index: int,
        dataset_total: int,
        dataset_name: str,
    ):

        nonlocal total
        nonlocal errors

        nonlocal ham_evaluated
        nonlocal spam_evaluated

        nonlocal ham_correctly_allowed
        nonlocal ham_escalated

        nonlocal spam_escalated
        nonlocal spam_allowed

        total += 1

        try:

            result = run_email_pipeline(
                path,
                predictor,
            )

            ml_result = result[
                "ml_result"
            ]

            decision = result[
                "decision"
            ]

            fields = extract_decision_fields(
                decision
            )


            ml_label = str(
                ml_result.get(
                    "label",
                    "unknown",
                )
            ).lower()

            ml_score = ml_result.get(
                "decision_score",
                ml_result.get(
                    "score",
                    None,
                ),
            )


            classification = fields[
                "classification"
            ]

            action = fields[
                "action"
            ]

            risk_score = fields[
                "risk_score"
            ]

            risk_level = fields[
                "risk_level"
            ]

            evidence = fields[
                "evidence"
            ]

            confidence = fields[
                "confidence"
            ]


            ml_counts[
                ml_label
            ] += 1

            final_counts[
                classification
            ] += 1

            action_counts[
                action
            ] += 1

            risk_counts[
                risk_level
            ] += 1

            evidence_counts[
                evidence
            ] += 1


            if expected == "ham":

                ham_evaluated += 1

                if classification == "HAM":

                    ham_correctly_allowed += 1

                else:

                    ham_escalated += 1

                    ham_escalations.append(
                        {
                            "file": str(path),
                            "ml_score": ml_score,
                            "classification": classification,
                            "action": action,
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "evidence": evidence,
                        }
                    )


            elif expected == "spam":

                spam_evaluated += 1

                if classification == "SPAM":

                    spam_escalated += 1

                else:

                    spam_allowed += 1

                    spam_allowed_records.append(
                        {
                            "file": str(path),
                            "ml_score": ml_score,
                            "classification": classification,
                            "action": action,
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "evidence": evidence,
                        }
                    )


            expected_final_ml = (
                "SPAM"
                if ml_label == "spam"
                else "HAM"
            )

            if classification != expected_final_ml:

                disagreements.append(
                    {
                        "file": str(path),
                        "expected": expected,
                        "ml_prediction": ml_label,
                        "ml_score": ml_score,
                        "final_class": classification,
                        "action": action,
                        "confidence": confidence,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "evidence": evidence,
                    }
                )

            print_progress(
                dataset_name,
                index,
                dataset_total,
            )

        except Exception as exc:

            errors += 1

            error_records.append(
                {
                    "file": str(path),
                    "expected": expected,
                    "error": str(exc),
                }
            )

            print_progress(
                dataset_name,
                index,
                dataset_total,
            )


    print(
        "Evaluating HAM emails..."
    )

    for index, path in enumerate(
        ham_files,
        start=1,
    ):

        process_file(
            path=path,
            expected="ham",
            index=index,
            dataset_total=len(ham_files),
            dataset_name="HAM",
        )

    print()


    print(
        "Evaluating SPAM emails..."
    )

    for index, path in enumerate(
        spam_files,
        start=1,
    ):

        process_file(
            path=path,
            expected="spam",
            index=index,
            dataset_total=len(spam_files),
            dataset_name="SPAM",
        )


    print()
    print("=" * 78)
    print(
        "EMAIL SECURITY AI — "
        "DECISION ENGINE V3.1 EVALUATION"
    )
    print("=" * 78)

    print()
    print(
        "DATASET RESULTS"
    )

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
        f"{ham_evaluated:,}"
    )

    print(
        f"SPAM evaluated         : "
        f"{spam_evaluated:,}"
    )


    print()
    print("=" * 78)
    print(
        "MACHINE LEARNING CLASSIFICATION"
    )
    print("=" * 78)

    print(
        f"ML HAM predictions     : "
        f"{ml_counts.get('ham', 0):,}"
    )

    print(
        f"ML SPAM predictions    : "
        f"{ml_counts.get('spam', 0):,}"
    )


    print()
    print("=" * 78)
    print(
        "V3.1 FINAL DECISION DISTRIBUTION"
    )
    print("=" * 78)

    final_ham = final_counts.get(
        "HAM",
        0,
    )

    final_spam = final_counts.get(
        "SPAM",
        0,
    )

    final_suspicious = final_counts.get(
        "SUSPICIOUS",
        0,
    )

    def percentage(value):
        if total == 0:
            return 0.0

        return (
            value / total
        ) * 100

    print(
        f"Final HAM              : "
        f"{final_ham:,} "
        f"{percentage(final_ham):7.2f}%"
    )

    print(
        f"Final SPAM             : "
        f"{final_spam:,} "
        f"{percentage(final_spam):7.2f}%"
    )

    print(
        f"Final SUSPICIOUS       : "
        f"{final_suspicious:,} "
        f"{percentage(final_suspicious):7.2f}%"
    )


    print()
    print("=" * 78)
    print(
        "HAM SAFETY"
    )
    print("=" * 78)

    ham_rate = (
        ham_escalated / ham_evaluated * 100
        if ham_evaluated
        else 0
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
        f"{ham_rate:.2f}%"
    )


    print()
    print("=" * 78)
    print(
        "SPAM SECURITY"
    )
    print("=" * 78)

    spam_rate = (
        spam_escalated / spam_evaluated * 100
        if spam_evaluated
        else 0
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
        f"{spam_rate:.2f}%"
    )


    print()
    print("=" * 78)
    print(
        "FINAL ACTION DISTRIBUTION"
    )
    print("=" * 78)

    for action in (
        "ALLOW",
        "BLOCK / REVIEW",
        "QUARANTINE / REVIEW",
        "REVIEW",
    ):

        count = action_counts.get(
            action,
            0,
        )

        print(
            f"{action:<30}"
            f"{count:>6,}"
            f"{percentage(count):>8.2f}%"
        )


    print()
    print("=" * 78)
    print(
        "RISK LEVEL DISTRIBUTION"
    )
    print("=" * 78)

    for level in (
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ):

        count = risk_counts.get(
            level,
            0,
        )

        print(
            f"{level:<24}"
            f"{count:>8,}"
            f"{percentage(count):>9.2f}%"
        )


    print()
    print("=" * 78)
    print(
        "SECURITY EVIDENCE DISTRIBUTION"
    )
    print("=" * 78)

    for evidence in (
        "STRONG",
        "MODERATE",
        "WEAK",
        "UNKNOWN",
    ):

        count = evidence_counts.get(
            evidence,
            0,
        )

        print(
            f"{evidence:<24}"
            f"{count:>8,}"
            f"{percentage(count):>9.2f}%"
        )


    print()
    print("=" * 78)
    print(
        "ML vs V3.1 FINAL DECISION DISAGREEMENTS"
    )
    print("=" * 78)

    print(
        f"Total disagreements : "
        f"{len(disagreements):,}"
    )

    if disagreements:

        print()
        print(
            "First 20 disagreements:"
        )

        print("-" * 78)

        for item in disagreements[:20]:

            print()
            print(
                f"File           : "
                f"{item['file']}"
            )

            print(
                f"Expected       : "
                f"{item['expected']}"
            )

            print(
                f"ML prediction  : "
                f"{item['ml_prediction']}"
            )

            print(
                f"ML score       : "
                f"{item['ml_score']}"
            )

            print(
                f"Final class    : "
                f"{item['final_class']}"
            )

            print(
                f"Action         : "
                f"{item['action']}"
            )

            print(
                f"Confidence     : "
                f"{item['confidence']}"
            )

            print(
                f"Risk score     : "
                f"{item['risk_score']}/100"
            )

            print(
                f"Risk level     : "
                f"{item['risk_level']}"
            )

            print(
                f"Evidence       : "
                f"{item['evidence']}"
            )


    print()
    print("=" * 78)
    print(
        "HAM ESCALATIONS"
    )
    print("=" * 78)

    for item in ham_escalations:

        print()
        print(
            f"File           : "
            f"{item['file']}"
        )

        print(
            f"ML score       : "
            f"{item['ml_score']}"
        )

        print(
            f"Classification : "
            f"{item['classification']}"
        )

        print(
            f"Action         : "
            f"{item['action']}"
        )

        print(
            f"Risk           : "
            f"{item['risk_score']}/100 "
            f"({item['risk_level']})"
        )

        print(
            f"Evidence       : "
            f"{item['evidence']}"
        )


    print()
    print("=" * 78)
    print(
        "SPAM ALLOWED"
    )
    print("=" * 78)

    for item in spam_allowed_records:

        print()
        print(
            f"File           : "
            f"{item['file']}"
        )

        print(
            f"ML score       : "
            f"{item['ml_score']}"
        )

        print(
            f"Classification : "
            f"{item['classification']}"
        )

        print(
            f"Action         : "
            f"{item['action']}"
        )

        print(
            f"Risk           : "
            f"{item['risk_score']}/100 "
            f"({item['risk_level']})"
        )

        print(
            f"Evidence       : "
            f"{item['evidence']}"
        )


    print()
    print("=" * 78)
    print(
        "EVALUATION STATUS"
    )
    print("=" * 78)

    if errors == 0:

        print()
        print(
            "Evaluation completed successfully."
        )

    else:

        print()
        print(
            f"Evaluation completed with "
            f"{errors:,} error(s)."
        )

        print()
        print(
            "First 20 errors:"
        )

        for item in error_records[:20]:

            print()

            print(
                f"File     : "
                f"{item['file']}"
            )

            print(
                f"Expected : "
                f"{item['expected']}"
            )

            print(
                f"Error    : "
                f"{item['error']}"
            )

    print()
    print(
        "No model artifacts were modified."
    )

    print(
        "No threshold was changed."
    )

    print(
        "No retraining was performed."
    )

    print("=" * 78)


if __name__ == "__main__":
    evaluate_dataset()
