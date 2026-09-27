"""
==============================================================================
EMAIL SECURITY AI
FINAL SYSTEM EVALUATION
==============================================================================

Purpose
-------
Perform a complete evaluation of the frozen Email Security AI production
pipeline against the SpamAssassin dataset.

Pipeline
--------
Raw .eml file
      |
      v
Email Parser
      |
      v
Email text construction
      |
      v
Frozen EmailPredictor
      |
      v
Security Analyzer
      |
      v
Risk Engine
      |
      v
Decision Engine V3.1
      |
      v
Final Security Decision

IMPORTANT
---------
This file is evaluation-only.

It does NOT:
    - retrain the model
    - modify the production model
    - modify the NLP pipeline
    - modify the production threshold
    - create a new model
    - overwrite model artifacts
    - tune thresholds
    - modify the production security engines

It uses:
    src.models.predict.EmailPredictor

The evaluation adapters only normalize dataclass/object results into
dictionary-compatible structures where required by the V3.1 evaluator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
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
    Safely retrieve a value from:

        - dictionary
        - dataclass
        - normal Python object
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


def object_to_dict(
    obj: Any,
) -> dict[str, Any]:
    """
    Convert dictionaries, dataclasses and normal objects into a dictionary.

    This is an evaluation compatibility layer.

    Production objects themselves are NOT modified.
    """

    if obj is None:
        return {}


    if isinstance(obj, dict):
        return dict(obj)


    if is_dataclass(obj):
        return asdict(obj)


    result: dict[str, Any] = {}

    for name in dir(obj):

        if name.startswith("_"):
            continue

        try:

            value = getattr(obj, name)

        except Exception:
            continue

        if callable(value):
            continue

        result[name] = value

    return result


def normalize_text(
    value: Any,
    default: str = "UNKNOWN",
) -> str:
    """
    Normalize a result value for reporting.
    """

    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text.upper()


def read_email_file(
    path: Path,
) -> str:
    """
    Read raw SpamAssassin email.

    SpamAssassin contains historical emails with mixed encodings.
    UTF-8 is attempted first and latin-1 is used as a tolerant fallback.
    """

    raw = path.read_bytes()

    try:

        return raw.decode("utf-8")

    except UnicodeDecodeError:

        return raw.decode(
            "latin-1",
            errors="replace",
        )


def build_email_text(
    subject: str,
    body: str,
    html_body: str,
) -> str:
    """
    Build the same general text representation used by the application
    before ML inference.
    """

    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    )


def discover_dataset() -> tuple[
    list[Path],
    list[Path],
]:
    """
    Discover HAM and SPAM emails.
    """

    if not HAM_DIR.exists():

        raise FileNotFoundError(
            f"HAM directory not found:\n{HAM_DIR}"
        )

    if not SPAM_DIR.exists():

        raise FileNotFoundError(
            f"SPAM directory not found:\n{SPAM_DIR}"
        )

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

    return ham_files, spam_files


def extract_email_fields(
    parsed_email: Any,
) -> tuple[
    str,
    str,
    str,
    Any,
]:
    """
    Extract the fields required by the security pipeline.
    """

    subject = str(
        get_value(
            parsed_email,
            "subject",
            default="",
        )
        or ""
    )

    body = str(
        get_value(
            parsed_email,
            "body",
            default="",
        )
        or ""
    )

    html_body = str(
        get_value(
            parsed_email,
            "html_body",
            default="",
        )
        or ""
    )

    attachments = get_value(
        parsed_email,
        "attachments",
        default=[],
    )

    if attachments is None:
        attachments = []

    return (
        subject,
        body,
        html_body,
        attachments,
    )


def normalize_ml_result(
    result: Any,
) -> dict[str, Any]:
    """
    Normalize the frozen EmailPredictor output.

    EmailPredictor currently returns:

        label
        spam
        decision_score
        threshold
    """

    label = normalize_text(
        get_value(
            result,
            "label",
            default="UNKNOWN",
        ),
        default="UNKNOWN",
    )

    spam_value = get_value(
        result,
        "spam",
        "is_spam",
        default=None,
    )

    if spam_value is None:

        spam_value = (
            label == "SPAM"
        )

    spam = bool(spam_value)

    score = get_value(
        result,
        "decision_score",
        "score",
        default=None,
    )

    threshold = get_value(
        result,
        "threshold",
        default=None,
    )

    return {
        "label": label.lower(),
        "spam": spam,
        "is_spam": spam,
        "decision_score": score,
        "score": score,
        "threshold": threshold,
    }


def normalize_security_result(
    security_result: Any,
) -> dict[str, Any]:
    """
    Convert SecurityAnalysis object/dataclass into a dictionary.

    V3.1 receives this compatibility representation.

    The original security_result remains untouched.
    """

    data = object_to_dict(
        security_result
    )


    if "ip_url_count" not in data:

        data["ip_url_count"] = data.get(
            "ip_based_url_count",
            0,
        )

    if "ip_based_url_count" not in data:

        data["ip_based_url_count"] = data.get(
            "ip_url_count",
            0,
        )

    if "suspicious_keyword_count" not in data:

        data["suspicious_keyword_count"] = data.get(
            "suspicious_keywords",
            0,
        )

    if "suspicious_keywords" not in data:

        data["suspicious_keywords"] = data.get(
            "suspicious_keyword_count",
            0,
        )

    if "attachment_count" not in data:

        data["attachment_count"] = data.get(
            "attachments",
            0,
        )

    if "attachments" not in data:

        data["attachments"] = data.get(
            "attachment_count",
            0,
        )

    return data


def normalize_risk_result(
    risk_result: Any,
) -> dict[str, Any]:
    """
    Convert RiskAssessment dataclass/object into a dictionary.

    This specifically prevents:

        'RiskAssessment' object has no attribute 'get'
    """

    data = object_to_dict(
        risk_result
    )

    return {
        "risk_score": data.get(
            "risk_score",
            0,
        ),
        "risk_level": data.get(
            "risk_level",
            "LOW",
        ),
        "reasons": data.get(
            "reasons",
            [],
        ),
        "recommended_action": data.get(
            "recommended_action",
            "",
        ),
    }


def normalize_decision_result(
    decision_result: Any,
) -> dict[str, Any]:
    """
    Normalize DecisionResultV3.1 or dictionary.
    """

    classification = normalize_text(
        get_value(
            decision_result,
            "classification",
            "final_classification",
            "final_class",
            "label",
            default="UNKNOWN",
        )
    )

    action = normalize_text(
        get_value(
            decision_result,
            "action",
            "final_action",
            "recommended_action",
            default="UNKNOWN",
        )
    )

    confidence = normalize_text(
        get_value(
            decision_result,
            "confidence",
            "decision_confidence",
            default="UNKNOWN",
        )
    )

    evidence = normalize_text(
        get_value(
            decision_result,
            "evidence",
            "evidence_level",
            "security_evidence",
            "security_evidence_strength",
            "evidence_strength",
            default="UNKNOWN",
        )
    )

    reason = get_value(
        decision_result,
        "reason",
        "explanation",
        "decision_reason",
        default="",
    )

    risk_score = get_value(
        decision_result,
        "risk_score",
        "score",
        default=None,
    )

    risk_level = normalize_text(
        get_value(
            decision_result,
            "risk_level",
            "level",
            default="UNKNOWN",
        )
    )

    return {
        "classification": classification,
        "action": action,
        "confidence": confidence,
        "evidence": evidence,
        "reason": str(reason),
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


def evaluate_email(
    predictor: EmailPredictor,
    path: Path,
    expected_label: str,
) -> dict[str, Any]:
    """
    Run one email through the complete production security pipeline.

    Correct pipeline:

        file
          ↓
        parser
          ↓
        email text
          ↓
        frozen ML predictor
          ↓
        security analyzer
          ↓
        risk engine
          ↓
        V3.1 decision engine
    """


    email_text = read_email_file(
        path
    )

    if not email_text.strip():

        raise ValueError(
            "Email file contains no usable text."
        )


    parsed_email = parse_email(
        path
    )


    (
        subject,
        body,
        html_body,
        attachments,
    ) = extract_email_fields(
        parsed_email
    )


    model_text = build_email_text(
        subject,
        body,
        html_body,
    )

    if not model_text.strip():

        raise ValueError(
            "Parsed email contains no usable text."
        )


    raw_ml_result = predictor.predict(
        model_text
    )

    ml_result = normalize_ml_result(
        raw_ml_result
    )


    raw_security_result = analyze_email(
        subject=subject,
        body=body,
        html_body=html_body,
        attachments=attachments,
    )

    security_result = normalize_security_result(
        raw_security_result
    )


    raw_risk_result = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_result,
    )

    risk_result = normalize_risk_result(
        raw_risk_result
    )


    raw_decision_result = make_decision_v31(
        ml_result=ml_result,
        security_analysis=security_result,
        risk_assessment=risk_result,
    )

    decision_result = normalize_decision_result(
        raw_decision_result
    )


    return {
        "path": str(path),

        "expected": expected_label.upper(),

        "ml_label": normalize_text(
            ml_result["label"]
        ),

        "ml_score": ml_result[
            "decision_score"
        ],

        "threshold": ml_result[
            "threshold"
        ],

        "final_classification":
            decision_result[
                "classification"
            ],

        "action":
            decision_result[
                "action"
            ],

        "confidence":
            decision_result[
                "confidence"
            ],

        "risk_score":
            risk_result[
                "risk_score"
            ],

        "risk_level":
            normalize_text(
                risk_result[
                    "risk_level"
                ]
            ),

        "evidence":
            decision_result[
                "evidence"
            ],

        "reason":
            decision_result[
                "reason"
            ],
    }


def process_dataset(
    predictor: EmailPredictor,
    files: list[Path],
    expected_label: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Evaluate every file in one dataset partition.
    """

    results: list[
        dict[str, Any]
    ] = []

    errors: list[
        dict[str, Any]
    ] = []

    total = len(files)

    for index, path in enumerate(
        files,
        start=1,
    ):

        try:

            result = evaluate_email(
                predictor,
                path,
                expected_label,
            )

            results.append(
                result
            )

        except Exception as exc:

            errors.append(
                {
                    "path": str(path),
                    "expected": expected_label,
                    "error": str(exc),
                }
            )

        if (
            index % PROGRESS_INTERVAL == 0
            or index == total
        ):

            print(
                f"  {expected_label.upper()} "
                f"processed: "
                f"{index:,}/{total:,}"
            )

    return (
        results,
        errors,
    )


def percentage(
    value: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return (
        value
        / total
        * 100.0
    )


def print_distribution(
    title: str,
    counter: Counter,
    labels: list[str] | None = None,
) -> None:

    print("\n")
    print("=" * 78)
    print(title)
    print("=" * 78)

    total = sum(
        counter.values()
    )

    if labels is None:

        labels = sorted(
            counter.keys()
        )

    for label in labels:

        count = counter.get(
            label,
            0,
        )

        pct = percentage(
            count,
            total,
        )

        print(
            f"{label:<35}"
            f"{count:>8,}"
            f"{pct:>10.2f}%"
        )


def print_item(
    item: dict[str, Any],
) -> None:

    print(
        f"\nFile           : "
        f"{item['path']}"
    )

    print(
        f"ML label       : "
        f"{item['ml_label']}"
    )

    print(
        f"ML score       : "
        f"{item['ml_score']}"
    )

    print(
        f"Classification : "
        f"{item['final_classification']}"
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
        f"Risk           : "
        f"{item['risk_score']}/100 "
        f"({item['risk_level']})"
    )

    print(
        f"Evidence       : "
        f"{item['evidence']}"
    )


def evaluate_system() -> None:


    print("=" * 78)
    print(
        "EMAIL SECURITY AI — FINAL SYSTEM EVALUATION"
    )
    print("=" * 78)

    print(
        "\nThis is an evaluation-only run."
    )

    print(
        "Production artifacts will not be modified."
    )


    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    print(
        "Production EmailPredictor loaded successfully."
    )


    print(
        "\nDiscovering SpamAssassin dataset..."
    )

    ham_files, spam_files = (
        discover_dataset()
    )

    print(
        f"\nHAM emails discovered : "
        f"{len(ham_files):,}"
    )

    print(
        f"SPAM emails discovered: "
        f"{len(spam_files):,}"
    )

    total_discovered = (
        len(ham_files)
        + len(spam_files)
    )


    print(
        "\nEvaluating HAM emails..."
    )

    ham_results, ham_errors = (
        process_dataset(
            predictor,
            ham_files,
            "ham",
        )
    )


    print(
        "\nEvaluating SPAM emails..."
    )

    spam_results, spam_errors = (
        process_dataset(
            predictor,
            spam_files,
            "spam",
        )
    )


    results = (
        ham_results
        + spam_results
    )

    errors = (
        ham_errors
        + spam_errors
    )


    ml_counter = Counter(
        item["ml_label"]
        for item in results
    )

    final_counter = Counter(
        item["final_classification"]
        for item in results
    )

    action_counter = Counter(
        item["action"]
        for item in results
    )

    risk_counter = Counter(
        item["risk_level"]
        for item in results
    )

    evidence_counter = Counter(
        item["evidence"]
        for item in results
    )


    evaluated_ham = [
        item
        for item in results
        if item["expected"] == "HAM"
    ]

    ham_allowed = [
        item
        for item in evaluated_ham
        if item[
            "final_classification"
        ] == "HAM"
    ]

    ham_escalated = [
        item
        for item in evaluated_ham
        if item[
            "final_classification"
        ] != "HAM"
    ]


    evaluated_spam = [
        item
        for item in results
        if item["expected"] == "SPAM"
    ]

    spam_escalated = [
        item
        for item in evaluated_spam
        if item[
            "final_classification"
        ] != "HAM"
    ]

    spam_allowed = [
        item
        for item in evaluated_spam
        if item[
            "final_classification"
        ] == "HAM"
    ]


    differences = [
        item
        for item in results
        if normalize_text(
            item["ml_label"]
        )
        != normalize_text(
            item["final_classification"]
        )
    ]


    print("\n")
    print("=" * 78)
    print("DATASET RESULTS")
    print("=" * 78)

    print(
        f"Total emails discovered : "
        f"{total_discovered:,}"
    )

    print(
        f"Total emails evaluated  : "
        f"{len(results):,}"
    )

    print(
        f"Evaluation errors       : "
        f"{len(errors):,}"
    )

    print(
        f"HAM evaluated           : "
        f"{len(evaluated_ham):,}"
    )

    print(
        f"SPAM evaluated          : "
        f"{len(evaluated_spam):,}"
    )


    print_distribution(
        "MACHINE LEARNING CLASSIFICATION",
        ml_counter,
        [
            "HAM",
            "SPAM",
        ],
    )


    print_distribution(
        "V3.1 FINAL DECISION DISTRIBUTION",
        final_counter,
        [
            "HAM",
            "SPAM",
            "SUSPICIOUS",
        ],
    )


    print("\n")
    print("=" * 78)
    print("HAM SAFETY")
    print("=" * 78)

    ham_rate = percentage(
        len(ham_escalated),
        len(evaluated_ham),
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
        f"{ham_rate:.2f}%"
    )


    print("\n")
    print("=" * 78)
    print("SPAM SECURITY")
    print("=" * 78)

    spam_rate = percentage(
        len(spam_escalated),
        len(evaluated_spam),
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
        f"{spam_rate:.2f}%"
    )


    print_distribution(
        "FINAL ACTION DISTRIBUTION",
        action_counter,
    )


    print_distribution(
        "RISK LEVEL DISTRIBUTION",
        risk_counter,
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
    )


    print_distribution(
        "SECURITY EVIDENCE DISTRIBUTION",
        evidence_counter,
        [
            "STRONG",
            "MODERATE",
            "WEAK",
        ],
    )


    print("\n")
    print("=" * 78)
    print(
        "ML vs V3.1 DECISION DIFFERENCES"
    )
    print("=" * 78)

    print(
        f"Total differences : "
        f"{len(differences):,}"
    )


    print("\n")
    print("=" * 78)
    print("HAM ESCALATIONS")
    print("=" * 78)

    if not ham_escalated:

        print(
            "None."
        )

    else:

        for item in ham_escalated:

            print_item(
                item
            )


    print("\n")
    print("=" * 78)
    print("SPAM ALLOWED")
    print("=" * 78)

    if not spam_allowed:

        print(
            "None."
        )

    else:

        for item in spam_allowed:

            print_item(
                item
            )


    print("\n")
    print("=" * 78)
    print("EVALUATION STATUS")
    print("=" * 78)

    if errors:

        print(
            f"\nEvaluation completed with "
            f"{len(errors):,} error(s)."
        )

        print(
            "\nFirst 20 errors:"
        )

        for error in errors[:20]:

            print(
                f"\nFile     : "
                f"{error['path']}"
            )

            print(
                f"Expected : "
                f"{error['expected']}"
            )

            print(
                f"Error    : "
                f"{error['error']}"
            )

    else:

        print(
            "\nFINAL SYSTEM EVALUATION "
            "COMPLETED SUCCESSFULLY."
        )

        print(
            "All discovered emails were "
            "processed without evaluation errors."
        )


    print(
        "\nNo model artifacts were modified."
    )

    print(
        "No production threshold was changed."
    )

    print(
        "No retraining was performed."
    )

    print(
        "The existing production EmailPredictor "
        "was used."
    )

    print("=" * 78)


if __name__ == "__main__":

    evaluate_system()
