from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.email_parser import parse_email
from src.features.text_features import transform_text


MODEL_PATH = Path("models/email_security_model.joblib")
PIPELINE_PATH = Path("models/text_pipeline.joblib")

THRESHOLD = -0.0975

EMAILS = [
    ("HAM false positive", "0068.f1c604a78739e4f966253d762c972dde", "ham"),
    ("HAM borderline", "0242.a02f8a0ce9077130c10d33db2a16ec36", "ham"),
    ("SPAM false negative", "0299.9d0b292172cb787eb2ed9e8855222edd", "spam"),
    ("SPAM false negative", "0228.23fc5aadfceb81d121d77dfe37f6929a", "spam"),
    ("SPAM false negative", "0281.7e8c08897b61b9b008238efec9ca8d15", "spam"),
    ("SPAM false negative", "0402.1290489e7e62ac9bb500677606540e5d", "spam"),
    ("SPAM false negative", "0095.e1db2d3556c2863ef7355faf49160219", "spam"),
]

HAM_DIR = Path("data/raw/spamassassin/easy_ham")
SPAM_DIR = Path("data/raw/spamassassin/spam")


def get_value(
    email_data: object,
    field: str,
    default: object = "",
) -> object:
    if isinstance(email_data, dict):
        return email_data.get(field, default)

    return getattr(email_data, field, default)


def build_email_text(email_data: object) -> str:
    subject = str(
        get_value(email_data, "subject", "") or ""
    )

    body = str(
        get_value(email_data, "body", "") or ""
    )

    html_body = str(
        get_value(email_data, "html_body", "") or ""
    )

    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    )


def find_vectorizers(
    obj: Any,
    path: str = "pipeline",
) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []

    if obj is None:
        return found

    if hasattr(obj, "get_feature_names_out"):
        found.append((path, obj))

    if hasattr(obj, "transformer_list"):
        for name, transformer in obj.transformer_list:
            found.extend(
                find_vectorizers(
                    transformer,
                    f"{path}.{name}",
                )
            )

    if hasattr(obj, "transformers_"):
        for name, transformer, columns in obj.transformers_:
            if transformer == "drop":
                continue

            if transformer == "passthrough":
                continue

            found.extend(
                find_vectorizers(
                    transformer,
                    f"{path}.{name}",
                )
            )

    if hasattr(obj, "steps"):
        for name, step in obj.steps:
            found.extend(
                find_vectorizers(
                    step,
                    f"{path}.{name}",
                )
            )

    return found


def extract_feature_names(
    pipeline: Any,
    feature_count: int,
) -> list[str] | None:

    candidates = find_vectorizers(pipeline)

    if not candidates:
        return None

    best_names: list[str] | None = None

    for path, transformer in candidates:
        try:
            names = transformer.get_feature_names_out()

            names = [
                str(name)
                for name in names
            ]

            if len(names) == feature_count:
                return names

            if best_names is None:
                best_names = names

            print(
                f"Found feature names at {path}: "
                f"{len(names)} features"
            )

        except Exception:
            continue

    return best_names


def get_nonzero_contributions(
    model: Any,
    features: Any,
    feature_names: list[str] | None,
    top_n: int = 20,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:

    coefficients = np.asarray(
        model.coef_[0]
    ).ravel()

    if hasattr(features, "toarray"):
        row = features.toarray()[0]
    else:
        row = np.asarray(features)[0]

    contributions = row * coefficients

    nonzero_indices = np.flatnonzero(
        row
    )

    positive: list[tuple[str, float]] = []
    negative: list[tuple[str, float]] = []

    for index in nonzero_indices:

        contribution = float(
            contributions[index]
        )

        if feature_names is not None:
            name = feature_names[index]
        else:
            name = f"feature_{index}"

        if contribution > 0:
            positive.append(
                (
                    name,
                    contribution,
                )
            )

        elif contribution < 0:
            negative.append(
                (
                    name,
                    contribution,
                )
            )

    positive.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    negative.sort(
        key=lambda item: item[1],
    )

    return (
        positive[:top_n],
        negative[:top_n],
    )


def print_features(
    title: str,
    features: list[tuple[str, float]],
) -> None:

    print("\n" + title)
    print("-" * 78)

    if not features:
        print("None")
        return

    for index, (name, value) in enumerate(
        features,
        start=1,
    ):
        print(
            f"{index:2d}. "
            f"{name:<55} "
            f"{value:+.6f}"
        )


def print_email_header(
    category: str,
    filename: str,
) -> None:

    print("\n" + "=" * 78)
    print(
        f"{category}: {filename}"
    )
    print("=" * 78)


def analyze_email(
    model: Any,
    pipeline: Any,
    file_path: Path,
    expected: str,
    feature_names: list[str] | None,
) -> None:

    email_data = parse_email(
        file_path
    )

    text = build_email_text(
        email_data
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

    features = transform_text(
        pipeline,
        [text],
    )

    score = float(
        model.decision_function(
            features
        )[0]
    )

    predicted = (
        "spam"
        if score >= THRESHOLD
        else "ham"
    )

    distance = abs(
        score - THRESHOLD
    )

    if predicted == expected:
        if distance <= 0.10:
            assessment = (
                "CORRECT — BORDERLINE"
            )
        else:
            assessment = "CORRECT"
    elif expected == "ham":
        assessment = (
            "FALSE POSITIVE — HAM -> SPAM"
        )
    else:
        assessment = (
            "FALSE NEGATIVE — SPAM -> HAM"
        )

    print(
        f"\nExpected label : "
        f"{expected.upper()}"
    )

    print(
        f"Predicted      : "
        f"{predicted.upper()}"
    )

    print(
        f"SVM score      : "
        f"{score:.4f}"
    )

    print(
        f"Threshold      : "
        f"{THRESHOLD:.4f}"
    )

    print(
        f"Distance       : "
        f"{distance:.4f}"
    )

    print(
        f"Assessment     : "
        f"{assessment}"
    )

    print("\nEMAIL FEATURES")
    print("-" * 78)

    print(
        f"Subject length : "
        f"{len(subject)}"
    )

    print(
        f"Body length    : "
        f"{len(body)}"
    )

    print(
        f"HTML length    : "
        f"{len(html_body)}"
    )

    print(
        "\nRunning production text transformation..."
    )

    if hasattr(features, "shape"):
        shape = features.shape
    else:
        shape = np.asarray(features).shape

    if hasattr(features, "nnz"):
        nonzero = features.nnz
    else:
        nonzero = int(
            np.count_nonzero(
                np.asarray(features)
            )
        )

    print("\nFEATURE MATRIX")
    print("-" * 78)

    print(
        f"Shape          : {shape}"
    )

    print(
        f"Non-zero       : {nonzero}"
    )

    coefficients = np.asarray(
        model.coef_[0]
    ).ravel()

    if hasattr(features, "toarray"):
        row = features.toarray()[0]
    else:
        row = np.asarray(features)[0]

    contributions = (
        row * coefficients
    )

    intercept = float(
        model.intercept_[0]
    )

    positive_total = float(
        contributions[
            contributions > 0
        ].sum()
    )

    negative_total = float(
        contributions[
            contributions < 0
        ].sum()
    )

    reconstructed = (
        intercept
        + positive_total
        + negative_total
    )

    print("\nMODEL CONTRIBUTION SUMMARY")
    print("-" * 78)

    print(
        f"Intercept              : "
        f"{intercept:+.6f}"
    )

    print(
        f"Positive contributions : "
        f"{positive_total:+.6f}"
    )

    print(
        f"Negative contributions : "
        f"{negative_total:+.6f}"
    )

    print(
        f"Reconstructed score    : "
        f"{reconstructed:+.6f}"
    )

    print(
        f"Actual SVM score       : "
        f"{score:+.6f}"
    )

    print(
        f"Reconstruction error   : "
        f"{abs(reconstructed - score):.10f}"
    )

    positive, negative = (
        get_nonzero_contributions(
            model=model,
            features=features,
            feature_names=feature_names,
        )
    )

    if feature_names is None:
        print(
            "\nWARNING: Feature names could not "
            "be recovered from the saved pipeline."
        )

    print_features(
        "TOP SPAM-PUSHING FEATURES",
        positive,
    )

    print_features(
        "TOP HAM-PUSHING FEATURES",
        negative,
    )


def load_artifacts() -> tuple[Any, Any]:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"Pipeline not found: {PIPELINE_PATH}"
        )

    print(
        "\nLoading frozen production artifacts..."
    )

    model = joblib.load(
        MODEL_PATH
    )

    pipeline = joblib.load(
        PIPELINE_PATH
    )

    print(
        "Artifacts loaded."
    )

    return model, pipeline


def main() -> None:

    print(
        "\n" + "=" * 78
    )

    print(
        "EMAIL SECURITY AI — "
        "FEATURE CONTRIBUTION ANALYSIS"
    )

    print(
        "=" * 78
    )

    model, pipeline = (
        load_artifacts()
    )

    print(
        "\nInspecting saved NLP pipeline..."
    )

    feature_names = extract_feature_names(
        pipeline,
        model.coef_.shape[1],
    )

    if feature_names is None:

        print(
            "Feature names unavailable."
        )

        print(
            "The analysis will still calculate "
            "exact numeric contributions."
        )

    else:

        print(
            f"Recovered "
            f"{len(feature_names):,} feature names."
        )

        print(
            f"Model expects "
            f"{model.coef_.shape[1]:,} features."
        )

    for category, filename, expected in EMAILS:

        if expected == "ham":
            base_dir = HAM_DIR
        else:
            base_dir = SPAM_DIR

        file_path = (
            base_dir / filename
        )

        if not file_path.exists():

            print(
                f"\nWARNING: Email not found:"
                f"\n{file_path}"
            )

            continue

        print_email_header(
            category,
            filename,
        )

        analyze_email(
            model=model,
            pipeline=pipeline,
            file_path=file_path,
            expected=expected,
            feature_names=feature_names,
        )

    print(
        "\n" + "=" * 78
    )

    print(
        "FEATURE CONTRIBUTION ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )

    print(
        "\nProduction model: Linear SVM"
    )

    print(
        f"Production threshold: "
        f"{THRESHOLD}"
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
