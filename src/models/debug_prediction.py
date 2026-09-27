from pathlib import Path

import joblib

from src.features.text_features import transform_text


MODEL_PATH = Path(
    "models/email_security_model.joblib"
)

PIPELINE_PATH = Path(
    "models/text_pipeline.joblib"
)


def main() -> None:
    model = joblib.load(
        MODEL_PATH
    )

    pipeline = joblib.load(
        PIPELINE_PATH
    )

    test_email = """
    hi see you soon.
    regards
    pritish
    """

    features = transform_text(
        pipeline,
        [test_email],
    )

    score = float(
        model.decision_function(
            features
        )[0]
    )

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — MODEL DIAGNOSTIC")
    print("=" * 70)

    print(
        f"\nDecision score: {score:.4f}"
    )

    print(
        f"Threshold     : -0.0975"
    )

    print(
        f"Prediction    : "
        f"{'SPAM' if score >= -0.0975 else 'HAM'}"
    )

    print(
        f"\nFeature matrix shape: {features.shape}"
    )

    print(
        f"Non-zero features  : {features.nnz}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
