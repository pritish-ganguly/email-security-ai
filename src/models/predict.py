"""
Email classification inference engine.

Loads the frozen NLP pipeline and Linear SVM model and
classifies new email text without retraining.
"""

from pathlib import Path

import joblib

from src.features.text_features import transform_text


MODEL_PATH = Path("models/email_security_model.joblib")
PIPELINE_PATH = Path("models/text_pipeline.joblib")

DECISION_THRESHOLD = -0.0975


class EmailPredictor:
    """Classify email text using the frozen security model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        pipeline_path: Path = PIPELINE_PATH,
        threshold: float = DECISION_THRESHOLD,
    ) -> None:

        print("\nLoading Email Security AI model...")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}"
            )

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Text pipeline artifact not found: {pipeline_path}"
            )

        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)
        self.threshold = threshold

    def predict(
        self,
        email_text: str,
    ) -> dict:
        """
        Classify a single email.

        The model uses the frozen Linear SVM decision function.
        The decision threshold was selected using validation data
        and must not be recalculated during inference.

        Returns:
            Dictionary containing:

            label:
                "spam" or "ham"

            spam:
                Boolean spam classification

            decision_score:
                Raw Linear SVM decision score

            threshold:
                Frozen classification threshold
        """

        if not isinstance(email_text, str):
            raise TypeError(
                "email_text must be a string."
            )

        if not email_text.strip():
            raise ValueError(
                "email_text cannot be empty."
            )


        features = transform_text(
            self.pipeline,
            [email_text],
        )


        decision_score = float(
            self.model.decision_function(
                features
            )[0]
        )


        is_spam = (
            decision_score >= self.threshold
        )

        label = (
            "spam"
            if is_spam
            else "ham"
        )

        return {
            "label": label,
            "spam": is_spam,
            "decision_score": decision_score,
            "threshold": self.threshold,
        }


def main() -> None:
    """Run a basic inference test."""

    predictor = EmailPredictor()

    test_email = """
    hi see you soon.
    regards
    pritish
    """

    print("Running ML prediction...")

    result = predictor.predict(
        test_email
    )

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — INFERENCE TEST")
    print("=" * 70)

    print(
        f"\nPrediction : "
        f"{result['label']}"
    )

    print(
        f"Spam       : "
        f"{result['spam']}"
    )

    print(
        f"Score      : "
        f"{result['decision_score']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{result['threshold']:.4f}"
    )


if __name__ == "__main__":
    main()
