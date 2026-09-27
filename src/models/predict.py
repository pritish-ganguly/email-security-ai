from pathlib import Path

import joblib

from src.features.text_features import transform_text


MODEL_PATH = Path("models/email_security_model.joblib")
PIPELINE_PATH = Path("models/text_pipeline.joblib")
DECISION_THRESHOLD = -0.0975


class EmailPredictor:
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

    def predict(self, email_text: str) -> dict:
        if not isinstance(email_text, str):
            raise TypeError("email_text must be a string.")

        if not email_text.strip():
            raise ValueError("email_text cannot be empty.")

        features = transform_text(
            self.pipeline,
            [email_text],
        )

        decision_score = float(
            self.model.decision_function(features)[0]
        )

        is_spam = decision_score >= self.threshold

        return {
            "label": "spam" if is_spam else "ham",
            "spam": is_spam,
            "decision_score": decision_score,
            "threshold": self.threshold,
        }