from src.models.predict import EmailPredictor
from src.security.security_analyzer import (
    analyze_email,
    print_security_analysis,
)
from src.security.risk_engine import (
    assess_email_risk,
    print_risk_assessment,
)
from src.security.decision_engine import (
    make_decision,
    print_decision,
)


def scan_text(
    email_text: str,
    predictor: EmailPredictor,
    subject: str = "",
):
    if not isinstance(email_text, str):
        raise TypeError(
            "email_text must be a string."
        )

    if not email_text.strip():
        raise ValueError(
            "Email text cannot be empty."
        )

    print("Running ML prediction...")

    ml_result = predictor.predict(
        email_text
    )

    print(
        "Running security analysis..."
    )

    security_analysis = analyze_email(
        subject=subject,
        body=email_text,
        html_body="",
        attachments=[],
    )

    print(
        "Running risk assessment..."
    )

    risk_assessment = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_analysis,
    )

    print(
        "Running decision engine..."
    )

    decision = make_decision(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EMAIL SECURITY AI — TEXT ANALYSIS"
    )

    print(
        "=" * 70
    )

    print("\nEmail information")
    print("-" * 70)

    print(
        f"\nSubject:\n"
        f"  {subject or '(No subject)'}"
    )

    print(
        f"\nText length:\n"
        f"  {len(email_text)} characters"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MACHINE LEARNING RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPrediction : "
        f"{ml_result['label']}"
    )

    print(
        f"Spam       : "
        f"{ml_result['spam']}"
    )

    print(
        f"Score      : "
        f"{ml_result['decision_score']:.4f}"
    )

    print(
        f"Threshold  : "
        f"{ml_result['threshold']:.4f}"
    )

    print_security_analysis(
        security_analysis
    )

    print_risk_assessment(
        risk_assessment
    )

    print_decision(
        decision
    )

    return {
        "ml_result": ml_result,
        "security_analysis": security_analysis,
        "risk_assessment": risk_assessment,
        "decision": decision,
    }


def main():
    print(
        "\nLoading Email Security AI model..."
    )

    predictor = EmailPredictor()

    subject = input(
        "\nEnter subject "
        "(press Enter if none): "
    ).strip()

    print(
        "\nPaste the email text below."
    )

    print(
        "Type END on a new line when finished.\n"
    )

    lines = []

    while True:
        try:
            line = input()

        except EOFError:
            break

        if line.strip() == "END":
            break

        lines.append(line)

    email_text = "\n".join(lines)

    scan_text(
        email_text=email_text,
        predictor=predictor,
        subject=subject,
    )


if __name__ == "__main__":
    main()
