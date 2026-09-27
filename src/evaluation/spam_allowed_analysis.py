from pathlib import Path

from src.data.email_parser import parse_email
from src.models.predict import EmailPredictor
from src.security.security_analyzer import analyze_email
from src.security.risk_engine import assess_email_risk
from src.security.decision_engine import make_decision


SPAM_DIR = Path(
    "data/raw/spamassassin/spam"
)


def get_value(
    data,
    field,
    default="",
):
    if isinstance(data, dict):
        return data.get(
            field,
            default,
        )

    return getattr(
        data,
        field,
        default,
    )


def build_email_text(
    email_data,
):
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


def analyze_spam_file(
    email_path,
    predictor,
):
    email_data = parse_email(
        email_path
    )

    email_text = build_email_text(
        email_data
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
        "path": email_path,
        "email_data": email_data,
        "ml_result": ml_result,
        "security_analysis": security_analysis,
        "risk_assessment": risk_assessment,
        "decision": decision,
    }


def print_indicator(
    analysis,
    name,
    label,
    default=0,
):
    value = get_value(
        analysis,
        name,
        default,
    )

    print(
        f"{label:<28}: {value}"
    )


def print_analysis(
    result,
    number,
):
    email_path = result["path"]
    ml_result = result["ml_result"]
    security = result["security_analysis"]
    risk = result["risk_assessment"]
    decision = result["decision"]
    email_data = result["email_data"]

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

    print(
        "\n" + "=" * 78
    )

    print(
        f"SPAM ALLOWED ANALYSIS #{number}"
    )

    print(
        "=" * 78
    )

    print(
        f"\nFile:\n  {email_path}"
    )

    print(
        f"\nSubject:\n  {subject}"
    )

    print(
        "\nMESSAGE SIZE"
    )

    print(
        "-" * 78
    )

    print(
        f"Body length                : "
        f"{len(body)}"
    )

    print(
        f"HTML length                : "
        f"{len(html_body)}"
    )

    print(
        "\nMACHINE LEARNING"
    )

    print(
        "-" * 78
    )

    print(
        f"Prediction                 : "
        f"{ml_result['label']}"
    )

    print(
        f"Spam                       : "
        f"{ml_result['spam']}"
    )

    print(
        f"Decision score             : "
        f"{ml_result['decision_score']:.4f}"
    )

    print(
        f"Threshold                  : "
        f"{ml_result['threshold']:.4f}"
    )

    print(
        "\nSECURITY INDICATORS"
    )

    print(
        "-" * 78
    )

    print_indicator(
        security,
        "subject_length",
        "Subject length",
    )

    print_indicator(
        security,
        "body_length",
        "Body length",
    )

    print_indicator(
        security,
        "word_count",
        "Word count",
    )

    print_indicator(
        security,
        "url_count",
        "URL count",
    )

    print_indicator(
        security,
        "ip_based_url_count",
        "IP-based URL count",
    )

    print_indicator(
        security,
        "suspicious_keywords",
        "Suspicious keywords",
    )

    print_indicator(
        security,
        "html_present",
        "HTML present",
        False,
    )

    print_indicator(
        security,
        "script_present",
        "Script present",
        False,
    )

    print_indicator(
        security,
        "attachments",
        "Attachments",
        0,
    )

    print_indicator(
        security,
        "exclamation_marks",
        "Exclamation marks",
    )

    print_indicator(
        security,
        "uppercase_ratio",
        "Uppercase ratio",
    )

    print(
        "\nRISK ASSESSMENT"
    )

    print(
        "-" * 78
    )

    print(
        f"Risk score                 : "
        f"{get_value(risk, 'risk_score', 0)}/100"
    )

    print(
        f"Risk level                 : "
        f"{get_value(risk, 'risk_level', 'UNKNOWN')}"
    )

    reasons = get_value(
        risk,
        "reasons",
        [],
    )

    if reasons:
        print(
            "\nRisk reasons:"
        )

        for index, reason in enumerate(
            reasons,
            start=1,
        ):
            print(
                f"  {index}. {reason}"
            )

    print(
        "\nFINAL DECISION"
    )

    print(
        "-" * 78
    )

    print(
        f"Classification             : "
        f"{decision.classification}"
    )

    print(
        f"Action                     : "
        f"{decision.action}"
    )

    print(
        f"Confidence                 : "
        f"{decision.confidence}"
    )

    print(
        f"Reason                     : "
        f"{decision.reason}"
    )

    print(
        "\nBODY PREVIEW"
    )

    print(
        "-" * 78
    )

    preview = body.strip()

    if not preview:
        preview = "[empty — inspect HTML content]"

    preview = preview[:1000]

    print(preview)

    if html_body:
        print(
            "\nHTML PREVIEW"
        )

        print(
            "-" * 78
        )

        print(
            html_body[:1000]
        )


def main():
    print(
        "\n" + "=" * 78
    )

    print(
        "EMAIL SECURITY AI — SPAM ALLOWED ANALYSIS"
    )

    print(
        "=" * 78
    )

    print(
        "\nLoading frozen production model..."
    )

    predictor = EmailPredictor()

    spam_files = sorted(
        path
        for path in SPAM_DIR.iterdir()
        if path.is_file()
    )

    print(
        f"\nSPAM emails discovered: "
        f"{len(spam_files):,}"
    )

    print(
        "\nFinding SPAM emails that receive ALLOW..."
    )

    allowed_results = []

    for email_path in spam_files:
        try:
            result = analyze_spam_file(
                email_path,
                predictor,
            )

            decision = result["decision"]

            if decision.action == "ALLOW":
                allowed_results.append(
                    result
                )

        except Exception as exc:
            print(
                f"\nWARNING: Could not analyze "
                f"{email_path}"
            )

            print(
                f"Reason: {exc}"
            )

    print(
        f"\nSPAM emails receiving ALLOW: "
        f"{len(allowed_results)}"
    )

    if not allowed_results:
        print(
            "\nNo SPAM emails were allowed."
        )

        return

    for index, result in enumerate(
        allowed_results,
        start=1,
    ):
        print_analysis(
            result,
            index,
        )

    print(
        "\n" + "=" * 78
    )

    print(
        "SPAM ALLOWED ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
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
