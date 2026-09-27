from src.security.security_analyzer import (
    analyze_email,
    print_security_analysis,
)


def main() -> None:
    """Run a simple security-analysis test."""

    analysis = analyze_email(
        subject="URGENT: Verify your account now!",
        body=(
            "Your account has been suspended. "
            "Click https://192.168.1.10/login "
            "to verify your password immediately."
        ),
        html_body="<html><body>Verify now!</body></html>",
        attachments=["invoice.pdf"],
    )

    print_security_analysis(analysis)


if __name__ == "__main__":
    main()
