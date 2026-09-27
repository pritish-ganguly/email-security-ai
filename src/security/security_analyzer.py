import re
from dataclasses import dataclass

import pandas as pd


SUSPICIOUS_KEYWORDS = {
    "urgent",
    "verify",
    "verification",
    "account",
    "password",
    "login",
    "suspended",
    "suspend",
    "click",
    "winner",
    "prize",
    "claim",
    "refund",
    "payment",
    "invoice",
    "bank",
    "credit",
    "debit",
    "security",
    "confirm",
    "limited time",
    "act now",
}


@dataclass
class SecurityAnalysis:
    """Human-readable security indicators extracted from an email."""

    subject_length: int
    body_length: int
    word_count: int
    url_count: int
    ip_url_count: int
    suspicious_keyword_count: int
    html_present: bool
    script_present: bool
    attachment_count: int
    exclamation_count: int
    uppercase_ratio: float

    @property
    def has_attachments(self) -> bool:
        """Return whether the email contains attachments."""

        return self.attachment_count > 0

    @property
    def has_suspicious_keywords(self) -> bool:
        """Return whether suspicious keywords were detected."""

        return self.suspicious_keyword_count > 0

    def to_dict(self) -> dict:
        """Convert the analysis to a dictionary."""

        return {
            "subject_length": self.subject_length,
            "body_length": self.body_length,
            "word_count": self.word_count,
            "url_count": self.url_count,
            "ip_url_count": self.ip_url_count,
            "suspicious_keyword_count": (
                self.suspicious_keyword_count
            ),
            "html_present": self.html_present,
            "script_present": self.script_present,
            "attachment_count": self.attachment_count,
            "exclamation_count": self.exclamation_count,
            "uppercase_ratio": self.uppercase_ratio,
        }


def _safe_text(value: object) -> str:
    """Convert a value to a safe text string."""

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value)


def _count_urls(text: str) -> int:
    """Count HTTP, HTTPS and WWW URLs."""

    pattern = re.compile(
        r"(?:https?://|www\.)[^\s<>\"]+",
        re.IGNORECASE,
    )

    return len(pattern.findall(text))


def _count_ip_urls(text: str) -> int:
    """Count URLs whose hostname is an IPv4 address."""

    pattern = re.compile(
        r"https?://"
        r"(?:\d{1,3}\.){3}\d{1,3}"
        r"(?::\d+)?(?:/[^\s<>\"]*)?",
        re.IGNORECASE,
    )

    return len(pattern.findall(text))


def _count_suspicious_keywords(text: str) -> int:
    """Count distinct suspicious keywords found in the email."""

    normalized_text = text.lower()

    return sum(
        1
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in normalized_text
    )


def _calculate_uppercase_ratio(text: str) -> float:
    """Calculate the ratio of uppercase alphabetic characters."""

    alphabetic_characters = [
        character
        for character in text
        if character.isalpha()
    ]

    if not alphabetic_characters:
        return 0.0

    uppercase_characters = sum(
        character.isupper()
        for character in alphabetic_characters
    )

    return uppercase_characters / len(
        alphabetic_characters
    )


def analyze_email(
    subject: object,
    body: object,
    html_body: object = "",
    attachments: object = None,
) -> SecurityAnalysis:
    """
    Extract security indicators from an email.

    Parameters
    ----------
    subject:
        Email subject.

    body:
        Plain-text email body.

    html_body:
        HTML representation of the email, if available.

    attachments:
        Attachment collection, if available.
    """

    subject_text = _safe_text(subject)
    body_text = _safe_text(body)
    html_text = _safe_text(html_body)

    combined_text = (
        f"{subject_text}\n{body_text}\n{html_text}"
    )

    words = re.findall(
        r"\b\w+\b",
        body_text,
        flags=re.UNICODE,
    )

    attachment_count = 0

    if attachments is not None:

        if isinstance(attachments, (list, tuple, set)):
            attachment_count = len(attachments)

        elif isinstance(attachments, str):
            attachment_count = (
                1 if attachments.strip() else 0
            )

    return SecurityAnalysis(
        subject_length=len(subject_text),
        body_length=len(body_text),
        word_count=len(words),
        url_count=_count_urls(combined_text),
        ip_url_count=_count_ip_urls(combined_text),
        suspicious_keyword_count=(
            _count_suspicious_keywords(combined_text)
        ),
        html_present=bool(
            html_text.strip()
            or re.search(
                r"<html\b|<body\b|<div\b|<p\b",
                combined_text,
                re.IGNORECASE,
            )
        ),
        script_present=bool(
            re.search(
                r"<script\b",
                combined_text,
                re.IGNORECASE,
            )
        ),
        attachment_count=attachment_count,
        exclamation_count=combined_text.count("!"),
        uppercase_ratio=_calculate_uppercase_ratio(
            combined_text
        ),
    )


def print_security_analysis(
    analysis: SecurityAnalysis,
) -> None:
    """Print a readable security-analysis report."""

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — SECURITY ANALYSIS")
    print("=" * 70)

    print("\nContent indicators:")

    print(
        f"  Subject length          : "
        f"{analysis.subject_length}"
    )

    print(
        f"  Body length             : "
        f"{analysis.body_length}"
    )

    print(
        f"  Word count              : "
        f"{analysis.word_count}"
    )

    print(
        f"  URL count               : "
        f"{analysis.url_count}"
    )

    print(
        f"  IP-based URL count      : "
        f"{analysis.ip_url_count}"
    )

    print(
        f"  Suspicious keywords     : "
        f"{analysis.suspicious_keyword_count}"
    )

    print(
        f"  HTML present            : "
        f"{analysis.html_present}"
    )

    print(
        f"  Script present          : "
        f"{analysis.script_present}"
    )

    print(
        f"  Attachments             : "
        f"{analysis.attachment_count}"
    )

    print(
        f"  Exclamation marks       : "
        f"{analysis.exclamation_count}"
    )

    print(
        f"  Uppercase ratio         : "
        f"{analysis.uppercase_ratio:.4f}"
    )


def analyze_dataframe_row(
    row: pd.Series,
) -> SecurityAnalysis:
    """
    Analyze an email represented by a dataframe row.

    This helper keeps the analyzer compatible with the
    standardized dataset used by the project.
    """

    return analyze_email(
        subject=row.get("subject", ""),
        body=row.get("body", ""),
        html_body=row.get("html_body", ""),
        attachments=row.get("attachments"),
    )
