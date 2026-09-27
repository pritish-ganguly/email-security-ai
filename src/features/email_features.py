"""
Email security feature engineering.

This module extracts security-relevant features from
standardized email records.

The features are designed for:
1. Supervised machine learning
2. Unsupervised anomaly detection
3. Explainable security analysis
"""

import re

import pandas as pd


SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "verification",
    "password",
    "account",
    "suspended",
    "click here",
    "login",
    "confirm",
    "winner",
    "prize",
    "free",
    "offer",
    "claim",
    "refund",
    "payment",
    "bank",
    "invoice",
    "crypto",
    "bitcoin",
]


URL_PATTERN = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    re.IGNORECASE,
)

IP_URL_PATTERN = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}",
    re.IGNORECASE,
)


def calculate_text_length(
    text: str,
) -> int:
    """Return the number of characters in text."""

    if not text:
        return 0

    return len(text)


def calculate_word_count(
    text: str,
) -> int:
    """Return the number of whitespace-separated words."""

    if not text:
        return 0

    return len(text.split())


def count_urls(
    text: str,
) -> int:
    """Count URLs contained in an email."""

    if not text:
        return 0

    return len(URL_PATTERN.findall(text))


def count_ip_urls(
    text: str,
) -> int:
    """Count URLs containing an IPv4 address."""

    if not text:
        return 0

    return len(IP_URL_PATTERN.findall(text))


def calculate_url_ratio(
    text: str,
) -> float:
    """
    Calculate the ratio of URL characters to total
    email characters.
    """

    if not text:
        return 0.0

    url_count = count_urls(text)

    if url_count == 0:
        return 0.0

    return url_count / max(calculate_word_count(text), 1)


def count_suspicious_keywords(
    text: str,
) -> int:
    """Count suspicious security-related keywords."""

    if not text:
        return 0

    text_lower = text.lower()

    return sum(
        1
        for keyword in SUSPICIOUS_KEYWORDS
        if keyword in text_lower
    )


def contains_html(
    html_body: str,
) -> int:
    """Return 1 when HTML content is present."""

    if not html_body:
        return 0

    return int(
        bool(
            re.search(
                r"<[^>]+>",
                html_body,
            )
        )
    )


def contains_script(
    html_body: str,
) -> int:
    """Detect script tags in HTML content."""

    if not html_body:
        return 0

    return int(
        bool(
            re.search(
                r"<script\b",
                html_body,
                re.IGNORECASE,
            )
        )
    )


def count_exclamation_marks(
    text: str,
) -> int:
    """Count exclamation marks."""

    if not text:
        return 0

    return text.count("!")


def calculate_uppercase_ratio(
    text: str,
) -> float:
    """
    Calculate the ratio of uppercase alphabetic
    characters to all alphabetic characters.
    """

    if not text:
        return 0.0

    alphabetic_characters = [
        character
        for character in text
        if character.isalpha()
    ]

    if not alphabetic_characters:
        return 0.0

    uppercase_characters = [
        character
        for character in alphabetic_characters
        if character.isupper()
    ]

    return (
        len(uppercase_characters)
        / len(alphabetic_characters)
    )


def extract_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract security-related numerical features.

    The original dataframe is preserved.

    Returns:
        DataFrame containing the original records plus
        engineered security features.
    """

    dataframe = dataframe.copy()

    dataframe["subject_length"] = (
        dataframe["subject"]
        .fillna("")
        .apply(calculate_text_length)
    )

    dataframe["body_length"] = (
        dataframe["body"]
        .fillna("")
        .apply(calculate_text_length)
    )

    dataframe["word_count"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(calculate_word_count)
    )

    dataframe["url_count"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(count_urls)
    )

    dataframe["ip_url_count"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(count_ip_urls)
    )

    dataframe["url_ratio"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(calculate_url_ratio)
    )

    dataframe["suspicious_keyword_count"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(count_suspicious_keywords)
    )

    dataframe["html_present"] = (
        dataframe["html_body"]
        .fillna("")
        .apply(contains_html)
    )

    dataframe["script_present"] = (
        dataframe["html_body"]
        .fillna("")
        .apply(contains_script)
    )

    dataframe["exclamation_count"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(count_exclamation_marks)
    )

    dataframe["uppercase_ratio"] = (
        dataframe["email_text"]
        .fillna("")
        .apply(calculate_uppercase_ratio)
    )

    return dataframe


def print_feature_summary(
    dataframe: pd.DataFrame,
) -> None:
    """Print the engineered feature summary."""

    feature_columns = [
        "subject_length",
        "body_length",
        "word_count",
        "url_count",
        "ip_url_count",
        "url_ratio",
        "suspicious_keyword_count",
        "html_present",
        "script_present",
        "exclamation_count",
        "uppercase_ratio",
    ]

    print("\n" + "=" * 70)
    print("EMAIL SECURITY AI — FEATURE SUMMARY")
    print("=" * 70)

    print("\nFeature columns:")

    for column in feature_columns:
        print(f"  - {column}")

    print("\nFeature statistics:")

    print(
        dataframe[feature_columns]
        .describe()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":

    from src.data.spamassassin_loader import (
        load_spamassassin,
    )

    from src.preprocessing.cleaner import (
        clean_dataset,
    )

    dataframe = load_spamassassin()

    dataframe, _ = clean_dataset(
        dataframe
    )

    dataframe = extract_features(
        dataframe
    )

    print_feature_summary(
        dataframe
    )
