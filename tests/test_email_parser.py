"""
Test the raw email parser.
"""

from pathlib import Path

from src.data.email_parser import parse_email


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPAMASSASSIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spamassassin"
)


def test_parse_spamassassin_email():

    email_files = list(
        (SPAMASSASSIN_DIR / "easy_ham").iterdir()
    )

    assert email_files

    email_data = parse_email(email_files[0])

    assert isinstance(email_data, dict)

    assert "sender" in email_data
    assert "receiver" in email_data
    assert "subject" in email_data
    assert "body" in email_data
    assert "html_body" in email_data
    assert "attachments" in email_data
