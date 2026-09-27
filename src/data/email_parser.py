"""
Email parser for the Email Security AI project.

Parses RFC-compatible .eml files into a standardized dictionary
that can be consumed by the inference layer.
"""

from email import policy
from email.parser import BytesParser
from email.message import Message
from pathlib import Path


def _decode_payload(part: Message) -> str:
    """Decode a text MIME part safely."""

    try:
        payload = part.get_payload(
            decode=True
        )

        if payload is None:
            return ""

        charset = part.get_content_charset()

        if charset:
            return payload.decode(
                charset,
                errors="replace",
            )

        return payload.decode(
            errors="replace"
        )

    except (UnicodeDecodeError, LookupError):
        return ""


def parse_email(file_path: str | Path) -> dict:
    """
    Parse an .eml file.

    Returns:
        Dictionary containing email metadata, text content,
        HTML content, and attachment information.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Email file not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Email path is not a file: {path}"
        )

    with path.open(
        "rb"
    ) as email_file:

        message = BytesParser(
            policy=policy.default
        ).parse(email_file)

    plain_text_parts = []
    html_parts = []
    attachments = []

    if message.is_multipart():

        for part in message.walk():

            content_disposition = (
                part.get_content_disposition()
            )

            if content_disposition == "attachment":

                filename = part.get_filename()

                if filename:
                    attachments.append(
                        filename
                    )

                continue

            content_type = (
                part.get_content_type()
            )

            if content_type == "text/plain":
                plain_text_parts.append(
                    _decode_payload(part)
                )

            elif content_type == "text/html":
                html_parts.append(
                    _decode_payload(part)
                )

    else:

        content_type = (
            message.get_content_type()
        )

        body = _decode_payload(message)

        if content_type == "text/html":
            html_parts.append(body)
        else:
            plain_text_parts.append(body)

    body = "\n".join(
        part
        for part in plain_text_parts
        if part.strip()
    )

    html_body = "\n".join(
        part
        for part in html_parts
        if part.strip()
    )

    subject = message.get(
        "Subject",
        "",
    )

    sender = message.get(
        "From",
        "",
    )

    receiver = message.get(
        "To",
        "",
    )

    date = message.get(
        "Date",
        "",
    )

    return {
        "sender": sender,
        "receiver": receiver,
        "subject": subject,
        "date": date,
        "body": body,
        "html_body": html_body,
        "attachments": attachments,
    }
