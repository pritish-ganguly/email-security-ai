from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from src.data.email_parser import parse_email


HAM_DIR = Path("data/raw/spamassassin/easy_ham")
SPAM_DIR = Path("data/raw/spamassassin/spam")

MAX_FILES = None


def get_value(
    data: object,
    field: str,
    default: object = "",
) -> object:
    if isinstance(data, dict):
        return data.get(field, default)

    return getattr(data, field, default)


def collect_emails(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
    ]

    files.sort()

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    return files


def build_text(email_data: object) -> str:
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


def count_words(text: str) -> int:
    return len(
        text.split()
    )


def count_urls(text: str) -> int:
    lowered = text.lower()

    return (
        lowered.count("http://")
        + lowered.count("https://")
        + lowered.count("www.")
    )


def is_html_present(
    html_body: str,
) -> bool:
    return bool(
        html_body.strip()
    )


def analyse_dataset(
    label: str,
    files: Iterable[Path],
) -> dict:
    total = 0
    parse_errors = 0

    word_counts: list[int] = []
    subject_lengths: list[int] = []
    body_lengths: list[int] = []
    html_lengths: list[int] = []
    url_counts: list[int] = []

    short_count = 0
    medium_count = 0
    long_count = 0
    html_count = 0

    error_files: list[str] = []

    for path in files:
        try:
            email_data = parse_email(path)

        except Exception:
            parse_errors += 1

            if len(error_files) < 20:
                error_files.append(
                    path.name
                )

            continue

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

        text = build_text(
            email_data
        )

        words = count_words(
            text
        )

        urls = count_urls(
            text
        )

        total += 1

        word_counts.append(words)

        subject_lengths.append(
            len(subject)
        )

        body_lengths.append(
            len(body)
        )

        html_lengths.append(
            len(html_body)
        )

        url_counts.append(
            urls
        )

        if words <= 40:
            short_count += 1

        elif words <= 200:
            medium_count += 1

        else:
            long_count += 1

        if is_html_present(
            html_body
        ):
            html_count += 1

    return {
        "label": label,
        "total": total,
        "parse_errors": parse_errors,
        "short": short_count,
        "medium": medium_count,
        "long": long_count,
        "html": html_count,
        "word_counts": word_counts,
        "subject_lengths": subject_lengths,
        "body_lengths": body_lengths,
        "html_lengths": html_lengths,
        "url_counts": url_counts,
        "error_files": error_files,
    }


def average(
    values: list[int],
) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def percentile(
    values: list[int],
    percentage: float,
) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)

    index = int(
        round(
            (percentage / 100)
            * (len(ordered) - 1)
        )
    )

    return float(
        ordered[index]
    )


def print_distribution(
    result: dict,
) -> None:
    label = result["label"]

    print()
    print("=" * 78)
    print(
        f"{label.upper()} TRAINING DATA"
    )
    print("=" * 78)

    print(
        f"Emails parsed       : "
        f"{result['total']}"
    )

    print(
        f"Parse errors        : "
        f"{result['parse_errors']}"
    )

    print()

    print(
        "MESSAGE LENGTH DISTRIBUTION"
    )

    print(
        f"<= 40 words         : "
        f"{result['short']}"
    )

    print(
        f"41-200 words        : "
        f"{result['medium']}"
    )

    print(
        f"> 200 words         : "
        f"{result['long']}"
    )

    print()

    print(
        "HTML"
    )

    print(
        f"HTML emails         : "
        f"{result['html']}"
    )

    html_percentage = (
        result["html"]
        / result["total"]
        * 100
        if result["total"]
        else 0
    )

    print(
        f"HTML percentage     : "
        f"{html_percentage:.2f}%"
    )

    print()

    print(
        "WORD COUNT"
    )

    words = result[
        "word_counts"
    ]

    print(
        f"Average             : "
        f"{average(words):.2f}"
    )

    print(
        f"Median-ish          : "
        f"{percentile(words, 50):.0f}"
    )

    print(
        f"P90                 : "
        f"{percentile(words, 90):.0f}"
    )

    print(
        f"P95                 : "
        f"{percentile(words, 95):.0f}"
    )

    print(
        f"Maximum             : "
        f"{max(words) if words else 0}"
    )

    print()

    print(
        "URL COUNT"
    )

    urls = result[
        "url_counts"
    ]

    print(
        f"Average             : "
        f"{average(urls):.2f}"
    )

    print(
        f"P50                 : "
        f"{percentile(urls, 50):.0f}"
    )

    print(
        f"P90                 : "
        f"{percentile(urls, 90):.0f}"
    )

    print(
        f"Maximum             : "
        f"{max(urls) if urls else 0}"
    )


def compare_distributions(
    ham: dict,
    spam: dict,
) -> None:
    print()
    print("=" * 78)
    print(
        "HAM vs SPAM TRAINING DISTRIBUTION"
    )
    print("=" * 78)

    print(
        f"{'Metric':<28}"
        f"{'HAM':>18}"
        f"{'SPAM':>18}"
    )

    print("-" * 78)

    rows = [
        (
            "Total emails",
            ham["total"],
            spam["total"],
        ),
        (
            "Short <=40 words",
            ham["short"],
            spam["short"],
        ),
        (
            "Medium 41-200",
            ham["medium"],
            spam["medium"],
        ),
        (
            "Long >200",
            ham["long"],
            spam["long"],
        ),
        (
            "HTML emails",
            ham["html"],
            spam["html"],
        ),
    ]

    for name, ham_value, spam_value in rows:
        print(
            f"{name:<28}"
            f"{ham_value:>18}"
            f"{spam_value:>18}"
        )

    print()

    ham_words = ham[
        "word_counts"
    ]

    spam_words = spam[
        "word_counts"
    ]

    print(
        f"{'Average word count':<28}"
        f"{average(ham_words):>18.2f}"
        f"{average(spam_words):>18.2f}"
    )

    print(
        f"{'P90 word count':<28}"
        f"{percentile(ham_words, 90):>18.0f}"
        f"{percentile(spam_words, 90):>18.0f}"
    )

    print(
        f"{'Average URL count':<28}"
        f"{average(ham['url_counts']):>18.2f}"
        f"{average(spam['url_counts']):>18.2f}"
    )

    print(
        f"{'Average HTML length':<28}"
        f"{average(ham['html_lengths']):>18.2f}"
        f"{average(spam['html_lengths']):>18.2f}"
    )


def main() -> None:
    print()
    print("=" * 78)
    print(
        "EMAIL SECURITY AI — TRAINING DATA AUDIT"
    )
    print("=" * 78)

    print()
    print(
        "Loading HAM emails..."
    )

    ham_files = collect_emails(
        HAM_DIR
    )

    print(
        f"HAM files discovered : "
        f"{len(ham_files)}"
    )

    print()
    print(
        "Loading SPAM emails..."
    )

    spam_files = collect_emails(
        SPAM_DIR
    )

    print(
        f"SPAM files discovered: "
        f"{len(spam_files)}"
    )

    print()
    print(
        "Analysing HAM dataset..."
    )

    ham = analyse_dataset(
        "HAM",
        ham_files,
    )

    print(
        "Analysing SPAM dataset..."
    )

    spam = analyse_dataset(
        "SPAM",
        spam_files,
    )

    print_distribution(
        ham
    )

    print_distribution(
        spam
    )

    compare_distributions(
        ham,
        spam,
    )

    print()
    print("=" * 78)
    print(
        "TRAINING DATA AUDIT COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        "This is diagnostic only."
    )

    print(
        "No model artifacts were modified."
    )


if __name__ == "__main__":
    main()
