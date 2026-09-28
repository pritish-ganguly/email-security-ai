from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.models.predict import EmailPredictor
from src.scanner.email_scanner import EmailScanner
from src.security.decision_engine_v31 import make_decision_v31
from src.security.risk_engine import assess_email_risk
from src.security.security_analyzer import analyze_email


APP_NAME = "Email Security AI"
APP_VERSION = "2.0"
PRODUCTION_MODEL = "Linear SVM"
DECISION_THRESHOLD = -0.0975
MAX_FILE_SIZE_MB = 10


st.set_page_config(
    page_title="Email Security AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at 5% 0%, rgba(14,165,233,.10), transparent 28%),
        radial-gradient(circle at 95% 5%, rgba(16,185,129,.08), transparent 26%),
        radial-gradient(circle at 50% 100%, rgba(37,99,235,.05), transparent 35%),
        #030811;
    color: #e5edf5;
}

.main .block-container {
    max-width: 1500px;
    padding: 2rem 2rem 4rem;
}

[data-testid="stHeader"] {
    background: rgba(3,8,17,.88);
}

[data-testid="stAppViewContainer"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#030811 0%,#06101c 48%,#030811 100%);
    border-right: 1px solid rgba(148,163,184,.08);
}

[data-testid="stSidebar"] * {
    color: #dce7f1;
}

.sidebar-brand {
    padding: .4rem 0 1rem;
}

.sidebar-brand-title {
    color: #f8fafc;
    font-size: 1.12rem;
    font-weight: 850;
}

.sidebar-brand-subtitle {
    color: #60768b;
    font-size: .66rem;
    margin-top: .25rem;
}

.sidebar-label {
    color: #50667a;
    font-size: .58rem;
    font-weight: 850;
    letter-spacing: .14em;
    text-transform: uppercase;
    margin-top: 1.2rem;
    margin-bottom: .5rem;
}

.sidebar-card {
    padding: .8rem;
    margin-bottom: .45rem;
    border-radius: 12px;
    border: 1px solid rgba(148,163,184,.08);
    background: linear-gradient(145deg,rgba(15,31,48,.72),rgba(6,17,29,.72));
}

.sidebar-card-title {
    color: #e5edf5;
    font-size: .74rem;
    font-weight: 800;
}

.sidebar-card-value {
    color: #67e8f9;
    font-size: .78rem;
    font-weight: 800;
    margin-top: .15rem;
}

.sidebar-card-caption {
    color: #53697e;
    font-size: .61rem;
    margin-top: .18rem;
}

.sidebar-footer {
    margin-top: 2rem;
    padding: .9rem;
    border-radius: 13px;
    text-align: center;
    border: 1px solid rgba(14,165,233,.09);
    background: rgba(14,165,233,.035);
}

.sidebar-footer-label {
    color: #465b70;
    font-size: .55rem;
    font-weight: 800;
    letter-spacing: .14em;
    text-transform: uppercase;
}

.sidebar-footer-name {
    color: #e8f1f8;
    font-size: .82rem;
    font-weight: 800;
    margin-top: .25rem;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.5rem 2.6rem;
    margin-bottom: 1.2rem;
    border-radius: 24px;
    border: 1px solid rgba(56,189,248,.15);
    background:
        linear-gradient(
            135deg,
            rgba(6,18,31,.98),
            rgba(5,28,43,.96),
            rgba(4,35,34,.94)
        );
    box-shadow:
        0 30px 80px rgba(0,0,0,.42),
        inset 0 1px 0 rgba(255,255,255,.035);
}

.hero:before {
    content: "";
    position: absolute;
    width: 390px;
    height: 390px;
    right: -140px;
    top: -190px;
    border-radius: 50%;
    background: radial-gradient(circle,rgba(34,211,238,.10),transparent 68%);
}

.hero:after {
    content: "";
    position: absolute;
    width: 300px;
    height: 300px;
    left: -170px;
    bottom: -190px;
    border-radius: 50%;
    background: radial-gradient(circle,rgba(14,165,233,.08),transparent 68%);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-eyebrow {
    color: #67e8f9;
    font-size: .62rem;
    font-weight: 850;
    letter-spacing: .16em;
    text-transform: uppercase;
    margin-bottom: .65rem;
}

.hero-title {
    color: #f8fafc;
    font-size: 2.7rem;
    font-weight: 900;
    line-height: 1.05;
    letter-spacing: -.055em;
}

.hero-subtitle {
    max-width: 900px;
    color: #91a7ba;
    font-size: .9rem;
    line-height: 1.7;
    margin-top: .75rem;
}

.hero-badges {
    display: flex;
    gap: .55rem;
    flex-wrap: wrap;
    margin-top: 1.15rem;
}

.hero-badge {
    display: inline-block;
    padding: .42rem .72rem;
    border-radius: 999px;
    color: #9eeaf7;
    background: rgba(14,165,233,.07);
    border: 1px solid rgba(14,165,233,.15);
    font-size: .58rem;
    font-weight: 850;
    letter-spacing: .06em;
}

.hero-badge-green {
    color: #7ce8bd;
    background: rgba(16,185,129,.07);
    border-color: rgba(16,185,129,.16);
}

.status-card {
    min-height: 105px;
    padding: 1rem 1.05rem;
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,.085);
    background: linear-gradient(145deg,rgba(12,28,44,.90),rgba(6,16,28,.92));
    box-shadow: 0 12px 30px rgba(0,0,0,.18);
}

.status-label {
    color: #5e7387;
    font-size: .59rem;
    font-weight: 850;
    letter-spacing: .12em;
    text-transform: uppercase;
}

.status-value {
    color: #f1f5f9;
    font-size: 1rem;
    font-weight: 850;
    margin-top: .45rem;
}

.status-caption {
    color: #556b80;
    font-size: .64rem;
    margin-top: .25rem;
}

.online-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    margin-right: 7px;
    box-shadow: 0 0 0 4px rgba(52,211,153,.07),0 0 15px rgba(52,211,153,.55);
}

.section-title {
    color: #f1f5f9;
    font-size: 1.25rem;
    font-weight: 850;
    letter-spacing: -.025em;
    margin-top: 1.6rem;
}

.section-caption {
    color: #61768a;
    font-size: .7rem;
    margin-top: .18rem;
    margin-bottom: .75rem;
}

.panel {
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid rgba(148,163,184,.08);
    background: linear-gradient(145deg,rgba(9,22,37,.90),rgba(4,12,22,.92));
    box-shadow: 0 12px 30px rgba(0,0,0,.16);
}

.info-panel {
    min-height: 105px;
    padding: 1rem;
    border-radius: 15px;
    border: 1px solid rgba(148,163,184,.08);
    background: linear-gradient(145deg,rgba(10,23,38,.92),rgba(6,15,27,.92));
}

.info-label {
    color: #5b7185;
    font-size: .58rem;
    font-weight: 850;
    letter-spacing: .1em;
    text-transform: uppercase;
}

.info-value {
    color: #dce7f1;
    font-size: .8rem;
    line-height: 1.5;
    margin-top: .3rem;
    overflow-wrap: anywhere;
}

[data-testid="stMetric"] {
    min-height: 100px;
    padding: .85rem .9rem;
    border-radius: 14px;
    border: 1px solid rgba(148,163,184,.085);
    background: linear-gradient(145deg,rgba(12,27,43,.95),rgba(6,16,28,.95));
    box-shadow: 0 10px 26px rgba(0,0,0,.16);
}

[data-testid="stMetricLabel"] {
    color: #61778b !important;
    font-size: .63rem !important;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 1.25rem !important;
    font-weight: 850 !important;
}

.stTextArea textarea {
    background: #050e19 !important;
    color: #dfeaf3 !important;
    border: 1px solid rgba(148,163,184,.13) !important;
    border-radius: 12px !important;
}

.stTextArea textarea:focus {
    border-color: rgba(56,189,248,.38) !important;
    box-shadow: 0 0 0 1px rgba(56,189,248,.08) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(5,15,27,.80);
    border: 1px dashed rgba(56,189,248,.18);
    border-radius: 13px;
}

.stButton > button {
    min-height: 2.65rem;
    border-radius: 10px;
    font-weight: 800;
    border: 1px solid rgba(56,189,248,.15);
}

.stButton > button:hover {
    border-color: rgba(56,189,248,.38);
}

button[data-baseweb="tab"] {
    color: #63788c !important;
    font-weight: 750 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #67e8f9 !important;
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#0891b2,#22c55e);
}

.footer {
    margin-top: 3rem;
    padding-top: 1.25rem;
    text-align: center;
    color: #465c70;
    font-size: .64rem;
}

.footer-line {
    height: 1px;
    margin-bottom: 1rem;
    background: linear-gradient(90deg,transparent,rgba(56,189,248,.15),transparent);
}

.footer-name {
    color: #71869a;
    font-weight: 800;
}

@media (max-width: 900px) {
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        padding: 1.7rem;
    }

    .hero-title {
        font-size: 2rem;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def to_dict(obj: Any) -> dict:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    method = getattr(obj, "to_dict", None)

    if callable(method):
        result = method()
        if isinstance(result, dict):
            return result

    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)

    raise TypeError(
        f"Expected dictionary-compatible result, got {type(obj).__name__}."
    )


def build_text(
    subject: str,
    body: str,
    html_body: str = "",
) -> str:
    return (
        f"Subject: {subject}\n\n"
        f"{body}\n\n"
        f"{html_body}"
    ).strip()


@st.cache_resource
def load_scanner() -> EmailScanner:
    return EmailScanner()


@st.cache_resource
def load_predictor() -> EmailPredictor:
    return EmailPredictor()


def analyze_email_components(
    predictor: EmailPredictor,
    subject: str,
    body: str,
    html_body: str = "",
    attachments: list | None = None,
    email_metadata: dict | None = None,
) -> dict:

    subject = str(subject or "")
    body = str(body or "")
    html_body = str(html_body or "")

    attachments = list(attachments or [])

    email_text = build_text(
        subject,
        body,
        html_body,
    )

    if not email_text.strip():
        raise ValueError(
            "The email does not contain usable text."
        )

    ml_result = predictor.predict(
        email_text
    )

    security_result = analyze_email(
        subject=subject,
        body=body,
        html_body=html_body,
        attachments=attachments,
    )

    security_analysis = to_dict(
        security_result
    )

    risk_result = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security_analysis,
    )

    risk_assessment = to_dict(
        risk_result
    )

    decision_result = make_decision_v31(
        ml_result=ml_result,
        security_analysis=security_analysis,
        risk_assessment=risk_assessment,
    )

    decision = to_dict(
        decision_result
    )

    metadata = {
        "sender": "",
        "receiver": "",
        "subject": subject,
        "date": "",
        "attachments": attachments,
    }

    if email_metadata:
        metadata.update(
            email_metadata
        )

    return {
        "email": metadata,
        "ml_result": to_dict(ml_result),
        "security_analysis": security_analysis,
        "risk_assessment": risk_assessment,
        "decision": decision,
    }


def normalize_result(
    result: Any,
) -> dict:

    if not isinstance(result, dict):
        raise TypeError(
            "Analysis result must be a dictionary."
        )

    normalized = dict(result)

    for key in (
        "email",
        "ml_result",
        "security_analysis",
        "risk_assessment",
        "decision",
    ):
        normalized[key] = to_dict(
            normalized.get(
                key,
                {},
            )
        )

    return normalized


def scan_uploaded(
    scanner: EmailScanner,
    uploaded_file: Any,
) -> dict:

    if uploaded_file is None:
        raise ValueError(
            "No email file was uploaded."
        )

    data = uploaded_file.getvalue()

    if not data:
        raise ValueError(
            "The uploaded file is empty."
        )

    max_size = (
        MAX_FILE_SIZE_MB
        * 1024
        * 1024
    )

    if len(data) > max_size:
        raise ValueError(
            f"File exceeds the {MAX_FILE_SIZE_MB} MB limit."
        )

    filename = str(
        uploaded_file.name or ""
    )

    if Path(filename).suffix.lower() != ".eml":
        raise ValueError(
            "Only .eml files are supported."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=".eml",
        ) as temp_file:

            temp_file.write(data)

            temp_path = Path(
                temp_file.name
            )

        result = scanner.scan(
            temp_path
        )

        return normalize_result(
            result
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(
                missing_ok=True
            )


def scan_text(
    predictor: EmailPredictor,
    raw_text: str,
) -> dict:

    text = str(
        raw_text or ""
    ).strip()

    if not text:
        raise ValueError(
            "Please paste an email before scanning."
        )

    subject = ""
    sender = ""
    receiver = ""
    date = ""
    body = text

    lines = text.splitlines()
    header_end = None

    for index, line in enumerate(lines):

        if not line.strip():
            header_end = index
            break

        lower = line.lower()

        if lower.startswith("subject:"):
            subject = line.split(
                ":",
                1,
            )[1].strip()

        elif lower.startswith("from:"):
            sender = line.split(
                ":",
                1,
            )[1].strip()

        elif lower.startswith("to:"):
            receiver = line.split(
                ":",
                1,
            )[1].strip()

        elif lower.startswith("date:"):
            date = line.split(
                ":",
                1,
            )[1].strip()

    if header_end is not None:
        body = "\n".join(
            lines[header_end + 1:]
        ).strip()

    result = analyze_email_components(
        predictor=predictor,
        subject=subject,
        body=body,
        html_body="",
        attachments=[],
        email_metadata={
            "sender": sender,
            "receiver": receiver,
            "subject": subject,
            "date": date,
            "attachments": [],
        },
    )

    return normalize_result(
        result
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-content">
                <div class="hero-eyebrow">SECURITY INTELLIGENCE PLATFORM</div>
                <div class="hero-title">Email Security AI</div>
                <div class="hero-subtitle">
                    Intelligent email threat detection powered by
                    machine-learning classification, security signal
                    analysis, risk assessment and policy-based
                    decision intelligence.
                </div>
                <div class="hero-badges">
                    <span class="hero-badge">● FROZEN PRODUCTION INFERENCE</span>
                    <span class="hero-badge hero-badge-green">● SECURITY ANALYSIS ENABLED</span>
                    <span class="hero-badge">● DECISION ENGINE V3.1</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_cards() -> None:

    col1, col2, col3 = st.columns(
        3,
        gap="medium",
    )

    with col1:

        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Engine Status
                </div>
                <div class="status-value">
                    <span class="online-dot"></span>
                    ONLINE
                </div>
                <div class="status-caption">
                    Production inference ready
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    Production Model
                </div>
                <div class="status-value">
                    Linear SVM
                </div>
                <div class="status-caption">
                    Frozen production classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="status-card">
                <div class="status-label">
                    NLP Representation
                </div>
                <div class="status-value">
                    WORD + CHAR TF-IDF
                </div>
                <div class="status-caption">
                    Production text representation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_section(
    title: str,
    caption: str,
) -> None:

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        <div class="section-caption">
            {caption}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:

    with st.sidebar:

        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">
                    🛡️ {APP_NAME}
                </div>
                <div class="sidebar-brand-subtitle">
                    Intelligent email threat analysis platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            f"Production Dashboard • v{APP_VERSION}"
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-label">SYSTEM</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    <span class="online-dot"></span>
                    Inference Engine
                </div>
                <div class="sidebar-card-value">
                    ONLINE
                </div>
                <div class="sidebar-card-caption">
                    Ready for email analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-label">MACHINE LEARNING</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    Production Model
                </div>
                <div class="sidebar-card-value">
                    {PRODUCTION_MODEL}
                </div>
                <div class="sidebar-card-caption">
                    Frozen classifier
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    NLP Features
                </div>
                <div class="sidebar-card-value">
                    WORD + CHAR
                </div>
                <div class="sidebar-card-caption">
                    TF-IDF representation
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    Decision Threshold
                </div>
                <div class="sidebar-card-value">
                    {DECISION_THRESHOLD:.4f}
                </div>
                <div class="sidebar-card-caption">
                    Frozen operating point
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-label">PIPELINE</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    01 · ML Classification
                </div>
                <div class="sidebar-card-caption">
                    Content classification
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    02 · Security Analysis
                </div>
                <div class="sidebar-card-caption">
                    Threat signal extraction
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    03 · Risk Assessment
                </div>
                <div class="sidebar-card-caption">
                    Security risk evaluation
                </div>
            </div>

            <div class="sidebar-card">
                <div class="sidebar-card-title">
                    04 · Decision Engine
                </div>
                <div class="sidebar-card-caption">
                    Policy-based decision
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="sidebar-footer">
                <div class="sidebar-footer-label">
                    Built by
                </div>
                <div class="sidebar-footer-name">
                    Pritish Ganguly
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_email_information(
    email_data: dict,
) -> None:

    render_section(
        "Email Intelligence",
        "Parsed message metadata and message context.",
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    values = [
        (
            "Sender",
            email_data.get(
                "sender",
                "",
            ) or "Unknown",
        ),
        (
            "Receiver",
            email_data.get(
                "receiver",
                "",
            ) or "Unknown",
        ),
        (
            "Subject",
            email_data.get(
                "subject",
                "",
            ) or "No subject",
        ),
        (
            "Date",
            email_data.get(
                "date",
                "",
            ) or "Unknown",
        ),
    ]

    for column, (
        label,
        content,
    ) in zip(
        (
            col1,
            col2,
            col3,
            col4,
        ),
        values,
    ):

        with column:

            st.markdown(
                f"""
                <div class="info-panel">
                    <div class="info-label">
                        {label}
                    </div>
                    <div class="info-value">
                        {content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    attachments = email_data.get(
        "attachments",
        [],
    )

    if attachments:

        st.write("")

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Attachments detected**"
        )

        for attachment in attachments:
            st.write(
                f"• {attachment}"
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


def render_ml_result(
    ml_result: dict,
) -> None:

    render_section(
        "Machine Learning Classification",
        "Frozen production classifier inference.",
    )

    prediction = str(
        ml_result.get(
            "label",
            ml_result.get(
                "prediction",
                "UNKNOWN",
            ),
        )
    ).upper()

    is_spam = bool(
        ml_result.get(
            "is_spam",
            ml_result.get(
                "spam",
                prediction == "SPAM",
            ),
        )
    )

    score = ml_result.get(
        "decision_score",
        ml_result.get(
            "score",
            "—",
        ),
    )

    threshold = ml_result.get(
        "threshold",
        DECISION_THRESHOLD,
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:
        st.metric(
            "Prediction",
            prediction,
        )

    with col2:

        if isinstance(
            score,
            (int, float),
        ):
            score_text = f"{float(score):.4f}"
        else:
            score_text = str(score)

        st.metric(
            "Decision Score",
            score_text,
        )

    with col3:

        if isinstance(
            threshold,
            (int, float),
        ):
            threshold_text = f"{float(threshold):.4f}"
        else:
            threshold_text = str(threshold)

        st.metric(
            "Threshold",
            threshold_text,
        )

    if is_spam:
        st.error(
            "Machine-learning classification: SPAM"
        )
    else:
        st.success(
            "Machine-learning classification: HAM / NON-SPAM"
        )


def render_security_analysis(
    security: dict,
) -> None:

    render_section(
        "Security Signal Analysis",
        "Content-level security indicators extracted from the message.",
    )

    indicators = [
        (
            "Subject Length",
            security.get(
                "subject_length",
                0,
            ),
        ),
        (
            "Body Length",
            security.get(
                "body_length",
                0,
            ),
        ),
        (
            "Word Count",
            security.get(
                "word_count",
                0,
            ),
        ),
        (
            "URL Count",
            security.get(
                "url_count",
                security.get(
                    "url_count_total",
                    0,
                ),
            ),
        ),
        (
            "IP-Based URLs",
            security.get(
                "ip_url_count",
                security.get(
                    "ip_based_url_count",
                    0,
                ),
            ),
        ),
        (
            "Attachments",
            security.get(
                "attachment_count",
                security.get(
                    "attachments_count",
                    0,
                ),
            ),
        ),
        (
            "Suspicious Keywords",
            security.get(
                "suspicious_keyword_count",
                security.get(
                    "suspicious_keywords",
                    0,
                ),
            ),
        ),
        (
            "Exclamation Marks",
            security.get(
                "exclamation_count",
                0,
            ),
        ),
        (
            "HTML Present",
            "YES"
            if security.get(
                "html_present",
                False,
            )
            else "NO",
        ),
        (
            "Script Present",
            "YES"
            if security.get(
                "script_present",
                False,
            )
            else "NO",
        ),
        (
            "Uppercase Ratio",
            f"{float(
                security.get(
                    "uppercase_ratio",
                    0.0,
                )
            ):.4f}",
        ),
    ]

    columns = st.columns(
        4
    )

    for index, (
        label,
        content,
    ) in enumerate(
        indicators
    ):

        with columns[
            index % 4
        ]:

            st.metric(
                label,
                str(content),
            )


def render_risk_assessment(
    risk: dict,
) -> None:

    render_section(
        "Risk Assessment",
        "Aggregated security risk derived from classification and security signals.",
    )

    risk_score = risk.get(
        "risk_score",
        0,
    )

    risk_level = str(
        risk.get(
            "risk_level",
            "UNKNOWN",
        )
    ).upper()

    recommended_action = str(
        risk.get(
            "recommended_action",
            "No recommendation available.",
        )
    )

    try:
        numeric_score = float(
            risk_score
        )
    except (
        TypeError,
        ValueError,
    ):
        numeric_score = 0.0

    numeric_score = max(
        0.0,
        min(
            100.0,
            numeric_score,
        ),
    )

    col1, col2 = st.columns(
        2
    )

    with col1:
        st.metric(
            "Risk Score",
            f"{numeric_score:.0f}/100",
        )

    with col2:
        st.metric(
            "Risk Level",
            risk_level,
        )

    st.progress(
        numeric_score / 100
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Recommended Action**"
    )

    st.info(
        recommended_action
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    reasons = risk.get(
        "reasons",
        [],
    )

    st.markdown(
        "#### Security Reasons"
    )

    if isinstance(
        reasons,
        (list, tuple),
    ) and reasons:

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

    else:

        st.caption(
            "No elevated security indicators were reported."
        )


def render_decision(
    decision: dict,
    prediction: str,
    action: str,
    confidence: str,
    risk_score: Any,
    risk_level: str,
) -> None:

    render_section(
        "Decision Engine V3.1",
        "Final policy decision generated from the complete security pipeline.",
    )

    classification = str(
        decision.get(
            "classification",
            prediction,
        )
    ).upper()

    final_action = str(
        decision.get(
            "action",
            action,
        )
    ).upper()

    final_confidence = str(
        decision.get(
            "confidence",
            confidence,
        )
    ).upper()

    final_risk_score = decision.get(
        "risk_score",
        risk_score,
    )

    final_risk_level = str(
        decision.get(
            "risk_level",
            risk_level,
        )
    ).upper()

    reason = str(
        decision.get(
            "reason",
            "",
        )
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:
        st.metric(
            "Classification",
            classification,
        )

    with col2:
        st.metric(
            "Action",
            final_action,
        )

    with col3:
        st.metric(
            "Confidence",
            final_confidence,
        )

    col4, col5 = st.columns(
        2
    )

    with col4:
        st.metric(
            "Risk Level",
            final_risk_level,
        )

    with col5:
        st.metric(
            "Risk Score",
            f"{final_risk_score}/100",
        )

    if reason:

        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        st.markdown(
            "**Decision Reason**"
        )

        st.info(
            reason
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


def render_result(
    result: dict,
) -> None:

    result = normalize_result(
        result
    )

    email_data = result[
        "email"
    ]

    ml_result = result[
        "ml_result"
    ]

    security = result[
        "security_analysis"
    ]

    risk = result[
        "risk_assessment"
    ]

    decision = result[
        "decision"
    ]

    render_email_information(
        email_data
    )

    render_ml_result(
        ml_result
    )

    render_security_analysis(
        security
    )

    render_risk_assessment(
        risk
    )

    prediction = str(
        ml_result.get(
            "label",
            ml_result.get(
                "prediction",
                "UNKNOWN",
            ),
        )
    ).upper()

    action = str(
        risk.get(
            "recommended_action",
            "REVIEW",
        )
    )

    confidence = str(
        decision.get(
            "confidence",
            "UNKNOWN",
        )
    )

    risk_score = risk.get(
        "risk_score",
        0,
    )

    risk_level = str(
        risk.get(
            "risk_level",
            "UNKNOWN",
        )
    ).upper()

    render_decision(
        decision=decision,
        prediction=prediction,
        action=action,
        confidence=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
    )


def main() -> None:

    render_sidebar()

    render_hero()

    render_status_cards()

    st.write("")

    render_section(
        "Email Analysis Console",
        "Upload a raw .eml message or paste email content directly for analysis.",
    )

    try:

        scanner = load_scanner()
        predictor = load_predictor()

    except Exception as exc:

        st.error(
            "Unable to initialize the Email Security AI engine."
        )

        with st.expander(
            "Technical details"
        ):

            st.exception(
                exc
            )

        st.stop()

    upload_tab, paste_tab = st.tabs(
        [
            "📎  UPLOAD .EML",
            "✍️  PASTE EMAIL",
        ]
    )

    with upload_tab:

        uploaded = st.file_uploader(
            "Choose an email file",
            type=["eml"],
            accept_multiple_files=False,
            help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB",
        )

        if uploaded is not None:

            st.caption(
                f"Selected: {uploaded.name} • "
                f"{len(uploaded.getvalue()):,} bytes"
            )

            if st.button(
                "🔍  SCAN EMAIL",
                type="primary",
                use_container_width=True,
                key="scan_uploaded_email",
            ):

                with st.spinner(
                    "Running ML classification, security analysis, risk assessment and Decision Engine V3.1..."
                ):

                    try:

                        result = scan_uploaded(
                            scanner,
                            uploaded,
                        )

                    except Exception as exc:

                        st.error(
                            "Email analysis failed."
                        )

                        with st.expander(
                            "Technical details"
                        ):

                            st.exception(
                                exc
                            )

                    else:

                        st.success(
                            "✓ Security analysis completed successfully."
                        )

                        render_result(
                            result
                        )

    with paste_tab:

        email_text = st.text_area(
            "Email content",
            height=330,
            placeholder=(
                "From: sender@example.com\n"
                "To: recipient@example.com\n"
                "Subject: Example subject\n\n"
                "Paste the email body here..."
            ),
            key="email_text_input",
        )

        if st.button(
            "🔍  ANALYZE EMAIL",
            type="primary",
            use_container_width=True,
            key="analyze_email_text",
        ):

            if not email_text.strip():

                st.warning(
                    "Please paste an email before starting the analysis."
                )

            else:

                with st.spinner(
                    "Running ML classification, security analysis, risk assessment and Decision Engine V3.1..."
                ):

                    try:

                        result = scan_text(
                            predictor,
                            email_text,
                        )

                    except Exception as exc:

                        st.error(
                            "Email analysis failed."
                        )

                        with st.expander(
                            "Technical details"
                        ):

                            st.exception(
                                exc
                            )

                    else:

                        st.success(
                            "✓ Security analysis completed successfully."
                        )

                        render_result(
                            result
                        )

    st.divider()

    st.caption(
        "Email Security AI  •  Production Security Analysis  •  "
        "Decision Engine V3.1  •  Version 2.0"
    )

    st.caption(
        "Built by Pritish Ganguly"
    )


if __name__ == "__main__":
    main()