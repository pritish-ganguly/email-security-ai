from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.models.predict import EmailPredictor
from src.scanner.email_scanner import EmailScanner
from src.security.risk_engine import assess_email_risk
from src.security.security_analyzer import analyze_email

st.set_page_config(
    page_title="Email Security AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Email Security AI"
APP_VERSION = "1.1"
PRODUCTION_MODEL = "Linear SVM"
DECISION_THRESHOLD = -0.0975
MAX_FILE_SIZE_MB = 10

st.markdown(
    """
<style>
:root{
    --bg:#07111f;
    --panel:#0d1929;
    --panel2:#101f32;
    --border:rgba(148,163,184,.14);
    --text:#f8fafc;
    --muted:#8da0b8;
    --blue:#38bdf8;
    --green:#34d399;
    --red:#fb7185;
    --amber:#fbbf24;
}
.stApp{
    background:
        radial-gradient(circle at 10% 0%,rgba(56,189,248,.08),transparent 25%),
        radial-gradient(circle at 90% 5%,rgba(52,211,153,.06),transparent 25%),
        var(--bg);
    color:var(--text);
}
[data-testid="stHeader"]{
    background:transparent;
}
[data-testid="stAppViewContainer"]>.main{
    padding-top:0;
}
.main .block-container{
    max-width:1500px;
    padding:1.5rem 2rem 4rem;
}
section[data-testid="stSidebar"]{
    background:#081321;
    border-right:1px solid var(--border);
}
.sidebar-brand{
    padding:4px 2px 24px;
}
.sidebar-logo{
    font-size:34px;
    line-height:1;
    margin-bottom:10px;
}
.sidebar-title{
    color:var(--text);
    font-size:20px;
    font-weight:800;
}
.sidebar-subtitle{
    color:#60738d;
    font-size:12px;
    margin-top:4px;
}
.side-section{
    color:#60738d;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.4px;
    text-transform:uppercase;
    margin:24px 0 9px;
}
.side-card{
    background:rgba(255,255,255,.025);
    border:1px solid var(--border);
    border-radius:12px;
    padding:12px;
    margin-bottom:9px;
}
.side-label{
    color:#6f839d;
    font-size:10px;
    text-transform:uppercase;
    letter-spacing:.8px;
}
.side-value{
    color:#e8f0fa;
    font-size:14px;
    font-weight:700;
    margin-top:4px;
}
.side-small{
    color:#70839b;
    font-size:11px;
    margin-top:3px;
}
.online{
    color:var(--green);
}
.dot{
    display:inline-block;
    width:7px;
    height:7px;
    border-radius:50%;
    background:var(--green);
    box-shadow:0 0 10px rgba(52,211,153,.8);
    margin-right:6px;
}
.hero{
    background:
        linear-gradient(135deg,rgba(15,31,51,.98),rgba(9,22,38,.98));
    border:1px solid var(--border);
    border-radius:22px;
    padding:30px 34px;
    margin-bottom:20px;
    box-shadow:0 20px 60px rgba(0,0,0,.2);
}
.hero-row{
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:20px;
}
.hero-title{
    color:#f8fafc;
    font-size:38px;
    font-weight:850;
    letter-spacing:-1.2px;
    line-height:1.1;
}
.hero-subtitle{
    color:#91a3ba;
    font-size:15px;
    margin-top:9px;
    max-width:720px;
    line-height:1.6;
}
.hero-meta{
    color:#627792;
    font-size:11px;
    margin-top:15px;
}
.status{
    display:inline-flex;
    align-items:center;
    padding:8px 13px;
    border-radius:999px;
    color:#4ade80;
    background:rgba(52,211,153,.08);
    border:1px solid rgba(52,211,153,.22);
    font-size:11px;
    font-weight:800;
    letter-spacing:.5px;
    white-space:nowrap;
}
.metric-grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin-bottom:20px;
}
.metric{
    background:var(--panel);
    border:1px solid var(--border);
    border-radius:16px;
    padding:18px;
}
.metric-label{
    color:#70839b;
    font-size:10px;
    text-transform:uppercase;
    letter-spacing:1px;
    font-weight:800;
}
.metric-value{
    color:#f1f5f9;
    font-size:19px;
    font-weight:800;
    margin-top:7px;
}
.metric-sub{
    color:#627792;
    font-size:11px;
    margin-top:4px;
}
.panel{
    background:rgba(13,25,41,.9);
    border:1px solid var(--border);
    border-radius:18px;
    padding:20px;
    margin-bottom:16px;
}
.panel-title{
    color:#f1f5f9;
    font-size:17px;
    font-weight:800;
}
.panel-subtitle{
    color:#6f839d;
    font-size:12px;
    margin-top:4px;
}
.result{
    border-radius:18px;
    padding:24px;
    border:1px solid var(--border);
    background:var(--panel);
}
.result-spam{
    border-color:rgba(251,113,133,.35);
    background:linear-gradient(135deg,rgba(127,29,55,.22),rgba(13,25,41,.95));
}
.result-ham{
    border-color:rgba(52,211,153,.3);
    background:linear-gradient(135deg,rgba(6,78,59,.18),rgba(13,25,41,.95));
}
.result-label{
    color:#71849c;
    font-size:10px;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:1px;
}
.result-value{
    font-size:30px;
    font-weight:850;
    margin-top:4px;
}
.risk-low{
    color:var(--green);
}
.risk-medium{
    color:var(--amber);
}
.risk-high,.risk-critical{
    color:var(--red);
}
.info-card{
    background:rgba(255,255,255,.025);
    border:1px solid var(--border);
    border-radius:13px;
    padding:14px;
    min-height:72px;
}
.info-label{
    color:#637791;
    font-size:10px;
    text-transform:uppercase;
    letter-spacing:.8px;
}
.info-value{
    color:#e5edf7;
    font-size:13px;
    margin-top:5px;
    word-break:break-word;
}
.reason{
    background:rgba(255,255,255,.025);
    border-left:3px solid var(--blue);
    padding:10px 13px;
    margin:7px 0;
    border-radius:0 8px 8px 0;
    color:#c8d4e2;
    font-size:12px;
}
.footer{
    text-align:center;
    color:#4f627b;
    font-size:11px;
    padding:30px 0 10px;
}
div[data-testid="stFileUploader"]{
    background:rgba(255,255,255,.018);
    border:1px dashed rgba(148,163,184,.25);
    border-radius:14px;
    padding:6px;
}
textarea{
    border-radius:12px!important;
}
@media(max-width:900px){
    .metric-grid{grid-template-columns:repeat(2,1fr)}
    .hero-row{flex-direction:column}
    .hero-title{font-size:30px}
}
@media(max-width:600px){
    .metric-grid{grid-template-columns:1fr}
    .main .block-container{padding:1rem}
}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_scanner() -> EmailScanner:
    return EmailScanner()

@st.cache_resource
def load_predictor() -> EmailPredictor:
    return EmailPredictor()

def value(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def indicator(obj: Any, name: str, default: Any = 0) -> Any:
    return value(obj, name, default)

def build_text(subject: str, body: str, html_body: str = "") -> str:
    return f"Subject: {subject}\n\n{body}\n\n{html_body}"

def scan_uploaded(scanner: EmailScanner, uploaded_file: Any):
    data = uploaded_file.getvalue()
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(
            f"File exceeds the {MAX_FILE_SIZE_MB} MB upload limit."
        )
    suffix = Path(uploaded_file.name).suffix or ".eml"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        temp_path = Path(f.name)
    try:
        return scanner.scan(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

def scan_text(predictor: EmailPredictor, raw_text: str):
    text = raw_text.strip()
    if not text:
        raise ValueError("Paste an email before scanning.")
    ml_result = predictor.predict(text)
    security = analyze_email(
        subject="",
        body=text,
        html_body="",
        attachments=[],
    )
    risk = assess_email_risk(
        ml_result=ml_result,
        security_analysis=security,
    )
    return {
        "file_path": None,
        "email_data": {
            "sender": "",
            "receiver": "",
            "subject": "",
            "date": "",
            "attachments": [],
            "body": text,
        },
        "ml_result": ml_result,
        "security_analysis": security,
        "risk_assessment": risk,
    }

def result_value(result: Any, name: str, default: Any = None) -> Any:
    return value(result, name, default)

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">🛡️</div>
                <div class="sidebar-title">Email Security AI</div>
                <div class="sidebar-subtitle">Security Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-section">Engine</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-card"><div class="side-value online"><span class="dot"></span>ML ENGINE ONLINE</div><div class="side-small">Inference service ready</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-section">Production Model</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="side-card"><div class="side-value">{PRODUCTION_MODEL}</div><div class="side-small">Decision threshold: {DECISION_THRESHOLD}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-section">Evaluated Algorithms</div>', unsafe_allow_html=True)
        for model in ["Naive Bayes", "Logistic Regression", "Linear SVM"]:
            st.markdown(
                f'<div class="side-card"><div class="side-value">{model}</div><div class="side-small">Supervised classifier</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="side-section">NLP</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="side-card"><div class="side-value">Word TF-IDF</div><div class="side-small">Text representation</div></div>'
            '<div class="side-card"><div class="side-value">Character TF-IDF</div><div class="side-small">Character representation</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-section">Security Layers</div>', unsafe_allow_html=True)
        for layer in ["ML Classification", "Security Indicators", "Risk Engine"]:
            st.markdown(
                f'<div class="side-card"><div class="side-value">✓ {layer}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div class="side-section">Version</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="side-card"><div class="side-value">v{APP_VERSION}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="footer">Built by Pritish Ganguly</div>',
            unsafe_allow_html=True,
        )

def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-row">
                <div>
                    <div class="hero-title">🛡️ Email Security AI</div>
                    <div class="hero-subtitle">Intelligent email threat detection, machine-learning classification and security risk assessment.</div>
                    <div class="hero-meta">Frozen production inference • No model retraining • Security analysis enabled</div>
                </div>
                <div class="status"><span class="dot"></span>ML ENGINE ONLINE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric">
                <div class="metric-label">Engine</div>
                <div class="metric-value">ONLINE</div>
                <div class="metric-sub">Inference ready</div>
            </div>
            <div class="metric">
                <div class="metric-label">Production Model</div>
                <div class="metric-value">{PRODUCTION_MODEL}</div>
                <div class="metric-sub">Frozen classifier</div>
            </div>
            <div class="metric">
                <div class="metric-label">NLP Features</div>
                <div class="metric-value">WORD + CHAR</div>
                <div class="metric-sub">TF-IDF representation</div>
            </div>
            <div class="metric">
                <div class="metric-label">Threshold</div>
                <div class="metric-value">{DECISION_THRESHOLD}</div>
                <div class="metric-sub">Frozen operating point</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_email_information(result: Any) -> None:
    data = result_value(result, "email_data", {})
    sender = value(data, "sender", "Not available")
    receiver = value(data, "receiver", "Not available")
    subject = value(data, "subject", "Not available")
    date = value(data, "date", "Not available")
    attachments = value(data, "attachments", [])
    st.markdown(
        '<div class="panel"><div class="panel-title">Email Information</div><div class="panel-subtitle">Parsed message metadata</div></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="info-card"><div class="info-label">Sender</div><div class="info-value">{sender}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="info-card"><div class="info-label">Receiver</div><div class="info-value">{receiver}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="info-card"><div class="info-label">Date</div><div class="info-value">{date}</div></div>',
            unsafe_allow_html=True,
        )
    c4, c5 = st.columns(2)
    with c4:
        st.markdown(
            f'<div class="info-card"><div class="info-label">Subject</div><div class="info-value">{subject}</div></div>',
            unsafe_allow_html=True,
        )
    with c5:
        attachment_text = ", ".join(map(str, attachments)) if attachments else "None"
        st.markdown(
            f'<div class="info-card"><div class="info-label">Attachments</div><div class="info-value">{attachment_text}</div></div>',
            unsafe_allow_html=True,
        )

def render_ml(result: Any) -> None:
    ml = result_value(result, "ml_result", {})
    label = value(ml, "label", "unknown")
    spam = value(ml, "spam", False)
    score = float(value(ml, "decision_score", 0.0))
    threshold = float(value(ml, "threshold", DECISION_THRESHOLD))
    css = "result-spam" if spam else "result-ham"
    color = "risk-high" if spam else "risk-low"
    text = "SPAM" if spam else "HAM"
    st.markdown(
        f"""
        <div class="result {css}">
            <div class="result-label">Machine Learning Classification</div>
            <div class="result-value {color}">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Prediction", str(label).upper())
    with c2:
        st.metric("Decision Score", f"{score:.4f}")
    with c3:
        st.metric("Threshold", f"{threshold:.4f}")

def render_security(result: Any) -> None:
    analysis = result_value(result, "security_analysis", None)
    st.markdown(
        '<div class="panel"><div class="panel-title">Security Indicators</div><div class="panel-subtitle">Content-level security signals extracted from the message</div></div>',
        unsafe_allow_html=True,
    )
    names = [
        ("Subject length", "subject_length"),
        ("Body length", "body_length"),
        ("Word count", "word_count"),
        ("URL count", "url_count"),
        ("IP-based URLs", "ip_url_count"),
        ("Suspicious keywords", "suspicious_keyword_count"),
        ("HTML present", "html_present"),
        ("Script present", "script_present"),
        ("Attachments", "attachment_count"),
        ("Exclamation marks", "exclamation_count"),
        ("Uppercase ratio", "uppercase_ratio"),
    ]
    cols = st.columns(4)
    for index, (label, field) in enumerate(names):
        raw = indicator(analysis, field, 0)
        if isinstance(raw, float):
            display = f"{raw:.4f}"
        elif isinstance(raw, bool):
            display = "Yes" if raw else "No"
        else:
            display = str(raw)
        with cols[index % 4]:
            st.markdown(
                f'<div class="info-card" style="margin-bottom:12px"><div class="info-label">{label}</div><div class="info-value">{display}</div></div>',
                unsafe_allow_html=True,
            )

def render_risk(result: Any) -> None:
    risk = result_value(result, "risk_assessment", None)
    score = int(value(risk, "risk_score", 0))
    level = str(value(risk, "risk_level", "UNKNOWN")).upper()
    action = str(value(risk, "recommended_action", "No recommendation available."))
    reasons = value(risk, "reasons", []) or []
    level_class = f"risk-{level.lower()}" if level.lower() in {"low", "medium", "high", "critical"} else ""
    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">Risk Assessment</div>
            <div class="result-value {level_class}">{level}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Risk Score", f"{score}/100")
    with c2:
        st.markdown(
            f'<div class="info-card"><div class="info-label">Recommended Action</div><div class="info-value">{action}</div></div>',
            unsafe_allow_html=True,
        )
    if reasons:
        st.markdown("#### Security Reasons")
        for reason in reasons:
            st.markdown(
                f'<div class="reason">{reason}</div>',
                unsafe_allow_html=True,
            )

def render_result(result: Any) -> None:
    render_email_information(result)
    render_ml(result)
    st.markdown("<br>", unsafe_allow_html=True)
    render_security(result)
    render_risk(result)

def main() -> None:
    render_sidebar()
    render_header()

    try:
        scanner = load_scanner()
        predictor = load_predictor()
    except Exception as exc:
        st.error("Unable to load the Email Security AI engine.")
        st.exception(exc)
        st.stop()

    st.markdown(
        '<div class="panel"><div class="panel-title">Analyze an Email</div><div class="panel-subtitle">Upload a raw .eml message or paste email content directly for analysis.</div></div>',
        unsafe_allow_html=True,
    )

    tab_upload, tab_text = st.tabs(["📎 Upload .EML", "✍️ Paste Email"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Choose an email file",
            type=["eml"],
            accept_multiple_files=False,
        )
        if uploaded:
            st.caption(
                f"{uploaded.name} • {len(uploaded.getvalue()):,} bytes"
            )
            if st.button(
                "🔍 Scan Uploaded Email",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analyzing email..."):
                    try:
                        result = scan_uploaded(scanner, uploaded)
                    except Exception as exc:
                        st.error("Email analysis failed.")
                        st.exception(exc)
                    else:
                        st.success("Analysis completed.")
                        render_result(result)

    with tab_text:
        email_text = st.text_area(
            "Paste email content",
            height=320,
            placeholder=(
                "Paste the email headers and body here...\n\n"
                "From: sender@example.com\n"
                "To: recipient@example.com\n"
                "Subject: Example\n\n"
                "Email content..."
            ),
        )
        if st.button(
            "🔍 Analyze Email Text",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner("Analyzing email text..."):
                try:
                    result = scan_text(predictor, email_text)
                except Exception as exc:
                    st.error("Email analysis failed.")
                    st.exception(exc)
                else:
                    st.success("Analysis completed.")
                    render_result(result)

    st.markdown(
        '<div class="footer">Email Security AI · Frozen production inference · Built by Pritish Ganguly</div>',
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
