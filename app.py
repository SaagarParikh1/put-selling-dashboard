import math
import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.analysis import analyze_stock, summarize_stock
from src.backtesting import (
    learning_signature,
    load_backtest_summary,
    load_learning_profile,
    run_automatic_learning_cycle,
)
from src.utils import (
    build_action_suggestion,
    build_avoid_reason,
    build_confidence_explanation,
    build_signal_takeaways,
    build_table_setup_note,
    colorize_signal_reason,
    group_signal_reasons,
    normalize_boolish,
)
from src.watchlist_manager import (
    load_watchlist,
    add_to_watchlist,
    remove_from_watchlist
)

st.set_page_config(
    page_title="Put Selling Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --bg-main: #0b1220;
        --bg-card: #111827;
        --bg-card-2: #0f172a;
        --bg-soft: #172033;
        --border: rgba(148, 163, 184, 0.18);
        --text-main: #f8fafc;
        --text-soft: #cbd5e1;
        --text-muted: #94a3b8;
        --green: #22c55e;
        --lime: #84cc16;
        --yellow: #facc15;
        --orange: #fb923c;
        --red: #ef4444;
        --rose-soft: #fca5a5;
        --blue: #60a5fa;
        --violet: #a78bfa;
        --cyan: #67e8f9;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(96,165,250,0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(167,139,250,0.08), transparent 22%),
            linear-gradient(180deg, #08101d 0%, #0b1220 38%, #0a1020 100%);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 3.25rem;
        padding-bottom: 2.5rem;
        max-width: 1460px;
    }

    .hero-title {
        font-size: 2.85rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        margin-top: 0.4rem;
        letter-spacing: -0.02em;
        color: #f8fbff;
    }

    .hero-subtitle {
        color: #b6c2d4;
        margin-bottom: 0.95rem;
        font-size: 1.02rem;
        line-height: 1.6;
    }

    .hero-note {
        color: #97a8bf;
        font-size: 0.9rem;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
    }

    .hero-shell {
        background:
            radial-gradient(circle at top left, rgba(96,165,250,0.14), transparent 32%),
            linear-gradient(135deg, rgba(15,23,42,0.98), rgba(17,24,39,0.96));
        border: 1px solid rgba(148,163,184,0.14);
        border-radius: 24px;
        padding: 1.35rem 1.4rem 1.15rem 1.4rem;
        box-shadow: 0 16px 40px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #93c5fd;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .summary-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.45rem 0.72rem;
        margin: 0.2rem 0.35rem 0.2rem 0;
        border-radius: 999px;
        border: 1px solid rgba(148,163,184,0.14);
        background: rgba(15,23,42,0.72);
        color: #dbe7f4;
        font-size: 0.88rem;
        font-weight: 700;
    }

    .subtle-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.82), rgba(17,24,39,0.78));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.9rem;
    }

    .empty-state {
        background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(17,24,39,0.90));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 22px;
        padding: 1.3rem 1.35rem;
        color: #d7e2ef;
        margin-top: 0.8rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.16);
    }

    .pill {
        display: inline-block;
        padding: 0.48rem 0.84rem;
        margin: 0.22rem 0.34rem 0.22rem 0;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(23,32,51,0.95));
        border: 1px solid rgba(148,163,184,0.15);
        font-size: 0.9rem;
        font-weight: 600;
        color: #e5edf7;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .section-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(17,24,39,0.96));
        border-radius: 18px;
        padding: 1rem 1.05rem;
        border: 1px solid rgba(148,163,184,0.14);
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }

    .section-header {
        margin-top: 0.35rem;
        margin-bottom: 0.8rem;
    }

    .section-eyebrow {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7dd3fc;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 0.15rem;
    }

    .section-subtitle {
        color: #9fb0c8;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .watchlist-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.98), rgba(17,24,39,0.98));
        border-radius: 18px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(148,163,184,0.14);
        margin-bottom: 1rem;
        box-shadow: 0 12px 32px rgba(0,0,0,0.16);
    }

    .sidebar-watchlist-shell {
        background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.98));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 0.9rem 0.9rem 0.8rem 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.14);
    }

    .sidebar-watchlist-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.15rem;
    }

    .sidebar-watchlist-subtitle {
        font-size: 0.8rem;
        color: #9fb0c8;
        margin-bottom: 0.2rem;
    }

    .sidebar-kicker {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7dd3fc;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }

    .sidebar-copy {
        font-size: 0.88rem;
        color: #9fb0c8;
        line-height: 1.55;
        margin-bottom: 0.2rem;
    }

    .sidebar-tip {
        font-size: 0.84rem;
        color: #cbd5e1;
        line-height: 1.5;
        margin-bottom: 0.45rem;
    }

    .soft-danger-text {
        color: #fca5a5;
        font-size: 0.9rem;
        margin-top: -0.25rem;
        margin-bottom: 0.45rem;
    }

    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #f8fafc;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,24,39,0.95));
        border: 1px solid rgba(148,163,184,0.12);
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.14);
    }

    div[data-testid="stMetricLabel"] {
        color: #9fb0c8 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(148,163,184,0.10);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(148,163,184,0.12) !important;
        background: linear-gradient(135deg, rgba(15,23,42,0.86), rgba(17,24,39,0.82));
        box-shadow: 0 10px 26px rgba(0,0,0,0.12);
    }

    div[data-testid="stTabs"] button {
        border-radius: 12px 12px 0 0 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(8,16,29,0.98), rgba(10,16,32,0.98));
        border-right: 1px solid rgba(148,163,184,0.08);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.3rem;
        padding-bottom: 1.2rem;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    div[data-baseweb="input"] input {
        background: rgba(15,23,42,0.75) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 700;
        transition: all 0.15s ease;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #0ea5e9) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 12px 26px rgba(37,99,235,0.28) !important;
        min-height: 2.9rem !important;
        font-size: 0.98rem !important;
        font-weight: 800 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #0284c7) !important;
    }

    section[data-testid="stSidebar"] form[data-testid="stForm"] button {
        background: linear-gradient(135deg, #16a34a, #22c55e) !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] form[data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #15803d, #16a34a) !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.15rem 0.35rem !important;
        min-height: 1.6rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: rgba(239, 68, 68, 0.08) !important;
        color: #f87171 !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        border-radius: 12px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
        border: none !important;
        box-shadow: 0 10px 26px rgba(59,130,246,0.24) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8, #6d28d9) !important;
    }

    div[data-testid="stButton"]:not(section[data-testid="stSidebar"] div[data-testid="stButton"]) button {
        border-radius: 12px !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 14px;
    }

    [data-testid="stPopover"] button {
        border-radius: 12px !important;
        border: 1px solid rgba(148,163,184,0.16) !important;
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(17,24,39,0.96)) !important;
        color: #e5edf7 !important;
        font-weight: 700 !important;
    }

    [data-testid="stCaptionContainer"] {
        color: #9fb0c8 !important;
    }

    .table-context {
        color: #9fb0c8;
        font-size: 0.9rem;
        margin-top: -0.3rem;
        margin-bottom: 0.75rem;
        line-height: 1.5;
    }

    .deep-dive-grid-gap {
        height: 0.7rem;
    }

    .analysis-note {
        color: #9fb0c8;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: 0.2rem;
        margin-bottom: 0.9rem;
    }

    .status-box {
        background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(19,30,50,0.92));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 0.9rem 0.95rem;
        min-height: 118px;
        margin-bottom: 0.25rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }

    .status-label {
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #93a7c2;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .status-value {
        font-size: 1.05rem;
        line-height: 1.4;
        font-weight: 800;
    }

    .metric-tile {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,24,39,0.93));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        min-height: 124px;
        margin-bottom: 0.25rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.12);
    }

    .metric-title {
        color: #93a7c2;
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.42rem;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.18rem;
        font-weight: 800;
        line-height: 1.35;
    }

    .metric-subtitle {
        color: #9fb0c8;
        font-size: 0.84rem;
        line-height: 1.45;
        margin-top: 0.42rem;
    }

    .queue-shell {
        background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(17,24,39,0.92));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.14);
        margin-bottom: 1rem;
    }

    .sidebar-status-shell {
        background: linear-gradient(135deg, rgba(13,20,36,0.98), rgba(18,28,46,0.98));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.85rem;
    }

    .sidebar-list-row {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.08);
        border-radius: 12px;
        padding: 0.5rem 0.7rem;
        min-height: 42px;
        display: flex;
        align-items: center;
        color: #d9e4f2;
        font-size: 0.92rem;
        font-weight: 700;
    }

    .sidebar-list-row-selected {
        background: linear-gradient(135deg, rgba(29,78,216,0.20), rgba(14,165,233,0.12));
        border-color: rgba(96,165,250,0.28);
        color: #f8fbff;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .decision-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(17,24,39,0.92));
        border-radius: 20px;
        border: 1px solid rgba(148,163,184,0.12);
        padding: 1rem 1.05rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.14);
    }

    .decision-label {
        font-size: 0.78rem;
        color: #93a7c2;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .decision-title {
        font-size: 1.06rem;
        font-weight: 800;
        line-height: 1.4;
    }

    .decision-body {
        margin-top: 0.45rem;
        color: #d6e1ef;
        line-height: 1.65;
        font-size: 0.94rem;
    }

    .insight-card {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.6rem;
        color: #dbe5f3;
        line-height: 1.6;
    }

    .reason-group-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(17,24,39,0.88));
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin-bottom: 0.75rem;
    }

    .reason-group-title {
        font-size: 0.86rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #d8e4f4;
        font-weight: 800;
        margin-bottom: 0.55rem;
    }

    .reason-bullet {
        color: #d5dfed;
        line-height: 1.6;
        margin-bottom: 0.45rem;
        font-size: 0.93rem;
    }

    .reason-bullet:last-child {
        margin-bottom: 0;
    }

    .guide-panel {
        background: linear-gradient(135deg, rgba(15,23,42,0.92), rgba(17,24,39,0.88));
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 1rem;
    }

    .guide-block-title {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .guide-block-copy {
        color: #cbd5e1;
        font-size: 0.93rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d"],
}


def is_valid_ticker(ticker: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z.\-]{1,10}", ticker.strip()))


def render_pills(items: list[str]) -> str:
    return "".join([f"<span class='pill'>{item}</span>" for item in items])


def label_color(label: str) -> str:
    colors = {
        "High Probability Put Sell": "#22c55e",
        "Put Sell Candidate": "#84cc16",
        "Stalk / Watchlist": "#60a5fa",
        "Neutral / Wait": "#facc15",
        "Downtrend Risk": "#fb923c",
        "Breakdown Risk": "#ef4444",
    }
    return colors.get(label, "#cbd5e1")


LABEL_PRIORITY = {
    "High Probability Put Sell": 0,
    "Put Sell Candidate": 1,
    "Stalk / Watchlist": 2,
    "Neutral / Wait": 3,
    "Downtrend Risk": 4,
    "Breakdown Risk": 5,
}


def entry_status_color(status: str) -> str:
    status = (status or "").strip().lower()
    if "in entry zone" in status:
        return "#22c55e"
    if "watch for stabilization" in status:
        return "#60a5fa"
    if "wait for pullback" in status:
        return "#facc15"
    if "support under pressure" in status:
        return "#fb923c"
    if "below support" in status or "caution" in status:
        return "#ef4444"
    return "#cbd5e1"


def market_regime_color(regime: str) -> str:
    if regime == "Bull":
        return "#22c55e"
    if regime == "Neutral":
        return "#facc15"
    if regime == "Bear":
        return "#ef4444"
    return "#cbd5e1"


def support_strength_label(value):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= 11:
        return "Exceptional"
    if value >= 9:
        return "Strong"
    if value >= 7:
        return "Solid"
    if value >= 6:
        return "Average"
    if value >= 4:
        return "Fragile"
    return "Very Weak"


def support_strength_color(label: str) -> str:
    colors = {
        "Exceptional": "#22c55e",
        "Strong": "#22c55e",
        "Solid": "#84cc16",
        "Average": "#facc15",
        "Fragile": "#fb923c",
        "Very Weak": "#ef4444",
    }
    return colors.get(label, "#cbd5e1")


def state_color(label: str) -> str:
    lookup = {
        "Pass": "#22c55e",
        "OK": "#22c55e",
        "Controlled": "#22c55e",
        "Caution": "#facc15",
        "Watch for stabilization": "#60a5fa",
        "Wait for pullback": "#facc15",
        "Support under pressure": "#fb923c",
        "Elevated": "#fb923c",
        "Thin": "#ef4444",
        "Below support / caution": "#ef4444",
    }
    return lookup.get(label, "#cbd5e1")


def render_status_box(title: str, value: str, color: str) -> str:
    return f"""
    <div class='status-box'>
        <div class='status-label'>{title}</div>
        <div class='status-value' style='color:{color};'>{value}</div>
    </div>
    """


def render_section_header(title: str, subtitle: str, eyebrow: str | None = None) -> None:
    eyebrow_html = f"<div class='section-eyebrow'>{eyebrow}</div>" if eyebrow else ""
    st.markdown(
        f"""
        <div class='section-header'>
            {eyebrow_html}
            <div class='section-title'>{title}</div>
            <div class='section-subtitle'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_tile(title: str, value: str, accent: str = "#f8fafc", subtitle: str | None = None) -> str:
    subtitle_html = f"<div class='metric-subtitle'>{subtitle}</div>" if subtitle else ""
    return f"""
    <div class='metric-tile'>
        <div class='metric-title'>{title}</div>
        <div class='metric-value' style='color:{accent};'>{value}</div>
        {subtitle_html}
    </div>
    """


DISPLAY_COLUMN_LABELS = {
    "symbol": "Symbol",
    "price": "Price",
    "primary_support": "Primary Support",
    "secondary_support": "Secondary Support",
    "recommended_entry": "Preferred Entry",
    "entry_status": "Entry Status",
    "support_strength_label": "Support",
    "bounce_signal": "Bounce Signal",
    "label": "Signal",
    "score": "Score",
    "confidence": "Conviction",
    "quality_score": "Quality",
    "entry_score": "Entry",
    "risk_score": "Risk",
    "setup_note": "Setup Note",
    "avoid_reason": "Why Avoid",
}


def sort_signal_df(df: pd.DataFrame, ascending_score: bool = False) -> pd.DataFrame:
    if df.empty or "label" not in df.columns:
        return df

    sorted_df = df.copy()
    sorted_df["_label_priority"] = sorted_df["label"].map(LABEL_PRIORITY).fillna(99)
    sorted_df = sorted_df.sort_values(
        ["_label_priority", "score", "confidence"],
        ascending=[True, ascending_score, ascending_score],
    )
    return sorted_df.drop(columns="_label_priority")


def prepare_display_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available_cols = [col for col in columns if col in df.columns]
    return df[available_cols].rename(columns=DISPLAY_COLUMN_LABELS)


def render_guide_content() -> None:
    render_section_header(
        "Scoring And Filter Guide",
        "These rules explain how the dashboard decides whether a setup is worth stalking, waiting on, or avoiding.",
        "Reference",
    )

    blocks = [
        (
            "How To Read The Signal Labels",
            """
            <div class='guide-block-copy'>
                The dashboard is now built around a seasoned put-seller idea:
                <b>good underlying I would not mind owning, near support, with risk still under control.</b><br><br>
                <b>High Probability Put Sell</b> means the stock, the support area, the bounce behavior, and the downside profile all line up unusually well.<br><br>
                <b>Put Sell Candidate</b> means the stock is near enough to support to be a legitimate put-selling idea, but it is still a notch below the cleanest trade-ready setups.<br><br>
                <b>Stalk / Watchlist</b> means the stock may be worth owning if assigned, but the timing is not ready enough yet for a disciplined put sale.<br><br>
                <b>Neutral / Wait</b> means there may be something to like, but the timing or risk profile is not ready yet.<br><br>
                <b>Downtrend Risk</b> and <b>Breakdown Risk</b> mean put sellers should get more defensive because support is less dependable.
            </div>
            """,
        ),
        (
            "What The Scores Mean",
            """
            <div class='guide-block-copy'>
                <b>Quality</b> asks whether this is a stock you would still be comfortable owning if assigned.
                Higher scores come from stronger trend structure, healthier moving averages, better relative strength, contained volatility, healthier money flow, and cleaner market context.<br><br>
                <b>Entry</b> asks whether the current price is actually a good place to sell a put.
                Higher scores come from price being near support, near the preferred entry area, and showing signs of stabilizing or bouncing.<br><br>
                <b>Risk</b> asks how likely the setup is to fail from here.
                More controlled setups score closer to zero or slightly positive, while broken, unstable, or highly speculative setups push the score further negative.
            </div>
            """,
        ),
        (
            "What Usually Qualifies As A Candidate",
            """
            <div class='guide-block-copy'>
                A typical <b>Put Sell Candidate</b> needs enough quality to justify assignment, a reasonable support map, and risk that is not already deteriorating.
                It does <b>not</b> need a perfect entry today, but it should already be near a real support decision area.
                In practice, better candidates usually have <b>Quality around 5+</b>, <b>Entry around 0 or better</b>, <b>Risk around -4 or better</b>, and price that is already testing or reclaiming support in a constructive way.
                <br><br>
                <b>Stalk / Watchlist</b> is for names that may still be acceptable to own on assignment, but are too early, too extended above support, or still waiting on bounce confirmation.
                <br><br>
                <b>High Probability Put Sell</b> is stricter. It usually needs stronger quality, a support-based entry that is already firming up, and a clearer bounce or stabilization signal with fewer obvious risk flags.
            </div>
            """,
        ),
        (
            "Support, Entry, And Bounce Behavior",
            """
            <div class='guide-block-copy'>
                <b>Primary Support</b> is the nearest important area where buyers have a realistic chance to defend price.
                <b>Secondary Support</b> is the next layer below it if the first level fails.<br><br>
                <b>Preferred Entry</b> is the area where put selling usually becomes more attractive than chasing price higher.
                <b>Entry Status</b> tells you whether price is already there, almost there, or still too early.
                <b>Bounce Signal</b> helps separate a clean rebound from a weak test that has not proven itself yet.
            </div>
            """,
        ),
        (
            "Indicators The Dashboard Uses",
            """
            <div class='guide-block-copy'>
                <b>EMA 9 / EMA 21</b> help judge short-term momentum.
                <b>SMA 50 / SMA 200</b> help judge medium- and long-term trend quality.
                <b>RSI</b> helps identify healthier pullback zones.
                <b>MACD</b> and <b>ADX</b> help confirm momentum and trend strength.
                <b>CMF</b> and <b>RS vs SPY</b> help show whether participation and leadership are supportive.
                <br><br>
                This dashboard still scores the <b>underlying</b>, not the actual option contract, so delta, IV rank, strike selection, and premium quality still need a trader's judgment.
            </div>
            """,
        ),
        (
            "Auto Backtest And Learning",
            """
            <div class='guide-block-copy'>
                On each full analysis cycle, the dashboard can replay historical signals across the watchlist, evaluate how those setups behaved afterward, and save a small local learning profile.
                <br><br>
                This is <b>not</b> full machine-learning retraining. It is a rule-tuning loop that adjusts scoring thresholds modestly based on recent historical hit rate, drawdown behavior, support-hold performance, and whether trade-ready setups actually stabilized enough for a put seller.
            </div>
            """,
        ),
    ]

    for title, body in blocks:
        st.markdown(
            f"""
            <div class='guide-panel'>
                <div class='guide-block-title'>{title}</div>
                {body}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_auto_backtest_summary(backtest_summary: dict | None, learning_profile: dict | None) -> None:
    render_section_header(
        "Auto Backtest And Learning",
        "The dashboard replays historical signals, scores how they behaved afterward, and uses that summary to self-tune the rules over time.",
        "Research",
    )

    if not backtest_summary:
        st.info("No automatic backtest summary is available yet. Run analysis to generate one.")
        return

    generated_at = backtest_summary.get("generated_at")
    refreshed_text = "Unavailable"
    if generated_at:
        try:
            refreshed_text = pd.to_datetime(generated_at).strftime("%b %d, %Y %I:%M %p UTC")
        except Exception:
            refreshed_text = str(generated_at)

    candidate_like = backtest_summary.get("candidate_like") or {}
    method_notes = (backtest_summary.get("methodology") or {}).get("notes") or []
    learning_notes = (learning_profile or {}).get("notes") or []
    source_summary = (learning_profile or {}).get("source_summary") or {}

    st.markdown(
        f"""
        <div class='section-card'>
            <div style='display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:flex-start;'>
                <div>
                    <div class='section-eyebrow'>Automatic Loop</div>
                    <div style='color:#e5edf7; font-size:1rem; font-weight:800; margin-bottom:0.2rem;'>Backtest refreshed: {refreshed_text}</div>
                    <div style='color:#9fb0c8; line-height:1.6;'>
                        This learning loop uses an <b>underlying-behavior proxy</b>, not actual option premium P/L. It checks whether historical signals held support and avoided materially weak forward price behavior.
                        The trade-ready labels are judged more strictly than a simple "did the stock go up?" test.
                    </div>
                </div>
                <div style='display:flex; flex-wrap:wrap; gap:0.45rem;'>
                    <span class='summary-chip'>History Window: {(backtest_summary.get("methodology") or {}).get("history_period", "N/A")}</span>
                    <span class='summary-chip'>Signals Tested: {candidate_like.get("signal_count", 0)}</span>
                    <span class='summary-chip'>Watchlist Size: {backtest_summary.get("watchlist_size", 0)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("Trade-Ready Hit Rate", f"{candidate_like.get('success_rate', 0):.1f}%" if candidate_like.get("success_rate") is not None else "N/A")
    m2.metric("Avg 20D Return", f"{candidate_like.get('avg_20d_return_pct', 0):.2f}%" if candidate_like.get("avg_20d_return_pct") is not None else "N/A")
    m3.metric("Avg Max Drawdown", f"{candidate_like.get('avg_max_drawdown_pct', 0):.2f}%" if candidate_like.get("avg_max_drawdown_pct") is not None else "N/A")
    m4.metric("Trade-Ready Samples", f"{candidate_like.get('signal_count', 0)}")

    left, right = st.columns([1.25, 1], gap="large")

    with left:
        labels = backtest_summary.get("labels") or {}
        if labels:
            label_rows = []
            for label, stats in labels.items():
                label_rows.append(
                    {
                        "Signal": label,
                        "Samples": stats.get("signal_count"),
                        "Hit Rate %": stats.get("success_rate"),
                        "Support Hold %": stats.get("support_hold_rate"),
                        "Avg 10D %": stats.get("avg_10d_return_pct"),
                        "Avg 20D %": stats.get("avg_20d_return_pct"),
                        "Avg Drawdown %": stats.get("avg_max_drawdown_pct"),
                    }
                )
            label_df = pd.DataFrame(label_rows)
            if not label_df.empty:
                label_df["_order"] = label_df["Signal"].map(LABEL_PRIORITY).fillna(99)
                label_df = label_df.sort_values(["_order", "Samples"], ascending=[True, False]).drop(columns="_order")
            st.markdown("<div class='table-context'>This is the historical scorecard by signal label using the dashboard's underlying proxy backtest.</div>", unsafe_allow_html=True)
            st.dataframe(style_ranked_table(label_df), use_container_width=True, hide_index=True)
        else:
            st.info("No label-level backtest results are available yet.")

    with right:
        note_blocks = []
        for note in learning_notes[:4]:
            note_blocks.append(f"<div class='reason-bullet'>- {colorize_signal_reason(note)}</div>")
        if source_summary:
            note_blocks.append(
                f"<div class='reason-bullet'>- Recent candidate sample count: {source_summary.get('candidate_signal_count', 0)} | success rate: {source_summary.get('candidate_success_rate', 'N/A')}%</div>"
            )
            note_blocks.append(
                f"<div class='reason-bullet'>- Recent high-probability sample count: {source_summary.get('high_probability_signal_count', 0)} | success rate: {source_summary.get('high_probability_success_rate', 'N/A')}%</div>"
            )

        st.markdown(
            f"""
            <div class='reason-group-card'>
                <div class='reason-group-title'>What The System Learned</div>
                {''.join(note_blocks) if note_blocks else "<div class='reason-bullet'>- No rule adjustments were needed from the latest backtest cycle.</div>"}
            </div>
            """,
            unsafe_allow_html=True,
        )

        method_cards = "".join([f"<div class='reason-bullet'>- {note}</div>" for note in method_notes[:3]])
        st.markdown(
            f"""
            <div class='reason-group-card'>
                <div class='reason-group-title'>Method Notes</div>
                {method_cards}
            </div>
            """,
            unsafe_allow_html=True,
        )


def liquidity_label(value):
    normalized = normalize_boolish(value)
    if normalized is True:
        return "OK"
    if normalized is False:
        return "Thin"
    return "N/A"


def build_ranked_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()

    if "support_strength" in display_df.columns:
        display_df["support_strength_label"] = display_df["support_strength"].apply(support_strength_label)

    display_df["setup_note"] = display_df.apply(build_table_setup_note, axis=1)

    if "liquidity_ok" in display_df.columns:
        display_df["liquidity_label"] = display_df["liquidity_ok"].apply(liquidity_label)

    return display_df


def build_avoid_display_df(df: pd.DataFrame) -> pd.DataFrame:
    display_df = build_ranked_display_df(df)
    if "reasons" in display_df.columns:
        display_df["avoid_reason"] = display_df.apply(build_avoid_reason, axis=1)
    else:
        display_df["avoid_reason"] = "Risk profile is unfavorable for a fresh put-selling entry."
    return display_df


def build_summary_counts(df: pd.DataFrame):
    return {
        "High Probability Put Sell": int((df["label"] == "High Probability Put Sell").sum()),
        "Put Sell Candidate": int((df["label"] == "Put Sell Candidate").sum()),
        "Stalk / Watchlist": int((df["label"] == "Stalk / Watchlist").sum()),
        "Neutral / Wait": int((df["label"] == "Neutral / Wait").sum()),
        "Downtrend Risk": int((df["label"] == "Downtrend Risk").sum()),
        "Breakdown Risk": int((df["label"] == "Breakdown Risk").sum()),
    }


def pct_return(df: pd.DataFrame, periods: int):
    if len(df) <= periods:
        return "N/A"
    start_price = df["close"].iloc[-periods - 1]
    end_price = df["close"].iloc[-1]
    if start_price == 0:
        return "N/A"
    val = ((end_price - start_price) / start_price) * 100
    return f"{val:.2f}%"


def fmt_price(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:.2f}"


def get_default_3m_range(df: pd.DataFrame):
    if df.empty:
        return None, None
    end_date = pd.to_datetime(df["timestamp"].max())
    start_candidate = end_date - pd.Timedelta(days=90)
    start_date = max(pd.to_datetime(df["timestamp"].min()), start_candidate)
    return start_date, end_date


def get_visible_y_range(df: pd.DataFrame, start_date, end_date, columns: list[str], padding_pct: float = 0.06):
    if df.empty:
        return None

    visible = df.copy()
    if start_date is not None:
        visible = visible[visible["timestamp"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        visible = visible[visible["timestamp"] <= pd.to_datetime(end_date)]

    if visible.empty:
        visible = df

    values = []
    for col in columns:
        if col in visible.columns:
            series = pd.to_numeric(visible[col], errors="coerce").dropna()
            if not series.empty:
                values.extend(series.tolist())

    if not values:
        return None

    low = min(values)
    high = max(values)

    if math.isclose(low, high):
        pad = max(abs(low) * padding_pct, 1)
        return [low - pad, high + pad]

    pad = (high - low) * padding_pct
    return [low - pad, high + pad]


def _safe_num(x):
    return None if x is None or pd.isna(x) else float(x)


def style_ranked_table(df: pd.DataFrame):
    def style_support_strength(v):
        if pd.isna(v):
            return ""
        if str(v) == "Exceptional":
            return "color: #22c55e; font-weight: 800;"
        if str(v) == "Strong":
            return "color: #22c55e; font-weight: 700;"
        if str(v) == "Solid":
            return "color: #84cc16; font-weight: 700;"
        if str(v) == "Average":
            return "color: #facc15; font-weight: 700;"
        if str(v) == "Fragile":
            return "color: #fb923c; font-weight: 700;"
        if str(v) == "Very Weak":
            return "color: #ef4444; font-weight: 700;"
        if str(v) == "Moderate":
            return "color: #facc15; font-weight: 700;"
        if str(v) == "Weak":
            return "color: #ef4444; font-weight: 700;"
        return ""

    def style_liquidity(v):
        if pd.isna(v):
            return ""
        if str(v) == "OK":
            return "color: #22c55e; font-weight: 700;"
        if str(v) == "Thin":
            return "color: #ef4444; font-weight: 700;"
        return ""

    def style_regime(v):
        if pd.isna(v):
            return ""
        if str(v) == "Bull":
            return "color: #22c55e; font-weight: 700;"
        if str(v) == "Neutral":
            return "color: #facc15; font-weight: 700;"
        if str(v) == "Bear":
            return "color: #ef4444; font-weight: 700;"
        return ""

    def style_entry_status(v):
        if pd.isna(v):
            return ""
        v = str(v).lower()
        if "in entry zone" in v:
            return "color: #22c55e; font-weight: 700;"
        if "watch for stabilization" in v:
            return "color: #60a5fa; font-weight: 700;"
        if "wait for pullback" in v:
            return "color: #facc15; font-weight: 700;"
        if "support under pressure" in v:
            return "color: #fb923c; font-weight: 700;"
        if "below support" in v or "caution" in v:
            return "color: #ef4444; font-weight: 700;"
        return ""

    def style_bounce(v):
        if pd.isna(v):
            return ""
        v = str(v).lower()
        if "confirmed bounce" in v:
            return "color: #22c55e; font-weight: 800;"
        if "early bounce" in v:
            return "color: #84cc16; font-weight: 700;"
        if "support test" in v or "stabilization" in v or "watch" in v:
            return "color: #60a5fa; font-weight: 700;"
        if "broken" in v or "below" in v:
            return "color: #ef4444; font-weight: 700;"
        return ""

    def style_label(v):
        if pd.isna(v):
            return ""
        return f"color: {label_color(str(v))}; font-weight: 800;"

    def style_quality(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n >= 7:
            return "color: #22c55e; font-weight: 700;"
        if n >= 4:
            return "color: #facc15; font-weight: 700;"
        return "color: #ef4444; font-weight: 700;"

    def style_entry(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n >= 5:
            return "color: #22c55e; font-weight: 700;"
        if n >= 2:
            return "color: #facc15; font-weight: 700;"
        return "color: #ef4444; font-weight: 700;"

    def style_risk(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n >= 0:
            return "color: #22c55e; font-weight: 700;"
        if n >= -3:
            return "color: #facc15; font-weight: 700;"
        if n >= -6:
            return "color: #fb923c; font-weight: 700;"
        return "color: #ef4444; font-weight: 800;"

    def style_rs(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n > 1:
            return "color: #22c55e; font-weight: 700;"
        if n < -1:
            return "color: #ef4444; font-weight: 700;"
        return "color: #facc15; font-weight: 700;"

    def style_cmf(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n > 0.05:
            return "color: #22c55e; font-weight: 700;"
        if n < -0.05:
            return "color: #ef4444; font-weight: 700;"
        return "color: #facc15; font-weight: 700;"

    def style_dist(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if -3.5 <= n <= 2:
            return "color: #22c55e; font-weight: 700;"
        if -6 <= n <= 8:
            return "color: #facc15; font-weight: 700;"
        return "color: #ef4444; font-weight: 700;"

    def style_score(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n >= 14:
            return "color: #22c55e; font-weight: 800;"
        if n >= 8:
            return "color: #84cc16; font-weight: 800;"
        if n >= 1:
            return "color: #facc15; font-weight: 800;"
        if n >= -8:
            return "color: #fb923c; font-weight: 800;"
        return "color: #ef4444; font-weight: 800;"

    def style_confidence(v):
        n = _safe_num(v)
        if n is None:
            return ""
        if n >= 80:
            return "color: #22c55e; font-weight: 700;"
        if n >= 65:
            return "color: #84cc16; font-weight: 700;"
        if n >= 50:
            return "color: #facc15; font-weight: 700;"
        return "color: #fb923c; font-weight: 700;"

    styled = df.style

    style_groups = [
        (("support_strength_label", "Support"), style_support_strength),
        (("liquidity_label", "Liquidity"), style_liquidity),
        (("market_regime", "Market Regime"), style_regime),
        (("entry_status", "Entry Status"), style_entry_status),
        (("label", "Signal"), style_label),
        (("quality_score", "Quality"), style_quality),
        (("entry_score", "Entry"), style_entry),
        (("risk_score", "Risk"), style_risk),
        (("rs_20", "RS 20", "rs_60", "RS 60"), style_rs),
        (("cmf_20", "CMF 20"), style_cmf),
        (("dist_sma50_pct", "Dist SMA50"), style_dist),
        (("dist_sma200_pct", "Dist SMA200"), style_dist),
        (("score", "Score"), style_score),
        (("confidence", "Conviction"), style_confidence),
        (("bounce_signal", "Bounce Signal"), style_bounce),
    ]

    for candidates, func in style_groups:
        matching = [col for col in candidates if col in df.columns]
        if matching:
            styled = styled.map(func, subset=matching)

    formatters = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if pd.api.types.is_integer_dtype(df[col]):
                formatters[col] = lambda v: "N/A" if pd.isna(v) else f"{int(v)}"
            else:
                formatters[col] = lambda v: "N/A" if pd.isna(v) else f"{float(v):.2f}"

    if formatters:
        styled = styled.format(formatters)

    return styled


def render_deep_dive_section(selected_symbol, stock_df, stock_signal, trade_levels, regime_data):
    latest = stock_df.iloc[-1]
    signal_hex = label_color(stock_signal["label"])
    regime_hex = market_regime_color(regime_data.get("market_regime"))
    chart_start, chart_end = get_default_3m_range(stock_df)
    price_y_range = get_visible_y_range(
        stock_df,
        chart_start,
        chart_end,
        ["low", "high", "ema_9", "ema_21", "sma_50", "sma_200", "bb_low", "bb_high"],
    )
    volume_y_range = get_visible_y_range(stock_df, chart_start, chart_end, ["volume"], padding_pct=0.1)
    rsi_y_range = get_visible_y_range(stock_df, chart_start, chart_end, ["rsi_14"], padding_pct=0.08)
    adx_y_range = get_visible_y_range(stock_df, chart_start, chart_end, ["adx", "adx_pos", "adx_neg"], padding_pct=0.08)
    macd_y_range = get_visible_y_range(stock_df, chart_start, chart_end, ["macd", "macd_signal", "macd_hist"], padding_pct=0.1)
    flow_y_range = get_visible_y_range(stock_df, chart_start, chart_end, ["cmf_20"], padding_pct=0.12)

    entry_status_hex = entry_status_color(trade_levels["entry_status"])
    support_strength_text = support_strength_label(trade_levels.get("support_strength"))
    support_strength_hex = support_strength_color(support_strength_text)
    deep_dive_regime = regime_data.get("market_regime", "Unknown")

    if deep_dive_regime == "Bull":
        regime_note = "Tailwind: broader conditions are supportive."
    elif deep_dive_regime == "Neutral":
        regime_note = "Mixed backdrop: stock-level support matters more."
    elif deep_dive_regime == "Bear":
        regime_note = "Headwind: support failures are more common here."
    else:
        regime_note = "Broader regime context is unavailable."

    action_color = signal_hex
    entry_status_lower = (trade_levels.get("entry_status") or "").lower()
    if "pressure" in entry_status_lower or stock_signal.get("risk_score", 0) <= -4:
        action_color = "#ef4444"
    elif stock_signal["label"] == "Stalk / Watchlist":
        action_color = "#60a5fa"
    elif "wait" in entry_status_lower:
        action_color = "#facc15"

    st.markdown(
        f"""
        <div class='section-card'>
            <div style='display:flex; justify-content:space-between; align-items:flex-start; gap:1.25rem; flex-wrap:wrap; margin-bottom:0.35rem;'>
                <div>
                    <div style='font-size:1.45rem; font-weight:800; color:{signal_hex}; margin-bottom:0.35rem;'>
                        {selected_symbol} — {stock_signal["label"]}
                    </div>
                    <div style='display:flex; flex-wrap:wrap; gap:0.45rem;'>
                        <span class='summary-chip'>Confidence: {stock_signal["confidence"]}%</span>
                        <span class='summary-chip'>Score: {stock_signal["score"]}</span>
                        <span class='summary-chip'>Quality: {stock_signal.get("quality_score", "N/A")}</span>
                        <span class='summary-chip'>Entry: {stock_signal.get("entry_score", "N/A")}</span>
                        <span class='summary-chip'>Risk: {stock_signal.get("risk_score", "N/A")}</span>
                        <span class='summary-chip' style='color:{regime_hex};'>Regime: {deep_dive_regime}</span>
                    </div>
                </div>
                <div style='min-width:190px; text-align:right;'>
                    <div style='font-size:0.8rem; letter-spacing:0.06em; text-transform:uppercase; color:#9fb0c8; font-weight:800; margin-bottom:0.2rem;'>
                        Latest Price
                    </div>
                    <div style='font-size:2.15rem; line-height:1.05; font-weight:900; color:#f8fbff; letter-spacing:-0.03em;'>
                        {fmt_price(latest["close"])}
                    </div>
                </div>
            </div>
            <div style='color:#cbd5e1; line-height:1.65;'>
                Price is sitting with
                <b style='color:{support_strength_hex};'>{support_strength_text}</b> support quality and
                <b style='color:{entry_status_hex};'>{trade_levels["entry_status"]}</b> entry status.
                <span style='color:#9fb0c8;'> {regime_note}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption(
        build_confidence_explanation(
            {
                "label": stock_signal["label"],
                "confidence": stock_signal["confidence"],
                "quality_score": stock_signal.get("quality_score"),
                "entry_score": stock_signal.get("entry_score"),
                "risk_score": stock_signal.get("risk_score"),
            }
        )
    )

    setup_tab, charts_tab, notes_tab = st.tabs(["Setup Overview", "Charts", "Decision Notes"])

    with setup_tab:
        trend_check = "Pass" if stock_signal.get("quality_score", 0) >= 6 else "Caution"
        liquidity_check = liquidity_label(latest.get("liquidity_ok"))
        support_check = trade_levels.get("entry_status", "N/A")
        risk_check = "Controlled" if stock_signal.get("risk_score", -99) >= -2 else "Elevated"
        bounce_signal = trade_levels.get("bounce_signal", "N/A")

        if "confirmed bounce" in (bounce_signal or "").lower():
            bounce_color = "#22c55e"
        elif "early bounce" in (bounce_signal or "").lower():
            bounce_color = "#84cc16"
        elif "test" in (bounce_signal or "").lower() or "watch" in (bounce_signal or "").lower():
            bounce_color = "#60a5fa"
        elif "broken" in (bounce_signal or "").lower():
            bounce_color = "#ef4444"
        else:
            bounce_color = "#cbd5e1"

        render_section_header(
            "Put-Seller Checklist",
            "Use these four checks first before you dig into indicators or charts.",
            "Decision Framework",
        )
        st.markdown(
            "<div class='analysis-note'>A stronger put-selling setup usually shows decent trend quality, acceptable liquidity, nearby support, and risk that is not already unraveling.</div>",
            unsafe_allow_html=True,
        )
        ck1, ck2, ck3, ck4 = st.columns(4, gap="large")
        ck1.markdown(render_status_box("Trend Quality", trend_check, state_color(trend_check)), unsafe_allow_html=True)
        ck2.markdown(render_status_box("Liquidity", liquidity_check, state_color(liquidity_check)), unsafe_allow_html=True)
        ck3.markdown(render_status_box("Support State", support_check, state_color(support_check)), unsafe_allow_html=True)
        ck4.markdown(render_status_box("Risk Posture", risk_check, state_color(risk_check)), unsafe_allow_html=True)

        st.markdown("<div class='deep-dive-grid-gap'></div>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4, gap="large")
        m1.markdown(
            render_metric_tile(
                "Latest Price",
                fmt_price(latest["close"]),
                "#f8fafc",
                "Current close used for the setup reading.",
            ),
            unsafe_allow_html=True,
        )
        m2.markdown(
            render_metric_tile(
                "Primary Support",
                fmt_price(trade_levels["primary_support"]),
                support_strength_hex,
                "Nearest area expected to hold if the setup stays intact.",
            ),
            unsafe_allow_html=True,
        )
        m3.markdown(
            render_metric_tile(
                "Preferred Entry",
                fmt_price(trade_levels["recommended_entry"]),
                entry_status_hex,
                "The dashboard's support-based zone for a cleaner put entry.",
            ),
            unsafe_allow_html=True,
        )
        m4.markdown(
            render_metric_tile(
                "Bounce Signal",
                bounce_signal,
                bounce_color,
                "This helps separate a simple support test from an improving rebound.",
            ),
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4, gap="large")
        r1.markdown(
            render_metric_tile(
                "Support Strength",
                support_strength_text,
                support_strength_hex,
                "Confluence of trend levels, structure, and nearby reference support.",
            ),
            unsafe_allow_html=True,
        )
        r2.markdown(
            render_metric_tile(
                "Support Distance",
                f'{trade_levels["support_distance_pct"]:.2f}%'
                if trade_levels.get("support_distance_pct") is not None else "N/A",
                "#60a5fa",
                "Smaller gaps usually mean price is trading closer to decision levels.",
            ),
            unsafe_allow_html=True,
        )
        r3.markdown(
            render_metric_tile(
                "Market Regime",
                deep_dive_regime,
                regime_hex,
                regime_note,
            ),
            unsafe_allow_html=True,
        )
        r4.markdown(
            render_metric_tile(
                "RS vs SPY (20D)",
                f'{latest["rs_20"]:.2f}' if pd.notna(latest.get("rs_20")) else "N/A",
                "#60a5fa" if pd.notna(latest.get("rs_20")) else "#cbd5e1",
                "Positive values suggest the stock is holding up better than SPY.",
            ),
            unsafe_allow_html=True,
        )

        perf1, perf2, perf3 = st.columns(3, gap="large")
        perf1.markdown(
            render_metric_tile("5-Day Return", pct_return(stock_df, 5), "#f8fafc"),
            unsafe_allow_html=True,
        )
        perf2.markdown(
            render_metric_tile("20-Day Return", pct_return(stock_df, 20), "#f8fafc"),
            unsafe_allow_html=True,
        )
        perf3.markdown(
            render_metric_tile("50-Day Return", pct_return(stock_df, 50), "#f8fafc"),
            unsafe_allow_html=True,
        )

    with charts_tab:
        render_section_header(
            "Price And Indicator View",
            "Start with price, support, and entry zone. Use the other charts to confirm momentum and participation.",
            "Chart Desk",
        )

        price_fig = go.Figure()
        price_fig.add_trace(go.Candlestick(
            x=stock_df["timestamp"],
            open=stock_df["open"],
            high=stock_df["high"],
            low=stock_df["low"],
            close=stock_df["close"],
            name="Candlestick"
        ))
        price_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["ema_9"], mode="lines", name="EMA 9"))
        price_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["ema_21"], mode="lines", name="EMA 21"))
        price_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["sma_50"], mode="lines", name="SMA 50"))
        if stock_df["sma_200"].notna().any():
            price_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["sma_200"], mode="lines", name="SMA 200"))
        price_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["bb_high"], mode="lines", name="BB High", line=dict(dash="dot")))
        price_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["bb_low"], mode="lines", name="BB Low", line=dict(dash="dot")))

        if trade_levels["primary_support"] is not None:
            price_fig.add_hline(
                y=trade_levels["primary_support"],
                line_dash="dash",
                annotation_text=f"Primary Support: {trade_levels['primary_support']}",
                annotation_position="top left"
            )
        if trade_levels["secondary_support"] is not None:
            price_fig.add_hline(
                y=trade_levels["secondary_support"],
                line_dash="dot",
                annotation_text=f"Secondary Support: {trade_levels['secondary_support']}",
                annotation_position="top left"
            )
        if trade_levels["recommended_entry"] is not None:
            price_fig.add_hline(
                y=trade_levels["recommended_entry"],
                line_dash="solid",
                annotation_text=f"Recommended Entry: {trade_levels['recommended_entry']}",
                annotation_position="top right"
            )
        if trade_levels["entry_zone_low"] is not None and trade_levels["entry_zone_high"] is not None:
            price_fig.add_hrect(
                y0=trade_levels["entry_zone_low"],
                y1=trade_levels["entry_zone_high"],
                fillcolor="rgba(34,197,94,0.10)",
                line_width=0,
                annotation_text="Entry Zone",
                annotation_position="top left"
            )

        price_fig.update_layout(
            title=f"{selected_symbol} Price Action",
            template="plotly_dark",
            height=560,
            dragmode="zoom",
            hovermode="x unified",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.55)",
            yaxis=dict(fixedrange=False, autorange=True, range=price_y_range),
            xaxis=dict(
                fixedrange=False,
                range=[chart_start, chart_end],
                rangeselector=dict(
                    buttons=[
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(step="year", stepmode="todate", label="YTD"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                )
            )
        )
        st.plotly_chart(price_fig, use_container_width=True, config=PLOT_CONFIG)

        col_left, col_right = st.columns(2)
        with col_left:
            volume_fig = go.Figure()
            volume_fig.add_trace(go.Bar(x=stock_df["timestamp"], y=stock_df["volume"], name="Volume"))
            volume_fig.update_layout(
                title="Volume",
                template="plotly_dark",
                height=280,
                dragmode="zoom",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Volume",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.55)",
                yaxis=dict(fixedrange=False, autorange=True, range=volume_y_range),
                xaxis=dict(fixedrange=False, range=[chart_start, chart_end])
            )
            st.plotly_chart(volume_fig, use_container_width=True, config=PLOT_CONFIG)

            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["rsi_14"], mode="lines", name="RSI"))
            rsi_fig.add_hline(y=55, line_dash="dash")
            rsi_fig.add_hline(y=40, line_dash="dash")
            rsi_fig.add_hline(y=35, line_dash="dot")
            rsi_fig.update_layout(
                title="RSI Pullback Zone",
                template="plotly_dark",
                height=300,
                dragmode="zoom",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="RSI",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.55)",
                yaxis=dict(fixedrange=False, autorange=True, range=rsi_y_range),
                xaxis=dict(fixedrange=False, range=[chart_start, chart_end])
            )
            st.plotly_chart(rsi_fig, use_container_width=True, config=PLOT_CONFIG)

            adx_fig = go.Figure()
            adx_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["adx"], mode="lines", name="ADX"))
            adx_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["adx_pos"], mode="lines", name="+DI"))
            adx_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["adx_neg"], mode="lines", name="-DI"))
            adx_fig.add_hline(y=20, line_dash="dash")
            adx_fig.update_layout(
                title="Trend Strength (ADX / DI)",
                template="plotly_dark",
                height=320,
                dragmode="zoom",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Value",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.55)",
                yaxis=dict(fixedrange=False, autorange=True, range=adx_y_range),
                xaxis=dict(fixedrange=False, range=[chart_start, chart_end])
            )
            st.plotly_chart(adx_fig, use_container_width=True, config=PLOT_CONFIG)

        with col_right:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["macd"], mode="lines", name="MACD"))
            macd_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["macd_signal"], mode="lines", name="Signal"))
            macd_fig.add_trace(go.Bar(x=stock_df["timestamp"], y=stock_df["macd_hist"], name="Histogram"))
            macd_fig.update_layout(
                title="MACD",
                template="plotly_dark",
                height=300,
                dragmode="zoom",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="Value",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.55)",
                yaxis=dict(fixedrange=False, autorange=True, range=macd_y_range),
                xaxis=dict(fixedrange=False, range=[chart_start, chart_end])
            )
            st.plotly_chart(macd_fig, use_container_width=True, config=PLOT_CONFIG)

            score_breakdown = pd.DataFrame({
                "Category": ["Quality", "Entry", "Risk", "Trend", "Support", "Money Flow"],
                "Score": [
                    stock_signal.get("quality_score", 0),
                    stock_signal.get("entry_score", 0),
                    stock_signal.get("risk_score", 0),
                    stock_signal["trend_score"],
                    stock_signal["support_score"],
                    stock_signal["flow_score"]
                ]
            })

            breakdown_fig = go.Figure()
            breakdown_fig.add_trace(go.Bar(x=score_breakdown["Category"], y=score_breakdown["Score"], name="Score Breakdown"))
            breakdown_fig.update_layout(
                title="Signal Breakdown",
                template="plotly_dark",
                height=300,
                dragmode="zoom",
                hovermode="x unified",
                xaxis_title="Component",
                yaxis_title="Score",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.55)",
                yaxis=dict(fixedrange=False, autorange=True),
                xaxis=dict(fixedrange=False)
            )
            st.plotly_chart(breakdown_fig, use_container_width=True, config=PLOT_CONFIG)

            flow_fig = go.Figure()
            flow_fig.add_trace(go.Scatter(x=stock_df["timestamp"], y=stock_df["cmf_20"], mode="lines", name="CMF"))
            flow_fig.add_hline(y=0.05, line_dash="dash")
            flow_fig.add_hline(y=0.0, line_dash="dot")
            flow_fig.add_hline(y=-0.05, line_dash="dash")
            flow_fig.update_layout(
                title="Chaikin Money Flow",
                template="plotly_dark",
                height=320,
                dragmode="zoom",
                hovermode="x unified",
                xaxis_title="Time",
                yaxis_title="CMF",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(15,23,42,0.55)",
                yaxis=dict(fixedrange=False, autorange=True, range=flow_y_range),
                xaxis=dict(fixedrange=False, range=[chart_start, chart_end])
            )
            st.plotly_chart(flow_fig, use_container_width=True, config=PLOT_CONFIG)

    with notes_tab:
        selected_row = {
            "symbol": selected_symbol,
            "label": stock_signal["label"],
            "confidence": stock_signal["confidence"],
            "score": stock_signal["score"],
            "reasons": " | ".join(stock_signal["reasons"]),
            "trend_score": stock_signal["trend_score"],
            "pullback_score": stock_signal["pullback_score"],
            "support_score": stock_signal["support_score"],
            "flow_score": stock_signal["flow_score"],
            "quality_score": stock_signal.get("quality_score"),
            "entry_score": stock_signal.get("entry_score"),
            "risk_score": stock_signal.get("risk_score"),
            "entry_status": trade_levels.get("entry_status"),
            "bounce_signal": trade_levels.get("bounce_signal"),
            "rsi": round(latest["rsi_14"], 2),
            "rs_20": round(latest["rs_20"], 2) if pd.notna(latest.get("rs_20")) else None,
            "liquidity_ok": normalize_boolish(latest.get("liquidity_ok")),
            "market_regime": regime_data.get("market_regime"),
            "candidate_blockers": stock_signal.get("candidate_blockers", []),
        }

        render_section_header(
            "Suggested Action",
            "This is the dashboard's best judgment on what a general put seller should do next.",
            "Playbook",
        )
        suggestion_title, suggestion_body = build_action_suggestion(selected_row)
        st.markdown(
            f"""
            <div class='decision-card' style='border-left: 4px solid {action_color};'>
                <div class='decision-label'>Suggested Action</div>
                <div class='decision-title' style='color:{action_color};'>{suggestion_title}</div>
                <div class='decision-body'>{suggestion_body}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        blockers = stock_signal.get("candidate_blockers") or []
        if stock_signal["label"] not in {"Put Sell Candidate", "High Probability Put Sell"} and blockers:
            render_section_header(
                "What Is Preventing A Trade-Ready Signal",
                "These are the main reasons the stock is not currently qualifying as a stronger put-selling setup.",
                "Blockers",
            )
            blocker_cards = "".join(
                [f"<div class='reason-bullet'>- {colorize_signal_reason(item)}</div>" for item in blockers[:5]]
            )
            st.markdown(
                f"""
                <div class='reason-group-card' style='border-left:4px solid #fb923c;'>
                    <div class='reason-group-title'>Candidate Blockers</div>
                    {blocker_cards}
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_section_header(
            "Why This Signal Was Assigned",
            "Read this from top to bottom: the summary sets the context, and the grouped reasons explain the actual score.",
            "Reasoning",
        )
        for takeaway in build_signal_takeaways(selected_row):
            st.markdown(
                f"<div class='insight-card'>{colorize_signal_reason(takeaway.replace('**', ''))}</div>",
                unsafe_allow_html=True,
            )

        grouped_reasons = group_signal_reasons(stock_signal["reasons"][:8])
        for heading, items in grouped_reasons.items():
            if not items:
                continue
            bullets = "".join(
                [f"<div class='reason-bullet'>- {colorize_signal_reason(reason)}</div>" for reason in items]
            )
            st.markdown(
                f"""
                <div class='reason-group-card'>
                    <div class='reason-group-title'>{heading}</div>
                    {bullets}
                </div>
                """,
                unsafe_allow_html=True,
            )


@st.cache_data(ttl=900)
def load_stock_analysis(symbol: str, learning_sig: str = ""):
    del learning_sig
    return analyze_stock(symbol, learning_profile=load_learning_profile())


@st.cache_data(ttl=900)
def load_stock_snapshot(symbol: str, learning_sig: str = ""):
    del learning_sig
    return summarize_stock(symbol, learning_profile=load_learning_profile())


def run_full_analysis():
    if not st.session_state.watchlist:
        st.session_state.analysis_df = pd.DataFrame()
        st.session_state.error_df = pd.DataFrame()
        st.session_state.analysis_ready = False
        st.error("Your watchlist is empty. Add at least one ticker.")
        return

    progress_text = st.empty()
    progress_bar = st.progress(0, text="Starting analysis...")
    tickers = list(st.session_state.watchlist)
    total = len(tickers)
    results = []

    progress_text.caption("Refreshing automatic backtest and learning profile...")
    progress_bar.progress(8, text="Running automatic backtest and learning...")
    learning_profile, backtest_summary, learning_refreshed = run_automatic_learning_cycle(tickers)
    st.session_state.learning_profile = learning_profile
    st.session_state.backtest_summary = backtest_summary

    learning_sig = learning_signature(learning_profile)
    if learning_refreshed:
        load_stock_snapshot.clear()
        load_stock_analysis.clear()

    progress_bar.progress(12, text="Scoring live watchlist with the latest learning profile...")

    for idx, symbol in enumerate(tickers, start=1):
        progress_pct = 12 + int(((idx - 1) / total) * 73)
        progress_text.caption(f"Analyzing {symbol} ({idx}/{total})...")
        progress_bar.progress(progress_pct, text=f"Running analysis for {symbol}...")

        try:
            results.append(load_stock_snapshot(symbol, learning_sig))
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e),
            })

        progress_pct = 12 + int((idx / total) * 73)
        progress_bar.progress(progress_pct, text=f"Finished {symbol}.")

    progress_bar.progress(88, text="Building watchlist results...")
    df = pd.DataFrame(results)

    if df.empty:
        progress_bar.empty()
        progress_text.empty()
        st.session_state.analysis_df = pd.DataFrame()
        st.session_state.error_df = pd.DataFrame()
        st.session_state.analysis_ready = False
        st.error("No data was returned from the analysis step.")
        return

    progress_bar.progress(93, text="Separating valid results from fetch errors...")
    if "error" in df.columns:
        error_df = df[df["error"].notna()].copy() if df["error"].notna().any() else pd.DataFrame()
        clean_df = df[df["error"].isna()].copy() if df["error"].notna().any() else df.copy()
    else:
        error_df = pd.DataFrame()
        clean_df = df.copy()

    progress_bar.progress(94, text="Saving analysis results to the dashboard...")
    st.session_state.analysis_df = clean_df
    st.session_state.error_df = error_df
    st.session_state.analysis_ready = not clean_df.empty
    st.session_state.learning_profile = learning_profile
    st.session_state.backtest_summary = backtest_summary

    if not clean_df.empty and st.session_state.selected_symbol not in clean_df["symbol"].tolist():
        st.session_state.selected_symbol = clean_df["symbol"].iloc[0]

    progress_bar.progress(100, text="Analysis complete.")
    progress_bar.empty()
    progress_text.empty()


if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = st.session_state.watchlist[0] if st.session_state.watchlist else None

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None

if "error_df" not in st.session_state:
    st.session_state.error_df = pd.DataFrame()

if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False

if "learning_profile" not in st.session_state:
    st.session_state.learning_profile = load_learning_profile()

if "backtest_summary" not in st.session_state:
    st.session_state.backtest_summary = load_backtest_summary()


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-kicker">Control Center</div>
        <div class="sidebar-title">Scan And Manage</div>
        <div class="sidebar-copy">Filter the dashboard, manage your watchlist, and rerun the full analysis from one place.</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sidebar-status-shell">
            <div class="sidebar-watchlist-title">Dashboard Status</div>
            <div class="sidebar-watchlist-subtitle">
                {len(st.session_state.watchlist)} ticker(s) loaded
                {'• analysis ready' if st.session_state.analysis_ready else '• run analysis to refresh rankings'}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    filter_panel = st.container(border=True)
    with filter_panel:
        st.markdown("### Filter View")
        st.caption("Choose which signal tier should drive the ranking and deep-dive selection.")
        signal_filter = st.selectbox(
            "Signal Filter",
            [
                "All",
                "High Probability Put Sell",
                "Put Sell Candidate",
                "Stalk / Watchlist",
                "Neutral / Wait",
                "Downtrend Risk",
                "Breakdown Risk",
            ],
            index=0
        )

    current_count = len(st.session_state.watchlist)

    add_panel = st.container(border=True)
    with add_panel:
        st.markdown(
            f"""
            <div class="sidebar-watchlist-shell" style="margin-top:0; margin-bottom:0.85rem;">
                <div class="sidebar-watchlist-title">Your Watchlist</div>
                <div class="sidebar-watchlist-subtitle">{current_count} ticker(s) saved</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.form("add_ticker_form", clear_on_submit=True):
            new_ticker = st.text_input(
                "Add a ticker",
                placeholder="Ex: NFLX"
            ).upper().strip()
            add_submitted = st.form_submit_button("Add")

    if add_submitted:
        if not new_ticker:
            st.warning("Enter a ticker first.")
        elif not is_valid_ticker(new_ticker):
            st.error("Ticker format looks invalid.")
        else:
            updated, message = add_to_watchlist(st.session_state.watchlist, new_ticker)
            st.session_state.watchlist = updated

            if st.session_state.selected_symbol is None and updated:
                st.session_state.selected_symbol = updated[0]

            if "Added" in message:
                st.success(message)
                st.rerun()
            else:
                st.warning(message)

    watchlist_container = st.container(border=True)

    with watchlist_container:
        st.markdown("### Current Watchlist")
        st.caption("Click the x to remove a name. The selected name becomes the default deep-dive stock.")
        if st.session_state.watchlist:
            for ticker in st.session_state.watchlist:
                row_col1, row_col2 = st.columns([6.2, 0.8])

                with row_col1:
                    is_selected = ticker == st.session_state.selected_symbol
                    if is_selected:
                        st.markdown(
                            f"<div class='sidebar-list-row sidebar-list-row-selected'>{ticker}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(f"<div class='sidebar-list-row'>{ticker}</div>", unsafe_allow_html=True)

                with row_col2:
                    remove_key = f"remove_{ticker}"
                    if st.button("x", key=remove_key, use_container_width=True, type="secondary"):
                        updated, message = remove_from_watchlist(st.session_state.watchlist, ticker)
                        st.session_state.watchlist = updated

                        if st.session_state.selected_symbol == ticker:
                            st.session_state.selected_symbol = updated[0] if updated else None

                        if st.session_state.analysis_df is not None and not st.session_state.analysis_df.empty:
                            st.session_state.analysis_df = st.session_state.analysis_df[
                                st.session_state.analysis_df["symbol"] != ticker
                            ].reset_index(drop=True)

                        st.success(message)
                        st.rerun()
        else:
            st.info("Your watchlist is empty.")

    tip_panel = st.container(border=True)
    with tip_panel:
        st.markdown("### Workflow Tip")
        st.markdown(
            "<div class='sidebar-tip'>Run analysis after updating the watchlist, then use the ranked view to sort candidates before opening the stock deep dive.</div>",
            unsafe_allow_html=True
        )

    sidebar_run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")
    st.caption("Refresh the watchlist ranking and deep-dive data.")

if sidebar_run_analysis:
    run_full_analysis()

hero_left, hero_right = st.columns([5.7, 1.45], vertical_alignment="center", gap="medium")

with hero_left:
    st.markdown(
        f"""
        <div class='hero-shell'>
            <div class='hero-kicker'>Cash-Secured Put Workflow</div>
            <div class='hero-title'>Put Selling Dashboard</div>
            <div class='hero-subtitle'>Screen watchlist names, separate live put-sale setups from stalk-list names, and keep the support-and-bounce decision process easy to read.</div>
            <div>
                <span class='summary-chip'>Watchlist: {len(st.session_state.watchlist)} ticker(s)</span>
                <span class='summary-chip'>Filter: {signal_filter}</span>
                <span class='summary-chip'>Default View: 3M</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with hero_right:
    st.markdown(
        """
        <div class='section-card' style='margin-top:0.15rem;'>
            <div class='section-eyebrow'>Refresh</div>
            <div style='color:#dbe6f4; font-size:0.94rem; line-height:1.55; margin-bottom:0.7rem;'>
                Re-run the screen after watchlist changes or after the market moves.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hero_run_analysis = st.button(
        "🚀 Run Analysis",
        key="hero_run_analysis",
        use_container_width=True,
        type="primary"
    )
    st.caption("Run or refresh the full watchlist analysis.")

if hero_run_analysis:
    run_full_analysis()

st.markdown(
    "<div class='hero-note'>Tip: start with the trade queue, then open the deep dive only for names that already look close to support or are beginning to bounce. Charts default to the last 3 months.</div>",
    unsafe_allow_html=True
)

render_section_header(
    "Current Watchlist",
    "These are the names currently being scanned whenever you run the full dashboard analysis.",
    "Workspace",
)
if st.session_state.watchlist:
    st.markdown(
        f"""
        <div class='watchlist-card'>
            {render_pills(st.session_state.watchlist)}
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.warning("Your watchlist is empty. Add a ticker from the sidebar.")

if st.session_state.analysis_ready and st.session_state.analysis_df is not None:
    clean_df = st.session_state.analysis_df.copy()
    error_df = st.session_state.error_df.copy()

    if clean_df.empty:
        st.warning("No valid stock data was returned for the watchlist.")
        if not error_df.empty:
            st.markdown("### Fetch Errors")
            st.dataframe(error_df, use_container_width=True, hide_index=True)
        st.stop()

    clean_df = sort_signal_df(clean_df).reset_index(drop=True)

    filtered_df = clean_df.copy()
    if signal_filter != "All":
        filtered_df = filtered_df[filtered_df["label"] == signal_filter].reset_index(drop=True)

    if filtered_df.empty:
        st.warning("No stocks match the selected filter.")
        if not error_df.empty:
            st.markdown("### Fetch Errors")
            st.dataframe(error_df, use_container_width=True, hide_index=True)
        st.stop()

    if st.session_state.selected_symbol not in filtered_df["symbol"].tolist():
        st.session_state.selected_symbol = filtered_df["symbol"].iloc[0]

    counts = build_summary_counts(clean_df)

    # Global market context from the first analyzed stock row
    market_regime = clean_df["market_regime"].iloc[0] if "market_regime" in clean_df.columns and not clean_df.empty else "Unknown"
    spy_close = clean_df["spy_close"].iloc[0] if "spy_close" in clean_df.columns and not clean_df.empty else None
    spy_dist = clean_df["spy_dist_sma200_pct"].iloc[0] if "spy_dist_sma200_pct" in clean_df.columns and not clean_df.empty else None

    regime_hex = market_regime_color(market_regime)

    if market_regime == "Bull":
        regime_note = "Broader conditions are supportive for cash-secured put selling, though stock-level quality and support still matter."
    elif market_regime == "Neutral":
        regime_note = "Broader conditions are mixed, so stronger support quality and cleaner entries matter more than usual."
    elif market_regime == "Bear":
        regime_note = "Broader conditions are unfavorable for put selling, and support breaks become more common."
    else:
        regime_note = "Market regime context is currently unavailable."

    qualified_put_sells = clean_df[
        clean_df["label"].isin(["High Probability Put Sell", "Put Sell Candidate"])
    ].sort_values(["score", "confidence"], ascending=[False, False]).head(5)

    stalk_names = clean_df[
        clean_df["label"] == "Stalk / Watchlist"
    ].sort_values(["score", "confidence"], ascending=[False, False]).head(6)

    avoid_names = clean_df[
        clean_df["label"].isin(["Downtrend Risk", "Breakdown Risk"])
    ].sort_values(["score", "confidence"], ascending=[True, True]).head(5)

    trade_queue_tab, ranking_tab, deep_dive_tab, backtest_tab, guide_tab = st.tabs(
        ["Trade Queue", "Full Ranking", "Stock Deep Dive", "Auto Backtest", "Guide"]
    )

    with trade_queue_tab:
        render_section_header(
            "Trade Queue",
            "Start with the broad backdrop, then review the short list of setups that are closest to an actual put-selling decision.",
            "Workflow",
        )
        st.markdown(
            "<div class='analysis-note'>Work left to right: confirm the market backdrop, scan the stronger setups, then avoid forcing anything that still looks pressured or structurally weak.</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4, c5, c6 = st.columns(6, gap="medium")
        c1.metric("High Conviction", counts["High Probability Put Sell"])
        c2.metric("Candidates", counts["Put Sell Candidate"])
        c3.metric("Stalk", counts["Stalk / Watchlist"])
        c4.metric("Neutral", counts["Neutral / Wait"])
        c5.metric("Downtrend Risk", counts["Downtrend Risk"])
        c6.metric("Breakdown Risk", counts["Breakdown Risk"])

        st.markdown(
            f"""
            <div class='queue-shell'>
                <div style='display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;'>
                    <div style='font-size:1.15rem; font-weight:800; color:{regime_hex};'>
                        Market Regime: {market_regime}
                    </div>
                    <div style='color:#d6e0ee; font-size:0.98rem;'>
                        SPY Close: <b>{fmt_price(spy_close)}</b> &nbsp; | &nbsp;
                        SPY Dist to 200 SMA: <b>{f"{spy_dist:.2f}%" if spy_dist is not None and not pd.isna(spy_dist) else "N/A"}</b>
                    </div>
                </div>
                <div style='margin-top:0.55rem; color:#9fb0c8; line-height:1.6;'>
                    {regime_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        left, right = st.columns([1.3, 1], gap="large")

        with left:
            render_section_header(
                "Actionable Setups",
                "These are the names closest to an actual put-selling decision right now.",
                "Short List",
            )
            if not qualified_put_sells.empty:
                st.markdown(
                    "<div class='table-context'>Read each row left to right: stock, nearby support, preferred entry, then entry readiness and conviction.</div>",
                    unsafe_allow_html=True,
                )
                left_display = prepare_display_table(
                    build_ranked_display_df(qualified_put_sells),
                    [
                        "symbol",
                        "price",
                        "primary_support",
                        "recommended_entry",
                        "entry_status",
                        "support_strength_label",
                        "bounce_signal",
                        "label",
                        "score",
                        "confidence",
                        "setup_note",
                    ],
                )
                st.dataframe(
                    style_ranked_table(left_display),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.markdown(
                    "<div class='soft-danger-text'>No stocks currently qualify as trade-ready put setups. Use the stalk list below instead of forcing an entry.</div>",
                    unsafe_allow_html=True
                )

            render_section_header(
                "Worth Stalking",
                "These names may still be worth owning on assignment, but the timing is not ready enough yet for a disciplined put sale.",
                "Pipeline",
            )
            if stalk_names.empty:
                st.info("No stalk-list names are standing out right now.")
            else:
                st.markdown(
                    "<div class='table-context'>These are better treated as monitored names than immediate trades. Wait for a cleaner pullback, support test, or bounce confirmation.</div>",
                    unsafe_allow_html=True,
                )
                stalk_display = prepare_display_table(
                    build_ranked_display_df(stalk_names),
                    [
                        "symbol",
                        "price",
                        "primary_support",
                        "recommended_entry",
                        "entry_status",
                        "support_strength_label",
                        "bounce_signal",
                        "label",
                        "score",
                        "confidence",
                        "setup_note",
                    ],
                )
                st.dataframe(
                    style_ranked_table(stalk_display),
                    use_container_width=True,
                    hide_index=True,
                )

        with right:
            render_section_header(
                "Avoid / Higher Risk",
                "These names are not lining up well enough for a fresh put sale right now.",
                "Risk Review",
            )
            if avoid_names.empty:
                st.info("No high-risk names found.")
            else:
                st.markdown(
                    "<div class='table-context'>This table emphasizes the structural reasons the setup is less attractive for put sellers.</div>",
                    unsafe_allow_html=True,
                )
                right_display = prepare_display_table(
                    build_avoid_display_df(avoid_names),
                    [
                        "symbol",
                        "price",
                        "primary_support",
                        "entry_status",
                        "support_strength_label",
                        "label",
                        "risk_score",
                        "confidence",
                        "avoid_reason",
                    ],
                )
                st.dataframe(
                    style_ranked_table(right_display),
                    use_container_width=True,
                    hide_index=True
                )

        if not error_df.empty:
            render_section_header(
                "Watchlist Fetch Errors",
                "These symbols could not be analyzed successfully in the latest run.",
                "Exceptions",
            )
            st.dataframe(error_df, use_container_width=True, hide_index=True)

    with ranking_tab:
        render_section_header(
            "Ranked Watchlist",
            "This is the full filtered ranking. The best trade-ready labels stay on top, then the stalk list, then the wait-or-avoid names.",
            "Ranking",
        )
        st.markdown(
            "<div class='table-context'>Focus first on label, support location, and bounce quality. Then use the score columns to judge assignment comfort, timing, and downside control.</div>",
            unsafe_allow_html=True,
        )
        ranked_display_df = prepare_display_table(
            build_ranked_display_df(filtered_df),
            [
                "symbol",
                "price",
                "primary_support",
                "secondary_support",
                "recommended_entry",
                "entry_status",
                "support_strength_label",
                "bounce_signal",
                "label",
                "score",
                "confidence",
                "quality_score",
                "entry_score",
                "risk_score",
                "setup_note",
            ],
        )

        st.dataframe(
            style_ranked_table(ranked_display_df),
            use_container_width=True,
            hide_index=True
        )

    with deep_dive_tab:
        render_section_header(
            "Stock Deep Dive",
            "Select one stock to inspect its support map, bounce behavior, chart context, and the dashboard's suggested next move.",
            "Deep Dive",
        )
        selection_col, helper_col = st.columns([2.2, 1], gap="medium")
        with selection_col:
            selected_symbol = st.selectbox(
                "Select stock",
                filtered_df["symbol"].tolist(),
                index=filtered_df["symbol"].tolist().index(st.session_state.selected_symbol),
                key="deep_dive_symbol"
            )
        with helper_col:
            st.markdown(
                """
                <div class='guide-panel' style='margin-bottom:0;'>
                    <div class='guide-block-title'>How To Use This View</div>
                    <div class='guide-block-copy'>
                        Confirm the support map first, then look at the bounce behavior, and only then use the indicators as confirmation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.session_state.selected_symbol = selected_symbol

        with st.spinner(f"Loading analysis for {selected_symbol}..."):
            try:
                active_learning_sig = learning_signature(st.session_state.learning_profile)
                stock_df, stock_signal, trade_levels, regime_data = load_stock_analysis(selected_symbol, active_learning_sig)
            except Exception as e:
                st.error(str(e))
                st.stop()

        render_deep_dive_section(selected_symbol, stock_df, stock_signal, trade_levels, regime_data)

    with backtest_tab:
        render_auto_backtest_summary(st.session_state.backtest_summary, st.session_state.learning_profile)

    with guide_tab:
        render_guide_content()

else:
    st.markdown(
        """
        <div class='empty-state'>
            <div class='section-title' style='font-size:1.25rem; margin-bottom:0.35rem;'>Run Your First Screen</div>
            <div class='section-subtitle'>
                Add or remove tickers from the sidebar, then click <b>Run Analysis</b>.
                The dashboard will rank names, surface stronger put-selling candidates, and open the deep dive for faster decision-making.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
