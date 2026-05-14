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
from src.fetch_data import fetch_benchmark_data, fetch_many_stock_data
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
    page_title="Quantitative Put Selling Dashboard",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

    :root {
        --bg-main: #08131a;
        --bg-panel: rgba(10, 23, 31, 0.90);
        --bg-panel-strong: rgba(12, 28, 37, 0.96);
        --border: rgba(145, 191, 201, 0.16);
        --text-main: #f4f7f4;
        --text-muted: #93aba5;
        --green: #22c55e;
        --lime: #84cc16;
        --yellow: #facc15;
        --orange: #fb923c;
        --red: #ef4444;
        --blue: #60a5fa;
        --shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
    }

    html, body, [class*="css"] {
        font-family: "Manrope", sans-serif;
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(45, 212, 191, 0.11), transparent 24%),
            radial-gradient(circle at 100% 0%, rgba(251, 113, 133, 0.10), transparent 22%),
            linear-gradient(180deg, #071015 0%, #09171f 44%, #0a151d 100%);
        color: var(--text-main);
    }

    .block-container {
        padding-top: 1.05rem;
        padding-bottom: 3.25rem;
        max-width: 1480px;
    }

    div[data-testid="stVerticalBlock"] {
        gap: 0.9rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1.15rem;
    }

    h1, h2, h3, h4, .hero-title, .section-title {
        font-family: "Space Grotesk", sans-serif !important;
    }

    .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: var(--text-main);
    }

    .desk-header,
    .workflow-strip,
    .hero-shell,
    .run-panel,
    .section-card,
    .watchlist-card,
    .queue-shell,
    .decision-card,
    .reason-group-card,
    .guide-panel,
    .empty-state,
    .workspace-strip,
    .ticker-rail,
    .operator-board,
    .market-cardlet,
    .priority-card,
    .sidebar-status-shell,
    .sidebar-watchlist-shell,
    .status-box,
    .metric-tile,
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--bg-panel-strong), rgba(10, 23, 31, 0.92));
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }

    .hero-shell {
        border-radius: 26px;
        padding: 1.12rem 1.22rem 1.05rem 1.22rem;
        position: relative;
        overflow: hidden;
        margin-bottom: 0.65rem;
    }

    .desk-header {
        border-radius: 26px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }

    .desk-header::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(90deg, rgba(45, 212, 191, 0.12), transparent 42%),
            radial-gradient(circle at right top, rgba(251, 146, 60, 0.11), transparent 30%);
        pointer-events: none;
    }

    .desk-header-top {
        position: relative;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .desk-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: 2.28rem;
        line-height: 1.02;
        font-weight: 700;
        letter-spacing: -0.045em;
        color: #f7fbf8;
        margin-bottom: 0.34rem;
    }

    .desk-subtitle {
        max-width: 54rem;
        color: #b8cbc6;
        font-size: 0.96rem;
        line-height: 1.55;
    }

    .desk-status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        padding: 0.55rem 0.82rem;
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid rgba(129, 230, 217, 0.18);
        background: rgba(8, 19, 26, 0.62);
        white-space: nowrap;
    }

    .desk-meta-row {
        position: relative;
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }

    .desk-meta-item {
        border-radius: 999px;
        border: 1px solid rgba(129, 230, 217, 0.12);
        background: rgba(8, 19, 26, 0.48);
        color: #dcebe6;
        padding: 0.42rem 0.68rem;
        font-size: 0.82rem;
        font-weight: 800;
    }

    .workflow-strip {
        border-radius: 22px;
        padding: 0.9rem;
        margin-bottom: 1.15rem;
    }

    .workflow-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
    }

    .workflow-step {
        border-radius: 16px;
        padding: 0.82rem 0.9rem;
        background: rgba(7, 19, 26, 0.42);
        border: 1px solid rgba(129, 230, 217, 0.09);
        min-height: 102px;
    }

    .workflow-step-active {
        border-color: rgba(45, 212, 191, 0.32);
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.13), rgba(96, 165, 250, 0.08));
    }

    .workflow-step-number {
        color: #81e6d9;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .workflow-step-title {
        color: var(--text-main);
        font-size: 0.92rem;
        font-weight: 900;
        margin-bottom: 0.18rem;
    }

    .workflow-step-copy {
        color: #93aba5;
        font-size: 0.8rem;
        line-height: 1.42;
    }

    .hero-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at top left, rgba(45, 212, 191, 0.18), transparent 32%),
            radial-gradient(circle at bottom right, rgba(251, 113, 133, 0.12), transparent 28%);
        pointer-events: none;
    }

    .hero-kicker,
    .section-eyebrow,
    .sidebar-kicker,
    .decision-label,
    .status-label,
    .metric-title,
    .reason-group-title {
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-weight: 800;
    }

    .hero-kicker,
    .section-eyebrow,
    .sidebar-kicker {
        color: #81e6d9;
    }

    .hero-title {
        position: relative;
        font-size: 2.36rem;
        line-height: 1.02;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #f7fbf8;
        margin: 0.18rem 0 0.34rem 0;
    }

    .hero-subtitle {
        position: relative;
        max-width: 50rem;
        color: #c0d3cd;
        font-size: 0.96rem;
        line-height: 1.55;
        margin: 0 0 0.78rem 0;
    }

    .hero-note {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.5;
        margin-top: 0.2rem;
        margin-bottom: 0.55rem;
    }

    .run-panel {
        border-radius: 24px;
        padding: 1rem 1.05rem;
        min-height: 100%;
    }

    .run-panel-title {
        color: var(--text-main);
        font-size: 1.05rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .run-panel-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.25rem;
        line-height: 1.05;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .run-panel-copy {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .summary-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.48rem 0.78rem;
        margin: 0.18rem 0.42rem 0.18rem 0;
        border-radius: 999px;
        border: 1px solid rgba(129, 230, 217, 0.14);
        background: rgba(10, 23, 31, 0.56);
        color: #dff3ed;
        font-size: 0.84rem;
        font-weight: 700;
    }

    .workspace-strip {
        border-radius: 22px;
        padding: 0.85rem 1rem 0.78rem 1rem;
        margin-bottom: 0.8rem;
    }

    .workspace-strip-title {
        color: var(--text-main);
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.18rem;
    }

    .workspace-strip-copy {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.55;
        margin-bottom: 0.5rem;
    }

    .pill {
        display: inline-block;
        padding: 0.42rem 0.74rem;
        margin: 0.14rem 0.24rem 0.14rem 0;
        border-radius: 999px;
        background: rgba(14, 33, 43, 0.88);
        border: 1px solid rgba(129, 230, 217, 0.12);
        font-size: 0.86rem;
        font-weight: 700;
        color: #e5f4ef;
    }

    .ticker-rail {
        border-radius: 22px;
        padding: 0.92rem 1.05rem;
        margin-bottom: 1.2rem;
    }

    .ticker-rail-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-bottom: 0.38rem;
    }

    .ticker-rail-title {
        color: var(--text-main);
        font-weight: 800;
        font-size: 0.92rem;
    }

    .ticker-rail-meta {
        color: var(--text-muted);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .operator-board {
        border-radius: 26px;
        padding: 1.1rem;
        margin-bottom: 1.25rem;
    }

    .operator-grid {
        display: grid;
        grid-template-columns: minmax(260px, 0.9fr) minmax(360px, 1.8fr);
        gap: 1rem;
        align-items: stretch;
    }

    .market-cardlet,
    .priority-card {
        border-radius: 20px;
        padding: 1.05rem 1.08rem;
        box-shadow: none;
    }

    .market-regime-line {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.32rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0.2rem 0 0.42rem 0;
    }

    .market-copy,
    .priority-copy {
        color: #b8cbc6;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    .signal-count-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.72rem;
        margin-bottom: 0.92rem;
    }

    .signal-count-tile {
        background: rgba(7, 19, 26, 0.54);
        border: 1px solid rgba(129, 230, 217, 0.10);
        border-radius: 17px;
        padding: 0.72rem 0.76rem;
        min-height: 88px;
    }

    .signal-count-label {
        color: #9db5af;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        line-height: 1.28;
        margin-bottom: 0.36rem;
    }

    .signal-count-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.55rem;
        line-height: 1;
        font-weight: 700;
    }

    .priority-title {
        font-size: 1.03rem;
        font-weight: 800;
        line-height: 1.38;
        margin: 0.18rem 0 0.28rem 0;
    }

    .context-chip-row {
        display: flex;
        gap: 0.42rem;
        flex-wrap: wrap;
        margin-top: 0.7rem;
    }

    .section-card,
    .watchlist-card,
    .queue-shell,
    .decision-card,
    .reason-group-card,
    .guide-panel,
    .empty-state {
        border-radius: 22px;
        padding: 1.05rem 1.12rem;
        margin-bottom: 1.2rem;
    }

    .section-header {
        margin-top: 0.45rem;
        margin-bottom: 0.92rem;
    }

    .section-title {
        font-size: 1.32rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--text-main);
        margin-bottom: 0.12rem;
    }

    .section-subtitle,
    .sidebar-copy,
    .sidebar-tip,
    .guide-block-copy,
    .decision-body,
    .analysis-note,
    .table-context {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }

    .sidebar-watchlist-title,
    .guide-block-title {
        color: var(--text-main);
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 0.22rem;
    }

    .sidebar-watchlist-subtitle {
        color: var(--text-muted);
        font-size: 0.8rem;
        line-height: 1.45;
    }

    .soft-danger-text {
        color: #ffb4b4;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-bottom: 0.65rem;
    }

    .empty-queue {
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px dashed rgba(250, 204, 21, 0.28);
        background: rgba(250, 204, 21, 0.065);
    }

    .empty-queue-title {
        color: #fde68a;
        font-size: 0.98rem;
        font-weight: 900;
        margin-bottom: 0.2rem;
    }

    .empty-queue-copy {
        color: #c9d8d3;
        font-size: 0.88rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        border-radius: 18px;
        padding: 0.85rem 0.95rem;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-muted) !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.02em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text-main) !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(129, 230, 217, 0.10);
        box-shadow: 0 10px 26px rgba(0, 0, 0, 0.14);
        margin-bottom: 1rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
        border: 1px solid rgba(129, 230, 217, 0.10) !important;
        background: linear-gradient(180deg, rgba(10, 23, 31, 0.76), rgba(10, 23, 31, 0.68));
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.10);
        margin-bottom: 0.8rem;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.45rem;
    }

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.45rem;
        padding-bottom: 0.7rem;
    }

    div[data-testid="stTabs"] button {
        border-radius: 999px !important;
        padding: 0.55rem 0.95rem !important;
        border: 1px solid rgba(129, 230, 217, 0.12) !important;
        background: rgba(10, 23, 31, 0.52) !important;
        color: #a9c1bb !important;
        font-weight: 800 !important;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.22), rgba(96, 165, 250, 0.18)) !important;
        border-color: rgba(129, 230, 217, 0.28) !important;
        color: #f6fbf8 !important;
    }

    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at top, rgba(45, 212, 191, 0.08), transparent 28%),
            linear-gradient(180deg, #071219 0%, #0b1820 100%);
        border-right: 1px solid rgba(129, 230, 217, 0.08);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1.35rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        border-radius: 14px;
        font-weight: 800;
        transition: all 0.18s ease;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #14b8a6, #0ea5e9) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 14px 26px rgba(20, 184, 166, 0.24) !important;
        min-height: 3rem !important;
        font-size: 1rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #0f766e, #0284c7) !important;
    }

    section[data-testid="stSidebar"] form[data-testid="stForm"] button {
        background: linear-gradient(135deg, #fb7185, #f59e0b) !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] form[data-testid="stForm"] button:hover {
        background: linear-gradient(135deg, #e11d48, #d97706) !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"] {
        background: transparent !important;
        color: #a0b7b1 !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 1.8rem !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: rgba(239, 68, 68, 0.10) !important;
        color: #fecaca !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        border-radius: 14px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #14b8a6, #0ea5e9) !important;
        border: none !important;
        box-shadow: 0 12px 26px rgba(20, 184, 166, 0.18) !important;
    }

    div[data-testid="stButton"] button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0f766e, #0284c7) !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] > div {
        border-radius: 14px !important;
    }

    div[data-baseweb="input"] input {
        background: rgba(9, 24, 32, 0.76) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 16px;
    }

    [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    [data-testid="stPopover"] button {
        border-radius: 14px !important;
        border: 1px solid rgba(129, 230, 217, 0.16) !important;
        background: rgba(10, 23, 31, 0.88) !important;
        color: #e7f5ef !important;
        font-weight: 700 !important;
    }

    .deep-dive-grid-gap {
        height: 1rem;
    }

    .status-box,
    .metric-tile {
        border-radius: 20px;
        padding: 0.95rem 1rem;
        min-height: 112px;
    }

    .status-label,
    .metric-title,
    .decision-label {
        color: #8fb4ac;
        margin-bottom: 0.42rem;
    }

    .status-value {
        font-size: 1.02rem;
        line-height: 1.42;
        font-weight: 800;
    }

    .metric-value {
        color: var(--text-main);
        font-size: 1.16rem;
        font-weight: 800;
        line-height: 1.32;
    }

    .metric-subtitle {
        color: var(--text-muted);
        font-size: 0.83rem;
        line-height: 1.48;
        margin-top: 0.38rem;
    }

    .sidebar-list-row {
        background: rgba(13, 30, 39, 0.84);
        border: 1px solid rgba(129, 230, 217, 0.08);
        border-radius: 14px;
        padding: 0.55rem 0.75rem;
        min-height: 42px;
        display: flex;
        align-items: center;
        color: #e2f1eb;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .sidebar-list-row-selected {
        background: linear-gradient(135deg, rgba(45, 212, 191, 0.16), rgba(96, 165, 250, 0.10));
        border-color: rgba(129, 230, 217, 0.28);
        color: #fbfffd;
    }

    .decision-title {
        font-size: 1.06rem;
        font-weight: 800;
        line-height: 1.42;
    }

    .insight-card {
        background: rgba(12, 28, 37, 0.74);
        border: 1px solid rgba(129, 230, 217, 0.10);
        border-radius: 18px;
        padding: 0.92rem 1rem;
        margin-bottom: 0.75rem;
        color: #e2efea;
        line-height: 1.62;
    }

    .reason-bullet {
        color: #dce9e5;
        line-height: 1.58;
        margin-bottom: 0.4rem;
        font-size: 0.92rem;
    }

    .reason-bullet:last-child {
        margin-bottom: 0;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 0.95rem;
        }

        .hero-title {
            font-size: 2.25rem;
        }

        .hero-shell,
        .desk-header,
        .workflow-strip,
        .run-panel,
        .section-card,
        .watchlist-card,
        .queue-shell,
        .decision-card,
        .reason-group-card,
        .guide-panel,
        .empty-state,
        .workspace-strip,
        .ticker-rail,
        .operator-board {
            border-radius: 20px;
        }

        .operator-grid,
        .signal-count-grid,
        .workflow-grid {
            grid-template-columns: 1fr;
        }
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


def render_compact_pills(items: list[str], limit: int = 16) -> str:
    shown = items[:limit]
    extra_count = max(0, len(items) - len(shown))
    extra = f"<span class='pill'>+{extra_count} more</span>" if extra_count else ""
    return render_pills(shown) + extra


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


def render_signal_count_grid(counts: dict) -> str:
    ordered_labels = [
        "High Probability Put Sell",
        "Put Sell Candidate",
        "Stalk / Watchlist",
        "Neutral / Wait",
        "Downtrend Risk",
        "Breakdown Risk",
    ]
    tiles = []
    for label in ordered_labels:
        tiles.append(
            "<div class='signal-count-tile'>"
            f"<div class='signal-count-label'>{label}</div>"
            f"<div class='signal-count-value' style='color:{label_color(label)};'>{counts.get(label, 0)}</div>"
            "</div>"
        )
    return "<div class='signal-count-grid'>" + "".join(tiles) + "</div>"


def render_workflow_strip(analysis_ready: bool) -> str:
    steps = [
        ("01", "Run Scan", "Refresh prices, scoring, and the learning profile."),
        ("02", "Check Queue", "Start with Ready Now, then stalk names, then avoid names."),
        ("03", "Deep Dive", "Confirm support, bounce quality, and downside risk."),
        ("04", "Decide", "Sell only when timing and assignment comfort line up."),
    ]
    active_idx = 1 if analysis_ready else 0
    step_html = []
    for idx, (number, title, copy) in enumerate(steps):
        active_class = " workflow-step-active" if idx == active_idx else ""
        step_html.append(
            f"<div class='workflow-step{active_class}'>"
            f"<div class='workflow-step-number'>{number}</div>"
            f"<div class='workflow-step-title'>{title}</div>"
            f"<div class='workflow-step-copy'>{copy}</div>"
            "</div>"
        )
    return "<div class='workflow-strip'><div class='workflow-grid'>" + "".join(step_html) + "</div></div>"


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
                In practice, better candidates usually have <b>Quality around 4+</b>, <b>Entry around -1 or better</b>, <b>Risk around -5 or better</b>, and price that is already testing or reclaiming support in a constructive way.
                <br><br>
                <b>Stalk / Watchlist</b> is for names that may still be acceptable to own on assignment, but are too early, too extended above support, or still waiting on bounce confirmation.
                <br><br>
                <b>High Probability Put Sell</b> is stricter. It usually needs stronger quality, price within a disciplined support zone, evidence that support is firming or bouncing, controlled volatility, and fewer obvious risk flags.
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

    if learning_refreshed:
        load_stock_snapshot.clear()
        load_stock_analysis.clear()

    progress_bar.progress(12, text="Loading shared market benchmark...")
    try:
        benchmark_df = fetch_benchmark_data()
    except Exception:
        benchmark_df = None

    progress_bar.progress(18, text="Downloading watchlist price history in one batch...")
    try:
        stock_data_map = fetch_many_stock_data(tickers)
    except Exception:
        stock_data_map = {}

    progress_bar.progress(22, text="Scoring live watchlist with the latest learning profile...")

    for idx, symbol in enumerate(tickers, start=1):
        progress_pct = 22 + int(((idx - 1) / total) * 63)
        progress_text.caption(f"Analyzing {symbol} ({idx}/{total})...")
        progress_bar.progress(progress_pct, text=f"Running analysis for {symbol}...")

        try:
            results.append(
                summarize_stock(
                    symbol,
                    learning_profile=learning_profile,
                    benchmark_df=benchmark_df,
                    stock_df=stock_data_map.get(symbol),
                )
            )
        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e),
            })

        progress_pct = 22 + int((idx / total) * 63)
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
        <div class="sidebar-copy">Keep the watchlist current, choose the signal view, and rerun the model when price action changes.</div>
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

    sidebar_run_analysis = st.button("Run Analysis", use_container_width=True, type="primary")
    st.caption("Refresh rankings, backtest learning, and deep-dive data.")

    filter_panel = st.container(border=True)
    with filter_panel:
        st.markdown("### Signal Focus")
        st.caption("Limit the ranking and deep dive to one label tier when you want a faster read.")
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
        st.caption("Remove names with the x. The selected name becomes the default deep-dive symbol.")
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
            "<div class='sidebar-tip'>Start with the trade queue, confirm whether anything is actually near support, then open the deep dive only for the names that still look actionable.</div>",
            unsafe_allow_html=True
        )

if sidebar_run_analysis:
    run_full_analysis()

analysis_status_text = "Analysis Ready" if st.session_state.analysis_ready else "Needs Refresh"
analysis_status_color = "#22c55e" if st.session_state.analysis_ready else "#f59e0b"
analysis_status_note = (
    "The latest scoring snapshot is loaded for this watchlist."
    if st.session_state.analysis_ready
    else "Run analysis from the sidebar to populate the trade queue, ranking, and deep dive."
)

desk_header_html = (
    "<div class='desk-header'>"
    "<div class='desk-header-top'>"
    "<div>"
    "<div class='hero-kicker'>Cash-Secured Put Workflow</div>"
    "<div class='desk-title'>Quantitative Put Selling Dashboard</div>"
    "<div class='desk-subtitle'>Scan the watchlist, find names near support, and decide whether the setup is worth selling a put or better left alone.</div>"
    "</div>"
    f"<div class='desk-status-pill' style='color:{analysis_status_color};'>{analysis_status_text}</div>"
    "</div>"
    "<div class='desk-meta-row'>"
    f"<span class='desk-meta-item'>Watchlist: {len(st.session_state.watchlist)} ticker(s)</span>"
    f"<span class='desk-meta-item'>Filter: {signal_filter}</span>"
    "<span class='desk-meta-item'>Chart Default: 3M</span>"
    f"<span class='desk-meta-item'>{analysis_status_note}</span>"
    "</div>"
    "</div>"
)
st.markdown(desk_header_html, unsafe_allow_html=True)
st.markdown(render_workflow_strip(st.session_state.analysis_ready), unsafe_allow_html=True)

if st.session_state.watchlist:
    st.markdown(
        f"""
        <div class='ticker-rail'>
            <div class='ticker-rail-top'>
                <div class='ticker-rail-title'>Active Watchlist</div>
                <div class='ticker-rail-meta'>{len(st.session_state.watchlist)} symbol(s) scanned on refresh</div>
            </div>
            {render_compact_pills(st.session_state.watchlist)}
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

    if not qualified_put_sells.empty:
        queue_title = f"{len(qualified_put_sells)} name(s) are close enough to support to review for a live put-selling decision."
        queue_color = "#22c55e"
    elif not stalk_names.empty:
        queue_title = "Nothing looks trade-ready right now, but there are names worth stalking for a cleaner pullback or bounce."
        queue_color = "#facc15"
    else:
        queue_title = "This screen is mostly defensive right now. Avoid forcing put sales until price behavior improves."
        queue_color = "#ef4444"

    spy_dist_label = f"{spy_dist:.2f}%" if spy_dist is not None and not pd.isna(spy_dist) else "N/A"
    operator_board_html = (
        "<div class='operator-board'>"
        "<div class='operator-grid'>"
        "<div class='market-cardlet'>"
        "<div class='section-eyebrow'>Market Backdrop</div>"
        f"<div class='market-regime-line' style='color:{regime_hex};'>{market_regime} Regime</div>"
        f"<div class='market-copy'>{regime_note}</div>"
        "<div class='context-chip-row'>"
        f"<span class='summary-chip'>SPY Close: {fmt_price(spy_close)}</span>"
        f"<span class='summary-chip'>SPY vs 200 SMA: {spy_dist_label}</span>"
        "</div>"
        "</div>"
        "<div class='priority-card'>"
        "<div class='section-eyebrow'>Today's Decision Board</div>"
        f"{render_signal_count_grid(counts)}"
        f"<div class='priority-title' style='color:{queue_color};'>{queue_title}</div>"
        "<div class='priority-copy'>Work left to right: live setups first, stalk-list names second, avoid-list names last.</div>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(operator_board_html, unsafe_allow_html=True)

    trade_queue_tab, ranking_tab, deep_dive_tab, backtest_tab, guide_tab = st.tabs(
        ["Queue", "Watchlist", "Deep Dive", "Backtest", "Guide"]
    )

    with trade_queue_tab:
        render_section_header(
            "Trade Queue",
            "This is the working board: review live setups first, then the stalk list, then the avoid list.",
            "Workflow",
        )
        st.markdown(
            "<div class='analysis-note'>A clean put sale usually needs two things at the same time: a stock you would still be okay owning and timing that is actually close enough to support to matter.</div>",
            unsafe_allow_html=True,
        )

        render_section_header(
            "Ready Now",
            "These are the names closest to a live put-selling decision right now.",
            "Priority",
        )
        if not qualified_put_sells.empty:
            st.markdown(
                "<div class='table-context'>Read left to right: stock, support, preferred entry, setup state, then conviction.</div>",
                unsafe_allow_html=True,
            )
            ready_display = prepare_display_table(
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
                style_ranked_table(ready_display),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.markdown(
                "<div class='empty-queue'>"
                "<div class='empty-queue-title'>No trade-ready put setups right now</div>"
                "<div class='empty-queue-copy'>That is a useful signal, not a failure. Review the stalk list for names that may become attractive after a cleaner pullback or bounce confirmation.</div>"
                "</div>",
                unsafe_allow_html=True
            )

        stalk_col, avoid_col = st.columns([1.15, 0.95], gap="large")

        with stalk_col:
            render_section_header(
                "Worth Stalking",
                "These names may still be acceptable on assignment, but the entry timing is not ready enough yet.",
                "Pipeline",
            )
            if stalk_names.empty:
                st.info("No stalk-list names are standing out right now.")
            else:
                st.markdown(
                    "<div class='table-context'>These belong on the monitor list until price reaches support more cleanly or the bounce improves.</div>",
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

        with avoid_col:
            render_section_header(
                "Avoid For Now",
                "These names are still too pressured or structurally weak for fresh put exposure.",
                "Risk Review",
            )
            if avoid_names.empty:
                st.info("No high-risk names found.")
            else:
                st.markdown(
                    "<div class='table-context'>This table focuses on the specific reasons the setup is less dependable for put sellers.</div>",
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
            "Full Watchlist Table",
            "This is the full filtered ranking. Trade-ready names stay on top, then stalk names, then the more defensive labels.",
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
            "Inspect one symbol at a time: support map first, bounce behavior second, then confirm with the charts.",
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
                        Start with support and entry status. If that part is weak, the indicators usually should not rescue the trade idea.
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
