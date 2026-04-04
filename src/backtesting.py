from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.analysis import calculate_trade_levels
from src.fetch_data import fetch_stock_data
from src.indicators import add_indicators
from src.regime import classify_market_regime
from src.scoring import score_stock


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_SUMMARY_PATH = DATA_DIR / "backtest_summary.json"
LEARNING_PROFILE_PATH = DATA_DIR / "learning_profile.json"

BACKTEST_PERIOD = "3y"
BACKTEST_INTERVAL = "1d"
BACKTEST_HORIZONS = (5, 10, 20)
MIN_SIGNAL_LOOKBACK = 220
MAX_PROFILE_DELTA = 2


DEFAULT_THRESHOLD_ADJUSTMENTS = {
    "candidate_min_score": 0,
    "candidate_min_quality": 0,
    "candidate_min_entry": 0,
    "candidate_min_risk": 0,
    "high_probability_min_score": 0,
    "high_probability_min_quality": 0,
    "high_probability_min_entry": 0,
    "high_probability_min_risk": 0,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _safe_write_json(path: Path, payload: dict) -> None:
    _ensure_data_dir()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_watchlist(watchlist: list[str] | None) -> list[str]:
    tickers = []
    for item in watchlist or []:
        symbol = str(item).strip().upper()
        if symbol and symbol not in tickers:
            tickers.append(symbol)
    return tickers


def _watchlist_signature(watchlist: list[str] | None) -> str:
    normalized = "|".join(sorted(_normalize_watchlist(watchlist)))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _clamp_int(value, low: int = -MAX_PROFILE_DELTA, high: int = MAX_PROFILE_DELTA) -> int:
    try:
        return max(low, min(high, int(round(float(value)))))
    except Exception:
        return 0


def load_learning_profile() -> dict:
    raw = _safe_read_json(LEARNING_PROFILE_PATH) or {}
    adjustments = DEFAULT_THRESHOLD_ADJUSTMENTS.copy()
    adjustments.update(
        {
            key: _clamp_int(value)
            for key, value in (raw.get("threshold_adjustments") or {}).items()
            if key in adjustments
        }
    )

    return {
        "version": 1,
        "generated_at": raw.get("generated_at"),
        "watchlist_signature": raw.get("watchlist_signature"),
        "threshold_adjustments": adjustments,
        "notes": raw.get("notes") or [],
        "source_summary": raw.get("source_summary") or {},
    }


def save_learning_profile(profile: dict) -> None:
    payload = {
        "version": 1,
        "generated_at": profile.get("generated_at") or _utc_now_iso(),
        "watchlist_signature": profile.get("watchlist_signature"),
        "threshold_adjustments": {
            key: _clamp_int(value)
            for key, value in (profile.get("threshold_adjustments") or {}).items()
            if key in DEFAULT_THRESHOLD_ADJUSTMENTS
        },
        "notes": profile.get("notes") or [],
        "source_summary": profile.get("source_summary") or {},
    }
    _safe_write_json(LEARNING_PROFILE_PATH, payload)


def learning_signature(profile: dict | None = None) -> str:
    active = profile or load_learning_profile()
    payload = json.dumps(active.get("threshold_adjustments") or {}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_backtest_summary() -> dict | None:
    return _safe_read_json(BACKTEST_SUMMARY_PATH)


def save_backtest_summary(summary: dict) -> None:
    _safe_write_json(BACKTEST_SUMMARY_PATH, summary)


def _summary_is_fresh(summary: dict | None, watchlist: list[str], max_age_hours: int = 12) -> bool:
    if not summary:
        return False

    if summary.get("watchlist_signature") != _watchlist_signature(watchlist):
        return False

    generated_at = summary.get("generated_at")
    if not generated_at:
        return False

    try:
        generated = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except Exception:
        return False

    return datetime.now(timezone.utc) - generated <= timedelta(hours=max_age_hours)


def _pct_change(current, base):
    if current is None or base is None or base == 0:
        return None
    return ((current - base) / base) * 100


def _future_metric(df: pd.DataFrame, start_idx: int, offset: int, column: str = "close"):
    target_idx = start_idx + offset
    if target_idx >= len(df):
        return None
    return float(df.iloc[target_idx][column])


def _allowed_drawdown_pct(signal_row: pd.Series) -> float:
    atr_pct = pd.to_numeric(signal_row.get("atr_pct"), errors="coerce")
    realized_vol_20 = pd.to_numeric(signal_row.get("realized_vol_20"), errors="coerce")

    dynamic_limit = 5.0
    if pd.notna(atr_pct):
        dynamic_limit = max(dynamic_limit, min(atr_pct * 1.35, 7.0))
    if pd.notna(realized_vol_20):
        dynamic_limit = max(dynamic_limit, min(float(realized_vol_20) * 10.0, 7.5))

    return -round(dynamic_limit, 2)


def _evaluate_signal_outcome(df: pd.DataFrame, signal_idx: int, trade_levels: dict) -> dict | None:
    max_horizon = max(BACKTEST_HORIZONS)
    if signal_idx + max_horizon >= len(df):
        return None

    signal_row = df.iloc[signal_idx]
    entry_price = float(df.iloc[signal_idx]["close"])
    future_slice = df.iloc[signal_idx + 1: signal_idx + 1 + max_horizon].copy()
    if future_slice.empty:
        return None

    future_low = float(future_slice["low"].min())
    future_high = float(future_slice["high"].max())
    ret_5d = _pct_change(_future_metric(df, signal_idx, 5), entry_price)
    ret_10d = _pct_change(_future_metric(df, signal_idx, 10), entry_price)
    ret_20d = _pct_change(_future_metric(df, signal_idx, 20), entry_price)
    max_drawdown_pct = _pct_change(future_low, entry_price)
    max_runup_pct = _pct_change(future_high, entry_price)

    primary_support = trade_levels.get("primary_support")
    if primary_support is not None:
        support_held = future_low >= float(primary_support) * 0.985
    else:
        support_held = max_drawdown_pct is not None and max_drawdown_pct >= -6

    allowed_drawdown_pct = _allowed_drawdown_pct(signal_row)
    drawdown_controlled = max_drawdown_pct is not None and max_drawdown_pct >= allowed_drawdown_pct
    stabilized_after_entry = (
        (ret_10d is None or ret_10d >= -1.5)
        and (ret_20d is None or ret_20d >= -3.0)
    )
    favorable_follow_through = any(
        condition
        for condition in (
            ret_5d is not None and ret_5d >= 0,
            ret_10d is not None and ret_10d >= 0,
            ret_20d is not None and ret_20d >= 0.5,
            max_runup_pct is not None and max_runup_pct >= 1.75,
        )
    )
    success = bool(
        support_held
        and drawdown_controlled
        and stabilized_after_entry
        and favorable_follow_through
    )

    return {
        "entry_price": round(entry_price, 2),
        "ret_5d_pct": round(ret_5d, 2) if ret_5d is not None else None,
        "ret_10d_pct": round(ret_10d, 2) if ret_10d is not None else None,
        "ret_20d_pct": round(ret_20d, 2) if ret_20d is not None else None,
        "max_drawdown_pct": round(max_drawdown_pct, 2) if max_drawdown_pct is not None else None,
        "max_runup_pct": round(max_runup_pct, 2) if max_runup_pct is not None else None,
        "allowed_drawdown_pct": allowed_drawdown_pct,
        "support_held": support_held,
        "success": success,
    }


def _aggregate_label_records(records: list[dict]) -> dict:
    labels = {}
    if not records:
        return labels

    df = pd.DataFrame(records)
    for label, group in df.groupby("label"):
        labels[label] = {
            "signal_count": int(len(group)),
            "success_rate": round(float(group["success"].mean() * 100), 1),
            "support_hold_rate": round(float(group["support_held"].mean() * 100), 1),
            "avg_5d_return_pct": round(float(group["ret_5d_pct"].dropna().mean()), 2) if group["ret_5d_pct"].notna().any() else None,
            "avg_10d_return_pct": round(float(group["ret_10d_pct"].dropna().mean()), 2) if group["ret_10d_pct"].notna().any() else None,
            "avg_20d_return_pct": round(float(group["ret_20d_pct"].dropna().mean()), 2) if group["ret_20d_pct"].notna().any() else None,
            "avg_max_drawdown_pct": round(float(group["max_drawdown_pct"].dropna().mean()), 2) if group["max_drawdown_pct"].notna().any() else None,
            "avg_max_runup_pct": round(float(group["max_runup_pct"].dropna().mean()), 2) if group["max_runup_pct"].notna().any() else None,
            "avg_score": round(float(group["score"].mean()), 2),
        }
    return labels


def backtest_symbol(symbol: str, learning_profile: dict | None = None) -> dict:
    payload = fetch_stock_data(
        symbol,
        include_benchmark=True,
        period=BACKTEST_PERIOD,
        interval=BACKTEST_INTERVAL,
    )
    stock_df = payload["stock_df"]
    benchmark_df = payload.get("benchmark_df")
    stock_df = add_indicators(stock_df, benchmark_df=benchmark_df)

    if len(stock_df) <= MIN_SIGNAL_LOOKBACK + max(BACKTEST_HORIZONS):
        return {
            "symbol": symbol,
            "error": f"Not enough history for backtesting after indicator warm-up. Got {len(stock_df)} rows.",
        }

    records = []
    start_idx = max(MIN_SIGNAL_LOOKBACK, 200)
    end_idx = len(stock_df) - max(BACKTEST_HORIZONS)

    for idx in range(start_idx, end_idx):
        history_df = stock_df.iloc[: idx + 1].copy()

        benchmark_history = None
        if benchmark_df is not None and not benchmark_df.empty:
            latest_ts = history_df["timestamp"].iloc[-1]
            benchmark_history = benchmark_df[benchmark_df["timestamp"] <= latest_ts].copy()

        regime_data = classify_market_regime(benchmark_history)
        trade_levels = calculate_trade_levels(history_df)

        latest = history_df.iloc[-1].copy()
        for key, value in trade_levels.items():
            latest[key] = value
        for key, value in regime_data.items():
            latest[key] = value

        signal = score_stock(latest, learning_profile=learning_profile)
        outcome = _evaluate_signal_outcome(stock_df, idx, trade_levels)
        if outcome is None:
            continue

        records.append(
            {
                "symbol": symbol,
                "timestamp": str(history_df["timestamp"].iloc[-1]),
                "label": signal["label"],
                "score": signal["score"],
                "quality_score": signal["quality_score"],
                "entry_score": signal["entry_score"],
                "risk_score": signal["risk_score"],
                **outcome,
            }
        )

    if not records:
        return {
            "symbol": symbol,
            "error": "No historical signals were available for backtest evaluation.",
        }

    label_summary = _aggregate_label_records(records)
    candidate_like = [
        record for record in records
        if record["label"] in {"High Probability Put Sell", "Put Sell Candidate"}
    ]

    candidate_like_summary = _aggregate_label_records(candidate_like)
    candidate_like_key = "candidate_like"
    if candidate_like:
        candidate_like_df = pd.DataFrame(candidate_like)
        candidate_like_summary[candidate_like_key] = {
            "signal_count": int(len(candidate_like_df)),
            "success_rate": round(float(candidate_like_df["success"].mean() * 100), 1),
            "avg_20d_return_pct": round(float(candidate_like_df["ret_20d_pct"].dropna().mean()), 2) if candidate_like_df["ret_20d_pct"].notna().any() else None,
            "avg_max_drawdown_pct": round(float(candidate_like_df["max_drawdown_pct"].dropna().mean()), 2) if candidate_like_df["max_drawdown_pct"].notna().any() else None,
        }

    return {
        "symbol": symbol,
        "signal_count": len(records),
        "labels": label_summary,
        "candidate_like": candidate_like_summary.get(candidate_like_key),
    }


def derive_learning_profile(backtest_summary: dict, watchlist: list[str]) -> dict:
    adjustments = DEFAULT_THRESHOLD_ADJUSTMENTS.copy()
    notes = []

    labels = backtest_summary.get("labels", {})
    candidate_stats = labels.get("Put Sell Candidate", {})
    high_stats = labels.get("High Probability Put Sell", {})

    candidate_count = int(candidate_stats.get("signal_count", 0) or 0)
    candidate_success = candidate_stats.get("success_rate")
    candidate_drawdown = candidate_stats.get("avg_max_drawdown_pct")
    candidate_support_hold = candidate_stats.get("support_hold_rate")

    high_count = int(high_stats.get("signal_count", 0) or 0)
    high_success = high_stats.get("success_rate")
    high_drawdown = high_stats.get("avg_max_drawdown_pct")

    if candidate_count >= 8:
        if (candidate_success is not None and candidate_success < 50) or (
            candidate_drawdown is not None and candidate_drawdown < -6.5
        ):
            adjustments["candidate_min_score"] += 1
            adjustments["candidate_min_quality"] += 1
            notes.append("Candidate threshold tightened because recent candidate outcomes were too weak.")
        elif (candidate_success is not None and candidate_success >= 68) and candidate_count < 20:
            adjustments["candidate_min_score"] -= 1
            notes.append("Candidate threshold loosened slightly because recent candidate outcomes held up well but signal count stayed light.")

        if candidate_support_hold is not None and candidate_support_hold < 55:
            adjustments["candidate_min_entry"] += 1
            adjustments["candidate_min_risk"] += 1
            notes.append("Candidate entry and risk floors tightened because support was not holding often enough.")

    if high_count >= 6:
        if (high_success is not None and high_success < 58) or (
            high_drawdown is not None and high_drawdown < -5.5
        ):
            adjustments["high_probability_min_score"] += 1
            adjustments["high_probability_min_quality"] += 1
            adjustments["high_probability_min_entry"] += 1
            notes.append("High-conviction threshold tightened because historical high-probability setups were not clean enough.")
        elif (high_success is not None and high_success >= 72) and high_count < 16:
            adjustments["high_probability_min_score"] -= 1
            notes.append("High-conviction threshold loosened slightly because recent high-probability outcomes were strong but sparse.")

    adjustments = {key: _clamp_int(value) for key, value in adjustments.items()}

    return {
        "version": 1,
        "generated_at": _utc_now_iso(),
        "watchlist_signature": _watchlist_signature(watchlist),
        "threshold_adjustments": adjustments,
        "notes": notes or ["No threshold adjustments were needed from the latest backtest cycle."],
        "source_summary": {
            "candidate_signal_count": candidate_count,
            "candidate_success_rate": candidate_success,
            "high_probability_signal_count": high_count,
            "high_probability_success_rate": high_success,
        },
    }


def _aggregate_watchlist_backtest(symbol_results: list[dict], watchlist: list[str]) -> dict:
    valid_results = [result for result in symbol_results if not result.get("error")]
    label_records = []
    for result in valid_results:
        for label, stats in (result.get("labels") or {}).items():
            label_records.append(
                {
                    "label": label,
                    "signal_count": stats.get("signal_count", 0),
                    "success_rate": stats.get("success_rate"),
                    "support_hold_rate": stats.get("support_hold_rate"),
                    "avg_5d_return_pct": stats.get("avg_5d_return_pct"),
                    "avg_10d_return_pct": stats.get("avg_10d_return_pct"),
                    "avg_20d_return_pct": stats.get("avg_20d_return_pct"),
                    "avg_max_drawdown_pct": stats.get("avg_max_drawdown_pct"),
                    "avg_max_runup_pct": stats.get("avg_max_runup_pct"),
                    "avg_score": stats.get("avg_score"),
                }
            )

    labels_summary = {}
    if label_records:
        label_df = pd.DataFrame(label_records)
        for label, group in label_df.groupby("label"):
            weights = pd.to_numeric(group["signal_count"], errors="coerce").fillna(0)
            total_weight = float(weights.sum())
            if total_weight <= 0:
                continue

            def weighted_avg(column: str):
                series = pd.to_numeric(group[column], errors="coerce")
                valid = series.notna() & (weights > 0)
                if not valid.any():
                    return None
                return round(float((series[valid] * weights[valid]).sum() / weights[valid].sum()), 2)

            labels_summary[label] = {
                "signal_count": int(total_weight),
                "success_rate": weighted_avg("success_rate"),
                "support_hold_rate": weighted_avg("support_hold_rate"),
                "avg_5d_return_pct": weighted_avg("avg_5d_return_pct"),
                "avg_10d_return_pct": weighted_avg("avg_10d_return_pct"),
                "avg_20d_return_pct": weighted_avg("avg_20d_return_pct"),
                "avg_max_drawdown_pct": weighted_avg("avg_max_drawdown_pct"),
                "avg_max_runup_pct": weighted_avg("avg_max_runup_pct"),
                "avg_score": weighted_avg("avg_score"),
            }

    candidate_like_results = [result.get("candidate_like") for result in valid_results if result.get("candidate_like")]
    candidate_like_summary = None
    if candidate_like_results:
        candidate_like_df = pd.DataFrame(candidate_like_results)
        weights = pd.to_numeric(candidate_like_df["signal_count"], errors="coerce").fillna(0)
        total_weight = float(weights.sum())
        if total_weight > 0:
            candidate_like_summary = {
                "signal_count": int(total_weight),
                "success_rate": round(float((candidate_like_df["success_rate"] * weights).sum() / total_weight), 2),
                "avg_20d_return_pct": round(float((candidate_like_df["avg_20d_return_pct"].fillna(0) * weights).sum() / total_weight), 2),
                "avg_max_drawdown_pct": round(float((candidate_like_df["avg_max_drawdown_pct"].fillna(0) * weights).sum() / total_weight), 2),
            }

    return {
        "version": 1,
        "generated_at": _utc_now_iso(),
        "watchlist_signature": _watchlist_signature(watchlist),
        "watchlist_size": len(_normalize_watchlist(watchlist)),
        "candidate_like": candidate_like_summary,
        "labels": labels_summary,
        "symbols": symbol_results,
        "methodology": {
            "history_period": BACKTEST_PERIOD,
            "horizons_days": list(BACKTEST_HORIZONS),
            "warmup_rows": MIN_SIGNAL_LOOKBACK,
            "notes": [
                "This is an underlying-behavior proxy backtest, not real options P/L.",
                "A historical signal counts as successful when support holds, drawdown stays controlled, and price stabilizes or bounces enough to resemble a workable short-put window.",
            ],
        },
    }


def run_automatic_learning_cycle(watchlist: list[str], max_age_hours: int = 12) -> tuple[dict, dict | None, bool]:
    normalized_watchlist = _normalize_watchlist(watchlist)
    if not normalized_watchlist:
        profile = load_learning_profile()
        return profile, load_backtest_summary(), False

    existing_summary = load_backtest_summary()
    if _summary_is_fresh(existing_summary, normalized_watchlist, max_age_hours=max_age_hours):
        return load_learning_profile(), existing_summary, False

    current_profile = load_learning_profile()
    symbol_results = []
    for symbol in normalized_watchlist:
        try:
            symbol_results.append(backtest_symbol(symbol, learning_profile=current_profile))
        except Exception as exc:
            symbol_results.append({"symbol": symbol, "error": str(exc)})

    backtest_summary = _aggregate_watchlist_backtest(symbol_results, normalized_watchlist)
    learning_profile = derive_learning_profile(backtest_summary, normalized_watchlist)

    save_backtest_summary(backtest_summary)
    save_learning_profile(learning_profile)

    return learning_profile, backtest_summary, True
