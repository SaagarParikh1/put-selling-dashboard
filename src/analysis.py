from src.fetch_data import fetch_stock_data
from src.indicators import add_indicators
from src.scoring import score_stock
from src.regime import classify_market_regime


def _is_valid_number(value) -> bool:
    return value is not None and value == value


def _safe_float(value):
    return float(value) if _is_valid_number(value) else None


def _safe_bool(value):
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return bool(value)


def _pct_diff(a, b):
    if a is None or b is None or b == 0:
        return None
    return ((a - b) / b) * 100


def _merge_nearby_levels(levels, threshold_pct=1.25):
    """
    Merge nearby levels into zones so highly similar support references
    are treated as one clustered support area rather than separate levels.
    """
    cleaned = sorted([float(x) for x in levels if x is not None and x > 0])
    if not cleaned:
        return []

    merged = []
    current_cluster = [cleaned[0]]

    for level in cleaned[1:]:
        cluster_avg = sum(current_cluster) / len(current_cluster)
        pct_gap = abs(level - cluster_avg) / cluster_avg * 100 if cluster_avg else 999

        if pct_gap <= threshold_pct:
            current_cluster.append(level)
        else:
            merged.append(current_cluster)
            current_cluster = [level]

    merged.append(current_cluster)
    return merged


def _score_support_cluster(
    cluster,
    close,
    recent_20_low,
    recent_50_low,
    sma_50,
    sma_200,
    bb_low,
    rolling_20_low,
    rolling_50_low,
):
    """
    Score a support cluster based on:
    - proximity to current price
    - confluence with recent lows / moving averages / Bollinger support
    - preference for levels slightly below price
    """
    center = round(sum(cluster) / len(cluster), 2)
    score = 0
    basis = []

    # Confluence from cluster size
    if len(cluster) >= 2:
        score += 2
        basis.append("multi-level confluence")
    else:
        score += 1

    dist_pct = _pct_diff(close, center)

    if dist_pct is not None:
        if 0 <= dist_pct <= 3:
            score += 4
            basis.append("close below price")
        elif 3 < dist_pct <= 8:
            score += 2
            basis.append("reasonable distance below price")
        elif dist_pct < 0:
            score -= 3
            basis.append("price below level")
        elif dist_pct > 8:
            score -= 1
            basis.append("support far below price")

    def near(level, label, threshold=1.5):
        nonlocal score, basis
        if level is None:
            return
        gap = abs(center - level) / level * 100 if level else 999
        if gap <= threshold:
            score += 2
            basis.append(label)

    near(recent_20_low, "20-day low")
    near(recent_50_low, "50-day low")
    near(rolling_20_low, "rolling 20-day low")
    near(rolling_50_low, "rolling 50-day low")
    near(sma_50, "50-day moving average")
    near(sma_200, "200-day moving average", threshold=2.0)
    near(bb_low, "lower Bollinger Band", threshold=1.75)

    return {
        "level": center,
        "score": score,
        "basis": basis,
        "confluence_count": len(basis),
        "distance_pct": round(dist_pct, 2) if dist_pct is not None else None,
    }


def calculate_trade_levels(df):
    """
    Build support and entry levels with more structure-aware logic.

    Output fields:
    - primary_support
    - secondary_support
    - recommended_entry
    - entry_zone_low
    - entry_zone_high
    - entry_status
    - support_strength
    - support_basis
    - support_distance_pct
    - bounce_score
    - bounce_signal
    - bounce_confirmed
    """
    if df.empty:
        return {
            "primary_support": None,
            "secondary_support": None,
            "recommended_entry": None,
            "entry_zone_low": None,
            "entry_zone_high": None,
            "entry_status": "N/A",
            "support_strength": None,
            "support_basis": "",
            "support_confluence_count": None,
            "support_distance_pct": None,
            "bounce_score": None,
            "bounce_signal": "N/A",
            "bounce_confirmed": False,
        }

    latest = df.iloc[-1]
    close = _safe_float(latest.get("close"))

    if close is None:
        return {
            "primary_support": None,
            "secondary_support": None,
            "recommended_entry": None,
            "entry_zone_low": None,
            "entry_zone_high": None,
            "entry_status": "N/A",
            "support_strength": None,
            "support_basis": "",
            "support_confluence_count": None,
            "support_distance_pct": None,
            "bounce_score": None,
            "bounce_signal": "N/A",
            "bounce_confirmed": False,
        }

    recent_20_low = float(df["low"].tail(20).min()) if len(df) >= 20 else float(df["low"].min())
    recent_50_low = float(df["low"].tail(50).min()) if len(df) >= 50 else float(df["low"].min())

    sma_50 = _safe_float(latest.get("sma_50"))
    sma_200 = _safe_float(latest.get("sma_200"))
    bb_low = _safe_float(latest.get("bb_low"))
    rolling_20_low = _safe_float(latest.get("rolling_20_low"))
    rolling_50_low = _safe_float(latest.get("rolling_50_low"))
    atr_14 = _safe_float(latest.get("atr_14"))

    candidate_supports = [
        recent_20_low,
        recent_50_low,
        rolling_20_low,
        rolling_50_low,
        sma_50,
        sma_200,
        bb_low,
    ]

    candidate_supports = [x for x in candidate_supports if x is not None and x > 0]

    if not candidate_supports:
        return {
            "primary_support": None,
            "secondary_support": None,
            "recommended_entry": None,
            "entry_zone_low": None,
            "entry_zone_high": None,
            "entry_status": "N/A",
            "support_strength": None,
            "support_basis": "",
            "support_confluence_count": None,
            "support_distance_pct": None,
            "bounce_score": None,
            "bounce_signal": "N/A",
            "bounce_confirmed": False,
        }

    clusters = _merge_nearby_levels(candidate_supports, threshold_pct=1.25)

    scored_clusters = []
    for cluster in clusters:
        scored = _score_support_cluster(
            cluster=cluster,
            close=close,
            recent_20_low=recent_20_low,
            recent_50_low=recent_50_low,
            sma_50=sma_50,
            sma_200=sma_200,
            bb_low=bb_low,
            rolling_20_low=rolling_20_low,
            rolling_50_low=rolling_50_low,
        )
        scored_clusters.append(scored)

    scored_clusters.sort(
        key=lambda x: (
            x["score"],
            -999 if x["distance_pct"] is None else -abs(x["distance_pct"]),
        ),
        reverse=True,
    )

    primary = scored_clusters[0] if scored_clusters else None
    secondary = scored_clusters[1] if len(scored_clusters) > 1 else None

    primary_support = primary["level"] if primary else None
    secondary_support = secondary["level"] if secondary else None

    recommended_entry = None
    entry_zone_low = None
    entry_zone_high = None
    entry_status = "N/A"
    support_strength = primary["score"] if primary else None
    support_basis = ", ".join(primary["basis"]) if primary else ""
    support_confluence_count = primary["confluence_count"] if primary else None
    support_distance_pct = primary["distance_pct"] if primary else None
    bounce_score = None
    bounce_signal = "N/A"
    bounce_confirmed = False

    if primary_support is not None:
        # Use ATR slightly when available so entry zone breathes with volatility
        if atr_14 is not None and atr_14 > 0:
            lower_buffer = min(atr_14 * 0.35, primary_support * 0.01)
            upper_buffer = min(atr_14 * 0.75, primary_support * 0.02)
            entry_zone_low = round(primary_support - lower_buffer, 2)
            entry_zone_high = round(primary_support + upper_buffer, 2)
            recommended_entry = round(primary_support + min(atr_14 * 0.25, primary_support * 0.005), 2)
        else:
            entry_zone_low = round(primary_support * 0.9925, 2)
            entry_zone_high = round(primary_support * 1.02, 2)
            recommended_entry = round(primary_support * 1.005, 2)

        if entry_zone_low <= close <= entry_zone_high:
            entry_status = "In entry zone"
        elif close > entry_zone_high:
            if support_distance_pct is not None and support_distance_pct <= 4:
                entry_status = "Watch for stabilization"
            else:
                entry_status = "Wait for pullback"
        elif close < entry_zone_low:
            if close >= primary_support * 0.985:
                entry_status = "Support under pressure"
            else:
                entry_status = "Below support / caution"

        latest_close_loc = _safe_float(latest.get("close_location"))
        latest_close_change_pct = _safe_float(latest.get("close_change_pct"))
        latest_green = _safe_bool(latest.get("green_candle"))
        latest_reversal = _safe_bool(latest.get("bullish_reversal_day"))
        latest_volume_ratio = _safe_float(latest.get("volume_ratio_20"))
        recent_lows = df["low"].tail(4)
        recent_support_test = recent_lows.min() <= primary_support * 1.01 if not recent_lows.empty else False
        reclaim_from_below = False

        if len(df) >= 2:
            prev = df.iloc[-2]
            prev_close = _safe_float(prev.get("close"))
            prev_low = _safe_float(prev.get("low"))
            if prev_close is not None and prev_low is not None:
                reclaim_from_below = (
                    (prev_close < primary_support or prev_low < primary_support)
                    and close >= primary_support
                )

        bounce_score = 0
        if recent_support_test:
            bounce_score += 2
        if latest_green:
            bounce_score += 1
        if latest_reversal:
            bounce_score += 2
        if latest_close_change_pct is not None and latest_close_change_pct > 0:
            bounce_score += 1
        if latest_close_loc is not None and latest_close_loc >= 0.6:
            bounce_score += 1
        if latest_volume_ratio is not None and latest_volume_ratio >= 1.05 and latest_close_change_pct is not None and latest_close_change_pct > 0:
            bounce_score += 1
        if reclaim_from_below:
            bounce_score += 2

        if close < primary_support * 0.99:
            bounce_signal = "Broken below support"
            bounce_confirmed = False
        elif recent_support_test and bounce_score >= 6:
            bounce_signal = "Confirmed bounce"
            bounce_confirmed = True
        elif recent_support_test and bounce_score >= 3:
            bounce_signal = "Early bounce"
        elif recent_support_test:
            bounce_signal = "At support, no bounce yet"
        elif support_distance_pct is not None and support_distance_pct <= 5:
            bounce_signal = "Near support"
        else:
            bounce_signal = "No bounce setup"

    return {
        "primary_support": round(primary_support, 2) if primary_support is not None else None,
        "secondary_support": round(secondary_support, 2) if secondary_support is not None else None,
        "recommended_entry": round(recommended_entry, 2) if recommended_entry is not None else None,
        "entry_zone_low": entry_zone_low,
        "entry_zone_high": entry_zone_high,
        "entry_status": entry_status,
        "support_strength": support_strength,
        "support_basis": support_basis,
        "support_confluence_count": support_confluence_count,
        "support_distance_pct": support_distance_pct,
        "bounce_score": bounce_score,
        "bounce_signal": bounce_signal,
        "bounce_confirmed": bounce_confirmed,
    }


def analyze_stock(symbol: str, learning_profile: dict | None = None):
    payload = fetch_stock_data(symbol, include_benchmark=True)

    df = payload["stock_df"]
    benchmark_df = payload.get("benchmark_df")

    df = add_indicators(df, benchmark_df=benchmark_df)

    if df.empty:
        raise ValueError(f"{symbol}: No valid stock data returned.")

    regime_data = classify_market_regime(benchmark_df)
    trade_levels = calculate_trade_levels(df)

    latest = df.iloc[-1].copy()

    for key, value in trade_levels.items():
        latest[key] = value

    for key, value in regime_data.items():
        latest[key] = value

    signal = score_stock(latest, learning_profile=learning_profile)

    return df, signal, trade_levels, regime_data


def summarize_stock(symbol: str, learning_profile: dict | None = None) -> dict:
    df, signal, trade_levels, regime_data = analyze_stock(symbol, learning_profile=learning_profile)
    latest = df.iloc[-1]

    return {
        "symbol": symbol,
        "price": round(latest["close"], 2),

        "rsi": round(latest["rsi_14"], 2) if _is_valid_number(latest.get("rsi_14")) else None,
        "adx": round(latest["adx"], 2) if _is_valid_number(latest.get("adx")) else None,
        "cmf_20": round(latest["cmf_20"], 3) if _is_valid_number(latest.get("cmf_20")) else None,

        "atr_14": round(latest["atr_14"], 2) if _is_valid_number(latest.get("atr_14")) else None,
        "atr_pct": round(latest["atr_pct"], 2) if _is_valid_number(latest.get("atr_pct")) else None,
        "realized_vol_20": round(latest["realized_vol_20"], 3) if _is_valid_number(latest.get("realized_vol_20")) else None,

        "rs_20": round(latest["rs_20"], 2) if _is_valid_number(latest.get("rs_20")) else None,
        "rs_60": round(latest["rs_60"], 2) if _is_valid_number(latest.get("rs_60")) else None,

        "dist_sma50_pct": round(latest["dist_sma50_pct"], 2) if _is_valid_number(latest.get("dist_sma50_pct")) else None,
        "dist_sma200_pct": round(latest["dist_sma200_pct"], 2) if _is_valid_number(latest.get("dist_sma200_pct")) else None,
        "pullback_from_126d_high_pct": round(latest["pullback_from_126d_high_pct"], 2) if _is_valid_number(latest.get("pullback_from_126d_high_pct")) else None,
        "pullback_from_252d_high_pct": round(latest["pullback_from_252d_high_pct"], 2) if _is_valid_number(latest.get("pullback_from_252d_high_pct")) else None,

        "volume_sma_20": round(latest["volume_sma_20"], 0) if _is_valid_number(latest.get("volume_sma_20")) else None,
        "avg_dollar_volume_20": round(latest["avg_dollar_volume_20"], 0) if _is_valid_number(latest.get("avg_dollar_volume_20")) else None,
        "liquidity_ok": bool(latest["liquidity_ok"]) if latest.get("liquidity_ok") == latest.get("liquidity_ok") else None,

        "market_regime": regime_data.get("market_regime"),
        "market_regime_score": regime_data.get("market_regime_score"),
        "spy_close": regime_data.get("spy_close"),
        "spy_sma_50": regime_data.get("spy_sma_50"),
        "spy_sma_200": regime_data.get("spy_sma_200"),
        "spy_dist_sma200_pct": regime_data.get("spy_dist_sma200_pct"),

        "primary_support": trade_levels["primary_support"],
        "secondary_support": trade_levels["secondary_support"],
        "recommended_entry": trade_levels["recommended_entry"],
        "entry_zone_low": trade_levels["entry_zone_low"],
        "entry_zone_high": trade_levels["entry_zone_high"],
        "entry_status": trade_levels["entry_status"],
        "support_strength": trade_levels["support_strength"],
        "support_basis": trade_levels["support_basis"],
        "support_confluence_count": trade_levels["support_confluence_count"],
        "support_distance_pct": trade_levels["support_distance_pct"],
        "bounce_score": trade_levels["bounce_score"],
        "bounce_signal": trade_levels["bounce_signal"],
        "bounce_confirmed": trade_levels["bounce_confirmed"],

        "score": signal["score"],
        "label": signal["label"],
        "confidence": signal["confidence"],
        "trend_score": signal["trend_score"],
        "pullback_score": signal["pullback_score"],
        "support_score": signal["support_score"],
        "flow_score": signal["flow_score"],
        "quality_score": signal.get("quality_score"),
        "entry_score": signal.get("entry_score"),
        "risk_score": signal.get("risk_score"),
        "candidate_blockers": " | ".join(signal.get("candidate_blockers", [])),

        "reasons": " | ".join(signal["reasons"]),
    }


def analyze_watchlist(watchlist: list[str] | None = None):
    tickers = watchlist or []
    results = []

    for symbol in tickers:
        try:
            results.append(summarize_stock(symbol))

        except Exception as e:
            results.append({
                "symbol": symbol,
                "error": str(e),
            })

    return results
