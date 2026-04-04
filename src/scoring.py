import math
import pandas as pd


def _is_valid(x) -> bool:
    return x is not None and not pd.isna(x) and not (isinstance(x, float) and math.isnan(x))


def _safe_get(row: pd.Series, key: str, default=None):
    return row[key] if key in row and _is_valid(row[key]) else default


def _pct_diff(a, b):
    if not _is_valid(a) or not _is_valid(b) or b == 0:
        return None
    return ((a - b) / b) * 100


def _profile_adjustment(profile: dict | None, key: str, low: int = -2, high: int = 2) -> int:
    if not profile:
        return 0

    adjustments = profile.get("threshold_adjustments") or {}
    try:
        value = int(round(float(adjustments.get(key, 0))))
    except Exception:
        return 0

    return max(low, min(high, value))


def score_stock(latest_row: pd.Series, learning_profile: dict | None = None) -> dict:
    """
    Score a stock specifically for cash-secured put selling.

    Philosophy:
    - Favor strong underlying trend / regime
    - Favor controlled pullbacks into support
    - Penalize breakdown behavior heavily
    - Reward positive money flow / healthy participation
    - Prefer names that are attractive to own if assigned
    - Use hard-fail gates for clearly broken setups

    Returns:
        {
            score,
            label,
            confidence,
            trend_score,
            pullback_score,
            support_score,
            flow_score,
            reasons,
            quality_score,
            entry_score,
            risk_score
        }
    """

    # -----------------------------
    # Pull core fields
    # -----------------------------
    close = _safe_get(latest_row, "close")
    rsi = _safe_get(latest_row, "rsi_14")
    macd = _safe_get(latest_row, "macd")
    macd_signal = _safe_get(latest_row, "macd_signal")
    macd_hist = _safe_get(latest_row, "macd_hist")
    ema_9 = _safe_get(latest_row, "ema_9")
    ema_21 = _safe_get(latest_row, "ema_21")
    sma_20 = _safe_get(latest_row, "sma_20")
    sma_50 = _safe_get(latest_row, "sma_50")
    sma_200 = _safe_get(latest_row, "sma_200")
    adx = _safe_get(latest_row, "adx")
    adx_pos = _safe_get(latest_row, "adx_pos")
    adx_neg = _safe_get(latest_row, "adx_neg")
    bb_low = _safe_get(latest_row, "bb_low")
    cmf = _safe_get(latest_row, "cmf_20")
    dist_sma50_pct = _safe_get(latest_row, "dist_sma50_pct")
    dist_sma200_pct = _safe_get(latest_row, "dist_sma200_pct")

    market_regime = _safe_get(latest_row, "market_regime", "Unknown")
    market_regime_score = _safe_get(latest_row, "market_regime_score", 0)

    atr_14 = _safe_get(latest_row, "atr_14")
    atr_pct = _safe_get(latest_row, "atr_pct")
    rs_20 = _safe_get(latest_row, "rs_20")
    rs_60 = _safe_get(latest_row, "rs_60")
    liquidity_ok = _safe_get(latest_row, "liquidity_ok")
    sma_50_slope_10 = _safe_get(latest_row, "sma_50_slope_10")
    sma_200_slope_20 = _safe_get(latest_row, "sma_200_slope_20")
    pullback_from_20d_high_pct = _safe_get(latest_row, "pullback_from_20d_high_pct")
    pullback_from_50d_high_pct = _safe_get(latest_row, "pullback_from_50d_high_pct")
    pullback_from_126d_high_pct = _safe_get(latest_row, "pullback_from_126d_high_pct")
    pullback_from_252d_high_pct = _safe_get(latest_row, "pullback_from_252d_high_pct")
    realized_vol_20 = _safe_get(latest_row, "realized_vol_20")
    downside_vol_ratio_20 = _safe_get(latest_row, "downside_vol_ratio_20")
    down_volume_ratio_20 = _safe_get(latest_row, "down_volume_ratio_20")
    volume_ratio_20 = _safe_get(latest_row, "volume_ratio_20")
    # Optional newer support / entry fields
    primary_support = _safe_get(latest_row, "primary_support")
    secondary_support = _safe_get(latest_row, "secondary_support")
    recommended_entry = _safe_get(latest_row, "recommended_entry")
    entry_status = _safe_get(latest_row, "entry_status", "")
    bounce_score = _safe_get(latest_row, "bounce_score")
    bounce_signal = _safe_get(latest_row, "bounce_signal", "")
    bounce_confirmed = _safe_get(latest_row, "bounce_confirmed", False)
    support_strength = _safe_get(latest_row, "support_strength")
    support_confluence_count = _safe_get(latest_row, "support_confluence_count")
    support_distance_pct = _safe_get(latest_row, "support_distance_pct")

    # -----------------------------
    # Initialize score buckets
    # -----------------------------
    score = 0

    # Legacy output buckets for compatibility
    trend_score = 0
    pullback_score = 0
    support_score = 0
    flow_score = 0

    # New internal buckets
    quality_score = 0
    entry_score = 0
    risk_score = 0
    reasons = []

    # -----------------------------
    # Regime state
    # -----------------------------
    below_50 = _is_valid(close) and _is_valid(sma_50) and close < sma_50
    below_200 = _is_valid(close) and _is_valid(sma_200) and close < sma_200
    below_both = below_50 and below_200

    # -----------------------------
    # Long-term regime
    # -----------------------------
    if _is_valid(close) and _is_valid(sma_200):
        if close > sma_200:
            score += 4
            trend_score += 4
            quality_score += 4
            reasons.append("Price is above the 200-day moving average, supporting a healthier long-term regime")
        else:
            score -= 7
            trend_score -= 7
            quality_score -= 7
            risk_score -= 3
            reasons.append("Price is below the 200-day moving average, which is a major warning for put selling")

    # -----------------------------
    # Intermediate trend
    # -----------------------------
    if _is_valid(close) and _is_valid(sma_50):
        if close > sma_50:
            score += 3
            trend_score += 3
            quality_score += 3
            reasons.append("Price is above the 50-day moving average, indicating intermediate trend support")
        else:
            score -= 5
            trend_score -= 5
            quality_score -= 4
            risk_score -= 2
            reasons.append("Price is below the 50-day moving average, weakening the support structure")

    # -----------------------------
    # Trend alignment
    # -----------------------------
    if _is_valid(close) and _is_valid(sma_20) and _is_valid(sma_50):
        if close > sma_20 and sma_20 > sma_50:
            score += 2
            trend_score += 2
            quality_score += 2
            reasons.append("Price and moving averages are aligned in a constructive uptrend")
        elif close < sma_20 and sma_20 < sma_50:
            score -= 3
            trend_score -= 3
            quality_score -= 2
            risk_score -= 1
            reasons.append("Short-term structure is misaligned below key averages")

    if _is_valid(sma_50) and _is_valid(sma_200):
        if sma_50 >= sma_200:
            score += 2
            trend_score += 1
            quality_score += 2
            reasons.append("The 50-day moving average is above the 200-day moving average, supporting better assignment quality")
        else:
            score -= 3
            trend_score -= 2
            quality_score -= 2
            risk_score -= 1
            reasons.append("The 50-day moving average is below the 200-day moving average, which weakens the longer-term backdrop")

    # -----------------------------
    # Short-term MA alignment
    # -----------------------------
    if _is_valid(ema_9) and _is_valid(ema_21):
        if ema_9 > ema_21:
            score += 1
            trend_score += 1
            quality_score += 1
            reasons.append("EMA 9 is above EMA 21, supporting near-term trend stability")
        else:
            score -= 2
            trend_score -= 2
            quality_score -= 1
            risk_score -= 1
            reasons.append("EMA 9 is below EMA 21, showing weaker short-term momentum")

    # -----------------------------
    # Momentum / trend confirmation
    # -----------------------------
    if _is_valid(macd) and _is_valid(macd_signal) and _is_valid(macd_hist):
        if macd > macd_signal and macd_hist >= 0:
            score += 2
            trend_score += 2
            quality_score += 1
            reasons.append("MACD is above signal with positive momentum confirmation")
        elif macd < macd_signal and macd_hist < 0:
            score -= 3
            trend_score -= 3
            risk_score -= 2
            reasons.append("MACD is below signal with negative momentum, raising breakdown risk")
        elif macd < macd_signal and macd_hist >= 0:
            score -= 1
            trend_score -= 1
            reasons.append("MACD remains below signal, though downside momentum is not accelerating")

    # -----------------------------
    # ADX / DI trend strength context
    # -----------------------------
    if _is_valid(adx) and _is_valid(adx_pos) and _is_valid(adx_neg):
        if adx >= 20 and adx_pos > adx_neg:
            score += 2
            trend_score += 2
            quality_score += 1
            reasons.append("ADX and directional movement support a strengthening bullish trend")
        elif adx >= 20 and adx_neg > adx_pos:
            score -= 4
            trend_score -= 4
            risk_score -= 2
            reasons.append("ADX confirms bearish directional pressure, which is unfavorable for put selling")
        elif adx < 18 and adx_pos > adx_neg:
            score += 1
            trend_score += 1
            reasons.append("Trend strength is modest, but directional bias still favors bulls")

    # -----------------------------
    # Moving-average slope quality
    # -----------------------------
    if _is_valid(sma_50_slope_10):
        if sma_50_slope_10 > 0.75:
            score += 2
            trend_score += 2
            quality_score += 2
            reasons.append("The 50-day moving average slope is rising, which supports a healthier trend backdrop")
        elif sma_50_slope_10 < -0.75:
            score -= 3
            trend_score -= 3
            quality_score -= 2
            risk_score -= 1
            reasons.append("The 50-day moving average slope is falling, which weakens the underlying trend quality")

    if _is_valid(sma_200_slope_20):
        if sma_200_slope_20 > 0.2:
            score += 1
            quality_score += 1
            reasons.append("The 200-day moving average slope is positive, supporting longer-term assignment quality")
        elif sma_200_slope_20 < -0.2:
            score -= 2
            quality_score -= 2
            risk_score -= 1
            reasons.append("The 200-day moving average slope is rolling over, which weakens the longer-term backdrop")

    # -----------------------------
    # Underlying ownership quality
    # -----------------------------
    if _is_valid(close):
        if close >= 20:
            score += 1
            quality_score += 1
            reasons.append("Share price is high enough to avoid the weakest low-priced-stock profile")
        elif close < 7:
            score -= 3
            quality_score -= 3
            risk_score -= 1
            reasons.append("Share price is very low, which often points to a more speculative assignment profile")
        elif close < 12:
            score -= 1
            quality_score -= 1
            reasons.append("Lower-priced stocks tend to behave less consistently for conservative put selling")

    # -----------------------------
    # Relative strength vs benchmark
    # -----------------------------
    if _is_valid(rs_20):
        if rs_20 > 5:
            score += 3
            quality_score += 2
            reasons.append("The stock is outperforming SPY over the last 20 days")
        elif rs_20 > 1:
            score += 2
            quality_score += 1
            reasons.append("Relative strength versus SPY is positive over the last 20 days")
        elif rs_20 < -5:
            score -= 3
            quality_score -= 2
            risk_score -= 1
            reasons.append("The stock is materially underperforming SPY over the last 20 days")
        elif rs_20 < -1:
            score -= 1
            quality_score -= 1
            reasons.append("Relative strength versus SPY is mildly negative")

    if _is_valid(rs_60):
        if rs_60 > 8:
            score += 2
            quality_score += 2
            reasons.append("Medium-term relative strength versus SPY remains strong")
        elif rs_60 < -8:
            score -= 2
            quality_score -= 2
            risk_score -= 1
            reasons.append("Medium-term relative strength versus SPY remains weak")

    # -----------------------------
    # Market regime filter
    # -----------------------------
    if market_regime == "Bull":
        score += 2
        quality_score += 2
        reasons.append("The broader market regime is bullish, which is more supportive for put selling")
    elif market_regime == "Neutral":
        score -= 1
        reasons.append("The broader market regime is neutral, so put-selling setups should be judged more selectively")
    elif market_regime == "Bear":
        score -= 6
        quality_score -= 3
        risk_score -= 3
        reasons.append("The broader market regime is bearish, which materially increases support-failure risk for put selling")

    # -----------------------------
    # Liquidity and execution quality
    # -----------------------------
    if liquidity_ok is True:
        score += 2
        quality_score += 1
        reasons.append("Screen liquidity looks acceptable for put-selling execution")
    elif liquidity_ok is False:
        score -= 6
        quality_score -= 3
        risk_score -= 2
        reasons.append("Screen liquidity looks thin, which makes put-selling execution less reliable")

    # -----------------------------
    # Pullback quality
    # -----------------------------
    if _is_valid(rsi):
        if 42 <= rsi <= 55:
            score += 4
            pullback_score += 4
            entry_score += 4
            reasons.append("RSI is in a strong pullback zone for cash-secured put entries")
        elif 38 <= rsi < 42:
            score += 2
            pullback_score += 2
            entry_score += 2
            reasons.append("RSI is mildly weak but still within a potentially favorable pullback range")
        elif 55 < rsi <= 65:
            score += 1
            pullback_score += 1
            entry_score += 1
            reasons.append("RSI remains healthy, though the pullback is less attractive")
        elif 32 <= rsi < 38:
            score -= 2
            pullback_score -= 2
            entry_score -= 1
            risk_score -= 1
            reasons.append("RSI is getting weak and may reflect a pullback that is becoming less controlled")
        elif rsi < 32:
            score -= 5
            pullback_score -= 5
            entry_score -= 3
            risk_score -= 2
            reasons.append("RSI is very weak, suggesting the stock may be in a more fragile decline")
        elif rsi > 70:
            score -= 2
            pullback_score -= 2
            entry_score -= 2
            reasons.append("RSI is extended, making it a less favorable put-selling entry right now")

    if _is_valid(pullback_from_20d_high_pct):
        if -12 <= pullback_from_20d_high_pct <= -3:
            score += 2
            pullback_score += 2
            entry_score += 2
            reasons.append("The stock is pulling back from its 20-day high in a healthier entry range")
        elif pullback_from_20d_high_pct < -18:
            score -= 3
            pullback_score -= 3
            risk_score -= 2
            reasons.append("The pullback from the 20-day high is getting too deep, which raises breakdown risk")
        elif pullback_from_20d_high_pct > -2:
            score -= 1
            entry_score -= 1
            reasons.append("Price is still close to its recent high, so the pullback is not yet very compelling")

    if _is_valid(pullback_from_50d_high_pct) and pullback_from_50d_high_pct < -22:
        score -= 2
        pullback_score -= 2
        risk_score -= 1
        reasons.append("The pullback from the 50-day high is deep enough to warrant more caution")

    if _is_valid(pullback_from_126d_high_pct):
        if -18 <= pullback_from_126d_high_pct <= -5:
            score += 1
            quality_score += 1
            reasons.append("The medium-term drawdown is still within a more normal pullback range")
        elif pullback_from_126d_high_pct < -30:
            score -= 3
            quality_score -= 2
            risk_score -= 2
            reasons.append("The stock is deeply below its medium-term highs, which weakens the case for comfortable assignment")

    if _is_valid(pullback_from_252d_high_pct):
        if pullback_from_252d_high_pct < -40:
            score -= 4
            quality_score -= 3
            risk_score -= 2
            reasons.append("The stock is still far below its yearly highs, which points to a weaker long-term ownership profile")

    # -----------------------------
    # Support proximity / structure
    # -----------------------------
    if _is_valid(dist_sma50_pct):
        if -3.5 <= dist_sma50_pct <= 1.5:
            score += 4
            support_score += 4
            entry_score += 3
            reasons.append("Price is sitting close to the 50-day area, which often acts as a useful support reference")
        elif -6 <= dist_sma50_pct < -3.5:
            score += 1
            support_score += 1
            entry_score += 1
            reasons.append("Price is moderately below the 50-day average and may still be testing support")
        elif 1.5 < dist_sma50_pct <= 8:
            score += 1
            support_score += 1
            reasons.append("Price remains above the 50-day average, though not in an ideal pullback zone")
        elif dist_sma50_pct < -6:
            score -= 5
            support_score -= 5
            risk_score -= 2
            reasons.append("Price is too far below the 50-day average, suggesting support may be failing")

    if _is_valid(dist_sma200_pct):
        if 0 <= dist_sma200_pct <= 15:
            score += 2
            support_score += 2
            quality_score += 1
            reasons.append("Price remains reasonably close to long-term support")
        elif dist_sma200_pct < 0:
            score -= 4
            support_score -= 4
            risk_score -= 2
            reasons.append("Price is below the 200-day support area, which weakens assignment attractiveness")

    if _is_valid(bb_low) and _is_valid(close):
        if close >= bb_low and close <= bb_low * 1.025 and not below_50:
            score += 1
            support_score += 1
            entry_score += 1
            reasons.append("Price is near the lower Bollinger Band while still holding broader support")
        elif close < bb_low and below_50:
            score -= 3
            support_score -= 3
            risk_score -= 2
            reasons.append("Price is below the lower Bollinger Band and key trend support, increasing downside risk")

    # -----------------------------
    # Volatility / downside behavior
    # -----------------------------
    if _is_valid(atr_pct):
        if 1.2 <= atr_pct <= 4.5:
            score += 1
            quality_score += 1
            reasons.append("ATR remains in a manageable range for a more controlled put-selling setup")
        elif 4.5 < atr_pct <= 6:
            score -= 1
            quality_score -= 1
            risk_score -= 1
            reasons.append("ATR is a bit elevated, so the stock may be less forgiving if support fails")
        elif atr_pct > 6:
            score -= 3
            risk_score -= 2
            reasons.append("ATR is elevated, which increases gap and support-failure risk")

    if _is_valid(realized_vol_20):
        if realized_vol_20 <= 0.38:
            score += 1
            quality_score += 1
            reasons.append("Realized volatility is relatively contained, which improves assignment comfort")
        elif realized_vol_20 >= 0.65:
            score -= 2
            quality_score -= 1
            risk_score -= 2
            reasons.append("Realized volatility is high, which makes the underlying less comfortable to own on assignment")

    if _is_valid(downside_vol_ratio_20):
        if downside_vol_ratio_20 <= 0.55:
            score += 1
            quality_score += 1
            reasons.append("Downside volatility is relatively controlled versus overall volatility")
        elif downside_vol_ratio_20 >= 0.85:
            score -= 3
            risk_score -= 2
            reasons.append("Downside volatility is dominating, which makes assignment risk less attractive")

    if _is_valid(down_volume_ratio_20):
        if down_volume_ratio_20 <= 0.95:
            score += 1
            flow_score += 1
            reasons.append("Down-day volume has stayed relatively contained, which supports support durability")
        elif down_volume_ratio_20 >= 1.15:
            score -= 3
            flow_score -= 2
            risk_score -= 2
            reasons.append("Down-day volume is heavy relative to normal volume, which weakens support confidence")

    if _is_valid(volume_ratio_20) and volume_ratio_20 >= 1.5 and below_50:
        score -= 2
        flow_score -= 1
        risk_score -= 1
        reasons.append("Volume is elevated while price is below support references, which adds caution")

    # -----------------------------
    # Explicit support / entry logic
    # -----------------------------
    if _is_valid(close) and _is_valid(primary_support):
        dist_primary = _pct_diff(close, primary_support)

        if dist_primary is not None:
            if 0 <= dist_primary <= 3:
                score += 3
                support_score += 3
                entry_score += 3
                reasons.append("Price is trading close to primary support, which is favorable for put-selling location")
            elif 3 < dist_primary <= 8:
                score += 1
                support_score += 1
                entry_score += 1
                reasons.append("Price is above support, though somewhat less precise as an entry")
            elif dist_primary < 0:
                score -= 5
                support_score -= 5
                risk_score -= 3
                reasons.append("Price is below primary support, which sharply increases breakdown risk")

    if _is_valid(close) and _is_valid(secondary_support):
        dist_secondary = _pct_diff(close, secondary_support)
        if dist_secondary is not None and 0 <= dist_secondary <= 10:
            score += 1
            support_score += 1
            reasons.append("Secondary support is still reasonably close beneath price")

    if _is_valid(support_strength):
        if support_strength >= 9:
            score += 3
            support_score += 3
            entry_score += 1
            reasons.append("Support quality is strong, with meaningful confluence beneath price")
        elif support_strength >= 6:
            score += 1
            support_score += 1
            reasons.append("Support quality is acceptable, which helps the put-selling setup")
        elif support_strength < 4:
            score -= 3
            support_score -= 3
            risk_score -= 2
            reasons.append("Support quality is weak, so the margin for error is much smaller")

    if _is_valid(support_confluence_count):
        if support_confluence_count >= 4:
            score += 1
            support_score += 1
            reasons.append("Several support references are lining up in the same area")
        elif support_confluence_count <= 1:
            score -= 1
            support_score -= 1
            reasons.append("Support relies on relatively little confluence")

    if _is_valid(close) and _is_valid(recommended_entry):
        dist_entry = _pct_diff(close, recommended_entry)

        if dist_entry is not None:
            if -1 <= dist_entry <= 2:
                score += 3
                entry_score += 3
                reasons.append("Price is near the recommended entry area")
            elif dist_entry > 2:
                score -= 1
                entry_score -= 1
                reasons.append("Price is somewhat above the recommended entry zone, so patience may improve setup quality")
            elif dist_entry < -1:
                score -= 3
                entry_score -= 2
                risk_score -= 2
                reasons.append("Price is below the recommended entry area, which may indicate support is failing")

    if isinstance(entry_status, str) and entry_status.strip():
        status = entry_status.strip().lower()

        if "in entry zone" in status:
            score += 3
            entry_score += 3
            reasons.append("The stock is currently in its preferred entry zone")
        elif "watch for stabilization" in status:
            score += 1
            entry_score += 1
            reasons.append("The stock is close to support, but further stabilization would improve the setup")
        elif "wait for pullback" in status:
            entry_score += 0
            reasons.append("The setup is constructive, but a better pullback entry may still be ahead")
        elif "support under pressure" in status:
            score -= 3
            entry_score -= 2
            risk_score -= 2
            reasons.append("Support is under pressure, which weakens the current put-selling setup")
        elif "below support" in status or "caution" in status:
            score -= 4
            entry_score -= 2
            risk_score -= 3
            reasons.append("Price is below support / caution territory, which is unfavorable for fresh put exposure")

    # -----------------------------
    # Bounce confirmation near support
    # -----------------------------
    if isinstance(bounce_signal, str) and bounce_signal.strip():
        signal = bounce_signal.strip().lower()

        if "confirmed bounce" in signal:
            score += 4
            support_score += 2
            entry_score += 4
            quality_score += 1
            reasons.append("Price has tested support and is showing a confirmed bounce")
        elif "early bounce" in signal:
            score += 2
            support_score += 1
            entry_score += 2
            reasons.append("Price is starting to bounce from support, though confirmation is still early")
        elif "at support, no bounce yet" in signal:
            score -= 2
            entry_score -= 2
            risk_score -= 1
            reasons.append("Price is at support, but bounce confirmation has not shown up yet")
        elif "near support" in signal:
            entry_score += 0
            reasons.append("Price is moving closer to support, but the actual support test is not complete yet")
        elif "no bounce setup" in signal:
            score -= 1
            entry_score -= 1
            reasons.append("The stock is not yet in a cleaner support-test area for fresh put selling")
        elif "broken below support" in signal:
            score -= 4
            support_score -= 3
            risk_score -= 3
            reasons.append("Support has broken, so the bounce thesis is not currently valid")

    # -----------------------------
    # Money flow
    # -----------------------------
    if _is_valid(cmf):
        if cmf > 0.08:
            score += 3
            flow_score += 3
            quality_score += 1
            reasons.append("Chaikin Money Flow is strongly positive, supporting accumulation")
        elif 0.02 < cmf <= 0.08:
            score += 2
            flow_score += 2
            reasons.append("Chaikin Money Flow is positive, supporting buyer participation")
        elif -0.02 <= cmf <= 0.02:
            flow_score += 0
            reasons.append("Money flow is roughly neutral")
        elif -0.08 <= cmf < -0.02:
            score -= 2
            flow_score -= 2
            risk_score -= 1
            reasons.append("Money flow is mildly negative, which weakens support confidence")
        elif cmf < -0.08:
            score -= 4
            flow_score -= 4
            risk_score -= 2
            reasons.append("Chaikin Money Flow is strongly negative, suggesting distribution risk")

    # -----------------------------
    # Assignment attractiveness / stretch penalties
    # -----------------------------
    if below_both:
        score -= 4
        risk_score -= 3
        reasons.append("Price is below both the 50-day and 200-day moving averages, making assignment less attractive")

    if _is_valid(close) and _is_valid(sma_200):
        long_term_stretch = _pct_diff(close, sma_200)
        if long_term_stretch is not None and long_term_stretch > 25:
            score -= 1
            entry_score -= 1
            reasons.append("Price is extended far above long-term support, reducing pullback-entry quality")

    # -----------------------------
    # Hard-fail gates
    # -----------------------------
    severe_fail_reasons = []
    warning_fail_reasons = []

    if _is_valid(close) and _is_valid(sma_200) and _is_valid(sma_50):
        if close < sma_200 and sma_50 < sma_200:
            severe_fail_reasons.append("Price is below the 200-day average and the 50-day average is below the 200-day average")

    if _is_valid(adx) and _is_valid(adx_pos) and _is_valid(adx_neg):
        if adx >= 30 and adx_neg > adx_pos:
            warning_fail_reasons.append("ADX confirms a strong bearish trend")

    if _is_valid(cmf) and cmf <= -0.12:
        warning_fail_reasons.append("Chaikin Money Flow shows strong distribution")

    if liquidity_ok is False:
        severe_fail_reasons.append("Screen liquidity is too thin for higher-confidence put selling")

    if (
        _is_valid(close)
        and _is_valid(sma_50)
        and _is_valid(atr_14)
        and close < (sma_50 - (2 * atr_14))
    ):
        warning_fail_reasons.append("Price is stretched far below the 50-day average relative to ATR")

    if _is_valid(close) and _is_valid(primary_support):
        if close < primary_support * 0.985:
            severe_fail_reasons.append("Price has broken meaningfully below primary support")

    if market_regime == "Bear" and _is_valid(close) and _is_valid(sma_200) and close < sma_50:
        severe_fail_reasons.append("Bearish market regime is combining with weak stock structure")

    if _is_valid(downside_vol_ratio_20) and downside_vol_ratio_20 >= 0.95:
        warning_fail_reasons.append("Downside volatility is dominating the recent tape")

    entry_status_lower = entry_status.lower() if isinstance(entry_status, str) else ""
    bounce_signal_lower = bounce_signal.lower() if isinstance(bounce_signal, str) else ""
    in_entry_zone = entry_status_lower == "in entry zone"
    watch_stabilization = entry_status_lower == "watch for stabilization"
    wait_for_pullback = entry_status_lower == "wait for pullback"
    confirmed_bounce = bounce_confirmed or "confirmed bounce" in bounce_signal_lower
    early_bounce = "early bounce" in bounce_signal_lower
    at_support_no_bounce = "at support, no bounce yet" in bounce_signal_lower
    near_support = "near support" in bounce_signal_lower
    no_bounce_setup = "no bounce setup" in bounce_signal_lower

    strong_timing_ready = confirmed_bounce or (early_bounce and in_entry_zone)
    candidate_timing_ready = (
        strong_timing_ready
        or early_bounce
        or in_entry_zone
        or watch_stabilization
        or at_support_no_bounce
    )
    stalk_timing_ready = (
        candidate_timing_ready
        or wait_for_pullback
        or near_support
        or no_bounce_setup
    )
    support_broken = (
        "support under pressure" in entry_status_lower
        or "below support" in entry_status_lower
        or "broken below support" in bounce_signal_lower
    )
    cautionary_fail = len(warning_fail_reasons) >= 1
    extreme_instability = (
        (_is_valid(atr_pct) and atr_pct > 8)
        or (_is_valid(realized_vol_20) and realized_vol_20 > 0.85)
        or (_is_valid(downside_vol_ratio_20) and downside_vol_ratio_20 >= 0.97)
    )
    controlled_volatility = (
        (not _is_valid(atr_pct) or atr_pct <= 6)
        and (not _is_valid(realized_vol_20) or realized_vol_20 <= 0.65)
        and (not _is_valid(downside_vol_ratio_20) or downside_vol_ratio_20 < 0.9)
    )
    stalk_min_score = 5
    stalk_min_quality = 5
    stalk_min_risk = -4
    candidate_min_score = 8 + _profile_adjustment(learning_profile, "candidate_min_score")
    candidate_min_quality = 5 + _profile_adjustment(learning_profile, "candidate_min_quality")
    candidate_min_entry = 0 + _profile_adjustment(learning_profile, "candidate_min_entry")
    candidate_min_risk = -4 + _profile_adjustment(learning_profile, "candidate_min_risk")
    high_probability_min_score_neutral = 14 + _profile_adjustment(learning_profile, "high_probability_min_score")
    high_probability_min_score_supportive = 13 + _profile_adjustment(learning_profile, "high_probability_min_score")
    high_probability_min_quality = 8 + _profile_adjustment(learning_profile, "high_probability_min_quality")
    high_probability_min_entry = 3 + _profile_adjustment(learning_profile, "high_probability_min_entry")
    high_probability_min_risk = -2 + _profile_adjustment(learning_profile, "high_probability_min_risk")
    assignment_ready = (
        liquidity_ok is True
        and quality_score >= stalk_min_quality
        and risk_score >= stalk_min_risk
        and not extreme_instability
    )
    support_ready = (
        not support_broken
        and (not _is_valid(support_strength) or support_strength >= 4)
    )
    candidate_support_ready = (
        support_ready
        and (not _is_valid(support_strength) or support_strength >= 5)
        and (support_distance_pct is None or support_distance_pct <= 6.5)
    )
    prime_support_ready = (
        support_ready
        and (not _is_valid(support_strength) or support_strength >= 7)
        and (
            support_distance_pct is None
            or support_distance_pct <= 4.0
            or in_entry_zone
        )
    )
    stalk_setup = (
        assignment_ready
        and support_ready
        and stalk_timing_ready
        and quality_score >= stalk_min_quality
        and risk_score >= stalk_min_risk
    )
    prime_setup = (
        assignment_ready
        and prime_support_ready
        and strong_timing_ready
        and quality_score >= high_probability_min_quality
        and entry_score >= high_probability_min_entry
        and risk_score >= high_probability_min_risk
        and controlled_volatility
        and not cautionary_fail
    )
    candidate_setup = (
        assignment_ready
        and candidate_support_ready
        and candidate_timing_ready
        and quality_score >= candidate_min_quality
        and entry_score >= candidate_min_entry
        and risk_score >= candidate_min_risk
        and controlled_volatility
    )

    candidate_blockers = []
    if liquidity_ok is False:
        candidate_blockers.append("Liquidity is too thin for a higher-quality put-selling setup.")
    if quality_score < candidate_min_quality:
        candidate_blockers.append("Underlying quality is not strong enough to justify assignment yet.")
    if risk_score < candidate_min_risk:
        candidate_blockers.append("Downside risk is still too elevated for a candidate label.")
    if extreme_instability:
        candidate_blockers.append("Volatility is too extreme right now for a conservative cash-secured put setup.")
    if support_broken:
        candidate_blockers.append("Support is failing or already broken, so the setup is not trustworthy enough.")
    elif _is_valid(support_strength) and support_strength < 4:
        candidate_blockers.append("Support quality is too weak and lacks enough confluence.")
    elif support_distance_pct is not None and support_distance_pct > 6.5:
        candidate_blockers.append("Price is still too far above support to count as a disciplined put-selling location.")
    if not candidate_timing_ready:
        candidate_blockers.append("Price is not yet close enough to a usable support-based entry area.")
    if entry_score < candidate_min_entry:
        candidate_blockers.append("Entry timing is still too poor for a put-selling candidate.")
    if not controlled_volatility:
        candidate_blockers.append("Volatility is still too active for a cleaner trade-ready put setup.")

    # -----------------------------
    # Labeling logic
    # -----------------------------
    if len(severe_fail_reasons) >= 2 or (len(severe_fail_reasons) >= 1 and len(warning_fail_reasons) >= 1):
        label = "Breakdown Risk"
        score = min(score, -10)
        reasons.extend((severe_fail_reasons + warning_fail_reasons)[:2])

    elif len(severe_fail_reasons) == 1 or len(warning_fail_reasons) >= 2:
        label = "Downtrend Risk"
        score = min(score, -4)
        reasons.extend((severe_fail_reasons + warning_fail_reasons)[:1])

    elif market_regime == "Bear":
        if warning_fail_reasons:
            reasons.extend(warning_fail_reasons[:1])
            score -= 1
            risk_score -= 1

        if (
            candidate_setup
            and quality_score >= max(6, candidate_min_quality)
            and entry_score >= 1
            and score >= max(10, candidate_min_score + 1)
        ):
            label = "Put Sell Candidate"
        elif stalk_setup and score >= max(5, stalk_min_score):
            label = "Stalk / Watchlist"
        elif score >= 2:
            label = "Neutral / Wait"
        elif score >= -8:
            label = "Downtrend Risk"
        else:
            label = "Breakdown Risk"

    elif market_regime == "Neutral":
        if warning_fail_reasons:
            reasons.extend(warning_fail_reasons[:1])
            score -= 1
            risk_score -= 1

        if prime_setup and score >= high_probability_min_score_neutral:
            label = "High Probability Put Sell"
        elif candidate_setup and score >= candidate_min_score:
            label = "Put Sell Candidate"
        elif stalk_setup and score >= stalk_min_score:
            label = "Stalk / Watchlist"
        elif score >= 1:
            label = "Neutral / Wait"
        elif score >= -8:
            label = "Downtrend Risk"
        else:
            label = "Breakdown Risk"

    else:
        if warning_fail_reasons:
            reasons.extend(warning_fail_reasons[:1])
            score -= 1
            risk_score -= 1

        if prime_setup and score >= high_probability_min_score_supportive:
            label = "High Probability Put Sell"
        elif candidate_setup and score >= max(5, candidate_min_score - 1):
            label = "Put Sell Candidate"
        elif stalk_setup and score >= max(4, stalk_min_score - 1):
            label = "Stalk / Watchlist"
        elif score >= 1:
            label = "Neutral / Wait"
        elif score >= -8:
            label = "Downtrend Risk"
        else:
            label = "Breakdown Risk"

    # -----------------------------
    # Confidence logic
    # -----------------------------
    confidence = 40
    confidence += min(abs(score) * 1.4, 18)
    confidence += min(abs(quality_score) * 1.6, 12)
    confidence += min(abs(entry_score) * 1.4, 10)
    confidence += min(abs(risk_score) * 1.6, 12)

    if label in {"High Probability Put Sell", "Put Sell Candidate"}:
        if quality_score >= 7:
            confidence += 7
        elif quality_score >= 5:
            confidence += 4

        if entry_score >= 4:
            confidence += 6
        elif entry_score >= 1:
            confidence += 3

        if risk_score >= -2:
            confidence += 6
        elif risk_score >= -4:
            confidence += 3
        else:
            confidence -= 8

        if severe_fail_reasons or warning_fail_reasons:
            confidence -= 12

    elif label == "Stalk / Watchlist":
        if quality_score >= 7:
            confidence += 6
        elif quality_score >= 5:
            confidence += 4

        if -1 <= entry_score <= 2:
            confidence += 4
        elif entry_score < -1:
            confidence -= 3

        if risk_score >= -2:
            confidence += 4
        elif risk_score <= -5:
            confidence -= 6

        if wait_for_pullback or near_support:
            confidence += 4
        if cautionary_fail:
            confidence -= 6

    elif label == "Neutral / Wait":
        if 3 <= quality_score <= 7:
            confidence += 5
        if -3 <= risk_score <= 0:
            confidence += 5
        if -1 <= entry_score <= 3:
            confidence += 4

    else:
        if severe_fail_reasons or warning_fail_reasons:
            confidence += 10
        if risk_score <= -5:
            confidence += 8
        elif risk_score <= -3:
            confidence += 4

        if quality_score <= 1:
            confidence += 6
        if support_score <= -3:
            confidence += 4
        if flow_score <= -2:
            confidence += 3

    inconsistency_penalty = 0
    if quality_score >= 7 and risk_score <= -4:
        inconsistency_penalty += 6
    if entry_score >= 4 and support_score <= 0:
        inconsistency_penalty += 4
    if label in {"High Probability Put Sell", "Put Sell Candidate", "Stalk / Watchlist"} and flow_score <= -2:
        inconsistency_penalty += 4
    if label in {"Downtrend Risk", "Breakdown Risk"} and quality_score >= 5:
        inconsistency_penalty += 3

    confidence -= inconsistency_penalty
    confidence = max(35, min(96, int(round(confidence))))

    return {
        "score": int(score),
        "label": label,
        "confidence": confidence,
        "trend_score": int(trend_score),
        "pullback_score": int(pullback_score),
        "support_score": int(support_score),
        "flow_score": int(flow_score),
        "quality_score": int(quality_score),
        "entry_score": int(entry_score),
        "risk_score": int(risk_score),
        "candidate_blockers": candidate_blockers,
        "reasons": reasons
    }
