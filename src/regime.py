import pandas as pd


def _is_valid_number(value) -> bool:
    return value is not None and value == value


def classify_market_regime(benchmark_df: pd.DataFrame) -> dict:
    if benchmark_df is None or benchmark_df.empty:
        return {
            "market_regime": "Unknown",
            "market_regime_score": 0,
            "spy_close": None,
            "spy_sma_50": None,
            "spy_sma_200": None,
            "spy_dist_sma200_pct": None,
            "spy_sma50_slope_10": None,
        }

    df = benchmark_df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["sma_50_slope_10"] = ((df["sma_50"] - df["sma_50"].shift(10)) / df["sma_50"].shift(10)) * 100

    latest = df.iloc[-1]

    close = latest.get("close")
    sma_50 = latest.get("sma_50")
    sma_200 = latest.get("sma_200")
    sma_50_slope_10 = latest.get("sma_50_slope_10")

    spy_dist_sma200_pct = None
    if _is_valid_number(close) and _is_valid_number(sma_200) and sma_200 != 0:
        spy_dist_sma200_pct = ((close - sma_200) / sma_200) * 100

    market_regime = "Neutral"
    market_regime_score = 0

    if _is_valid_number(close) and _is_valid_number(sma_200) and _is_valid_number(sma_50):
        if close < sma_200 and sma_50 < sma_200:
            market_regime = "Bear"
            market_regime_score = -2

        elif close > sma_200:
            if (_is_valid_number(sma_50_slope_10) and sma_50_slope_10 > 0) or sma_50 >= sma_200:
                market_regime = "Bull"
                market_regime_score = 2
            else:
                market_regime = "Neutral"
                market_regime_score = 0

    return {
        "market_regime": market_regime,
        "market_regime_score": market_regime_score,
        "spy_close": round(float(close), 2) if _is_valid_number(close) else None,
        "spy_sma_50": round(float(sma_50), 2) if _is_valid_number(sma_50) else None,
        "spy_sma_200": round(float(sma_200), 2) if _is_valid_number(sma_200) else None,
        "spy_dist_sma200_pct": round(float(spy_dist_sma200_pct), 2) if _is_valid_number(spy_dist_sma200_pct) else None,
        "spy_sma50_slope_10": round(float(sma_50_slope_10), 2) if _is_valid_number(sma_50_slope_10) else None,
    }