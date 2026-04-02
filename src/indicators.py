import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import ChaikinMoneyFlowIndicator


def add_indicators(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------
    # Clean numeric columns
    # -----------------------------
    required_cols = ["open", "high", "low", "close", "volume"]
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(f"Not enough usable rows after cleaning. Got {len(df)}.")

    # -----------------------------
    # Core momentum indicators
    # -----------------------------
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()

    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # -----------------------------
    # Moving averages
    # -----------------------------
    df["ema_9"] = EMAIndicator(close=df["close"], window=9).ema_indicator()
    df["ema_21"] = EMAIndicator(close=df["close"], window=21).ema_indicator()
    df["sma_20"] = SMAIndicator(close=df["close"], window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close=df["close"], window=50).sma_indicator()

    if len(df) >= 200:
        df["sma_200"] = SMAIndicator(close=df["close"], window=200).sma_indicator()
    else:
        df["sma_200"] = np.nan

    # Helpful regime flags
    df["above_sma_50"] = df["close"] > df["sma_50"]
    df["above_sma_200"] = df["close"] > df["sma_200"]
    df["bullish_ma_stack"] = (df["close"] > df["sma_20"]) & (df["sma_20"] > df["sma_50"])
    df["death_cross_regime"] = df["sma_50"] < df["sma_200"]

    # -----------------------------
    # Trend strength
    # -----------------------------
    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # -----------------------------
    # Bollinger Bands
    # -----------------------------
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    bb_range = df["bb_high"] - df["bb_low"]
    df["bb_position"] = np.where(
        bb_range > 0,
        (df["close"] - df["bb_low"]) / bb_range,
        np.nan
    )

    # -----------------------------
    # ATR context
    # -----------------------------
    atr = AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )
    df["atr_14"] = atr.average_true_range()

    df["atr_pct"] = np.where(
        df["close"] != 0,
        (df["atr_14"] / df["close"]) * 100,
        np.nan
    )

    # -----------------------------
    # Money flow
    # -----------------------------
    cmf = ChaikinMoneyFlowIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        volume=df["volume"],
        window=20
    )
    df["cmf_20"] = cmf.chaikin_money_flow()

    # -----------------------------
    # Distance from key moving averages
    # -----------------------------
    df["dist_sma50_pct"] = np.where(
        df["sma_50"].notna() & (df["sma_50"] != 0),
        ((df["close"] - df["sma_50"]) / df["sma_50"]) * 100,
        np.nan
    )

    df["dist_sma200_pct"] = np.where(
        df["sma_200"].notna() & (df["sma_200"] != 0),
        ((df["close"] - df["sma_200"]) / df["sma_200"]) * 100,
        np.nan
    )

    # -----------------------------
    # MA slope / regime context
    # -----------------------------
    df["sma_20_slope_5"] = np.where(
        df["sma_20"].shift(5).notna() & (df["sma_20"].shift(5) != 0),
        ((df["sma_20"] - df["sma_20"].shift(5)) / df["sma_20"].shift(5)) * 100,
        np.nan
    )

    df["sma_50_slope_10"] = np.where(
        df["sma_50"].shift(10).notna() & (df["sma_50"].shift(10) != 0),
        ((df["sma_50"] - df["sma_50"].shift(10)) / df["sma_50"].shift(10)) * 100,
        np.nan
    )

    df["sma_200_slope_20"] = np.where(
        df["sma_200"].shift(20).notna() & (df["sma_200"].shift(20) != 0),
        ((df["sma_200"] - df["sma_200"].shift(20)) / df["sma_200"].shift(20)) * 100,
        np.nan
    )

    # -----------------------------
    # Return / pullback context
    # -----------------------------
    df["return_5d_pct"] = df["close"].pct_change(5) * 100
    df["return_20d_pct"] = df["close"].pct_change(20) * 100
    df["return_60d_pct"] = df["close"].pct_change(60) * 100

    rolling_20_high = df["close"].rolling(20).max()
    rolling_50_high = df["close"].rolling(50).max()
    rolling_126_high = df["close"].rolling(126).max()
    rolling_252_high = df["close"].rolling(252).max()

    df["pullback_from_20d_high_pct"] = np.where(
        rolling_20_high.notna() & (rolling_20_high != 0),
        ((df["close"] - rolling_20_high) / rolling_20_high) * 100,
        np.nan
    )

    df["pullback_from_50d_high_pct"] = np.where(
        rolling_50_high.notna() & (rolling_50_high != 0),
        ((df["close"] - rolling_50_high) / rolling_50_high) * 100,
        np.nan
    )

    df["pullback_from_126d_high_pct"] = np.where(
        rolling_126_high.notna() & (rolling_126_high != 0),
        ((df["close"] - rolling_126_high) / rolling_126_high) * 100,
        np.nan
    )

    df["pullback_from_252d_high_pct"] = np.where(
        rolling_252_high.notna() & (rolling_252_high != 0),
        ((df["close"] - rolling_252_high) / rolling_252_high) * 100,
        np.nan
    )

    # -----------------------------
    # Volatility / downside behavior
    # -----------------------------
    df["daily_return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["realized_vol_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["realized_vol_60"] = df["log_return"].rolling(60).std() * np.sqrt(252)

    downside_log_returns = df["log_return"].where(df["log_return"] < 0)
    df["downside_vol_20"] = downside_log_returns.rolling(20).std() * np.sqrt(252)

    df["downside_vol_ratio_20"] = np.where(
        df["realized_vol_20"].notna() & (df["realized_vol_20"] != 0),
        df["downside_vol_20"] / df["realized_vol_20"],
        np.nan
    )

    # -----------------------------
    # Candle / volume behavior
    # -----------------------------
    df["range_pct"] = np.where(
        df["close"] != 0,
        ((df["high"] - df["low"]) / df["close"]) * 100,
        np.nan
    )

    candle_range = df["high"] - df["low"]
    df["close_location"] = np.where(
        candle_range > 0,
        (df["close"] - df["low"]) / candle_range,
        np.nan
    )

    df["body_pct"] = np.where(
        df["close"] != 0,
        (abs(df["close"] - df["open"]) / df["close"]) * 100,
        np.nan
    )

    df["close_change_pct"] = df["close"].pct_change() * 100
    df["green_candle"] = df["close"] > df["open"]
    df["bullish_reversal_day"] = (
        (df["close"] > df["open"]) &
        (df["close"] > df["close"].shift(1)) &
        (df["close_location"] >= 0.6)
    )

    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio_20"] = np.where(
        df["volume_sma_20"].notna() & (df["volume_sma_20"] != 0),
        df["volume"] / df["volume_sma_20"],
        np.nan
    )

    df["dollar_volume"] = df["close"] * df["volume"]
    df["avg_dollar_volume_20"] = df["dollar_volume"].rolling(20).mean()

    df["down_day"] = (df["close"] < df["open"]).astype(int)
    df["down_day_volume"] = np.where(df["close"] < df["open"], df["volume"], np.nan)

    down_day_volume_mean_20 = df["down_day_volume"].rolling(20).mean()
    df["down_volume_ratio_20"] = np.where(
        df["volume_sma_20"].notna() & (df["volume_sma_20"] != 0),
        down_day_volume_mean_20 / df["volume_sma_20"],
        np.nan
    )

    df["liquidity_ok"] = (
        (df["volume_sma_20"] >= 500_000) &
        (df["avg_dollar_volume_20"] >= 5_000_000)
    )

    # -----------------------------
    # Support reference helpers
    # -----------------------------
    df["rolling_20_low"] = df["low"].rolling(20).min()
    df["rolling_50_low"] = df["low"].rolling(50).min()
    df["rolling_20_high"] = df["high"].rolling(20).max()
    df["rolling_50_high"] = df["high"].rolling(50).max()

    df["dist_20d_low_pct"] = np.where(
        df["rolling_20_low"].notna() & (df["rolling_20_low"] != 0),
        ((df["close"] - df["rolling_20_low"]) / df["rolling_20_low"]) * 100,
        np.nan
    )

    df["dist_50d_low_pct"] = np.where(
        df["rolling_50_low"].notna() & (df["rolling_50_low"] != 0),
        ((df["close"] - df["rolling_50_low"]) / df["rolling_50_low"]) * 100,
        np.nan
    )

    # -----------------------------
    # Relative strength vs SPY
    # -----------------------------
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df[["timestamp", "close"]].copy()
        bench = bench.rename(columns={"close": "benchmark_close"})

        df = df.merge(bench, on="timestamp", how="left")

        df["benchmark_return_20d_pct"] = df["benchmark_close"].pct_change(20) * 100
        df["benchmark_return_60d_pct"] = df["benchmark_close"].pct_change(60) * 100

        df["rs_20"] = df["return_20d_pct"] - df["benchmark_return_20d_pct"]
        df["rs_60"] = df["return_60d_pct"] - df["benchmark_return_60d_pct"]
    else:
        df["benchmark_close"] = np.nan
        df["benchmark_return_20d_pct"] = np.nan
        df["benchmark_return_60d_pct"] = np.nan
        df["rs_20"] = np.nan
        df["rs_60"] = np.nan

    # -----------------------------
    # Clean infinities
    # -----------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df
