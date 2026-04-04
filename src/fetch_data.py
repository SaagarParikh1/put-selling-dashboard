import pandas as pd
import yfinance as yf


FIXED_PERIOD = "1y"
FIXED_INTERVAL = "1d"
BENCHMARK_SYMBOL = "SPY"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        flattened = []
        for col in df.columns:
            parts = [str(x) for x in col if x and str(x) != "None"]
            flattened.append(parts[0] if parts else "")
        df.columns = flattened
    return df


def _normalize_history(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(f"{symbol}: No data returned from yfinance.")

    df = _flatten_columns(df)
    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "timestamp"})
    elif "Date" in df.columns:
        df = df.rename(columns={"Date": "timestamp"})
    else:
        raise ValueError(f"{symbol}: Missing timestamp column.")

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: Missing expected columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["symbol"] = symbol.upper()

    df = df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]].copy()
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{symbol}: Data became empty after cleaning.")

    return df


def _download_history(
    symbol: str,
    period: str = FIXED_PERIOD,
    interval: str = FIXED_INTERVAL,
) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
        multi_level_index=False,
    )

    if df is None or df.empty:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
        )

    return _normalize_history(df, symbol)


def fetch_benchmark_data(
    symbol: str = BENCHMARK_SYMBOL,
    period: str = FIXED_PERIOD,
    interval: str = FIXED_INTERVAL,
) -> pd.DataFrame:
    return _download_history(symbol, period=period, interval=interval)


def fetch_stock_data(
    symbol: str,
    include_benchmark: bool = False,
    period: str = FIXED_PERIOD,
    interval: str = FIXED_INTERVAL,
):
    """
    Default behavior remains backward-compatible:
    - fetch_stock_data("AAPL") -> DataFrame

    Optional richer behavior:
    - fetch_stock_data("AAPL", include_benchmark=True)
      -> {
            "stock_df": ...,
            "benchmark_df": ...,
         }
    """
    stock_df = _download_history(symbol, period=period, interval=interval)

    if not include_benchmark:
        return stock_df

    benchmark_df = None
    if include_benchmark:
        try:
            benchmark_df = fetch_benchmark_data(period=period, interval=interval)
        except Exception:
            benchmark_df = None

    return {
        "stock_df": stock_df,
        "benchmark_df": benchmark_df,
    }
