from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
import yfinance as yf


FIXED_PERIOD = "1y"
FIXED_INTERVAL = "1d"
BENCHMARK_SYMBOL = "SPY"
CACHE_TTL_SECONDS = 900


def _cache_bucket() -> int:
    return int(datetime.now(timezone.utc).timestamp() // CACHE_TTL_SECONDS)


def _normalize_symbols(symbols: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for item in symbols or []:
        symbol = str(item).strip().upper()
        if symbol and symbol not in normalized:
            normalized.append(symbol)
    return normalized


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


def _extract_symbol_frame(
    download_df: pd.DataFrame,
    symbol: str,
    allow_flat: bool = False,
) -> pd.DataFrame:
    if download_df is None or download_df.empty:
        raise ValueError(f"{symbol}: No batch data returned from yfinance.")

    if not isinstance(download_df.columns, pd.MultiIndex):
        if not allow_flat:
            raise ValueError(f"{symbol}: Batch data was not split by symbol.")
        return download_df.copy()

    upper_symbol = symbol.upper()
    for level in range(download_df.columns.nlevels):
        level_values = list(download_df.columns.get_level_values(level).unique())
        match = next((value for value in level_values if str(value).upper() == upper_symbol), None)
        if match is not None:
            return download_df.xs(match, axis=1, level=level, drop_level=True).copy()

    raise ValueError(f"{symbol}: Symbol missing from batch download.")


def _download_history_uncached(
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


@lru_cache(maxsize=512)
def _download_history_cached(
    symbol: str,
    period: str,
    interval: str,
    cache_bucket: int,
) -> pd.DataFrame:
    del cache_bucket
    return _download_history_uncached(symbol, period=period, interval=interval)


def _download_history(
    symbol: str,
    period: str = FIXED_PERIOD,
    interval: str = FIXED_INTERVAL,
) -> pd.DataFrame:
    normalized_symbol = str(symbol).strip().upper()
    return _download_history_cached(
        normalized_symbol,
        period,
        interval,
        _cache_bucket(),
    ).copy()


def fetch_benchmark_data(
    symbol: str = BENCHMARK_SYMBOL,
    period: str = FIXED_PERIOD,
    interval: str = FIXED_INTERVAL,
) -> pd.DataFrame:
    return _download_history(symbol, period=period, interval=interval)


def fetch_many_stock_data(
    symbols: list[str] | tuple[str, ...],
    period: str = FIXED_PERIOD,
    interval: str = FIXED_INTERVAL,
) -> dict[str, pd.DataFrame]:
    normalized_symbols = _normalize_symbols(symbols)
    if not normalized_symbols:
        return {}

    if len(normalized_symbols) == 1:
        symbol = normalized_symbols[0]
        return {symbol: _download_history(symbol, period=period, interval=interval)}

    raw_batch = None
    try:
        raw_batch = yf.download(
            tickers=" ".join(normalized_symbols),
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        raw_batch = None

    results = {}
    for symbol in normalized_symbols:
        try:
            symbol_df = _extract_symbol_frame(raw_batch, symbol)
            results[symbol] = _normalize_history(symbol_df, symbol)
        except Exception:
            results[symbol] = _download_history(symbol, period=period, interval=interval)

    return results


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
