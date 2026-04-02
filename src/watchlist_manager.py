import json
import os

WATCHLIST_FILE = "data/watchlist.json"
DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "SPY"]


def ensure_watchlist_file():
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(WATCHLIST_FILE):
        save_watchlist(DEFAULT_WATCHLIST)


def load_watchlist() -> list[str]:
    ensure_watchlist_file()

    try:
        with open(WATCHLIST_FILE, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return DEFAULT_WATCHLIST.copy()

        cleaned = []
        for ticker in data:
            if isinstance(ticker, str):
                t = ticker.upper().strip()
                if t and t not in cleaned:
                    cleaned.append(t)

        return cleaned if cleaned else DEFAULT_WATCHLIST.copy()

    except Exception:
        return DEFAULT_WATCHLIST.copy()


def save_watchlist(watchlist: list[str]) -> None:
    os.makedirs("data", exist_ok=True)

    cleaned = []
    for ticker in watchlist:
        t = str(ticker).upper().strip()
        if t and t not in cleaned:
            cleaned.append(t)

    with open(WATCHLIST_FILE, "w") as f:
        json.dump(cleaned, f, indent=2)


def add_to_watchlist(watchlist: list[str], ticker: str) -> tuple[list[str], str]:
    ticker = ticker.upper().strip()

    if not ticker:
        return watchlist, "Enter a ticker first."

    if ticker in watchlist:
        return watchlist, f"{ticker} is already in your watchlist."

    updated = watchlist + [ticker]
    save_watchlist(updated)
    return updated, f"Added {ticker} to your watchlist."


def remove_from_watchlist(watchlist: list[str], ticker: str) -> tuple[list[str], str]:
    ticker = ticker.upper().strip()

    if ticker not in watchlist:
        return watchlist, f"{ticker} is not in your watchlist."

    updated = [t for t in watchlist if t != ticker]
    save_watchlist(updated)
    return updated, f"Removed {ticker} from your watchlist."