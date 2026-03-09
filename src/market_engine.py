"""
market_engine.py — Module 1: Market Data Engine
================================================
Responsibilities:
- Fetch live NSE/BSE prices via yfinance (appends .NS for NSE symbols).
- Pull OHLC historical data for any symbol.
- Support Nifty 50, Bank Nifty indices, and individual stocks.
- TTL-based caching to respect API rate limits.

NOTE: All yfinance calls for NSE stocks use the '<SYMBOL>.NS' ticker format.
      BSE symbols use '<SYMBOL>.BO'.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any

import yfinance as yf
import pandas as pd
from cachetools import TTLCache, cached
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_PRICE_TTL = int(os.getenv("PRICE_CACHE_TTL", "30"))  # seconds
_price_cache: TTLCache = TTLCache(maxsize=256, ttl=_PRICE_TTL)

# ── Well-known index tickers ──────────────────────────────────────────────────
INDEX_TICKERS: dict[str, str] = {
    "NIFTY50":   "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX":    "^BSESN",
    "NIFTYIT":   "^CNXIT",
    "NIFTYMID":  "^NSEMDCP50",
}

# Nifty 50 constituents (used by scan_market / sector heatmap)
NIFTY50_SYMBOLS: list[str] = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "HCLTECH",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "ULTRACEMCO", "WIPRO",
    "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "TATAMOTORS",
    "JSWSTEEL", "TATASTEEL", "M&M", "ADANIPORTS", "COALINDIA",
    "BAJAJFINSV", "HEROMOTOCO", "DRREDDY", "CIPLA", "DIVISLAB",
    "TECHM", "GRASIM", "INDUSINDBK", "EICHERMOT", "APOLLOHOSP",
    "BRITANNIA", "HDFCLIFE", "SBILIFE", "BPCL", "TATACONSUM",
    "ADANIENT", "BAJAJ-AUTO", "UPL", "LTIM", "HAL",
]

SECTOR_MAP: dict[str, list[str]] = {
    "IT":          ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"],
    "Banking":     ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK"],
    "FMCG":        ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"],
    "Auto":        ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "Pharma":      ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "Energy":      ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "COALINDIA"],
    "Metals":      ["JSWSTEEL", "TATASTEEL"],
    "Financials":  ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE"],
    "Infra":       ["LT", "ADANIPORTS", "ADANIENT"],
    "Consumer":    ["TITAN", "ASIANPAINT", "ULTRATECH", "GRASIM"],
}


def _nse_ticker(symbol: str) -> str:
    """Return the yfinance ticker for an NSE symbol, handling indices."""
    upper = symbol.upper()
    if upper in INDEX_TICKERS:
        return INDEX_TICKERS[upper]
    if upper.endswith(".NS") or upper.endswith(".BO") or upper.startswith("^"):
        return upper
    return f"{upper}.NS"


# ── Public API ────────────────────────────────────────────────────────────────

def get_live_price(symbol: str) -> dict[str, Any]:
    """
    Return the latest trade price, day change %, and volume for *symbol*.

    Parameters
    ----------
    symbol : str
        NSE symbol (e.g. 'RELIANCE'), index alias ('NIFTY50'), or full
        yfinance ticker ('RELIANCE.NS').

    Returns
    -------
    dict with keys: symbol, ticker, price, prev_close, change_pct, volume,
                    currency, market_state, timestamp
    """
    ticker_str = _nse_ticker(symbol)

    # Serve from TTL cache if available
    cache_key = ticker_str
    if cache_key in _price_cache:
        return _price_cache[cache_key]

    try:
        tk = yf.Ticker(ticker_str)
        info = tk.fast_info  # lightweight; avoids heavy .info call

        price: float = float(info.last_price or 0)
        prev_close: float = float(info.previous_close or 0)
        volume: int = int(info.three_month_average_volume or 0)

        change_pct: float = 0.0
        if prev_close and prev_close != 0:
            change_pct = round((price - prev_close) / prev_close * 100, 2)

        result: dict[str, Any] = {
            "symbol":       symbol.upper(),
            "ticker":       ticker_str,
            "price":        round(price, 2),
            "prev_close":   round(prev_close, 2),
            "change_pct":   change_pct,
            "volume":       volume,
            "currency":     getattr(info, "currency", "INR"),
            "market_state": getattr(info, "market_state", "UNKNOWN"),
            "timestamp":    pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        }

        _price_cache[cache_key] = result
        return result

    except Exception as exc:
        logger.error("get_live_price(%s) failed: %s", symbol, exc)
        raise RuntimeError(f"Could not fetch price for '{symbol}': {exc}") from exc


def get_historical_ohlc(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV historical data for *symbol*.

    Parameters
    ----------
    symbol   : NSE symbol or index alias.
    period   : yfinance period string — '1d','5d','1mo','3mo','6mo','1y','2y','5y','max'.
    interval : yfinance interval — '1m','5m','15m','30m','1h','1d','1wk','1mo'.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date.
    """
    ticker_str = _nse_ticker(symbol)
    try:
        df = yf.download(
            ticker_str,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            raise ValueError(f"No data returned for '{symbol}' ({ticker_str})")
        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:
        logger.error("get_historical_ohlc(%s) failed: %s", symbol, exc)
        raise RuntimeError(f"Could not fetch OHLC for '{symbol}': {exc}") from exc


def get_multiple_prices(symbols: list[str]) -> list[dict[str, Any]]:
    """Batch-fetch live prices for a list of symbols (returns results for each)."""
    results = []
    for sym in symbols:
        try:
            results.append(get_live_price(sym))
        except Exception as exc:
            results.append({"symbol": sym, "error": str(exc)})
    return results


def get_sector_changes() -> dict[str, dict[str, Any]]:
    """
    Return average % price change across each sector in SECTOR_MAP.
    Used by the get_sector_heatmap MCP tool.
    """
    sector_results: dict[str, dict[str, Any]] = {}
    for sector, symbols in SECTOR_MAP.items():
        changes: list[float] = []
        for sym in symbols:
            try:
                data = get_live_price(sym)
                changes.append(data["change_pct"])
            except Exception:
                pass
        avg = round(sum(changes) / len(changes), 2) if changes else 0.0
        sector_results[sector] = {
            "avg_change_pct": avg,
            "stocks_sampled": len(changes),
        }
    return sector_results
