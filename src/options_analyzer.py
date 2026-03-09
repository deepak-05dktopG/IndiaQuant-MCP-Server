"""
options_analyzer.py — Module 3: Options Chain Analyzer
=======================================================
Responsibilities:
- Pull live options chain via yfinance (NSE F&O symbols).
- Calculate Black-Scholes Greeks per contract via quant_logic.
- Detect unusual volume/OI spikes ("unusual activity").
- Calculate max pain point for each expiry.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from src.market_engine import get_live_price, _nse_ticker
from src.quant_logic import calculate_greeks

logger = logging.getLogger(__name__)

# Risk-free rate — approx. NSE 91-day T-bill yield (update periodically)
RISK_FREE_RATE: float = 0.065   # 6.5% p.a.
# Dividend yield assumption for index options
DIV_YIELD: float = 0.011        # 1.1% p.a. for Nifty


def _days_to_expiry(expiry_str: str) -> int:
    """Return calendar days from today to *expiry_str* (format: 'YYYY-MM-DD')."""
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return max((exp - date.today()).days, 1)


def _implied_vol_estimate(symbol: str, period: str = "3mo") -> float:
    """
    Proxy for IV: compute 30-day historical volatility (annualised).
    Used when IV is not directly available from the options chain.
    """
    try:
        from src.market_engine import get_historical_ohlc
        df = get_historical_ohlc(symbol, period=period, interval="1d")
        close = df["Close"].squeeze()
        log_ret = np.log(close / close.shift(1)).dropna()
        hv = float(log_ret.std() * np.sqrt(252))
        return max(hv, 0.05)  # floor at 5%
    except Exception:
        return 0.20   # fallback 20% IV


# ── Public API ────────────────────────────────────────────────────────────────

def get_options_chain(symbol: str, expiry: str | None = None) -> dict[str, Any]:
    """
    Fetch the live NSE options chain for *symbol*.

    Parameters
    ----------
    symbol : NSE symbol (e.g. 'NIFTY', 'RELIANCE') — '.NS' appended automatically.
    expiry : expiry date string 'YYYY-MM-DD'. If None, uses the nearest expiry.

    Returns
    -------
    dict with keys:
      expiry, underlying_price, calls (list), puts (list)
    Each contract dict contains: strike, expiry, type, OI, volume, ltp, iv_pct,
                                  delta, gamma, vega, theta
    """
    ticker_str = _nse_ticker(symbol)
    tk = yf.Ticker(ticker_str)

    # Resolve expiry
    available = tk.options   # tuple of 'YYYY-MM-DD' strings
    if not available:
        raise RuntimeError(f"No options data available for '{symbol}'.")

    if expiry is None:
        expiry = available[0]   # nearest expiry
    elif expiry not in available:
        raise ValueError(
            f"Expiry '{expiry}' not available. Choose from: {list(available[:6])}"
        )

    chain = tk.option_chain(expiry)
    calls_df: pd.DataFrame = chain.calls
    puts_df:  pd.DataFrame = chain.puts

    # Underlying spot price
    spot = get_live_price(symbol)["price"]
    T_days = _days_to_expiry(expiry)
    iv_est = _implied_vol_estimate(symbol)

    def _enrich(row: pd.Series, opt_type: str) -> dict[str, Any]:
        strike = float(row.get("strike", 0))
        iv  = float(row.get("impliedVolatility", iv_est)) or iv_est
        ltp = float(row.get("lastPrice", 0))
        oi  = int(row.get("openInterest", 0))
        vol = int(row.get("volume", 0) or 0)

        try:
            greeks = calculate_greeks(
                S=spot, K=strike, T_days=T_days,
                r=RISK_FREE_RATE, sigma=iv,
                option_type=opt_type,
            )
        except Exception:
            greeks = {"delta": None, "gamma": None, "vega": None, "theta": None, "price": None}

        return {
            "strike":     strike,
            "expiry":     expiry,
            "type":       opt_type,
            "ltp":        round(ltp, 2),
            "oi":         oi,
            "volume":     vol,
            "iv_pct":     round(iv * 100, 2),
            **greeks,
        }

    calls = [_enrich(row, "CE") for _, row in calls_df.iterrows()]
    puts  = [_enrich(row, "PE") for _, row in puts_df.iterrows()]

    return {
        "symbol":           symbol.upper(),
        "expiry":           expiry,
        "underlying_price": spot,
        "days_to_expiry":   T_days,
        "available_expiries": list(available[:6]),
        "calls":            calls,
        "puts":             puts,
    }


def calculate_max_pain(options_chain: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate the max pain strike for an expiry.

    Max pain = the strike at which total option buyer loss (writer profit) is maximised.
    Formula: for each strike K*, sum up intrinsic value of all ITM calls + ITM puts.
    The strike K* that minimises this total is max pain.

    Parameters
    ----------
    options_chain : output from get_options_chain().

    Returns
    -------
    dict: max_pain_strike, expiry, underlying_price, pain_table (sorted by strike)
    """
    calls = options_chain["calls"]
    puts  = options_chain["puts"]
    spot  = options_chain["underlying_price"]

    # Build OI maps by strike
    call_oi: dict[float, int] = {c["strike"]: c["oi"] for c in calls}
    put_oi:  dict[float, int] = {p["strike"]: p["oi"] for p in puts}

    all_strikes = sorted(set(call_oi.keys()) | set(put_oi.keys()))

    pain_table: list[dict[str, Any]] = []
    for K in all_strikes:
        # Total call pain at strike K = Σ max(K - strike_i, 0) * call_OI_i  for all i
        call_pain = sum(
            max(K - s, 0) * call_oi.get(s, 0) for s in all_strikes
        )
        put_pain  = sum(
            max(s - K, 0) * put_oi.get(s, 0)  for s in all_strikes
        )
        pain_table.append({
            "strike":     K,
            "call_pain":  call_pain,
            "put_pain":   put_pain,
            "total_pain": call_pain + put_pain,
        })

    # Min total pain = max pain strike
    min_pain_row = min(pain_table, key=lambda x: x["total_pain"])

    return {
        "symbol":           options_chain["symbol"],
        "expiry":           options_chain["expiry"],
        "underlying_price": spot,
        "max_pain_strike":  min_pain_row["strike"],
        "pain_table":       pain_table,
    }


def detect_unusual_activity(symbol: str) -> dict[str, Any]:
    """
    Detect unusual options activity for *symbol* — OI spikes and volume anomalies.

    Logic:
      - Fetch options chain for nearest expiry.
      - Flag contracts where volume > 3× median volume (across all strikes, that type).
      - Flag contracts where OI crosses a z-score threshold > 1.5.

    Returns
    -------
    dict: symbol, alerts (list of unusual contracts), summary
    """
    chain = get_options_chain(symbol)
    alerts: list[dict[str, Any]] = []

    for opt_type, contracts in [("CE", chain["calls"]), ("PE", chain["puts"])]:
        df = pd.DataFrame(contracts)
        if df.empty:
            continue

        median_vol = df["volume"].median()
        mean_oi    = df["oi"].mean()
        std_oi     = df["oi"].std() or 1.0

        for _, row in df.iterrows():
            reason: list[str] = []
            vol_ratio = (row["volume"] / median_vol) if median_vol > 0 else 0
            oi_z      = (row["oi"] - mean_oi) / std_oi

            if vol_ratio >= 3.0:
                reason.append(f"Volume {vol_ratio:.1f}× median ({int(row['volume'])} vs {int(median_vol)})")
            if oi_z >= 1.5:
                reason.append(f"OI z-score {oi_z:.2f} (OI={int(row['oi'])})")

            if reason:
                alerts.append({
                    "strike":    row["strike"],
                    "type":      opt_type,
                    "ltp":       row["ltp"],
                    "volume":    int(row["volume"]),
                    "oi":        int(row["oi"]),
                    "vol_ratio": round(vol_ratio, 2),
                    "oi_zscore": round(oi_z, 2),
                    "reasons":   reason,
                })

    summary = (
        f"{len(alerts)} unusual contracts detected for {symbol.upper()} "
        f"(expiry: {chain['expiry']})."
    ) if alerts else f"No unusual activity detected for {symbol.upper()}."

    return {
        "symbol":  symbol.upper(),
        "expiry":  chain["expiry"],
        "alerts":  alerts,
        "summary": summary,
    }
