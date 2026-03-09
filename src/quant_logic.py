"""
quant_logic.py — Pure-Math Quant Library
=========================================
Implements:
  1. Black-Scholes Greeks (Delta, Gamma, Vega, Theta) using ONLY
     math, numpy, and scipy.stats.norm.  NO vollib, NO mibian, NO shortcuts.

  Exact formulae used (standard European, no dividend yield in Greeks):
    d1    = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
    d2    = d1 - σ·√T
    Delta = N(d1)                              [Call]  |  N(d1)-1  [Put]
    Gamma = φ(d1) / (S·σ·√T)
    Vega  = S·φ(d1)·√T  / 100
    Theta = [−S·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2)] / 365  [Call]

  2. RSI (Relative Strength Index)
  3. MACD (Moving Average Convergence Divergence)
  4. Bollinger Bands
"""

from __future__ import annotations

import math
import numpy as np
from scipy.stats import norm
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BLACK-SCHOLES GREEKS
#     Reference: Black & Scholes (1973)
# ══════════════════════════════════════════════════════════════════════════════

def _bs_d1_d2(
    S: float,      # Current underlying price
    K: float,      # Strike price
    T: float,      # Time to expiry in years
    r: float,      # Risk-free rate (annualised, e.g. 0.065 for 6.5%)
    sigma: float,  # Implied / historical volatility (annualised)
) -> tuple[float, float]:
    """Compute d1 and d2 per the standard Black-Scholes formula."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError(
            f"Invalid BS inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}"
        )
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Black-Scholes theoretical price for a European option.

    Parameters
    ----------
    option_type : 'CE' (Call) or 'PE' (Put).
    """
    d1, d2   = _bs_d1_d2(S, K, T, r, sigma)
    discount = math.exp(-r * T)

    if option_type.upper() in ("CE", "CALL", "C"):
        return S * norm.cdf(d1) - K * discount * norm.cdf(d2)
    elif option_type.upper() in ("PE", "PUT", "P"):
        return K * discount * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Unknown option_type '{option_type}'. Use 'CE' or 'PE'.")


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Delta — rate of change of option price w.r.t. underlying price.
      Call Delta = N(d1)       ∈ (0, 1)
      Put  Delta = N(d1) − 1  ∈ (-1, 0)
    """
    d1, _ = _bs_d1_d2(S, K, T, r, sigma)
    if option_type.upper() in ("CE", "CALL", "C"):
        return norm.cdf(d1)
    else:  # PE / Put
        return norm.cdf(d1) - 1


def bs_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Gamma — rate of change of Delta w.r.t. underlying price.
    Same for calls and puts.
      Γ = φ(d1) / (S · σ · √T)
    """
    d1, _ = _bs_d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(
    S: float, K: float, T: float, r: float, sigma: float,
) -> float:
    """
    Vega — rate of change of option price w.r.t. implied volatility.
    Same for calls and puts. Returned per 1% σ change (÷ 100).
      ν = S · φ(d1) · √T  / 100
    """
    d1, _ = _bs_d1_d2(S, K, T, r, sigma)
    return (S * norm.pdf(d1) * math.sqrt(T)) / 100.0


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: str = "CE",
) -> float:
    """
    Theta — rate of change of option price w.r.t. passage of time.
    Returned per calendar day (divided by 365). Typically negative.

      Θ_call = [−S·φ(d1)·σ/(2√T) − r·K·e^(−rT)·N( d2)] / 365
      Θ_put  = [−S·φ(d1)·σ/(2√T) + r·K·e^(−rT)·N(−d2)] / 365
    """
    d1, d2   = _bs_d1_d2(S, K, T, r, sigma)
    discount = math.exp(-r * T)
    common   = -S * norm.pdf(d1) * sigma / (2 * math.sqrt(T))

    if option_type.upper() in ("CE", "CALL", "C"):
        theta = common - r * K * discount * norm.cdf(d2)
    else:  # PE / Put
        theta = common + r * K * discount * norm.cdf(-d2)

    return theta / 365.0  # per calendar day


def calculate_greeks(
    S: float, K: float, T_days: float, r: float, sigma: float,
    option_type: str = "CE",
) -> dict[str, float]:
    """
    Convenience wrapper — compute all four Greeks for one option contract.

    Parameters
    ----------
    T_days : time to expiry in *calendar days* (converted internally to years).
    """
    T = T_days / 365.0
    return {
        "delta": round(bs_delta(S, K, T, r, sigma, option_type), 6),
        "gamma": round(bs_gamma(S, K, T, r, sigma), 6),
        "vega":  round(bs_vega(S, K, T, r, sigma), 6),
        "theta": round(bs_theta(S, K, T, r, sigma, option_type), 6),
        "price": round(bs_price(S, K, T, r, sigma, option_type), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RSI — Relative Strength Index
# ══════════════════════════════════════════════════════════════════════════════

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Standard Wilder RSI.

    Parameters
    ----------
    close  : pd.Series of closing prices (chronological order).
    period : look-back window, default 14.

    Returns
    -------
    pd.Series of RSI values in [0, 100], same index as *close*.
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # neutral fill until enough data


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MACD — Moving Average Convergence Divergence
# ══════════════════════════════════════════════════════════════════════════════

def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """
    Standard MACD.

    Returns
    -------
    dict with keys:
      'macd'       — MACD line (fast EMA − slow EMA)
      'signal'     — Signal line (EMA of MACD)
      'histogram'  — Histogram (MACD − signal)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  BOLLINGER BANDS
# ══════════════════════════════════════════════════════════════════════════════

def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> dict[str, pd.Series]:
    """
    Bollinger Bands (simple moving average ± k standard deviations).

    Returns
    -------
    dict with keys: 'upper', 'middle' (SMA), 'lower', 'bandwidth', '%B'
    """
    sma   = close.rolling(window=period).mean()
    std   = close.rolling(window=period).std(ddof=0)
    upper = sma + std_dev * std
    lower = sma - std_dev * std

    bandwidth = (upper - lower) / sma
    pct_b     = (close - lower) / (upper - lower + 1e-10)

    return {
        "upper":     upper,
        "middle":    sma,
        "lower":     lower,
        "bandwidth": bandwidth,
        "%B":        pct_b,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  UTILITY — latest scalar from a pd.Series
# ══════════════════════════════════════════════════════════════════════════════

def last(series: pd.Series) -> float:
    """Return the last non-NaN value of a Series as a Python float."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else float("nan")
