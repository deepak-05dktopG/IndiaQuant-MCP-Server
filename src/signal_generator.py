"""
signal_generator.py — Module 2: AI Trade Signal Generator
===========================================================
Responsibilities:
- Compute RSI, MACD, Bollinger Bands for a given symbol.
- Fetch news headlines from NewsAPI and score sentiment with TextBlob.
- Combine technicals + sentiment into a BUY/SELL/HOLD signal with a
  0–100 confidence score.
- Detect simple chart patterns (Double Top/Bottom, overbought/oversold).
"""

from __future__ import annotations

import os
import logging
import requests
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from textblob import TextBlob
from dotenv import load_dotenv

from src.market_engine import get_historical_ohlc
from src.quant_logic import (
    compute_rsi, compute_macd, compute_bollinger_bands, last
)

load_dotenv()
logger = logging.getLogger(__name__)

NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL: str = "https://newsapi.org/v2/everything"

# ── Scoring weights ──────────────────────────────────────────────────────────
# Total weight = 100 pts
W_RSI       = 25   # RSI zone (oversold/neutral/overbought)
W_MACD      = 25   # MACD histogram direction + crossover
W_BB        = 20   # Bollinger %B position
W_SENTIMENT = 30   # News sentiment score


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_news(query: str, days_back: int = 3) -> list[str]:
    """Fetch recent headlines for *query* from NewsAPI (up to 10 articles)."""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set — skipping sentiment fetch.")
        return []
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q":        query,
        "from":     from_date,
        "language": "en",
        "sortBy":   "publishedAt",
        "pageSize": 10,
        "apiKey":   NEWS_API_KEY,
    }
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=8)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title", "") + ". " + (a.get("description") or "") for a in articles]
    except Exception as exc:
        logger.error("NewsAPI fetch failed: %s", exc)
        return []


def _sentiment_score(headlines: list[str]) -> float:
    """
    Aggregate TextBlob polarity across headlines.
    Returns a float in [-1, 1]:  -1 = very negative, +1 = very positive.
    """
    if not headlines:
        return 0.0
    polarities = [TextBlob(h).sentiment.polarity for h in headlines if h.strip()]
    return sum(polarities) / len(polarities) if polarities else 0.0


def _rsi_score(rsi_val: float) -> tuple[float, str]:
    """Map RSI to a 0–25 bullish score and a textual label."""
    if rsi_val <= 30:
        return W_RSI, "Oversold (bullish)"     # strong buy zone
    elif rsi_val <= 45:
        return W_RSI * 0.65, "Mild oversold"
    elif rsi_val <= 55:
        return W_RSI * 0.5, "Neutral"
    elif rsi_val <= 70:
        return W_RSI * 0.35, "Mild overbought"
    else:
        return 0.0, "Overbought (bearish)"     # strong sell zone


def _macd_score(macd: float, signal: float, histogram: float) -> tuple[float, str]:
    """Map MACD to a 0–25 bullish score."""
    if macd > signal and histogram > 0:
        return W_MACD, "Bullish crossover"
    elif macd > signal:
        return W_MACD * 0.6, "Bullish but weakening"
    elif macd < signal and histogram < 0:
        return 0.0, "Bearish crossover"
    else:
        return W_MACD * 0.4, "Bearish but easing"


def _bb_score(pct_b: float) -> tuple[float, str]:
    """Map Bollinger %B (0-1 typically) to a 0–20 bullish score."""
    if pct_b <= 0.05:
        return W_BB, "At lower band (oversold)"
    elif pct_b <= 0.35:
        return W_BB * 0.7, "Lower half"
    elif pct_b <= 0.65:
        return W_BB * 0.5, "Mid-band"
    elif pct_b <= 0.95:
        return W_BB * 0.3, "Upper half"
    else:
        return 0.0, "At upper band (overbought)"


def _sentiment_to_score(polarity: float) -> tuple[float, str]:
    """Map [-1,1] polarity to a 0–30 bullish score."""
    # Linear scale: polarity=-1 → 0pts, polarity=0 → 15pts, polarity=+1 → 30pts
    score = (polarity + 1) / 2 * W_SENTIMENT
    label = (
        "Very positive" if polarity > 0.4 else
        "Positive" if polarity > 0.1 else
        "Neutral" if polarity > -0.1 else
        "Negative" if polarity > -0.4 else "Very negative"
    )
    return score, label


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_sentiment(symbol: str) -> dict[str, Any]:
    """
    Fetch recent news for *symbol* and return sentiment analysis.

    Returns
    -------
    dict: symbol, polarity, label, headlines (list), signal (BULLISH/NEUTRAL/BEARISH)
    """
    headlines = _fetch_news(symbol)
    polarity  = _sentiment_score(headlines)
    signal    = "BULLISH" if polarity > 0.1 else ("BEARISH" if polarity < -0.1 else "NEUTRAL")
    return {
        "symbol":    symbol.upper(),
        "polarity":  round(polarity, 4),
        "label":     signal,
        "headlines": headlines[:5],   # top 5 for display
        "signal":    signal,
    }


def generate_signal(symbol: str, timeframe: str = "1d") -> dict[str, Any]:
    """
    Combine technical indicators + news sentiment into a BUY/SELL/HOLD signal.

    Parameters
    ----------
    symbol    : NSE symbol (e.g. 'RELIANCE').
    timeframe : yfinance interval — '1d', '1h', '15m', etc.

    Returns
    -------
    dict: symbol, signal (BUY/SELL/HOLD), confidence (0–100), breakdown, indicators
    """
    period = "6mo" if timeframe in ("1d", "1wk") else "5d"
    df = get_historical_ohlc(symbol, period=period, interval=timeframe)

    close = df["Close"].squeeze()

    # Calculate indicators
    rsi_series  = compute_rsi(close)
    macd_dict   = compute_macd(close)
    bb_dict     = compute_bollinger_bands(close)

    rsi_val   = last(rsi_series)
    macd_val  = last(macd_dict["macd"])
    sig_val   = last(macd_dict["signal"])
    hist_val  = last(macd_dict["histogram"])
    pct_b_val = last(bb_dict["%B"])

    # Sentiment
    headlines  = _fetch_news(symbol)
    polarity   = _sentiment_score(headlines)

    # Score each component
    rsi_pts,  rsi_label  = _rsi_score(rsi_val)
    macd_pts, macd_label = _macd_score(macd_val, sig_val, hist_val)
    bb_pts,   bb_label   = _bb_score(pct_b_val)
    sent_pts, sent_label = _sentiment_to_score(polarity)

    confidence = round(rsi_pts + macd_pts + bb_pts + sent_pts, 1)

    # Decide signal
    if confidence >= 62:
        signal = "BUY"
    elif confidence <= 38:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "symbol":     symbol.upper(),
        "timeframe":  timeframe,
        "signal":     signal,
        "confidence": confidence,
        "breakdown": {
            "rsi":       {"score": round(rsi_pts,  2), "label": rsi_label,  "value": round(rsi_val, 2)},
            "macd":      {"score": round(macd_pts, 2), "label": macd_label, "macd": round(macd_val, 4), "signal_line": round(sig_val, 4)},
            "bollinger": {"score": round(bb_pts,   2), "label": bb_label,   "pct_b": round(pct_b_val, 4)},
            "sentiment": {"score": round(sent_pts, 2), "label": sent_label, "polarity": round(polarity, 4)},
        },
        "raw_indicators": {
            "rsi":      round(rsi_val, 2),
            "macd":     round(macd_val, 4),
            "pct_b":    round(pct_b_val, 4),
            "bb_upper": round(last(bb_dict["upper"]), 2),
            "bb_lower": round(last(bb_dict["lower"]), 2),
        },
    }
