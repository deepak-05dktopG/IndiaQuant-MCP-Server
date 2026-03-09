"""
portfolio_manager.py — Module 4: Virtual Portfolio & Risk Manager
==================================================================
Responsibilities:
- Maintain a virtual portfolio via SQLite (positions + cash).
- Place virtual trades (buy/sell) with order IDs.
- Real-time P&L from live prices.
- Auto stop-loss / target hit detection.
- Risk score per position (based on historical volatility).
"""

from __future__ import annotations

import os
import uuid
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

from dotenv import load_dotenv

from src.market_engine import get_live_price, get_historical_ohlc

load_dotenv()
logger = logging.getLogger(__name__)

DB_PATH: str = os.getenv("DB_PATH", "data/portfolio.db")
INITIAL_CASH: float = float(os.getenv("PORTFOLIO_INITIAL_CASH", "1000000"))


# ── DB Setup ──────────────────────────────────────────────────────────────────

def _ensure_db_dir() -> None:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id            INTEGER PRIMARY KEY,
                cash          REAL    NOT NULL DEFAULT 1000000.0,
                updated_at    TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS positions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol        TEXT    NOT NULL,
                qty           REAL    NOT NULL,
                avg_price     REAL    NOT NULL,
                side          TEXT    NOT NULL CHECK(side IN ('BUY','SELL')),
                stop_loss     REAL,
                target        REAL,
                opened_at     TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id      TEXT    PRIMARY KEY,
                symbol        TEXT    NOT NULL,
                qty           REAL    NOT NULL,
                side          TEXT    NOT NULL,
                price         REAL    NOT NULL,
                status        TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            );
        """)
        # Seed cash row if empty
        row = conn.execute("SELECT id FROM portfolio LIMIT 1").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO portfolio (cash, updated_at) VALUES (?, ?)",
                (INITIAL_CASH, _now()),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_cash(conn: sqlite3.Connection) -> float:
    row = conn.execute("SELECT cash FROM portfolio ORDER BY id LIMIT 1").fetchone()
    return float(row["cash"]) if row else INITIAL_CASH


def _update_cash(conn: sqlite3.Connection, new_cash: float) -> None:
    conn.execute("UPDATE portfolio SET cash = ?, updated_at = ?", (new_cash, _now()))


def _historical_volatility(symbol: str) -> float:
    """Annualised 30-day historical volatility (used in risk scoring)."""
    import numpy as np
    try:
        df = get_historical_ohlc(symbol, period="3mo", interval="1d")
        close = df["Close"].squeeze()
        log_ret = (close / close.shift(1)).apply(lambda x: float("nan") if x <= 0 else __import__("math").log(x)).dropna()
        return float(log_ret.std() * (252 ** 0.5))
    except Exception:
        return 0.25  # fallback 25%


# ── Public API ────────────────────────────────────────────────────────────────

def place_virtual_trade(
    symbol: str,
    qty: float,
    side: str,
    stop_loss: float | None = None,
    target: float | None = None,
) -> dict[str, Any]:
    """
    Execute a virtual trade and persist it to SQLite.

    Parameters
    ----------
    symbol    : NSE symbol (e.g. 'RELIANCE').
    qty       : Number of shares / lots.
    side      : 'BUY' or 'SELL'.
    stop_loss : Optional stop-loss price (INR).
    target    : Optional target price (INR).

    Returns
    -------
    dict: order_id, symbol, qty, side, price, status, cash_remaining
    """
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("side must be 'BUY' or 'SELL'")

    price_data = get_live_price(symbol)
    exec_price = price_data["price"]
    order_id   = str(uuid.uuid4())[:12].upper()
    order_value = exec_price * qty

    initialize_db()
    with _get_conn() as conn:
        cash = _get_cash(conn)

        if side == "BUY":
            if order_value > cash:
                raise ValueError(
                    f"Insufficient cash: need ₹{order_value:,.2f}, have ₹{cash:,.2f}"
                )
            new_cash = cash - order_value
        else:
            # SELL — add proceeds (short-selling allowed for simplicity)
            new_cash = cash + order_value

        # Upsert position
        existing = conn.execute(
            "SELECT id, qty, avg_price FROM positions WHERE symbol = ? AND side = ?",
            (symbol.upper(), side),
        ).fetchone()

        if existing:
            new_qty   = existing["qty"] + qty
            new_avg   = (existing["avg_price"] * existing["qty"] + exec_price * qty) / new_qty
            conn.execute(
                "UPDATE positions SET qty = ?, avg_price = ?, stop_loss = ?, target = ? WHERE id = ?",
                (new_qty, new_avg, stop_loss, target, existing["id"]),
            )
        else:
            conn.execute(
                "INSERT INTO positions (symbol, qty, avg_price, side, stop_loss, target, opened_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (symbol.upper(), qty, exec_price, side, stop_loss, target, _now()),
            )

        _update_cash(conn, new_cash)

        conn.execute(
            "INSERT INTO orders (order_id, symbol, qty, side, price, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'EXECUTED', ?)",
            (order_id, symbol.upper(), qty, side, exec_price, _now()),
        )

    return {
        "order_id":       order_id,
        "symbol":         symbol.upper(),
        "qty":            qty,
        "side":           side,
        "exec_price":     round(exec_price, 2),
        "order_value":    round(order_value, 2),
        "status":         "EXECUTED",
        "cash_remaining": round(new_cash, 2),
        "stop_loss":      stop_loss,
        "target":         target,
        "timestamp":      _now(),
    }


def get_portfolio_pnl() -> dict[str, Any]:
    """
    Return real-time P&L for all virtual positions.

    Returns
    -------
    dict: cash, positions (list with live P&L), total_invested, total_current_value,
          total_pnl, total_pnl_pct, risk_alerts (stop-loss / target breaches)
    """
    initialize_db()
    with _get_conn() as conn:
        cash = _get_cash(conn)
        rows = conn.execute(
            "SELECT * FROM positions ORDER BY opened_at"
        ).fetchall()

    positions_out: list[dict[str, Any]] = []
    total_invested = 0.0
    total_current  = 0.0
    risk_alerts:   list[str] = []

    for row in rows:
        symbol = row["symbol"]
        qty    = float(row["qty"])
        avg    = float(row["avg_price"])
        side   = row["side"]
        sl     = float(row["stop_loss"]) if row["stop_loss"] else None
        tgt    = float(row["target"])    if row["target"]    else None

        try:
            live = get_live_price(symbol)
            cmp  = live["price"]
        except Exception:
            cmp = avg   # fallback to cost price

        invested = avg * qty
        current  = cmp * qty
        pnl      = (current - invested) if side == "BUY" else (invested - current)
        pnl_pct  = round(pnl / invested * 100, 2) if invested else 0.0

        # Risk alerts
        if sl and side == "BUY" and cmp <= sl:
            risk_alerts.append(f"⚠️ STOP-LOSS HIT: {symbol} CMP {cmp} ≤ SL {sl}")
        if tgt and side == "BUY" and cmp >= tgt:
            risk_alerts.append(f"🎯 TARGET HIT: {symbol} CMP {cmp} ≥ Target {tgt}")

        # Risk score: based on annualised HV
        hv = _historical_volatility(symbol)
        risk_score = min(round(hv * 100, 1), 100)  # 0–100

        total_invested += invested
        total_current  += current

        positions_out.append({
            "symbol":       symbol,
            "qty":          qty,
            "avg_price":    round(avg, 2),
            "cmp":          round(cmp, 2),
            "side":         side,
            "invested":     round(invested, 2),
            "current_val":  round(current, 2),
            "pnl":          round(pnl, 2),
            "pnl_pct":      pnl_pct,
            "stop_loss":    sl,
            "target":       tgt,
            "risk_score":   risk_score,   # 0=low, 100=high
            "hv_pct":       round(hv * 100, 2),
        })

    total_pnl     = total_current - total_invested
    total_pnl_pct = round(total_pnl / total_invested * 100, 2) if total_invested else 0.0
    portfolio_val = cash + total_current

    return {
        "cash":              round(cash, 2),
        "total_invested":    round(total_invested, 2),
        "total_current_val": round(total_current, 2),
        "total_pnl":         round(total_pnl, 2),
        "total_pnl_pct":     total_pnl_pct,
        "portfolio_value":   round(portfolio_val, 2),
        "positions":         positions_out,
        "risk_alerts":       risk_alerts,
        "timestamp":         _now(),
    }


def scan_market(
    rsi_max: float = 100,
    rsi_min: float = 0,
    min_change_pct: float = -100,
    max_change_pct: float = 100,
    sector: str | None = None,
) -> dict[str, Any]:
    """
    Scan Nifty 50 (or a specific sector) for stocks matching filter criteria.

    Parameters
    ----------
    rsi_max        : Maximum RSI threshold (e.g. 30 for oversold).
    rsi_min        : Minimum RSI threshold.
    min_change_pct : Minimum day % change.
    max_change_pct : Maximum day % change.
    sector         : Optional sector name to restrict scan.

    Returns
    -------
    dict: filter_criteria, matches (list), total_scanned
    """
    from src.market_engine import NIFTY50_SYMBOLS, SECTOR_MAP
    from src.quant_logic import compute_rsi, last
    from src.market_engine import get_historical_ohlc

    symbols = SECTOR_MAP.get(sector, NIFTY50_SYMBOLS) if sector else NIFTY50_SYMBOLS
    matches: list[dict[str, Any]] = []

    for sym in symbols:
        try:
            price_data = get_live_price(sym)
            chg = price_data["change_pct"]
            if not (min_change_pct <= chg <= max_change_pct):
                continue

            # RSI
            df    = get_historical_ohlc(sym, period="3mo", interval="1d")
            close = df["Close"].squeeze()
            rsi   = last(compute_rsi(close))

            if rsi_min <= rsi <= rsi_max:
                matches.append({
                    "symbol":     sym,
                    "price":      price_data["price"],
                    "change_pct": chg,
                    "rsi":        round(rsi, 2),
                })
        except Exception as exc:
            logger.debug("scan_market skip %s: %s", sym, exc)

    return {
        "filter_criteria": {
            "rsi_min": rsi_min,
            "rsi_max": rsi_max,
            "min_change_pct": min_change_pct,
            "max_change_pct": max_change_pct,
            "sector": sector,
        },
        "total_scanned": len(symbols),
        "matches":       matches,
    }
