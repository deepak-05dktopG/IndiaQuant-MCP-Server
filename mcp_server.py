"""
mcp_server.py — IndiaQuant MCP Server Entry Point
===================================================
Registers all 10 MCP tools and starts the stdio server
compatible with Claude Desktop and any MCP-capable agent.

Tools:
  1. get_live_price
  2. get_options_chain
  3. analyze_sentiment
  4. generate_signal
  5. get_portfolio_pnl
  6. place_virtual_trade
  7. calculate_greeks
  8. detect_unusual_activity
  9. scan_market
 10. get_sector_heatmap
"""

from __future__ import annotations

# ── PATH FIX: ensure src.* imports resolve no matter what CWD Claude uses ──────
import os as _os
import sys as _sys
_PROJECT_ROOT = _os.path.dirname(_os.path.abspath(__file__))
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
# ────────────────────────────────────────────────────────────────────────────────

import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

# ── Internal modules ───────────────────────────────────────────────────────────
from src.market_engine import (
    get_live_price as _get_live_price,
    get_sector_changes,
    NIFTY50_SYMBOLS,
)
from src.signal_generator import analyze_sentiment as _analyze_sentiment, generate_signal as _generate_signal
from src.options_analyzer import get_options_chain as _get_options_chain, calculate_max_pain, detect_unusual_activity as _detect_unusual_activity
from src.portfolio_manager import place_virtual_trade as _place_virtual_trade, get_portfolio_pnl as _get_portfolio_pnl, scan_market as _scan_market, initialize_db
from src.quant_logic import calculate_greeks as _calculate_greeks

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("indiaquant-mcp")

# ── MCP Server instance ────────────────────────────────────────────────────────
app = Server("indiaquant-mcp")


# ── Tool Definitions ──────────────────────────────────────────────────────────

TOOLS: list[Tool] = [

    Tool(
        name="get_live_price",
        description=(
            "Fetch the live price, day change %, and volume for an NSE/BSE stock or index. "
            "Supports Nifty50 ('NIFTY50'), BankNifty ('BANKNIFTY'), and any NSE symbol like 'RELIANCE'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE symbol (e.g. 'RELIANCE', 'HDFCBANK') or index alias ('NIFTY50', 'BANKNIFTY').",
                },
            },
            "required": ["symbol"],
        },
    ),

    Tool(
        name="get_options_chain",
        description=(
            "Pull the live NSE options chain for a symbol and expiry. "
            "Returns CE and PE contracts with strike, OI, volume, LTP, IV, and Black-Scholes Greeks. "
            "Also returns max pain strike for the expiry."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE F&O symbol — e.g. 'NIFTY', 'BANKNIFTY', 'RELIANCE'.",
                },
                "expiry": {
                    "type": "string",
                    "description": "Expiry date in 'YYYY-MM-DD' format. Omit to use nearest expiry.",
                },
            },
            "required": ["symbol"],
        },
    ),

    Tool(
        name="analyze_sentiment",
        description=(
            "Fetch recent news headlines for a stock and run NLP sentiment analysis. "
            "Returns a polarity score, BULLISH/NEUTRAL/BEARISH label, and top headlines."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE symbol or company name (e.g. 'INFY', 'Infosys').",
                },
            },
            "required": ["symbol"],
        },
    ),

    Tool(
        name="generate_signal",
        description=(
            "Generate a BUY/SELL/HOLD trade signal with a 0–100 confidence score by combining "
            "RSI, MACD, Bollinger Bands, and news sentiment analysis."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "NSE symbol (e.g. 'TCS', 'NIFTY50').",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Candle interval: '1d' (default), '1h', '15m', '5m'.",
                    "default": "1d",
                },
            },
            "required": ["symbol"],
        },
    ),

    Tool(
        name="get_portfolio_pnl",
        description=(
            "Show real-time P&L for all virtual portfolio positions with live CMPs. "
            "Includes total P&L, risk scores, and stop-loss / target breach alerts."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),

    Tool(
        name="place_virtual_trade",
        description=(
            "Execute a virtual BUY or SELL trade at the current live market price. "
            "Persisted to SQLite. Optionally set stop-loss and target prices."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":    {"type": "string",  "description": "NSE symbol (e.g. 'WIPRO')."},
                "qty":       {"type": "number",  "description": "Number of shares to trade."},
                "side":      {"type": "string",  "description": "'BUY' or 'SELL'."},
                "stop_loss": {"type": "number",  "description": "Optional stop-loss price (INR)."},
                "target":    {"type": "number",  "description": "Optional target price (INR)."},
            },
            "required": ["symbol", "qty", "side"],
        },
    ),

    Tool(
        name="calculate_greeks",
        description=(
            "Calculate Black-Scholes option Greeks (Delta, Gamma, Vega, Theta) and theoretical price "
            "for a specific option contract. Implemented from scratch using pure math."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol":      {"type": "string",  "description": "Underlying NSE symbol (e.g. 'NIFTY50')."},
                "strike":      {"type": "number",  "description": "Strike price (INR)."},
                "expiry":      {"type": "string",  "description": "Expiry date 'YYYY-MM-DD'."},
                "option_type": {"type": "string",  "description": "'CE' (Call) or 'PE' (Put)."},
                "sigma":       {"type": "number",  "description": "Implied volatility as decimal (e.g. 0.18 for 18%). Optional — uses historical vol if omitted."},
                "risk_free_rate": {"type": "number", "description": "Annual risk-free rate (e.g. 0.065). Defaults to 6.5%."},
            },
            "required": ["symbol", "strike", "expiry", "option_type"],
        },
    ),

    Tool(
        name="detect_unusual_activity",
        description=(
            "Detect unusual options activity for a symbol — OI spikes and abnormal volume. "
            "Flags contracts with volume ≥ 3× median or OI z-score ≥ 1.5."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "NSE F&O symbol (e.g. 'INFY', 'BANKNIFTY')."},
            },
            "required": ["symbol"],
        },
    ),

    Tool(
        name="scan_market",
        description=(
            "Scan Nifty 50 stocks (or a specific sector) for stocks matching filter criteria "
            "like RSI range and day % change. Example: find oversold IT stocks with RSI < 30."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "rsi_min":        {"type": "number", "description": "Minimum RSI (default 0)."},
                "rsi_max":        {"type": "number", "description": "Maximum RSI (default 100). Set to 30 to find oversold stocks."},
                "min_change_pct": {"type": "number", "description": "Minimum day % change (default -100)."},
                "max_change_pct": {"type": "number", "description": "Maximum day % change (default +100)."},
                "sector":         {"type": "string", "description": "Optional sector: 'IT','Banking','FMCG','Auto','Pharma','Energy','Metals','Financials','Infra','Consumer'."},
            },
            "required": [],
        },
    ),

    Tool(
        name="get_sector_heatmap",
        description=(
            "Show a heatmap of average % price change across major NSE sectors "
            "(IT, Banking, FMCG, Auto, Pharma, Energy, Metals, Financials, Infra, Consumer)."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


# ── Tool Handlers ─────────────────────────────────────────────────────────────

def _ok(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, indent=2, default=str))]


def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}, indent=2))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    logger.info("Tool called: %s | args: %s", name, arguments)
    try:
        # ── 1. get_live_price ───────────────────────────────────────────────
        if name == "get_live_price":
            return _ok(_get_live_price(arguments["symbol"]))

        # ── 2. get_options_chain ────────────────────────────────────────────
        elif name == "get_options_chain":
            chain = _get_options_chain(
                symbol=arguments["symbol"],
                expiry=arguments.get("expiry"),
            )
            max_pain = calculate_max_pain(chain)
            chain["max_pain_strike"] = max_pain["max_pain_strike"]
            return _ok(chain)

        # ── 3. analyze_sentiment ────────────────────────────────────────────
        elif name == "analyze_sentiment":
            return _ok(_analyze_sentiment(arguments["symbol"]))

        # ── 4. generate_signal ──────────────────────────────────────────────
        elif name == "generate_signal":
            return _ok(_generate_signal(
                symbol=arguments["symbol"],
                timeframe=arguments.get("timeframe", "1d"),
            ))

        # ── 5. get_portfolio_pnl ────────────────────────────────────────────
        elif name == "get_portfolio_pnl":
            return _ok(_get_portfolio_pnl())

        # ── 6. place_virtual_trade ──────────────────────────────────────────
        elif name == "place_virtual_trade":
            return _ok(_place_virtual_trade(
                symbol=arguments["symbol"],
                qty=float(arguments["qty"]),
                side=arguments["side"],
                stop_loss=arguments.get("stop_loss"),
                target=arguments.get("target"),
            ))

        # ── 7. calculate_greeks ─────────────────────────────────────────────
        elif name == "calculate_greeks":
            from datetime import date, datetime as dt
            from src.options_analyzer import _implied_vol_estimate, RISK_FREE_RATE, DIV_YIELD
            symbol      = arguments["symbol"]
            strike      = float(arguments["strike"])
            expiry_str  = arguments["expiry"]
            opt_type    = arguments["option_type"]
            sigma       = float(arguments.get("sigma") or _implied_vol_estimate(symbol))
            rfr         = float(arguments.get("risk_free_rate", RISK_FREE_RATE))

            expiry_date = dt.strptime(expiry_str, "%Y-%m-%d").date()
            T_days      = max((expiry_date - date.today()).days, 1)
            spot        = _get_live_price(symbol)["price"]

            greeks = _calculate_greeks(
                S=spot, K=strike, T_days=T_days,
                r=rfr, sigma=sigma,
                option_type=opt_type,
            )
            greeks.update({
                "symbol": symbol.upper(),
                "strike": strike,
                "expiry": expiry_str,
                "type":   opt_type,
                "spot":   spot,
                "sigma_used": round(sigma, 4),
                "T_days": T_days,
            })
            return _ok(greeks)

        # ── 8. detect_unusual_activity ──────────────────────────────────────
        elif name == "detect_unusual_activity":
            return _ok(_detect_unusual_activity(arguments["symbol"]))

        # ── 9. scan_market ─────────────────────────────────────────────────
        elif name == "scan_market":
            return _ok(_scan_market(
                rsi_max=float(arguments.get("rsi_max", 100)),
                rsi_min=float(arguments.get("rsi_min", 0)),
                min_change_pct=float(arguments.get("min_change_pct", -100)),
                max_change_pct=float(arguments.get("max_change_pct", 100)),
                sector=arguments.get("sector"),
            ))

        # ── 10. get_sector_heatmap ──────────────────────────────────────────
        elif name == "get_sector_heatmap":
            return _ok({
                "heatmap":   get_sector_changes(),
                "timestamp": __import__("pandas").Timestamp.now(tz="Asia/Kolkata").isoformat(),
            })

        else:
            return _err(f"Unknown tool: '{name}'")

    except Exception as exc:
        logger.exception("Tool '%s' raised an error", name)
        return _err(str(exc))


# ── Entry Point ───────────────────────────────────────────────────────────────

async def main() -> None:
    initialize_db()
    logger.info("IndiaQuant MCP Server starting …")
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
