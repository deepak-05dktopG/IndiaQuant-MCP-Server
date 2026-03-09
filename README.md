# IndiaQuant MCP Server 🚀

A real-time Indian stock market AI assistant built via the Model Context Protocol (MCP). It connects directly to Claude Desktop (or any MCP-compatible agent) to provide live NSE/BSE market intelligence, Options Greeks calculations (from scratch), portfolio management, and quantitative trade signals.

---

## 🏗️ Architecture & Modules

The system is built on **100% free APIs** (yfinance, NewsAPI, Alpha Vantage) and operates completely locally via SQLite. The architecture consists of 5 core modules exposed through 10 distinct MCP tools.

### 1. Market Data Engine (`market_engine.py`)
- Pulls live prices, volume, and day-change metrics using `yfinance`.
- Automatically handles NSE ticker translation (`.NS` suffix).
- Implements a TTL cache (`cachetools`) to prevent rate-limit bans from rapid consecutive queries by the AI.

### 2. Quant Logic Library (`quant_logic.py`)
- **Black-Scholes from Scratch:** Computes option prices, Delta, Gamma, Vega, and Theta using pure mathematical derivation (via `math`, `numpy`, `scipy.stats.norm`). No black-box options libraries are used.
- Technical indicators (RSI, MACD, Bollinger Bands) derived manually over OHLC data.

### 3. AI Trade Signal Generator (`signal_generator.py`)
- Combines technicals constraints (oversold RSI bounds, MACD bullish crossovers, %B support level) with NewsAPI sentiment analysis.
- NLP sentiment scored via `TextBlob`.
- Aggregates technical and sentiment markers into a weighted 1–100 Confidence Score mapping to a strict `BUY`, `SELL`, or `HOLD` recommendation.

### 4. Options Chain Analyzer (`options_analyzer.py`)
- Live fetching of CE/PE chains.
- Scans up to the 6 nearest expiries.
- Computes **Max Pain** by aggregating intrinsic values across all strikes.
- Detects unusual volume (> 3× median) and OI spikes (z-score > 1.5).

### 5. Portfolio Risk Manager (`portfolio_manager.py`)
- SQLite-backed state management for virtual trades (`data/portfolio.db`).
- Tracks real-time P&L against live CMPs.
- Automatic trailing/breach detection for set Stops & Targets.
- Calculates an automated Risk Score (0-100) per position using unweighted annualised Historical Volatility over 30 days.

---

## 🛠️ The 10 MCP Tools

| Tool Name | Description |
|-----------|-------------|
| `get_live_price` | Fetch live quote, % change, and volume for an NSE symbol or Index. |
| `get_options_chain` | Pull live expiry chain, enriched with LTP, OI, and IV. |
| `analyze_sentiment` | Aggregates and scores recent news headlines. |
| `generate_signal` | Generates a 0-100 confidence Buy/Sell signal. |
| `get_portfolio_pnl` | Real-time P&L table for the virtual portfolio. |
| `place_virtual_trade` | Open a mock long/short position (persisted). |
| `calculate_greeks` | Run pure Black-Scholes math given Spot, Strike, Expiry, and IV. |
| `detect_unusual_activity`| Scan chain for institutional OI/Volume whales. |
| `scan_market` | Filter Nifty50/Sectors by % change and RSI constraints. |
| `get_sector_heatmap` | Average price variation across 10 major NSE sectors. |

---

## 🚀 Setup & Installation

### Requirements
- Python 3.10+
- Claude Desktop installed

### 1. Clone & Install
```bash
git clone https://github.com/your-repo/IndiaQuant-MCP-Server.git
cd IndiaQuant-MCP-Server
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Variables
Create a file named `.env` in the root folder based off `.env.example`:
```ini
NEWS_API_KEY=your_key_here
ALPHA_VANTAGE_KEY=your_key_here
```
*(Both are free tiers. If you omit NewsAPI, sentiment scores will default to Neutral 0.0)*

### 3. Connect to Claude Desktop (Windows)
Open `%APPDATA%\Claude\claude_desktop_config.json` and add:

```json
{
  "mcpServers": {
    "indiaquant": {
      "command": "C:/Path/To/Your/Python/python.exe",
      "args": [
        "C:/Path/To/IndiaQuant-MCP-Server/mcp_server.py"
      ]
    }
  }
}
```
*Note: Make sure to replace the paths above with your actual absolute paths, using forward slashes `/`. Then fully restart Claude Desktop.*

---

## 🛑 Trade-Offs & Known Limitations

**1. Latency & Free APIs**
`yfinance` is heavily utilized. While caching prevents redundant identical calls, fetching full options chains + OHLC logic for Nifty 50 scans concurrent requests can be slow. A production system would ingest realtime UDP multicast ticks instead of polling scraping APIs.

**2. Constant Risk Free Rates**
Currently, `RISK_FREE_RATE` is statically mapped to ~6.5% standard NSE T-Bill yields. A true quant stack would interpolate the zero-coupon yield curve mapped to the exact days-to-expiry for discount factor generation.

**3. Implied Volatility**
yfinance IV data is frequently stale or malformed. When missing, the server degrades gracefully by attempting to calculate 30-day Historical Volatility of the underlying and subbing it into Black-Scholes.

**4. Concurrent DB Writes**
The SQLite database `portfolio.db` uses simple locking. If Claude calls `place_virtual_trade` highly asynchronously in parallel, `sqlite3.OperationalError` (database locked) may occur. This is acceptable for a single-user agentic interface but not for a high-frequency websocket backend.
