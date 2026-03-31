# 🚀 IndiaQuant: Complete Interview Preparation Guide

This guide is designed to help you explain every single file and workflow in your project to an interviewer. It breaks down the architecture, the "why" behind the code, and how data flows from the user to the Indian Stock Market and back.

---

## 🏗️ Architecture Overview

**What is this project?**
IndiaQuant is an **MCP (Model Context Protocol) Server**. MCP is an open standard created by Anthropic that allows AI models (like Claude) to connect to external data sources and tools. 
Instead of Claude just guessing stock prices, this server acts as a "bridge" between Claude and live Indian stock market data (NSE/BSE).

**The Workflow (How it works end-to-end):**
1. **User Prompt**: You type "What is the RSI of Reliance?" into Claude Desktop.
2. **Claude's Decision**: Claude realizes it doesn't know the answer, but it knows the `indiaquant` MCP server has a tool called `generate_signal` or `scan_market`.
3. **MCP Request**: Claude sends a JSON request to `mcp_server.py` via standard input/output (stdio).
4. **Python Execution**: `mcp_server.py` routes the request to the correct module in the `src/` folder (e.g., `signal_generator.py`).
5. **Data Fetch**: The module uses `yfinance` to download live market data from Yahoo Finance.
6. **Math/Logic**: The module calculates the RSI using pure math (pandas/numpy).
7. **MCP Response**: The Python script returns the data as a JSON string back to Claude.
8. **Final Output**: Claude reads the JSON and writes a natural language response to you.

---

## 📂 File-by-File Breakdown

If an interviewer points to a random file, here is exactly what you should say about it:

### 1. `mcp_server.py` (The Director / The Bridge)
*   **What it does:** This is the entry point of the entire application. It runs continuously in the background. It defines all 10 tools that Claude is allowed to use and handles the communication protocol (stdio).
*   **How it works:** It uses the official `@modelcontextprotocol/sdk` (via the `mcp` python package). It decorates functions with `@app.call_tool()` to map a tool name (like `get_live_price`) to the actual python function. It intercepts Claude's requests, pulls the arguments, calls the `src/` modules, and packages the result into `TextContent` JSON to send back.

### 2. `src/market_engine.py` (The Data Gatherer)
*   **What it does:** This handles the raw data fetching. Think of it as the project's direct pipeline to the stock exchanges.
*   **How it works:** It heavily relies on the `yfinance` library. It contains functions like `get_historical_ohlc` (Open, High, Low, Close) and `get_live_price`. It also handles mapping Indian stock symbols properly (appending `.NS` to make sure Yahoo finance knows it's an NSE stock, e.g., `RELIANCE.NS`).

### 3. `src/quant_logic.py` (The Math Brain)
*   **What it does:** This is the pure-math library of your project. It calculates technical indicators and complex theoretical pricing.
*   **How it works:** 
    *   **Technical Indicators:** Uses `pandas` and `numpy` to manually calculate RSI, MACD, and Bollinger Bands based on standard formulas (e.g., Exponential Moving Averages).
    *   **Black-Scholes Options Pricing:** Uses `scipy.stats.norm` to map out the exact mathematical formulas for Options Greeks (Delta, Gamma, Vega, Theta). *Interview Flex: Emphasize that you didn't just use a pre-built library for Greeks; you implemented the core Black-Scholes math equations directly in Python to prove your quant skills.*

### 4. `src/signal_generator.py` (The AI Analyst)
*   **What it does:** This acts like a virtual trader making a decision. It combines math and news to spit out a `BUY`, `SELL`, or `HOLD` signal alongside a confidence score (0-100).
*   **How it works:** 
    *   It imports data from `market_engine.py` and indicators from `quant_logic.py`.
    *   It fetches live news headlines from `NewsAPI`.
    *   It uses `TextBlob` (NLP) to perform sentiment analysis on the news headlines (scoring text as positive or negative).
    *   It assigns weights (e.g., RSI is worth 25 points, News is worth 30 points) and adds them up to generate the final confidence score.

### 5. `src/options_analyzer.py` (The Derivatives Expert)
*   **What it does:** Analyzes complex F&O (Futures & Options) data.
*   **How it works:** It pulls the options chain for an expiry. It calculates "Max Pain" (the strike price where options buyers would lose the most money, widely used by options sellers). It also contains logic `detect_unusual_activity` to flag option contracts that suddenly have massively abnormal Open Interest (OI) or Volume compared to normal levels.

### 6. `src/portfolio_manager.py` (The Virtual Broker / Database)
*   **What it does:** Allows Claude to place "paper trades" (virtual trades) and tracks their performance over time.
*   **How it works:** 
    *   It uses Python's built-in `sqlite3` to maintain a local database file (`portfolio.db`).
    *   It writes `BUY`/`SELL` orders to a `trades` table.
    *   For the P&L (Profit & Loss), it queries the database for all open positions, fetches the *current* live price via `market_engine.py`, and calculates the unrealized profit or loss in real-time.

---

## 🎯 Key Interview Talking Points (Flexes)

When discussing this project, bring up these points to sound senior:

1. **"I built this using an Agentic Architecture."**
   * *Explanation:* Instead of hardcoding a chatbot conversational flow, I gave the AI native tools. The AI decides *when* and *how* to use them dynamically based on user prompts.
2. **"I decoupled the logic into micro-modules."**
   * *Explanation:* Notice how `mcp_server.py` has no business logic. It's just a router. If I wanted to rip out MCP and build a FastAPI web server tomorrow, I wouldn't have to touch `quant_logic.py` or `portfolio_manager.py`. That's clean architecture.
3. **"I implemented core Quant math from scratch."**
   * *Explanation:* Tell them about `quant_logic.py`. Showing you understand *how* RSI or Black-Scholes is calculated under the hood using `numpy` arrays is much more impressive than just typing `import library`.
4. **"It handles real-time data persistence."**
   * *Explanation:* Mention `portfolio_manager.py` and SQLite. It proves you know how to maintain state, write SQL queries, and calculate real-time mark-to-market P&L.
