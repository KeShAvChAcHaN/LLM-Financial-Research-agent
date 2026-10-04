"""The tools the agent can call: a JSON schema for the LLM plus a Python function for each."""
import json
import math
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yfinance as yf

import ingest
import rag


@dataclass
class Context:
    """Per-session state the tools need but the LLM must not control."""
    user_id: str = "guest"
    portfolio: pd.DataFrame | None = None           # columns: ticker, quantity, buy_price
    charts: list = field(default_factory=list)      # price series fetched this turn, drawn by the UI
    cache: dict = field(default_factory=dict)       # retrieval cache: don't search the same thing twice


def _num(x, digits=2):
    """Round to a plain float; None for missing/NaN so the JSON stays clean."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else round(x, digits)


# ---------- tool implementations ----------

def get_stock_data(ctx, ticker, period="1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    if hist.empty:
        raise ValueError(
            f"No price data for '{ticker}'. Check the Yahoo Finance symbol "
            "(NSE stocks end in .NS, BSE in .BO, US stocks have no suffix)."
        )
    close = hist["Close"].dropna()
    daily = close.pct_change().dropna()
    try:
        currency = stock.history_metadata.get("currency")
    except Exception:
        currency = None

    ctx.charts.append({
        "ticker": ticker.upper(),
        "currency": currency,
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "close": [round(float(v), 2) for v in close],
    })
    return {
        "ticker": ticker.upper(),
        "currency": currency,
        "period": period,
        "as_of": close.index[-1].strftime("%Y-%m-%d"),
        "last_close": _num(close.iloc[-1]),
        "period_start_close": _num(close.iloc[0]),
        "period_return_pct": _num((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "period_high": _num(close.max()),
        "period_low": _num(close.min()),
        "annualised_volatility_pct": _num(daily.std() * math.sqrt(252) * 100),
        "max_drawdown_pct": _num((close / close.cummax() - 1).min() * 100),
        "sma_50": _num(close.tail(50).mean()) if len(close) >= 50 else None,
        "sma_200": _num(close.tail(200).mean()) if len(close) >= 200 else None,
    }


INFO_FIELDS = [
    "longName", "sector", "industry", "currency", "marketCap", "trailingPE", "forwardPE",
    "priceToBook", "trailingEps", "returnOnEquity", "returnOnAssets", "profitMargins",
    "operatingMargins", "revenueGrowth", "earningsGrowth", "debtToEquity", "currentRatio",
    "dividendYield", "payoutRatio", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
]
STATEMENT_ROWS = {
    "income_stmt": ["Total Revenue", "Operating Income", "Net Income"],
    "balance_sheet": ["Total Debt", "Stockholders Equity", "Cash And Cash Equivalents"],
    "cashflow": ["Operating Cash Flow", "Free Cash Flow"],
}


def get_financials(ctx, ticker):
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    out = {"ticker": ticker.upper(), "ratios": {k: info[k] for k in INFO_FIELDS if info.get(k) is not None}}
    for statement, rows in STATEMENT_ROWS.items():
        df = getattr(stock, statement)
        for row in rows:
            if df is not None and row in df.index:
                series = df.loc[row].dropna()
                out.setdefault("annual", {})[row] = {
                    str(date.year): _num(value, 0) for date, value in series.items()
                }
    if not out["ratios"] and "annual" not in out:
        raise ValueError(f"No fundamentals found for '{ticker}'. Check the Yahoo Finance symbol.")
    return out


def get_news(ctx, query, max_items=6):
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    if not feed.entries:
        raise ValueError(f"No news found for '{query}'.")
    return [
        {
            "title": e.get("title"),
            "source": e.get("source", {}).get("title"),
            "published": e.get("published"),
            "link": e.get("link"),
        }
        for e in feed.entries[:max_items]
    ]


def _fetch_report(company, year):
    """Download and index a report that is not in the knowledge base yet: NSE first, then SEC EDGAR."""
    try:
        ingest.ingest_nse(company, year)
        return
    except ValueError as e:
        nse_error = e
    try:
        ingest.ingest_edgar(company)
    except Exception as e:
        raise ValueError(f"Could not download a report for '{company}'. NSE: {nse_error} SEC EDGAR: {e}")


def search_annual_report(ctx, query, company, year=None):
    company = ingest.nse_symbol(company)
    key = f"{company}|{year}|{query.lower()}"
    if key in ctx.cache:
        return ctx.cache[key]
    chunks = rag.search_reports(query, company, year)
    fetched = f"fetched|{company}|{year}"
    if not chunks and fetched not in ctx.cache:
        _fetch_report(company, year)
        ctx.cache[fetched] = True
        chunks = rag.search_reports(query, company, year)
    if not chunks:
        available = ", ".join(f"{c} {y}" for c, y in rag.list_reports()) or "none"
        raise ValueError(
            f"No annual report indexed for company='{company}' year={year}. Indexed reports: {available}."
        )
    ctx.cache[key] = chunks
    return chunks


def save_memory(ctx, fact, replaces_id=None):
    memory_id = rag.save_memory(fact, ctx.user_id, replaces_id)
    return {"saved": fact, "id": memory_id, "replaced": replaces_id}


def get_portfolio(ctx):
    if ctx.portfolio is None:
        raise ValueError("No portfolio uploaded. Ask the user to upload a holdings CSV in the sidebar.")
    rows = []
    for row in ctx.portfolio.itertuples():
        hist = yf.Ticker(row.ticker).history(period="1y")["Close"].dropna()
        price = float(hist.iloc[-1]) if not hist.empty else None
        value = price * row.quantity if price else None
        rows.append({
            "ticker": row.ticker,
            "quantity": row.quantity,
            "buy_price": row.buy_price,
            "last_price": _num(price),
            "value": _num(value),
            "pnl_pct": _num((price / row.buy_price - 1) * 100) if price else None,
            "volatility_1y_pct": _num(hist.pct_change().std() * math.sqrt(252) * 100) if price else None,
        })
    total = sum(r["value"] for r in rows if r["value"])
    invested = sum(r["quantity"] * r["buy_price"] for r in rows if r["value"])
    for r in rows:
        r["weight_pct"] = _num(r["value"] / total * 100) if r["value"] and total else None
    return {
        "holdings": rows,
        "total_value": _num(total),
        "total_invested": _num(invested),
        "total_pnl_pct": _num((total / invested - 1) * 100) if invested else None,
        "note": "Values are in each stock's own trading currency; totals assume a single currency.",
    }


# ---------- schemas shown to the LLM ----------

TOOLS = [
    {
        "name": "get_stock_data",
        "description": (
            "Get price history statistics for one stock from Yahoo Finance: last close, return over the "
            "period, high/low, annualised volatility, max drawdown and moving averages. Also draws a price "
            "chart for the user. Call it once per ticker; call it for several tickers to compare them."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Yahoo Finance symbol. NSE stocks end in .NS (TCS.NS, RELIANCE.NS), "
                                   "BSE in .BO, US stocks have no suffix (AAPL).",
                },
                "period": {
                    "type": "string",
                    "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                    "description": "Look-back window. Defaults to 1y.",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_financials",
        "description": (
            "Get fundamentals for one stock from Yahoo Finance: valuation and profitability ratios "
            "(P/E, P/B, ROE, margins, debt-to-equity, dividend yield) plus the last few years of revenue, "
            "net income, debt, equity and cash flow. Ratios such as returnOnEquity and profitMargins are "
            "fractions (0.25 = 25%); debtToEquity and dividendYield are already percentages."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"ticker": {"type": "string", "description": "Yahoo Finance symbol, e.g. INFY.NS"}},
            "required": ["ticker"],
        },
    },
    {
        "name": "get_news",
        "description": "Get recent news headlines from Google News for a company or topic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, e.g. 'Tata Motors' or 'RBI repo rate'"},
                "max_items": {"type": "integer", "description": "Number of headlines, default 6"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_annual_report",
        "description": (
            "Search a company's annual report for passages. Use it for qualitative questions the numbers "
            "cannot answer: risks, strategy, management commentary, segment details, outlook. Write a "
            "clear, specific search query and resolve pronouns such as 'they' or 'it' to the company name "
            "using the conversation. Each result carries the page number it came from. The reports already "
            "indexed are listed in the context of each user message. A report that is not indexed yet is "
            "downloaded and indexed automatically on the first search (from NSE for Indian companies, SEC "
            "EDGAR for US ones), which takes a minute or two; later searches are instant."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'key business risks'"},
                "company": {
                    "type": "string",
                    "description": "NSE symbol for an Indian company (TCS, INFY, TATAMOTORS) or US ticker (AAPL)",
                },
                "year": {
                    "type": "integer",
                    "description": (
                        "Year the financial year ends in (FY 2024-25 is 2025). Omit to search all indexed "
                        "years, or to get the latest report when it has to be downloaded."
                    ),
                },
            },
            "required": ["query", "company"],
        },
    },
    {
        "name": "save_memory",
        "description": (
            "Save one durable fact about the user to long-term memory so it is available in future "
            "conversations: holdings, goals, risk appetite, time horizon, preferences. Write it as a short "
            "self-contained sentence. Do not save small talk or facts about companies. If the new fact "
            "updates or contradicts a remembered one, pass that memory's id as replaces_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "e.g. 'User owns 50 TCS shares bought at Rs 3500'"},
                "replaces_id": {"type": "string", "description": "id of the outdated memory this replaces"},
            },
            "required": ["fact"],
        },
    },
    {
        "name": "get_portfolio",
        "description": (
            "Value the holdings CSV the user uploaded: current price, value, profit/loss, weight and "
            "1-year volatility per holding, plus totals. Use it for portfolio, diversification or "
            "concentration questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

_FUNCTIONS = {
    "get_stock_data": get_stock_data,
    "get_financials": get_financials,
    "get_news": get_news,
    "search_annual_report": search_annual_report,
    "save_memory": save_memory,
    "get_portfolio": get_portfolio,
}


def run_tool(name, args, ctx):
    """Execute one tool call. Returns (result_text, is_error); never raises."""
    try:
        result = _FUNCTIONS[name](ctx, **args)
        return json.dumps(result, default=str), False
    except Exception as e:
        return f"Error: {e}", True
