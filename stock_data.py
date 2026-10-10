
"""
stock_data.py  —  Inspirit AI · Meme Stock Investigator
=========================================================
Fetches live stock price history and company info using yfinance.
Free, no API key needed.

Public API
----------
    get_price_history(ticker, period)    -> DataFrame of OHLCV price history
    get_stock_info(ticker)               -> dict of company metadata + price
    get_current_price(ticker)            -> float (latest close)
    get_multi_ticker_summary(tickers)    -> DataFrame comparing multiple stocks
    load_ticker_prices()                 -> loads ticker_prices.csv snapshot
"""

import yfinance as yf
import pandas as pd
import os


def get_price_history(ticker, period="1mo", interval="1d"):
    """
    Fetch historical OHLCV price data for a ticker.

    Parameters
    ----------
    ticker   : e.g. "GME", "NVDA", "AAPL"
    period   : "1wk", "1mo", "3mo", "6mo", "1y", "5y"
    interval : "1d" (daily) works for all periods

    Returns a DataFrame with columns Open, High, Low, Close, Volume.
    Returns empty DataFrame on failure.
    """
    try:
        hist = yf.Ticker(ticker.upper()).history(period=period, interval=interval)
        if hist.empty:
            return pd.DataFrame()
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
        hist = hist[cols].copy()
        hist.index.name = "Date"
        return hist
    except Exception as e:
        print(f"[stock_data] Error fetching {ticker}: {e}")
        return pd.DataFrame()


def get_stock_info(ticker):
    """
    Fetch company metadata and latest price for a ticker.

    Returns a dict with keys:
        ticker, name, sector, industry, current_price, previous_close,
        price_change, price_change_pct, market_cap, week_52_high,
        week_52_low, description, error
    """
    try:
        raw      = yf.Ticker(ticker.upper()).info
        current  = (raw.get("currentPrice") or raw.get("regularMarketPrice")
                    or raw.get("previousClose") or 0.0)
        prev     = raw.get("previousClose") or 0.0
        change   = current - prev
        chg_pct  = (change / prev * 100) if prev else 0.0
        return {
            "ticker"          : ticker.upper(),
            "name"            : raw.get("longName") or raw.get("shortName") or ticker.upper(),
            "sector"          : raw.get("sector", "Unknown"),
            "industry"        : raw.get("industry", "Unknown"),
            "current_price"   : round(current, 2),
            "previous_close"  : round(prev, 2),
            "price_change"    : round(change, 2),
            "price_change_pct": round(chg_pct, 2),
            "market_cap"      : raw.get("marketCap"),
            "week_52_high"    : raw.get("fiftyTwoWeekHigh"),
            "week_52_low"     : raw.get("fiftyTwoWeekLow"),
            "description"     : (raw.get("longBusinessSummary") or "")[:400],
            "error"           : None,
        }
    except Exception as e:
        return {
            "ticker": ticker.upper(), "name": ticker.upper(),
            "sector": "Unknown", "industry": "Unknown",
            "current_price": 0.0, "previous_close": 0.0,
            "price_change": 0.0, "price_change_pct": 0.0,
            "market_cap": None, "week_52_high": None,
            "week_52_low": None, "description": "", "error": str(e),
        }


def get_current_price(ticker):
    """Returns just the latest price as a float. Returns 0.0 on failure."""
    return get_stock_info(ticker)["current_price"]


def get_multi_ticker_summary(tickers):
    """
    Fetch a price summary for a list of tickers.
    Returns a DataFrame with ticker, name, current_price, price_change_pct,
    market_cap, week_52_high, week_52_low, sector.
    """
    rows = []
    for t in tickers:
        info = get_stock_info(t)
        rows.append({
            "ticker"          : info["ticker"],
            "name"            : info["name"],
            "current_price"   : info["current_price"] or None,
            "price_change_pct": info["price_change_pct"] or None,
            "market_cap"      : info["market_cap"],
            "week_52_high"    : info["week_52_high"],
            "week_52_low"     : info["week_52_low"],
            "sector"          : info["sector"],
        })
    return pd.DataFrame(rows)


def load_ticker_prices(path=None):
    """
    Load the pre-fetched ticker price snapshot from Notebook 3 (ticker_prices.csv).
    Use this for the initial dashboard load — call get_stock_info() for live updates.
    """
    if path is None:
        try:
            import model_setup
            path = model_setup.paths.get("ticker_prices.csv")
        except ImportError:
            pass
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ticker_prices.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ticker_prices.csv not found at {path}.")
    return pd.read_csv(path)
