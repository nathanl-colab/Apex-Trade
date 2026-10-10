"""
recommender.py  —  Apex Trade signal engine
============================================
Fuses two model components into a single BUY / HOLD / SELL call:

  1. Technical price indicators (from stock_data.py / yfinance):
       - short-term momentum
       - moving-average crossover (5d vs 20d)
       - position within the 52-week range
  2. Social sentiment (from meme_stock_classifier.py + wsb_labeled.csv):
       - net sentiment of WallStreetBets posts that mention the ticker
       - falls back to overall community mood when a ticker is rarely discussed

The blended score lives in [-1, 1]. We bucket it into SELL / HOLD / BUY and
expose every sub-signal so the UI can show *why* a call was made.
"""

import re
import pandas as pd

import stock_data
import meme_stock_classifier as msc


# ----------------------------------------------------------------------------
# Sentiment side
# ----------------------------------------------------------------------------

# Common ticker -> company keywords so we can match posts that mention the
# company by name rather than by $TICKER.
_COMPANY_KEYWORDS = {
    "GME": ["gamestop", "gme"],
    "AMC": ["amc"],
    "TSLA": ["tesla", "tsla", "musk"],
    "NVDA": ["nvidia", "nvda"],
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "QCOM": ["qualcomm", "qcom"],
    "MP": ["mp materials"],
    "SWBI": ["smith & wesson", "smith and wesson", "swbi"],
    "SPCX": ["spacex", "space exploration", "spcx"],
    "MU": ["micron", "mu "],
    "HOOD": ["robinhood", "hood"],
}


def overall_market_mood(df):
    """Net community mood across the whole labeled dataset, in [-1, 1]."""
    summary = msc.get_sentiment_summary(df)
    total = summary.get("total", 0) or 1
    pos = summary["Positive"]["count"]
    neg = summary["Negative"]["count"]
    return (pos - neg) / total, summary


def ticker_sentiment(df, ticker):
    """
    Net sentiment for a single ticker based on posts that mention it.

    Returns dict: score [-1,1], positive, negative, neutral, mentions, basis.
    Falls back to the overall community mood (down-weighted) when the ticker is
    barely discussed.
    """
    ticker = ticker.upper().strip()
    keywords = _COMPANY_KEYWORDS.get(ticker, [ticker.lower()])
    pattern = re.compile(
        r"(\$" + re.escape(ticker) + r"\b)|("
        + "|".join(re.escape(k) for k in keywords) + r")",
        re.IGNORECASE,
    )

    text = (df.get("title", "").fillna("") + " " + df.get("body", "").fillna(""))
    mask = text.apply(lambda t: bool(pattern.search(str(t))))
    hits = df[mask]

    pos = int((hits["sentiment"] == "Positive").sum())
    neg = int((hits["sentiment"] == "Negative").sum())
    neu = int((hits["sentiment"] == "Neutral").sum())
    mentions = len(hits)

    if mentions >= 2:
        score = (pos - neg) / mentions
        basis = "ticker"
    else:
        # Not enough ticker-specific chatter: lean on overall mood, softened.
        mood, _ = overall_market_mood(df)
        score = mood * 0.4
        basis = "market"

    return {
        "score": round(score, 3),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "mentions": mentions,
        "basis": basis,
    }


# ----------------------------------------------------------------------------
# Technical side
# ----------------------------------------------------------------------------

def technical_signals(history, info=None):
    """
    Compute technical indicators from an OHLCV history DataFrame.

    Returns dict with momentum_pct, ma_signal, range_position and a blended
    technical score in [-1, 1]. Tolerates short/empty histories.
    """
    out = {
        "momentum_pct": 0.0,
        "ma_signal": 0.0,
        "range_position": 0.5,
        "score": 0.0,
        "has_data": False,
    }
    if history is None or history.empty or "Close" not in history:
        # Fall back to the snapshot's daily change if we have it.
        if info and info.get("price_change_pct"):
            chg = info["price_change_pct"]
            out["momentum_pct"] = chg
            out["score"] = max(-1.0, min(1.0, chg / 8.0))
        return out

    close = history["Close"].dropna()
    if len(close) < 2:
        return out

    out["has_data"] = True

    # Momentum: total return across the visible window.
    momentum = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0
    out["momentum_pct"] = round(float(momentum), 2)
    momentum_score = max(-1.0, min(1.0, momentum / 15.0))

    # Moving-average crossover (short above long = bullish).
    short_win = min(5, len(close))
    long_win = min(20, len(close))
    short_ma = close.tail(short_win).mean()
    long_ma = close.tail(long_win).mean()
    if long_ma:
        ma_gap = (short_ma - long_ma) / long_ma
        out["ma_signal"] = round(float(ma_gap) * 100, 2)
        ma_score = max(-1.0, min(1.0, ma_gap * 12.0))
    else:
        ma_score = 0.0

    # Position within the visible range (0 = at lows, 1 = at highs).
    lo, hi = close.min(), close.max()
    if hi > lo:
        pos = (close.iloc[-1] - lo) / (hi - lo)
    else:
        pos = 0.5
    out["range_position"] = round(float(pos), 3)
    # Mild mean-reversion: being pinned at the very top is a slight caution.
    range_score = (pos - 0.5) * 0.6

    out["score"] = round(
        0.5 * momentum_score + 0.35 * ma_score + 0.15 * range_score, 3
    )
    out["score"] = max(-1.0, min(1.0, out["score"]))
    return out


# ----------------------------------------------------------------------------
# Blend
# ----------------------------------------------------------------------------

def _bucket(score):
    if score >= 0.18:
        return "BUY"
    if score <= -0.18:
        return "SELL"
    return "HOLD"


def recommend(ticker, labeled_df, history=None, info=None, period="3mo"):
    """
    Produce a full recommendation for a ticker.

    If history/info are not supplied they are fetched live via stock_data.
    Returns a dict consumed directly by the UI.
    """
    ticker = ticker.upper().strip()

    if info is None:
        info = stock_data.get_stock_info(ticker)
    if history is None:
        history = stock_data.get_price_history(ticker, period=period)

    tech = technical_signals(history, info)
    sent = ticker_sentiment(labeled_df, ticker)

    # Whether we actually have live market data to back this call. When price
    # data is unavailable (network/yfinance outage, delisted/invalid ticker),
    # the recommendation rests on sentiment alone and should be flagged so the
    # UI can suppress or clearly mark it rather than present a false signal.
    data_ok = bool(tech["has_data"]) or bool(info.get("current_price"))

    # Technicals carry more weight; sentiment tilts the call.
    blended = 0.6 * tech["score"] + 0.4 * sent["score"]
    blended = max(-1.0, min(1.0, blended))
    action = _bucket(blended) if data_ok else "NO DATA"
    confidence = int(round(min(1.0, abs(blended) * 1.6 + 0.1) * 100))

    return {
        "ticker": ticker,
        "action": action,
        "score": round(blended, 3),
        "confidence": confidence,
        "data_ok": data_ok,
        "technical": tech,
        "sentiment": sent,
        "info": info,
        "history": history,
    }


def rank_universe(tickers, labeled_df, period="3mo"):
    """
    Build a ranked recommendation table for a list of tickers — powers the
    dashboard's curated Buy / Hold / Sell lists. Uses live data per ticker.
    """
    rows = []
    for t in tickers:
        try:
            rec = recommend(t, labeled_df, period=period)
        except Exception:
            continue
        info = rec["info"]
        rows.append({
            "ticker": rec["ticker"],
            "name": info.get("name", rec["ticker"]),
            "action": rec["action"],
            "data_ok": rec["data_ok"],
            "score": rec["score"],
            "confidence": rec["confidence"],
            "price": info.get("current_price", 0.0),
            "change_pct": info.get("price_change_pct", 0.0),
            "momentum_pct": rec["technical"]["momentum_pct"],
            "sentiment_score": rec["sentiment"]["score"],
            "mentions": rec["sentiment"]["mentions"],
            "sector": info.get("sector", "Unknown"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
