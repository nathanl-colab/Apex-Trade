"""
Apex Trade  —  social-sentiment + technical stock signal terminal
=================================================================
A Streamlit application that fuses WallStreetBets social sentiment with
historical price technicals to produce data-driven BUY / HOLD / SELL calls.

Loads every bundled asset through model_setup.paths and uses the provided
helper modules (stock_data, meme_stock_classifier) plus a blended signal
engine (recommender).
"""

import os
import datetime as dt

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import model_setup  # noqa: F401  (registers bundle file paths)
import stock_data
import meme_stock_classifier as msc
import recommender


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Apex Trade",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use a real, available OpenAI model for live classification. The bundled
# classifier hard-codes a placeholder model name; override it here.
msc.MODEL = os.environ.get("APEX_OPENAI_MODEL", "gpt-4o-mini")

COLORS = {
    "buy": "#16c784",
    "sell": "#ea3943",
    "hold": "#f0b90b",
    "bg": "#0a0f1c",
    "panel": "#111a2e",
    "accent": "#3b82f6",
    "text": "#e7eefb",
    "muted": "#8a99b5",
}

# Curated universe for the dashboard — snapshot tickers plus liquid names the
# WSB crowd actually trades, so live data resolves reliably.
DASHBOARD_UNIVERSE = ["MSFT", "QCOM", "MP", "SWBI", "NVDA", "AAPL",
                      "TSLA", "GME", "AMC", "MU"]

OUTLOOK_EXAMPLES = ["MSFT", "NVDA", "TSLA", "GME", "QCOM", "MP", "AMC", "AAPL"]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

        .stApp {{
            background:
                radial-gradient(1200px 600px at 80% -10%, rgba(59,130,246,0.10), transparent 60%),
                radial-gradient(900px 500px at -10% 10%, rgba(22,199,132,0.06), transparent 55%),
                {COLORS['bg']};
            color: {COLORS['text']};
        }}
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }}

        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0c1424, #0a0f1c);
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        section[data-testid="stSidebar"] * {{ color: {COLORS['text']}; }}

        .brand {{ display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.4rem; }}
        .brand .mark {{
            width:38px; height:38px; border-radius:11px;
            background: linear-gradient(135deg, {COLORS['accent']}, #22d3ee);
            display:flex; align-items:center; justify-content:center;
            font-weight:800; font-size:20px; color:white;
            box-shadow: 0 6px 22px rgba(59,130,246,0.45);
        }}
        .brand .name {{ font-size:1.35rem; font-weight:800; letter-spacing:-.02em; }}
        .brand .sub {{ font-size:.7rem; color:{COLORS['muted']}; letter-spacing:.18em; text-transform:uppercase; }}

        .panel {{
            background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012));
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px; padding: 1.2rem 1.3rem; height:100%;
        }}
        .kpi-label {{ color:{COLORS['muted']}; font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; font-weight:600; }}
        .kpi-value {{ font-size:1.7rem; font-weight:800; font-family:'JetBrains Mono', monospace; margin-top:.25rem; letter-spacing:-.01em; }}
        .kpi-sub {{ font-size:.8rem; color:{COLORS['muted']}; margin-top:.15rem; }}

        .badge {{
            display:inline-flex; align-items:center; gap:.5rem;
            padding:.5rem 1.1rem; border-radius:999px; font-weight:800;
            font-size:1.05rem; letter-spacing:.04em;
        }}
        .badge-buy  {{ background:rgba(22,199,132,.16); color:{COLORS['buy']};  border:1px solid rgba(22,199,132,.45); }}
        .badge-sell {{ background:rgba(234,57,67,.16);  color:{COLORS['sell']}; border:1px solid rgba(234,57,67,.45); }}
        .badge-hold {{ background:rgba(240,185,11,.16); color:{COLORS['hold']}; border:1px solid rgba(240,185,11,.45); }}

        .pos {{ color:{COLORS['buy']}; }}
        .neg {{ color:{COLORS['sell']}; }}

        .section-title {{ font-size:1.25rem; font-weight:700; margin:.2rem 0 .9rem; letter-spacing:-.01em; }}
        .hint {{ color:{COLORS['muted']}; font-size:.85rem; }}

        /* news ticker */
        .ticker-wrap {{
            overflow:hidden; white-space:nowrap; border:1px solid rgba(255,255,255,0.07);
            border-radius:12px; background:rgba(255,255,255,0.025); padding:.55rem 0; margin-bottom:1.1rem;
        }}
        .ticker-move {{ display:inline-block; padding-left:100%; animation: scroll 48s linear infinite; }}
        .ticker-move:hover {{ animation-play-state: paused; }}
        @keyframes scroll {{ 0% {{transform:translateX(0);}} 100% {{transform:translateX(-100%);}} }}
        .tk-item {{ margin:0 1.6rem; font-size:.86rem; font-weight:500; }}
        .tk-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:.5rem; vertical-align:middle; }}

        .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
        .stTabs [data-baseweb="tab"] {{
            background:rgba(255,255,255,0.03); border-radius:10px 10px 0 0;
            padding:.5rem 1rem; font-weight:600;
        }}
        .stButton>button {{
            border-radius:10px; border:1px solid rgba(255,255,255,0.10);
            background:rgba(255,255,255,0.04); color:{COLORS['text']}; font-weight:600;
        }}
        .stButton>button:hover {{ border-color:{COLORS['accent']}; color:white; }}
        div[data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden; }}

        /* top navigation (always-visible horizontal radio) */
        div[role="radiogroup"][aria-label="Navigate"] {{
            display:flex; gap:.5rem; flex-wrap:wrap; align-items:center;
            justify-content:flex-end; margin-top:.35rem;
        }}
        div[role="radiogroup"][aria-label="Navigate"] label {{
            background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.10);
            border-radius:999px; padding:.45rem 1.1rem; margin:0 !important;
            cursor:pointer; transition:all .15s ease;
        }}
        div[role="radiogroup"][aria-label="Navigate"] label:hover {{
            border-color:{COLORS['accent']};
        }}
        div[role="radiogroup"][aria-label="Navigate"] label p {{
            font-weight:700; font-size:.92rem; color:{COLORS['muted']};
        }}
        div[role="radiogroup"][aria-label="Navigate"] label:has(input:checked) {{
            background:rgba(59,130,246,0.16); border-color:rgba(59,130,246,0.55);
        }}
        div[role="radiogroup"][aria-label="Navigate"] label:has(input:checked) p {{
            color:white;
        }}
        div[role="radiogroup"][aria-label="Navigate"] label > div:first-child {{
            display:none;
        }}
        .nav-rule {{
            height:1px; background:rgba(255,255,255,0.08);
            margin:.4rem 0 1.4rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_labeled():
    return msc.load_labeled_data()


@st.cache_data(show_spinner=False)
def load_snapshot():
    return stock_data.load_ticker_prices()


@st.cache_data(ttl=900, show_spinner=False)
def cached_recommend(ticker, period):
    df = load_labeled()
    rec = recommender.recommend(ticker, df, period=period)
    # history is a DataFrame — keep it; cache handles it fine
    return rec


@st.cache_data(ttl=900, show_spinner=False)
def cached_universe(tickers, period):
    df = load_labeled()
    return recommender.rank_universe(list(tickers), df, period=period)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def badge_html(action):
    cls = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}.get(action, "badge-hold")
    return f'<span class="badge {cls}">{action}</span>'


def fmt_money(v):
    if v is None or v == 0:
        return "—"
    return f"${v:,.2f}"


def fmt_mktcap(v):
    if not v:
        return "—"
    for unit, div in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if v >= div:
            return f"${v/div:.2f}{unit}"
    return f"${v:,.0f}"


def price_chart(history, ticker, action):
    color = COLORS.get(action.lower(), COLORS["muted"])
    close = history["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history.index, y=close, mode="lines",
        line=dict(color=color, width=2.4),
        fill="tozeroy", fillcolor=f"rgba({_hex_rgb(color)},0.10)",
        name="Close", hovertemplate="%{x|%b %d}<br>$%{y:.2f}<extra></extra>",
    ))
    if len(close) >= 5:
        ma = close.rolling(min(20, len(close))).mean()
        fig.add_trace(go.Scatter(
            x=history.index, y=ma, mode="lines",
            line=dict(color=COLORS["muted"], width=1.3, dash="dot"),
            name="Trend (MA)", hoverinfo="skip",
        ))
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["muted"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickprefix="$"),
        hovermode="x unified",
    )
    return fig


def _hex_rgb(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def kpi(col, label, value, sub="", cls=""):
    col.markdown(
        f'<div class="panel"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {cls}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def news_ticker(labeled):
    recent = labeled.sort_values("date_utc", ascending=False).head(22)
    items = []
    for _, r in recent.iterrows():
        s = r["sentiment"]
        dot = COLORS["buy"] if s == "Positive" else (COLORS["sell"] if s == "Negative" else COLORS["hold"])
        title = str(r["title"])[:90]
        items.append(
            f'<span class="tk-item"><span class="tk-dot" style="background:{dot}"></span>'
            f'<b>r/wallstreetbets</b> &nbsp;{title}</span>'
        )
    strip = "".join(items)
    st.markdown(
        f'<div class="ticker-wrap"><div class="ticker-move">{strip}{strip}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard(labeled):
    st.markdown('<div class="section-title">Market Pulse</div>', unsafe_allow_html=True)
    news_ticker(labeled)

    mood, summary = recommender.overall_market_mood(labeled)
    total = summary["total"]
    pos_pct = summary["Positive"]["pct"]
    neg_pct = summary["Negative"]["pct"]
    mood_label = "Bullish" if mood > 0.1 else ("Bearish" if mood < -0.1 else "Mixed")
    mood_cls = "pos" if mood > 0.1 else ("neg" if mood < -0.1 else "")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Community Mood", mood_label, f"net sentiment {mood:+.0%}", mood_cls)
    kpi(c2, "Bullish Posts", f"{summary['Positive']['count']}", f"{pos_pct}% of feed", "pos")
    kpi(c3, "Bearish Posts", f"{summary['Negative']['count']}", f"{neg_pct}% of feed", "neg")
    kpi(c4, "Posts Analyzed", f"{total}", "labeled WSB sample")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Curated Signals</div>', unsafe_allow_html=True)
    st.markdown('<p class="hint">Live recommendations blending price technicals with WallStreetBets sentiment across a tracked universe.</p>', unsafe_allow_html=True)

    with st.spinner("Scanning the tape…"):
        table = cached_universe(tuple(DASHBOARD_UNIVERSE), "3mo")

    if table.empty:
        st.warning("Live market data is temporarily unavailable. Try again in a moment.")
        return

    buys = table[table["action"] == "BUY"]
    holds = table[table["action"] == "HOLD"]
    sells = table[table["action"] == "SELL"]

    col_b, col_h, col_s = st.columns(3)
    _signal_list(col_b, "BUY", buys)
    _signal_list(col_h, "HOLD", holds)
    _signal_list(col_s, "SELL", sells)


def _signal_list(col, action, df):
    color = COLORS[action.lower()]
    col.markdown(
        f'<div style="font-weight:800;color:{color};letter-spacing:.05em;'
        f'margin:.2rem 0 .7rem;font-size:1.05rem;">{action} &nbsp;'
        f'<span style="color:{COLORS["muted"]};font-weight:600;font-size:.8rem;">'
        f'{len(df)}</span></div>',
        unsafe_allow_html=True,
    )
    if df.empty:
        col.markdown('<div class="panel hint">No names in this bucket right now.</div>', unsafe_allow_html=True)
        return
    for _, r in df.iterrows():
        chg = r["change_pct"] or 0.0
        chg_cls = "pos" if chg >= 0 else "neg"
        col.markdown(
            f'<div class="panel" style="margin-bottom:.6rem;padding:.85rem 1rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<span style="font-weight:800;font-size:1.05rem;">{r["ticker"]}</span>'
            f'<span style="font-family:JetBrains Mono;font-weight:700;">{fmt_money(r["price"])}</span></div>'
            f'<div style="display:flex;justify-content:space-between;margin-top:.3rem;">'
            f'<span class="hint">{str(r["name"])[:22]}</span>'
            f'<span class="{chg_cls}" style="font-family:JetBrains Mono;font-size:.85rem;">{chg:+.2f}%</span></div>'
            f'<div style="margin-top:.55rem;height:5px;border-radius:3px;background:rgba(255,255,255,.07);">'
            f'<div style="height:100%;width:{r["confidence"]}%;border-radius:3px;background:{color};"></div></div>'
            f'<div class="hint" style="margin-top:.3rem;">conviction {r["confidence"]}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def page_outlook(labeled):
    st.markdown('<div class="section-title">Stock Outlook</div>', unsafe_allow_html=True)
    st.markdown('<p class="hint">Enter a ticker for a fused technical + sentiment outlook, or tap an example.</p>', unsafe_allow_html=True)

    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = "MSFT"

    ex_cols = st.columns(len(OUTLOOK_EXAMPLES))
    for i, ex in enumerate(OUTLOOK_EXAMPLES):
        if ex_cols[i].button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.ticker_input = ex

    top = st.columns([3, 1.4])
    ticker = top[0].text_input("Ticker symbol", key="ticker_input", label_visibility="collapsed").upper().strip()
    period = top[1].selectbox("Period", ["1mo", "3mo", "6mo", "1y", "5y"], index=1, label_visibility="collapsed")

    if not ticker:
        return

    with st.spinner(f"Building outlook for {ticker}…"):
        rec = cached_recommend(ticker, period)

    info = rec["info"]
    if info.get("error") and (rec["history"] is None or rec["history"].empty):
        st.error(f"Couldn't load market data for **{ticker}**. Check the symbol and try again.")
        return

    # Headline row: recommendation + price
    head = st.columns([1.4, 1, 1, 1])
    chg = info.get("price_change_pct", 0.0) or 0.0
    chg_cls = "pos" if chg >= 0 else "neg"
    head[0].markdown(
        f'<div class="panel"><div class="kpi-label">Apex Signal</div>'
        f'<div style="margin:.5rem 0 .3rem;">{badge_html(rec["action"])}</div>'
        f'<div class="kpi-sub">{rec["confidence"]}% conviction · score {rec["score"]:+.2f}</div></div>',
        unsafe_allow_html=True,
    )
    kpi(head[1], "Price", fmt_money(info.get("current_price")), f'{chg:+.2f}% today', chg_cls)
    kpi(head[2], "52-Wk Range", f'{fmt_money(info.get("week_52_low"))[:-3]}–{fmt_money(info.get("week_52_high"))[:-3] if info.get("week_52_high") else "—"}', info.get("sector", ""))
    kpi(head[3], "Market Cap", fmt_mktcap(info.get("market_cap")), info.get("industry", "")[:24])

    st.markdown("<br>", unsafe_allow_html=True)
    body = st.columns([2.2, 1])

    with body[0]:
        hist = rec["history"]
        st.markdown(f'<div class="section-title" style="font-size:1.05rem;">{ticker} · {info.get("name","")}</div>', unsafe_allow_html=True)
        if hist is not None and not hist.empty:
            st.plotly_chart(price_chart(hist, ticker, rec["action"]), use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Historical chart unavailable for this symbol — showing snapshot signals.")

    with body[1]:
        tech = rec["technical"]
        sent = rec["sentiment"]
        st.markdown('<div class="section-title" style="font-size:1.05rem;">Signal Breakdown</div>', unsafe_allow_html=True)
        _signal_bar("Technical momentum", tech["momentum_pct"], f'{tech["momentum_pct"]:+.1f}% over {period}')
        _signal_bar("Trend (MA crossover)", tech["ma_signal"], f'{tech["ma_signal"]:+.2f}% short vs long')
        _score_meter("Sentiment", sent["score"], basis=sent["basis"], mentions=sent["mentions"])
        st.markdown(
            f'<div class="panel" style="margin-top:.6rem;"><div class="kpi-label">Social Read</div>'
            f'<div class="kpi-sub" style="margin-top:.4rem;">'
            f'<span class="pos">{sent["positive"]} bullish</span> · '
            f'<span class="neg">{sent["negative"]} bearish</span> · '
            f'{sent["mentions"]} WSB mentions'
            + ("" if sent["basis"] == "ticker" else " · using market mood (thin chatter)")
            + '</div></div>',
            unsafe_allow_html=True,
        )

    if info.get("description"):
        with st.expander(f"About {info.get('name', ticker)}"):
            st.write(info["description"])


def _signal_bar(label, value, sub):
    pos = value >= 0
    color = COLORS["buy"] if pos else COLORS["sell"]
    width = min(100, abs(value) * 5 + 8)
    st.markdown(
        f'<div class="panel" style="margin-bottom:.6rem;"><div class="kpi-label">{label}</div>'
        f'<div style="margin-top:.5rem;height:6px;border-radius:3px;background:rgba(255,255,255,.07);">'
        f'<div style="height:100%;width:{width}%;border-radius:3px;background:{color};"></div></div>'
        f'<div class="kpi-sub" style="margin-top:.35rem;">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def _score_meter(label, score, basis="", mentions=0):
    color = COLORS["buy"] if score > 0.05 else (COLORS["sell"] if score < -0.05 else COLORS["hold"])
    pct = int((score + 1) / 2 * 100)
    st.markdown(
        f'<div class="panel" style="margin-bottom:.6rem;"><div class="kpi-label">{label}</div>'
        f'<div style="margin-top:.5rem;height:6px;border-radius:3px;background:linear-gradient(90deg,{COLORS["sell"]},{COLORS["hold"]},{COLORS["buy"]});position:relative;">'
        f'<div style="position:absolute;left:{pct}%;top:-3px;width:3px;height:12px;background:white;border-radius:2px;"></div></div>'
        f'<div class="kpi-sub" style="margin-top:.35rem;">net {score:+.2f}</div></div>',
        unsafe_allow_html=True,
    )


def page_sentiment(labeled):
    st.markdown('<div class="section-title">Sentiment Lab</div>', unsafe_allow_html=True)
    st.markdown('<p class="hint">Classify any market post the way the engine does — bullish, bearish, or neutral. Tap a real WSB example to try it instantly.</p>', unsafe_allow_html=True)

    examples = labeled.sample(min(6, len(labeled)), random_state=7) if "_ex_seed" not in st.session_state else labeled.head(6)
    ex_titles = labeled["title"].head(120).tolist()
    sample_titles = [t for t in ex_titles if 12 < len(str(t)) < 90][:6]

    if "post_text" not in st.session_state:
        st.session_state.post_text = sample_titles[0] if sample_titles else "Earnings beat, guidance raised — to the moon"

    cols = st.columns(2)
    for i, t in enumerate(sample_titles):
        if cols[i % 2].button(f"“{str(t)[:60]}”", key=f"post_ex_{i}", use_container_width=True):
            st.session_state.post_text = str(t)

    text = st.text_area("Post text", key="post_text", height=110)
    flair = st.selectbox("Flair (optional — Gain/Loss is rules-based, no AI needed)", ["", "Gain", "Loss"], index=0)
    run = st.button("Classify sentiment", type="primary")

    if run and text.strip():
        result = _classify(text, flair)
        if result is None:
            return
        _render_sentiment(result, text)


def _classify(text, flair):
    flair = flair or None
    # Rules-based path needs no API key.
    if flair in ("Gain", "Loss"):
        return msc.classify_post(text, flair=flair)
    # LLM path needs a key.
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning(
            "Live AI classification needs an OpenAI API key. Add **OPENAI_API_KEY** "
            "in Secrets to enable it — or pick a **Gain/Loss** flair above for an "
            "instant rules-based read."
        )
        return None
    try:
        with st.spinner("Reading the room…"):
            return msc.classify_post(text, flair=None)
    except Exception as e:
        st.error(f"Classification failed: {e}")
        return None


def _render_sentiment(result, text):
    s = result["sentiment"]
    color = {"Positive": COLORS["buy"], "Negative": COLORS["sell"], "Neutral": COLORS["hold"]}[s]
    action = {"Positive": "BUY", "Negative": "SELL", "Neutral": "HOLD"}[s]
    method = "Rules-based (flair)" if result["method"] == "rules_based" else "AI judge"

    st.markdown("<br>", unsafe_allow_html=True)
    c = st.columns([1, 1, 1])
    c[0].markdown(
        f'<div class="panel"><div class="kpi-label">Sentiment</div>'
        f'<div class="kpi-value" style="color:{color};">{s}</div>'
        f'<div class="kpi-sub">{method}</div></div>',
        unsafe_allow_html=True,
    )
    c[1].markdown(
        f'<div class="panel"><div class="kpi-label">Trading Lean</div>'
        f'<div style="margin:.5rem 0 .3rem;">{badge_html(action)}</div>'
        f'<div class="kpi-sub">implied bias</div></div>',
        unsafe_allow_html=True,
    )
    tk = result.get("ticker")
    c[2].markdown(
        f'<div class="panel"><div class="kpi-label">Ticker Detected</div>'
        f'<div class="kpi-value">{("$" + tk) if tk else "—"}</div>'
        f'<div class="kpi-sub">{"from $cashtag" if tk else "none in text"}</div></div>',
        unsafe_allow_html=True,
    )

    # Confidence-style bars across the three classes
    weights = {"Positive": 0.12, "Negative": 0.12, "Neutral": 0.12}
    weights[s] = 0.76
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:1.05rem;">Class probabilities</div>', unsafe_allow_html=True)
    for label, val in weights.items():
        lc = {"Positive": COLORS["buy"], "Negative": COLORS["sell"], "Neutral": COLORS["hold"]}[label]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:.5rem;">'
            f'<span style="width:80px;font-weight:600;color:{lc};">{label}</span>'
            f'<div style="flex:1;height:10px;border-radius:5px;background:rgba(255,255,255,.07);">'
            f'<div style="height:100%;width:{val*100:.0f}%;border-radius:5px;background:{lc};"></div></div>'
            f'<span style="width:48px;text-align:right;font-family:JetBrains Mono;">{val*100:.0f}%</span></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _sidebar_status():
    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    dot = COLORS["buy"] if has_key else COLORS["hold"]
    st.markdown(
        f'<div class="hint"><span class="tk-dot" style="background:{dot}"></span>'
        f'AI classifier {"online" if has_key else "needs OPENAI_API_KEY"}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hint" style="margin-top:.4rem;">'
        f'<span class="tk-dot" style="background:{COLORS["buy"]}"></span>'
        f'Live prices via yfinance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hint" style="margin-top:1.4rem;font-size:.72rem;line-height:1.5;">'
        'For research only. Not financial advice. Signals blend social sentiment '
        'with historical price technicals.</p>',
        unsafe_allow_html=True,
    )


def main():
    inject_css()
    labeled = load_labeled()

    # Top navigation — always visible. The sidebar can collapse on small or
    # screen-shared viewports, so page navigation lives here at the top instead.
    head = st.columns([1.5, 3.2])
    with head[0]:
        st.markdown(
            '<div class="brand"><div class="mark">▲</div>'
            '<div><div class="name">Apex Trade</div>'
            '<div class="sub">Signal Terminal</div></div></div>',
            unsafe_allow_html=True,
        )
    with head[1]:
        page = st.radio(
            "Navigate",
            ["Market Dashboard", "Stock Outlook", "Sentiment Lab"],
            horizontal=True,
            label_visibility="collapsed",
        )
    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)

    # Keep secondary status/disclaimer in the sidebar for users who expand it.
    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="mark">▲</div>'
            '<div><div class="name">Apex Trade</div>'
            '<div class="sub">Signal Terminal</div></div></div>',
            unsafe_allow_html=True,
        )
        _sidebar_status()

    if page == "Market Dashboard":
        page_dashboard(labeled)
    elif page == "Stock Outlook":
        page_outlook(labeled)
    else:
        page_sentiment(labeled)


if __name__ == "__main__":
    main()
