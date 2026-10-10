# Apex Trade

A financial signal terminal that blends WallStreetBets social sentiment with historical price technicals to produce BUY / HOLD / SELL recommendations for stocks.

## Run & Operate

- App runs via the `artifacts/apex-trade: web` workflow (Streamlit, bound to `0.0.0.0` on the artifact's assigned `$PORT`, served at `/`).
- `pnpm --filter @workspace/apex-trade run dev` — start the Streamlit app (uses `$PORT`, falls back to 5000)
- `pnpm --filter @workspace/apex-trade run start` — production start (headless Streamlit)
- Optional secret: `OPENAI_API_KEY` — enables live AI sentiment classification in the Sentiment Lab. Without it, the app degrades gracefully to a rules-based classifier.

## Stack

- Python 3.11 + Streamlit (single-service web artifact)
- Data: `yfinance` for live/historical prices, pandas for processing, plotly for charts
- AI: OpenAI SDK for sentiment classification (model overridden to `gpt-4o-mini`)
- Routed through the shared proxy at `/` via `artifacts/apex-trade/.replit-artifact/artifact.toml`

## Where things live

- `artifacts/apex-trade/app.py` — main Streamlit UI (3 pages: Market Dashboard, Stock Outlook, Sentiment Lab)
- `artifacts/apex-trade/recommender.py` — signal engine blending technicals + sentiment into BUY/HOLD/SELL with conviction
- `artifacts/apex-trade/meme_stock_classifier.py` — OpenAI sentiment classifier (bundle file)
- `artifacts/apex-trade/stock_data.py` — yfinance price/info access (bundle file)
- `artifacts/apex-trade/model_setup.py` — builds a paths dict over bundle files via `os.walk` (bundle file)
- `artifacts/apex-trade/wsb_labeled.csv`, `ticker_prices.csv`, `model_config.json` — bundle data
- `artifacts/apex-trade/.streamlit/config.toml` — server (headless, 0.0.0.0) + dark theme
- `artifacts/apex-trade/.replit-artifact/artifact.toml` — proxy routing + dev/production run commands

## Architecture decisions

- The user-provided bundle is a Python/Streamlit app, so Apex Trade is a Python web artifact. There is no Python artifact template, so a `react-vite` (kind `web`) artifact was created at `/` and its scaffold was replaced with the Streamlit app; the `artifact.toml` run commands were repointed to Streamlit (kind cannot be changed after creation).
- `meme_stock_classifier.py` hardcodes a non-existent model; `app.py` overrides `MODEL` to `gpt-4o-mini` at runtime.
- Preloaded examples use real tickers (MSFT/NVDA/TSLA/GME/QCOM/MP/AMC/AAPL) because the bundle's fictional snapshot tickers do not resolve in yfinance.
- All bundle files are copied into the artifact dir so `model_setup.py`'s `os.walk` stays scoped to the app.

## Product

- **Market Dashboard** — community mood from the labeled WSB sample, a live news ticker, and curated BUY/HOLD/SELL signals across a tracked universe with conviction bars.
- **Stock Outlook** — per-ticker recommendation combining live price technicals and sentiment.
- **Sentiment Lab** — classify arbitrary WSB-style text (AI when `OPENAI_API_KEY` is set, rules-based otherwise).

## User preferences

_None recorded yet._

## Gotchas

- The artifact's run command executes with the artifact directory as CWD — use `pnpm --filter` scripts (CWD-independent), not `cd artifacts/...`.
- Do not edit `artifact.toml` directly; use `verifyAndReplaceArtifactToml`. Kind and `integratedSkills` cannot be changed after creation.
- yfinance requires live network access; fictional tickers will not resolve.

## Pointers

- See the `pnpm-workspace` skill for workspace structure and proxy routing.
