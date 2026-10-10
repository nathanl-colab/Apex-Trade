# The Trend Terminal: deployment notes for the build agent

This app has a Python backend: it classifies posts with OpenAI and fetches live
prices with yfinance (stock_data.py), so it is NOT a static site. A Python
process must be running for the app to work.

DEPLOYMENT (must do):
- If you split the app into a separate backend service, give that service a
  PRODUCTION run command, not only a development one. A backend with only a dev
  run config starts in the workspace but never starts when published, which
  appears as an "API offline" or PY_API_STATUS error in the live app.
- Confirm the published deployment starts every service (frontend AND Python
  backend) together.

SECRETS:
- The classifier needs an OpenAI API key from the OPENAI_API_KEY environment
  variable. Set it in the DEPLOYMENT (published) Secrets, not just the dev
  workspace. Those are separate in Replit.

KEY FILES:
- meme_stock_classifier.py : sentiment classification
- stock_data.py            : live prices via yfinance
- wsb_labeled.csv          : pre-labeled posts for the initial load
- ticker_prices.csv        : pre-fetched price snapshot
- model_setup.py           : load files via model_setup.paths
