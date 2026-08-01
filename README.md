# Sports Over/Under ML App

Fully functional Streamlit web app that:

- Loads the last ~5 years of **NFL** games with **real closing over/under lines** and final scores (from public nflverse / habitatring data).
- Loads **NBA** and **MLB** scores (via free public APIs) with estimated totals lines for demonstration.
- Shows scores, the over/under line that was offered, and whether the over hit.
- Trains a simple logistic regression model on historical outcomes.
- Reports **historical over-hit percentage** and a **projected probability that the over hits in the next season / future games**.
- Interactive charts (season trends, actual vs line scatter, probability gauge).

## Quick Start (local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Free Hosting (recommended)

### Streamlit Community Cloud (easiest)

1. Create a free GitHub account / new public repository.
2. Upload the contents of this folder (`app.py`, `requirements.txt`, `README.md`).
3. Go to https://share.streamlit.io → "New app" → select the repo → main file `app.py` → Deploy.
4. Your public URL is ready in ~1–2 minutes. It auto-redeploys on every git push.

### Hugging Face Spaces

1. Create a new Space → SDK = Streamlit.
2. Upload the same files.
3. The Space will build and run for free.

## Data Notes

- **NFL**: Real, high-quality closing totals (`total_line`) + scores. Updates as the source CSV is refreshed. Covers regular season + playoffs.
- **NBA / MLB**: Live API pulls (rate-limited). Over/under lines are *estimated* around recent league averages so the ML pipeline can still run end-to-end. For production-grade NBA/MLB lines you would need a paid odds API (The Odds API, SportsDataIO, etc.) and an API key.

## Extending the ML

The current model is intentionally simple (logistic regression on the closing line + season trend). Easy upgrades:

- Add team offensive/defensive ratings, pace, weather (NFL), rest days, etc.
- Switch to XGBoost / LightGBM.
- Walk-forward validation instead of random split.
- Calibrate probabilities (Platt / isotonic).
- Predict the actual total (regression) then convert to P(over) under a distribution.

## Disclaimer

Educational / research tool only. Not betting advice. Past results do not guarantee future outcomes. Always verify lines with licensed sportsbooks.
