# NFL Result Predictor

![Project Screenshot](NFLResultPredictor.png)

A machine learning system for predicting NFL game outcomes, built for the **2026 season**.  
Live at **[nfl-result-predictor-am11.vercel.app](https://nfl-result-predictor-am11.vercel.app)**

---

## What it does

- Fetches the full NFL schedule via `nfl_data_py`
- Engineers features from betting spreads and home-field advantage
- Trains a logistic regression model on historical results
- Generates weekly win probabilities and predicted winners
- Serves an interactive web dashboard deployed on Vercel

---

## Web App

The dashboard is deployed at **https://nfl-result-predictor-am11.vercel.app**

Features:
- Week selector (defaults to latest week)
- Adjustable pick threshold (0.40–0.60)
- Sort by confidence, win probability, or home win probability
- Interactive horizontal bar chart per matchup
- CSV download of filtered predictions

---

## Project Structure

```
NFLResultPredictor/
├── api/                        # Vercel serverless functions
│   ├── predictions.py          # GET /api/predictions?week=X&threshold=Y
│   └── weeks.py                # GET /api/weeks
│                               # (both auto-detect the highest-numbered SeasonNN/ folder)
├── public/
│   └── index.html              # Web dashboard frontend
├── Season26/                    # Current season (mirror this folder each year)
│   ├── config.yaml             # season: 2026, train_seasons: [2025]
│   ├── data/
│   │   ├── raw/                # schedule_train_*.csv, schedule_2026.csv
│   │   └── processed/          # predictions_2026_wkN.csv
│   ├── models/
│   │   ├── artifacts/          # baseline_logreg.pkl
│   │   └── reports/            # baseline_metrics.txt
│   ├── scripts/
│   │   ├── refresh_all.py      # Full retrain pipeline
│   │   ├── get_week.py         # CLI wrapper for predictions
│   │   └── serve_streamlit.py  # Legacy local Streamlit UI
│   ├── src/
│   │   ├── data.py             # Schedule fetching & I/O (cwd-independent paths)
│   │   ├── features.py         # Feature engineering
│   │   ├── train.py            # Model training
│   │   └── predict.py          # Prediction generation
│   └── requirements.txt        # Model-pipeline deps (pinned)
├── Season25/                    # Prior season (kept for history)
├── requirements.txt            # Streamlit dashboard deps
└── vercel.json
```

---

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Vercel CLI)

### Setup

```bash
git clone https://github.com/alexmekhail/NFLResultPredictor.git
cd NFLResultPredictor

python3.11 -m venv Season26/.venv
source Season26/.venv/bin/activate      # macOS/Linux
# .\Season26\.venv\Scripts\Activate.ps1  # Windows

pip install -r Season26/requirements.txt
```

Scripts resolve their paths relative to the season folder, so they can be run
from anywhere (from inside `Season26/` or as `python Season26/scripts/...` from
the repo root) and always read/write under `Season26/`.

### Retrain the model

```bash
python Season26/scripts/refresh_all.py
```

Fetches the `train_seasons` schedules (see `Season26/config.yaml`), trains the
model on completed games, and saves metrics to `Season26/models/reports/`.

### Generate predictions for a week

```bash
python Season26/scripts/get_week.py --season 2026 --week 1
```

Saves results to `Season26/data/processed/`. Weeks without a published betting
spread yet will produce fewer (or no) rows until closer to game time.

### Run the local Streamlit UI (legacy)

```bash
pip install -r requirements.txt
streamlit run Season26/scripts/serve_streamlit.py
```

---

## Model

| Detail | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Features | Betting spread (home perspective), home-field indicator |
| Preprocessing | StandardScaler |
| Training data | 2025 season, completed games (228 train / 57 test) |
| Test Accuracy | 67% |
| ROC-AUC | 72% |

Metrics are for the current `Season26/models/reports/baseline_metrics.txt` and
will change when the model is retrained (e.g. once 2026 games are added to
`train_seasons`).

---

## Deploy

```bash
npm i -g vercel
vercel --prod
```

The `api/` directory contains Python serverless functions; `public/` serves the static frontend.

---

## Roadmap

- Rolling team statistics (yards/play, turnovers, rest days, EPA/play)
- Injury and weather data integration
- Advanced models (XGBoost / LightGBM ensembles)
- Probability calibration
- Walk-forward backtesting for season-long evaluation
- Automated weekly retraining via GitHub Actions
