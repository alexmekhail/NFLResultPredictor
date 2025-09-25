# NFLResultPredictor

A predictive system for NFL outcomes, updated and improved for the **2025 season**.  
Uses machine learning to estimate win probabilities and generate picks based on game spreads and historical trends.

---

## 🚀 New for 2025 Season

- Added **prediction of outright winners** (not just probabilities) with configurable thresholds.  
- Streamlit UI enhanced to display matchups, predicted winners, confidence scores, and interactive bar charts.  
- Improved feature extraction with robust handling of nfl_data_py schema changes.  
- End-to-end pipeline for fetching schedules, training models, generating weekly predictions, and visualizing results.  
- Modular design to support richer features in the future (rolling stats, injuries, weather, backtesting, etc.).

---

## 🔧 Setup & Usage

### Prerequisites
- Python 3.11 (recommended for compatibility with dependencies)  
- Git  
- VS Code or another IDE (optional but recommended)

### Installation
```
git clone https://github.com/alexmekhail/NFLResultPredictor.git
cd NFLResultPredictor

python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

cp .env.example .env
```

## 🏋️ Training a Model

```
python scripts/refresh_all.py
```
This fetches current season data, builds features, trains the baseline model, and saves metrics to models/reports/.

## 🎲 Making Predictions
```
python src/predict.py --season 2025 --week 1
```
Generates win probabilities and predicted winners for Week 1, saving results to data/processed/.

## 📊 Viewing Results in the UI
```
streamlit run scripts/serve_streamlit.py
```
Choose a predictions file (e.g. predictions_2025_wk1.csv)

Browse probabilities, picks, and confidence scores

Visualize results with an interactive bar chart

## 📈 Features & Modeling Notes
- Baseline features: betting spread & home-field indicator
- Predictions: threshold-based winner assignment (default 0.5, adjustable)
- Dynamic feature mapping: supports schedule schema changes automatically
- Streamlit dashboard: tabular results, bar charts, and CSV download
- Extensible design: add new features, swap models, or expand evaluation

## 🎯 Roadmap
- Add rolling team statistics (yards/play, turnovers, rest days, EPA/play)
- Incorporate injury and weather data
- Introduce advanced models (XGBoost / LightGBM ensembles)
- Calibrate probability outputs for better accuracy
- Implement walk-forward backtesting for season-long evaluation
-Automate weekly retraining and predictions with GitHub Actions
