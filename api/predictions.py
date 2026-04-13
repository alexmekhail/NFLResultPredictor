from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import glob
import re
from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "Season25" / "data" / "processed"


def get_week_files():
    raw_files = glob.glob(str(DATA_DIR / "predictions_*_wk*.csv"))
    pairs = []
    for f in raw_files:
        m = re.search(r"_wk(\d+)\.csv$", f)
        if m:
            pairs.append((int(m.group(1)), f))
    pairs.sort(key=lambda x: x[0])
    return {w: f for w, f in pairs}


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)

        week = int(params.get("week", [1])[0])
        threshold = float(params.get("threshold", [0.5])[0])

        week_files = get_week_files()
        if week not in week_files:
            self.send_response(404)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Week not found"}).encode())
            return

        df = pd.read_csv(week_files[week])

        # Filter to the requested week if multiple weeks are in the file
        if "week" in df.columns:
            df = df[df["week"] == week].copy()

        # Derive predicted_winner using threshold if missing
        if "predicted_winner" not in df.columns:
            df["predicted_winner"] = np.where(
                df["home_win_prob"] >= threshold,
                df["home_team"],
                df["away_team"],
            )

        # Probability for the predicted winner
        df["predicted_win_prob"] = np.where(
            df["predicted_winner"] == df["home_team"],
            df["home_win_prob"],
            1 - df["home_win_prob"],
        )

        # Confidence: distance from 0.5, scaled to 0–1
        if "confidence" not in df.columns:
            df["confidence"] = (df["predicted_win_prob"] - 0.5).abs() * 2

        df["matchup"] = df["away_team"].astype(str) + " @ " + df["home_team"].astype(str)

        records = df[
            [
                "matchup",
                "home_team",
                "away_team",
                "predicted_winner",
                "home_win_prob",
                "predicted_win_prob",
                "confidence",
            ]
        ].to_dict(orient="records")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"predictions": records}).encode())

    def log_message(self, format, *args):
        pass
