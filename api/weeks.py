from http.server import BaseHTTPRequestHandler
import json
import glob
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_season_dir() -> Path:
    """Highest-numbered SeasonNN/ folder in the repo, so a new season only
    requires adding a SeasonNN/ folder — this file never needs editing."""
    seasons = []
    for p in REPO_ROOT.glob("Season*"):
        if p.is_dir() and re.fullmatch(r"Season\d+", p.name):
            seasons.append(p)
    if not seasons:
        raise RuntimeError(f"No SeasonNN/ folder found under {REPO_ROOT}")
    return max(seasons, key=lambda p: int(p.name.replace("Season", "")))


DATA_DIR = latest_season_dir() / "data" / "processed"


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        raw_files = glob.glob(str(DATA_DIR / "predictions_*_wk*.csv"))
        weeks = []
        for f in raw_files:
            m = re.search(r"_wk(\d+)\.csv$", f)
            if m:
                weeks.append(int(m.group(1)))
        weeks.sort()

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"weeks": weeks}).encode())

    def log_message(self, format, *args):
        pass
