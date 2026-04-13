from http.server import BaseHTTPRequestHandler
import json
import glob
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "Season25" / "data" / "processed"


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
