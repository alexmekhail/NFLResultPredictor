import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json

from _nfl import grade_week, week_files


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        try:
            week = int(params.get("week", [1])[0])
        except ValueError:
            week = 1
        try:
            threshold = float(params.get("threshold", [0.5])[0])
        except ValueError:
            threshold = 0.5

        try:
            if week not in week_files():
                self._send(404, {"error": "Week not found"})
                return
            result = grade_week(week, threshold)
        except Exception as exc:
            self._send(500, {"error": "prediction load failed", "detail": repr(exc)})
            return

        # keep the historical key name for the games list
        self._send(200, {
            "week": result["week"],
            "record": result["record"],
            "predictions": result["games"],
        })

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=60, stale-while-revalidate=300")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass
