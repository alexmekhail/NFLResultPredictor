import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json

from _nfl import season_summary


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        try:
            threshold = float(params.get("threshold", [0.5])[0])
        except ValueError:
            threshold = 0.5

        try:
            payload = season_summary(threshold)
        except Exception as exc:
            self._send(500, {"error": "summary failed", "detail": repr(exc)})
            return

        self._send(200, payload)

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
