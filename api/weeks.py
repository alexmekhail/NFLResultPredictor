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

        payload = season_summary(threshold)

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=60, stale-while-revalidate=300")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass
