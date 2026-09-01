"""Local preview of the site — emulates the routing in vercel.json so you can
test the dashboard without the Vercel CLI.

    python devserver.py            # http://localhost:8000
    python devserver.py 5000       # custom port

Routes:
    /api/weeks        -> api/weeks.py       (handler class)
    /api/predictions  -> api/predictions.py (handler class)
    /*                -> public/
"""
import importlib.util
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "api"))


def _load_handler(name):
    spec = importlib.util.spec_from_file_location(f"api_{name}", ROOT / "api" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.handler


API = {
    "/api/weeks": _load_handler("weeks"),
    "/api/predictions": _load_handler("predictions"),
}


class Router(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        rel = urlparse(path).path.lstrip("/") or "index.html"
        return str(ROOT / "public" / rel)

    def do_GET(self):
        route = urlparse(self.path).path
        api = API.get(route)
        if api is not None:
            proxy = api.__new__(api)
            proxy.__dict__ = self.__dict__
            return proxy.do_GET()
        if route == "/":
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving http://localhost:{port}  (Ctrl-C to stop)")
    HTTPServer(("127.0.0.1", port), Router).serve_forever()
