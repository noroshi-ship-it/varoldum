"""Varoldum Web Dashboard — serves simulation output as interactive charts."""

import http.server
import json
import csv
import os
import sys
import webbrowser
from urllib.parse import urlparse, parse_qs

OUTPUT_DIR = "output"
PORT = 8420


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
            with open(html_path, "rb") as f:
                self.wfile.write(f.read())
            return

        if path == "/api/population":
            self._serve_csv("population.csv")
            return
        if path == "/api/traits":
            self._serve_csv("traits.csv")
            return
        if path == "/api/consciousness":
            self._serve_csv("consciousness.csv")
            return
        if path == "/api/discovered_rules":
            self._serve_csv("discovered_rules.csv")
            return
        if path == "/api/composable_rules":
            self._serve_csv("composable_rules.csv")
            return
        if path == "/api/hall_of_fame":
            self._serve_json("hall_of_fame.json")
            return
        if path == "/api/events":
            self._serve_json("events.json")
            return
        if path.startswith("/api/snapshot"):
            params = parse_qs(parsed.query)
            tick = params.get("tick", ["100"])[0]
            self._serve_json(f"snapshot_{tick}.json")
            return
        if path == "/api/snapshots":
            self._serve_snapshot_list()
            return

        super().do_GET()

    def _serve_csv(self, filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            self._json_response({"error": f"{filename} not found"}, 404)
            return
        rows = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    try:
                        parsed_row[k] = float(v)
                    except (ValueError, TypeError):
                        parsed_row[k] = v
                rows.append(parsed_row)
        self._json_response(rows)

    def _serve_json(self, filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            self._json_response({"error": f"{filename} not found"}, 404)
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        self._json_response(data)

    def _serve_snapshot_list(self):
        snaps = []
        for f in os.listdir(OUTPUT_DIR):
            if f.startswith("snapshot_") and f.endswith(".json"):
                tick = f.replace("snapshot_", "").replace(".json", "")
                try:
                    snaps.append(int(tick))
                except ValueError:
                    pass
        snaps.sort()
        self._json_response(snaps)

    def _json_response(self, data, code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    global OUTPUT_DIR, PORT
    if len(sys.argv) > 1:
        OUTPUT_DIR = sys.argv[1]
    if len(sys.argv) > 2:
        PORT = int(sys.argv[2])

    if not os.path.exists(OUTPUT_DIR):
        print(f"Output directory '{OUTPUT_DIR}' not found.")
        sys.exit(1)

    server = http.server.HTTPServer(("", PORT), DashboardHandler)
    url = f"http://localhost:{PORT}"
    print(f"Varoldum Dashboard: {url}")
    print(f"Data: {os.path.abspath(OUTPUT_DIR)}")
    print("Ctrl+C to stop")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
