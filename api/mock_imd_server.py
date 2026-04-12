"""
api/mock_imd_server.py — Mock IMD (India Met Dept) Weather Forecast API
=======================================================================
Lightweight HTTP server using Python's built-in http.server.
No fastapi/uvicorn required.

Endpoints:
    GET /forecast?year=2023&day=15&days=3
        Returns JSON array of {rain_mm, temp_c, solar_mj, power_cut} for
        the requested number of days, with realistic forecast noise.

    GET /health
        Returns {"status": "ok", "rows": N}

    GET /summary?year=2023
        Returns season summary stats for a year.

Run:
    python api/mock_imd_server.py            # default port 8765
    python api/mock_imd_server.py --port 9000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
RAINFALL_CSV = os.path.join(DATA_DIR, "rainfall.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Load dataset once at startup
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if os.path.exists(RAINFALL_CSV):
        return pd.read_csv(RAINFALL_CSV)
    # inline synthetic fallback
    rng = np.random.default_rng(0)
    records = []
    for year in range(2021, 2027):
        for day in range(1, 91):
            rain  = float(max(0, rng.gamma(1.2, 4.5)) if rng.random() < .45 else 0)
            temp  = float(np.clip(rng.normal(30, 2.5), 22, 42))
            solar = float(max(5, rng.normal(18 - rain*0.2, 2)))
            records.append({"year": year, "season_day": day,
                            "rainfall_mm": round(rain, 2),
                            "temp_celsius": round(temp, 1),
                            "solar_mj":     round(solar, 1),
                            "power_cut":    int(rng.random() < .10)})
    return pd.DataFrame(records)


WEATHER_DB: pd.DataFrame = pd.DataFrame()  # populated at startup


def get_forecast(year: int, start_day: int, days: int = 3) -> list[dict[str, Any]]:
    """Return forecast with realistic noise."""
    rng    = np.random.default_rng(year * 1000 + start_day)
    subset = WEATHER_DB[WEATHER_DB["year"] == year].reset_index(drop=True)

    if subset.empty:
        # Use nearest available year
        available = WEATHER_DB["year"].unique()
        year      = int(available[np.argmin(np.abs(available - year))])
        subset    = WEATHER_DB[WEATHER_DB["year"] == year].reset_index(drop=True)

    result = []
    for offset in range(days):
        idx = min(start_day - 1 + offset, len(subset) - 1)
        row = subset.iloc[idx]
        # Add forecast uncertainty (further = noisier)
        noise_scale = 1.0 + offset * 0.5
        rain  = float(max(0, row["rainfall_mm"]  + rng.normal(0, 1.5 * noise_scale)))
        temp  = float(row["temp_celsius"] + rng.normal(0, 0.4 * noise_scale))
        solar = float(max(3, row["solar_mj"]     + rng.normal(0, 0.8 * noise_scale)))
        result.append({
            "day":        int(start_day + offset),
            "rain_mm":    round(rain, 2),
            "temp_c":     round(temp, 1),
            "solar_mj":   round(solar, 1),
            "power_cut":  int(row["power_cut"]),
            "year":       year,
            "source":     "mock-imd-v1",
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────

class IMDHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Suppress default verbose logging
        pass

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400):
        self._send_json({"error": msg}, status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        path   = parsed.path.rstrip("/")

        if path == "/health":
            self._send_json({"status": "ok", "rows": len(WEATHER_DB),
                             "years": sorted(WEATHER_DB["year"].unique().tolist())})

        elif path == "/forecast":
            try:
                year      = int(params.get("year", 2026))
                start_day = int(params.get("day",  1))
                days      = int(params.get("days", 3))
                days      = min(days, 7)   # cap at 7
            except ValueError:
                self._send_error("year, day, days must be integers")
                return
            data = get_forecast(year, start_day, days)
            self._send_json(data)

        elif path == "/summary":
            try:
                year = int(params.get("year", 2026))
            except ValueError:
                self._send_error("year must be integer")
                return
            sub = WEATHER_DB[WEATHER_DB["year"] == year]
            if sub.empty:
                self._send_error(f"No data for year {year}", 404)
                return
            self._send_json({
                "year":          year,
                "total_rain_mm": round(float(sub["rainfall_mm"].sum()), 1),
                "mean_temp_c":   round(float(sub["temp_celsius"].mean()), 1),
                "power_cut_days": int(sub["power_cut"].sum()),
                "monsoon_days":  int((sub["rainfall_mm"] > 10).sum()),
                "flash_flood_days": int((sub["rainfall_mm"] > 25).sum()),
            })

        else:
            self._send_error(f"Unknown endpoint: {path}", 404)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def start_server(port: int = 8765, host: str = "localhost"):
    global WEATHER_DB
    WEATHER_DB = load_data()
    print(f"🌦️  Mock IMD API — {len(WEATHER_DB)} weather records loaded")
    print(f"   Listening on http://{host}:{port}")
    print(f"   Endpoints: /health  /forecast?year=2026&day=1&days=3  /summary?year=2026")

    server = HTTPServer((host, port), IMDHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n⛔  Server stopped.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock IMD Weather API Server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()
    start_server(args.port, args.host)
