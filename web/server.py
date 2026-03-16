
import os
import sys
import json
import csv
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from flask import Flask, jsonify, request, send_from_directory
except ImportError:
    print("Flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))

OUTPUT_DIR = str(Path(__file__).parent.parent / "output")


def read_csv_tail(filepath, n=100):
    rows = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
            return rows[-n:]
    except (FileNotFoundError, StopIteration):
        return []


def read_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_latest_snapshot():
    import glob
    pattern = os.path.join(OUTPUT_DIR, "snapshot_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    return read_json(files[-1])


@app.route('/')
def index():
    return send_from_directory(str(Path(__file__).parent / "static"), "dashboard.html")


@app.route('/api/status')
def api_status():
    pop_data = read_csv_tail(os.path.join(OUTPUT_DIR, "population.csv"), 1)
    events = read_json(os.path.join(OUTPUT_DIR, "events.json"))

    status = {}
    if pop_data:
        row = pop_data[0]
        status = {
            "tick": int(row.get("tick", 0)),
            "population": int(row.get("count", 0)),
            "mean_energy": float(row.get("mean_energy", 0)),
            "max_generation": int(row.get("max_generation", 0)),
            "mean_wm_accuracy": float(row.get("mean_wm_accuracy", 0)),
            "mean_think_steps": float(row.get("mean_think_steps", 0)),
            "mean_bottleneck": float(row.get("mean_bottleneck_size", 0)),
            "mean_self_model": float(row.get("mean_self_model_accuracy", 0)),
            "n_concept_rules": int(row.get("n_concept_rules", 0)),
            "structures": int(row.get("struct_total", 0)),
            "signals_sent": int(row.get("signals_sent", 0)),
            "diversity": float(row.get("diversity", 0)),
            "season": float(row.get("season", 0)),
            "births": int(row.get("births", 0)),
            "deaths": int(row.get("deaths", 0)),
            "total_resource": float(row.get("total_resource", 0)),
        }
    status["n_events"] = len(events.get("events", []))
    status["timestamp"] = time.time()
    return jsonify(status)


@app.route('/api/events')
def api_events():
    since = request.args.get("since", 0, type=int)
    events = read_json(os.path.join(OUTPUT_DIR, "events.json"))
    all_events = events.get("events", [])
    if since > 0:
        all_events = [e for e in all_events if e.get("tick", 0) >= since]
    return jsonify({"events": all_events})


@app.route('/api/agents')
def api_agents():
    snapshot = read_latest_snapshot()
    agents = snapshot.get("agents", [])
    agents.sort(key=lambda a: a.get("total_reward", 0), reverse=True)
    return jsonify({"agents": agents[:50]})


@app.route('/api/population')
def api_population():
    n = request.args.get("n", 200, type=int)
    rows = read_csv_tail(os.path.join(OUTPUT_DIR, "population.csv"), n)
    return jsonify({"data": rows})


@app.route('/api/rules')
def api_rules():
    rows = read_csv_tail(os.path.join(OUTPUT_DIR, "discovered_rules.csv"), 50)
    return jsonify({"rules": rows})


@app.route('/api/consciousness')
def api_consciousness():
    rows = read_csv_tail(os.path.join(OUTPUT_DIR, "consciousness.csv"), 50)
    return jsonify({"data": rows})


@app.route('/api/hall_of_fame')
def api_hall_of_fame():
    data = read_json(os.path.join(OUTPUT_DIR, "hall_of_fame.json"))
    entries = data.get("entries", [])
    category = request.args.get("category", None)
    if category:
        entries = [e for e in entries if e.get("category") == category]
    return jsonify({"entries": entries, "total": len(entries)})


_start_time = time.time()


def get_gpu_info():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,name",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return {
                "gpu_temp": int(parts[0]),
                "gpu_util": int(parts[1]),
                "gpu_mem_used": int(parts[2]),
                "gpu_mem_total": int(parts[3]),
                "gpu_name": parts[4],
            }
    except (FileNotFoundError, subprocess.TimeoutExpired, (IndexError, ValueError)):
        pass
    return {}


@app.route('/api/system')
def api_system():
    uptime = time.time() - _start_time
    gpu = get_gpu_info()
    return jsonify({
        "uptime": uptime,
        "gpu": gpu,
    })


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    global OUTPUT_DIR
    if args.output:
        OUTPUT_DIR = args.output

    print(f"Varoldum Dashboard: http://localhost:{args.port}")
    print(f"Reading from: {OUTPUT_DIR}")
    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
