
import json
import os
import csv
from typing import Any


class Logger:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._csv_files: dict[str, Any] = {}
        self._csv_writers: dict[str, csv.DictWriter] = {}
        self._initialized: set[str] = set()

    def log_dict(self, category: str, tick: int, data: dict):
        row = {"tick": tick, **data}

        if category not in self._initialized:
            path = os.path.join(self.output_dir, f"{category}.csv")
            f = open(path, "w", newline="")
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            self._csv_files[category] = f
            self._csv_writers[category] = writer
            self._initialized.add(category)

        self._csv_writers[category].writerow(row)

    def log_snapshot(self, tick: int, agents: list, label: str = "snapshot"):
        if not agents:
            return
        top = sorted(agents, key=lambda a: a.total_reward, reverse=True)[:10]
        data = {
            "tick": tick,
            "agents": [
                {
                    "id": a.id,
                    "generation": a.generation,
                    "age": a.body.age,
                    "energy": float(a.body.energy),
                    "total_reward": float(a.total_reward),
                    "children": a.children_count,
                    "self_model_accuracy": float(a.self_model.cumulative_accuracy),
                    "genome_traits": a.genome[:12].tolist(),
                }
                for a in top
            ],
        }
        path = os.path.join(self.output_dir, f"{label}_{tick}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def flush(self):
        for f in self._csv_files.values():
            f.flush()

    def close(self):
        for f in self._csv_files.values():
            f.close()
