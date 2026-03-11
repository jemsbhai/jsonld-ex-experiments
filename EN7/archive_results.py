#!/usr/bin/env python
"""Archive experiment results with timestamped filenames.

Copies the latest en7_1_results.json to a timestamped version
in the same directory for permanent archival.

Usage:
    python experiments/EN7/archive_results.py
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def archive_latest():
    latest = RESULTS_DIR / "en7_1_results.json"
    if not latest.exists():
        print(f"No results file found at {latest}")
        return

    with open(latest) as f:
        data = json.load(f)

    ts = data["timestamp"][:19].replace(":", "").replace("-", "").replace("T", "_")
    archived = RESULTS_DIR / f"en7_1_results_{ts}.json"

    if archived.exists():
        print(f"Archive already exists: {archived.name}")
        return

    shutil.copy2(latest, archived)
    print(f"Archived: {archived.name}")

if __name__ == "__main__":
    archive_latest()
