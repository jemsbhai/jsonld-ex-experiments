"""Pytest configuration for EN8 tests.

Adds the necessary paths so that:
  - ``from en8_4b_beir_benchmarks import ...`` works (via EN8/ on sys.path)
  - ``from infra.* import ...`` works (via experiments/ on sys.path)
  - ``from jsonld_ex.* import ...`` works (installed package)
"""
from __future__ import annotations

import sys
from pathlib import Path

_en8_dir = Path(__file__).resolve().parent.parent
_experiments_root = _en8_dir.parent

for p in [str(_en8_dir), str(_experiments_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
