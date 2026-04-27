"""Pytest configuration for EN5 tests.

Adds the necessary paths so that:
  - ``from EN5.en5_1_core import ...`` works  (via experiments/ on sys.path)
  - ``from EN5.en5_core import ...``   works  (shared utilities)
  - ``from infra.* import ...``        works  (via experiments/ on sys.path)
  - ``from jsonld_ex.* import ...``    works  (installed package)
"""
from __future__ import annotations

import sys
from pathlib import Path

_en5_dir = Path(__file__).resolve().parent.parent
_experiments_root = _en5_dir.parent

for p in [str(_en5_dir), str(_experiments_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
