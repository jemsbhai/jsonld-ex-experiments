"""Pytest configuration for EN4 tests.

Adds the necessary paths so that:
  - ``from en4_2_ds_comparison import ...`` works (via EN4/ on sys.path)
  - ``from infra.* import ...`` works (via experiments/ on sys.path)
  - ``from jsonld_ex.* import ...`` works (installed package)
"""
from __future__ import annotations

import sys
from pathlib import Path

_en4_dir = Path(__file__).resolve().parent.parent
_experiments_root = _en4_dir.parent

for p in [str(_en4_dir), str(_experiments_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)
