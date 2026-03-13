"""Pytest configuration for EN1 tests.

Adds the necessary paths so that:
  - ``from EN1.en1_3_core import ...`` works (via experiments/ on sys.path)
  - ``from infra.* import ...`` works (via experiments/ on sys.path)
  - ``from jsonld_ex.* import ...`` works (installed package)
"""
from __future__ import annotations

import sys
from pathlib import Path

_experiments_root = Path(__file__).resolve().parent.parent.parent
if str(_experiments_root) not in sys.path:
    sys.path.insert(0, str(_experiments_root))
