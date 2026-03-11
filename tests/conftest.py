"""Pytest configuration for experiment infrastructure tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path so `from experiments.infra import ...` works.
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
