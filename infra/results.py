"""Experiment result schema and persistence.

Provides a standard container for experiment outputs with JSON
serialization for reproducibility and paper-readiness.

Usage::

    from experiments.infra.results import ExperimentResult

    result = ExperimentResult(
        experiment_id="EN1.1",
        parameters={"model": "spacy", "threshold": 0.5},
        metrics={"f1": 0.85},
    )
    result.save_json("results/EN1_1.json")

    loaded = ExperimentResult.load_json("results/EN1_1.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class ExperimentResult:
    """Standard container for a single experiment run.

    Attributes:
        experiment_id: Identifier matching the roadmap (e.g. "EN1.1").
        parameters:    All hyperparameters and configuration for this run.
        metrics:       Computed metrics (F1, accuracy, ECE, etc.).
        timestamp:     ISO-8601 string auto-generated at creation time.
        raw_data:      Optional raw predictions, intermediate values, etc.
        environment:   Optional environment snapshot from ``log_environment()``.
        notes:         Optional free-text notes about the run.
    """

    experiment_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_data: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)

    def save_json(self, path: str, indent: int = 2) -> None:
        """Save to a JSON file.

        Args:
            path:   File path to write.
            indent: JSON indentation (default 2).
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str) -> ExperimentResult:
        """Load from a JSON file.

        Args:
            path: File path to read.

        Returns:
            Reconstructed ExperimentResult.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            experiment_id=data["experiment_id"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            timestamp=data.get("timestamp", ""),
            raw_data=data.get("raw_data"),
            environment=data.get("environment"),
            notes=data.get("notes"),
        )
