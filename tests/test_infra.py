"""Tests for experiment infrastructure utilities.

RED phase: all tests should fail until infra modules are implemented.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest


# ═══════════════════════════════════════════════════════════════════
# config.py — Reproducibility seed management
# ═══════════════════════════════════════════════════════════════════


class TestSeedConfig:
    """Global seed configuration for reproducibility."""

    def test_set_global_seed_makes_random_deterministic(self) -> None:
        """After set_global_seed(), random.random() is deterministic."""
        import random
        from experiments.infra.config import set_global_seed

        set_global_seed(42)
        a = [random.random() for _ in range(5)]

        set_global_seed(42)
        b = [random.random() for _ in range(5)]

        assert a == b

    def test_set_global_seed_makes_numpy_deterministic(self) -> None:
        """After set_global_seed(), numpy random is deterministic."""
        import numpy as np
        from experiments.infra.config import set_global_seed

        set_global_seed(42)
        a = np.random.rand(5).tolist()

        set_global_seed(42)
        b = np.random.rand(5).tolist()

        assert a == b

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds produce different random sequences."""
        import random
        from experiments.infra.config import set_global_seed

        set_global_seed(42)
        a = [random.random() for _ in range(5)]

        set_global_seed(99)
        b = [random.random() for _ in range(5)]

        assert a != b

    def test_get_global_seed_returns_set_value(self) -> None:
        """get_global_seed() returns the most recently set seed."""
        from experiments.infra.config import set_global_seed, get_global_seed

        set_global_seed(123)
        assert get_global_seed() == 123

    def test_get_global_seed_default_is_none(self) -> None:
        """Before any set_global_seed(), get_global_seed() returns None."""
        from experiments.infra import config as cfg

        # Reset module state
        cfg._global_seed = None
        assert cfg.get_global_seed() is None


# ═══════════════════════════════════════════════════════════════════
# env_log.py — Environment capture
# ═══════════════════════════════════════════════════════════════════


class TestEnvLog:
    """Capture reproducibility-critical environment details."""

    def test_log_environment_returns_dict(self) -> None:
        """log_environment() returns a dict."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        assert isinstance(env, dict)

    def test_log_environment_has_python_version(self) -> None:
        """Contains Python version string."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        assert "python_version" in env
        assert isinstance(env["python_version"], str)
        assert "." in env["python_version"]  # e.g. "3.11.5"

    def test_log_environment_has_os_info(self) -> None:
        """Contains OS platform string."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        assert "platform" in env
        assert isinstance(env["platform"], str)

    def test_log_environment_has_jsonld_ex_version(self) -> None:
        """Contains the installed jsonld-ex version."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        assert "jsonld_ex_version" in env

    def test_log_environment_has_cpu_info(self) -> None:
        """Contains CPU information."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        assert "cpu" in env

    def test_log_environment_has_ram(self) -> None:
        """Contains RAM information in GB."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        assert "ram_gb" in env
        assert isinstance(env["ram_gb"], (int, float))
        assert env["ram_gb"] > 0

    def test_log_environment_is_json_serializable(self) -> None:
        """The entire environment dict can be JSON-serialized."""
        from experiments.infra.env_log import log_environment

        env = log_environment()
        serialized = json.dumps(env)
        assert isinstance(serialized, str)


# ═══════════════════════════════════════════════════════════════════
# results.py — Experiment result schema and persistence
# ═══════════════════════════════════════════════════════════════════


class TestExperimentResult:
    """Standard experiment result container."""

    def test_create_result(self) -> None:
        """Can construct an ExperimentResult."""
        from experiments.infra.results import ExperimentResult

        r = ExperimentResult(
            experiment_id="EN1.1",
            parameters={"model": "spacy", "threshold": 0.5},
            metrics={"f1": 0.85, "precision": 0.90},
        )
        assert r.experiment_id == "EN1.1"
        assert r.metrics["f1"] == 0.85

    def test_result_has_timestamp(self) -> None:
        """ExperimentResult auto-generates a timestamp."""
        from experiments.infra.results import ExperimentResult

        r = ExperimentResult(
            experiment_id="EN1.1",
            parameters={},
            metrics={},
        )
        assert r.timestamp is not None
        assert isinstance(r.timestamp, str)

    def test_save_and_load_roundtrip(self) -> None:
        """save_json() / load_json() round-trip preserves data."""
        from experiments.infra.results import ExperimentResult

        r = ExperimentResult(
            experiment_id="EN1.1",
            parameters={"seed": 42, "model": "spacy"},
            metrics={"f1": 0.85, "precision": 0.90},
            raw_data={"predictions": [1, 0, 1]},
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            r.save_json(path)
            loaded = ExperimentResult.load_json(path)

            assert loaded.experiment_id == r.experiment_id
            assert loaded.parameters == r.parameters
            assert loaded.metrics == r.metrics
            assert loaded.raw_data == r.raw_data
            assert loaded.timestamp == r.timestamp
        finally:
            os.unlink(path)

    def test_save_json_creates_file(self) -> None:
        """save_json() creates a readable JSON file."""
        from experiments.infra.results import ExperimentResult

        r = ExperimentResult(
            experiment_id="test",
            parameters={},
            metrics={"acc": 0.9},
        )

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name

        try:
            r.save_json(path)
            with open(path, "r") as f:
                data = json.load(f)
            assert data["experiment_id"] == "test"
            assert data["metrics"]["acc"] == 0.9
        finally:
            os.unlink(path)

    def test_result_includes_environment(self) -> None:
        """ExperimentResult can optionally include environment info."""
        from experiments.infra.results import ExperimentResult

        r = ExperimentResult(
            experiment_id="test",
            parameters={},
            metrics={},
            environment={"python_version": "3.11.5"},
        )
        assert r.environment["python_version"] == "3.11.5"


# ═══════════════════════════════════════════════════════════════════
# stats.py — Statistical utilities
# ═══════════════════════════════════════════════════════════════════


class TestBootstrapCI:
    """Bootstrap confidence interval computation."""

    def test_bootstrap_ci_returns_three_values(self) -> None:
        """bootstrap_ci() returns (lower, mean, upper)."""
        from experiments.infra.stats import bootstrap_ci

        data = [0.8, 0.85, 0.82, 0.88, 0.79, 0.83, 0.87, 0.81]
        lower, mean, upper = bootstrap_ci(data)

        assert isinstance(lower, float)
        assert isinstance(mean, float)
        assert isinstance(upper, float)

    def test_bootstrap_ci_ordering(self) -> None:
        """lower <= mean <= upper."""
        from experiments.infra.stats import bootstrap_ci

        data = [0.8, 0.85, 0.82, 0.88, 0.79, 0.83, 0.87, 0.81]
        lower, mean, upper = bootstrap_ci(data, n_bootstrap=1000, seed=42)

        assert lower <= mean <= upper

    def test_bootstrap_ci_deterministic_with_seed(self) -> None:
        """Same seed produces same CI."""
        from experiments.infra.stats import bootstrap_ci

        data = [0.8, 0.85, 0.82, 0.88, 0.79, 0.83]
        a = bootstrap_ci(data, n_bootstrap=500, seed=42)
        b = bootstrap_ci(data, n_bootstrap=500, seed=42)

        assert a == b

    def test_bootstrap_ci_constant_data(self) -> None:
        """Constant data produces zero-width CI."""
        from experiments.infra.stats import bootstrap_ci

        data = [0.5] * 20
        lower, mean, upper = bootstrap_ci(data, n_bootstrap=500, seed=42)

        assert abs(lower - 0.5) < 1e-9
        assert abs(mean - 0.5) < 1e-9
        assert abs(upper - 0.5) < 1e-9

    def test_bootstrap_ci_custom_alpha(self) -> None:
        """alpha=0.01 produces wider CI than alpha=0.05."""
        from experiments.infra.stats import bootstrap_ci
        import numpy as np

        np.random.seed(42)
        data = np.random.normal(0.8, 0.05, 100).tolist()

        l_95, _, u_95 = bootstrap_ci(data, alpha=0.05, seed=42)
        l_99, _, u_99 = bootstrap_ci(data, alpha=0.01, seed=42)

        width_95 = u_95 - l_95
        width_99 = u_99 - l_99
        assert width_99 >= width_95

    def test_bootstrap_ci_single_element(self) -> None:
        """Single element: CI collapses to that value."""
        from experiments.infra.stats import bootstrap_ci

        lower, mean, upper = bootstrap_ci([0.7], n_bootstrap=100, seed=42)

        assert abs(lower - 0.7) < 1e-9
        assert abs(mean - 0.7) < 1e-9
        assert abs(upper - 0.7) < 1e-9
