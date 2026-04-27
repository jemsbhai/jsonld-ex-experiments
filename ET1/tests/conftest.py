"""Shared fixtures for ET1 tests."""

import pytest
import yaml
from pathlib import Path


@pytest.fixture
def config():
    """Load the training config."""
    config_path = Path(__file__).parent.parent / "configs" / "training_config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def kb_config(config):
    """Just the knowledge_base section of config."""
    return config["knowledge_base"]


@pytest.fixture
def small_kb_config():
    """A minimal KB config for fast unit tests."""
    return {
        "seed": 42,
        "world_name": "Meridian",
        "total_facts": 100,
        "splits": {
            "train": 50,
            "val": 15,
            "test_id": 20,
            "test_ood": 15,
        },
        "confidence_tiers": {
            "T1_established": {"belief_range": [0.90, 0.99], "fraction": 0.30},
            "T2_probable": {"belief_range": [0.70, 0.89], "fraction": 0.25},
            "T3_uncertain": {"belief_range": [0.40, 0.69], "fraction": 0.20},
            "T4_speculative": {"belief_range": [0.15, 0.39], "fraction": 0.15},
            "T5_contested": {
                "belief_range": [0.30, 0.60],
                "disbelief_range": [0.20, 0.50],
                "fraction": 0.10,
            },
        },
    }
