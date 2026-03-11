"""Global seed configuration for experiment reproducibility.

Sets seeds for Python's ``random``, ``numpy``, and (optionally)
``torch`` to guarantee deterministic results across runs.

Usage::

    from experiments.infra.config import set_global_seed
    set_global_seed(42)
"""

from __future__ import annotations

import random
from typing import Optional

_global_seed: Optional[int] = None


def set_global_seed(seed: int) -> None:
    """Set the global random seed for all supported RNGs.

    Seeds: ``random``, ``numpy.random``.  If ``torch`` is installed,
    also seeds ``torch.manual_seed`` and ``torch.cuda.manual_seed_all``.

    Args:
        seed: Non-negative integer seed value.
    """
    global _global_seed
    _global_seed = seed

    # Python stdlib
    random.seed(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch (optional)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_global_seed() -> Optional[int]:
    """Return the most recently set global seed, or None if unset."""
    return _global_seed
