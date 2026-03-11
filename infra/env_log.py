"""Environment logging for experiment reproducibility.

Captures Python version, OS, CPU, RAM, GPU, and key package versions
into a JSON-serializable dict.  Included in every experiment result
for full reproducibility.

Usage::

    from experiments.infra.env_log import log_environment
    env = log_environment()
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def log_environment() -> Dict[str, Any]:
    """Capture reproducibility-critical environment details.

    Returns:
        A JSON-serializable dict with keys:

        - ``python_version``: e.g. ``"3.11.5"``
        - ``platform``: e.g. ``"Windows-10-10.0.19045-SP0"``
        - ``os``: e.g. ``"Windows"``
        - ``cpu``: processor string
        - ``ram_gb``: total physical RAM in GB (float)
        - ``jsonld_ex_version``: installed jsonld-ex version
        - ``numpy_version``: installed numpy version (or None)
        - ``gpu``: GPU name if torch+CUDA available (or None)
    """
    env: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": _get_ram_gb(),
        "jsonld_ex_version": _get_package_version("jsonld_ex"),
        "numpy_version": _get_package_version("numpy"),
        "gpu": _get_gpu_name(),
    }
    return env


def _get_ram_gb() -> float:
    """Return total physical RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        pass

    # Fallback: platform-specific
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass
    elif sys.platform in ("linux", "darwin"):
        try:
            import os
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round((pages * page_size) / (1024 ** 3), 2)
        except Exception:
            pass

    return 0.0


def _get_package_version(package_name: str) -> str | None:
    """Return installed version of a package, or None."""
    try:
        from importlib.metadata import version
        return version(package_name.replace("_", "-"))
    except Exception:
        try:
            import importlib
            mod = importlib.import_module(package_name)
            return getattr(mod, "__version__", None)
        except ImportError:
            return None


def _get_gpu_name() -> str | None:
    """Return GPU name if torch+CUDA is available, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None
