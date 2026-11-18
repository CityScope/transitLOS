"""Where the heavy array work runs: on a CUDA GPU when one is usable, otherwise on the CPU.

    device()   'cuda' or 'cpu'      (environment variable TRANSITLOS_DEVICE = cpu | cuda | auto, default auto)
    xp()       the array module: CuPy on 'cuda', NumPy on 'cpu'

CuPy is imported lazily and only when needed; a missing package, a missing driver or a failing device all fall back to the CPU, never to an error.
"""
from __future__ import annotations

import os
from geohierarchy._backend import to_numpy  # noqa: F401  (one implementation, in geohierarchy)

_cache: dict = {}


def device() -> str:
    """``"cuda"`` when CuPy and a usable GPU are found (and ``TRANSITLOS_DEVICE`` / ``FLEX_BACKEND`` allow it), else
    ``"cpu"``; decided once per process.

    Returns:
        The device name.
    """
    if "device" in _cache:
        return _cache["device"]
    want = os.environ.get("TRANSITLOS_DEVICE", os.environ.get("FLEX_BACKEND", "auto")).lower()
    dev = "cpu"
    if want in ("auto", "cuda", "gpu", "cupy"):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                cp.zeros(1)   # fails here, not later, if the driver cannot be used
                dev = "cuda"
        except Exception:
            dev = "cpu"
    _cache["device"] = dev
    return dev


def xp():
    """The array library for the current device: CuPy on ``"cuda"``, NumPy otherwise.

    Returns:
        The ``cupy`` or ``numpy`` module.
    """
    if device() == "cuda":
        import cupy as cp
        return cp
    import numpy as np
    return np


