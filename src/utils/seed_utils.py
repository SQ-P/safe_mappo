"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


_THREADS_CONFIGURED = False


def configure_torch_threads(num_threads: int | None = None) -> None:
    """Use a conservative thread count for small MAPF MLP experiments.

    On some Windows/CPU setups, PyTorch's default high thread count makes tiny
    categorical-sampling workloads extremely slow. A single or small number of
    threads is faster and more predictable for this project.
    """
    global _THREADS_CONFIGURED
    if _THREADS_CONFIGURED:
        return
    value = int(num_threads or os.environ.get("FCRM_TORCH_THREADS", "1"))
    value = max(1, value)
    torch.set_num_threads(value)
    try:
        torch.set_num_interop_threads(value)
    except RuntimeError:
        pass
    _THREADS_CONFIGURED = True


def set_global_seeds(seed: int) -> None:
    configure_torch_threads()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
