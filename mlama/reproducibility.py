"""Runtime helpers for reproducible M-LAMA experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    seed: int
    device: str
    cuda_available: bool


def set_seed(seed: int = 42, deterministic: bool = True) -> RuntimeInfo:
    """Seed Python, NumPy, and PyTorch, returning the selected runtime."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return RuntimeInfo(seed=seed, device=device, cuda_available=torch.cuda.is_available())


def configure_tokenizers(parallelism: bool = False) -> None:
    """Set a stable default for Hugging Face tokenizer worker behavior."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true" if parallelism else "false")
