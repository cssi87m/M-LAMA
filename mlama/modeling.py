"""Small shared helpers for model construction and checkpoint loading."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def model_kwargs_from_config(model_cls: type, model_config: Any) -> dict[str, Any]:
    """Filter a config dataclass/dict to arguments accepted by a model class."""
    values = model_config if isinstance(model_config, dict) else vars(model_config)
    valid_params = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
    return {key: value for key, value in values.items() if key in valid_params}


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    """Move loaded optimizer state tensors to the active device."""
    target = torch.device(device)
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(target)


def find_checkpoint(explicit: str | None, configured: str | None, save_dir: str) -> str:
    """Resolve a checkpoint path from CLI, config, or latest best checkpoint."""
    if explicit:
        return explicit
    if configured:
        return configured

    save_path = Path(save_dir)
    if not save_path.exists():
        raise FileNotFoundError(f"No checkpoint provided and save_dir does not exist: {save_path}")

    best = sorted(save_path.glob("model_best_mae_*.pth"))
    if not best:
        raise FileNotFoundError(f"No model_best_mae_*.pth checkpoint found in {save_path}")
    return str(best[-1])
