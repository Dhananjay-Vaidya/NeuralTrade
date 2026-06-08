"""Shared dependency — lazy-loaded model singleton.

Keeps the SB3 model in memory across requests so we don't reload it every call.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from neuraltrade.config import cfg

_lock = threading.Lock()
_model: Any | None = None


def get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        model_path = cfg.model.abs(cfg.model.save_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. "
                "Run `python -m neuraltrade.train` first."
            )
        algo = cfg.model.algorithm
        if algo == "PPO":
            from stable_baselines3 import PPO
            _model = PPO.load(
                str(model_path),
                custom_objects={"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
            )
        else:
            from sb3_contrib import RecurrentPPO
            _model = RecurrentPPO.load(
                str(model_path),
                custom_objects={"lr_schedule": lambda _: 0.0, "clip_range": lambda _: 0.0},
            )
        return _model


def model_available() -> bool:
    return cfg.model.abs(cfg.model.save_path).exists()


def data_available() -> bool:
    parquet = cfg.data.abs(cfg.data.engineered_path).with_suffix(".parquet")
    csv = cfg.data.abs(cfg.data.engineered_path)
    return parquet.exists() or csv.exists()
