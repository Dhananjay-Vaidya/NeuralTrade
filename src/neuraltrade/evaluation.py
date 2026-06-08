"""Evaluate a trained model on the held-out test set."""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from neuraltrade.config import Settings, cfg
from neuraltrade.data_splits import DataSplit, load_and_split
from neuraltrade.environment import TradingEnv
from neuraltrade.metrics import compute_all
from neuraltrade.seeding import seed_everything


def _load_model(settings: Settings):
    model_path = settings.model.abs(settings.model.save_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    custom_objects = {
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }
    algo = settings.model.algorithm
    if algo == "PPO":
        return PPO.load(str(model_path), custom_objects=custom_objects)
    if algo == "RecurrentPPO":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(str(model_path), custom_objects=custom_objects)
    raise ValueError(f"Unknown algorithm {algo!r}")


def run_evaluation(settings: Settings | None = None) -> tuple[dict, TradingEnv, DataSplit]:
    """Evaluate on test split, return (metrics, env, split)."""
    s = settings or cfg
    seed_everything(s.model.seed)

    split = load_and_split(s.data)
    model = _load_model(s)

    env = TradingEnv(
        df=split.test,
        price_col="close",
        env_cfg=s.env,
        seed=s.model.seed,
    )
    env._prices = split.test_prices.to_numpy(dtype=np.float64)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    metrics = compute_all(env.equity_curve, env.trade_log, split.test_prices.to_numpy())
    return metrics, env, split


if __name__ == "__main__":
    metrics, env, split = run_evaluation()
    print("\n=== Test-set Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:<28} {v:+.4f}")
