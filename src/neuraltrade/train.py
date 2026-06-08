"""Training entry point — PPO or RecurrentPPO with MLflow tracking."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from neuraltrade.config import Settings, cfg
from neuraltrade.data_splits import DataSplit, load_and_split, save_scaler
from neuraltrade.environment import TradingEnv
from neuraltrade.metrics import compute_all
from neuraltrade.seeding import seed_everything


def _make_env(df: pd.DataFrame, prices: pd.Series, settings: Settings) -> TradingEnv:
    env = TradingEnv(
        df=df,
        price_col="close",
        env_cfg=settings.env,
        seed=settings.model.seed,
    )
    # Inject real prices so the env's internal price array matches unscaled values
    env._prices = prices.to_numpy(dtype=np.float64)
    return env


def _load_algorithm(algorithm: str):
    if algorithm == "PPO":
        return PPO
    if algorithm == "RecurrentPPO":
        try:
            from sb3_contrib import RecurrentPPO

            return RecurrentPPO
        except ImportError as e:
            raise ImportError("Install sb3-contrib for RecurrentPPO support") from e
    raise ValueError(f"Unknown algorithm: {algorithm!r}. Choose PPO or RecurrentPPO.")


def train(settings: Settings | None = None) -> Path:
    """Train the RL model and return the saved model path."""
    s = settings or cfg
    seed_everything(s.model.seed)

    # --- Data ---
    split: DataSplit = load_and_split(s.data)
    print(
        f"Split sizes - train: {len(split.train)}, "
        f"val: {len(split.val)}, test: {len(split.test)}"
    )

    # Save scaler next to model
    model_path = s.model.abs(s.model.save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path = model_path.parent / "scaler.pkl"
    save_scaler(split.scaler, scaler_path)

    # --- Environment ---
    train_env = _make_env(split.train, split.train_prices, s)
    vec_env = DummyVecEnv([lambda: train_env])

    # --- Model ---
    AlgoCls = _load_algorithm(s.model.algorithm)
    hyperparams = {
        "policy": s.model.policy,
        "env": vec_env,
        "seed": s.model.seed,
        "learning_rate": s.model.learning_rate,
        "n_steps": s.model.n_steps,
        "batch_size": s.model.batch_size,
        "n_epochs": s.model.n_epochs,
        "gamma": s.model.gamma,
        "gae_lambda": s.model.gae_lambda,
        "clip_range": s.model.clip_range,
        "ent_coef": s.model.ent_coef,
        "vf_coef": s.model.vf_coef,
        "max_grad_norm": s.model.max_grad_norm,
        "verbose": 1,
    }
    # RecurrentPPO doesn't accept some PPO-only args
    if s.model.algorithm == "RecurrentPPO":
        hyperparams.pop("gae_lambda", None)

    model = AlgoCls(**hyperparams)

    # --- MLflow run ---
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(s.mlflow.tracking_uri)
    mlflow.set_experiment(s.mlflow.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "algorithm": s.model.algorithm,
                "total_timesteps": s.model.total_timesteps,
                "seed": s.model.seed,
                "learning_rate": s.model.learning_rate,
                "n_steps": s.model.n_steps,
                "batch_size": s.model.batch_size,
                "n_epochs": s.model.n_epochs,
                "gamma": s.model.gamma,
                "initial_cash": s.env.initial_cash,
                "transaction_cost_bps": s.env.transaction_cost_bps,
                "slippage_bps": s.env.slippage_bps,
                "sizing": s.env.sizing,
                "reward_type": s.env.reward_type,
                "primary_ticker": s.data.primary_ticker,
                "train_ratio": s.data.train_ratio,
                "val_ratio": s.data.val_ratio,
            }
        )

        print(f"Training {s.model.algorithm} for {s.model.total_timesteps} timesteps...")
        model.learn(total_timesteps=s.model.total_timesteps)
        print("Training complete.")

        # --- Save model ---
        model.save(str(model_path))
        print(f"Model saved -> {model_path}")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(scaler_path))

        # --- Validation evaluation ---
        val_metrics = _evaluate_split(model, split.val, split.val_prices, s, "val")
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # --- Test evaluation ---
        test_metrics = _evaluate_split(model, split.test, split.test_prices, s, "test")
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        _print_metrics("Validation", val_metrics)
        _print_metrics("Test      ", test_metrics)

    return model_path


def _evaluate_split(
    model,
    df: pd.DataFrame,
    prices: pd.Series,
    s: Settings,
    split_name: str,
) -> dict[str, float]:
    env = _make_env(df, prices, s)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    return compute_all(env.equity_curve, env.trade_log, prices.to_numpy())


def _print_metrics(label: str, m: dict[str, float]) -> None:
    print(
        f"{label} | "
        f"Return={m['total_return']:+.2%}  "
        f"Sharpe={m['sharpe_ratio']:.3f}  "
        f"MaxDD={m['max_drawdown']:.2%}  "
        f"WinRate={m['win_rate']:.2%}  "
        f"B&H={m['buy_and_hold_return']:+.2%}"
    )


if __name__ == "__main__":
    train()
