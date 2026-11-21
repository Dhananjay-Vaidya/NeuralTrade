"""Train a PPO agent on the TRAIN period (2019-2022).

Usage: python src/train_rl_model.py [--timesteps 200000]
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from common import (
    SEED,
    TICKER,
    compute_norm_stats,
    load_features,
    model_path,
    norm_path,
    save_norm_stats,
    split_train_test,
)
from environment import TradingEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    args = parser.parse_args()

    train_df, _ = split_train_test(load_features())
    stats = compute_norm_stats(train_df)  # train-set statistics only
    save_norm_stats(stats)
    print(f"Training {TICKER} on {len(train_df)} days "
          f"({train_df.index[0].date()} to {train_df.index[-1].date()}), "
          f"{len(stats['feature_columns'])} features")

    env = Monitor(TradingEnv(train_df, stats))
    model = PPO("MlpPolicy", env, seed=SEED, device="cpu", verbose=1)
    model.learn(total_timesteps=args.timesteps)
    model.save(model_path())
    print(f"Saved model to {model_path()} and normalisation stats to {norm_path()}")


if __name__ == "__main__":
    main()
