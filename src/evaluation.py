"""Evaluate the trained agent on the TEST period (2023-2024) against buy-and-hold.

Usage: python src/evaluation.py
"""

import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from common import (
    RESULTS_DIR,
    TICKER,
    load_features,
    load_norm_stats,
    model_path,
    split_train_test,
)
from environment import BUY, TradingEnv


def run_policy(env, policy):
    """Run one episode. Returns a DataFrame indexed by date with value, action, trade."""
    obs, info = env.reset()
    values, actions, trades = [info["portfolio_value"]], [], []
    done = False
    while not done:
        action = policy(obs)
        obs, _, terminated, truncated, info = env.step(action)
        values.append(info["portfolio_value"])
        actions.append(int(action))
        trades.append(info["trade"])
        done = terminated or truncated
    # action/trade at row t is the decision taken at close t; the last day has none
    actions.append(-1)
    trades.append(False)
    return pd.DataFrame(
        {"price": env.prices, "value": values, "action": actions, "trade": trades},
        index=env.dates,
    )


def compute_metrics(result):
    value = result["value"].to_numpy()
    daily = np.diff(value) / value[:-1]
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    drawdown = value / np.maximum.accumulate(value) - 1

    # Round trips: buy -> sell; an open position is marked to market on the last day.
    wins, round_trips, entry_value = 0, 0, None
    for i, (traded, act) in enumerate(zip(result["trade"], result["action"])):
        if not traded:
            continue
        if act == BUY:
            entry_value = value[i]
        else:
            round_trips += 1
            wins += value[i + 1] > entry_value
            entry_value = None
    if entry_value is not None:
        round_trips += 1
        wins += value[-1] > entry_value

    return {
        "total_return_pct": (value[-1] / value[0] - 1) * 100,
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "num_trades": int(result["trade"].sum()),
        "win_rate_pct": (wins / round_trips * 100) if round_trips else 0.0,
        "final_value": float(value[-1]),
    }


def evaluate(ticker=TICKER):
    _, test_df = split_train_test(load_features(ticker))
    stats = load_norm_stats(ticker)
    model = PPO.load(model_path(ticker), device="cpu")

    agent = run_policy(TradingEnv(test_df, stats), lambda o: model.predict(o, deterministic=True)[0])
    bh = run_policy(TradingEnv(test_df, stats), lambda o: BUY)
    return test_df, agent, bh


def format_table(metrics):
    rows = [
        ("Total return (%)", "total_return_pct", "{:.2f}"),
        ("Sharpe (ann.)", "sharpe", "{:.2f}"),
        ("Max drawdown (%)", "max_drawdown_pct", "{:.2f}"),
        ("Trades", "num_trades", "{:d}"),
        ("Win rate (%)", "win_rate_pct", "{:.1f}"),
        ("Final value ($)", "final_value", "{:,.0f}"),
    ]
    lines = [f"{'Metric':<20}{'PPO agent':>14}{'Buy & hold':>14}", "-" * 48]
    for label, key, fmt in rows:
        a = fmt.format(metrics["agent"][key])
        b = fmt.format(metrics["buy_and_hold"][key])
        lines.append(f"{label:<20}{a:>14}{b:>14}")
    return "\n".join(lines)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test_df, agent, bh = evaluate()
    metrics = {
        "ticker": TICKER,
        "test_period": [str(test_df.index[0].date()), str(test_df.index[-1].date())],
        "agent": compute_metrics(agent),
        "buy_and_hold": compute_metrics(bh),
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(agent.index, agent["value"], label="PPO agent")
    ax.plot(bh.index, bh["value"], label="Buy & hold", linestyle="--")
    ax.set_title(f"{TICKER} test period {metrics['test_period'][0]} to {metrics['test_period'][1]}")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "equity_curve.png", dpi=120)

    print(f"\n{TICKER} test period: {metrics['test_period'][0]} to {metrics['test_period'][1]}\n")
    print(format_table(metrics))
    print(f"\nSaved {RESULTS_DIR / 'metrics.json'} and {RESULTS_DIR / 'equity_curve.png'}")


if __name__ == "__main__":
    main()
