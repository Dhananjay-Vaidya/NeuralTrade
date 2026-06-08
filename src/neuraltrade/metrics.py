"""Portfolio performance metrics used for evaluation and MLflow logging."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def total_return(equity_curve: list[float] | np.ndarray) -> float:
    """(final - initial) / initial."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2 or eq[0] == 0:
        return 0.0
    return (eq[-1] - eq[0]) / eq[0]


def sharpe_ratio(
    equity_curve: list[float] | np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio from an equity curve (daily steps assumed)."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return 0.0
    log_rets = np.diff(np.log(np.clip(eq, 1e-8, None)))
    excess = log_rets - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: list[float] | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a positive fraction (0–1)."""
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak == 0, 1, peak)
    return float(dd.max())


def win_rate(trade_log: list[dict]) -> float:
    """Fraction of completed round-trips (buy→sell) that were profitable."""
    buys: list[float] = []
    wins = 0
    total = 0
    for trade in trade_log:
        if trade["action"] == "BUY":
            buys.append(trade["price"])
        elif trade["action"] == "SELL" and buys:
            buy_price = buys.pop(0)
            total += 1
            if trade["price"] > buy_price:
                wins += 1
    return wins / total if total > 0 else 0.0


def buy_and_hold_return(prices: pd.Series | np.ndarray) -> float:
    """Simple buy-and-hold return on the price series (no fees)."""
    p = np.asarray(prices, dtype=float)
    if len(p) < 2 or p[0] == 0:
        return 0.0
    return (p[-1] - p[0]) / p[0]


def compute_all(
    equity_curve: list[float] | np.ndarray,
    trade_log: list[dict],
    prices: pd.Series | np.ndarray,
) -> dict[str, float]:
    return {
        "total_return": total_return(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trade_log),
        "buy_and_hold_return": buy_and_hold_return(prices),
    }
