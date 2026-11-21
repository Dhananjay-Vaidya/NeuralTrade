"""Single-asset, long-only daily trading environment (gymnasium API).

Timing: at step t the agent sees indicators computed from data up to close t,
trades at close t, and is rewarded with the log change in portfolio value from
close t to close t+1. No look-ahead.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from common import INITIAL_CASH, TRANSACTION_COST

HOLD, BUY, SELL = 0, 1, 2


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df, norm_stats, initial_cash=INITIAL_CASH, transaction_cost=TRANSACTION_COST):
        super().__init__()
        cols = norm_stats["feature_columns"]
        mean = np.array([norm_stats["mean"][c] for c in cols])
        std = np.array([norm_stats["std"][c] for c in cols])

        self.dates = df.index
        self.prices = df["Close"].to_numpy(dtype=np.float64)
        self.features = ((df[cols].to_numpy(dtype=np.float64) - mean) / std).astype(np.float32)
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost

        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy (all-in), 2=sell (go flat)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(cols) + 2,), dtype=np.float32
        )

    def _portfolio_value(self):
        return self.cash + self.shares * self.prices[self.t]

    def _obs(self):
        pv = self._portfolio_value()
        extra = np.array([float(self.shares > 0), self.cash / pv], dtype=np.float32)
        return np.concatenate([self.features[self.t], extra])

    def _info(self, trade=False):
        return {
            "step": self.t,
            "portfolio_value": self._portfolio_value(),
            "position": int(self.shares > 0),
            "price": self.prices[self.t],
            "trade": trade,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.cash = self.initial_cash
        self.shares = 0.0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        price = self.prices[self.t]
        value_before = self._portfolio_value()

        trade = False
        if action == BUY and self.shares == 0:
            self.shares = self.cash * (1 - self.transaction_cost) / price
            self.cash = 0.0
            trade = True
        elif action == SELL and self.shares > 0:
            self.cash = self.shares * price * (1 - self.transaction_cost)
            self.shares = 0.0
            trade = True

        self.t += 1
        value_after = self._portfolio_value()
        reward = float(np.log(value_after / value_before))
        terminated = self.t >= len(self.prices) - 1
        info = self._info(trade)
        info["action"] = action
        return self._obs(), reward, terminated, False, info
