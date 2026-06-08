"""TradingEnv — single-asset portfolio simulator.

Observation  : feature_row ++ [position_flag, cash_norm]   shape=(n_features+2,)
Action space : Discrete(3)  — 0=Buy, 1=Hold, 2=Sell
Reward       : log-return of portfolio value per step
               (optionally risk-adjusted via differential Sharpe / drawdown penalty)
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from neuraltrade.config import EnvConfig, cfg

# Canonical action codes
BUY = 0
HOLD = 1
SELL = 2

_DEFAULT_ENV_CFG = cfg.env


class TradingEnv(gym.Env):
    """Single-asset portfolio environment.

    Parameters
    ----------
    df:
        DataFrame whose columns are *feature columns only* (no Date index).
        The first column is treated as the asset close price used for P&L.
    price_col:
        Name of the close-price column inside *df*.
    env_cfg:
        EnvConfig instance; defaults to the global config.
    seed:
        RNG seed for reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        env_cfg: EnvConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self._cfg = env_cfg or _DEFAULT_ENV_CFG
        self._df = df.reset_index(drop=True).copy()

        # --- resolve price column (case-insensitive fallback) ---
        if price_col not in self._df.columns:
            lower_map = {c.lower(): c for c in self._df.columns}
            if price_col.lower() in lower_map:
                price_col = lower_map[price_col.lower()]
            else:
                # fall back to first column
                price_col = self._df.columns[0]
        self._price_col = price_col

        self._prices: np.ndarray = self._df[price_col].to_numpy(dtype=np.float64)
        self._n_steps = len(self._df)
        self._n_features = self._df.shape[1]

        # Spaces
        # obs = feature_row (n_features) + position_flag (1) + cash_norm (1)
        n_obs = self._n_features + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        # Portfolio state (initialised in reset)
        self._cash: float = 0.0
        self._shares: float = 0.0
        self._portfolio_value: float = 0.0
        self._prev_portfolio_value: float = 0.0
        self._peak_portfolio_value: float = 0.0
        self._current_step: int = 0

        # Equity curve recorded per episode
        self.equity_curve: list[float] = []
        self.trade_log: list[dict[str, Any]] = []

        # Cost params (convert bps → fraction)
        self._tc_frac = self._cfg.transaction_cost_bps / 10_000.0
        self._slip_frac = self._cfg.slippage_bps / 10_000.0

        # Seeding
        self._np_rng, _ = gym.utils.seeding.np_random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_rng, _ = gym.utils.seeding.np_random(seed)

        self._cash = float(self._cfg.initial_cash)
        self._shares = 0.0
        self._portfolio_value = self._cfg.initial_cash
        self._prev_portfolio_value = self._cfg.initial_cash
        self._peak_portfolio_value = self._cfg.initial_cash
        self._current_step = 0
        self.equity_curve = [self._portfolio_value]
        self.trade_log = []

        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        price = float(self._prices[self._current_step])
        executed_price = self._apply_slippage(price, action)

        self._execute_action(action, executed_price)

        self._current_step += 1
        terminated = self._current_step >= self._n_steps - 1
        truncated = False

        # Mark-to-market portfolio value at new price
        new_price = float(self._prices[self._current_step])
        self._portfolio_value = self._cash + self._shares * new_price
        self._peak_portfolio_value = max(self._peak_portfolio_value, self._portfolio_value)
        self.equity_curve.append(self._portfolio_value)

        reward = self._compute_reward()
        self._prev_portfolio_value = self._portfolio_value

        info: dict[str, Any] = {
            "portfolio_value": self._portfolio_value,
            "cash": self._cash,
            "shares": self._shares,
            "price": new_price,
            "step": self._current_step,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        print(
            f"Step {self._current_step:4d} | "
            f"Price {self._prices[self._current_step]:.2f} | "
            f"Cash {self._cash:.2f} | "
            f"Shares {self._shares:.4f} | "
            f"Portfolio {self._portfolio_value:.2f}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        feature_row = self._df.iloc[self._current_step].to_numpy(dtype=np.float32)
        position_flag = np.float32(1.0 if self._shares > 0 else 0.0)
        cash_norm = np.float32(self._cash / self._cfg.initial_cash)
        return np.concatenate([feature_row, [position_flag, cash_norm]])

    def _apply_slippage(self, price: float, action: int) -> float:
        """Widen price against the trader on each trade."""
        if action == BUY:
            return price * (1.0 + self._slip_frac)
        if action == SELL:
            return price * (1.0 - self._slip_frac)
        return price

    def _execute_action(self, action: int, exec_price: float) -> None:
        if action == BUY and self._shares == 0:
            affordable = self._cash / (exec_price * (1.0 + self._tc_frac))
            shares_to_buy = self._trade_size(affordable)
            cost = shares_to_buy * exec_price * (1.0 + self._tc_frac)
            self._cash -= cost
            self._shares += shares_to_buy
            self.trade_log.append(
                {
                    "step": self._current_step,
                    "action": "BUY",
                    "price": exec_price,
                    "shares": shares_to_buy,
                    "cost": cost,
                }
            )

        elif action == SELL and self._shares > 0:
            proceeds = self._shares * exec_price * (1.0 - self._tc_frac)
            self.trade_log.append(
                {
                    "step": self._current_step,
                    "action": "SELL",
                    "price": exec_price,
                    "shares": self._shares,
                    "proceeds": proceeds,
                }
            )
            self._cash += proceeds
            self._shares = 0.0

    def _trade_size(self, max_shares: float) -> float:
        """Return shares to buy based on sizing config."""
        if self._cfg.sizing == "all_in":
            return max_shares
        # fixed_fraction: use that fraction of current portfolio
        portfolio = self._cash + self._shares * self._prices[self._current_step]
        target_value = portfolio * self._cfg.fixed_fraction
        price = self._prices[self._current_step]
        return min(max_shares, target_value / price)

    def _compute_reward(self) -> float:
        prev = self._prev_portfolio_value
        curr = self._portfolio_value

        # Protect against zero / negative portfolio
        if prev <= 0:
            return 0.0

        log_ret = math.log(curr / prev) if curr > 0 else -10.0

        if self._cfg.reward_type == "log_return":
            return log_ret

        # risk_adjusted: log_return minus drawdown penalty
        drawdown = (self._peak_portfolio_value - curr) / self._peak_portfolio_value
        penalty = self._cfg.risk_adjusted_drawdown_penalty * drawdown
        return log_ret - penalty
