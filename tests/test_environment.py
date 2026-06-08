"""Unit tests for TradingEnv.

Verified properties
-------------------
1. buy_and_hold_matches_asset_return  — buy at step 0, sell at last step,
   zero fees → final equity ≈ initial_cash * (p_last / p_first)
2. hold_keeps_equity_flat             — always HOLD → equity stays at initial_cash
3. fees_reduce_returns                — identical B&H with fees < without fees
4. gymnasium_api_5tuple               — step() returns 5-tuple, reset() returns 2-tuple
5. obs_shape                          — observation has n_features + 2 elements
6. no_crash_full_episode              — random policy completes without IndexError
7. reward_is_not_random               — same seed → identical reward sequence
8. risk_adjusted_reward_differs       — risk_adjusted reward ≠ log_return reward
"""

from __future__ import annotations

import math

import numpy as np

from neuraltrade.config import EnvConfig
from neuraltrade.environment import BUY, HOLD, SELL, TradingEnv
from neuraltrade.feature_engineering import FEATURE_COLUMNS
from tests.conftest import make_feature_df

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _run_policy(env: TradingEnv, actions: list[int]) -> list[float]:
    """Drive env with a fixed action sequence; return equity curve."""
    env.reset(seed=0)
    for a in actions:
        _, _, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            break
    return env.equity_curve


def _make_flat_env(n: int = 50, price: float = 100.0, cfg: EnvConfig | None = None) -> TradingEnv:
    """Env where the asset price never moves."""
    df = make_feature_df(n)
    env = TradingEnv(df=df, price_col="close", env_cfg=cfg)
    env._prices = np.full(n, price, dtype=np.float64)
    return env


def _make_linear_env(
    n: int = 100,
    start: float = 100.0,
    end: float = 200.0,
    cfg: EnvConfig | None = None,
) -> TradingEnv:
    """Env where price rises linearly from start to end."""
    df = make_feature_df(n)
    env = TradingEnv(df=df, price_col="close", env_cfg=cfg)
    env._prices = np.linspace(start, end, n, dtype=np.float64)
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestGymnasiumAPI:
    def test_reset_returns_two_tuple(self):
        df = make_feature_df(50)
        env = TradingEnv(df=df)
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2, (
            "reset() must return (obs, info)"
        )
        obs, info = result
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        df = make_feature_df(50)
        env = TradingEnv(df=df)
        env.reset(seed=0)
        result = env.step(HOLD)
        assert len(result) == 5, "step() must return (obs, reward, terminated, truncated, info)"
        _, _, terminated, truncated, info = result
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_obs_shape(self):
        df = make_feature_df(50)
        env = TradingEnv(df=df)
        obs, _ = env.reset(seed=0)
        expected = len(FEATURE_COLUMNS) + 2  # features + position_flag + cash_norm
        assert obs.shape == (expected,), f"Expected obs shape ({expected},), got {obs.shape}"

    def test_action_space_contains_all_actions(self):
        df = make_feature_df(50)
        env = TradingEnv(df=df)
        for a in (BUY, HOLD, SELL):
            assert env.action_space.contains(a)


class TestHoldKeepsEquityFlat:
    """Doing nothing (all HOLDs) with a flat price → equity never changes."""

    def test_hold_equity_flat(self):
        n = 50
        initial_cash = 10_000.0
        no_fee_cfg = EnvConfig(
            initial_cash=initial_cash,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        env = _make_flat_env(n=n, price=100.0, cfg=no_fee_cfg)
        env.reset(seed=0)
        for _ in range(n - 2):
            _, _, terminated, truncated, _ = env.step(HOLD)
            if terminated or truncated:
                break

        equity = np.array(env.equity_curve)
        np.testing.assert_allclose(
            equity, initial_cash, rtol=1e-6,
            err_msg="HOLDing with flat price must keep equity exactly at initial_cash",
        )

    def test_hold_reward_zero(self):
        """log(V_t / V_{t-1}) = log(1) = 0 when nothing changes."""
        n = 30
        no_fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        env = _make_flat_env(n=n, cfg=no_fee_cfg)
        env.reset(seed=0)
        rewards = []
        for _ in range(n - 2):
            _, r, terminated, truncated, _ = env.step(HOLD)
            rewards.append(r)
            if terminated or truncated:
                break
        np.testing.assert_allclose(
            rewards, 0.0, atol=1e-8,
            err_msg="All rewards must be 0 when HOLDing with flat price and zero fees",
        )


class TestBuyAndHoldMatchesAssetReturn:
    """Buy at step 0, hold, sell at last step (no fees) → equity == BnH equity."""

    def test_bnh_return_no_fees(self):
        n = 100
        start_price = 100.0
        end_price = 150.0
        no_fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        env = _make_linear_env(n=n, start=start_price, end=end_price, cfg=no_fee_cfg)
        env.reset(seed=0)

        # Buy on step 0, hold until terminal
        _, _, terminated, truncated, _ = env.step(BUY)
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(HOLD)

        final_equity = env.equity_curve[-1]
        expected_equity = no_fee_cfg.initial_cash * (end_price / start_price)

        assert math.isclose(final_equity, expected_equity, rel_tol=1e-4), (
            f"Expected B&H equity ≈ {expected_equity:.2f}, got {final_equity:.2f}"
        )

    def test_bnh_total_return(self):
        from neuraltrade.metrics import buy_and_hold_return, total_return

        n = 100
        no_fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        env = _make_linear_env(n=n, start=100.0, end=200.0, cfg=no_fee_cfg)
        prices = env._prices.copy()
        env.reset(seed=0)

        _, _, terminated, truncated, _ = env.step(BUY)
        while not (terminated or truncated):
            _, _, terminated, truncated, _ = env.step(HOLD)

        rl_return = total_return(env.equity_curve)
        bh_return = buy_and_hold_return(prices)
        # Allow 0.1% tolerance for integer-share rounding
        assert math.isclose(rl_return, bh_return, rel_tol=1e-3), (
            f"RL B&H return {rl_return:.4f} should ≈ asset B&H return {bh_return:.4f}"
        )


class TestFeesReduceReturns:
    """Fees must make the final equity strictly lower than the zero-fee case."""

    def test_fees_reduce_equity(self):
        n = 100
        no_fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=20,
            slippage_bps=10,
            sizing="all_in",
            reward_type="log_return",
        )

        def run_bnh(cfg):
            env = _make_linear_env(n=n, start=100.0, end=150.0, cfg=cfg)
            env.reset(seed=0)
            env.step(BUY)
            for _ in range(n - 3):
                _, _, terminated, truncated, _ = env.step(HOLD)
                if terminated or truncated:
                    break
            return env.equity_curve[-1]

        equity_no_fee = run_bnh(no_fee_cfg)
        equity_with_fee = run_bnh(fee_cfg)

        assert equity_with_fee < equity_no_fee, (
            f"Fees must reduce equity: {equity_with_fee:.2f} should < {equity_no_fee:.2f}"
        )

    def test_fees_are_nonzero_on_trade(self):
        """Single buy: slippage + tc must consume cash vs no-fee case."""
        fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=10,
            slippage_bps=5,
            sizing="all_in",
            reward_type="log_return",
        )
        no_fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        env_fee = _make_flat_env(n=20, price=100.0, cfg=fee_cfg)
        env_nofee = _make_flat_env(n=20, price=100.0, cfg=no_fee_cfg)
        env_fee.reset(seed=0)
        env_nofee.reset(seed=0)
        env_fee.step(BUY)
        env_nofee.step(BUY)

        # all_in exhausts cash in both cases; fees mean fewer shares were affordable
        assert env_fee._shares < env_nofee._shares, (
            f"Fees must result in fewer shares: {env_fee._shares:.4f} should < {env_nofee._shares:.4f}"
        )


class TestRewardDeterminism:
    """Same seed → identical reward sequence (no random rewards)."""

    def test_same_seed_same_rewards(self):
        df = make_feature_df(80)

        def collect_rewards(seed):
            env = TradingEnv(df=df, seed=seed)
            env.reset(seed=seed)
            rewards = []
            for a in [BUY] + [HOLD] * 30 + [SELL] + [HOLD] * 20:
                _, r, terminated, truncated, _ = env.step(a)
                rewards.append(r)
                if terminated or truncated:
                    break
            return rewards

        r1 = collect_rewards(42)
        r2 = collect_rewards(42)
        assert r1 == r2, "Rewards must be identical for same seed"

    def test_rewards_not_random_noise(self):
        """Rewards must not look like N(0,1) noise — check std is small for HOLD."""
        n = 50
        no_fee_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        env = _make_flat_env(n=n, price=100.0, cfg=no_fee_cfg)
        env.reset(seed=0)
        rewards = []
        for _ in range(n - 2):
            _, r, terminated, truncated, _ = env.step(HOLD)
            rewards.append(r)
            if terminated or truncated:
                break
        std = float(np.std(rewards))
        assert std < 1e-6, (
            f"HOLD rewards on flat price must be ≈ 0.0, got std={std:.6f}. "
            "Likely residual random reward."
        )


class TestRiskAdjustedReward:
    def test_risk_adjusted_differs_from_log_return(self):
        n = 80
        log_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="log_return",
        )
        ra_cfg = EnvConfig(
            initial_cash=10_000.0,
            transaction_cost_bps=0,
            slippage_bps=0,
            sizing="all_in",
            reward_type="risk_adjusted",
            risk_adjusted_drawdown_penalty=0.5,
        )
        actions = [BUY] + [HOLD] * (n - 4) + [SELL]

        def rewards_for(cfg):
            env = _make_linear_env(n=n, start=100.0, end=80.0, cfg=cfg)  # falling market
            env.reset(seed=0)
            rs = []
            for a in actions:
                _, r, terminated, truncated, _ = env.step(a)
                rs.append(r)
                if terminated or truncated:
                    break
            return rs

        r_log = rewards_for(log_cfg)
        r_ra = rewards_for(ra_cfg)
        # During a drawdown, risk-adjusted reward should be lower
        assert r_ra != r_log, "risk_adjusted rewards must differ from log_return rewards"


class TestFullEpisode:
    def test_random_policy_no_crash(self):
        rng = np.random.default_rng(99)
        df = make_feature_df(150)
        env = TradingEnv(df=df, seed=42)
        obs, _ = env.reset(seed=42)
        steps = 0
        while True:
            action = int(rng.integers(0, 3))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            assert not np.isnan(reward), f"NaN reward at step {steps}"
            assert obs.shape == env.observation_space.shape
            if terminated or truncated:
                break
        assert steps > 0
        assert len(env.equity_curve) == steps + 1  # +1 for initial value
