"""NeuralTrade — Streamlit backtesting dashboard.

⚠️  This is backtested research output, not live trading advice.
    Past performance does not guarantee future results.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import RobustScaler

from neuraltrade.config import cfg
from neuraltrade.environment import TradingEnv
from neuraltrade.feature_engineering import FEATURE_COLUMNS, add_indicators
from neuraltrade.metrics import compute_all


def _constant_schedule(_: float) -> float:
    return 0.0

st.set_page_config(layout="wide", page_title="NeuralTrade Backtester")

# --- Warning banner ---
st.warning(
    "⚠️ **Research / backtesting tool only.** "
    "This is not financial advice and does not constitute a live trading system."
)
st.title("📈 NeuralTrade — AI Trading Backtester")

# ─────────────────────────── Sidebar ────────────────────────────────────────
st.sidebar.header("Configuration")

uploaded = st.sidebar.file_uploader(
    "Upload CSV (raw OHLCV or pre-engineered)", type=["csv"]
)
use_default = st.sidebar.checkbox("Use default test split from config", value=uploaded is None)
ticker_label = st.sidebar.text_input("Price column (close-price name)", value="close")


# ─────────────────────────── Data loading ───────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str):
    custom_objects = {
        "lr_schedule": _constant_schedule,
        "clip_range": _constant_schedule,
    }
    algo = cfg.model.algorithm
    if algo == "PPO":
        from stable_baselines3 import PPO
        return PPO.load(model_path, custom_objects=custom_objects)
    from sb3_contrib import RecurrentPPO
    return RecurrentPPO.load(model_path, custom_objects=custom_objects)


@st.cache_data(show_spinner="Processing data…")
def prepare_from_upload(csv_bytes: bytes, price_col: str):
    """Accept either raw OHLCV or a pre-engineered CSV and return scaled features + raw prices."""
    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Parse date index if present
    for date_col in ("date", "unnamed: 0", "index"):
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.set_index(date_col)
            break

    # If raw OHLCV: run feature engineering in-memory
    raw_ohlcv_cols = {"open", "high", "low", "close", "volume"}
    if raw_ohlcv_cols.issubset(set(df.columns)):
        st.info("Detected raw OHLCV — running feature engineering pipeline…")
        df = add_indicators(df)

    # Keep only expected feature columns that are present
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available) < 5:
        st.error(
            f"Could not find enough feature columns. Found: {available}. "
            f"Expected: {FEATURE_COLUMNS}"
        )
        st.stop()
    df = df[available].dropna()

    # Scale using RobustScaler fitted on this data (no future data available)
    scaler = RobustScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=available,
    )

    price_col_actual = price_col if price_col in df.columns else available[0]
    raw_prices = df[price_col_actual]
    return scaled, raw_prices


@st.cache_data(show_spinner="Loading test split…")
def prepare_from_config():
    from neuraltrade.data_splits import load_and_split
    split = load_and_split(cfg.data)
    return split.test, split.test_prices


# ─────────────────────────── Run evaluation ─────────────────────────────────
model_path = str(cfg.model.abs(cfg.model.save_path))
if not Path(model_path).exists():
    st.error(f"Trained model not found at `{model_path}`. Run `python -m neuraltrade.train` first.")
    st.stop()

model = load_model(model_path)

if uploaded is not None:
    features_df, raw_prices = prepare_from_upload(uploaded.read(), ticker_label)
elif use_default:
    try:
        features_df, raw_prices = prepare_from_config()
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e))
        st.stop()
else:
    st.info("Upload a CSV or check 'Use default test split' to begin.")
    st.stop()

env = TradingEnv(
    df=features_df,
    price_col="close",
    env_cfg=cfg.env,
    seed=cfg.model.seed,
)
env._prices = raw_prices.to_numpy(dtype=np.float64)

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(int(action))
    done = terminated or truncated

metrics = compute_all(env.equity_curve, env.trade_log, raw_prices.to_numpy())

# ─────────────────────────── Build trade DataFrame ──────────────────────────
prices_arr = raw_prices.to_numpy()
dates = raw_prices.index if hasattr(raw_prices, "index") else np.arange(len(prices_arr))

# Reconstruct open-prices array for candlestick (prev close = open)
opens = np.roll(prices_arr, 1)
opens[0] = prices_arr[0]

# Need real high/low — try to pull from features
has_hl = "high" in features_df.columns and "low" in features_df.columns

# Align equity curve to dates (env starts at step 0 and goes to len-1)
equity = np.array(env.equity_curve)
bh_equity = cfg.env.initial_cash * (prices_arr / prices_arr[0])

# Trade markers
buy_steps = [t["step"] for t in env.trade_log if t["action"] == "BUY"]
sell_steps = [t["step"] for t in env.trade_log if t["action"] == "SELL"]

# ─────────────────────────── Metrics table ──────────────────────────────────
st.subheader("📊 Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Return", f"{metrics['total_return']:+.2%}")
col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
col4.metric("Win Rate", f"{metrics['win_rate']:.2%}")
col5.metric("Buy & Hold", f"{metrics['buy_and_hold_return']:+.2%}")

# ─────────────────────────── Candlestick + trades ───────────────────────────
st.subheader("📉 Price Chart with Trade Signals")
fig_price = go.Figure()

if has_hl:
    # Reconstruct unscaled high/low via the scaler — not available here; use price ±0
    highs = prices_arr  # fallback
    lows = prices_arr
else:
    highs = prices_arr
    lows = prices_arr

fig_price.add_trace(
    go.Candlestick(
        x=dates,
        open=opens,
        high=highs,
        low=lows,
        close=prices_arr,
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )
)

if buy_steps:
    fig_price.add_trace(
        go.Scatter(
            x=[dates[i] for i in buy_steps if i < len(dates)],
            y=[prices_arr[i] * 0.99 for i in buy_steps if i < len(prices_arr)],
            mode="markers",
            marker={"symbol": "triangle-up", "size": 12, "color": "lime"},
            name="Buy",
        )
    )
if sell_steps:
    fig_price.add_trace(
        go.Scatter(
            x=[dates[i] for i in sell_steps if i < len(dates)],
            y=[prices_arr[i] * 1.01 for i in sell_steps if i < len(prices_arr)],
            mode="markers",
            marker={"symbol": "triangle-down", "size": 12, "color": "red"},
            name="Sell",
        )
    )

fig_price.update_layout(xaxis_rangeslider_visible=False, height=450)
st.plotly_chart(fig_price, use_container_width=True)

# ─────────────────────────── Equity curve ───────────────────────────────────
st.subheader("💰 Equity Curve vs Buy-and-Hold")
fig_equity = go.Figure()
eq_dates = dates[: len(equity)]
fig_equity.add_trace(
    go.Scatter(x=eq_dates, y=equity, name="RL Strategy", line={"color": "#1f77b4", "width": 2})
)
fig_equity.add_trace(
    go.Scatter(
        x=dates[: len(bh_equity)],
        y=bh_equity,
        name="Buy & Hold",
        line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
    )
)
fig_equity.update_layout(yaxis_title="Portfolio Value ($)", height=350)
st.plotly_chart(fig_equity, use_container_width=True)

# ─────────────────────────── Drawdown chart ─────────────────────────────────
st.subheader("📉 Drawdown")
peaks = np.maximum.accumulate(equity)
drawdown = (peaks - equity) / np.where(peaks == 0, 1, peaks)
fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(
        x=eq_dates,
        y=-drawdown,
        fill="tozeroy",
        fillcolor="rgba(239,83,80,0.3)",
        line={"color": "#ef5350"},
        name="Drawdown",
    )
)
fig_dd.update_layout(yaxis_tickformat=".1%", height=250)
st.plotly_chart(fig_dd, use_container_width=True)

# ─────────────────────────── Trade log ──────────────────────────────────────
if env.trade_log:
    st.subheader("📋 Trade Log")
    st.dataframe(pd.DataFrame(env.trade_log), use_container_width=True)
