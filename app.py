"""NeuralTrade - Streamlit backtest dashboard.

Loads the trained PPO model and the TEST period (2023-2024) automatically.
Optionally upload a raw OHLCV CSV (Date,Open,High,Low,Close,Volume) to backtest
on other data; it is normalised with the saved TRAIN-set statistics.

Run: streamlit run app.py
Research tool only - not financial advice.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from common import TICKER, load_norm_stats, model_path  # noqa: E402
from environment import BUY, SELL, TradingEnv  # noqa: E402
from evaluation import compute_metrics, evaluate, run_policy  # noqa: E402
from feature_engineering import add_indicators  # noqa: E402

st.set_page_config(layout="wide", page_title="NeuralTrade Backtester")
st.title(f"NeuralTrade - PPO agent vs buy-and-hold ({TICKER})")
st.caption("Backtested research output, not financial advice.")

if not model_path().exists():
    st.error(f"No trained model at `{model_path()}`. Run the pipeline in the README first.")
    st.stop()

uploaded = st.sidebar.file_uploader("Optional: upload raw OHLCV CSV", type=["csv"])


@st.cache_data
def default_results():
    test_df, agent, bh = evaluate()
    return test_df, agent, bh


@st.cache_data
def uploaded_results(csv_bytes):
    from io import BytesIO

    from stable_baselines3 import PPO

    df = pd.read_csv(BytesIO(csv_bytes), index_col="Date", parse_dates=True)
    df = add_indicators(df)
    stats = load_norm_stats()
    model = PPO.load(model_path(), device="cpu")
    agent = run_policy(TradingEnv(df, stats), lambda o: model.predict(o, deterministic=True)[0])
    bh = run_policy(TradingEnv(df, stats), lambda o: BUY)
    return df, agent, bh


try:
    if uploaded is not None:
        data, agent, bh = uploaded_results(uploaded.getvalue())
        st.info("Showing results for the uploaded file.")
    else:
        data, agent, bh = default_results()
except Exception as e:
    st.error(f"Could not run backtest: {e}")
    st.stop()

st.write(f"Period: **{data.index[0].date()}** to **{data.index[-1].date()}** ({len(data)} trading days)")

# Metrics table
m_agent, m_bh = compute_metrics(agent), compute_metrics(bh)
table = pd.DataFrame(
    {"PPO agent": m_agent, "Buy & hold": m_bh},
).rename(index={
    "total_return_pct": "Total return (%)",
    "sharpe": "Sharpe (annualised)",
    "max_drawdown_pct": "Max drawdown (%)",
    "num_trades": "Trades",
    "win_rate_pct": "Win rate (%)",
    "final_value": "Final value ($)",
})
st.subheader("Metrics")
st.dataframe(table.style.format("{:,.2f}"), width="stretch")

# Equity curve
st.subheader("Equity curve")
fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=agent.index, y=agent["value"], name="PPO agent"))
fig_eq.add_trace(go.Scatter(x=bh.index, y=bh["value"], name="Buy & hold", line={"dash": "dash"}))
fig_eq.update_layout(yaxis_title="Portfolio value ($)", height=380)
st.plotly_chart(fig_eq, width="stretch")

# Price chart with executed trades
st.subheader("Price with agent trades")
buys = agent[agent["trade"] & (agent["action"] == BUY)]
sells = agent[agent["trade"] & (agent["action"] == SELL)]
fig_px = go.Figure()
fig_px.add_trace(go.Candlestick(
    x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
    name="Price",
))
fig_px.add_trace(go.Scatter(
    x=buys.index, y=buys["price"], mode="markers", name="Buy",
    marker={"symbol": "triangle-up", "size": 11, "color": "green"},
))
fig_px.add_trace(go.Scatter(
    x=sells.index, y=sells["price"], mode="markers", name="Sell",
    marker={"symbol": "triangle-down", "size": 11, "color": "red"},
))
fig_px.update_layout(xaxis_rangeslider_visible=False, height=450)
st.plotly_chart(fig_px, width="stretch")
