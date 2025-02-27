import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from stable_baselines3 import PPO
from src.environment import TradingEnv

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📈 AI-Powered Stock Trading Simulator")
st.sidebar.header("Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    if 'Date' in df_test.columns:
        df_test.drop(columns=['Date'], inplace=True)
else:
    st.warning("Please upload a test dataset to proceed.")
    st.stop()

# Load trained model
model = PPO.load("model/ppo_trading_model.zip")
env = TradingEnv(df_test)

# Run evaluation
obs = env.reset()
done = False
total_reward = 0
trade_data = []

for step in range(len(df_test)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    trade_data.append((step, action, reward))
    if done:
        break

trade_df = pd.DataFrame(trade_data, columns=["Step", "Action", "Reward"])

# Stock Prices vs. Trade Actions (Advanced Candlestick Chart)
st.subheader("📊 Stock Prices & Trade Actions")
fig = go.Figure()
fig.add_trace(go.Candlestick(x=df_test.index,
                open=df_test['AAPL'],
                high=df_test['High'],
                low=df_test['Low'],
                close=df_test['AAPL'],
                name='AAPL'))
buy_steps = trade_df[trade_df['Action'] == 0]['Step']
sell_steps = trade_df[trade_df['Action'] == 2]['Step']
fig.add_trace(go.Scatter(x=buy_steps, y=df_test['AAPL'][buy_steps],
                         mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                         name='Buy'))
fig.add_trace(go.Scatter(x=sell_steps, y=df_test['AAPL'][sell_steps],
                         mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
                         name='Sell'))
st.plotly_chart(fig, use_container_width=True)

# Portfolio Value Over Time
st.subheader("💰 Portfolio Value Over Time")
portfolio_values = np.cumsum(trade_df["Reward"].values)
fig = px.area(x=trade_df["Step"], y=portfolio_values, labels={'x': 'Step', 'y': 'Portfolio Value'},
              title='Portfolio Growth', line_shape='spline')
st.plotly_chart(fig, use_container_width=True)

# Reward Trends
st.subheader("📈 Reward Trends")
fig = px.line(trade_df, x="Step", y="Reward", title="Reward Per Trade", markers=True)
st.plotly_chart(fig, use_container_width=True)

# Action Distribution
st.subheader("📊 Action Distribution")
action_counts = trade_df['Action'].value_counts().reset_index()
action_counts.columns = ['Action', 'Count']
fig = px.pie(action_counts, names='Action', values='Count', title='Action Distribution')
st.plotly_chart(fig, use_container_width=True)

st.success(f"Evaluation completed! Total Reward: {total_reward:.2f}")
