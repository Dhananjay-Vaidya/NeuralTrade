import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import TradingEnv

# Load dataset
df = pd.read_csv("engineered_stock_data.csv")  # Ensure this file exists
print("First few rows of DataFrame:", df.head(), flush=True)
print("Columns in DataFrame:", df.columns, flush=True)

# Initialize environment
env = TradingEnv(df)

# Wrap in DummyVecEnv (Stable-Baselines3 requirement)
env = DummyVecEnv([lambda: env])

# Define PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
print("Starting training...", flush=True)
model.learn(total_timesteps=50000)
print("Training complete.", flush=True)

# Save trained model
model.save("ppo_trading_model")
print("Model saved successfully!", flush=True)
