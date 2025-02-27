import numpy as np
import pandas as pd
import gym
from stable_baselines3 import PPO
from environment import TradingEnv
import os

# Load the test dataset
test_data_path = "engineered_stock_data.csv"  # Update if needed
assert os.path.exists(test_data_path), f"Test data file not found: {test_data_path}"

df_test = pd.read_csv(test_data_path)

# Debugging: Check if the columns match training data
print("Columns in df_test before dropping 'Date':", df_test.columns)
print("df_test shape:", df_test.shape)

# Drop the 'Date' column if it exists
if 'Date' in df_test.columns:
    df_test = df_test.drop(columns=['Date'])

# Debugging: Confirm the shape after dropping
print("Columns in df_test after dropping 'Date':", df_test.columns)
print("df_test shape after dropping 'Date':", df_test.shape)

# Ensure df_test has the same number of features as training
num_features = df_test.shape[1]  # Dynamically determine feature count
assert num_features == 17, f"Expected 17 features, but got {num_features}"

# Initialize the environment
env = TradingEnv(df_test)

# Load the trained RL model
model_path = "model/ppo_trading_model.zip"
assert os.path.exists(model_path), f"Trained model not found: {model_path}"
model = PPO.load(model_path)

# Evaluate the model
obs = env.reset()
done = False
total_reward = 0

i = 0
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    print(f"Step {i}: Action={action}, Reward={reward}")
    i += 1

print("Evaluation completed. Total Reward:", total_reward)
