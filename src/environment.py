import gym
import numpy as np
import pandas as pd
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()
        
        # Debug: Print the initial columns
        print("Columns in df passed to TradingEnv:", df.columns, flush=True)

        self.df = df.copy()  # Avoid modifying the original dataframe
        self.current_step = 0

        # Ensure 'Date' is not present
        if 'Date' in self.df.columns:
            self.df = self.df.drop(columns=['Date'])

        # Assign num_features dynamically
        self.num_features = len(self.df.columns)

        # Debugging: Check if we have the expected number of features
        print(f"Final columns after processing: {self.df.columns}", flush=True)
        print(f"Final shape after processing: {self.df.shape}", flush=True)

        assert self.num_features == 17, f"Expected 17 features, but got {self.num_features}"

        # Define action & observation spaces
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        print(f"Resetting environment... Step set to {self.current_step}", flush=True)
        return self._next_observation()

    def _next_observation(self):
        print(f"Fetching next observation at step {self.current_step}...", flush=True)
        
        # Extract observation without assuming "Date" column
        obs = self.df.iloc[self.current_step].values
        obs = np.array(obs, dtype=np.float32)

        # Debug: Ensure shape is correct
        print(f"Observation Shape: {obs.shape}\nObservation: {obs}", flush=True)

        if obs.shape[0] != 17:
            raise ValueError(f"Unexpected observation shape: {obs.shape}, expected (17,)")

        return obs

    def step(self, action):
        print(f"Performing action {action} at step {self.current_step}...", flush=True)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = np.random.randn()  # Placeholder reward calculation
        obs = self._next_observation()

        return obs, reward, done, {}

    def render(self, mode="human"):
        pass
