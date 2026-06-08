# NeuralTrade Codebase Guide

## 1. Overview

NeuralTrade is a small end-to-end reinforcement-learning stock trading prototype. The repository is organized around a linear pipeline:

1. Download historical stock prices from Yahoo Finance.
2. Clean the exported CSV into a tabular training dataset.
3. Add technical indicators for one target stock.
4. Train a PPO reinforcement-learning policy on a custom Gym environment.
5. Evaluate the trained policy.
6. Visualize evaluation output in a Streamlit dashboard.

The codebase is closer to a proof of concept than a production trading system. The training environment, reward function, and dashboard are functional scaffolding rather than a realistic portfolio simulator.

## 2. Repository Layout

```text
NeuralTrade/
├── app.py
├── README.md
├── requirements.txt
├── codebase.md
├── data/
│   ├── stock_data1.csv
│   ├── clean_stock_data.csv
│   └── engineered_stock_data.csv
├── model/
│   └── ppo_trading_model.zip
└── src/
    ├── datacollection.py
    ├── data_cleaning.py
    ├── feature_engineering.py
    ├── environment.py
    ├── train_rl_model.py
    └── evaluation.py
```

## 3. High-Level Architecture

### Data pipeline

- `src/datacollection.py` fetches adjusted daily close prices for `AAPL`, `MSFT`, and `GOOGL`.
- `src/data_cleaning.py` reshapes the exported CSV into a cleaner dated table.
- `src/feature_engineering.py` computes technical indicators using only `AAPL` as the target feature source.

### RL pipeline

- `src/environment.py` defines a custom `TradingEnv` Gym environment with:
  - `3` discrete actions: buy, hold, sell.
  - `17` expected features per observation.
  - random rewards instead of market-driven rewards.
- `src/train_rl_model.py` loads engineered data, wraps the environment in `DummyVecEnv`, and trains a PPO policy.
- `src/evaluation.py` reloads the model and dataset, runs inference through the same environment, and prints action/reward steps.

### UI layer

- `app.py` exposes a Streamlit dashboard that:
  - accepts a CSV upload,
  - loads the saved PPO model from `model/ppo_trading_model.zip`,
  - replays one evaluation loop,
  - charts action markers, cumulative rewards, and action distribution.

## 4. Execution Flow

### Step 1: Data collection

File: [src/datacollection.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/datacollection.py:1)

Behavior:

- Imports `yfinance`, `pandas`, and `time`.
- Defines `stock_symbols = ["AAPL", "MSFT", "GOOGL"]`.
- Defines `download_data(ticker)` with up to 3 retry attempts.
- Downloads data from `2019-01-01` through `2024-01-01` with `auto_adjust=True`.
- Extracts only the `"Close"` column from each ticker dataset.
- Concatenates close-price series into a single multi-column DataFrame.
- Saves the result to `data/stock_data1.csv`.

Important implementation detail:

- Because `pd.concat({k: v["Close"] for k, v in data_dict.items()}, axis=1)` uses dictionary keys, the resulting CSV includes an unusual two-row header structure when written to disk. That odd header shape is the reason the next script uses `skiprows=1` and parses a column named `Ticker`.

Observed output:

- `data/stock_data1.csv` has shape `(1260, 4)`.
- Columns after a naive load are `['Unnamed: 0', 'AAPL', 'MSFT', 'GOOGL']`.
- The first rows reveal two header-like lines before actual dated records.

### Step 2: Data cleaning

File: [src/data_cleaning.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/data_cleaning.py:1)

Behavior:

- Reads `data/stock_data1.csv` with:
  - `skiprows=1`
  - `parse_dates=['Ticker']`
- Renames `Ticker` to `Date`.
- Sorts by `Date`.
- Forward-fills missing values with `fillna(method='ffill')`.
- Drops any remaining null rows.
- Writes output to `clean_stock_data.csv`.

Important implementation detail:

- The script writes to `clean_stock_data.csv` in the repository root, not to `data/clean_stock_data.csv`.
- The repository also already contains `data/clean_stock_data.csv`, so there are two competing path conventions in the project.

Observed output in `data/clean_stock_data.csv`:

- Shape `(1259, 4)`.
- Columns `['Date', 'AAPL', 'MSFT', 'GOOGL']`.

### Step 3: Feature engineering

File: [src/feature_engineering.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/feature_engineering.py:1)

Behavior inside `add_technical_indicators(df, column='Stock_Price')`:

- Converts `Date` to datetime.
- Drops rows with invalid dates.
- Sets `Date` as index.
- Validates that the target price column exists.
- Synthesizes:
  - `High = close * 1.02`
  - `Low = close * 0.98`
- Calculates indicators with the `ta` library:
  - `SMA_10`
  - `SMA_50`
  - `EMA_20`
  - `RSI_14`
  - `MACD`
  - `MACD_Signal`
  - `MACD_Hist`
  - `BB_Middle`
  - `BB_Upper`
  - `BB_Lower`
  - `ATR_14`
  - `ADX_14`
- Drops the initial rows made incomplete by rolling indicator windows.

Script-level behavior:

- Reads `clean_stock_data.csv` from the repository root.
- Force-renames columns to `['Date', 'AAPL', 'MSFT', 'GOOGL']`.
- Hard-codes `target_stock = "AAPL"`.
- Writes output to `engineered_stock_data.csv` in the repository root.

Important implementation detail:

- Only `AAPL` is used to derive `High`, `Low`, and all technical indicators.
- `MSFT` and `GOOGL` remain as raw close-price features without derived indicators.
- The project also stores `data/engineered_stock_data.csv`, so again there is a root-vs-`data/` path mismatch.

Observed output in `data/engineered_stock_data.csv`:

- Shape `(1209, 18)`.
- Columns:
  - `Date`
  - `AAPL`
  - `MSFT`
  - `GOOGL`
  - `High`
  - `Low`
  - `SMA_10`
  - `SMA_50`
  - `EMA_20`
  - `RSI_14`
  - `MACD`
  - `MACD_Signal`
  - `MACD_Hist`
  - `BB_Middle`
  - `BB_Upper`
  - `BB_Lower`
  - `ATR_14`
  - `ADX_14`

This explains the environment’s expectation of `17` numeric features after dropping `Date`.

### Step 4: Custom Gym environment

File: [src/environment.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/environment.py:1)

`TradingEnv` class behavior:

- Inherits from `gym.Env`.
- Copies the provided DataFrame.
- Removes `Date` if present.
- Computes `self.num_features`.
- Asserts that `self.num_features == 17`.
- Defines:
  - `action_space = spaces.Discrete(3)`
  - `observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)`

Observation flow:

- `reset()` sets `current_step = 0` and returns the first observation.
- `_next_observation()` returns the current row as a `float32` NumPy array.
- It raises if the observation width is not `17`.

Step flow:

- `step(action)` increments `current_step`.
- Sets `done` when the agent reaches the final row.
- Generates reward with `np.random.randn()`.
- Returns the next observation, random reward, done flag, and empty info dict.

What the environment actually models:

- It does not track:
  - cash,
  - position size,
  - portfolio value,
  - transaction costs,
  - slippage,
  - realized or unrealized PnL,
  - action validity.
- The action argument does not affect reward.
- Rewards are random noise, so PPO is optimizing against a stochastic placeholder rather than a trading objective.

Why this matters:

- The model can learn patterns in observation-to-action output only insofar as PPO tries to fit random rewards. There is no financial signal grounding the policy.
- Dashboard plots labeled as portfolio growth are cumulative random rewards, not actual portfolio valuation.

### Step 5: Training

File: [src/train_rl_model.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/train_rl_model.py:1)

Behavior:

- Imports `PPO` and `DummyVecEnv`.
- Imports `TradingEnv` using `from environment import TradingEnv`.
- Reads `engineered_stock_data.csv` from the repository root.
- Creates `TradingEnv(df)`.
- Wraps the same environment object in `DummyVecEnv([lambda: env])`.
- Instantiates `PPO("MlpPolicy", env, verbose=1)`.
- Trains for `50000` timesteps.
- Saves the model to `ppo_trading_model`.

Important implementation details:

- Import path is relative to running the script from inside `src/`, not from repository root. From the root directory, `from environment import TradingEnv` will fail unless Python path is adjusted.
- Saved model path is `ppo_trading_model`, while the app expects `model/ppo_trading_model.zip`.
- The repository already contains `model/ppo_trading_model.zip`, implying the checked-in artifact was moved or generated separately from this script’s current save path.

### Step 6: CLI evaluation

File: [src/evaluation.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/evaluation.py:1)

Behavior:

- Reads `engineered_stock_data.csv` from repository root.
- Drops `Date` if present.
- Asserts `17` features remain.
- Builds `TradingEnv(df_test)`.
- Loads model from `model/ppo_trading_model.zip`.
- Runs a prediction loop until `done`.
- Prints per-step `action` and `reward`.
- Prints `total_reward`.

Important implementation details:

- The script drops `Date` before constructing `TradingEnv`, but `TradingEnv` also drops `Date` if present. The duplicate handling is harmless.
- Import path has the same issue as training: `from environment import TradingEnv` assumes a particular working directory or Python path setup.
- Since the environment reward is random, evaluation is nondeterministic even when the policy is deterministic.

### Step 7: Streamlit app

File: [app.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/app.py:1)

Behavior:

- Configures a wide Streamlit layout.
- Prompts the user to upload a CSV.
- Loads the upload into a DataFrame.
- Drops `Date` if present.
- Loads the PPO model from `model/ppo_trading_model.zip`.
- Creates `TradingEnv(df_test)`.
- Runs a fixed-length evaluation loop.
- Builds four Plotly visualizations:
  - candlestick plus buy/sell markers,
  - cumulative reward area chart,
  - reward-per-step line chart,
  - action distribution pie chart.

Chart assumptions:

- Candlestick uses:
  - `open = df_test['AAPL']`
  - `high = df_test['High']`
  - `low = df_test['Low']`
  - `close = df_test['AAPL']`
- Buy markers use action `0`.
- Sell markers use action `2`.
- Action `1` is implicitly hold.

Important implementation details:

- The app assumes the uploaded CSV already contains the engineered columns required by `TradingEnv`, especially `High` and `Low`.
- If a user uploads raw or merely cleaned data, the environment assertion will fail because there will not be `17` features.
- The candlestick is synthetic because `open` and `close` both use `AAPL`, while `High` and `Low` were manufactured during feature engineering.
- Portfolio growth is computed with `np.cumsum(trade_df["Reward"])`, which is cumulative random reward rather than account equity.

## 5. Data Contracts

### Raw dataset contract

Source file:

- `data/stock_data1.csv`

Content:

- Date-like index exported from Yahoo Finance.
- Adjusted close prices for `AAPL`, `MSFT`, `GOOGL`.
- Extra header irregularity caused by concatenation/export format.

### Clean dataset contract

Expected columns:

- `Date`
- `AAPL`
- `MSFT`
- `GOOGL`

Purpose:

- Remove header irregularity.
- Ensure chronological ordering.
- Fill forward missing values.

### Engineered dataset contract

Expected columns:

- `Date`
- 17 numeric feature columns after dropping `Date`

Used by:

- `TradingEnv`
- training
- evaluation
- Streamlit upload path

## 6. Dependencies and Their Roles

File: [requirements.txt](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/requirements.txt:1)

Main packages:

- `numpy`, `pandas`: tabular data handling and numeric arrays.
- `yfinance`: historical stock data download.
- `ta`: technical analysis indicators.
- `gym`: RL environment interface.
- `stable-baselines3`: PPO implementation.
- `torch`: deep learning backend for Stable-Baselines3.
- `streamlit`: app framework.
- `plotly`: interactive plotting.
- `matplotlib`, `seaborn`: listed but not currently used by checked-in code.
- `joblib`, `scikit-learn`: listed but not currently used by checked-in code.

Versioning note:

- `python==3.8` in `requirements.txt` is not a valid pip package requirement for pinning an interpreter in the normal way. It documents intent but does not enforce Python 3.8 via pip alone.

## 7. Path and Import Conventions

The codebase mixes root-relative and `data/` or `model/` paths.

Current path behavior:

- `src/datacollection.py` writes `data/stock_data1.csv`
- `src/data_cleaning.py` reads `data/stock_data1.csv` but writes `clean_stock_data.csv`
- `src/feature_engineering.py` reads `clean_stock_data.csv` and writes `engineered_stock_data.csv`
- `src/train_rl_model.py` reads `engineered_stock_data.csv` and writes `ppo_trading_model`
- `src/evaluation.py` reads `engineered_stock_data.csv` and loads `model/ppo_trading_model.zip`
- `app.py` loads `model/ppo_trading_model.zip`

This means the scripts are not fully aligned with each other unless files are manually copied or execution happens with extra file management.

Import behavior:

- `app.py` correctly imports `from src.environment import TradingEnv`.
- `src/train_rl_model.py` and `src/evaluation.py` import `from environment import TradingEnv`, which is fragile and depends on the current working directory.

## 8. Actual Model Semantics

Despite the project naming, the current system is not a live trading bot and not yet a financially meaningful backtester.

What the PPO policy currently learns from:

- observations: the 17 engineered numeric features
- actions: buy/hold/sell labels
- rewards: Gaussian random numbers from `np.random.randn()`

What is missing for a real trading agent:

- stateful inventory and capital management
- action-to-position transition logic
- reward linked to returns or risk-adjusted returns
- train/validation/test data split
- benchmark comparison
- transaction fees
- reproducibility controls and seeding
- model configuration tracking
- experiment logging
- proper model artifact versioning

## 9. Operational Notes

### If you run the scripts as written

Likely execution sequence:

1. `python src/datacollection.py`
2. `python src/data_cleaning.py`
3. `python src/feature_engineering.py`
4. `python src/train_rl_model.py`
5. `python src/evaluation.py`
6. `streamlit run app.py`

But in practice you may need to correct:

- output/input file paths between steps,
- module imports in training and evaluation,
- location of the saved model artifact.

### Existing checked-in artifacts

The repository already contains:

- `data/stock_data1.csv`
- `data/clean_stock_data.csv`
- `data/engineered_stock_data.csv`
- `model/ppo_trading_model.zip`

These artifacts make the app and evaluation easier to inspect, but they also hide the fact that the scripts themselves are not fully path-consistent.

## 10. Main Technical Gaps

These are the most important engineering limitations in the current codebase:

1. Reward is random.
   The RL problem is not tied to trading performance.

2. Environment is not a portfolio simulator.
   Buy/sell/hold actions do not change holdings or capital.

3. File paths are inconsistent.
   Several scripts read and write different directories than the rest of the pipeline expects.

4. Import paths are inconsistent.
   `app.py` uses `src.environment`; training and evaluation do not.

5. The app expects engineered input.
   A user-uploaded raw CSV will not satisfy the environment contract.

6. Candlestick data is synthetic.
   High/low are estimated from close; open equals close.

7. Documentation overstates current capability.
   The README describes an AI-powered trading bot, while the code is a prototype with placeholder reward mechanics.

## 11. Suggested Next Refactor Targets

If the goal is to evolve this into a coherent trading research project, the most valuable changes would be:

1. Standardize all file paths under `data/` and `model/`.
2. Make imports package-consistent across all scripts.
3. Replace random rewards with portfolio-return-based rewards.
4. Track cash, positions, portfolio value, and transaction costs in `TradingEnv`.
5. Separate train, validation, and test datasets.
6. Save model artifacts to the exact path consumed by evaluation and Streamlit.
7. Add configuration instead of hard-coding `AAPL` and date ranges.
8. Add deterministic seeding and reproducible experiment settings.

## 12. Quick Reference

### Main entry points

- Data download: [src/datacollection.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/datacollection.py:1)
- Data cleaning: [src/data_cleaning.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/data_cleaning.py:1)
- Feature engineering: [src/feature_engineering.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/feature_engineering.py:1)
- Environment: [src/environment.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/environment.py:1)
- Training: [src/train_rl_model.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/train_rl_model.py:1)
- Evaluation: [src/evaluation.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/src/evaluation.py:1)
- Dashboard: [app.py](C:/Users/Dhananjay%20Vaidya/Downloads/NeuralTrade/app.py:1)

### Core assumptions hard-coded in code

- Tickers: `AAPL`, `MSFT`, `GOOGL`
- Main engineered stock: `AAPL`
- Action space: `3`
- Observation width: `17`
- Training algorithm: `PPO`
- Training horizon: `50000` timesteps
- Historical data window: `2019-01-01` to `2024-01-01`

## 13. Bottom Line

This repository implements the skeleton of an RL-based trading workflow:

- data ingestion,
- preprocessing,
- indicator generation,
- RL environment wiring,
- PPO training,
- evaluation,
- UI visualization.

The codebase is understandable and runnable with adjustments, but the current environment logic makes it a demonstrator rather than a valid trading system. The most important fact about the code is that the learning signal is currently random, so every downstream metric should be interpreted as prototype output, not trading performance.
