# NeuralTrade — Architecture & Bug Audit

## 1. System Overview

A Reinforcement Learning stock-trading prototype with a five-stage pipeline:

```
datacollection.py  →  data_cleaning.py  →  feature_engineering.py
                                                    ↓
                                          train_rl_model.py
                                                    ↓
                                           evaluation.py  /  app.py
```

The trained model is a PPO agent (Stable-Baselines3) that acts in a custom
OpenAI Gym environment (`TradingEnv`). The Streamlit app (`app.py`) re-uses
the same environment for live demo evaluation.

---

## 2. Component Descriptions

### `src/datacollection.py`
- Downloads OHLC data for AAPL, MSFT, GOOGL from Yahoo Finance (2019-01-01 → 2024-01-01).
- Keeps only `Close` prices → 3-column wide DataFrame.
- Saves to **`data/stock_data1.csv`** (index = ticker row + date rows).

### `src/data_cleaning.py`
- Reads `data/stock_data1.csv` with `skiprows=1`, treating the second header
  row as the first real row.
- Renames the first column (`Ticker`) to `Date`.
- Forward-fills then drops NaN rows.
- Saves to **`clean_stock_data.csv`** (project root, not `data/`).

### `src/feature_engineering.py`
- Reads `clean_stock_data.csv` (project root).
- Hard-codes column rename to `['Date', 'AAPL', 'MSFT', 'GOOGL']`.
- Derives `High = AAPL * 1.02`, `Low = AAPL * 0.98` (synthetic, not real OHLC).
- Adds 12 technical indicators: SMA_10, SMA_50, EMA_20, RSI_14, MACD,
  MACD_Signal, MACD_Hist, BB_Middle, BB_Upper, BB_Lower, ATR_14, ADX_14.
- Sets `Date` as the index.
- Saves to **`engineered_stock_data.csv`** (project root, not `data/`).
- Resulting shape: **1209 rows × 18 columns** (Date index + 17 feature columns).

### `src/environment.py` — `TradingEnv(gym.Env)`
- Accepts a DataFrame, drops `Date` if present.
- Hard-asserts exactly **17 features**.
- `action_space`: Discrete(3) — 0=Buy, 1=Hold, 2=Sell.
- `observation_space`: Box(17,).
- `reset()`: sets `current_step = 0`, returns first observation row.
- `step(action)`: increments step, computes reward, returns next obs.
- `done` flag: `current_step >= len(df) - 1`.

### `src/train_rl_model.py`
- Reads `engineered_stock_data.csv` (no path prefix — CWD-relative).
- Imports `TradingEnv` from `environment` (bare module name, no `src.` prefix).
- Wraps env in `DummyVecEnv`.
- Trains PPO for 50,000 timesteps.
- Saves model to **`ppo_trading_model`** (no directory, no `.zip`).

### `src/evaluation.py`
- Reads `engineered_stock_data.csv` (CWD-relative).
- Imports `TradingEnv` from `environment` (bare module name).
- Loads model from **`model/ppo_trading_model.zip`**.
- Runs a deterministic rollout and prints total reward.

### `app.py`
- Streamlit dashboard — accepts a user-uploaded CSV.
- Drops `Date` column before passing to `TradingEnv`.
- Imports `TradingEnv` from `src.environment` (correct package-qualified path).
- Loads model from **`model/ppo_trading_model.zip`**.
- Runs evaluation loop and renders candlestick chart + portfolio/reward/action charts.

---

## 3. Data Contracts Between Scripts

| Stage | Produces | Expected by next stage |
|---|---|---|
| `datacollection.py` | `data/stock_data1.csv` — 4 cols (index, AAPL, MSFT, GOOGL), double header | `data_cleaning.py` reads with `skiprows=1` |
| `data_cleaning.py` | `clean_stock_data.csv` (root) — 4 cols: Date, AAPL, MSFT, GOOGL | `feature_engineering.py` reads from root |
| `feature_engineering.py` | `engineered_stock_data.csv` (root) — 18 cols: Date + 17 features | `train_rl_model.py` & `evaluation.py` read from root |
| `train_rl_model.py` | `ppo_trading_model` (root, no `.zip`) | `evaluation.py` & `app.py` look for `model/ppo_trading_model.zip` |
| `TradingEnv` | obs shape `(17,)` | PPO policy input — must stay consistent between train & eval |

---

## 4. Bugs & Inconsistencies Checklist

### Critical — will cause crashes or wrong results

- [ ] **BUG-01 · Reward function is random noise**
  `environment.py:59` — `reward = np.random.randn()`. This is a commented
  placeholder that was never replaced with an actual trading reward (e.g.
  mark-to-market P&L). The agent trains on pure white noise, making any
  learned policy meaningless.

- [ ] **BUG-02 · Model save path mismatch (train vs. eval)**
  `train_rl_model.py:26` saves to `"ppo_trading_model"` (CWD root, no
  directory). `evaluation.py:34` and `app.py:24` both load from
  `"model/ppo_trading_model.zip"`. The file is never written to `model/`,
  so evaluation will `AssertionError` / `FileNotFoundError` on a clean run.

- [ ] **BUG-03 · Feature count hard-coded to 17, but engineered CSV has 18 columns**
  `engineered_stock_data.csv` has 18 columns when read with default
  `pd.read_csv` (Date + 17 features). `feature_engineering.py` sets `Date`
  as the index before saving, but `train_rl_model.py` and `evaluation.py`
  read the file without `index_col=0`, so `Date` comes back as a normal
  column → 18 columns. `TradingEnv.__init__` asserts `num_features == 17`
  and crashes. (`app.py` sidesteps this only because it explicitly drops
  `Date` before constructing the env.)

- [ ] **BUG-04 · `data_cleaning.py` uses deprecated `fillna(method=)`**
  `data_cleaning.py:14` calls `df.fillna(method='ffill')`, which raises a
  `FutureWarning` in Pandas ≥ 2.1 and will error in a future release. The
  correct call is `df.ffill()`.

- [ ] **BUG-05 · Intermediate CSVs saved to project root, not `data/`**
  `data_cleaning.py` and `feature_engineering.py` write `clean_stock_data.csv`
  and `engineered_stock_data.csv` to CWD (the project root). The README and
  `data/` directory imply they should live under `data/`. This inconsistency
  means the files are only found if scripts are run from the root, and the
  repo's `data/` folder is inconsistently populated.

- [ ] **BUG-06 · `train_rl_model.py` and `evaluation.py` use bare `from environment import`**
  These scripts do `from environment import TradingEnv` — a bare (non-package)
  import that only works if the scripts are run with `src/` as CWD. Running
  them from the project root as `python src/train_rl_model.py` (as the README
  instructs) will raise `ModuleNotFoundError`. `app.py` correctly uses
  `from src.environment import TradingEnv`.

### Logic / Design Issues

- [ ] **BUG-07 · `High` and `Low` are synthetic, not real OHLC data**
  `feature_engineering.py:21-22` fabricates `High = AAPL * 1.02` and
  `Low = AAPL * 0.98`. ATR and ADX are therefore computed from fake spread,
  making those indicators unreliable. Real high/low prices are available from
  `yfinance` and are not fetched.

- [ ] **BUG-08 · Candlestick chart in `app.py` uses wrong columns**
  `app.py:47-51` passes `open=df_test['AAPL']` and `close=df_test['AAPL']`
  (same column for both), so every candlestick has zero body. The `High` and
  `Low` columns exist in the engineered data and are used correctly, but
  there is no real `Open` column — it should at minimum use the previous
  close as `open`, or the chart should be a line chart instead.

- [ ] **BUG-09 · Portfolio value calculated as cumulative sum of random rewards**
  `app.py:64` computes `portfolio_values = np.cumsum(trade_df["Reward"])`.
  Because rewards are `np.random.randn()` (BUG-01), this is a random walk,
  not a portfolio value.

- [ ] **BUG-10 · `done` flag off-by-one: last row is never observed**
  `environment.py:57` — `done = self.current_step >= len(self.df) - 1`.
  After incrementing the step, `_next_observation` is called while
  `current_step == len(df) - 1`, fetching the last row. But on the *next*
  call, `done` is set before the agent sees that final observation, meaning
  the loop exits one step early. Conventionally `done` should be
  `current_step >= len(self.df)` (strictly equal or after) and the
  observation should be returned at the terminal step.

- [ ] **BUG-11 · `step()` calls `_next_observation()` even after `done=True`**
  When `current_step == len(df) - 1`, `done` is `True` but `_next_observation`
  is still called, which will access `df.iloc[len(df) - 1]` without issue
  this cycle — but on the very next step (if the caller ignores `done`) it
  will `IndexError`. There is no guard against stepping past the end.

- [ ] **BUG-12 · `evaluation.py` has a broken import for unused `gym`**
  `evaluation.py:3` — `import gym` is present but `gym` is never used
  directly in the file. Minor, but indicates copy-paste residue.

- [ ] **BUG-13 · `requirements.txt` pins `python==3.8` as a pip dependency**
  `requirements.txt:1` — `python==3.8` is not a valid pip requirement; pip
  cannot install a Python version. This causes a harmless warning but could
  confuse automated installers.

- [ ] **BUG-14 · `app.py` action encoding: Buy=0, Sell=2, but env has no semantic mapping**
  `app.py:52-53` treats action `0` as Buy and action `2` as Sell. This
  ordering is undocumented in `environment.py` — there is no `ACTION_BUY`,
  `ACTION_HOLD`, `ACTION_SELL` constant. If the mapping is ever changed, the
  dashboard silently shows wrong markers.

- [ ] **BUG-15 · `data_cleaning.py` `parse_dates` applied to wrong column**
  `data_cleaning.py:5` — `pd.read_csv(..., skiprows=1, parse_dates=['Ticker'])`.
  After `skiprows=1` the first column is the date column but it is still
  named `Ticker` in the raw CSV header. `parse_dates=['Ticker']` works only
  because the raw file happens to name that column `Ticker`, but the intent
  (parse the date column) would break if the column name ever changes. The
  rename to `Date` happens on line 8, after parsing, so this is fragile.

---

## 5. Dependency & Environment Notes

- `gym` is imported directly; `stable-baselines3` ≥ 1.7 ships its own
  `gymnasium`-compatible shim — mixing `gym` and `gymnasium` APIs can cause
  silent incompatibilities on newer SB3 versions.
- No `gymnasium` is listed in `requirements.txt`; `gym` (legacy OpenAI Gym)
  is used throughout.
- `torch` is listed without a version pin, which can break with CUDA/CPU
  incompatibilities depending on the install environment.
- `ta` library is listed with a trailing space in `requirements.txt` (`ta `);
  harmless but untidy.
