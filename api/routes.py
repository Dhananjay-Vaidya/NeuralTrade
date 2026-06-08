"""FastAPI route handlers."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from neuraltrade.config import cfg
from neuraltrade.data_splits import load_and_split
from neuraltrade.environment import BUY, HOLD, SELL, TradingEnv
from neuraltrade.feature_engineering import FEATURE_COLUMNS, add_indicators
from neuraltrade.metrics import compute_all
from neuraltrade.seeding import seed_everything

from .deps import data_available, get_model, model_available
from .models import (
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    Metrics,
    OHLCVBar,
    PredictResponse,
    TickersResponse,
    Trade,
)

router = APIRouter(prefix="/api")

_ACTION_NAMES = {BUY: "BUY", HOLD: "HOLD", SELL: "SELL"}


# ── GET /api/health ────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_available(),
        data_available=data_available(),
    )


# ── GET /api/tickers ───────────────────────────────────────────────────────────

@router.get("/tickers", response_model=TickersResponse)
def tickers() -> TickersResponse:
    return TickersResponse(
        tickers=cfg.data.tickers,
        primary=cfg.data.primary_ticker,
    )


# ── POST /api/backtest ────────────────────────────────────────────────────────

@router.post("/backtest", response_model=BacktestResponse)
def backtest(req: BacktestRequest) -> BacktestResponse:
    if not model_available():
        raise HTTPException(
            503,
            detail="Model not trained yet. Run `python -m neuraltrade.train`.",
        )
    if not data_available():
        raise HTTPException(
            503,
            detail="Engineered data not found. Run the data pipeline first.",
        )

    ticker = req.ticker
    if ticker not in cfg.data.tickers:
        raise HTTPException(
            404,
            detail=f"Ticker '{ticker}' not in configured list: {cfg.data.tickers}",
        )

    seed_everything(cfg.model.seed)

    # --- Load data split for the requested ticker ---
    try:
        split = load_and_split(cfg.data, ticker=ticker)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    # --- Run policy rollout on test split ---
    model = get_model()
    env = TradingEnv(
        df=split.test,
        price_col="close",
        env_cfg=cfg.env,
        seed=cfg.model.seed,
    )
    env._prices = split.test_prices.to_numpy(dtype=np.float64)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

    # --- Build metrics ---
    raw_metrics = compute_all(env.equity_curve, env.trade_log, split.test_prices.to_numpy())

    # --- Build OHLCV bars (unscaled prices from test_prices; reconstruct H/L/O from features) ---
    prices_arr = split.test_prices.to_numpy(dtype=float)
    n = len(prices_arr)

    # test_raw stored in split as unscaled DataFrame — we need to pull from engineered data
    # We reconstruct approximate open/high/low from test_prices (close) for display.
    # When the real parquet has H/L/O columns they're in split.test (scaled); we use test_prices
    # as close and approximate the rest.
    opens = np.roll(prices_arr, 1)
    opens[0] = prices_arr[0]

    dates = split.test_prices.index
    date_strs = [d.strftime("%Y-%m-%dT%H:%M:%S") if hasattr(d, "strftime") else str(d) for d in dates]

    # Pull unscaled high/low from the parquet if available
    highs, lows, volumes = _get_unscaled_hl(ticker, len(prices_arr), split.test_prices.index)

    ohlcv = [
        OHLCVBar(
            timestamp=date_strs[i],
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(prices_arr[i]),
            volume=float(volumes[i]),
        )
        for i in range(n)
    ]

    # --- Build trade list ---
    trades: list[Trade] = []
    for t in env.trade_log:
        step_idx = t["step"]
        if step_idx < len(date_strs):
            trades.append(
                Trade(
                    timestamp=date_strs[step_idx],
                    side=t["action"],
                    price=float(t["price"]),
                    shares=float(t["shares"]),
                )
            )

    # --- Equity / B&H curves ---
    equity = list(map(float, env.equity_curve))
    bh_curve = [float(cfg.env.initial_cash * (p / prices_arr[0])) for p in prices_arr]

    # Align equity_curve length to n (equity has n+1 values including initial)
    eq_dates = date_strs[: len(equity)]

    return BacktestResponse(
        ticker=ticker,
        ohlcv=ohlcv,
        trades=trades,
        equity_curve=equity,
        buy_and_hold_curve=bh_curve,
        timestamps=eq_dates,
        metrics=Metrics(**raw_metrics),
        initial_cash=float(cfg.env.initial_cash),
    )


# ── GET /api/predict/{ticker} ─────────────────────────────────────────────────

@router.get("/predict/{ticker}", response_model=PredictResponse)
def predict(ticker: str) -> PredictResponse:
    if not model_available():
        raise HTTPException(
            503,
            detail="Model not trained yet. Run `python -m neuraltrade.train`.",
        )
    if not data_available():
        raise HTTPException(
            503,
            detail="Engineered data not found. Run the data pipeline first.",
        )

    if ticker not in cfg.data.tickers:
        raise HTTPException(
            404,
            detail=f"Ticker '{ticker}' not in configured list: {cfg.data.tickers}",
        )

    seed_everything(cfg.model.seed)

    try:
        split = load_and_split(cfg.data, ticker=ticker)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, detail=str(exc)) from exc

    # Use the latest bar of the test set as the observation
    model = get_model()
    env = TradingEnv(
        df=split.test,
        price_col="close",
        env_cfg=cfg.env,
        seed=cfg.model.seed,
    )
    env._prices = split.test_prices.to_numpy(dtype=np.float64)

    # Fast-forward to the last step by running the full episode
    obs, _ = env.reset()
    last_action = HOLD
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        last_action = int(action)
        obs, _, terminated, truncated, _ = env.step(last_action)
        done = terminated or truncated

    latest_price = float(split.test_prices.iloc[-1])
    latest_date = str(split.test_prices.index[-1])[:10]

    return PredictResponse(
        ticker=ticker,
        action=_ACTION_NAMES[last_action],
        action_code=last_action,
        confidence=None,
        latest_close=latest_price,
        latest_date=latest_date,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _get_unscaled_hl(
    ticker: str,
    n: int,
    index,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Try to pull real high/low/volume from the engineered parquet; fall back to close."""
    try:
        import pandas as pd
        parquet_path = cfg.data.abs(cfg.data.engineered_path).with_suffix(".parquet")
        if parquet_path.exists():
            combined = pd.read_parquet(parquet_path)
            df = combined[combined["ticker"] == ticker]
            if not df.empty and "high" in df.columns and "low" in df.columns:
                # Align by index
                df = df.loc[df.index.isin(index)]
                if len(df) >= n:
                    df = df.iloc[-n:]
                    return (
                        df["high"].to_numpy(dtype=float),
                        df["low"].to_numpy(dtype=float),
                        df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.ones(n),
                    )
    except Exception:
        pass

    # Fallback: use close price for H/L, volume=1
    prices = np.ones(n, dtype=float)
    return prices, prices, np.ones(n, dtype=float)
