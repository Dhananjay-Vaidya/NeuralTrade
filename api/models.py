"""Pydantic v2 response/request models for the NeuralTrade API.

All shapes defined here are mirrored in frontend/src/lib/api.ts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g. 'AAPL'")
    start: str | None = Field(None, description="ISO date 'YYYY-MM-DD'; defaults to config")
    end: str | None = Field(None, description="ISO date 'YYYY-MM-DD'; defaults to config")


# ── Shared primitives ──────────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    timestamp: str          # ISO-8601 date string
    open: float
    high: float
    low: float
    close: float
    volume: float


class Trade(BaseModel):
    timestamp: str
    side: Literal["BUY", "SELL"]
    price: float
    shares: float


class Metrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    buy_and_hold_return: float


# ── Endpoint responses ─────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    data_available: bool


class TickersResponse(BaseModel):
    tickers: list[str]
    primary: str


class BacktestResponse(BaseModel):
    ticker: str
    ohlcv: list[OHLCVBar]
    trades: list[Trade]
    equity_curve: list[float]
    buy_and_hold_curve: list[float]
    timestamps: list[str]      # aligned with equity_curve / buy_and_hold_curve
    metrics: Metrics
    initial_cash: float


class PredictResponse(BaseModel):
    ticker: str
    action: Literal["BUY", "HOLD", "SELL"]
    action_code: int           # 0=BUY 1=HOLD 2=SELL
    confidence: float | None   # softmax max prob when available
    latest_close: float
    latest_date: str
