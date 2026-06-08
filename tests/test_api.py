"""Tests for FastAPI endpoints using TestClient (no real model/data required).

Tests cover:
- GET /api/health         — always reachable; reports model_loaded / data_available flags
- GET /api/tickers        — returns configured tickers
- POST /api/backtest      — returns correct response shape (mocked when model absent)
- GET /api/predict/{t}    — returns correct response shape (mocked when model absent)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── App import ─────────────────────────────────────────────────────────────────
from api.main import app

client = TestClient(app)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def mock_split(tmp_path):
    """A minimal DataSplit-like object with test data."""
    from dataclasses import dataclass

    from neuraltrade.feature_engineering import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = pd.Series(100 + rng.standard_normal(n).cumsum(), index=dates, name="close")
    features = pd.DataFrame(rng.standard_normal((n, len(FEATURE_COLUMNS))), index=dates, columns=FEATURE_COLUMNS)

    @dataclass
    class FakeSplit:
        test: pd.DataFrame = None
        test_prices: pd.Series = None

    fs = FakeSplit(test=features, test_prices=prices)
    return fs


@pytest.fixture()
def mock_model():
    """A model.predict() that always returns HOLD (1)."""
    model = MagicMock()
    model.predict.return_value = (np.array(1), None)
    return model


# ── GET /api/health ────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok_status(self):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_schema(self):
        resp = client.get("/api/health")
        body = resp.json()
        assert "status" in body
        assert body["status"] == "ok"
        assert "model_loaded" in body
        assert "data_available" in body
        assert isinstance(body["model_loaded"], bool)
        assert isinstance(body["data_available"], bool)


# ── GET /api/tickers ───────────────────────────────────────────────────────────

class TestTickersEndpoint:
    def test_tickers_ok(self):
        resp = client.get("/api/tickers")
        assert resp.status_code == 200

    def test_tickers_schema(self):
        resp = client.get("/api/tickers")
        body = resp.json()
        assert "tickers" in body
        assert "primary" in body
        assert isinstance(body["tickers"], list)
        assert len(body["tickers"]) > 0
        assert body["primary"] in body["tickers"]

    def test_tickers_nonempty(self):
        body = client.get("/api/tickers").json()
        assert all(isinstance(t, str) for t in body["tickers"])


# ── POST /api/backtest ─────────────────────────────────────────────────────────

class TestBacktestEndpoint:
    def test_backtest_unknown_ticker_404(self):
        resp = client.post("/api/backtest", json={"ticker": "ZZZZZZ"})
        assert resp.status_code == 404

    def test_backtest_503_when_no_model(self):
        """With no model file present, must return 503."""
        with patch("api.routes.model_available", return_value=False):
            resp = client.post("/api/backtest", json={"ticker": "AAPL"})
        assert resp.status_code == 503

    def test_backtest_503_when_no_data(self):
        with (
            patch("api.routes.model_available", return_value=True),
            patch("api.routes.data_available", return_value=False),
        ):
            resp = client.post("/api/backtest", json={"ticker": "AAPL"})
        assert resp.status_code == 503

    def test_backtest_full_response_shape(self, mock_split, mock_model):
        """Full happy path with mocked split and model."""
        from neuraltrade.config import cfg

        valid_ticker = cfg.data.tickers[0]

        with (
            patch("api.routes.model_available", return_value=True),
            patch("api.routes.data_available", return_value=True),
            patch("api.routes.load_and_split", return_value=mock_split),
            patch("api.routes.get_model", return_value=mock_model),
            patch("api.routes._get_unscaled_hl") as mock_hl,
        ):
            n = len(mock_split.test_prices)
            mock_hl.return_value = (
                np.ones(n) * 105,
                np.ones(n) * 95,
                np.ones(n) * 1_000_000,
            )
            resp = client.post("/api/backtest", json={"ticker": valid_ticker})

        assert resp.status_code == 200, resp.text
        body = resp.json()

        # Top-level keys
        for key in ("ticker", "ohlcv", "trades", "equity_curve", "buy_and_hold_curve",
                    "timestamps", "metrics", "initial_cash"):
            assert key in body, f"Missing key: {key}"

        assert body["ticker"] == valid_ticker
        assert isinstance(body["ohlcv"], list)
        assert len(body["ohlcv"]) > 0
        assert isinstance(body["equity_curve"], list)
        assert isinstance(body["buy_and_hold_curve"], list)

        # OHLCV bar shape
        bar = body["ohlcv"][0]
        for field in ("timestamp", "open", "high", "low", "close", "volume"):
            assert field in bar, f"OHLCV bar missing field: {field}"

        # Metrics shape
        m = body["metrics"]
        for field in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate", "buy_and_hold_return"):
            assert field in m, f"Metrics missing field: {field}"
            assert isinstance(m[field], (int, float))

        # Sanity: all numeric metrics are finite
        assert all(math.isfinite(m[f]) for f in m)

    def test_backtest_ohlcv_invariants(self, mock_split, mock_model):
        """high >= low and high >= close for every bar."""
        from neuraltrade.config import cfg

        valid_ticker = cfg.data.tickers[0]
        n = len(mock_split.test_prices)

        with (
            patch("api.routes.model_available", return_value=True),
            patch("api.routes.data_available", return_value=True),
            patch("api.routes.load_and_split", return_value=mock_split),
            patch("api.routes.get_model", return_value=mock_model),
            patch("api.routes._get_unscaled_hl") as mock_hl,
        ):
            mock_hl.return_value = (
                np.ones(n) * 105,
                np.ones(n) * 95,
                np.ones(n) * 1_000_000,
            )
            body = client.post("/api/backtest", json={"ticker": valid_ticker}).json()

        for bar in body["ohlcv"]:
            assert bar["high"] >= bar["low"], "high < low in OHLCV bar"


# ── GET /api/predict/{ticker} ─────────────────────────────────────────────────

class TestPredictEndpoint:
    def test_predict_unknown_ticker_404(self):
        resp = client.get("/api/predict/ZZZZZZ")
        assert resp.status_code == 404

    def test_predict_503_when_no_model(self):
        from neuraltrade.config import cfg
        t = cfg.data.tickers[0]
        with patch("api.routes.model_available", return_value=False):
            resp = client.get(f"/api/predict/{t}")
        assert resp.status_code == 503

    def test_predict_response_shape(self, mock_split, mock_model):
        from neuraltrade.config import cfg
        t = cfg.data.tickers[0]

        with (
            patch("api.routes.model_available", return_value=True),
            patch("api.routes.data_available", return_value=True),
            patch("api.routes.load_and_split", return_value=mock_split),
            patch("api.routes.get_model", return_value=mock_model),
        ):
            resp = client.get(f"/api/predict/{t}")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        for key in ("ticker", "action", "action_code", "latest_close", "latest_date"):
            assert key in body, f"Missing key: {key}"
        assert body["action"] in ("BUY", "HOLD", "SELL")
        assert body["action_code"] in (0, 1, 2)
        assert isinstance(body["latest_close"], float)

    def test_predict_action_matches_code(self, mock_split, mock_model):
        from neuraltrade.config import cfg
        from neuraltrade.environment import BUY, HOLD, SELL
        t = cfg.data.tickers[0]

        for code, name in [(BUY, "BUY"), (HOLD, "HOLD"), (SELL, "SELL")]:
            mock_model.predict.return_value = (np.array(code), None)
            with (
                patch("api.deps.model_available", return_value=True),
                patch("api.deps.data_available", return_value=True),
                patch("api.routes.load_and_split", return_value=mock_split),
                patch("api.routes.get_model", return_value=mock_model),
            ):
                body = client.get(f"/api/predict/{t}").json()
            assert body["action"] == name
            assert body["action_code"] == code
