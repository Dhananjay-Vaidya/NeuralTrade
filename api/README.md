# NeuralTrade API — Contract Reference

> ⚠️ **Research / backtesting output only. Not financial advice.**

Run the server:
```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/docs

---

## Endpoints

### `GET /api/health`

Check backend status.

**Response `200`**
```json
{
  "status": "ok",
  "model_loaded": true,
  "data_available": true
}
```

| Field | Type | Notes |
|---|---|---|
| `status` | `"ok"` | Always `"ok"` when server is running |
| `model_loaded` | `boolean` | True when `model/ppo_trading_model.zip` exists |
| `data_available` | `boolean` | True when engineered data parquet/CSV exists |

---

### `GET /api/tickers`

Return configured ticker symbols.

**Response `200`**
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "primary": "AAPL"
}
```

| Field | Type | Notes |
|---|---|---|
| `tickers` | `string[]` | All configured tickers from `config.yaml` |
| `primary` | `string` | Default ticker (subset of `tickers`) |

---

### `POST /api/backtest`

Run a full backtest rollout on the test split for a given ticker and return
OHLCV data, trade signals, equity curve, and performance metrics.

**Request body**
```json
{
  "ticker": "AAPL",
  "start": "2022-01-01",
  "end": "2023-12-31"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `ticker` | `string` | ✅ | Must be in the configured tickers list |
| `start` | `string \| null` | ❌ | ISO date; defaults to config |
| `end` | `string \| null` | ❌ | ISO date; defaults to config |

**Response `200`**
```json
{
  "ticker": "AAPL",
  "ohlcv": [
    {
      "timestamp": "2023-01-03T00:00:00",
      "open": 130.28,
      "high": 130.90,
      "low": 124.17,
      "close": 125.07,
      "volume": 112117500
    }
  ],
  "trades": [
    {
      "timestamp": "2023-01-04T00:00:00",
      "side": "BUY",
      "price": 126.36,
      "shares": 79.15
    }
  ],
  "equity_curve": [10000.0, 10012.3, ...],
  "buy_and_hold_curve": [10000.0, 10009.7, ...],
  "timestamps": ["2023-01-03T00:00:00", ...],
  "metrics": {
    "total_return": 0.142,
    "sharpe_ratio": 1.23,
    "max_drawdown": 0.087,
    "win_rate": 0.6,
    "buy_and_hold_return": 0.118
  },
  "initial_cash": 10000.0
}
```

**OHLCVBar**

| Field | Type |
|---|---|
| `timestamp` | `string` (ISO-8601) |
| `open` | `number` |
| `high` | `number` |
| `low` | `number` |
| `close` | `number` |
| `volume` | `number` |

**Trade**

| Field | Type | Notes |
|---|---|---|
| `timestamp` | `string` (ISO-8601) | Date of trade execution |
| `side` | `"BUY" \| "SELL"` | |
| `price` | `number` | Execution price including slippage |
| `shares` | `number` | Shares transacted |

**Metrics**

| Field | Type | Notes |
|---|---|---|
| `total_return` | `number` | `(final - initial) / initial` |
| `sharpe_ratio` | `number` | Annualised, daily-step assumed |
| `max_drawdown` | `number` | Peak-to-trough fraction (0–1) |
| `win_rate` | `number` | Fraction of profitable round-trips |
| `buy_and_hold_return` | `number` | Naive B&H return for comparison |

**Error responses**

| Code | When |
|---|---|
| `404` | Ticker not in configured list |
| `503` | Model not trained or data not available |

---

### `GET /api/predict/{ticker}`

Return the action signal for the most recent bar in the test split.

**Path parameter:** `ticker` — must match a configured ticker

**Response `200`**
```json
{
  "ticker": "AAPL",
  "action": "BUY",
  "action_code": 0,
  "confidence": null,
  "latest_close": 187.44,
  "latest_date": "2023-12-29"
}
```

| Field | Type | Notes |
|---|---|---|
| `ticker` | `string` | |
| `action` | `"BUY" \| "HOLD" \| "SELL"` | Last action from the deterministic policy |
| `action_code` | `0 \| 1 \| 2` | 0=BUY, 1=HOLD, 2=SELL |
| `confidence` | `number \| null` | Softmax max probability; `null` for MlpPolicy |
| `latest_close` | `number` | Most recent close price |
| `latest_date` | `string` | ISO date string (YYYY-MM-DD) |

**Error responses**

| Code | When |
|---|---|
| `404` | Ticker not in configured list |
| `503` | Model not trained or data not available |

---

## TypeScript types (mirrored in `frontend/src/lib/api.ts`)

```typescript
export interface HealthResponse {
  status: "ok";
  model_loaded: boolean;
  data_available: boolean;
}

export interface TickersResponse {
  tickers: string[];
  primary: string;
}

export interface BacktestRequest {
  ticker: string;
  start?: string | null;
  end?: string | null;
}

export interface OHLCVBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  timestamp: string;
  side: "BUY" | "SELL";
  price: number;
  shares: number;
}

export interface Metrics {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  buy_and_hold_return: number;
}

export interface BacktestResponse {
  ticker: string;
  ohlcv: OHLCVBar[];
  trades: Trade[];
  equity_curve: number[];
  buy_and_hold_curve: number[];
  timestamps: string[];
  metrics: Metrics;
  initial_cash: number;
}

export interface PredictResponse {
  ticker: string;
  action: "BUY" | "HOLD" | "SELL";
  action_code: 0 | 1 | 2;
  confidence: number | null;
  latest_close: number;
  latest_date: string;
}
```
