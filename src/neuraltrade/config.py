"""Central configuration — load once, import everywhere."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

_PROJECT_ROOT = Path(__file__).parent.parent.parent


class DataConfig(BaseModel):
    tickers: list[str] = ["AAPL", "MSFT", "GOOGL"]
    primary_ticker: str = "AAPL"
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"
    raw_path: Path = Path("data/stock_data1.csv")
    clean_path: Path = Path("data/clean_stock_data.csv")
    engineered_path: Path = Path("data/engineered_stock_data.csv")
    train_ratio: float = Field(0.7, gt=0, lt=1)
    val_ratio: float = Field(0.15, gt=0, lt=1)

    @property
    def test_ratio(self) -> float:
        return 1.0 - self.train_ratio - self.val_ratio

    def abs(self, p: Path) -> Path:
        """Return absolute path relative to project root."""
        return _PROJECT_ROOT / p


class ModelConfig(BaseModel):
    save_path: Path = Path("model/ppo_trading_model.zip")
    algorithm: Literal["PPO", "RecurrentPPO"] = "PPO"
    total_timesteps: int = 50_000
    seed: int = 42
    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    def abs(self, p: Path) -> Path:
        return _PROJECT_ROOT / p


class EnvConfig(BaseModel):
    initial_cash: float = 10_000.0
    transaction_cost_bps: int = 10
    slippage_bps: int = 5
    sizing: Literal["all_in", "fixed_fraction"] = "all_in"
    fixed_fraction: float = Field(0.25, gt=0, le=1)
    reward_type: Literal["log_return", "risk_adjusted"] = "log_return"
    risk_adjusted_drawdown_penalty: float = 0.1


class MLflowConfig(BaseModel):
    tracking_uri: str = "mlruns"
    experiment_name: str = "neuraltrade"


class Settings(BaseSettings):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    env: EnvConfig = EnvConfig()
    mlflow: MLflowConfig = MLflowConfig()

    model_config = {"arbitrary_types_allowed": True}


def load_settings(config_path: Path | None = None) -> Settings:
    """Load settings from config.yaml, falling back to defaults."""
    path = config_path or (_PROJECT_ROOT / "config.yaml")
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f)
        return Settings(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            env=EnvConfig(**raw.get("env", {})),
            mlflow=MLflowConfig(**raw.get("mlflow", {})),
        )
    return Settings()


# Module-level singleton — import cfg from neuraltrade.config
cfg = load_settings()
