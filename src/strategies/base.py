from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

SignalConfig = Dict[str, Any]

DEFAULT_SIGNAL_CONFIG: SignalConfig = {
    "enable_trend_long": True,
    "enable_pullback_long": True,
    "regime_filter_enabled": True,
    "entry_score_floor": {"trend_long": 0.0, "pullback_long": 0.0},
    "risk_scalars": {"trend_long": 1.10, "pullback_long": 0.88},
}

PROFILE_CONFIGS: Dict[str, SignalConfig] = {
    "trend_only": {**DEFAULT_SIGNAL_CONFIG, "enable_pullback_long": False},
    "full_system": {**DEFAULT_SIGNAL_CONFIG, "enable_pullback_long": True},
    "next_research": {**DEFAULT_SIGNAL_CONFIG, "enable_pullback_long": True},
}


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def merge_signal_config(signal_config: Optional[SignalConfig]) -> SignalConfig:
    merged: SignalConfig = {
        **DEFAULT_SIGNAL_CONFIG,
        "entry_score_floor": dict(DEFAULT_SIGNAL_CONFIG["entry_score_floor"]),
        "risk_scalars": dict(DEFAULT_SIGNAL_CONFIG["risk_scalars"]),
    }
    if signal_config:
        for key, value in signal_config.items():
            if key in {"entry_score_floor", "risk_scalars"} and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["atr_pct"] = (out["ATR"] / out["close"]).replace([np.inf, -np.inf], np.nan)
    out["sma_spread"] = ((out["SMA20"] - out["SMA50"]) / out["close"]).replace([np.inf, -np.inf], np.nan)
    out["long_trend"] = ((out["close"] > out["SMA20"]) & (out["SMA20"] > out["SMA50"])).fillna(False)
    out["bull_regime"] = (out["long_trend"] & (out["ADX"] >= 18)).fillna(False)

    range_20 = (out["high"].rolling(20, min_periods=20).max() - out["low"].rolling(20, min_periods=20).min()) / out["close"]
    atr_norm = (out["ATR"] / out["close"]).replace([np.inf, -np.inf], np.nan)
    compression = (range_20 <= range_20.rolling(120, min_periods=40).quantile(0.35)).fillna(False)
    expansion = (atr_norm >= atr_norm.rolling(120, min_periods=40).quantile(0.55)).fillna(False)

    out["trend_strength"] = (
        1.8 * out["sma_spread"].clip(lower=0).fillna(0.0) +
        1.2 * (out["ADX"] / 100.0).fillna(0.0)
    )
    out["vol_compression"] = compression
    out["vol_expansion"] = expansion
    out["pullback_regime"] = (
        out["bull_regime"] &
        (out["close"] >= out["SMA50"] * 0.97) &
        (out["RSI"].between(28, 48))
    ).fillna(False)
    return out


def initialize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "signal": 0,
        "entry_score": 0.0,
        "risk_scalar": 0.0,
        "pullback_signal": 0,
        "pullback_entry_score": 0.0,
        "pullback_risk_scalar": 0.0,
    }
    for col, value in defaults.items():
        out[col] = value
    return out


def disable_unused_sleeves(df: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    out = df.copy()
    if not cfg.get("enable_trend_long", True):
        out["signal"] = 0
        out["entry_score"] = 0.0
        out["risk_scalar"] = 0.0
    if not cfg.get("enable_pullback_long", True):
        out["pullback_signal"] = 0
        out["pullback_entry_score"] = 0.0
        out["pullback_risk_scalar"] = 0.0
    return out
