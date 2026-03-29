from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def build_pullback_long(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    dist_sma20 = ((out["close"] - out["SMA20"]) / out["close"]).replace([np.inf, -np.inf], np.nan)
    dist_sma50 = ((out["close"] - out["SMA50"]) / out["close"]).replace([np.inf, -np.inf], np.nan)
    deep_pullback = (dist_sma20 <= -0.018) & (dist_sma50 >= -0.035)
    rebound = (out["close"] > out["high"].shift(1)) | ((out["close"] > out["open"]) & (out["close"] > out["close"].shift(1)))
    strong_trend_context = out["bull_regime"] & (out["ADX"] >= 22)
    reset_rsi = out["RSI"].between(30, 44)

    signal = (
        strong_trend_context &
        out["pullback_regime"] &
        deep_pullback.fillna(False) &
        reset_rsi.fillna(False) &
        rebound.fillna(False)
    )

    out["pullback_signal"] = signal.astype(int)
    out["pullback_entry_score"] = (
        1.0 * out["trend_strength"].clip(0, 1.2).fillna(0.0) +
        0.8 * ((45 - out["RSI"]).clip(0, 20) / 20.0).fillna(0.0) +
        0.9 * (-dist_sma20).clip(0, 0.05).fillna(0.0) / 0.05 +
        0.3 * (out["ATR"] / out["close"]).clip(0, 0.05).fillna(0.0) / 0.05
    )
    out["pullback_risk_scalar"] = (0.72 + 0.20 * out["pullback_entry_score"].clip(0, 1)).fillna(0.72)
    return out
