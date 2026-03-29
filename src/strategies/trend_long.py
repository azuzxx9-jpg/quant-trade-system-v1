from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def build_trend_long(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    breakout = out["close"] > out["high"].rolling(20, min_periods=20).max().shift(1)
    momentum_20 = (out["close"] / out["close"].shift(20) - 1.0).clip(lower=0)
    momentum_60 = (out["close"] / out["close"].shift(60) - 1.0).clip(lower=0)

    long_signal = (
        out["bull_regime"] &
        out["vol_expansion"] &
        breakout.fillna(False) &
        (momentum_20 > 0.03).fillna(False) &
        (momentum_60 > 0.05).fillna(False)
    )

    out["signal"] = long_signal.astype(int)
    out["entry_score"] = (
        1.2 * out["trend_strength"].clip(lower=0).fillna(0.0) +
        0.9 * momentum_20.fillna(0.0) +
        0.7 * momentum_60.fillna(0.0) +
        0.4 * out["vol_expansion"].astype(int)
    )
    out["risk_scalar"] = float(cfg.get("risk_scalars", {}).get("trend_long", 1.10))
    return out
