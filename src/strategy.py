from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from src.ranking import compute_symbol_score
from src.strategies import SignalConfig, build_pullback_long, build_trend_long
from src.strategies.base import (
    compute_rsi,
    disable_unused_sleeves,
    initialize_output_columns,
    merge_signal_config,
    regime_columns,
)

__all__ = ["SignalConfig", "generate_signals"]


def generate_signals(df: pd.DataFrame, signal_config: Optional[SignalConfig] = None) -> pd.DataFrame:
    cfg: Dict[str, Any] = merge_signal_config(signal_config)
    out = df.copy()
    out["RSI"] = compute_rsi(out["close"])
    out = regime_columns(out)
    out = initialize_output_columns(out)
    out = build_trend_long(out, cfg)
    out = build_pullback_long(out, cfg)
    out = disable_unused_sleeves(out, cfg)
    out["symbol_score"] = compute_symbol_score(out)
    return out
