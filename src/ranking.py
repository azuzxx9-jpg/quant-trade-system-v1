from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd


def compute_symbol_score(df: pd.DataFrame) -> pd.Series:
    trend = ((df["SMA20"] - df["SMA50"]) / df["close"]).clip(lower=0)
    mom_20 = df["close"].pct_change(20).clip(lower=0)
    vol = (df["ATR"] / df["close"]).clip(lower=0)
    return 2.0 * trend.fillna(0) + 1.5 * mom_20.fillna(0) + 0.5 * vol.fillna(0)


def compute_mr_symbol_score(df: pd.DataFrame) -> pd.Series:
    atr_pct = (df["ATR"] / df["close"]).clip(lower=0)
    chop = (22.0 - df["ADX"]).clip(lower=0) / 22.0
    flat = (0.03 - ((df["SMA20"] - df["SMA50"]).abs() / df["close"])).clip(lower=0) / 0.03
    return 1.4 * chop.fillna(0) + 1.0 * flat.fillna(0) + 0.4 * atr_pct.fillna(0)


def rank_symbols(data: Dict[str, pd.DataFrame], timestamp, score_column: str = "symbol_score") -> List[Tuple[str, float]]:
    scores: List[Tuple[str, float]] = []
    for symbol, df in data.items():
        if timestamp not in df.index:
            continue
        scores.append((symbol, float(df.loc[timestamp].get(score_column, 0.0))))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
