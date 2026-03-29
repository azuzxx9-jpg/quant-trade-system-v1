from __future__ import annotations

from typing import Dict, Set

import pandas as pd


def _safe_profit_factor(pnl_series: pd.Series) -> float:
    wins = pnl_series[pnl_series > 0].sum()
    losses = -pnl_series[pnl_series < 0].sum()
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)


def build_symbol_trade_stats(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["symbol", "trades", "net_pnl", "profit_factor", "avg_trade_pnl"])

    df = trades_df.copy()
    if "pnl" not in df.columns:
        return pd.DataFrame(columns=["symbol", "trades", "net_pnl", "profit_factor", "avg_trade_pnl"])

    realized_types = {"stop", "time_exit", "final_exit"}
    if "type" in df.columns:
        df = df[df["type"].isin(realized_types)]

    if df.empty:
        return pd.DataFrame(columns=["symbol", "trades", "net_pnl", "profit_factor", "avg_trade_pnl"])

    rows = []
    for symbol, g in df.groupby("symbol"):
        pnl = g["pnl"].fillna(0.0)
        rows.append(
            {
                "symbol": symbol,
                "trades": int(len(g)),
                "net_pnl": float(pnl.sum()),
                "profit_factor": _safe_profit_factor(pnl),
                "avg_trade_pnl": float(pnl.mean()) if len(g) else 0.0,
            }
        )

    out = pd.DataFrame(rows).sort_values(["net_pnl", "profit_factor"], ascending=False)
    return out.reset_index(drop=True)


def eligible_symbols_from_history(
    trades_df: pd.DataFrame,
    min_trades: int = 20,
    min_profit_factor: float = 1.05,
    min_net_pnl: float = 0.0,
) -> Set[str]:
    stats = build_symbol_trade_stats(trades_df)
    if stats.empty:
        return set()

    eligible = stats[
        (stats["trades"] >= int(min_trades))
        & (stats["profit_factor"] >= float(min_profit_factor))
        & (stats["net_pnl"] > float(min_net_pnl))
    ]
    return set(eligible["symbol"].tolist())


def filter_market_data_by_symbols(market_data: Dict[str, pd.DataFrame], eligible_symbols: Set[str]) -> Dict[str, pd.DataFrame]:
    if not eligible_symbols:
        return market_data
    return {symbol: df for symbol, df in market_data.items() if symbol in eligible_symbols}
