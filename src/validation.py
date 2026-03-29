from __future__ import annotations
from typing import Any, Dict, List
import pandas as pd

def _max_drawdown_pct(equity_curve: pd.DataFrame) -> float:
    if equity_curve is None or equity_curve.empty or "equity" not in equity_curve.columns:
        return 0.0
    eq = equity_curve["equity"].astype(float)
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min() * 100.0)

def _profit_factor_from_trades(trades: List[dict]) -> float:
    realized = [float(t["pnl"]) for t in trades if t.get("pnl") is not None and t.get("type") in {"stop","time_exit","final_exit"}]
    wins = sum(x for x in realized if x > 0)
    losses = -sum(x for x in realized if x < 0)
    if losses <= 0:
        return 999.0 if wins > 0 else 0.0
    return float(wins / losses)

def compute_summary_stats(trades: List[dict], equity_curve: pd.DataFrame, starting_equity: float, label: str, start_ts, end_ts) -> Dict[str, Any]:
    final_equity = float(equity_curve["equity"].iloc[-1]) if equity_curve is not None and not equity_curve.empty else float(starting_equity)
    total_return_pct = ((final_equity / float(starting_equity)) - 1.0) * 100.0 if starting_equity else 0.0
    realized = [t for t in trades if t.get("type") in {"stop","time_exit","final_exit"}]
    avg_open_positions = float(equity_curve["open_positions"].mean()) if equity_curve is not None and not equity_curve.empty and "open_positions" in equity_curve.columns else 0.0
    return {"window": label, "start": str(start_ts), "end": str(end_ts), "final_equity": round(final_equity,2), "return_pct": round(total_return_pct,2), "max_dd_pct": round(_max_drawdown_pct(equity_curve),2), "profit_factor": round(_profit_factor_from_trades(trades),3), "realized_trades": int(len(realized)), "avg_open_positions": round(avg_open_positions,3)}

def build_window_slices(market_data: Dict[str, pd.DataFrame], train_months: int, test_months: int, start_after, final_end, embargo_bars: int = 1) -> List[Dict[str, Any]]:
    idx = pd.DatetimeIndex(sorted(set().union(*[set(df.index) for df in market_data.values()])))
    start_after_ts = pd.Timestamp(start_after)
    final_end_ts = pd.Timestamp(final_end)
    bar_delta = pd.Timedelta(hours=4 * max(embargo_bars + 1, 1))
    test_starts = pd.date_range(start=start_after_ts + bar_delta, end=final_end_ts, freq=f"{test_months}MS")
    windows: List[Dict[str, Any]] = []
    for test_start in test_starts:
        train_end = test_start - bar_delta
        train_start = train_end - pd.DateOffset(months=train_months)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(seconds=1)
        if test_end > final_end_ts:
            test_end = final_end_ts
        if train_start < idx.min() or test_start >= test_end:
            continue
        windows.append({"train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end})
    return windows

def print_validation_report(rows: List[dict]) -> None:
    summaries = [r["summary"] for r in rows if r.get("summary") is not None]
    if not summaries:
        print("No validation rows produced."); return
    df = pd.DataFrame(summaries)
    print("\n=== INSTITUTIONAL VALIDATION REPORT ===")
    print(df.to_string(index=False))
    wf = df[df["window"].str.startswith("wf_")].copy()
    if not wf.empty:
        print("\n=== WALK-FORWARD AGGREGATES ===")
        print(f"windows={len(wf)}, avg_return_pct={wf['return_pct'].mean():.2f}, median_return_pct={wf['return_pct'].median():.2f}, avg_max_dd_pct={wf['max_dd_pct'].mean():.2f}, avg_pf={wf['profit_factor'].mean():.3f}")
