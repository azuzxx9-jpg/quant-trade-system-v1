import pandas as pd

TRADE_CLOSE_TYPES = {"stop", "time_exit", "target", "final_exit"}
TRADE_ENTRY_TYPES = {"entry", "pyramid"}

def build_trade_dataframe(trades):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades).copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    if "pnl" not in df.columns:
        df["pnl"] = 0.0
    if "symbol" not in df.columns:
        df["symbol"] = "UNKNOWN"
    if "strategy" not in df.columns:
        df["strategy"] = "UNKNOWN"
    return df

def _build_trade_lifecycle_dataframe(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    entries = trades_df[trades_df["type"].isin(TRADE_ENTRY_TYPES)].copy()
    closes = trades_df[trades_df["type"].isin(TRADE_CLOSE_TYPES)].copy()
    partials = trades_df[trades_df["type"] == "partial"].copy()
    if entries.empty or closes.empty:
        return pd.DataFrame()
    entry_agg = entries.groupby("trade_id").agg(symbol=("symbol","last"), strategy=("strategy","last"), entry_score=("entry_score","first"), entry_fee=("fee","sum")).to_dict("index")
    partial_map = partials.groupby("trade_id")["pnl"].sum().to_dict()
    rows = []
    for _, close_row in closes.iterrows():
        trade_id = close_row["trade_id"]
        if trade_id not in entry_agg:
            continue
        entry_row = entry_agg[trade_id]
        total_trade_pnl = float(partial_map.get(trade_id, 0.0)) + float(close_row.get("pnl", 0.0)) - float(entry_row.get("entry_fee", 0.0))
        rows.append({"trade_id": trade_id, "symbol": close_row.get("symbol", entry_row.get("symbol","UNKNOWN")), "strategy": close_row.get("strategy", entry_row.get("strategy","UNKNOWN")), "trade_pnl": total_trade_pnl, "entry_score": entry_row.get("entry_score"), "bars_in_trade": close_row.get("bars_in_trade")})
    return pd.DataFrame(rows)

def summarize_by_symbol(trades_df):
    lifecycle = _build_trade_lifecycle_dataframe(trades_df)
    if lifecycle.empty:
        return pd.DataFrame()
    rows = []
    for symbol, group in lifecycle.groupby("symbol"):
        pnls = group["trade_pnl"].tolist()
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gp = sum(wins); gl = abs(sum(losses)); pf = gp / gl if gl > 0 else None
        rows.append({"symbol": symbol, "trades": len(pnls), "win_rate_pct": round((len(wins)/len(pnls))*100,2) if pnls else 0.0, "gross_profit": round(gp,2), "gross_loss": round(gl,2), "net_pnl": round(sum(pnls),2), "avg_trade_pnl": round(sum(pnls)/len(pnls),2) if pnls else 0.0, "avg_entry_score": round(float(group["entry_score"].dropna().mean()),3) if group["entry_score"].notna().any() else None, "avg_bars_in_trade": round(float(group["bars_in_trade"].dropna().mean()),2) if group["bars_in_trade"].notna().any() else None, "profit_factor": round(pf,3) if pf is not None else None})
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)

def summarize_by_strategy(trades_df):
    lifecycle = _build_trade_lifecycle_dataframe(trades_df)
    if lifecycle.empty:
        return pd.DataFrame()
    rows = []
    for strategy, group in lifecycle.groupby("strategy"):
        pnls = group["trade_pnl"].tolist()
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gp = sum(wins); gl = abs(sum(losses)); pf = gp / gl if gl > 0 else None
        rows.append({"strategy": strategy, "trades": len(pnls), "win_rate_pct": round((len(wins)/len(pnls))*100,2) if pnls else 0.0, "gross_profit": round(gp,2), "gross_loss": round(gl,2), "net_pnl": round(sum(pnls),2), "avg_trade_pnl": round(sum(pnls)/len(pnls),2) if pnls else 0.0, "avg_entry_score": round(float(group["entry_score"].dropna().mean()),3) if group["entry_score"].notna().any() else None, "avg_bars_in_trade": round(float(group["bars_in_trade"].dropna().mean()),2) if group["bars_in_trade"].notna().any() else None, "profit_factor": round(pf,3) if pf is not None else None})
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)

def summarize_by_month(equity_curve_df):
    if equity_curve_df.empty:
        return pd.DataFrame()
    df = equity_curve_df.copy(); df.attrs = {}
    monthly_equity = df["equity"].resample("ME").last().dropna()
    monthly_returns = monthly_equity.pct_change().dropna() * 100.0
    out = monthly_returns.to_frame(name="return_pct")
    out["year"] = out.index.year; out["month"] = out.index.month; out["month_name"] = out.index.strftime("%b"); out["return_pct"] = out["return_pct"].round(2)
    return out.reset_index().rename(columns={"time":"month_end"})

def summarize_by_year(equity_curve_df):
    if equity_curve_df.empty:
        return pd.DataFrame()
    df = equity_curve_df.copy(); df.attrs = {}; df = df.sort_index()
    yearly = []
    for year in sorted(df.index.year.unique()):
        year_df = df[df.index.year == year]
        if year_df.empty: continue
        start_equity = year_df["equity"].iloc[0]; end_equity = year_df["equity"].iloc[-1]
        ret = ((end_equity / start_equity) - 1.0) * 100.0
        yearly.append({"year":year, "return_pct": round(ret,2), "start_equity": round(start_equity,2), "end_equity": round(end_equity,2)})
    return pd.DataFrame(yearly)

def compute_drawdown_table(equity_curve_df, top_n=10):
    if equity_curve_df.empty:
        return pd.DataFrame()
    df = equity_curve_df.copy(); df.attrs = {}
    df["running_peak"] = df["equity"].cummax()
    df["drawdown_pct"] = ((df["equity"] / df["running_peak"]) - 1.0) * 100.0
    cols = ["equity", "running_peak", "drawdown_pct"]
    for optional in ["open_positions", "open_risk", "risk_scalar"]:
        if optional in df.columns:
            cols.append(optional)
    worst = df[["drawdown_pct"] + [c for c in cols if c != "drawdown_pct"]].copy()
    worst.attrs = {}
    worst = worst.nsmallest(top_n, "drawdown_pct").reset_index()
    numeric_cols = [c for c in cols if c != "time"]
    worst[numeric_cols] = worst[numeric_cols].round(4)
    return worst[["time", *cols]]

def compute_exposure_stats(equity_curve_df):
    if equity_curve_df.empty:
        return {}
    open_positions = equity_curve_df["open_positions"]
    stats = {"avg_open_positions": round(float(open_positions.mean()),3), "max_open_positions": int(open_positions.max()), "pct_time_with_positions": round(float((open_positions > 0).mean() * 100.0),2), "pct_time_fully_allocated": round(float((open_positions == open_positions.max()).mean() * 100.0),2)}
    if "open_risk" in equity_curve_df.columns:
        stats["avg_open_risk"] = round(float(equity_curve_df["open_risk"].mean()),2); stats["max_open_risk"] = round(float(equity_curve_df["open_risk"].max()),2)
    if "drawdown_pct" in equity_curve_df.columns:
        stats["worst_drawdown_pct"] = round(float(equity_curve_df["drawdown_pct"].min()),2)
    if "risk_scalar" in equity_curve_df.columns:
        stats["avg_risk_scalar"] = round(float(equity_curve_df["risk_scalar"].mean()),3)
    return stats

def print_table(title, df, max_rows=20):
    print(f"\n=== {title} ===")
    if df is None or getattr(df, "empty", False):
        print("No data."); return
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False)); print(f"... ({len(df)} rows total)")
    else:
        print(df.to_string(index=False))

def print_exposure_stats(stats):
    print("\n=== EXPOSURE STATS ===")
    if not stats:
        print("No data."); return
    for key, value in stats.items():
        print(f"{key}: {value}")
