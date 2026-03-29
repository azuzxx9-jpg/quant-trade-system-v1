import math
from collections import defaultdict

import pandas as pd


TRADE_CLOSE_TYPES = {"stop", "time_exit", "target", "final_exit"}
TRADE_ENTRY_TYPES = {"entry", "pyramid"}


def _round_or_none(value, digits=4):
    if value is None:
        return None
    try:
        if math.isnan(value):
            return None
    except TypeError:
        pass
    return round(float(value), digits)


def _build_trade_lifecycle_table(trades):
    trade_map = defaultdict(lambda: {
        "symbol": None,
        "strategy": None,
        "entry_time": None,
        "exit_time": None,
        "entry_fee": 0.0,
        "realized_pnl": 0.0,
        "initial_risk_cash": 0.0,
        "bars_in_trade": None,
    })

    for event in trades:
        trade_id = event.get("trade_id")
        if trade_id is None:
            continue

        row = trade_map[trade_id]
        row["symbol"] = event.get("symbol", row["symbol"])
        row["strategy"] = event.get("strategy", row["strategy"])

        if event["type"] in TRADE_ENTRY_TYPES:
            if row["entry_time"] is None:
                row["entry_time"] = event.get("time")
            row["entry_fee"] += float(event.get("fee", 0.0))
            risk_per_unit = event.get("risk_per_unit")
            size = event.get("size")
            if risk_per_unit is not None and size is not None:
                row["initial_risk_cash"] += float(risk_per_unit) * float(size)

        elif event["type"] in TRADE_CLOSE_TYPES:
            row["exit_time"] = event.get("time")
            row["realized_pnl"] += float(event.get("pnl", 0.0))
            row["bars_in_trade"] = int(event.get("bars_in_trade", 0))

        elif event["type"] == "partial":
            row["realized_pnl"] += float(event.get("pnl", 0.0))

    lifecycle_rows = []
    for trade_id, item in trade_map.items():
        if item["entry_time"] is None or item["exit_time"] is None:
            continue

        total_trade_pnl = item["realized_pnl"] - item["entry_fee"]
        initial_risk_cash = item["initial_risk_cash"] if item["initial_risk_cash"] > 0 else None
        trade_r_multiple = None
        if initial_risk_cash is not None:
            trade_r_multiple = total_trade_pnl / initial_risk_cash

        lifecycle_rows.append({
            "trade_id": trade_id,
            "symbol": item["symbol"],
            "strategy": item["strategy"],
            "entry_time": item["entry_time"],
            "exit_time": item["exit_time"],
            "trade_pnl": total_trade_pnl,
            "trade_r_multiple": trade_r_multiple,
            "bars_in_trade": item["bars_in_trade"],
        })

    if not lifecycle_rows:
        return pd.DataFrame()

    df = pd.DataFrame(lifecycle_rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    return df.sort_values("entry_time").reset_index(drop=True)


def analyze_trades(trades, starting_equity=10000, equity_curve_df=None):
    event_realized = [
        t for t in trades
        if t["type"] in {"partial", "stop", "time_exit", "target", "final_exit"}
    ]
    entry_events = [t for t in trades if t["type"] in TRADE_ENTRY_TYPES]

    event_pnls = [float(t["pnl"]) for t in event_realized if "pnl" in t]
    event_winners = [p for p in event_pnls if p > 0]
    event_losers = [p for p in event_pnls if p < 0]

    event_gross_profit = sum(event_winners)
    event_gross_loss = abs(sum(event_losers))
    event_profit_factor = (event_gross_profit / event_gross_loss) if event_gross_loss > 0 else None
    event_win_rate = (len(event_winners) / len(event_pnls)) if event_pnls else 0.0
    avg_event_pnl = (sum(event_pnls) / len(event_pnls)) if event_pnls else 0.0

    trade_df = _build_trade_lifecycle_table(trades)
    trade_pnls = trade_df["trade_pnl"].tolist() if not trade_df.empty else []
    trade_r_values = [x for x in trade_df["trade_r_multiple"].tolist() if pd.notna(x)] if not trade_df.empty else []

    trade_winners = [p for p in trade_pnls if p > 0]
    trade_losers = [p for p in trade_pnls if p < 0]

    trade_gross_profit = sum(trade_winners)
    trade_gross_loss = abs(sum(trade_losers))
    trade_profit_factor = (trade_gross_profit / trade_gross_loss) if trade_gross_loss > 0 else None
    trade_win_rate = (len(trade_winners) / len(trade_pnls)) if trade_pnls else 0.0
    avg_trade_pnl = (sum(trade_pnls) / len(trade_pnls)) if trade_pnls else 0.0
    avg_trade_r = (sum(trade_r_values) / len(trade_r_values)) if trade_r_values else None
    avg_bars_in_trade = trade_df["bars_in_trade"].mean() if (not trade_df.empty and "bars_in_trade" in trade_df.columns) else None

    final_equity = trades[-1]["equity"] if trades else starting_equity
    total_return_pct = ((final_equity / starting_equity) - 1.0) * 100.0
    symbols = sorted(set(t.get("symbol") for t in trades if t.get("symbol")))

    max_drawdown_pct = None
    if equity_curve_df is not None and not equity_curve_df.empty:
        running_peak = equity_curve_df["equity"].cummax()
        drawdown = (equity_curve_df["equity"] / running_peak) - 1.0
        max_drawdown_pct = float(drawdown.min() * 100.0)

    summary = {
        "starting_equity": _round_or_none(starting_equity, 2),
        "final_equity": _round_or_none(final_equity, 2),
        "total_return_pct": _round_or_none(total_return_pct, 2),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "total_entries": len(entry_events),
        "trade_lifecycles": int(len(trade_df)),
        "total_realized_events": len(event_pnls),
        "event_win_rate_pct": _round_or_none(event_win_rate * 100.0, 2),
        "trade_win_rate_pct": _round_or_none(trade_win_rate * 100.0, 2),
        "avg_event_pnl": _round_or_none(avg_event_pnl, 4),
        "avg_trade_pnl": _round_or_none(avg_trade_pnl, 4),
        "avg_trade_r": _round_or_none(avg_trade_r, 4),
        "trade_expectancy": _round_or_none(avg_trade_pnl, 4),
        "event_profit_factor": _round_or_none(event_profit_factor, 4),
        "trade_profit_factor": _round_or_none(trade_profit_factor, 4),
        "gross_profit_trade": _round_or_none(trade_gross_profit, 4),
        "gross_loss_trade": _round_or_none(trade_gross_loss, 4),
        "avg_bars_in_trade": _round_or_none(avg_bars_in_trade, 2),
        "max_drawdown_pct": _round_or_none(max_drawdown_pct, 2),
    }

    return summary


def print_performance_report(summary):
    print("\n=== PERFORMANCE REPORT ===")
    print(f"Starting equity:        {summary['starting_equity']}")
    print(f"Final equity:           {summary['final_equity']}")
    print(f"Total return (%):       {summary['total_return_pct']}")
    print(f"Symbols traded:         {summary['symbol_count']} -> {summary['symbols']}")
    print(f"Entries:                {summary['total_entries']}")
    print(f"Trade lifecycles:       {summary['trade_lifecycles']}")
    print(f"Realized events:        {summary['total_realized_events']}")
    print(f"Event win rate (%):     {summary['event_win_rate_pct']}")
    print(f"Trade win rate (%):     {summary['trade_win_rate_pct']}")
    print(f"Average event pnl:      {summary['avg_event_pnl']}")
    print(f"Average trade pnl:      {summary['avg_trade_pnl']}")
    print(f"Average trade R:        {summary['avg_trade_r']}")
    print(f"Trade expectancy:       {summary['trade_expectancy']}")
    print(f"Event profit factor:    {summary['event_profit_factor']}")
    print(f"Trade profit factor:    {summary['trade_profit_factor']}")
    print(f"Gross profit (trade):   {summary['gross_profit_trade']}")
    print(f"Gross loss (trade):     {summary['gross_loss_trade']}")
    print(f"Average bars/trade:     {summary['avg_bars_in_trade']}")
    print(f"Max drawdown (%):       {summary['max_drawdown_pct']}")
