from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path
from typing import Any, Dict

from src.data_loader import load_data
from src.experiment_config import build_config
from src.indicators import compute_indicators
from src.metrics import analyze_trades, print_performance_report
from src.portfolio import run_portfolio_backtest
from src.reporting import (
    build_trade_dataframe,
    compute_drawdown_table,
    compute_exposure_stats,
    print_exposure_stats,
    print_table,
    summarize_by_month,
    summarize_by_strategy,
    summarize_by_symbol,
    summarize_by_year,
)
from src.strategy import generate_signals


BASE_CONFIG: Dict[str, Any] = {
    "starting_equity": 10_000,
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "AVAXUSDT"],
    "timeframe": "4h",
    "data_dir": "data",
    "risk_per_trade": 0.0105,
    "fee_rate": 0.0004,
    "entry_slippage_bps": 5.0,
    "exit_slippage_bps": 5.0,
    "cooldown_bars": 6,
    "trailing_atr_multiple": 5.0,
    "trend_time_stop_bars": 96,
    "pullback_time_stop_bars": 56,
    "max_portfolio_risk": 0.055,
    "max_positions": 8,
    "max_position_notional_fraction": 0.55,
    "max_gross_exposure": 1.40,
    "correlation_lookback": 90,
    "trend_correlation_threshold": 0.82,
    "pullback_correlation_threshold": 0.92,
    "pyramid_enabled": True,
    "pyramid_trigger_r": 2.0,
    "pyramid_risk_fraction": 0.25,
    "pyramid_size_fraction": 0.18,
    "pyramid_max_adds": 1,
    "print_last_trade_events": 20,
    "enable_symbol_ranking": True,
    "symbol_rank_top_n_long": 6,
    "enable_symbol_filtering": False,
    "strategy_notional_budget": {
        "trend_long": 0.95,
        "pullback_long": 0.45,
    },
    "strategy_risk_budget": {
        "trend_long": 0.72,
        "pullback_long": 0.28,
    },
    "strategy_max_positions": {
        "trend_long": 5,
        "pullback_long": 3,
    },
    "strategy_min_positions": {
        "trend_long": 0,
        "pullback_long": 1,
    },
    "enable_score_based_sizing": True,
    "score_sizing_floor": 0.90,
    "score_sizing_ceiling": 1.25,
    "score_sizing_center": 0.26,
    "score_sizing_width": 0.16,
    "allow_cross_sleeve_symbol_overlap": True,
    "trend_rank_bias": 0.28,
    "pullback_rank_bias": 0.22,
    "candidate_corr_penalty_trend": 0.20,
    "candidate_corr_penalty_pullback": 0.05,
    "pullback_score_floor": 0.48,
    "trend_score_floor": 0.12,
    "embargo_bars": 1,
}
CONFIG = BASE_CONFIG


def _call_build_config(base_config: Dict[str, Any], profile: str) -> Dict[str, Any]:
    sig = inspect.signature(build_config)
    params = sig.parameters
    if "base_config" in params and "profile" in params:
        return build_config(base_config=base_config, profile=profile)
    if "base_config" in params and "profile_name" in params:
        return build_config(base_config=base_config, profile_name=profile)
    if "profile" in params:
        return build_config(profile=profile)
    if "profile_name" in params:
        return build_config(profile_name=profile)
    return build_config(base_config, profile)


def prepare_symbol_dataframe(symbol: str, data_dir: str, timeframe: str, signal_config: Dict[str, Any]):
    csv_path = Path(data_dir) / f"{symbol.lower()}_{timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing data file for {symbol}: {csv_path}")
    df = load_data(str(csv_path))
    df = compute_indicators(df)
    df = generate_signals(df, signal_config=signal_config)
    return df


def build_market_data(config: Dict[str, Any]):
    market_data = {}
    print(f"\nProfile: {config['profile_name']}")
    print("Loading and preparing data...\n")
    for symbol in config["symbols"]:
        df = prepare_symbol_dataframe(symbol, config["data_dir"], config["timeframe"], config["signal_config"])
        market_data[symbol] = df
        trend_signals = int(df["signal"].sum()) if "signal" in df.columns else 0
        pullback_long = int(df["pullback_signal"].sum()) if "pullback_signal" in df.columns else 0
        print(
            f"{symbol}: rows={len(df)}, trend_signals={trend_signals}, pullback_long={pullback_long}, "
            f"start={df.index.min()}, end={df.index.max()}"
        )
    return market_data


def run_research_pipeline(config: Dict[str, Any]):
    market_data = build_market_data(config)
    print("\nRunning portfolio backtest...\n")

    trades, final_equity, equity_curve = run_portfolio_backtest(
        market_data=market_data,
        **{k: v for k, v in config.items() if k not in {"symbols", "timeframe", "data_dir", "signal_config", "profile_name", "signal_profile"}}
    )

    print(f"\nFinal portfolio equity: {final_equity:.2f}")
    print(f"Total trade events: {len(trades)}")

    n_tail = int(config.get("print_last_trade_events", 20))
    if trades and n_tail > 0:
        print(f"\nLast {min(n_tail, len(trades))} trade events:")
        for trade in trades[-n_tail:]:
            print(trade)

    summary = analyze_trades(trades=trades, starting_equity=config["starting_equity"], equity_curve_df=equity_curve)
    print_performance_report(summary)

    diagnostics = equity_curve.attrs.get("signal_pipeline_diagnostics")
    rejection_reasons = equity_curve.attrs.get("rejection_reasons")

    trades_df = build_trade_dataframe(trades)
    print_table("SIGNAL PIPELINE DIAGNOSTICS", diagnostics)
    print_table("REJECTION REASONS", rejection_reasons)
    print_table("P&L BY SYMBOL", summarize_by_symbol(trades_df))
    print_table("P&L BY STRATEGY", summarize_by_strategy(trades_df))
    print_table("MONTHLY RETURNS", summarize_by_month(equity_curve), max_rows=62)
    print_table("YEARLY RETURNS", summarize_by_year(equity_curve), max_rows=20)
    print_table("WORST DRAWDOWNS", compute_drawdown_table(equity_curve, top_n=10), max_rows=10)
    print_exposure_stats(compute_exposure_stats(equity_curve))
    return {"summary": summary, "final_equity": final_equity, "equity_curve": equity_curve, "trades": trades}


def main():
    parser = argparse.ArgumentParser(description="Run research profile for the crypto trend + pullback system.")
    parser.add_argument("--profile", default="full_system", choices=["trend_only", "full_system", "next_research"])
    args = parser.parse_args()
    config = _call_build_config(copy.deepcopy(BASE_CONFIG), args.profile)
    run_research_pipeline(config)


if __name__ == "__main__":
    main()
