from __future__ import annotations

import argparse
import copy
import inspect
from pathlib import Path
from typing import Any, Dict, List

from src.data_loader import load_data
from src.experiment_config import build_config
from src.indicators import compute_indicators
from src.portfolio import run_portfolio_backtest
from src.strategy import generate_signals
from src.validation import build_window_slices, compute_summary_stats, print_validation_report


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


def slice_market_data(market_data: Dict[str, Any], start_ts, end_ts):
    out = {}
    for symbol, df in market_data.items():
        part = df.copy()
        if start_ts is not None:
            part = part.loc[part.index >= start_ts]
        if end_ts is not None:
            part = part.loc[part.index <= end_ts]
        if len(part) > 250:
            out[symbol] = part
    return out


def run_window(config: Dict[str, Any], market_data: Dict[str, Any], label: str, start_ts, end_ts):
    if not market_data:
        return {"summary": None}
    trades, _final_equity, equity_curve = run_portfolio_backtest(
        market_data=market_data,
        **{k: v for k, v in config.items() if k not in {"symbols", "timeframe", "data_dir", "signal_config", "profile_name", "signal_profile"}}
    )
    return {"summary": compute_summary_stats(trades, equity_curve, config["starting_equity"], label, start_ts, end_ts)}


def main():
    parser = argparse.ArgumentParser(description="Institutional validation runner")
    parser.add_argument("--profile", default="full_system", choices=["trend_only", "full_system", "next_research"])
    parser.add_argument("--dev-end", default="2023-12-31")
    parser.add_argument("--val-end", default="2024-12-31")
    parser.add_argument("--holdout-end", default="2026-03-25")
    parser.add_argument("--wf-train-months", type=int, default=12)
    parser.add_argument("--wf-test-months", type=int, default=3)
    parser.add_argument("--embargo-bars", type=int, default=1)
    args = parser.parse_args()

    config = _call_build_config(copy.deepcopy(BASE_CONFIG), args.profile)
    market_data = build_market_data(config)

    import pandas as pd
    start_ts = min(df.index.min() for df in market_data.values())
    dev_end = pd.Timestamp(args.dev_end)
    val_end = pd.Timestamp(args.val_end)
    holdout_end = pd.Timestamp(args.holdout_end)
    step = pd.Timedelta(hours=4 * (max(int(args.embargo_bars), 0) + 1))
    rows: List[dict] = []

    rows.append(run_window(config, slice_market_data(market_data, start_ts, dev_end), "development", start_ts, dev_end))
    rows.append(run_window(config, slice_market_data(market_data, dev_end + step, val_end), "validation", dev_end + step, val_end))
    rows.append(run_window(config, slice_market_data(market_data, val_end + step, holdout_end), "holdout", val_end + step, holdout_end))

    wf_windows = build_window_slices(
        market_data=market_data,
        train_months=args.wf_train_months,
        test_months=args.wf_test_months,
        start_after=dev_end,
        final_end=holdout_end,
        embargo_bars=int(args.embargo_bars),
    )
    for i, window in enumerate(wf_windows, start=1):
        rows.append(run_window(config, slice_market_data(market_data, window["test_start"], window["test_end"]), f"wf_{i:02d}", window["test_start"], window["test_end"]))

    print_validation_report(rows)


if __name__ == "__main__":
    main()
