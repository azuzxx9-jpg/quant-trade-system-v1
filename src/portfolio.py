from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _slip_frac(bps: float) -> float:
    return float(bps) / 10000.0

def _apply_entry(price: float, side: str, bps: float) -> float:
    f = _slip_frac(bps)
    return price * (1.0 + f) if side == "long" else price * (1.0 - f)

def _apply_exit(price: float, side: str, bps: float) -> float:
    f = _slip_frac(bps)
    return price * (1.0 - f) if side == "long" else price * (1.0 + f)

def _position_unrealized_pnl(position, mark_price: float) -> float:
    return (mark_price - position["entry_price"]) * position["size"]

def _realized_pnl(position, exit_price: float, size_closed: float, exit_fee: float) -> float:
    return (exit_price - position["entry_price"]) * size_closed - exit_fee

def _mark_to_market_equity(cash: float, positions: Dict[Tuple[str, str, int], dict], market_data: Dict[str, pd.DataFrame], timestamp) -> float:
    equity = cash
    for position in positions.values():
        df = market_data[position["symbol"]]
        if timestamp in df.index:
            equity += _position_unrealized_pnl(position, float(df.loc[timestamp, "close"]))
    return equity

def _current_open_risk(positions):
    return sum(abs(p["entry_price"] - p["stop_price"]) * p["size"] for p in positions.values())

def _current_gross_notional(positions, market_data, timestamp):
    gross = 0.0
    for position in positions.values():
        df = market_data[position["symbol"]]
        if timestamp in df.index:
            gross += position["size"] * float(df.loc[timestamp, "close"])
    return gross

def _strategy_open_positions(strategy_name, positions):
    return sum(1 for p in positions.values() if p["strategy"] == strategy_name)

def _strategy_open_notional(strategy_name, positions, market_data, timestamp):
    total = 0.0
    for p in positions.values():
        if p["strategy"] != strategy_name:
            continue
        df = market_data[p["symbol"]]
        if timestamp in df.index:
            total += p["size"] * float(df.loc[timestamp, "close"])
    return total

def _strategy_open_risk(strategy_name, positions):
    return sum(abs(p["entry_price"] - p["stop_price"]) * p["size"] for p in positions.values() if p["strategy"] == strategy_name)

def _risk_scalar_from_drawdown(current_equity, peak_equity):
    if peak_equity <= 0:
        return 1.0
    dd = 1.0 - (current_equity / peak_equity)
    if dd >= 0.18:
        return 0.50
    if dd >= 0.12:
        return 0.75
    return 1.0

def _symbol_scalar(symbol: str) -> float:
    return 0.45 if symbol in {"BTCUSDT", "ETHUSDT"} else 1.0

def _score_size_multiplier(score, floor, ceiling, center, width):
    if width <= 0:
        return 1.0
    z = (float(score) - float(center)) / float(width)
    sigmoid = 1.0 / (1.0 + math.exp(-z))
    return float(floor) + (float(ceiling) - float(floor)) * sigmoid

def _estimate_average_correlation(symbol, positions, market_data, timestamp, lookback):
    if not positions:
        return 0.0
    candidate_df = market_data[symbol]
    if timestamp not in candidate_df.index:
        return 0.0
    candidate_idx = candidate_df.index.get_loc(timestamp)
    if isinstance(candidate_idx, slice) or candidate_idx < lookback:
        return 0.0
    candidate_returns = candidate_df["close"].pct_change().iloc[candidate_idx - lookback:candidate_idx]
    corrs = []
    for p in positions.values():
        open_df = market_data[p["symbol"]]
        if timestamp not in open_df.index:
            continue
        open_idx = open_df.index.get_loc(timestamp)
        if isinstance(open_idx, slice) or open_idx < lookback:
            continue
        open_returns = open_df["close"].pct_change().iloc[open_idx - lookback:open_idx]
        joined = pd.concat([candidate_returns, open_returns], axis=1).dropna()
        if len(joined) < max(lookback // 3, 20):
            continue
        corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
        if pd.notna(corr):
            corrs.append(float(corr))
    return float(np.mean(corrs)) if corrs else 0.0

def _candidate_rank(candidate, trend_rank_bias, pullback_rank_bias, corr_penalty_trend, corr_penalty_pullback):
    base = float(candidate["score"])
    corr = max(float(candidate["avg_book_corr"]), 0.0)
    if candidate["strategy"] == "trend_long":
        return base + trend_rank_bias + 0.25 * float(candidate.get("symbol_score", 0.0)) - corr_penalty_trend * corr
    return base + pullback_rank_bias + 0.20 * float(candidate.get("symbol_score", 0.0)) - corr_penalty_pullback * corr

def _log_trade_event(trades, trade_id, symbol, strategy, event_type, timestamp, price, size, pnl, fee, equity, bars_in_trade, extra=None):
    payload = {"trade_id": int(trade_id), "symbol": symbol, "strategy": strategy, "side": "long", "type": event_type, "time": str(timestamp), "price": float(price), "size": float(size), "fee": float(fee), "equity": float(equity), "bars_in_trade": int(bars_in_trade)}
    if pnl is not None:
        payload["pnl"] = float(pnl)
    if extra:
        payload.update(extra)
    trades.append(payload)

def run_portfolio_backtest(
    market_data,
    starting_equity=10000,
    risk_per_trade=0.0105,
    fee_rate=0.0004,
    entry_slippage_bps=5.0,
    exit_slippage_bps=5.0,
    cooldown_bars=6,
    trailing_atr_multiple=5.0,
    trend_time_stop_bars=96,
    pullback_time_stop_bars=56,
    max_portfolio_risk=0.055,
    max_positions=8,
    max_position_notional_fraction=0.55,
    max_gross_exposure=1.40,
    correlation_lookback=90,
    trend_correlation_threshold=0.82,
    pullback_correlation_threshold=0.92,
    pyramid_enabled=True,
    pyramid_trigger_r=2.0,
    pyramid_risk_fraction=0.25,
    pyramid_size_fraction=0.18,
    pyramid_max_adds=1,
    enable_symbol_ranking=True,
    symbol_rank_top_n_long=6,
    enable_symbol_filtering=False,
    strategy_notional_budget=None,
    strategy_risk_budget=None,
    strategy_max_positions=None,
    strategy_min_positions=None,
    enable_score_based_sizing=True,
    score_sizing_floor=0.90,
    score_sizing_ceiling=1.25,
    score_sizing_center=0.26,
    score_sizing_width=0.16,
    allow_cross_sleeve_symbol_overlap=True,
    trend_rank_bias=0.28,
    pullback_rank_bias=0.22,
    candidate_corr_penalty_trend=0.20,
    candidate_corr_penalty_pullback=0.05,
    pullback_score_floor=0.48,
    trend_score_floor=0.12,
    **kwargs,
):
    strategy_notional_budget = strategy_notional_budget or {"trend_long": 0.95, "pullback_long": 0.45}
    strategy_risk_budget = strategy_risk_budget or {"trend_long": 0.72, "pullback_long": 0.28}
    strategy_max_positions = strategy_max_positions or {"trend_long": 5, "pullback_long": 3}
    strategy_min_positions = strategy_min_positions or {"trend_long": 0, "pullback_long": 1}
    sleeves_order = ["pullback_long", "trend_long"]

    cash = float(starting_equity)
    positions = {}
    trades = []
    cooldown_until = defaultdict(lambda: -1)
    equity_curve = []
    trade_id_counter = 0
    peak_equity = float(starting_equity)
    rejection_counts = defaultdict(int)
    raw_signal_counts = defaultdict(int)
    entry_counts = defaultdict(int)

    all_timestamps = sorted(set().union(*[set(df.index) for df in market_data.values()]))
    index_maps = {symbol: {ts: i for i, ts in enumerate(df.index)} for symbol, df in market_data.items()}

    for timestamp in all_timestamps:
        trend_ranked = []
        if enable_symbol_ranking:
            for symbol, df in market_data.items():
                if timestamp in df.index:
                    trend_ranked.append((symbol, float(df.loc[timestamp].get("symbol_score", 0.0))))
            trend_ranked.sort(key=lambda x: x[1], reverse=True)
        rank_allowed = set(market_data.keys()) if not enable_symbol_ranking else {s for s, _ in trend_ranked[:max(int(symbol_rank_top_n_long), 0)]}

        for key in list(positions.keys()):
            position = positions[key]
            df = market_data[position["symbol"]]
            if timestamp not in df.index:
                continue
            row = df.loc[timestamp]
            high, low, close, open_px = float(row["high"]), float(row["low"]), float(row["close"]), float(row["open"])
            atr = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
            position["bars_in_trade"] += 1
            position["max_price"] = max(position["max_price"], high)
            stop_hit = low <= position["stop_price"]

            if stop_hit:
                raw_stop = position["stop_price"]
                if open_px < raw_stop:
                    raw_stop = open_px
                exit_price = _apply_exit(raw_stop, "long", exit_slippage_bps)
                size_closed = position["size"]
                exit_fee = size_closed * exit_price * fee_rate
                pnl = _realized_pnl(position, exit_price, size_closed, exit_fee)
                cash += pnl
                _log_trade_event(trades, position["trade_id"], position["symbol"], position["strategy"], "stop", timestamp, exit_price, size_closed, pnl, exit_fee, cash, position["bars_in_trade"], extra={"entry_score": position["entry_score"]})
                del positions[key]
                cooldown_until[(position["symbol"], position["strategy"])] = index_maps[position["symbol"]][timestamp] + cooldown_bars
                continue

            if position["strategy"] == "trend_long":
                if not position["partial_taken"]:
                    target_price = position["base_entry_price"] + 2.2 * position["base_initial_risk"]
                    if high >= target_price:
                        size_closed = position["size"] * 0.5
                        exit_price = _apply_exit(target_price, "long", exit_slippage_bps)
                        exit_fee = size_closed * exit_price * fee_rate
                        pnl = _realized_pnl(position, exit_price, size_closed, exit_fee)
                        cash += pnl
                        position["size"] -= size_closed
                        position["partial_taken"] = True
                        position["stop_price"] = max(position["stop_price"], position["base_entry_price"])
                        _log_trade_event(trades, position["trade_id"], position["symbol"], position["strategy"], "partial", timestamp, exit_price, size_closed, pnl, exit_fee, cash, position["bars_in_trade"], extra={"entry_score": position["entry_score"]})
                if pd.notna(atr):
                    position["stop_price"] = max(position["stop_price"], position["max_price"] - position["trailing_atr_multiple"] * atr)
                if pyramid_enabled and position["partial_taken"] and position["pyramids_added"] < pyramid_max_adds and position["stop_price"] >= position["base_entry_price"]:
                    close_ready = close >= (position["base_entry_price"] + pyramid_trigger_r * position["base_initial_risk"])
                    if close_ready:
                        add_entry_price = _apply_entry(close, "long", entry_slippage_bps)
                        add_risk_per_unit = max(add_entry_price - position["stop_price"], 0.0)
                        if add_risk_per_unit > 0:
                            current_equity = _mark_to_market_equity(cash, positions, market_data, timestamp)
                            current_open_risk = _current_open_risk(positions)
                            current_gross_notional = _current_gross_notional(positions, market_data, timestamp)
                            add_size = min((current_equity * risk_per_trade * pyramid_risk_fraction) / add_risk_per_unit, position["initial_size"] * pyramid_size_fraction)
                            add_notional = add_size * add_entry_price
                            add_risk_cash = add_size * add_risk_per_unit
                            if add_size > 0 and current_open_risk + add_risk_cash <= current_equity * max_portfolio_risk and current_gross_notional + add_notional <= current_equity * max_gross_exposure:
                                add_fee = add_notional * fee_rate
                                cash -= add_fee
                                old_size = position["size"]
                                new_size = old_size + add_size
                                position["entry_price"] = ((position["entry_price"] * old_size) + (add_entry_price * add_size)) / new_size
                                position["size"] = new_size
                                position["pyramids_added"] += 1
                                position["max_price"] = max(position["max_price"], add_entry_price)
                                _log_trade_event(trades, position["trade_id"], position["symbol"], position["strategy"], "pyramid", timestamp, add_entry_price, add_size, None, add_fee, cash, position["bars_in_trade"], extra={"entry_score": position["entry_score"]})
                if position["bars_in_trade"] > position["time_stop_bars"]:
                    exit_price = _apply_exit(close, "long", exit_slippage_bps)
                    size_closed = position["size"]
                    exit_fee = size_closed * exit_price * fee_rate
                    pnl = _realized_pnl(position, exit_price, size_closed, exit_fee)
                    cash += pnl
                    _log_trade_event(trades, position["trade_id"], position["symbol"], position["strategy"], "time_exit", timestamp, exit_price, size_closed, pnl, exit_fee, cash, position["bars_in_trade"], extra={"entry_score": position["entry_score"]})
                    del positions[key]
                    cooldown_until[(position["symbol"], position["strategy"])] = index_maps[position["symbol"]][timestamp] + cooldown_bars
                    continue
            else:
                target_hit = high >= position["target_price"]
                if target_hit or position["bars_in_trade"] > position["time_stop_bars"] or row.get("RSI", 50) >= 62:
                    exit_price = _apply_exit(position["target_price"] if target_hit else close, "long", exit_slippage_bps)
                    size_closed = position["size"]
                    exit_fee = size_closed * exit_price * fee_rate
                    pnl = _realized_pnl(position, exit_price, size_closed, exit_fee)
                    cash += pnl
                    _log_trade_event(trades, position["trade_id"], position["symbol"], position["strategy"], "time_exit", timestamp, exit_price, size_closed, pnl, exit_fee, cash, position["bars_in_trade"], extra={"entry_score": position["entry_score"]})
                    del positions[key]
                    cooldown_until[(position["symbol"], position["strategy"])] = index_maps[position["symbol"]][timestamp] + cooldown_bars
                    continue

        portfolio_equity = _mark_to_market_equity(cash, positions, market_data, timestamp)
        peak_equity = max(peak_equity, portfolio_equity)
        equity_curve.append({"time": timestamp, "equity": portfolio_equity, "cash": cash, "open_positions": len(positions), "open_risk": _current_open_risk(positions), "drawdown_pct": ((portfolio_equity / peak_equity) - 1.0) * 100.0 if peak_equity > 0 else 0.0, "risk_scalar": _risk_scalar_from_drawdown(portfolio_equity, peak_equity)})

        candidates_by_sleeve = {s: [] for s in sleeves_order}
        for symbol, df in market_data.items():
            if timestamp not in df.index:
                continue
            i = index_maps[symbol][timestamp]
            if i <= 200 or i >= len(df) - 1:
                continue
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            avg_corr = _estimate_average_correlation(symbol, positions, market_data, timestamp, correlation_lookback)
            symbol_score = float(row.get("symbol_score", 0.0))
            trend_strength = float(row.get("trend_strength", 0.0))

            if int(row.get("signal", 0)) == 1:
                raw_signal_counts["trend_long"] += 1
                if symbol not in rank_allowed:
                    rejection_counts[("trend_long", "ranking_or_filter")] += 1
                elif float(row.get("entry_score", 0.0)) < float(trend_score_floor):
                    rejection_counts[("trend_long", "score_floor")] += 1
                else:
                    candidates_by_sleeve["trend_long"].append({"symbol": symbol, "strategy": "trend_long", "row": row, "next_row": next_row, "timestamp": df.index[i + 1], "score": float(row.get("entry_score", 0.0)), "avg_book_corr": avg_corr, "risk_scalar": float(row.get("risk_scalar", 1.1)), "symbol_score": symbol_score, "trend_strength": trend_strength})

            if int(row.get("pullback_signal", 0)) == 1:
                raw_signal_counts["pullback_long"] += 1
                if float(row.get("pullback_entry_score", 0.0)) < float(pullback_score_floor):
                    rejection_counts[("pullback_long", "score_floor")] += 1
                else:
                    candidates_by_sleeve["pullback_long"].append({"symbol": symbol, "strategy": "pullback_long", "row": row, "next_row": next_row, "timestamp": df.index[i + 1], "score": float(row.get("pullback_entry_score", 0.0)), "avg_book_corr": avg_corr, "risk_scalar": float(row.get("pullback_risk_scalar", 0.88)), "symbol_score": symbol_score, "trend_strength": trend_strength})

        for sleeve in sleeves_order:
            candidates = sorted(candidates_by_sleeve[sleeve], key=lambda c: _candidate_rank(c, trend_rank_bias, pullback_rank_bias, candidate_corr_penalty_trend, candidate_corr_penalty_pullback), reverse=True)
            if not candidates:
                continue
            current_equity = _mark_to_market_equity(cash, positions, market_data, timestamp)
            current_open_risk = _current_open_risk(positions)
            current_gross_notional = _current_gross_notional(positions, market_data, timestamp)
            sleeve_open_notional = _strategy_open_notional(sleeve, positions, market_data, timestamp)
            sleeve_open_risk = _strategy_open_risk(sleeve, positions)
            sleeve_position_count = _strategy_open_positions(sleeve, positions)
            sleeve_max_positions = int(strategy_max_positions.get(sleeve, 0))
            sleeve_min_positions = int(strategy_min_positions.get(sleeve, 0))
            sleeve_risk_budget_cash = current_equity * max_portfolio_risk * float(strategy_risk_budget.get(sleeve, 0.0))
            sleeve_notional_budget_cash = current_equity * float(strategy_notional_budget.get(sleeve, 0.0))
            must_deploy = sleeve_position_count < sleeve_min_positions

            for candidate in candidates:
                if len(positions) >= max_positions:
                    rejection_counts[(sleeve, "max_positions")] += 1
                    continue
                same_sleeve_symbol = any(p["symbol"] == candidate["symbol"] and p["strategy"] == sleeve for p in positions.values())
                if same_sleeve_symbol:
                    rejection_counts[(sleeve, "existing_same_sleeve_symbol")] += 1
                    continue
                if not allow_cross_sleeve_symbol_overlap and any(p["symbol"] == candidate["symbol"] for p in positions.values()):
                    rejection_counts[(sleeve, "existing_symbol")] += 1
                    continue
                if sleeve_position_count >= sleeve_max_positions:
                    rejection_counts[(sleeve, "strategy_position_cap")] += 1
                    continue
                if index_maps[candidate["symbol"]][timestamp] <= cooldown_until[(candidate["symbol"], sleeve)]:
                    rejection_counts[(sleeve, "cooldown")] += 1
                    continue

                corr_threshold = pullback_correlation_threshold if sleeve == "pullback_long" else trend_correlation_threshold
                if candidate["avg_book_corr"] > (0.97 if must_deploy and sleeve == "pullback_long" else corr_threshold):
                    rejection_counts[(sleeve, "correlation_gate")] += 1
                    continue

                row = candidate["row"]
                entry_price = _apply_entry(float(candidate["next_row"]["open"]), "long", entry_slippage_bps)
                atr = float(row["ATR"])
                if sleeve == "trend_long":
                    stop_price = min(float(row["SMA20"]) - 1.1 * atr, float(row["low"]))
                    risk_per_unit = entry_price - stop_price
                    target_price = math.nan
                    local_time_stop = int(trend_time_stop_bars)
                    trailing_mult = float(trailing_atr_multiple)
                else:
                    stop_price = min(float(row["SMA50"]) - 0.8 * atr, entry_price - 1.0 * atr)
                    risk_per_unit = entry_price - stop_price
                    target_price = entry_price + 1.8 * atr
                    local_time_stop = int(pullback_time_stop_bars)
                    trailing_mult = 0.0

                if pd.isna(risk_per_unit) or risk_per_unit <= 0 or (risk_per_unit / entry_price) < 0.0015:
                    rejection_counts[(sleeve, "invalid_risk")] += 1
                    continue

                drawdown_scalar = _risk_scalar_from_drawdown(current_equity, peak_equity)
                score_mult = _score_size_multiplier(candidate["score"], score_sizing_floor, score_sizing_ceiling, score_sizing_center, score_sizing_width) if enable_score_based_sizing else 1.0
                effective_risk_fraction = risk_per_trade * float(candidate["risk_scalar"]) * drawdown_scalar * _symbol_scalar(candidate["symbol"]) * score_mult
                proposed_size = min((current_equity * effective_risk_fraction) / risk_per_unit, (current_equity * max_position_notional_fraction) / entry_price)
                proposed_notional = proposed_size * entry_price
                proposed_risk_cash = proposed_size * risk_per_unit

                if proposed_size <= 0:
                    rejection_counts[(sleeve, "size_zero")] += 1
                    continue
                if sleeve_open_notional + proposed_notional > sleeve_notional_budget_cash:
                    rejection_counts[(sleeve, "strategy_notional_budget")] += 1
                    continue
                if sleeve_open_risk + proposed_risk_cash > sleeve_risk_budget_cash:
                    rejection_counts[(sleeve, "strategy_risk_budget")] += 1
                    continue
                if current_open_risk + proposed_risk_cash > current_equity * max_portfolio_risk:
                    rejection_counts[(sleeve, "portfolio_risk_budget")] += 1
                    continue
                if current_gross_notional + proposed_notional > current_equity * max_gross_exposure:
                    rejection_counts[(sleeve, "gross_exposure")] += 1
                    continue

                entry_fee = proposed_notional * fee_rate
                cash -= entry_fee
                trade_id_counter += 1
                key = (candidate["symbol"], sleeve, trade_id_counter)
                positions[key] = {"trade_id": trade_id_counter, "symbol": candidate["symbol"], "strategy": sleeve, "entry_time": candidate["timestamp"], "entry_price": entry_price, "base_entry_price": entry_price, "stop_price": stop_price, "target_price": target_price, "size": proposed_size, "initial_size": proposed_size, "base_initial_risk": risk_per_unit, "max_price": entry_price, "partial_taken": False, "bars_in_trade": 0, "time_stop_bars": local_time_stop, "trailing_atr_multiple": trailing_mult, "entry_score": candidate["score"], "avg_book_corr": candidate["avg_book_corr"], "pyramids_added": 0}
                entry_counts[sleeve] += 1
                _log_trade_event(trades, trade_id_counter, candidate["symbol"], sleeve, "entry", candidate["timestamp"], entry_price, proposed_size, None, entry_fee, cash, 0, extra={"risk_per_unit": float(risk_per_unit), "entry_score": float(candidate["score"]), "avg_book_corr": float(candidate["avg_book_corr"]), "risk_scalar": float(candidate["risk_scalar"])})
                sleeve_position_count += 1
                sleeve_open_notional += proposed_notional
                sleeve_open_risk += proposed_risk_cash
                current_open_risk += proposed_risk_cash
                current_gross_notional += proposed_notional
                must_deploy = sleeve_position_count < sleeve_min_positions

    if all_timestamps:
        final_timestamp = all_timestamps[-1]
        for key in list(positions.keys()):
            position = positions[key]
            df = market_data[position["symbol"]]
            if final_timestamp not in df.index:
                continue
            close_px = float(df.loc[final_timestamp, "close"])
            exit_price = _apply_exit(close_px, "long", exit_slippage_bps)
            size_closed = position["size"]
            exit_fee = size_closed * exit_price * fee_rate
            pnl = _realized_pnl(position, exit_price, size_closed, exit_fee)
            cash += pnl
            _log_trade_event(trades, position["trade_id"], position["symbol"], position["strategy"], "final_exit", final_timestamp, exit_price, size_closed, pnl, exit_fee, cash, position["bars_in_trade"], extra={"entry_score": position["entry_score"]})
            del positions[key]

    diag_rows = []
    for sleeve in sleeves_order:
        raw = int(raw_signal_counts.get(sleeve, 0))
        entries = int(entry_counts.get(sleeve, 0))
        blocked = max(raw - entries, 0)
        diag_rows.append({"strategy": sleeve, "raw_signals": raw, "entries": entries, "blocked": blocked, "conversion_pct": round((entries / raw) * 100.0, 2) if raw else 0.0})
    reject_rows = [{"strategy": strategy, "reason": reason, "count": count} for (strategy, reason), count in sorted(rejection_counts.items(), key=lambda x: (x[0][0], -x[1], x[0][1]))]

    equity_curve_df = pd.DataFrame(equity_curve)
    if not equity_curve_df.empty:
        equity_curve_df["time"] = pd.to_datetime(equity_curve_df["time"])
        equity_curve_df.set_index("time", inplace=True)
        equity_curve_df.attrs["signal_pipeline_diagnostics"] = pd.DataFrame(diag_rows)
        equity_curve_df.attrs["rejection_reasons"] = pd.DataFrame(reject_rows)
    return trades, cash, equity_curve_df
