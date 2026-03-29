from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_SLIPPAGE_BPS = 5.0


def _slippage_fraction(bps: float) -> float:
    return bps / 10000.0


def _apply_exit_slippage(price: float, slippage_bps: float) -> float:
    return price * (1.0 - _slippage_fraction(slippage_bps))


def _apply_entry_slippage(price: float, slippage_bps: float) -> float:
    return price * (1.0 + _slippage_fraction(slippage_bps))


def _make_trade_event(trade_id: int, event_type: str, timestamp, symbol: str, **kwargs) -> Dict:
    event = {
        "trade_id": trade_id,
        "type": event_type,
        "time": str(timestamp),
        "symbol": symbol,
    }
    event.update(kwargs)
    return event


def run_backtest(
    df: pd.DataFrame,
    symbol: str = "SINGLE_ASSET",
    starting_equity: float = 10000,
    risk_per_trade: float = 0.005,
    fee_rate: float = 0.0004,
    cooldown_bars: int = 10,
    trailing_atr_multiple: float = 6.0,
    time_stop_bars: int = 120,
    entry_slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    exit_slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    stop_atr_buffer: float = 0.10,
    partial_r_multiple: float = 2.0,
    partial_close_fraction: float = 0.5,
    min_risk_distance_pct: float = 0.002,
    conservative_same_bar: bool = True,
) -> Tuple[List[Dict], float, pd.DataFrame]:
    data = df.copy()

    equity = float(starting_equity)
    position: Optional[Dict] = None
    trades: List[Dict] = []
    cooldown_until = -1
    equity_curve: List[Dict] = []
    next_trade_id = 1

    for i in range(200, len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]
        timestamp = data.index[i]

        if position is None and i > cooldown_until and int(row.get("signal", 0)) == 1:
            atr = row["ATR"]
            sma50 = row["SMA50"]
            signal_low = row["low"]
            signal_high = row["high"]
            entry_score = float(row.get("entry_score", 0.0))

            if pd.isna(atr) or atr <= 0:
                continue

            stop_reference = min(signal_low, sma50 - atr) - stop_atr_buffer * atr
            entry_price = _apply_entry_slippage(float(next_row["open"]), entry_slippage_bps)
            risk_per_unit = entry_price - stop_reference

            if pd.isna(risk_per_unit) or risk_per_unit <= 0:
                continue

            if (risk_per_unit / entry_price) < min_risk_distance_pct:
                continue

            # Scale slightly by signal quality but cap tightly to avoid unstable size jumps.
            quality_scalar = min(max(0.75 + 0.10 * entry_score, 0.75), 1.15)
            risk_budget = equity * risk_per_trade * quality_scalar
            position_size = risk_budget / risk_per_unit
            entry_notional = position_size * entry_price
            entry_fee = entry_notional * fee_rate

            equity -= entry_fee
            trade_id = next_trade_id
            next_trade_id += 1

            position = {
                "trade_id": trade_id,
                "entry_time": data.index[i + 1],
                "entry_price": entry_price,
                "stop_price": stop_reference,
                "pending_stop_price": stop_reference,
                "size": position_size,
                "initial_size": position_size,
                "initial_risk": risk_per_unit,
                "max_price": entry_price,
                "partial_taken": False,
                "bars_in_trade": 0,
                "entry_score": entry_score,
                "signal_high": signal_high,
            }

            trades.append(
                _make_trade_event(
                    trade_id,
                    "entry",
                    data.index[i + 1],
                    symbol,
                    price=float(entry_price),
                    size=float(position_size),
                    fee=float(entry_fee),
                    risk_per_unit=float(risk_per_unit),
                    equity=float(equity),
                    entry_score=entry_score,
                )
            )
            continue

        if position is not None:
            position["bars_in_trade"] += 1

            high = float(row["high"])
            low = float(row["low"])
            close = float(row["close"])
            atr = float(row["ATR"])

            old_stop = float(position["stop_price"])
            old_max_price = float(position["max_price"])
            new_max_price = max(old_max_price, high)
            position["max_price"] = new_max_price

            target_price = position["entry_price"] + partial_r_multiple * position["initial_risk"]
            stop_touched = low <= old_stop
            target_touched = (not position["partial_taken"]) and (high >= target_price)

            if conservative_same_bar and stop_touched:
                exit_price = _apply_exit_slippage(old_stop, exit_slippage_bps)
                size_closed = position["size"]
                exit_notional = size_closed * exit_price
                exit_fee = exit_notional * fee_rate
                pnl = (exit_price - position["entry_price"]) * size_closed - exit_fee
                equity += pnl

                trades.append(
                    _make_trade_event(
                        position["trade_id"],
                        "stop",
                        timestamp,
                        symbol,
                        price=float(exit_price),
                        size=float(size_closed),
                        pnl=float(pnl),
                        fee=float(exit_fee),
                        equity=float(equity),
                        bars_in_trade=int(position["bars_in_trade"]),
                    )
                )
                position = None
                cooldown_until = i + cooldown_bars
            else:
                if target_touched:
                    size_closed = position["size"] * partial_close_fraction
                    exit_price = _apply_exit_slippage(target_price, exit_slippage_bps)
                    exit_notional = size_closed * exit_price
                    exit_fee = exit_notional * fee_rate
                    pnl = (exit_price - position["entry_price"]) * size_closed - exit_fee
                    equity += pnl

                    position["size"] -= size_closed
                    position["partial_taken"] = True
                    position["stop_price"] = max(position["stop_price"], position["entry_price"])

                    trades.append(
                        _make_trade_event(
                            position["trade_id"],
                            "partial",
                            timestamp,
                            symbol,
                            price=float(exit_price),
                            size=float(size_closed),
                            pnl=float(pnl),
                            fee=float(exit_fee),
                            equity=float(equity),
                            bars_in_trade=int(position["bars_in_trade"]),
                        )
                    )

                current_stop = float(position["stop_price"])
                if low <= current_stop:
                    exit_price = _apply_exit_slippage(current_stop, exit_slippage_bps)
                    size_closed = position["size"]
                    exit_notional = size_closed * exit_price
                    exit_fee = exit_notional * fee_rate
                    pnl = (exit_price - position["entry_price"]) * size_closed - exit_fee
                    equity += pnl

                    trades.append(
                        _make_trade_event(
                            position["trade_id"],
                            "stop",
                            timestamp,
                            symbol,
                            price=float(exit_price),
                            size=float(size_closed),
                            pnl=float(pnl),
                            fee=float(exit_fee),
                            equity=float(equity),
                            bars_in_trade=int(position["bars_in_trade"]),
                        )
                    )
                    position = None
                    cooldown_until = i + cooldown_bars
                else:
                    trailing_stop = new_max_price - trailing_atr_multiple * atr
                    if not math.isnan(trailing_stop):
                        position["pending_stop_price"] = max(position["stop_price"], trailing_stop)
                        position["stop_price"] = position["pending_stop_price"]

                    if position["bars_in_trade"] > time_stop_bars:
                        exit_price = _apply_exit_slippage(close, exit_slippage_bps)
                        size_closed = position["size"]
                        exit_notional = size_closed * exit_price
                        exit_fee = exit_notional * fee_rate
                        pnl = (exit_price - position["entry_price"]) * size_closed - exit_fee
                        equity += pnl

                        trades.append(
                            _make_trade_event(
                                position["trade_id"],
                                "time_exit",
                                timestamp,
                                symbol,
                                price=float(exit_price),
                                size=float(size_closed),
                                pnl=float(pnl),
                                fee=float(exit_fee),
                                equity=float(equity),
                                bars_in_trade=int(position["bars_in_trade"]),
                            )
                        )
                        position = None
                        cooldown_until = i + cooldown_bars

        marked_equity = equity
        if position is not None:
            marked_equity += (close - position["entry_price"]) * position["size"]

        equity_curve.append(
            {
                "time": timestamp,
                "equity": float(marked_equity),
                "cash": float(equity),
                "open_positions": 0 if position is None else 1,
            }
        )

    if position is not None:
        final_timestamp = data.index[-1]
        final_close = _apply_exit_slippage(float(data.iloc[-1]["close"]), exit_slippage_bps)
        size_closed = position["size"]
        exit_notional = size_closed * final_close
        exit_fee = exit_notional * fee_rate
        pnl = (final_close - position["entry_price"]) * size_closed - exit_fee
        equity += pnl
        trades.append(
            _make_trade_event(
                position["trade_id"],
                "final_exit",
                final_timestamp,
                symbol,
                price=float(final_close),
                size=float(size_closed),
                pnl=float(pnl),
                fee=float(exit_fee),
                equity=float(equity),
                bars_in_trade=int(position["bars_in_trade"]),
            )
        )

    equity_curve_df = pd.DataFrame(equity_curve)
    if not equity_curve_df.empty:
        equity_curve_df["time"] = pd.to_datetime(equity_curve_df["time"])
        equity_curve_df.set_index("time", inplace=True)

    return trades, equity, equity_curve_df
