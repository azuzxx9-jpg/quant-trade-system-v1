import numpy as np
import pandas as pd


EPSILON = 1e-12


def compute_sma(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].rolling(window=period, min_periods=period).mean()


def compute_ema(df: pd.DataFrame, period: int) -> pd.Series:
    return df["close"].ewm(span=period, adjust=False, min_periods=period).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing is preferable for trading systems.
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_smoothed / (atr + EPSILON))
    minus_di = 100.0 * (minus_dm_smoothed / (atr + EPSILON))
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPSILON)

    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["SMA20"] = compute_sma(out, 20)
    out["SMA50"] = compute_sma(out, 50)
    out["SMA200"] = compute_sma(out, 200)

    out["EMA20"] = compute_ema(out, 20)
    out["ATR"] = compute_atr(out, 20)
    out["ADX"] = compute_adx(out, 14)

    out["returns_1"] = out["close"].pct_change(1)
    out["returns_5"] = out["close"].pct_change(5)
    out["returns_20"] = out["close"].pct_change(20)
    out["volatility_20"] = out["returns_1"].rolling(20).std()
    out["atr_pct"] = out["ATR"] / out["close"]
    out["sma20_distance_atr"] = (out["close"] - out["SMA20"]) / out["ATR"]
    out["sma50_distance_atr"] = (out["close"] - out["SMA50"]) / out["ATR"]

    return out
