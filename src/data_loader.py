import os
from typing import Optional

import pandas as pd
from binance.client import Client
from dotenv import load_dotenv


REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

load_dotenv()


def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=REQUIRED_COLUMNS)

    bad_rows = (out["high"] < out[["open", "close", "low"]].max(axis=1)) | (
        out["low"] > out[["open", "close", "high"]].min(axis=1)
    )
    if bad_rows.any():
        raise ValueError("Invalid OHLC relationships detected in input data.")

    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    return out


def fetch_binance_data(
    symbol: str = "BTCUSDT",
    interval: str = "4h",
    start_str: str = "1 Jan 2021",
    end_str: Optional[str] = None,
) -> pd.DataFrame:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    client = Client(api_key, api_secret)
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)

    df = pd.DataFrame(
        klines,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(None)
    df.set_index("open_time", inplace=True)

    return _validate_ohlcv(df)


def save_data(df: pd.DataFrame, filename: str) -> None:
    out = _validate_ohlcv(df)
    out.to_csv(filename, index=True)


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=[0], index_col=0)
    df.index.name = "open_time"
    return _validate_ohlcv(df)
