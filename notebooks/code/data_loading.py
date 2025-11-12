"""Fonctions de chargement des données pour le projet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import pandas as pd

from .paths import DATA_DIR


@dataclass(frozen=True)
class DataPaths:
    """Chemins standards des jeux de données."""

    polymarket: str = str(DATA_DIR / "BTC.csv")
    ohlc_1m: str = str(DATA_DIR / "btc_1m_OHLC.csv")


def load_polymarket_data(
    path: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Charge les données Polymarket (granularité 1s)."""

    usecols = list(columns) if columns is not None else None
    df = pd.read_csv(
        path or DataPaths().polymarket,
        usecols=usecols,
        parse_dates=["timestamp"],
    )
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_ohlc_1m_data(
    path: Optional[str] = None,
    columns: Optional[Iterable[str]] = None,
    tz: str = "UTC",
) -> pd.DataFrame:
    """Charge les données OHLC en 1 minute."""

    usecols = list(columns) if columns is not None else None
    df = pd.read_csv(
        path or DataPaths().ohlc_1m,
        usecols=usecols,
        parse_dates=["timestamp"],
    )
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def resample_seconds_to_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Pivote les données par seconde vers des bougies 1m."""

    if df.empty:
        return df.copy()

    features = {}
    if "spot_price" in df.columns:
        features.update(
            {
                "open": ("spot_price", "first"),
                "high": ("spot_price", "max"),
                "low": ("spot_price", "min"),
                "close": ("spot_price", "last"),
                "volume_proxy": ("spot_price", "count"),
            }
        )

    ohlc = (
        df.set_index("timestamp")
        .resample("1min")
        .agg(**features)
        .dropna(subset=["open", "high", "low", "close"], how="all")
        .reset_index()
    )
    return ohlc


def align_to_ohlc(
    per_second: pd.DataFrame,
    ohlc: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta(minutes=1),
) -> pd.DataFrame:
    """Fusionne les caractéristiques 1m sur les points 1s."""

    if per_second.empty or ohlc.empty:
        return per_second.copy()

    merged = pd.merge_asof(
        per_second.sort_values("timestamp"),
        ohlc.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=tolerance,
        suffixes=("", "_ohlc"),
    )
    return merged


__all__ = [
    "DataPaths",
    "load_polymarket_data",
    "load_ohlc_1m_data",
    "resample_seconds_to_minutes",
    "align_to_ohlc",
]

