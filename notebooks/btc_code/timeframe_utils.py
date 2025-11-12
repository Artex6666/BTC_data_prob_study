"""Utilitaires pour les contrats Polymarket (m15, h1, daily)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from pandas import Timedelta

Timeframe = Literal["m15", "h1", "daily"]


@dataclass(frozen=True)
class TimeframeSpec:
    label: Timeframe
    duration: Timedelta
    resample_rule: str


TIMEFRAME_SPECS: Dict[Timeframe, TimeframeSpec] = {
    "m15": TimeframeSpec("m15", Timedelta(minutes=15), "15min"),
    "h1": TimeframeSpec("h1", Timedelta(hours=1), "1h"),
    "daily": TimeframeSpec("daily", Timedelta(hours=24), "1d"),
}


def _daily_contract_close(ts: pd.Series) -> pd.Series:
    """Calcule les clôtures à 12h ET pour les contrats daily."""

    ts_est = ts.dt.tz_convert("US/Eastern")
    close_est = (ts_est - Timedelta(hours=12)).dt.floor("D") + Timedelta(hours=12)
    close_est = close_est.where(close_est > ts_est, close_est + Timedelta(days=1))
    close_utc = close_est.dt.tz_convert("UTC")
    return close_utc


def assign_contracts(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Assigne les identifiants de contrat et le temps restant."""

    if "timestamp" not in df.columns:
        raise ValueError("La colonne 'timestamp' est obligatoire.")

    spec = TIMEFRAME_SPECS[timeframe]
    df = df.copy()
    df = df.sort_values("timestamp")

    if timeframe == "daily":
        contract_close = _daily_contract_close(df["timestamp"])
    else:
        floor = df["timestamp"].dt.floor(spec.resample_rule)
        contract_close = floor + spec.duration

    contract_start = contract_close - spec.duration
    time_elapsed = (df["timestamp"] - contract_start).dt.total_seconds()
    total_time = spec.duration.total_seconds()
    df[f"{timeframe}_contract_close"] = contract_close
    df[f"{timeframe}_contract_start"] = contract_start
    df[f"{timeframe}_time_remaining_ratio"] = 1 - np.clip(time_elapsed / total_time, 0, 1)
    df[f"{timeframe}_time_elapsed_ratio"] = 1 - df[f"{timeframe}_time_remaining_ratio"]
    df[f"{timeframe}_contract_id"] = contract_close.astype(str)

    return df


def compute_contract_price_features(
    df: pd.DataFrame,
    timeframe: Timeframe,
    price_col: str = "spot_price",
) -> pd.DataFrame:
    """Ajoute les features de prix intra-contrat (open/high/low)."""

    df = assign_contracts(df, timeframe)
    contract_id = f"{timeframe}_contract_id"

    df[f"{timeframe}_tf_open"] = df.groupby(contract_id)[price_col].transform("first")
    df[f"{timeframe}_tf_high_to_now"] = df.groupby(contract_id)[price_col].cummax()
    df[f"{timeframe}_tf_low_to_now"] = df.groupby(contract_id)[price_col].cummin()
    df[f"{timeframe}_tf_close_to_now"] = df[price_col]

    return df


def compute_forward_returns(
    df: pd.DataFrame,
    timeframe: Timeframe,
    price_col: str = "spot_price",
) -> pd.DataFrame:
    """Calcule les retours futurs et les labels binaires up/down."""

    df = assign_contracts(df, timeframe)
    spec = TIMEFRAME_SPECS[timeframe]

    df = df.sort_values("timestamp")
    future = (
        df.set_index("timestamp")[price_col]
        .shift(freq=spec.duration)
        .reindex(df["timestamp"])
    )
    df[f"{timeframe}_future_price"] = future.values
    df[f"{timeframe}_future_return"] = (
        df[f"{timeframe}_future_price"] / df[price_col] - 1
    )
    df[f"{timeframe}_target_up"] = (df[f"{timeframe}_future_return"] >= 0).astype(int)
    return df


__all__ = [
    "Timeframe",
    "TimeframeSpec",
    "TIMEFRAME_SPECS",
    "assign_contracts",
    "compute_contract_price_features",
    "compute_forward_returns",
]

