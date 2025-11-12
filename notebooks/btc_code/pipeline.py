"""Pipeline de préparation des données et des modèles."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_loading import (
    align_to_ohlc,
    load_ohlc_1m_data,
    load_polymarket_data,
)
from .feature_engineering import (
    FeatureConfig,
    add_time_features,
    build_feature_matrix,
    compute_atr,
    compute_consecutive_moves,
)
from .model_training import compute_market_probabilities
from .timeframe_utils import (
    Timeframe,
    compute_contract_price_features,
    compute_forward_returns,
)


def load_all_data(tz: str = "UTC") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge l'ensemble des données nécessaires."""

    polymarket = load_polymarket_data(tz=tz)
    ohlc = load_ohlc_1m_data(tz=tz)
    return polymarket, ohlc


def select_recent_rows(df: pd.DataFrame, n: int, stride: int = 1, offset: int = 0) -> pd.DataFrame:
    """Retourne une copie des n dernières lignes avec possibilité de sous-échantillonnage.

    Args:
        df: DataFrame source trié chronologiquement.
        n: nombre d'échantillons désirés après sous-échantillonnage.
        stride: pas d'échantillonnage (1 = toutes les lignes, 2 = une ligne sur deux, etc.).
        offset: décalage de départ (0 <= offset < stride).
    """

    if n <= 0:
        subset = df.copy()
    else:
        subset = df.tail(n * max(stride, 1)).copy()

    stride = max(stride, 1)
    offset = offset % stride
    if stride > 1 or offset:
        subset = subset.iloc[offset::stride]
    return subset.reset_index(drop=True)


def estimate_average_spreads(dataset: pd.DataFrame, timeframes: List[Timeframe]) -> Dict[Timeframe, Dict[str, float]]:
    """Calcule les spreads moyens up/down pour chaque timeframe."""

    spreads = {}
    for tf in timeframes:
        spreads[tf] = {
            "spread_up": float(dataset[f"{tf}_spread_up"].mean()),
            "spread_down": float(dataset[f"{tf}_spread_down"].mean()),
        }
    return spreads


def prepare_ohlc_features(ohlc_df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features 1m."""

    ohlc_df = ohlc_df.sort_values("timestamp").reset_index(drop=True)
    config = FeatureConfig(
        price_col="close",
        open_col="open",
        high_col="high",
        low_col="low",
        volume_col="volume",
        prefix="m1",
        timestamp_col="timestamp",
    )
    features = build_feature_matrix(ohlc_df, config, dropna=False)
    features["atr_15m"] = compute_atr(
        ohlc_df["high"],
        ohlc_df["low"],
        ohlc_df["close"],
        window=15,
    )
    features = features.dropna().reset_index(drop=True)
    return features


def enrich_polymarket_with_features(
    polymarket_df: pd.DataFrame,
    ohlc_features: pd.DataFrame,
    stride: int = 1,
    offset: int = 0,
) -> pd.DataFrame:
    """Ajoute les features 1m sur les données Polymarket (1s) et des signaux microstructure."""

    stride = max(stride, 1)
    offset = offset % stride
    base = polymarket_df.sort_values("timestamp").reset_index(drop=True)
    if stride > 1 or offset:
        base = base.iloc[offset::stride].reset_index(drop=True)

    merged = align_to_ohlc(base, ohlc_features, tolerance=pd.Timedelta(minutes=5))
    if "atr_15m_ohlc" in merged.columns and "atr_15m" not in merged.columns:
        merged["atr_15m"] = merged["atr_15m_ohlc"]
    merged = add_time_features(merged, "timestamp")

    merged["spot_second"] = merged["timestamp"].dt.second
    merged["spot_position_in_minute"] = merged["spot_second"] / 60.0
    merged["spot_session_day"] = (
        (merged["timestamp"].dt.hour >= 8) & (merged["timestamp"].dt.hour < 20)
    ).astype(int)

    merged["spot_return_1s"] = merged["spot_price"].pct_change()
    merged["spot_return_5s"] = merged["spot_price"].pct_change(5)
    merged["spot_return_30s"] = merged["spot_price"].pct_change(30)
    merged["spot_return_60s"] = merged["spot_price"].pct_change(60)

    merged["spot_momentum_5s"] = merged["spot_price"] - merged["spot_price"].shift(5)
    merged["spot_momentum_15s"] = merged["spot_price"] - merged["spot_price"].shift(15)
    merged["spot_momentum_30s"] = merged["spot_price"] - merged["spot_price"].shift(30)
    merged["spot_momentum_60s"] = merged["spot_price"] - merged["spot_price"].shift(60)

    up_run, down_run = compute_consecutive_moves(merged["spot_price"])
    merged["spot_consecutive_up"] = up_run
    merged["spot_consecutive_down"] = down_run

    merged["spot_volatility_10s"] = merged["spot_return_1s"].rolling(10).std()
    merged["spot_volatility_30s"] = merged["spot_return_1s"].rolling(30).std()
    merged["spot_volatility_60s"] = merged["spot_return_1s"].rolling(60).std()
    merged["spot_volatility_120s"] = merged["spot_return_1s"].rolling(120).std()

    merged["spot_realized_vol_60s"] = merged["spot_volatility_60s"]
    vol60 = merged["spot_volatility_60s"]
    merged["spot_vol_regime_high"] = (vol60 > vol60.median()).astype(int)

    merged["spot_price_max_60s"] = merged["spot_price"].rolling(60).max()
    merged["spot_price_min_60s"] = merged["spot_price"].rolling(60).min()
    merged["spot_range_ratio_60s"] = (
        merged["spot_price_max_60s"] - merged["spot_price_min_60s"]
    ) / (merged["spot_price"] + 1e-9)

    merged["spot_zscore_120s"] = (
        merged["spot_price"] - merged["spot_price"].rolling(120).mean()
    ) / (merged["spot_price"].rolling(120).std() + 1e-9)

    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    if "atr_15m" in merged.columns:
        merged["atr_15m"] = merged["atr_15m"].ffill()
    elif "atr_15m_ohlc" in merged.columns:
        merged["atr_15m"] = merged["atr_15m_ohlc"].ffill()
    essential_cols = [col for col in ["spot_price"] if col in merged.columns]
    merged = merged.dropna(subset=essential_cols).reset_index(drop=True)
    return merged


def prepare_timeframe_tables(
    dataset: pd.DataFrame,
    timeframe: Timeframe,
    price_col: str = "spot_price",
) -> pd.DataFrame:
    """Ajoute les colonnes spécifiques au timeframe (contrats, temps restant)."""

    df = dataset.copy()
    df = compute_market_probabilities(df, timeframe)
    df = compute_contract_price_features(df, timeframe, price_col=price_col)
    df = compute_forward_returns(df, timeframe, price_col=price_col)

    if "atr_15m" not in df.columns:
        if "atr_15m_ohlc" in df.columns:
            df["atr_15m"] = df["atr_15m_ohlc"]
        elif "m1_atr_14" in df.columns:
            df["atr_15m"] = df["m1_atr_14"]
        else:
            atr = compute_atr(
                df.get("high", df[price_col]),
                df.get("low", df[price_col]),
                df[price_col],
                window=15,
            )
            df["atr_15m"] = atr
    df["atr_15m"] = df["atr_15m"].ffill()

    df[f"{timeframe}_prob_base"] = df[f"{timeframe}_prob_up_market"]
    df[f"{timeframe}_contract_id_str"] = df[f"{timeframe}_contract_id"]
    df[f"{timeframe}_time_remaining_ratio"] = df[f"{timeframe}_time_remaining_ratio"]
    return df


def make_fomo_input(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Prépare les colonnes nécessaires à la simulation FOMO."""

    columns = {
        f"{timeframe}_prob_up_market": "prob_up",
        f"{timeframe}_time_remaining_ratio": "time_remaining_ratio",
        "atr_15m": "atr_15m",
        f"{timeframe}_tf_close_to_now": "tf_close_to_now",
        f"{timeframe}_tf_open": "tf_open",
        f"{timeframe}_tf_high_to_now": "tf_high_to_now",
        f"{timeframe}_tf_low_to_now": "tf_low_to_now",
        f"{timeframe}_contract_id": "contract_id",
    }
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour make_fomo_input: {missing}")
    fomo_df = df[list(columns.keys()) + ["timestamp"]].rename(columns=columns)
    return fomo_df


def build_regression_dataset(df: pd.DataFrame, timeframe: Timeframe):
    """Construit le dataset pour la régression des cotes.

    Retourne un tuple (dataset, feature_cols, target_cols).
    """

    target_cols = [
        f"{timeframe}_price_up_mid",
        f"{timeframe}_price_down_mid",
        f"{timeframe}_prob_up_market",
    ]
    feature_cols = [
        col
        for col in df.columns
        if col
        not in target_cols
        + [
            "timestamp",
            f"{timeframe}_future_price",
            f"{timeframe}_future_return",
            f"{timeframe}_target_up",
            f"{timeframe}_contract_id",
        ]
    ]
    subset = df[["timestamp"] + feature_cols + target_cols].dropna()
    dataset = subset.reset_index().rename(columns={"index": "original_index"})
    return dataset, feature_cols, target_cols


def build_classification_dataset(df: pd.DataFrame, timeframe: Timeframe):
    """Construit le dataset de classification (probabilité de clôture up).

    Retourne un tuple (dataset, feature_cols, target_col).
    """

    target_col = f"{timeframe}_target_up"
    allowed_prefixes = (
        "m1_",
        "hour",
        "minute",
        "second",
        "day_of_week",
        "is_weekend",
        "spot_",
        "atr_15m",
    )
    disallow = {
        "timestamp",
        target_col,
        f"{timeframe}_future_price",
        f"{timeframe}_future_return",
        f"{timeframe}_contract_id",
    }

    feature_cols = []
    for col in df.columns:
        if col in disallow:
            continue
        if col.startswith(f"{timeframe}_price") or col.startswith(f"{timeframe}_prob"):
            continue
        if col.startswith(allowed_prefixes):
            feature_cols.append(col)

    subset = df[["timestamp"] + feature_cols + [target_col]].dropna()
    dataset = subset.reset_index().rename(columns={"index": "original_index"})
    return dataset, feature_cols, target_col


def prepare_minute_history(
    ohlc_features: pd.DataFrame,
    timeframe: Timeframe,
) -> pd.DataFrame:
    """Construit un dataset minute pour dérouler le backtest historique."""

    df = ohlc_features.copy()
    df["spot_price"] = df["close"]
    df = compute_contract_price_features(df, timeframe, price_col="spot_price")
    df = compute_forward_returns(df, timeframe, price_col="spot_price")
    df = df.dropna().reset_index(drop=True)
    return df


__all__ = [
    "load_all_data",
    "prepare_ohlc_features",
    "enrich_polymarket_with_features",
    "prepare_timeframe_tables",
    "make_fomo_input",
    "build_regression_dataset",
    "build_classification_dataset",
    "prepare_minute_history",
    "select_recent_rows",
    "estimate_average_spreads",
]

