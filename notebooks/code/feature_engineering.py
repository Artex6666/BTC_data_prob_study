"""Ingénierie de caractéristiques pour les modèles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9
LOG_2 = np.log(2)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / (loss + EPS)
    return 100 - (100 / (1 + rs))


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean()
    return atr


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low + EPS)
    percent_d = percent_k.rolling(3).mean()
    return percent_k, percent_d


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    lowest_low = low.rolling(window).min()
    highest_high = high.rolling(window).max()
    return -100 * (highest_high - close) / (highest_high - lowest_low + EPS)


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window).mean()
    mad = (typical_price - sma_tp).abs().rolling(window).mean()
    return (typical_price - sma_tp) / (0.015 * mad + EPS)


def compute_vwap(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series],
    window: int,
) -> pd.Series:
    typical_price = (high + low + close) / 3
    if volume is None:
        volume = pd.Series(1.0, index=close.index)
    weighted_price = typical_price * volume
    cum_price = weighted_price.rolling(window).sum()
    cum_volume = volume.rolling(window).sum()
    return cum_price / (cum_volume + EPS)


def _compute_consecutive_counts(series: pd.Series, condition: Iterable[bool]) -> pd.Series:
    count = 0
    results = []
    for flag in condition:
        if flag:
            count += 1
        else:
            count = 0
        results.append(count)
    return pd.Series(results, index=series.index, dtype=float)


def compute_consecutive_moves(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    delta = series.diff()
    up = _compute_consecutive_counts(series, (delta > 0).fillna(False))
    down = _compute_consecutive_counts(series, (delta < 0).fillna(False))
    return up, down


def _zscore_by_group(values: pd.Series, group: pd.Series) -> pd.Series:
    df = pd.DataFrame({"value": values, "group": group})

    def _transform(s: pd.Series) -> pd.Series:
        std = s.std()
        if std is None or np.isnan(std) or std < EPS:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    return df.groupby("group", group_keys=False)["value"].apply(_transform)


def _label_from_window(window: int) -> str:
    if window < 60:
        return f"{window}m"
    hours = window // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


@dataclass
class FeatureConfig:
    price_col: str = "close"
    open_col: Optional[str] = "open"
    high_col: str = "high"
    low_col: str = "low"
    volume_col: Optional[str] = "volume"
    prefix: str = "feat"
    timestamp_col: str = "timestamp"


def add_price_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Ajoute les features basées sur le prix."""

    df = df.copy()
    price = df[config.price_col]
    open_col = (
        df[config.open_col]
        if config.open_col and config.open_col in df.columns
        else price.shift(1)
    )
    high = df[config.high_col]
    low = df[config.low_col]
    volume = df[config.volume_col] if config.volume_col and config.volume_col in df.columns else None
    prefix = config.prefix

    # Log returns et momentum
    log_return_windows = [1, 5, 15, 30, 60]
    for window in log_return_windows:
        df[f"{prefix}_log_return_{window}"] = np.log(price / price.shift(window))

    momentum_windows = [1, 3, 5, 10, 20, 60, 120]
    for window in momentum_windows:
        df[f"{prefix}_momentum_{window}"] = price - price.shift(window)

    # Moyennes mobiles et Bollinger
    sma_windows = [5, 20, 60, 120, 240]
    for window in sma_windows:
        label = _label_from_window(window)
        sma = price.rolling(window).mean()
        ema = _ema(price, window)
        std = price.rolling(window).std()

        df[f"{prefix}_sma_{label}"] = sma
        df[f"{prefix}_ema_{label}"] = ema
        df[f"{prefix}_boll_mid_{label}"] = sma
        df[f"{prefix}_boll_up_{label}"] = sma + 2 * std
        df[f"{prefix}_boll_low_{label}"] = sma - 2 * std
        df[f"{prefix}_boll_width_{label}"] = (2 * std) / (sma + EPS)
        df[f"{prefix}_close_over_sma_{label}"] = price / (sma + EPS) - 1
        df[f"{prefix}_close_over_ema_{label}"] = price / (ema + EPS) - 1

    # RSI multi-périodes
    for window in (3, 7, 14, 21):
        df[f"{prefix}_rsi_{window}"] = compute_rsi(price, window)

    # MACD standard
    macd_line, signal_line, hist = compute_macd(price)
    df[f"{prefix}_macd_line"] = macd_line
    df[f"{prefix}_macd_signal"] = signal_line
    df[f"{prefix}_macd_hist"] = hist

    # ATR et dérivées
    atr = compute_atr(high, low, price, 14)
    df[f"{prefix}_atr_14"] = atr
    df[f"{prefix}_atr_slope_1"] = atr.diff()
    df[f"{prefix}_atr_slope_5"] = atr.diff(5)

    # Volatilité réalisée
    for window in (10, 30, 60, 120):
        df[f"{prefix}_realized_vol_{window}"] = (
            df[f"{prefix}_log_return_1"].rolling(window).std()
        )

    # Parkinson et Garman-Klass
    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(price / (open_col + EPS)).replace([np.inf, -np.inf], np.nan)
    parkinson = (log_hl**2) / (4 * LOG_2)
    garman_klass = 0.5 * log_hl**2 - (2 * LOG_2 - 1) * log_co**2
    for window in (15, 30, 60, 120):
        df[f"{prefix}_parkinson_vol_{window}"] = parkinson.rolling(window).mean()
        df[f"{prefix}_gk_vol_{window}"] = garman_klass.rolling(window).mean()

    # Ratios de bougie
    body = (price - open_col).abs()
    range_ = (high - low).abs()
    upper_wick = (high - np.maximum(price, open_col)).clip(lower=0)
    lower_wick = (np.minimum(price, open_col) - low).clip(lower=0)

    df[f"{prefix}_range"] = range_
    df[f"{prefix}_body_abs"] = body
    df[f"{prefix}_range_ratio"] = range_ / (body + EPS)
    df[f"{prefix}_upper_wick"] = upper_wick
    df[f"{prefix}_lower_wick"] = lower_wick
    df[f"{prefix}_wick_ratio"] = (upper_wick + lower_wick) / (body + EPS)
    df[f"{prefix}_position_in_range"] = (price - low) / (range_ + EPS)
    df[f"{prefix}_close_over_open"] = (price / (open_col + EPS)) - 1
    df[f"{prefix}_range_pct"] = range_ / (price + EPS)

    # Prises de liquidité intra-bougie
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    df[f"{prefix}_liquidity_grab_high"] = (
        (high > prev_high) & (price < prev_high)
    ).astype(int)
    df[f"{prefix}_liquidity_grab_low"] = (
        (low < prev_low) & (price > prev_low)
    ).astype(int)

    # Consécutifs up/down
    up_run, down_run = compute_consecutive_moves(price)
    df[f"{prefix}_consecutive_up"] = up_run
    df[f"{prefix}_consecutive_down"] = down_run
    df[f"{prefix}_trend_direction"] = np.sign(price.diff()).fillna(0)
    df[f"{prefix}_trend_persistence"] = up_run - down_run

    # Oscillateurs
    for window in (14, 28):
        percent_k, percent_d = compute_stochastic(high, low, price, window)
        df[f"{prefix}_stoch_k_{window}"] = percent_k
        df[f"{prefix}_stoch_d_{window}"] = percent_d
        df[f"{prefix}_williams_r_{window}"] = compute_williams_r(high, low, price, window)
        df[f"{prefix}_cci_{window}"] = compute_cci(high, low, price, window)

    # Volume features et VWAP
    if volume is not None:
        df[f"{prefix}_vol_sma_20"] = volume.rolling(20).mean()
        df[f"{prefix}_vol_zscore"] = (volume - volume.rolling(60).mean()) / (
            volume.rolling(60).std() + EPS
        )
        for window in (60, 240):
            label = _label_from_window(window)
            vwap = compute_vwap(price, high, low, volume, window)
            df[f"{prefix}_vwap_{label}"] = vwap
            df[f"{prefix}_close_over_vwap_{label}"] = price / (vwap + EPS) - 1

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """Ajoute des features temporelles (sinus/cosinus, heure, jour, secondes)."""

    ts = df[timestamp_col]
    if not np.issubdtype(ts.dtype, np.datetime64):
        raise ValueError("La colonne de timestamp doit être de type datetime.")

    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

    if hasattr(ts.dt, "second"):
        df["second"] = ts.dt.second
        df["second_sin"] = np.sin(2 * np.pi * df["second"] / 60)
        df["second_cos"] = np.cos(2 * np.pi * df["second"] / 60)

    return df


def add_regime_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Standardise certaines variables selon les régimes jour/nuit et vol."""

    prefix = config.prefix
    if "hour" not in df.columns:
        return df

    session_day = ((df["hour"] >= 8) & (df["hour"] < 20)).astype(int)
    df[f"{prefix}_session_day"] = session_day
    df[f"{prefix}_session_night"] = 1 - session_day

    vol_ref_col = f"{prefix}_realized_vol_60"
    if vol_ref_col in df.columns:
        rolling_median = df[vol_ref_col].rolling(720, min_periods=60).median()
        high_vol = (df[vol_ref_col] > rolling_median).astype(int)
        df[f"{prefix}_vol_regime_high"] = high_vol
        df[f"{prefix}_vol_regime_low"] = 1 - high_vol

        for col in (f"{prefix}_log_return_1", f"{prefix}_log_return_5"):
            if col in df.columns:
                df[f"{col}_session_z"] = _zscore_by_group(df[col], session_day)
                df[f"{col}_volreg_z"] = _zscore_by_group(df[col], high_vol)

    return df


def add_macd_from_resample(
    df: pd.DataFrame,
    config: FeatureConfig,
    rules: Dict[str, str],
) -> pd.DataFrame:
    """Ajoute des MACD calculés sur des séries resamplées."""

    if config.timestamp_col not in df.columns:
        return df

    indexed = df.set_index(config.timestamp_col)
    for rule, label in rules.items():
        resampled = (
            indexed[config.price_col]
            .resample(rule)
            .last()
            .dropna()
        )
        if resampled.empty:
            continue
        macd_line, signal_line, hist = compute_macd(resampled)
        macd_df = pd.DataFrame(
            {
                f"{config.prefix}_macd_line_{label}": macd_line,
                f"{config.prefix}_macd_signal_{label}": signal_line,
                f"{config.prefix}_macd_hist_{label}": hist,
            }
        )
        macd_df[f"{config.prefix}_macd_slope_{label}"] = macd_line.diff()
        macd_df[f"{config.prefix}_macd_cross_{label}"] = np.sign(macd_line - signal_line)
        macd_df = macd_df.reindex(indexed.index, method="ffill")
        for col in macd_df.columns:
            df[col] = macd_df[col].values

    return df


def add_liquidity_features(
    df: pd.DataFrame,
    config: FeatureConfig,
    rules: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Ajoute des features liées aux prises de liquidité sur des horizons supérieurs."""

    if rules is None:
        rules = {"15min": "15m", "1h": "1h", "4h": "4h"}

    if config.timestamp_col not in df.columns:
        return df

    base = df.set_index(config.timestamp_col)
    select_cols = [c for c in [config.high_col, config.low_col, config.price_col] if c in df.columns]
    if len(select_cols) < 3:
        return df

    for rule, label in rules.items():
        agg = base[select_cols].resample(rule).agg(
            {
                config.high_col: "max",
                config.low_col: "min",
                config.price_col: "last",
            }
        ).dropna()
        if agg.empty:
            continue
        agg.rename(
            columns={
                config.high_col: "htf_high",
                config.low_col: "htf_low",
                config.price_col: "htf_close",
            },
            inplace=True,
        )
        agg["prev_high"] = agg["htf_high"].shift(1)
        agg["prev_low"] = agg["htf_low"].shift(1)
        expanded = agg.reindex(base.index, method="ffill")

        df[f"{config.prefix}_htf_high_{label}"] = expanded["htf_high"].values
        df[f"{config.prefix}_htf_low_{label}"] = expanded["htf_low"].values
        df[f"{config.prefix}_close_over_htf_high_{label}"] = (
            df[config.price_col] / (expanded["htf_high"] + EPS) - 1
        )
        df[f"{config.prefix}_close_over_htf_low_{label}"] = (
            df[config.price_col] / (expanded["htf_low"] + EPS) - 1
        )
        df[f"{config.prefix}_liquidity_grab_htf_high_{label}"] = (
            (df[config.high_col] > expanded["prev_high"])
            & (df[config.price_col] < expanded["prev_high"])
            & expanded["prev_high"].notna()
        ).astype(int)
        df[f"{config.prefix}_liquidity_grab_htf_low_{label}"] = (
            (df[config.low_col] < expanded["prev_low"])
            & (df[config.price_col] > expanded["prev_low"])
            & expanded["prev_low"].notna()
        ).astype(int)

    return df


def add_multi_timeframe_context(
    df: pd.DataFrame,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Ajoute les contextes multi-timeframes (MACD, tendances relatives)."""

    macd_rules = {"5min": "5m", "15min": "15m"}
    df = add_macd_from_resample(df, config, macd_rules)
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    config: FeatureConfig,
    dropna: bool = True,
) -> pd.DataFrame:
    """Construit la matrice finale de features."""

    df = df.copy()
    df = add_price_features(df, config)
    df = add_time_features(df, config.timestamp_col)
    df = add_regime_features(df, config)
    df = add_multi_timeframe_context(df, config)
    df = add_liquidity_features(df, config)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


__all__ = [
    "FeatureConfig",
    "add_price_features",
    "add_time_features",
    "add_regime_features",
    "add_multi_timeframe_context",
    "add_liquidity_features",
    "build_feature_matrix",
    "compute_rsi",
    "compute_atr",
    "compute_macd",
    "compute_consecutive_moves",
]

