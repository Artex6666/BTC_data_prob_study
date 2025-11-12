"""Fonctions de visualisation pour le projet."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure

FRENCH_MONTHS = {
    1: "janvier",
    2: "février",
    3: "mars",
    4: "avril",
    5: "mai",
    6: "juin",
    7: "juillet",
    8: "août",
    9: "septembre",
    10: "octobre",
    11: "novembre",
    12: "décembre",
}


def _format_french_date(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return ""
    ts_local = ts
    if ts.tzinfo is not None:
        try:
            ts_local = ts.tz_convert("Europe/Paris")
        except Exception:
            ts_local = ts.tz_convert("UTC")
    day = ts_local.day
    month = FRENCH_MONTHS.get(ts_local.month, "")
    return f"{day} {month}"


def plot_odds_comparison(
    df: pd.DataFrame,
    timeframe_label: str,
    predicted_cols: Iterable[str],
    actual_col: str,
    contract_col: str,
    contract_ids: Optional[Sequence[str]] = None,
    title_prefix: str = "Comparaisons des cotes",
    date_label: Optional[str] = None,
    label_map: Optional[Dict[str, str]] = None,
) -> Figure:
    """Affiche la comparaison entre cotes réelles et prédites."""

    label_map = label_map or {}
    data = df.sort_values("timestamp")
    if contract_ids:
        data = data[data[contract_col].isin(contract_ids)]

    if data.empty:
        raise ValueError("Aucune donnée disponible pour les contrats sélectionnés.")

    if contract_ids is None:
        contract_sequence = data[contract_col].unique().tolist()
    else:
        contract_sequence = list(dict.fromkeys(contract_ids))

    first_ts = data["timestamp"].iloc[0]
    subtitle = date_label or _format_french_date(first_ts)

    fig, ax = plt.subplots(figsize=(12, 4))

    actual_label_used = False
    predicted_labels_used = {col: False for col in predicted_cols}

    for contract_id in contract_sequence:
        segment = data[data[contract_col] == contract_id].sort_values("timestamp")
        if segment.empty:
            continue
        segment = segment.dropna(subset=[actual_col] + list(predicted_cols))
        if segment.empty:
            continue
        line_label = label_map.get(actual_col, f"Cote marché ({timeframe_label})") if not actual_label_used else None
        ax.plot(
            segment["timestamp"],
            segment[actual_col],
            label=line_label,
            linewidth=2.2,
        )
        actual_label_used = True

        for col in predicted_cols:
            label = label_map.get(col, col)
            display_label = label if not predicted_labels_used[col] else None
            ax.plot(
                segment["timestamp"],
                segment[col],
                linestyle="--",
                linewidth=1.8,
                label=display_label,
            )
            predicted_labels_used[col] = True

    ax.set_ylabel("Probabilité de clôture UP")
    ax.set_xlabel("Heure")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    fig.suptitle(f"{title_prefix} {timeframe_label}\n{subtitle}", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    return fig


def plot_equity_curves(backtest_results) -> Figure:
    """Affiche les courbes d'équité des deux stratégies."""

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(backtest_results.equity_fractional.values, label="2% capital par trade")
    ax.plot(backtest_results.equity_share.values, label="4% de parts fixes")
    ax.set_title("Évolution du capital")
    ax.set_xlabel("Trades exécutés")
    ax.set_ylabel("Capital")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_calibration_curve(
    df: pd.DataFrame,
    proba_col: str,
    target_col: str,
    n_bins: int = 10,
) -> Figure:
    """Affiche la courbe de calibration du classifieur."""

    df = df.dropna(subset=[proba_col, target_col])
    bins = pd.qcut(df[proba_col], q=n_bins, duplicates="drop")
    grouped = df.groupby(bins)[[proba_col, target_col]].mean()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(grouped[proba_col], grouped[target_col], marker="o", label="Empirique")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Idéal")
    ax.set_title("Courbe de calibration")
    ax.set_xlabel("Probabilité prédite")
    ax.set_ylabel("Fréquence observée")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


__all__ = ["plot_odds_comparison", "plot_equity_curves", "plot_calibration_curve"]

