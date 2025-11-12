"""Fonctions de visualisation pour le projet."""

from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_odds_comparison(
    df: pd.DataFrame,
    timeframe: str,
    predicted_cols: Iterable[str],
    actual_col: str,
    n_samples: int = 500,
) -> Figure:
    """Affiche la comparaison entre cotes réelles et prédites."""

    sample = df.sort_values("timestamp").tail(n_samples)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(sample["timestamp"], sample[actual_col], label=f"Cote réelle ({timeframe})", linewidth=2)

    for col in predicted_cols:
        ax.plot(sample["timestamp"], sample[col], label=col, alpha=0.7)

    ax.set_title(f"Comparaison des cotes {timeframe}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Probabilité")
    ax.legend()
    ax.grid(True, alpha=0.3)
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

