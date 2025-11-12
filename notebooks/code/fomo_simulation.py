"""Simulation de scénarios FOMO pour les cotes Polymarket."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FomoScenario:
    name: str
    fomo_index: float
    aggressiveness: float
    stickiness: float
    noise: float = 0.0
    alpha: float = 4.0
    beta: float = 2.0
    gamma: float = 1.0
    k_atr: float = 1.0


REQUIRED_COLUMNS = {
    "prob_up",
    "time_remaining_ratio",
    "atr_15m",
    "tf_close_to_now",
    "tf_open",
    "tf_high_to_now",
    "tf_low_to_now",
}


def simulate_fomo_odds(
    df: pd.DataFrame,
    scenarios: Iterable[FomoScenario],
    prob_column: str = "prob_up",
    time_remaining_col: str = "time_remaining_ratio",
    atr_col: str = "atr_15m",
    close_col: str = "tf_close_to_now",
    open_col: str = "tf_open",
    high_col: str = "tf_high_to_now",
    low_col: str = "tf_low_to_now",
    contract_col: str = "contract_id",
) -> pd.DataFrame:
    """Génère des cotes simulées minute par minute pour chaque scénario."""

    missing = REQUIRED_COLUMNS - set(
        {
            prob_column,
            time_remaining_col,
            atr_col,
            close_col,
            open_col,
            high_col,
            low_col,
        }
    )
    if missing:
        raise ValueError(f"Colonnes manquantes pour la simulation FOMO: {missing}")

    if contract_col not in df.columns:
        raise ValueError(f"La colonne '{contract_col}' est obligatoire.")

    simulated = df.copy()

    def _simulate_group(group: pd.DataFrame, scenario: FomoScenario) -> np.ndarray:
        odds = []
        prev_odds = None
        eps = 1e-6
        for row in group.itertuples():
            base = getattr(row, prob_column)
            time_decay = getattr(row, time_remaining_col)
            atr = max(getattr(row, atr_col) * scenario.k_atr, eps)
            z_dist = (getattr(row, close_col) - getattr(row, open_col)) / atr
            z_range = (
                getattr(row, high_col) - getattr(row, low_col)
            ) / atr  # range normalisée

            end_boost = (1 - time_decay) ** scenario.gamma
            bias = (
                scenario.aggressiveness
                * np.tanh(scenario.alpha * z_dist + scenario.beta * z_range)
                * end_boost
            )

            target = float(np.clip(base + bias, 1e-4, 1 - 1e-4))

            if prev_odds is None:
                proposal = target
            else:
                proposal = scenario.stickiness * prev_odds + (1 - scenario.stickiness) * target
            blended = scenario.fomo_index * base + (1 - scenario.fomo_index) * proposal

            if scenario.noise > 0:
                blended += np.random.normal(0, scenario.noise)
            blended = float(np.clip(blended, 1e-4, 1 - 1e-4))
            odds.append(blended)
            prev_odds = blended
        return np.array(odds)

    for scenario in scenarios:
        column = f"odds_{scenario.name}"
        simulated[column] = np.nan
        for contract_id, group in simulated.groupby(contract_col):
            series = pd.Series(_simulate_group(group, scenario), index=group.index)
            simulated.loc[group.index, column] = series

    return simulated


def make_default_scenarios(timeframe: str) -> List[FomoScenario]:
    """Retourne un set de scénarios par défaut en fonction du timeframe."""

    if timeframe == "m15":
        return [
            FomoScenario("m15_moderate", fomo_index=0.5, aggressiveness=0.15, stickiness=0.6),
            FomoScenario("m15_aggressive", fomo_index=0.3, aggressiveness=0.25, stickiness=0.4),
            FomoScenario("m15_conservative", fomo_index=0.7, aggressiveness=0.1, stickiness=0.75),
        ]
    if timeframe == "h1":
        return [
            FomoScenario("h1_moderate", fomo_index=0.55, aggressiveness=0.12, stickiness=0.65),
            FomoScenario("h1_trend", fomo_index=0.4, aggressiveness=0.2, stickiness=0.5),
        ]
    if timeframe == "daily":
        return [
            FomoScenario("daily_slow", fomo_index=0.65, aggressiveness=0.1, stickiness=0.8),
            FomoScenario("daily_impulse", fomo_index=0.45, aggressiveness=0.18, stickiness=0.6),
        ]
    raise ValueError(f"Timeframe inconnu: {timeframe}")


__all__ = ["FomoScenario", "simulate_fomo_odds", "make_default_scenarios"]

