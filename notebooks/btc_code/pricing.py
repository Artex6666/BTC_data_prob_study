"""Conversion entre probabilités et prix de parts Polymarket."""

from __future__ import annotations

import numpy as np
import pandas as pd


def probabilities_to_prices(
    prob_up: pd.Series,
    spread_up: float,
    spread_down: float,
) -> pd.DataFrame:
    """Convertit des probabilités en prix ask pour Up/Down."""

    prob_up = prob_up.clip(1e-4, 1 - 1e-4)
    prob_down = (1 - prob_up).clip(1e-4, 1 - 1e-4)

    ask_up = np.clip(prob_up + spread_up / 2, 1e-4, 1 - 1e-4)
    bid_up = np.clip(ask_up - spread_up, 1e-4, 1 - 1e-4)

    ask_down = np.clip(prob_down + spread_down / 2, 1e-4, 1 - 1e-4)
    bid_down = np.clip(ask_down - spread_down, 1e-4, 1 - 1e-4)

    return pd.DataFrame(
        {
            "price_up_ask": ask_up,
            "price_up_bid": bid_up,
            "price_down_ask": ask_down,
            "price_down_bid": bid_down,
            "prob_up_market": prob_up,
            "prob_down_market": prob_down,
        }
    )


__all__ = ["probabilities_to_prices"]

