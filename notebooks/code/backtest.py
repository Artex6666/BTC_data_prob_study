"""Backtests des stratégies basées sur les probabilités prédites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class BacktestParams:
    timeframe: str
    prediction_col: str
    market_prob_col: str
    outcome_col: str
    price_up_col: str
    price_down_col: str
    threshold: float = 0.05
    initial_capital: float = 1_000.0
    capital_risk_fraction: float = 0.02
    share_fraction: float = 0.04


@dataclass
class BacktestResults:
    trades: pd.DataFrame
    equity_fractional: pd.Series
    equity_share: pd.Series
    summary: pd.DataFrame


def _simulate_trade_outcome(
    bet_side: str,
    price_up: float,
    price_down: float,
    outcome_up: int,
) -> float:
    """Retourne le payoff par part en fonction du résultat."""

    if bet_side == "up":
        payoff = 1.0 if outcome_up == 1 else 0.0
        price = price_up
    else:
        payoff = 1.0 if outcome_up == 0 else 0.0
        price = price_down
    return payoff - price


def run_backtest(params: BacktestParams, dataset: pd.DataFrame) -> BacktestResults:
    """Exécute le backtest sur le dataset donné."""

    trades = []
    capital_fractional = params.initial_capital
    capital_share = params.initial_capital
    share_size = params.initial_capital * params.share_fraction

    equity_fractional = []
    equity_share = []

    for row in dataset.itertuples(index=False):
        pred = getattr(row, params.prediction_col)
        market = getattr(row, params.market_prob_col)
        price_up = getattr(row, params.price_up_col)
        price_down = getattr(row, params.price_down_col)
        outcome_up = getattr(row, params.outcome_col)

        edge_up = pred - market
        edge_down = (1 - pred) - (1 - market)

        if edge_up > params.threshold:
            side = "up"
        elif edge_down > params.threshold:
            side = "down"
        else:
            equity_fractional.append(capital_fractional)
            equity_share.append(capital_share)
            continue

        payoff_per_share = _simulate_trade_outcome(side, price_up, price_down, outcome_up)

        # Stratégie capital fractionnel
        capital_to_risk = capital_fractional * params.capital_risk_fraction
        if side == "up":
            shares_fractional = capital_to_risk / max(price_up, 1e-6)
        else:
            shares_fractional = capital_to_risk / max(price_down, 1e-6)
        pnl_fractional = shares_fractional * payoff_per_share
        capital_fractional += pnl_fractional

        # Stratégie nombre de parts fixe
        shares_fixed = share_size
        pnl_share = shares_fixed * payoff_per_share
        capital_share += pnl_share

        trades.append(
            {
                "timestamp": getattr(row, "timestamp"),
                "side": side,
                "pred": pred,
                "market_prob": market,
                "edge": edge_up if side == "up" else edge_down,
                "price_up": price_up,
                "price_down": price_down,
                "payoff_per_share": payoff_per_share,
                "pnl_fractional": pnl_fractional,
                "pnl_share": pnl_share,
                "capital_fractional": capital_fractional,
                "capital_share": capital_share,
                "outcome_up": outcome_up,
            }
        )

        equity_fractional.append(capital_fractional)
        equity_share.append(capital_share)

    trades_df = pd.DataFrame(trades)
    equity_frac_series = pd.Series(equity_fractional, name="equity_fractional")
    equity_share_series = pd.Series(equity_share, name="equity_share")

    if not trades_df.empty:
        winrate_fractional = (trades_df["pnl_fractional"] > 0).mean()
        winrate_share = (trades_df["pnl_share"] > 0).mean()
        summary = pd.DataFrame(
            [
                {
                    "strategy": "fractional",
                    "trades": len(trades_df),
                    "winrate": winrate_fractional,
                    "total_pnl": trades_df["pnl_fractional"].sum(),
                    "final_equity": capital_fractional,
                },
                {
                    "strategy": "share_fixed",
                    "trades": len(trades_df),
                    "winrate": winrate_share,
                    "total_pnl": trades_df["pnl_share"].sum(),
                    "final_equity": capital_share,
                },
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "strategy": "fractional",
                    "trades": 0,
                    "winrate": np.nan,
                    "total_pnl": 0.0,
                    "final_equity": capital_fractional,
                },
                {
                    "strategy": "share_fixed",
                    "trades": 0,
                    "winrate": np.nan,
                    "total_pnl": 0.0,
                    "final_equity": capital_share,
                },
            ]
        )

    return BacktestResults(
        trades=trades_df,
        equity_fractional=equity_frac_series,
        equity_share=equity_share_series,
        summary=summary,
    )


__all__ = ["BacktestParams", "BacktestResults", "run_backtest"]

