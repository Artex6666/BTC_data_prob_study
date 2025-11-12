"""Entraînement et évaluation des modèles ML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FeatureConfig, build_feature_matrix
from .timeframe_utils import Timeframe, TIMEFRAME_SPECS, assign_contracts


POLY_COLUMNS = {
    "m15": ("m15_buy", "m15_sell", "m15_spread_up", "m15_spread_down"),
    "h1": ("h1_buy", "h1_sell", "h1_spread_up", "h1_spread_down"),
    "daily": ("daily_buy", "daily_sell", "daily_spread_up", "daily_spread_down"),
}


def compute_market_probabilities(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Calcule les probabilités implicites à partir des cotations."""

    buy_col, sell_col, spread_up_col, spread_down_col = POLY_COLUMNS[timeframe]
    ask_up = df[buy_col].astype(float)
    ask_down = df[sell_col].astype(float)
    spread_up = df[spread_up_col].astype(float)
    spread_down = df[spread_down_col].astype(float)

    bid_up = np.clip(ask_up - spread_up, 0, 1)
    bid_down = np.clip(ask_down - spread_down, 0, 1)

    mid_up = (ask_up + bid_up) / 2
    mid_down = (ask_down + bid_down) / 2

    total = mid_up + mid_down
    total = np.where(total == 0, 1, total)
    prob_up = (mid_up / total).clip(1e-4, 1 - 1e-4)
    prob_down = (mid_down / total).clip(1e-4, 1 - 1e-4)

    df[f"{timeframe}_prob_up_market"] = prob_up
    df[f"{timeframe}_prob_down_market"] = prob_down
    df[f"{timeframe}_price_up_ask"] = ask_up
    df[f"{timeframe}_price_down_ask"] = ask_down
    df[f"{timeframe}_price_up_bid"] = bid_up
    df[f"{timeframe}_price_down_bid"] = bid_down
    df[f"{timeframe}_price_up_mid"] = mid_up
    df[f"{timeframe}_price_down_mid"] = mid_down
    df[f"{timeframe}_spread_up"] = spread_up
    df[f"{timeframe}_spread_down"] = spread_down
    return df


def prepare_feature_set(
    df: pd.DataFrame,
    feature_config: FeatureConfig,
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Construit la matrice de caractéristiques prête pour l'entraînement."""

    feat_df = build_feature_matrix(df, feature_config, dropna=dropna)
    feature_cols = [
        col
        for col in feat_df.columns
        if col not in {"timestamp", "open", "high", "low", "close"}
    ]
    return feat_df, feature_cols


@dataclass
class RegressionArtifacts:
    models: Dict[str, Pipeline]
    feature_columns: List[str]
    target_columns: List[str]
    metrics: pd.DataFrame


def train_odds_regressors(
    dataset: pd.DataFrame,
    feature_cols: Iterable[str],
    targets: Iterable[str],
    test_size: float = 0.2,
    random_state: int = 17,
) -> RegressionArtifacts:
    """Entraîne un régresseur par cible pour estimer les cotes."""

    X = dataset[list(feature_cols)].values
    models: Dict[str, Pipeline] = {}
    metrics = []

    X_train, X_test, _, _ = train_test_split(
        X,
        dataset[list(targets)],
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    for target in targets:
        y = dataset[target].values
        X_tr, X_ts, y_tr, y_ts = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", HistGradientBoostingRegressor(random_state=random_state)),
            ]
        )
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_ts)
        mae = mean_absolute_error(y_ts, y_pred)
        rmse = mean_squared_error(y_ts, y_pred, squared=False)
        metrics.append({"target": target, "mae": mae, "rmse": rmse})
        models[target] = pipeline

    metrics_df = pd.DataFrame(metrics)
    return RegressionArtifacts(
        models=models,
        feature_columns=list(feature_cols),
        target_columns=list(targets),
        metrics=metrics_df,
    )


@dataclass
class ClassificationArtifacts:
    models: Dict[str, Pipeline]
    feature_columns: List[str]
    target_columns: List[str]
    metrics: pd.DataFrame


def train_outcome_classifiers(
    dataset: pd.DataFrame,
    feature_cols: Iterable[str],
    targets: Dict[str, str],
    test_size: float = 0.3,
    random_state: int = 42,
) -> ClassificationArtifacts:
    """Entraîne des classifieurs probabilistes pour les sens de clôture."""

    models: Dict[str, Pipeline] = {}
    metrics = []

    X = dataset[list(feature_cols)].values

    for name, target_col in targets.items():
        y = dataset[target_col].values
        mask = ~np.isnan(y)
        X_valid = X[mask]
        y_valid = y[mask]

        X_tr, X_ts, y_tr, y_ts = train_test_split(
            X_valid,
            y_valid,
            test_size=test_size,
            random_state=random_state,
            stratify=y_valid,
        )

        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", HistGradientBoostingClassifier(random_state=random_state)),
            ]
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_ts)[:, 1]
        auc = roc_auc_score(y_ts, proba)
        acc = accuracy_score(y_ts, (proba >= 0.5).astype(int))
        brier = mean_squared_error(y_ts, proba)
        metrics.append({"target": target_col, "auc": auc, "accuracy": acc, "brier": brier})
        models[name] = clf

    metrics_df = pd.DataFrame(metrics)
    return ClassificationArtifacts(
        models=models,
        feature_columns=list(feature_cols),
        target_columns=list(targets.values()),
        metrics=metrics_df,
    )


def predict_regressions(
    artifacts: RegressionArtifacts,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Prédit les cibles de régression."""

    results = dataset.copy()
    X = dataset[artifacts.feature_columns]

    for target, model in artifacts.models.items():
        results[f"pred_{target}"] = model.predict(X)

    return results


def predict_classifications(
    artifacts: ClassificationArtifacts,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Prédit les probabilités (classe 1) pour les classifieurs."""

    results = dataset.copy()
    X = dataset[artifacts.feature_columns]

    for name, model in artifacts.models.items():
        proba = model.predict_proba(X)[:, 1]
        results[f"pred_proba_{name}"] = proba

    return results


__all__ = [
    "compute_market_probabilities",
    "prepare_feature_set",
    "train_odds_regressors",
    "RegressionArtifacts",
    "train_outcome_classifiers",
    "ClassificationArtifacts",
    "predict_regressions",
    "predict_classifications",
]

