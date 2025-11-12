"""Exemple d'utilisation des modèles entraînés pour un flux live."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .persistence import load_trained_models
from .pipeline import (
    build_classification_dataset,
    enrich_polymarket_with_features,
    load_all_data,
    prepare_ohlc_features,
    prepare_timeframe_tables,
)
from .timeframe_utils import Timeframe


def run_live_inference(
    meta_path: Path,
    timeframe: Timeframe = "m15",
    last_seconds: int = 900,
) -> pd.DataFrame:
    """Charge les modèles exportés et produit des probabilités live."""

    bundle = load_trained_models(meta_path)
    model = bundle["models"].get(timeframe)
    if model is None:
        raise ValueError(f"Aucun modèle trouvé pour le timeframe {timeframe}.")

    polymarket, ohlc = load_all_data()
    ohlc_features = prepare_ohlc_features(ohlc)
    recent = polymarket.tail(last_seconds)
    enriched = enrich_polymarket_with_features(recent, ohlc_features)
    timeframe_df = prepare_timeframe_tables(enriched, timeframe)
    dataset = build_classification_dataset(timeframe_df, timeframe)

    X = dataset[bundle["feature_columns"]]
    probabilities = model.predict_proba(X)[:, 1]
    dataset["predicted_prob_up"] = probabilities
    dataset = dataset.assign(
        edge=dataset["predicted_prob_up"] - timeframe_df.loc[dataset.index, f"{timeframe}_prob_up_market"]
    )
    return dataset[["timestamp", "predicted_prob_up", "edge"]].tail(50)


if __name__ == "__main__":
    default_meta = Path(__file__).resolve().parents[2] / "models" / "outcome_meta.json"
    if default_meta.exists():
        preds = run_live_inference(default_meta)
        print(preds.tail(10))
    else:
        print("Aucun modèle exporté trouvé. Entraînez et exportez d'abord le modèle outcome.")

