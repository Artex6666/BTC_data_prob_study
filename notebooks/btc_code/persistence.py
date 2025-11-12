"""Sauvegarde et chargement des modèles entraînés."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import joblib

from .model_training import ClassificationArtifacts, RegressionArtifacts
from .paths import MODEL_DIR


@dataclass
class SavedModelInfo:
    model_paths: Dict[str, str]
    feature_columns: list
    target_columns: list
    metrics_path: str


def _export_models(artifacts, prefix: str) -> SavedModelInfo:
    model_paths = {}
    for name, model in artifacts.models.items():
        model_path = MODEL_DIR / f"{prefix}_{name}.joblib"
        joblib.dump(model, model_path)
        model_paths[name] = str(model_path)

    metrics_path = MODEL_DIR / f"{prefix}_metrics.parquet"
    artifacts.metrics.to_parquet(metrics_path, index=False)

    info = SavedModelInfo(
        model_paths=model_paths,
        feature_columns=artifacts.feature_columns,
        target_columns=artifacts.target_columns,
        metrics_path=str(metrics_path),
    )

    meta_path = MODEL_DIR / f"{prefix}_meta.json"
    meta_path.write_text(json.dumps(asdict(info), indent=2))
    return info


def export_regression_artifacts(artifacts: RegressionArtifacts, prefix: str = "odds") -> SavedModelInfo:
    return _export_models(artifacts, prefix)


def export_classification_artifacts(
    artifacts: ClassificationArtifacts,
    prefix: str = "outcome",
) -> SavedModelInfo:
    return _export_models(artifacts, prefix)


def load_trained_models(meta_path: Path) -> Dict[str, object]:
    """Recharge les modèles à partir d'un fichier meta."""

    info = json.loads(Path(meta_path).read_text())
    models = {name: joblib.load(path) for name, path in info["model_paths"].items()}
    return {
        "models": models,
        "feature_columns": info["feature_columns"],
        "target_columns": info["target_columns"],
        "metrics_path": info["metrics_path"],
    }


__all__ = [
    "SavedModelInfo",
    "export_regression_artifacts",
    "export_classification_artifacts",
    "load_trained_models",
]

