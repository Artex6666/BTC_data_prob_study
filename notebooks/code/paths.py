"""Gestion centralisée des chemins du projet."""

from pathlib import Path

from . import get_project_root


ROOT_DIR: Path = get_project_root()
DATA_DIR: Path = ROOT_DIR / "data"
MODEL_DIR: Path = ROOT_DIR / "models"
CACHE_DIR: Path = ROOT_DIR / "notebooks" / "cache"

for directory in (MODEL_DIR, CACHE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


__all__ = ["ROOT_DIR", "DATA_DIR", "MODEL_DIR", "CACHE_DIR"]

