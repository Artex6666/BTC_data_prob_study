"""Modules utilitaires pour le projet BtcUpDownStudy.

Ce package regroupe tout le code réutilisable depuis les notebooks.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Retourne la racine du projet."""
    return Path(__file__).resolve().parents[2]


__all__ = ["get_project_root"]

