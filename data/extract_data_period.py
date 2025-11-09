"""
Extraction de données minute Bitcoin depuis le dépôt bitstamp-btcusd-minute-data.
Ce script combine l'historique compressé et les mises à jour, puis filtre une
période donnée (ici, de 2023 à aujourd'hui) pour générer un CSV exploitable.
"""
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DATA_ROOT = Path("data")
HISTORICAL_PATH = DATA_ROOT / "historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
UPDATES_PATH = DATA_ROOT / "updates/btcusd_bitstamp_1min_latest.csv"

OUTPUT_FILE = Path("data/btcusd_bitstamp_1min_2023-present.csv")
START_DATE = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DATE = datetime.now(timezone.utc)


def load_source_data() -> pd.DataFrame:
    if not HISTORICAL_PATH.exists():
        print(f"[ERROR] Fichier historique manquant: {HISTORICAL_PATH}")
        sys.exit(1)

    try:
        historical_df = pd.read_csv(HISTORICAL_PATH, compression="gzip")
        print(
            f"[INFO] Historique chargé ({len(historical_df)} lignes) depuis {HISTORICAL_PATH}"
        )
    except Exception as exc:
        print(f"[ERROR] Lecture historique échouée: {exc}")
        sys.exit(1)

    if UPDATES_PATH.exists():
        try:
            updates_df = pd.read_csv(UPDATES_PATH)
            print(
                f"[INFO] Mises à jour chargées ({len(updates_df)} lignes) depuis {UPDATES_PATH}"
            )
        except Exception as exc:
            print(f"[WARN] Lecture des mises à jour échouée ({exc}), poursuite sans")
            updates_df = pd.DataFrame(columns=historical_df.columns)
    else:
        print(f"[WARN] Fichier de mises à jour absent ({UPDATES_PATH}), poursuite sans")
        updates_df = pd.DataFrame(columns=historical_df.columns)

    df = pd.concat([historical_df, updates_df], ignore_index=True)
    df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
    df.sort_values(by="timestamp", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"[INFO] Dataset combiné: {len(df)} lignes")
    return df


def filter_by_date_range(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    filtered_df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
    print(
        f"[INFO] Filtrage entre {start_date.isoformat()} et {end_date.isoformat()} -> {len(filtered_df)} lignes"
    )
    return filtered_df


def main() -> None:
    df = load_source_data()
    filtered_df = filter_by_date_range(df, START_DATE, END_DATE)

    if filtered_df.empty:
        print("[WARN] Aucune donnée trouvée pour l'intervalle demandé")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        filtered_df.to_csv(OUTPUT_FILE, index=False)
        print(f"[INFO] Sauvegarde: {OUTPUT_FILE} ({len(filtered_df)} lignes)")
    except Exception as exc:
        print(f"[ERROR] Echec lors de la sauvegarde: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

