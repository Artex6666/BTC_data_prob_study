"""
Extraction de données de prix Bitcoin pour une période donnée.
Je vais récupérer les données 1 minute pour la période du 1er janvier 2024 au 30 octobre 2025.
"""
import os
import sys
from datetime import datetime, timezone

import pandas as pd


def load_merged_data() -> pd.DataFrame:
    data_dir = "data"
    bulk_path = os.path.join(
        data_dir, "historical", "btcusd_bitstamp_1min_2012-2025.csv.gz"
    )
    updates_path = os.path.join(data_dir, "updates", "btcusd_bitstamp_1min_latest.csv")

    try:
        bulk_df = pd.read_csv(bulk_path, compression="gzip")
    except Exception as e:
        print(f"✗ Erreur lors du chargement: {e}")
        sys.exit(1)

    try:
        if os.path.exists(updates_path):
            updated_df = pd.read_csv(updates_path)
        else:
            updated_df = pd.DataFrame(columns=bulk_df.columns)
    except Exception as e:
        updated_df = pd.DataFrame(columns=bulk_df.columns)

    df = pd.concat([bulk_df, updated_df], ignore_index=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df = df.sort_values(by="timestamp", ascending=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def filter_by_date_range(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    filtered_df = df[
        (df["timestamp"] >= start_timestamp) & (df["timestamp"] <= end_timestamp)
    ].copy()

    return filtered_df


def main() -> None:
    start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime(2025, 10, 30, 23, 59, 59, tzinfo=timezone.utc)

    df = load_merged_data()
    filtered_df = filter_by_date_range(df, start_date, end_date)

    if len(filtered_df) == 0:
        print("✗ Aucune donnée trouvée pour cette période")
        sys.exit(1)

    output_dir = "data"
    output_file = os.path.join(
        output_dir, "btcusd_bitstamp_1min_2024-10-2025.csv"
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        filtered_df.to_csv(output_file, index=False)
        print(f"✓ Extrait {len(filtered_df)} points de données → {output_file}")
    except Exception as e:
        print(f"✗ Erreur de sauvegarde: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

