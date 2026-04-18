from __future__ import annotations

import argparse
import json
from pathlib import Path

from ordresmarkets_common import OUTPUT_DIR, build_feature_frame, summarize_buy_patterns


def extract_asset(asset: str, sample_every_s: int) -> Path:
    df = build_feature_frame(asset=asset, sample_every_s=sample_every_s)
    out_csv = OUTPUT_DIR / f"features_{asset}.csv"
    df.to_csv(out_csv, index=False)

    summary = {
        "asset": asset,
        "rows": int(len(df)),
        "buy_rows": int(df["buy_label"].sum()) if not df.empty else 0,
        "markets": int(df["market_key"].nunique()) if not df.empty else 0,
        "patterns": summarize_buy_patterns(df),
    }
    out_json = OUTPUT_DIR / f"features_{asset}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{asset}: rows={summary['rows']} buys={summary['buy_rows']} markets={summary['markets']}")
    print(f"  -> {out_csv}")
    print(f"  -> {out_json}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ordresMarkets feature datasets")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=["btc", "eth"],
        choices=["btc", "eth"],
        help="Assets to process.",
    )
    parser.add_argument(
        "--sample-every-s",
        type=int,
        default=5,
        help="Negative/control sampling interval in seconds inside each contract.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for asset in args.assets:
        extract_asset(asset, sample_every_s=max(1, int(args.sample_every_s)))


if __name__ == "__main__":
    main()
