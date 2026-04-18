from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from chart_utils import TIMEFRAMES, attach_settlement_outcomes, load_contracts

BASE = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE / "Datas" / "report"
LIVE_DIR = BASE / "reportLive" / "safeChase"
ARCHIVE_DIR = BASE / "Datas" / "csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "ordresmarkets_strategy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSET_BY_LABEL = {
    "Bitcoin": "btc",
    "Ethereum": "eth",
}

TF_INFO = {
    "m5": {
        "tf_floor": "5min",
        "minutes": 5,
        "bid_up_col": "m5_up_bid",
        "bid_down_col": "m5_down_bid",
        "ask_up_col": "m5_up_ask",
        "ask_down_col": "m5_down_ask",
    },
    "m15": {
        "tf_floor": "15min",
        "minutes": 15,
        "bid_up_col": "m15_up_bid",
        "bid_down_col": "m15_down_bid",
        "ask_up_col": "m15_up_ask",
        "ask_down_col": "m15_down_ask",
    },
    "h1": {
        "tf_floor": "1h",
        "minutes": 60,
        "bid_up_col": "h1_up_bid",
        "bid_down_col": "h1_down_bid",
        "ask_up_col": "h1_up_ask",
        "ask_down_col": "h1_down_ask",
    },
}

MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

MARKET_RE = re.compile(
    r"^(Bitcoin|Ethereum)_Up_or_Down_-_([A-Za-z]+)_(\d+)_([0-9]+(?:AM|PM))(?:-([0-9]+(?:AM|PM)))?_ET$"
)
TRADE_LINE_RE = re.compile(r"^\s*\d+\s*\|")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class MarketMeta:
    asset: str
    report_path: Path
    market_folder: str
    market_title: str
    resolution: str
    open_ts: pd.Timestamp
    ce_ts: pd.Timestamp
    tf_slug: str
    tf_floor: str
    duration_s: int
    report_pnl: float
    report_spent: float
    remaining_yes: float
    remaining_no: float

    @property
    def market_key(self) -> str:
        return f"{self.asset}:{self.tf_slug}:{int(self.open_ts.timestamp())}"


def _parse_ampm(token: str) -> tuple[int, int]:
    match = re.match(r"(\d{1,2})(?:(\d{2}))?(AM|PM)", token)
    if not match:
        raise ValueError(f"Invalid AM/PM token: {token}")
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ap = match.group(3)
    if ap == "AM":
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour, minute


def _parse_market_folder(folder: str) -> tuple[str, pd.Timestamp, pd.Timestamp, str]:
    match = MARKET_RE.match(folder)
    if not match:
        raise ValueError(f"Unsupported market folder: {folder}")
    asset_label, month_name, day_s, start_token, end_token = match.groups()
    hour, minute = _parse_ampm(start_token)
    start_et = pd.Timestamp(
        year=2026,
        month=MONTHS[month_name],
        day=int(day_s),
        hour=hour,
        minute=minute,
    )
    if end_token:
        end_hour, end_minute = _parse_ampm(end_token)
        end_et = pd.Timestamp(
            year=2026,
            month=MONTHS[month_name],
            day=int(day_s),
            hour=end_hour,
            minute=end_minute,
        )
    else:
        end_et = start_et + pd.Timedelta(hours=1)
    start_utc = (start_et + pd.Timedelta(hours=4)).tz_localize("UTC")
    end_utc = (end_et + pd.Timedelta(hours=4)).tz_localize("UTC")
    duration_min = int((end_utc - start_utc).total_seconds() // 60)
    tf_slug = {5: "m5", 15: "m15", 60: "h1"}.get(duration_min)
    if tf_slug is None:
        raise ValueError(f"Unsupported duration: {duration_min} min")
    return ASSET_BY_LABEL[asset_label], start_utc, end_utc, tf_slug


def _parse_report_summary(lines: list[str]) -> tuple[str, str, float, float, float, float]:
    title = ""
    resolution = ""
    pnl = spent = remaining_yes = remaining_no = 0.0
    for line in lines:
        if line.startswith("MARKET:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("RESOLUTION:"):
            resolution = line.split(":", 1)[1].strip()
        elif line.startswith("FINAL PNL:"):
            pnl = float(NUMBER_RE.search(line).group())
        elif line.startswith("Total spent (net exposure):"):
            spent = float(NUMBER_RE.search(line).group())
        elif line.startswith("Remaining YES shares:"):
            remaining_yes = float(NUMBER_RE.search(line).group())
        elif line.startswith("Remaining NO shares:"):
            remaining_no = float(NUMBER_RE.search(line).group())
    return title, resolution, pnl, spent, remaining_yes, remaining_no


def parse_report_buys(report_path: Path) -> list[dict]:
    lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
    buys: list[dict] = []
    in_table = False
    for line in lines:
        if line.startswith("Idx | Time"):
            in_table = True
            continue
        if not in_table or not TRADE_LINE_RE.match(line):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 7 or parts[2] != "Buy":
            continue
        ts = pd.Timestamp(parts[1]).tz_localize("Europe/Paris").tz_convert("UTC")
        buys.append(
            {
                "timestamp": ts,
                "side": parts[3],
                "price_c": float(parts[4]),
                "shares": float(parts[5]),
                "cost_usd": float(NUMBER_RE.search(parts[6]).group()),
            }
        )
    return buys


def list_market_reports(asset: str | None = None, label: str = "ordresMarkets") -> list[MarketMeta]:
    out: list[MarketMeta] = []
    pattern = f"**/report-{label}.txt"
    for report_path in sorted(REPORT_DIR.glob(pattern)):
        try:
            asset_slug, open_ts, ce_ts, tf_slug = _parse_market_folder(report_path.parent.name)
        except ValueError:
            continue
        if asset is not None and asset_slug != asset:
            continue
        lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
        title, resolution, pnl, spent, rem_yes, rem_no = _parse_report_summary(lines)
        out.append(
            MarketMeta(
                asset=asset_slug,
                report_path=report_path,
                market_folder=report_path.parent.name,
                market_title=title,
                resolution=resolution,
                open_ts=open_ts,
                ce_ts=ce_ts,
                tf_slug=tf_slug,
                tf_floor=TF_INFO[tf_slug]["tf_floor"],
                duration_s=int((ce_ts - open_ts).total_seconds()),
                report_pnl=pnl,
                report_spent=spent,
                remaining_yes=rem_yes,
                remaining_no=rem_no,
            )
        )
    return out


@lru_cache(maxsize=8)
def load_asset_ticks(asset: str) -> pd.DataFrame:
    symbol = asset.upper()
    paths: list[Path] = []
    archive = ARCHIVE_DIR / f"{symbol}.csv"
    live = LIVE_DIR / f"{symbol}.csv"
    if archive.exists():
        paths.append(archive)
    if live.exists():
        paths.append(live)
    if not paths:
        raise FileNotFoundError(f"No CSV found for asset {asset}")

    cols = ["timestamp", "spot_price"]
    for info in TF_INFO.values():
        cols.extend(
            [
                info["ask_up_col"],
                info["bid_up_col"],
                info["ask_down_col"],
                info["bid_down_col"],
            ]
        )

    frames = [pd.read_csv(path, usecols=cols, parse_dates=["timestamp"]) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["ts_ns"] = df["timestamp"].astype("int64")
    return df


@lru_cache(maxsize=8)
def load_full_contract_map(asset: str) -> dict[str, dict]:
    symbol = asset.upper()
    paths: list[str] = []
    archive = ARCHIVE_DIR / f"{symbol}.csv"
    live = LIVE_DIR / f"{symbol}.csv"
    if archive.exists():
        paths.append(str(archive))
    if live.exists():
        paths.append(str(live))
    out: dict[str, dict] = {}
    for tf_floor, bid_up, bid_down, ask_up, ask_down in TIMEFRAMES:
        contracts = load_contracts(
            paths,
            tf_floor,
            bid_up,
            bid_down,
            ask_up,
            ask_down,
            window_s=None,
        )
        attach_settlement_outcomes(contracts, asset, tf=tf_floor)
        tf_slug = {"5min": "m5", "15min": "m15", "1h": "h1"}[tf_floor]
        for c in contracts:
            key = f"{asset}:{tf_slug}:{int(c['open_ts'])}"
            out[key] = c
    return out


@lru_cache(maxsize=8)
def build_report_contract_map(asset: str, label: str = "ordresMarkets") -> dict[str, dict]:
    df = load_asset_ticks(asset)
    all_ns = df["ts_ns"].to_numpy()
    out: dict[str, dict] = {}
    for meta in list_market_reports(asset=asset, label=label):
        start_idx = np.searchsorted(all_ns, meta.open_ts.value, side="right") - 1
        end_idx = np.searchsorted(all_ns, meta.ce_ts.value, side="right") - 1
        if start_idx < 0 or end_idx <= start_idx:
            continue
        contract_df = df.iloc[start_idx : end_idx + 1].copy()
        if len(contract_df) < 2:
            continue
        info = TF_INFO[meta.tf_slug]
        out[meta.market_key] = {
            "ce": meta.ce_ts.tz_convert(None),
            "ce_ts": meta.ce_ts.timestamp(),
            "open_ts": meta.open_ts.timestamp(),
            "op": float(contract_df["spot_price"].iloc[0]),
            "spot": contract_df["spot_price"].to_numpy(dtype=np.float64),
            "ts": contract_df["timestamp"].dt.tz_convert(None).to_numpy(),
            "up_bid": contract_df[info["bid_up_col"]].to_numpy(dtype=np.float64),
            "down_bid": contract_df[info["bid_down_col"]].to_numpy(dtype=np.float64),
            "up_ask": contract_df[info["ask_up_col"]].to_numpy(dtype=np.float64),
            "down_ask": contract_df[info["ask_down_col"]].to_numpy(dtype=np.float64),
            "_settlement_won_up": meta.resolution == "YES",
        }
    return out


def build_feature_frame(
    asset: str,
    label: str = "ordresMarkets",
    sample_every_s: int = 5,
) -> pd.DataFrame:
    df = load_asset_ticks(asset)
    all_ns = df["ts_ns"].to_numpy()
    min_ts = df["timestamp"].iloc[0]
    max_ts = df["timestamp"].iloc[-1]
    metas = list_market_reports(asset=asset, label=label)
    rows: list[dict] = []

    for meta in metas:
        if meta.open_ts < min_ts or meta.ce_ts > max_ts:
            continue

        start_idx = np.searchsorted(all_ns, meta.open_ts.value, side="right") - 1
        end_idx = np.searchsorted(all_ns, meta.ce_ts.value, side="right") - 1
        if start_idx < 0 or end_idx <= start_idx:
            continue

        contract_df = df.iloc[start_idx : end_idx + 1].copy()
        contract_ns = contract_df["ts_ns"].to_numpy()
        spots = contract_df["spot_price"].to_numpy(dtype=float)
        if len(spots) < 3:
            continue
        open_spot = float(spots[0])

        side_buy_map: dict[tuple[int, str], dict[str, float]] = {}
        for buy in parse_report_buys(meta.report_path):
            j = np.searchsorted(contract_ns, buy["timestamp"].value, side="right") - 1
            if j < 0 or j >= len(contract_df):
                continue
            key = (j, buy["side"])
            agg = side_buy_map.setdefault(
                key,
                {"buy_count": 0, "buy_cost_usd": 0.0, "buy_shares": 0.0, "buy_price_c_sum": 0.0},
            )
            agg["buy_count"] += 1
            agg["buy_cost_usd"] += buy["cost_usd"]
            agg["buy_shares"] += buy["shares"]
            agg["buy_price_c_sum"] += buy["price_c"]

        if not side_buy_map:
            continue

        ts_sec = (contract_ns - contract_ns[0]) / 1e9
        sample_mask = np.zeros(len(contract_df), dtype=bool)
        sample_mask[0] = True
        last_sample_t = ts_sec[0]
        for j in range(1, len(contract_df)):
            if ts_sec[j] - last_sample_t >= sample_every_s:
                sample_mask[j] = True
                last_sample_t = ts_sec[j]
        for j, _side in side_buy_map:
            sample_mask[j] = True

        above = spots > open_spot
        below = spots < open_spot
        cross_up = np.zeros(len(contract_df), dtype=bool)
        cross_down = np.zeros(len(contract_df), dtype=bool)
        cross_up[1:] = below[:-1] & above[1:]
        cross_down[1:] = above[:-1] & below[1:]
        recross_up = np.cumsum(cross_up) > 0
        recross_down = np.cumsum(cross_down) > 0

        idx_10 = np.searchsorted(ts_sec, ts_sec - 10.0, side="left")
        idx_30 = np.searchsorted(ts_sec, ts_sec - 30.0, side="left")
        move_10_pct = (spots - spots[idx_10]) / open_spot * 100.0
        move_30_pct = (spots - spots[idx_30]) / open_spot * 100.0

        info = TF_INFO[meta.tf_slug]
        up_ask = contract_df[info["ask_up_col"]].to_numpy(dtype=float)
        down_ask = contract_df[info["ask_down_col"]].to_numpy(dtype=float)
        up_bid = contract_df[info["bid_up_col"]].to_numpy(dtype=float)
        down_bid = contract_df[info["bid_down_col"]].to_numpy(dtype=float)
        side_won = meta.resolution == "YES"

        for j in np.flatnonzero(sample_mask):
            elapsed_s = float((contract_df.iloc[j]["timestamp"] - meta.open_ts).total_seconds())
            remain_s = float((meta.ce_ts - contract_df.iloc[j]["timestamp"]).total_seconds())
            if remain_s <= 0:
                continue
            for side in ("Up", "Down"):
                ask = up_ask[j] if side == "Up" else down_ask[j]
                bid = up_bid[j] if side == "Up" else down_bid[j]
                if not np.isfinite(ask) or ask <= 0 or not np.isfinite(bid):
                    continue

                signed_move_pct = move_30_pct[j] if side == "Up" else -move_30_pct[j]
                signed_open_pct = ((spots[j] - open_spot) / open_spot * 100.0)
                if side == "Down":
                    signed_open_pct = -signed_open_pct

                if side == "Up":
                    recross = bool(recross_up[j])
                    side_hit = side_won
                else:
                    recross = bool(recross_down[j])
                    side_hit = not side_won

                hold_pnl_per_usd = (1.0 / ask - 1.0) if side_hit else -1.0
                buy_stats = side_buy_map.get((j, side), None)
                rows.append(
                    {
                        "asset": asset,
                        "market_key": meta.market_key,
                        "market_title": meta.market_title,
                        "market_folder": meta.market_folder,
                        "report_path": str(meta.report_path),
                        "tf_slug": meta.tf_slug,
                        "tf_floor": meta.tf_floor,
                        "side": side,
                        "contract_open_ts": int(meta.open_ts.timestamp()),
                        "contract_ce_ts": int(meta.ce_ts.timestamp()),
                        "contract_j": int(j),
                        "timestamp": contract_df.iloc[j]["timestamp"].isoformat(),
                        "elapsed_s": elapsed_s,
                        "elapsed_frac": elapsed_s / max(meta.duration_s, 1),
                        "remain_s": remain_s,
                        "spot": float(spots[j]),
                        "open_spot": open_spot,
                        "signed_open_pct": signed_open_pct,
                        "signed_move_10s_pct": move_10_pct[j] if side == "Up" else -move_10_pct[j],
                        "signed_move_30s_pct": signed_move_pct,
                        "aligned": int(signed_open_pct > 0),
                        "recross": int(recross),
                        "ask_c": ask * 100.0,
                        "bid_c": bid * 100.0,
                        "spread_c": (ask - bid) * 100.0,
                        "sum_ask_c": (up_ask[j] + down_ask[j]) * 100.0,
                        "side_won": int(side_hit),
                        "hold_pnl_per_usd": hold_pnl_per_usd,
                        "buy_label": int(buy_stats is not None),
                        "buy_count": int(buy_stats["buy_count"]) if buy_stats else 0,
                        "buy_cost_usd": float(buy_stats["buy_cost_usd"]) if buy_stats else 0.0,
                        "buy_shares": float(buy_stats["buy_shares"]) if buy_stats else 0.0,
                        "buy_avg_price_c": (
                            float(buy_stats["buy_price_c_sum"]) / float(buy_stats["buy_count"])
                            if buy_stats
                            else np.nan
                        ),
                    }
                )

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.sort_values(["market_key", "timestamp", "side"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def summarize_buy_patterns(df: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    if df.empty:
        return out
    for keys, sub in df.groupby(["asset", "tf_slug", "side"]):
        buys = sub[sub["buy_label"] == 1]
        if buys.empty:
            continue
        weights = buys["buy_cost_usd"].to_numpy()
        wsum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0
        out["|".join(keys)] = {
            "n_buys": int(len(buys)),
            "weighted_aligned": float(np.dot(buys["aligned"], weights) / wsum),
            "weighted_recross": float(np.dot(buys["recross"], weights) / wsum),
            "median_signed_open_pct": float(buys["signed_open_pct"].median()),
            "median_signed_move_30s_pct": float(buys["signed_move_30s_pct"].median()),
            "median_ask_c": float(buys["ask_c"].median()),
            "median_elapsed_frac": float(buys["elapsed_frac"].median()),
            "median_buy_cost_usd": float(buys["buy_cost_usd"].median()),
        }
    return out
