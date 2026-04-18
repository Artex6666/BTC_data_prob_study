"""
Pour chaque contrat M5 : compare issue perdant/gagnant (backtest CSV vs live JSONL).
Même logique que run_safechase_default.py pour le backtest.
Live : perdant si expected_pnl < 0 (fenêtre avec position ; sinon ignoré pour l'alignement).
"""
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
JSONL_BTC = BASE / "reportLive" / "safeChase" / "60_lb0.5h" / "btc" / "m5"
JSONL_ETH = BASE / "reportLive" / "safeChase" / "60_lb0.5h" / "eth" / "m5"
CSV_BTC = BASE / "reportLive" / "safeChase" / "60_lb0.5h" / "BTC.csv"
CSV_ETH = BASE / "reportLive" / "safeChase" / "60_lb0.5h" / "ETH.csv"

MIN_BID_FLOOR = 0.65
MAX_FILLS = 3
POS_SIZE = 50.0
SAFETY_INTERCEPT_BTC = 0.5
SAFETY_INTERCEPT_ETH = 1.0
SAFETY_SLOPE_BTC = 2.0
SAFETY_SLOPE_ETH = 0.05
BTC_RANGE_LB = "30min"
BTC_RANGE_THRESH = 60.0


def _btc_contracts(df):
    df = df.copy()
    df["ce"] = df["timestamp"].dt.floor("5min") + pd.Timedelta(minutes=5)
    contracts = []
    for ce in df["ce"].unique():
        c_df = df[(df["ce"] == ce) & (df["timestamp"] >= (ce - pd.Timedelta(seconds=60)))]
        if len(c_df) < 5:
            continue
        op = df[df["ce"] == ce]["spot_price"].iloc[0]
        last_tick = c_df.iloc[-1]
        won_up = last_tick["m5_up_bid"] > last_tick["m5_down_bid"]
        contracts.append({
            "op": op,
            "won_up": won_up,
            "spots": c_df["spot_price"].values,
            "ups": c_df["m5_up_bid"].values,
            "dns": c_df["m5_down_bid"].values,
            "ts": c_df["timestamp"].values,
            "r30": c_df["spot_30m_range"].values,
            "ce": ce,
        })
    return contracts


def _eth_contracts(df):
    df = df.copy()
    df["ce"] = df["timestamp"].dt.floor("5min") + pd.Timedelta(minutes=5)
    contracts = []
    for ce in sorted(df["ce"].unique()):
        full_df = df[df["ce"] == ce].sort_values("timestamp")
        if len(full_df) < 2:
            continue
        op = full_df["spot_price"].iloc[0]
        last_tick = full_df.iloc[-1]
        won_up = last_tick["m5_up_bid"] > last_tick["m5_down_bid"]
        t60_df = full_df[full_df["timestamp"] >= (ce - pd.Timedelta(seconds=60))]
        if t60_df.empty:
            continue
        contracts.append({
            "ce": ce,
            "op": op,
            "won_up": won_up,
            "spots": t60_df["spot_price"].values,
            "ups": t60_df["m5_up_bid"].values,
            "dns": t60_df["m5_down_bid"].values,
            "ts": t60_df["timestamp"].values,
        })
    return contracts


def simulate_one(c, slope, intercept, pos_size, require_btc_range):
    activated_side = None
    f_count = 0
    our_limit = None
    for j in range(len(c["ts"])):
        s = c["spots"][j]
        if require_btc_range:
            r30 = float(c["r30"][j])
            if np.isnan(r30) or r30 < BTC_RANGE_THRESH:
                continue
        remain = (c["ce"] - c["ts"][j]) / np.timedelta64(1, "s")
        if remain <= 0:
            break
        thresh = (slope * remain) + float(intercept)
        if activated_side:
            buf = (s - c["op"]) if activated_side == "UP" else (c["op"] - s)
            if buf < thresh:
                our_limit = None
                activated_side = None
        if not activated_side:
            if s - c["op"] >= thresh:
                activated_side = "UP"
            elif c["op"] - s >= thresh:
                activated_side = "DOWN"
        if not activated_side:
            continue
        bid = round(float(c["ups"][j] if activated_side == "UP" else c["dns"][j]), 2)
        if our_limit is not None and bid <= round(our_limit - 0.01, 2):
            f_count += 1
            our_limit = None
            if f_count >= MAX_FILLS:
                break
        if bid >= MIN_BID_FLOOR:
            proposed = min(bid, 0.99)
            if our_limit is None or proposed > our_limit:
                our_limit = proposed
    if f_count == 0:
        return None
    won = c["won_up"] == (activated_side == "UP")
    return {"fills": f_count, "won": won, "lost": not won}


def parse_jsonl(path):
    with open(path, encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    wo = next((e for e in events if e.get("event") == "window_open"), None)
    we = next((e for e in events if e.get("event") in ("window_ended", "window_settled")), None)
    if not wo or not we:
        return None
    ts = pd.Timestamp(wo["ts"].replace("Z", "+00:00")).tz_convert(None)
    ce = ts.floor("5min") + pd.Timedelta(minutes=5)
    pnl = float(we.get("expected_pnl", 0) or 0)
    uc = float(we.get("up_cost", 0) or 0)
    dc = float(we.get("down_cost", 0) or 0)
    us = float(we.get("up_shares", 0) or 0)
    ds = float(we.get("down_shares", 0) or 0)
    has_pos = (uc + dc) > 1e-9 or (us + ds) > 1e-9
    live_loss = pnl < 0
    live_win = pnl > 0
    return {"ce": ce, "expected_pnl": pnl, "has_pos": has_pos, "live_loss": live_loss, "live_win": live_win}


def run_for_asset(label, csv_path, jsonl_dir, require_btc_range, slope, intercept):
    cols = ["timestamp", "spot_price", "m5_up_bid", "m5_down_bid"]
    df = pd.read_csv(csv_path, usecols=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.dropna().sort_values("timestamp")
    if require_btc_range:
        df = df.sort_values("timestamp").copy()
        df.set_index("timestamp", inplace=True)
        df["spot_30m_range"] = (
            df["spot_price"].rolling(BTC_RANGE_LB).max() - df["spot_price"].rolling(BTC_RANGE_LB).min()
        )
        df.reset_index(inplace=True)
        contracts = _btc_contracts(df)
    else:
        contracts = _eth_contracts(df)

    by_ce = {c["ce"]: c for c in contracts}

    files = sorted(glob.glob(str(jsonl_dir / "*.jsonl")))
    both_filled = 0
    bt_loss_live_loss = 0
    bt_loss_live_win = 0
    bt_win_live_loss = 0
    mismatches = []
    no_bt = 0
    no_live_pos = 0

    for fp in files:
        p = parse_jsonl(fp)
        if p is None:
            continue
        ce = p["ce"]
        if not p["has_pos"]:
            no_live_pos += 1
            continue
        if ce not in by_ce:
            no_bt += 1
            continue
        bt = simulate_one(by_ce[ce], slope, intercept, POS_SIZE, require_btc_range)
        if bt is None:
            continue
        both_filled += 1
        bt_lost = bt["lost"]
        lv_lost = p["live_loss"]
        if bt_lost and lv_lost:
            bt_loss_live_loss += 1
        elif bt_lost and not lv_lost:
            bt_loss_live_win += 1
            mismatches.append((fp, ce, "bt_loss_live_not_loss", p["expected_pnl"], bt))
        elif not bt_lost and lv_lost:
            bt_win_live_loss += 1
            mismatches.append((fp, ce, "bt_win_live_loss", p["expected_pnl"], bt))
        # bt win + live win : ok

    print(f"\n--- {label} ---")
    print(f"  JSONL fichiers        : {len(files)}")
    print(f"  Live sans position    : {no_live_pos}")
    print(f"  CE hors CSV (ticks)   : {no_bt}")
    print(f"  Alignés (fill BT+pos live) : {both_filled}")
    print(f"  Perdant BT & perdant live  : {bt_loss_live_loss}")
    print(f"  Perdant BT mais live gagne  : {bt_loss_live_win}")
    print(f"  Gagnant BT mais live perd   : {bt_win_live_loss}")
    if both_filled:
        agree = both_filled - bt_loss_live_win - bt_win_live_loss
        print(f"  Même signe PnL (W/W ou L/L) : {agree} ({100*agree/both_filled:.2f}%)")

    if mismatches:
        print(f"\n  Décalages (max 8) :")
        for row in mismatches[:8]:
            fp, ce, kind, epnl, bt = row
            print(f"    {kind}  pnl_live={epnl:.4f}  ce={ce}  file={Path(fp).name}")

    return {
        "both_filled": both_filled,
        "bt_loss_live_loss": bt_loss_live_loss,
        "mismatches": len(mismatches),
    }


if __name__ == "__main__":
    run_for_asset(
        "BTC",
        CSV_BTC,
        JSONL_BTC,
        True,
        SAFETY_SLOPE_BTC,
        SAFETY_INTERCEPT_BTC,
    )
    run_for_asset(
        "ETH",
        CSV_ETH,
        JSONL_ETH,
        False,
        SAFETY_SLOPE_ETH,
        SAFETY_INTERCEPT_ETH,
    )
