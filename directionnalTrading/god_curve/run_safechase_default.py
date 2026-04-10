"""
Backtest God Curve — paramètres défaut (USER) :
  BTC : SAFETY_SLOPE_BTC=2.0,  SAFETY_INTERCEPT_BTC=0.5
        + filtre range BTC : spot_30m_range >= $60 (lookback 30m)
  ETH : SAFETY_SLOPE_ETH=0.05, SAFETY_INTERCEPT_ETH=1.0
  Min bid floor : 0.65
  Position : $50 / fill
  Max fills : 3
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

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
    """Même construction que run_5_point_sweep.py."""
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
    """Même construction que final_eth_sweep_definitive.py."""
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


def simulate_contracts(contracts, slope, intercept, pos_size, label, *, require_btc_range=False):
    wins, losses = 0, 0
    total_payout, total_cost = 0.0, 0.0
    total_fills = 0
    dropped_by_range = 0

    for c in contracts:
        activated_side = None
        f_count = 0
        our_limit = None

        for j in range(len(c["ts"])):
            s = c["spots"][j]
            if require_btc_range:
                r30 = float(c["r30"][j])
                if np.isnan(r30) or r30 < BTC_RANGE_THRESH:
                    dropped_by_range += 1
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
                total_cost += pos_size * our_limit
                our_limit = None
                if f_count >= MAX_FILLS:
                    break

            if bid >= MIN_BID_FLOOR:
                proposed = min(bid, 0.99)
                if our_limit is None or proposed > our_limit:
                    our_limit = proposed

        if f_count > 0:
            total_fills += f_count
            won = c["won_up"] == (activated_side == "UP")
            if won:
                wins += 1
                total_payout += pos_size * f_count
            else:
                losses += 1

    n = wins + losses
    wr = (wins / n * 100.0) if n else 0.0
    pnl = total_payout - total_cost
    print(f"\n=== {label} ===")
    print(f"  slope={slope}  intercept=${intercept}  position=${pos_size}")
    if require_btc_range:
        print(f"  Filtre BTC: range({BTC_RANGE_LB}) >= ${BTC_RANGE_THRESH}")
    print(f"  Contrats avec fill : {n}  |  Wins: {wins}  Losses: {losses}  |  WR: {wr:.2f}%")
    print(f"  Fills totaux : {total_fills}  |  PnL : ${pnl:,.2f}")
    if require_btc_range:
        print(f"  Ticks ignorés (filtre range) : {dropped_by_range}")


def main():
    cols = ["timestamp", "spot_price", "m5_up_bid", "m5_down_bid"]

    btc_path = BASE / "reportLive" / "safeChase" / "BTC.csv"
    eth_path = BASE / "reportLive" / "safeChase" / "ETH.csv"

    print("Chargement CSV safeChase (colonnes M5 bid)...")
    dfb = pd.read_csv(btc_path, usecols=cols)
    dfb["timestamp"] = pd.to_datetime(dfb["timestamp"], utc=True).dt.tz_localize(None)
    dfb = dfb.dropna().sort_values("timestamp")

    dfe = pd.read_csv(eth_path, usecols=cols)
    dfe["timestamp"] = pd.to_datetime(dfe["timestamp"], utc=True).dt.tz_localize(None)
    dfe = dfe.dropna().sort_values("timestamp")

    # Pré-calc BTC range 30m (max-min) pour le filtre.
    dfb = dfb.sort_values("timestamp").copy()
    dfb.set_index("timestamp", inplace=True)
    dfb["spot_30m_range"] = dfb["spot_price"].rolling(BTC_RANGE_LB).max() - dfb["spot_price"].rolling(BTC_RANGE_LB).min()
    dfb.reset_index(inplace=True)

    bc = _btc_contracts(dfb)
    ec = _eth_contracts(dfe)
    print(f"Contrats (fenêtre T-60s) — BTC: {len(bc)}  ETH: {len(ec)}")

    simulate_contracts(
        bc,
        slope=SAFETY_SLOPE_BTC,
        intercept=SAFETY_INTERCEPT_BTC,
        pos_size=POS_SIZE,
        label="BTC (default user)",
        require_btc_range=True,
    )
    simulate_contracts(
        ec,
        slope=SAFETY_SLOPE_ETH,
        intercept=SAFETY_INTERCEPT_ETH,
        pos_size=POS_SIZE,
        label="ETH (default user)",
        require_btc_range=False,
    )


if __name__ == "__main__":
    main()
