"""
analyze_entry_prices.py — Distribution des prix d'entrée réels pour les configs lottery.

Pour chaque config cible, montre:
- Distribution des prix d'entrée (0.01 / 0.02 / ... / 0.05)
- WR et EV par tranche de prix d'entrée
- Impact sur le payout moyen réel
"""

import sys
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Datas" / "csv"
SETTLEMENT_CSV = DATA_DIR / "settlement.csv"

CSV_PATHS = [
    DATA_DIR / "BTC.csv",
    BASE_DIR / "reportLive" / "safeChase" / "BTC.csv",
]
SIZE = 10.0
SCAN_WINDOW_S = 60

# ── Configs à analyser ────────────────────────────────────────────────────────
CONFIGS = [
    # (label,              price_max, slope, intercept, secs_lo, secs_hi)
    ("v2 best RF  p≤0.05 0.25xT+5  t=0-60s",  0.05, 0.25,  5.0, 0, 60),
    ("v2 best EV  p≤0.01 0.50xT+0  t=5-20s",  0.01, 0.50,  0.0, 5, 20),
    ("v2 p≤0.02   0.50xT+5  t=0-30s",          0.02, 0.50,  5.0, 0, 30),
    ("v2 p≤0.05   0.25xT+15 t=0-60s",          0.05, 0.25, 15.0, 0, 60),
    ("v2 p≤0.03   0.50xT+0  t=5-30s",          0.03, 0.50,  0.0, 5, 30),
    ("v1 ref p≤0.01 prox=25 t=20-60s",         0.01, None,  25.0, 20, 60),  # static v1 (slope=None)
]


def load_settlement():
    df = pd.read_csv(SETTLEMENT_CSV)
    df = df[(df['asset'] == 'btc') & (df['tf'] == '5min')]
    return {int(r.open_ts): (float(r.outcome_up) > 0.5) for _, r in df.iterrows()}


def inst_vol_array(spots, ts_ns, window_s=30.0):
    n   = len(spots)
    out = np.zeros(n, dtype=np.float64)
    dq  = deque()
    ws  = window_s * 1e9
    for j in range(n):
        t = float(ts_ns[j])
        while dq and dq[0][0] < t - ws:
            dq.popleft()
        dq.append((t, float(spots[j])))
        vals = [x[1] for x in dq]
        out[j] = max(vals) - min(vals) if vals else 0.0
    return out


def main():
    print("[*] Chargement CSV ...", flush=True)
    dfs = []
    for p in CSV_PATHS:
        if not p.exists():
            continue
        d = pd.read_csv(p, usecols=['timestamp','spot_price','m5_up_bid','m5_down_bid'])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
        print(f"    {p.name}: {len(d):,} ticks")

    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract_ce'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    outcomes = load_settlement()
    print(f"    Settlement: {len(outcomes):,} contrats\n")

    # Construire les contrats une seule fois
    contracts = []
    for ce, grp in df.groupby('contract_ce', sort=True):
        open_ts = int((ce - pd.Timedelta(minutes=5)).timestamp())
        if open_ts not in outcomes:
            continue
        op_spot = float(grp['spot_price'].iloc[0])
        wnd = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=SCAN_WINDOW_S)].reset_index(drop=True)
        if len(wnd) < 2:
            continue
        spots    = wnd['spot_price'].values.astype(np.float64)
        ts_ns    = wnd['timestamp'].values.astype(np.int64)
        ce_ns    = ce.value
        secs_left = (ce_ns - ts_ns) / 1e9
        contracts.append({
            'won_up':    outcomes[open_ts],
            'op':        op_spot,
            'spot':      spots,
            'secs_left': secs_left,
            'up_bid':    wnd['m5_up_bid'].values.astype(np.float64),
            'down_bid':  wnd['m5_down_bid'].values.astype(np.float64),
            'ce':        ce,
        })
    print(f"[*] {len(contracts):,} contrats prêts.\n")
    del df

    SEP = "─" * 100

    for label, pmax, slope, intercept, slo, shi in CONFIGS:
        # Collecter tous les trades
        price_buckets = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
        all_prices = []
        all_secs   = []
        all_bufs   = []
        wins = 0; losses = 0; total_pnl = 0.0

        for c in contracts:
            op       = c['op']
            spots    = c['spot']
            up_bid   = c['up_bid']
            down_bid = c['down_bid']
            secs_left= c['secs_left']
            won_up   = c['won_up']

            for j in range(len(spots)):
                sl = secs_left[j]
                if sl < slo or sl > shi:
                    continue

                # Plafond de prox (dynamique ou statique)
                if slope is not None:
                    prox_limit = slope * sl + intercept
                else:
                    prox_limit = intercept  # v1 static

                buf = abs(spots[j] - op)
                if buf > prox_limit:
                    continue

                ub = up_bid[j]; db = down_bid[j]
                if ub <= 0.0 and db <= 0.0:
                    continue
                if ub <= 0.0:
                    cheap_bid, cheap_is_up = db, False
                elif db <= 0.0:
                    cheap_bid, cheap_is_up = ub, True
                elif ub <= db:
                    cheap_bid, cheap_is_up = ub, True
                else:
                    cheap_bid, cheap_is_up = db, False

                if cheap_bid > pmax:
                    continue

                won = won_up if cheap_is_up else (not won_up)
                price = max(cheap_bid, 1e-4)
                pnl   = SIZE * (1.0/price - 1.0) if won else -SIZE

                # Bucket par centième
                bucket = round(cheap_bid, 2)
                price_buckets[bucket]['n']    += 1
                price_buckets[bucket]['pnl']  += pnl
                if won:
                    price_buckets[bucket]['wins'] += 1
                    wins += 1
                else:
                    losses += 1
                total_pnl += pnl
                all_prices.append(cheap_bid)
                all_secs.append(sl)
                all_bufs.append(buf)
                break  # 1 trade par contrat

        n_trades = wins + losses
        if n_trades == 0:
            print(f"[{label}] — 0 trades\n")
            continue

        wr = wins / n_trades * 100
        ev = total_pnl / n_trades
        be = pmax * 100

        print(f"\n{'='*100}")
        print(f"  {label}")
        print(f"  price_max={pmax:.2f}  slope={slope}  intercept={intercept}  t=[{slo}s-{shi}s]")
        print(f"{'='*100}")
        print(f"  {n_trades} trades  |  WR={wr:.1f}%  BE={be:.1f}%  edge={wr-be:+.1f}%  EV={ev:+.2f}$/trade  PNL={total_pnl:+.0f}$")
        print(f"  Prix moyen d'entrée : {np.mean(all_prices):.4f}  |  médiane : {np.median(all_prices):.4f}")
        print(f"  T restant moyen     : {np.mean(all_secs):.1f}s    |  buf$  moyen  : {np.mean(all_bufs):.1f}$")
        print(f"\n  Distribution des prix d'entrée :")
        print(f"  {'Prix':>8}  {'N trades':>9}  {'%total':>7}  {'Wins':>6}  {'WR%':>7}  {'EV$/t':>8}  {'PNL':>10}  {'Payout si win':>14}")
        print(f"  {SEP[:90]}")

        total_n = sum(v['n'] for v in price_buckets.values())
        for price in sorted(price_buckets.keys()):
            bkt = price_buckets[price]
            n   = bkt['n']
            w   = bkt['wins']
            p   = price if price > 1e-4 else 1e-4
            bkt_wr  = w/n*100 if n > 0 else 0
            bkt_ev  = bkt['pnl']/n if n > 0 else 0
            pct     = n/total_n*100
            payout  = SIZE * (1.0/p - 1.0)  # payout si win (pour SIZE$)
            print(f"  {price:>8.3f}  {n:>9d}  {pct:>6.1f}%  {w:>6d}  {bkt_wr:>6.1f}%  {bkt_ev:>+8.2f}$  {bkt['pnl']:>+10.0f}$  {payout:>+13.0f}$")

        print(f"\n  Résumé EV pondéré par la distribution réelle :")
        print(f"  → PNL moyen par trade (pondéré) = {ev:+.2f}$  (vs +{SIZE*(1.0/pmax - 1.0)*wr/100 - SIZE*(1-wr/100):.2f}$ théorique au prix max fixe)")
        print(f"  → Si on ne rentrait QU'à {pmax:.2f} : EV = {SIZE*(1.0/pmax-1.0)*wr/100 - SIZE*(1-wr/100):+.2f}$/trade")

    print("\n\nDone.")


if __name__ == '__main__':
    main()
