"""
analyze_low_value.py — Analyse "Lottery Long" : acheter le côté perdant (bid < PRICE_MAX)
en fin de contrat quand le spot est proche de l'open et la volatilité est faible.

Idée : si le DOWN trade à 0.01 mais qu'on est à $100 de l'open avec très peu de vol,
la vraie probabilité de crossing dépasse peut-être les 2% (break-even à 0.02).

Usage:
    python analyze_low_value.py [--tf 5min] [--asset btc]
                                [--price-max 0.05]  (prix max d'achat)
                                [--prox 500]        (buffer max spot-open, $)
                                [--vol 100]         (vol instantanée max sur 30s, $)
                                [--window 60]       (secondes avant expiry à analyser)
"""

import sys, argparse
from pathlib import Path
from collections import deque

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE  = Path(__file__).resolve().parent.parent
DATA  = BASE / "Datas" / "csv"
SETTLEMENT_CSV = DATA / "settlement.csv"

# ── colonnes disponibles par TF ──────────────────────────────────────────────
TF_COLS = {
    '5min':  ('m5_up_bid',  'm5_down_bid',  'm5_up_ask',  'm5_down_ask'),
    '15min': ('m15_up_bid', 'm15_down_bid', 'm15_up_ask', 'm15_down_ask'),
    '1h':    ('h1_up_bid',  'h1_down_bid',  'h1_up_ask',  'h1_down_ask'),
}

ASSET_CSV = {'btc': 'BTC.csv', 'eth': 'ETH.csv', 'sol': 'SOL.csv'}


# ── Vol instantanée (range max-min sur window_s secondes) ────────────────────
def inst_vol(spots, ts_ns, window_s=30.0):
    """Retourne le range $ sur les `window_s` dernières secondes pour chaque tick."""
    n = len(spots)
    out = np.zeros(n)
    dq = deque()
    for j in range(n):
        t = float(ts_ns[j])
        while dq and dq[0][0] < t - window_s * 1e9:
            dq.popleft()
        dq.append((t, float(spots[j])))
        vals = [x[1] for x in dq]
        out[j] = max(vals) - min(vals) if vals else 0.0
    return out


# ── Chargement settlement ─────────────────────────────────────────────────────
def load_settlement(asset, tf):
    df = pd.read_csv(SETTLEMENT_CSV)
    df = df[(df['asset'] == asset) & (df['tf'] == tf)]
    return {int(r.open_ts): (float(r.outcome_up) > 0.5) for _, r in df.iterrows()}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf',        default='5min',  choices=list(TF_COLS))
    ap.add_argument('--asset',     default='btc',   choices=list(ASSET_CSV))
    ap.add_argument('--price-max', type=float, default=0.05,  help='Prix bid max pour entrée (ex: 0.05)')
    ap.add_argument('--prox',      type=float, default=500.0, help='Buffer max |spot-open| en $ pour entrer')
    ap.add_argument('--vol',       type=float, default=100.0, help='Vol inst 30s max en $ pour entrer')
    ap.add_argument('--window',    type=int,   default=60,    help='Secondes avant expiry à scanner')
    ap.add_argument('--size',      type=float, default=10.0,  help='Taille de position en $ par trade')
    args = ap.parse_args()

    # ── Chargement CSV tick ──────────────────────────────────────────────────
    csv_path = DATA / ASSET_CSV[args.asset]
    print(f"[*] Chargement {csv_path.name} …", flush=True)
    bid_up, bid_dn, ask_up, ask_dn = TF_COLS[args.tf]
    df = pd.read_csv(csv_path, usecols=['timestamp', 'spot_price', bid_up, bid_dn, ask_up, ask_dn])
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna().sort_values('timestamp')
    print(f"    {len(df):,} ticks chargés.", flush=True)

    # ── Settlement ───────────────────────────────────────────────────────────
    outcomes = load_settlement(args.asset, args.tf)
    print(f"    {len(outcomes):,} contrats avec settlement connu.", flush=True)

    # ── Grouper par contrat ──────────────────────────────────────────────────
    tf_pd = args.tf.replace('min', 'T') if 'min' in args.tf else args.tf
    offset = pd.tseries.frequencies.to_offset(tf_pd)
    df['contract_ce'] = df['timestamp'].dt.floor(tf_pd) + offset

    WINDOW_S = args.window
    PRICE_MAX = args.price_max
    PROX      = args.prox
    VOL_MAX   = args.vol
    SIZE      = args.size

    # Grilles de sweep pour afficher la sensibilité
    price_grid = [0.01, 0.02, 0.03, 0.05, 0.10]
    prox_grid  = [50, 100, 200, 500, 1000, 9999]
    vol_grid   = [20, 50, 100, 200, 9999]

    # Stockage pour sweep
    sweep_results = {}  # (price_max, prox, vol) -> {trades, wins, pnl}

    trades_detail = []  # trades pour le filtre principal

    print(f"\n[*] Scanning contrats (TF={args.tf}, asset={args.asset.upper()}) …")
    print(f"    Paramètres principaux : price_max={PRICE_MAX}, prox=${PROX}, vol<${VOL_MAX}, window={WINDOW_S}s\n")

    n_contracts = 0
    n_with_settlement = 0

    for ce, grp in df.groupby('contract_ce', sort=True):
        open_ts = int((ce - offset).timestamp())
        if open_ts not in outcomes:
            continue

        won_up = outcomes[open_ts]
        op_spot = float(grp['spot_price'].iloc[0])

        # Fenêtre finale WINDOW_S secondes
        window_mask = grp['timestamp'] >= ce - pd.Timedelta(seconds=WINDOW_S)
        wnd = grp[window_mask].reset_index(drop=True)
        if len(wnd) < 2:
            continue

        n_contracts += 1
        n_with_settlement += 1

        spots  = wnd['spot_price'].values.astype(np.float64)
        ts_ns  = wnd['timestamp'].values.astype(np.int64)
        up_bid = wnd[bid_up].values.astype(np.float64)
        dn_bid = wnd[bid_dn].values.astype(np.float64)
        up_ask = wnd[ask_up].values.astype(np.float64)
        dn_ask = wnd[ask_dn].values.astype(np.float64)

        vol30 = inst_vol(spots, ts_ns, window_s=30.0)

        # --- Sweep complet ---
        for pmax in price_grid:
            for prox in prox_grid:
                for vmax in vol_grid:
                    key = (pmax, prox, vmax)
                    if key not in sweep_results:
                        sweep_results[key] = {'n': 0, 'wins': 0, 'pnl': 0.0}
                    # Premier tick qui satisfait les conditions (entrée unique par contrat)
                    for j in range(len(wnd)):
                        buf = abs(spots[j] - op_spot)
                        # Essayer d'acheter la direction PERDANTE
                        # Cas A : UP trade cher → DOWN est le perdant (dn_bid bas)
                        # Cas B : DOWN trade cher → UP est le perdant (up_bid bas)
                        # On cherche LE PLUS BAS des deux bids (la direction la moins chère)
                        cheap_bid = min(up_bid[j], dn_bid[j])
                        cheap_is_up = (up_bid[j] < dn_bid[j])

                        if cheap_bid <= pmax and buf <= prox and vol30[j] <= vmax:
                            won = won_up if cheap_is_up else (not won_up)
                            sr = sweep_results[key]
                            sr['n'] += 1
                            if won:
                                sr['wins'] += 1
                                # Achat de SIZE$ à prix P → SIZE/P shares → gain = SIZE*(1/P - 1)
                                p = max(cheap_bid, 1e-4)
                                sr['pnl'] += SIZE * (1.0 / p - 1.0)
                            else:
                                sr['pnl'] -= SIZE
                            break  # 1 trade par contrat

        # --- Trade détaillé avec les params principaux ---
        for j in range(len(wnd)):
            buf = abs(spots[j] - op_spot)
            cheap_bid = min(up_bid[j], dn_bid[j])
            cheap_is_up = (up_bid[j] < dn_bid[j])

            if cheap_bid <= PRICE_MAX and buf <= PROX and vol30[j] <= VOL_MAX:
                won = won_up if cheap_is_up else (not won_up)
                secs_left = (ce - wnd['timestamp'].iloc[j]).total_seconds()
                trades_detail.append({
                    'ce':         str(ce),
                    'direction':  'UP' if cheap_is_up else 'DOWN',
                    'price':      round(cheap_bid, 4),
                    'buf_$':      round(buf, 1),
                    'vol30s_$':   round(vol30[j], 1),
                    'secs_left':  round(secs_left, 1),
                    'won_up':     won_up,
                    'won_trade':  won,
                })
                break

    print(f"[*] {n_with_settlement} contrats analysés.\n")

    # ── Résultats sweep ──────────────────────────────────────────────────────
    print("=" * 90)
    print(f"{'SWEEP':<6}  {'price_max':>10}  {'prox($)':>8}  {'vol30s($)':>10}  {'trades':>7}  {'wins':>6}  {'WR%':>7}  {'break_even%':>12}  {'PNL_$':>9}")
    print("-" * 90)

    # Trier par PNL décroissant parmi les configs ayant ≥10 trades
    rows = []
    for (pmax, prox, vmax), sr in sweep_results.items():
        if sr['n'] == 0:
            continue
        wr = sr['wins'] / sr['n'] * 100
        be = pmax * 100  # break-even en %
        rows.append((pmax, prox, vmax, sr['n'], sr['wins'], wr, be, sr['pnl']))

    rows.sort(key=lambda r: (r[4] >= 1, r[7]), reverse=True)  # tri: au moins 1 win, puis PNL

    for pmax, prox, vmax, n, wins, wr, be, pnl in rows[:60]:
        marker = " ★" if wr > be and n >= 5 else ""
        print(f"{'':6}  {pmax:>10.2f}  {prox:>8.0f}  {vmax:>10.0f}  {n:>7}  {wins:>6}  {wr:>6.1f}%  {be:>11.1f}%  {pnl:>+9.1f}{marker}")

    # ── Détail trades (params principaux) ───────────────────────────────────
    if trades_detail:
        print(f"\n{'=' * 90}")
        print(f"DÉTAIL TRADES (price_max={PRICE_MAX}, prox=${PROX}, vol<${VOL_MAX})")
        print(f"{'CE':>22}  {'DIR':>5}  {'price':>6}  {'buf$':>7}  {'vol30s$':>8}  {'secs_left':>10}  {'WON':>5}")
        print("-" * 90)
        for t in trades_detail[:50]:
            w = "WIN" if t['won_trade'] else "loss"
            print(f"{t['ce']:>22}  {t['direction']:>5}  {t['price']:>6.3f}  {t['buf_$']:>7.1f}  {t['vol30s_$']:>8.1f}  {t['secs_left']:>10.1f}  {w:>5}")

        wins_main = sum(1 for t in trades_detail if t['won_trade'])
        n_main    = len(trades_detail)
        wr_main   = wins_main / n_main * 100 if n_main else 0
        pnl_main  = sum(SIZE * (1.0 / max(t['price'], 1e-4) - 1.0) if t['won_trade'] else -SIZE for t in trades_detail)
        print(f"\n  → {n_main} trades, {wins_main} wins ({wr_main:.1f}%), break-even {PRICE_MAX*100:.1f}%, PNL total: {pnl_main:+.2f}$")

        # Distribution par direction
        up_trades  = [t for t in trades_detail if t['direction'] == 'UP']
        dn_trades  = [t for t in trades_detail if t['direction'] == 'DOWN']
        up_wins    = sum(1 for t in up_trades if t['won_trade'])
        dn_wins    = sum(1 for t in dn_trades if t['won_trade'])
        if up_trades: print(f"  → UP  cheap : {len(up_trades)} trades, {up_wins} wins ({up_wins/len(up_trades)*100:.1f}%)")
        if dn_trades: print(f"  → DOWN cheap : {len(dn_trades)} trades, {dn_wins} wins ({dn_wins/len(dn_trades)*100:.1f}%)")

        # Distribution buf$
        bufs = [t['buf_$'] for t in trades_detail]
        print(f"\n  Buffer |spot-open| : min={min(bufs):.0f}$ med={np.median(bufs):.0f}$ max={max(bufs):.0f}$")

        # Histogramme par tranche de secondes restantes
        secs_bins = [0, 5, 10, 20, 30, 60]
        print(f"\n  Distribution secs_left :")
        for i in range(len(secs_bins) - 1):
            lo, hi = secs_bins[i], secs_bins[i+1]
            sub = [t for t in trades_detail if lo <= t['secs_left'] < hi]
            sw = sum(1 for t in sub if t['won_trade'])
            if sub:
                print(f"    {lo:2d}s – {hi:2d}s : {len(sub):4d} trades, {sw:3d} wins ({sw/len(sub)*100:.1f}%)")
    else:
        print(f"\n[!] Aucun trade trouvé avec price_max={PRICE_MAX}, prox=${PROX}, vol<${VOL_MAX}.")
        print("    Essayez --prox 9999 ou --price-max 0.10 pour élargir.")

    print("\nDone.")


if __name__ == '__main__':
    main()
