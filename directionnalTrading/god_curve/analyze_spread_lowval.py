"""
analyze_spread_lowval.py — Analyse des spreads bid/ask sur les basses valeurs.

Questions :
1. Quand bid ≤ 0.05, quel est l'ask correspondant ? (spread réel)
2. Est-ce qu'il y a toujours un ask dispo ?
3. Quel est le prix réel de fill si on prend le ask (taker) ?
4. L'edge survit-il au ask ?
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
CSV_PATHS = [DATA_DIR / "BTC.csv", BASE_DIR / "reportLive" / "safeChase" / "BTC.csv"]

SIZE         = 10.0
SCAN_WINDOW_S = 60
# Config de référence v2
SLOPE     = 0.50
INTERCEPT = 0.0
SECS_LO   = 5.0
SECS_HI   = 20.0
PRICE_MAX_BID = 0.05  # on regarde tout ≤ 0.05 bid


def load_settlement():
    df = pd.read_csv(SETTLEMENT_CSV)
    df = df[(df['asset'] == 'btc') & (df['tf'] == '5min')]
    return {int(r.open_ts): (float(r.outcome_up) > 0.5) for _, r in df.iterrows()}


def main():
    print("[*] Chargement ...", flush=True)
    dfs = []
    for p in CSV_PATHS:
        if not p.exists(): continue
        d = pd.read_csv(p, usecols=[
            'timestamp','spot_price',
            'm5_up_bid','m5_down_bid',
            'm5_up_ask','m5_down_ask',
        ])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
        print(f"    {p.name}: {len(d):,} ticks")

    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract_ce'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)
    outcomes = load_settlement()
    print(f"    Settlement: {len(outcomes):,} contrats\n")

    # ── Collecte des ticks éligibles ─────────────────────────────────────────
    records = []  # {bid, ask, won, secs_left, buf}

    for ce, grp in df.groupby('contract_ce', sort=True):
        open_ts = int((ce - pd.Timedelta(minutes=5)).timestamp())
        if open_ts not in outcomes:
            continue
        won_up  = outcomes[open_ts]
        op_spot = float(grp['spot_price'].iloc[0])
        wnd = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=SCAN_WINDOW_S)].reset_index(drop=True)
        if len(wnd) < 2:
            continue

        spots     = wnd['spot_price'].values.astype(np.float64)
        ts_ns     = wnd['timestamp'].values.astype(np.int64)
        ce_ns     = ce.value
        secs_left = (ce_ns - ts_ns) / 1e9
        up_bid    = wnd['m5_up_bid'].values.astype(np.float64)
        dn_bid    = wnd['m5_down_bid'].values.astype(np.float64)
        up_ask    = wnd['m5_up_ask'].values.astype(np.float64)
        dn_ask    = wnd['m5_down_ask'].values.astype(np.float64)

        for j in range(len(spots)):
            sl = secs_left[j]
            if sl < SECS_LO or sl > SECS_HI:
                continue

            prox_limit = SLOPE * sl + INTERCEPT
            buf = abs(spots[j] - op_spot)
            if buf > prox_limit:
                continue

            ub = up_bid[j]; db = dn_bid[j]
            ua = up_ask[j]; da = dn_ask[j]
            if ub <= 0.0 and db <= 0.0:
                continue

            # Côté le moins cher (bid)
            if ub <= 0.0:
                bid, ask, is_up = db, da, False
            elif db <= 0.0:
                bid, ask, is_up = ub, ua, True
            elif ub <= db:
                bid, ask, is_up = ub, ua, True
            else:
                bid, ask, is_up = db, da, False

            if bid > PRICE_MAX_BID:
                continue

            won = won_up if is_up else (not won_up)
            records.append({
                'bid':  round(bid,  3),
                'ask':  round(ask,  3),
                'won':  won,
                'secs': sl,
                'buf':  buf,
            })
            break  # 1 tick par contrat

    print(f"[*] {len(records)} ticks éligibles collectés (bid≤{PRICE_MAX_BID}, formule {SLOPE}xT+{INTERCEPT}, t=[{SECS_LO}s-{SECS_HI}s])\n")

    if not records:
        print("Aucun tick."); return

    bids = np.array([r['bid'] for r in records])
    asks = np.array([r['ask'] for r in records])
    spreads = asks - bids
    wons = np.array([r['won'] for r in records])

    # ── Stats globales spread ─────────────────────────────────────────────────
    no_ask = np.sum(asks <= 0)
    print(f"{'='*80}")
    print(f"  SPREAD BID/ASK — analyse globale")
    print(f"{'='*80}")
    print(f"  Ticks sans ask (ask=0) : {no_ask} ({no_ask/len(records)*100:.1f}%)")
    valid = asks > 0
    if valid.sum():
        print(f"  Spread moyen  : {spreads[valid].mean():.4f}  ({spreads[valid].mean()*100:.2f} cents)")
        print(f"  Spread médian : {np.median(spreads[valid]):.4f}  ({np.median(spreads[valid])*100:.2f} cents)")
        print(f"  Spread max    : {spreads[valid].max():.4f}")
        print(f"  Ask moyen     : {asks[valid].mean():.4f}")
        print(f"  Ask médian    : {np.median(asks[valid]):.4f}")

    # ── Par tranche de bid ────────────────────────────────────────────────────
    print(f"\n  Distribution BID → ASK réel :")
    print(f"  {'bid':>6}  {'N':>5}  {'ask=0%':>7}  {'ask_moy':>8}  {'spread_moy':>11}  {'WR_bid%':>9}  {'WR_ask%':>9}  {'EV@bid':>8}  {'EV@ask':>8}")
    print(f"  {'─'*90}")

    bid_buckets = [0.01, 0.02, 0.03, 0.04, 0.05]
    for b in bid_buckets:
        mask = (bids == b)
        if mask.sum() == 0:
            continue
        sub_ask    = asks[mask]
        sub_won    = wons[mask]
        sub_spread = spreads[mask]
        n          = mask.sum()
        no_ask_n   = (sub_ask <= 0).sum()
        valid_ask  = sub_ask[sub_ask > 0]

        wr  = sub_won.mean() * 100
        ask_moy = valid_ask.mean() if len(valid_ask) else float('nan')
        sp_moy  = sub_spread[sub_ask > 0].mean() if len(valid_ask) else float('nan')

        # EV si fill au bid (current assumption)
        ev_bid = SIZE * (1.0/b - 1.0) * wr/100 - SIZE * (1 - wr/100)
        # EV si fill au ask (réaliste taker)
        if not np.isnan(ask_moy) and ask_moy > 0:
            ev_ask = SIZE * (1.0/ask_moy - 1.0) * wr/100 - SIZE * (1 - wr/100)
        else:
            ev_ask = float('nan')

        print(f"  {b:>6.3f}  {n:>5d}  {no_ask_n/n*100:>6.1f}%  {ask_moy:>8.4f}  {sp_moy:>11.4f}  {wr:>8.1f}%  {wr:>8.1f}%  {ev_bid:>+8.2f}$  {ev_ask:>+8.2f}$")

    # ── Simulation complète : bid vs ask vs midpoint ──────────────────────────
    print(f"\n{'='*80}")
    print(f"  SIMULATION PNL — 3 hypothèses de fill")
    print(f"{'='*80}")

    for fill_label, use_ask in [("Fill au BID (limit, maker)", False), ("Fill au ASK (market, taker)", True)]:
        pnl_total = 0.0
        n_filled  = 0
        n_no_ask  = 0
        wins_f    = 0
        price_dist = defaultdict(int)

        for r in records:
            if use_ask:
                fill_p = r['ask']
                if fill_p <= 0:
                    n_no_ask += 1
                    continue
            else:
                fill_p = r['bid']
                if fill_p <= 0:
                    continue

            fill_p = max(fill_p, 1e-4)
            price_dist[round(fill_p, 2)] += 1
            won = r['won']
            pnl = SIZE * (1.0/fill_p - 1.0) if won else -SIZE
            pnl_total += pnl
            n_filled  += 1
            if won: wins_f += 1

        if n_filled == 0:
            print(f"\n  [{fill_label}] : 0 fills")
            continue

        wr_f  = wins_f / n_filled * 100
        ev_f  = pnl_total / n_filled
        days  = 54.6

        print(f"\n  [{fill_label}]")
        if use_ask:
            print(f"    No-ask (skip) : {n_no_ask} ({n_no_ask/(n_filled+n_no_ask)*100:.1f}%)")
        print(f"    Fills      : {n_filled}  ({n_filled/days:.1f}/j)")
        print(f"    WR         : {wr_f:.1f}%  (BE={PRICE_MAX_BID*100:.0f}%)")
        print(f"    EV/trade   : {ev_f:+.2f}$")
        print(f"    PNL total  : {pnl_total:+.0f}$  ({pnl_total/days:+.1f}$/j)")
        print(f"    Prix fills : ", end='')
        for p, cnt in sorted(price_dist.items()):
            print(f"{p:.2f}→{cnt}  ", end='')
        print()

    print("\nDone.")


if __name__ == '__main__':
    main()
