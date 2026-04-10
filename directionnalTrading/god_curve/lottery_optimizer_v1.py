"""
lottery_optimizer_v1.py — Optimiseur "Lottery Long" BTC M5

Stratégie miroir du god_curve : acheter le CÔTÉ PERDANT (bid très bas) en fin de contrat
quand le spot est proche du strike. Si le marché sous-price les reversals, c'est rentable.

Logique d'entrée :
  - On scanne les N dernières secondes du contrat
  - On prend la position dès que cheap_bid <= price_max AND |spot-open| <= prox_max
    AND vol_30s <= vol30s_max AND secs_lo <= secs_left <= secs_hi
  - 1 trade par contrat (premier tick éligible dans la fenêtre)
  - Sizing fixe : SIZE $ par trade (on achète SIZE/price shares → gain = SIZE*(1/price-1) si WIN)

Grid :
  price_max  : 0.01, 0.02, 0.03, 0.05
  prox_max   : 25, 50, 100, 200, 500
  vol30s_max : 20, 50, 100, 200, 9999
  secs_lo/hi : 18 paires valides (secs_lo < secs_hi)

Total : 4 × 5 × 5 × 18 = 1800 configs
"""

import argparse
import pickle
import sys
import time
import multiprocessing as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Datas" / "csv"
SETTLEMENT_CSV = DATA_DIR / "settlement.csv"

DEFAULT_CSV_PATHS = [
    str(DATA_DIR / "BTC.csv"),
    str(BASE_DIR / "reportLive" / "safeChase" / "BTC.csv"),
]
DEFAULT_N_WORKERS = max(1, mp.cpu_count() - 2)
DEFAULT_SIZE      = 10.0   # $ par trade
VOL_WINDOW_S      = 30.0   # fenêtre vol instantanée
SCAN_WINDOW_S     = 60     # secondes avant expiry à charger

_CONTRACTS    = None
_TOTAL_HOURS  = None


# ─────────────────────────────────────────────────────────────────────────────
# Grid de configs
# ─────────────────────────────────────────────────────────────────────────────
def build_configs():
    configs = []
    price_grid  = [0.01, 0.02, 0.03, 0.05]
    prox_grid   = [25, 50, 100, 200, 500]
    vol_grid    = [20, 50, 100, 200, 9999]
    secs_lo_g   = [0, 5, 10, 20]
    secs_hi_g   = [15, 20, 30, 45, 60]

    for pmax in price_grid:
        for prox in prox_grid:
            for vmax in vol_grid:
                for slo in secs_lo_g:
                    for shi in secs_hi_g:
                        if slo >= shi:
                            continue
                        configs.append(dict(
                            price_max  = pmax,
                            prox_max   = float(prox),
                            vol30s_max = float(vmax),
                            secs_lo    = float(slo),
                            secs_hi    = float(shi),
                        ))
    return configs  # 1800 configs


# ─────────────────────────────────────────────────────────────────────────────
# Chargement dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_settlement(asset='btc', tf='5min'):
    df = pd.read_csv(SETTLEMENT_CSV)
    df = df[(df['asset'] == asset) & (df['tf'] == tf)]
    return {int(r.open_ts): (float(r.outcome_up) > 0.5) for _, r in df.iterrows()}


def inst_vol_array(spots, ts_ns, window_s=30.0):
    """Range max-min sur `window_s` secondes glissantes (tick par tick)."""
    n   = len(spots)
    out = np.zeros(n, dtype=np.float64)
    dq  = deque()
    ws  = window_s * 1e9  # ns
    for j in range(n):
        t = float(ts_ns[j])
        while dq and dq[0][0] < t - ws:
            dq.popleft()
        dq.append((t, float(spots[j])))
        vals = [x[1] for x in dq]
        out[j] = max(vals) - min(vals) if vals else 0.0
    return out


def load_dataset(csv_paths):
    """Charge CSV + settlement, construit les contrats avec precomputes. Appelé 1 fois."""
    global _CONTRACTS, _TOTAL_HOURS

    print(f"[*] Chargement CSV ...", flush=True)
    dfs = []
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            print(f"    [SKIP] {p} (introuvable)", flush=True)
            continue
        d = pd.read_csv(path, usecols=[
            'timestamp', 'spot_price',
            'm5_up_bid', 'm5_down_bid',
            'm5_up_ask', 'm5_down_ask',
        ])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
        print(f"    {path.name}: {len(d):,} ticks", flush=True)

    if not dfs:
        raise FileNotFoundError("Aucun CSV trouvé.")

    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract_ce'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    outcomes = load_settlement('btc', '5min')
    print(f"    Settlement: {len(outcomes):,} contrats connus.", flush=True)

    contracts = []
    for ce, grp in df.groupby('contract_ce', sort=True):
        open_ts = int((ce - pd.Timedelta(minutes=5)).timestamp())
        if open_ts not in outcomes:
            continue

        op_spot = float(grp['spot_price'].iloc[0])
        wnd = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=SCAN_WINDOW_S)].reset_index(drop=True)
        if len(wnd) < 2:
            continue

        spots  = wnd['spot_price'].values.astype(np.float64)
        ts_ns  = wnd['timestamp'].values.astype(np.int64)
        # secs_left par tick
        ce_ns    = ce.value  # timestamp en ns
        secs_left = (ce_ns - ts_ns) / 1e9  # positif = avant expiry

        vol30 = inst_vol_array(spots, ts_ns, VOL_WINDOW_S)

        contracts.append({
            'won_up':   outcomes[open_ts],
            'op':       op_spot,
            'spot':     spots,
            'ts_ns':    ts_ns,
            'secs_left':secs_left,
            'vol30':    vol30,
            'up_bid':   wnd['m5_up_bid'].values.astype(np.float64),
            'down_bid': wnd['m5_down_bid'].values.astype(np.float64),
            'up_ask':   wnd['m5_up_ask'].values.astype(np.float64),
            'down_ask': wnd['m5_down_ask'].values.astype(np.float64),
            'ce_date':  ce.date(),
            'ce':       ce,
        })

    del df
    _CONTRACTS   = contracts
    _TOTAL_HOURS = (
        (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 3600
        if len(contracts) > 1 else 1.0
    )
    print(f"    Contrats prêts: {len(contracts):,}  ({_TOTAL_HOURS/24:.1f}j)", flush=True)
    return len(contracts)


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────
def _init_worker(pickle_path: str):
    global _CONTRACTS, _TOTAL_HOURS
    with open(pickle_path, 'rb') as f:
        payload = pickle.load(f)
    _CONTRACTS   = payload['contracts']
    _TOTAL_HOURS = payload['total_hours']


def _simulate_config(c, cfg):
    """
    Retourne (won, entry_price) ou None si pas de signal.
    won=True si le côté acheté a gagné.
    entry_price = prix auquel on a acheté (cheap_bid).
    """
    pmax = cfg['price_max']
    prox = cfg['prox_max']
    vmax = cfg['vol30s_max']
    slo  = cfg['secs_lo']
    shi  = cfg['secs_hi']
    op   = c['op']

    spots     = c['spot']
    up_bid    = c['up_bid']
    down_bid  = c['down_bid']
    vol30     = c['vol30']
    secs_left = c['secs_left']
    won_up    = c['won_up']

    for j in range(len(spots)):
        sl = secs_left[j]
        if sl < slo or sl > shi:
            continue

        buf = abs(spots[j] - op)
        if buf > prox:
            continue

        if vol30[j] > vmax:
            continue

        # Côté le moins cher (le "perdant" selon le marché)
        ub = up_bid[j]
        db = down_bid[j]

        # Filtrer les bids nuls (pas de marché)
        if ub <= 0.0 and db <= 0.0:
            continue

        if ub <= 0.0:
            cheap_bid  = db
            cheap_is_up = False
        elif db <= 0.0:
            cheap_bid  = ub
            cheap_is_up = True
        elif ub <= db:
            cheap_bid  = ub
            cheap_is_up = True
        else:
            cheap_bid  = db
            cheap_is_up = False

        if cheap_bid > pmax:
            continue

        # On a un signal
        won = won_up if cheap_is_up else (not won_up)
        return won, cheap_bid

    return None


def worker_fn(config_batch):
    global _CONTRACTS, _TOTAL_HOURS
    contracts = _CONTRACTS
    size      = DEFAULT_SIZE
    results   = []

    for cfg in config_batch:
        wins = 0; losses = 0
        total_pnl = 0.0
        contract_pnls = []
        day_pnls = {}

        for c in contracts:
            res = _simulate_config(c, cfg)
            if res is None:
                continue
            won, price = res
            price = max(price, 1e-4)  # sécurité div/0

            # PnL : achat de SIZE$ à price P → SIZE/P shares → net = SIZE*(1/P-1) si won, -SIZE si lost
            if won:
                pnl = size * (1.0 / price - 1.0)
                wins += 1
            else:
                pnl = -size
                losses += 1

            total_pnl += pnl
            contract_pnls.append(pnl)
            day_pnls[c['ce_date']] = day_pnls.get(c['ce_date'], 0.0) + pnl

        trades = wins + losses
        if trades == 0:
            continue

        wr         = wins / trades * 100.0
        break_even = cfg['price_max'] * 100.0
        edge       = wr - break_even
        ev_trade   = total_pnl / trades

        losing_days = sum(1 for v in day_pnls.values() if v < 0)
        worst_day   = min(day_pnls.values()) if day_pnls else 0.0
        best_day    = max(day_pnls.values()) if day_pnls else 0.0

        # Max drawdown peak-to-trough
        cum = np.cumsum([0.0] + contract_pnls)
        peak = 0.0; max_dd = 0.0
        for v in cum:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd

        rf = (total_pnl / max(max_dd, size)) if max_dd > 0 else 0.0

        results.append({**cfg,
            'trades':      trades,
            'wins':        wins,
            'losses':      losses,
            'wr':          wr,
            'break_even':  break_even,
            'edge':        edge,
            'edge_ratio':  wr / break_even if break_even > 0 else 0.0,
            'total_pnl':   total_pnl,
            'ev_trade':    ev_trade,
            'losing_days': losing_days,
            'worst_day':   worst_day,
            'best_day':    best_day,
            'max_dd':      max_dd,
            'rf':          rf,
            'total_hours': float(_TOTAL_HOURS),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Formatage
# ─────────────────────────────────────────────────────────────────────────────
def fmt_row(i, r, days):
    vol_s = f"vol<{r['vol30s_max']:.0f}" if r['vol30s_max'] < 9999 else "vol=any"
    edge_m = "★" if r['edge'] > 0 else " "
    return (
        f"{i:>4d} | "
        f"p≤{r['price_max']:.2f} prox≤{r['prox_max']:>4.0f}$ {vol_s:>8s} "
        f"t=[{r['secs_lo']:>2.0f}s-{r['secs_hi']:>2.0f}s] | "
        f"T={r['trades']:>5d} ({r['trades']/days:>4.1f}/j) "
        f"WR={r['wr']:>5.1f}% BE={r['break_even']:>4.1f}% edge={r['edge']:>+5.1f}% {edge_m} "
        f"EV={r['ev_trade']:>+7.2f}$/t | "
        f"PNL={r['total_pnl']:>+10.0f}$ ({r['total_pnl']/days:>+7.1f}$/j) "
        f"maxDD={r['max_dd']:>7.0f}$ RF={r['rf']:>5.1f}x "
        f"loseDay={r['losing_days']:>3d}/{int(days):>3d} wrstDay={r['worst_day']:>+8.0f}$\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Lottery Optimizer v1 (BTC M5)")
    ap.add_argument('--csv',     nargs='+', default=DEFAULT_CSV_PATHS)
    ap.add_argument('--workers', type=int,  default=DEFAULT_N_WORKERS)
    ap.add_argument('--threads', action='store_true')
    ap.add_argument('--size',    type=float, default=10.0, help='$ par trade')
    args = ap.parse_args()

    global DEFAULT_SIZE, _CONTRACTS, _TOTAL_HOURS
    DEFAULT_SIZE = args.size

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / f"lottery_btc_v1_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"lottery_results_{run_ts}.txt"

    t0 = time.time()
    configs = build_configs()
    n_workers = max(1, args.workers)

    print(f"\n{'='*80}")
    print(f"  Lottery Optimizer v1 — BTC M5")
    print(f"  Run: {run_ts}  |  Configs: {len(configs)}  |  Workers: {n_workers}")
    print(f"  Size: {args.size}$/trade  |  Sortie: {run_dir.name}")
    print(f"{'='*80}\n")

    n_ct = load_dataset(args.csv)

    batch_size = max(1, len(configs) // n_workers + 1)
    batches = [configs[i:i+batch_size] for i in range(0, len(configs), batch_size)]
    print(f"\n[*] {len(batches)} batches × ~{batch_size} configs\n")

    all_results = []

    if args.threads:
        print("  Backend: threads\n", flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker_fn, b) for b in batches]
            for i, fut in enumerate(as_completed(futures), 1):
                batch_res = fut.result()
                all_results.extend(batch_res)
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i}/{len(batches)} done — {len(all_results)} résultats", flush=True)
    else:
        print("  Backend: processus (vrai multi-coeur)\n", flush=True)
        payload_path = run_dir / "_worker_data.pkl"
        with open(payload_path, 'wb') as f:
            pickle.dump({'contracts': _CONTRACTS, 'total_hours': _TOTAL_HOURS}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb = payload_path.stat().st_size / (1024*1024)
        print(f"  Snapshot: {sz_mb:.1f} MiB  (×{n_workers} workers)\n", flush=True)
        _CONTRACTS = None; _TOTAL_HOURS = None
        try:
            with mp.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(str(payload_path.resolve()),),
            ) as pool:
                for i, br in enumerate(pool.imap_unordered(worker_fn, batches), 1):
                    all_results.extend(br)
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i}/{len(batches)} done — {len(all_results)} résultats", flush=True)
        finally:
            payload_path.unlink(missing_ok=True)

    if not all_results:
        print("[!] Aucun résultat."); return

    elapsed = time.time() - t0
    days    = all_results[0]['total_hours'] / 24

    # ── Tris ──────────────────────────────────────────────────────────────────
    by_edge_ratio = sorted(
        [r for r in all_results if r['edge'] > 0],
        key=lambda x: (-x['edge_ratio'], -x['ev_trade'])
    )
    by_ev   = sorted(all_results, key=lambda x: -x['ev_trade'])
    by_rf   = sorted([r for r in all_results if r['total_pnl'] > 0], key=lambda x: -x['rf'])
    by_pnl  = sorted(all_results, key=lambda x: -x['total_pnl'])
    by_cons = sorted(
        [r for r in all_results if r['trades'] >= 50],
        key=lambda x: (x['losing_days'], -x['ev_trade'])
    )

    sep = "─" * 150 + "\n"
    hdr = (
        f"  {'#':>4}   {'p_max':>5} {'prox':>6} {'vol30s':>8} {'t_range':>10}   "
        f"{'trades':>7} {'T/j':>5}   "
        f"{'WR':>6} {'BE':>5} {'edge':>7}★   "
        f"{'EV$/t':>8}   "
        f"{'PNL_tot':>11} {'PNL/j':>9}   "
        f"{'maxDD':>8} {'RF':>6}   "
        f"{'loseDays':>9} {'worstDay':>10}\n"
    )

    lines = [
        f"Lottery Optimizer v1 — BTC M5  |  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  run_id={run_ts}  |  "
        f"Data: {days:.1f}j  |  "
        f"Configs évaluées: {len(all_results)}/{len(configs)}  |  "
        f"Size: {args.size}$/trade  |  {elapsed:.0f}s\n\n",
    ]

    sections = [
        (f"TOP 40 — Edge Ratio (WR/break_even, edge>0 seulement)", by_edge_ratio, 40),
        (f"TOP 40 — EV par trade ($/trade)", by_ev, 40),
        (f"TOP 40 — Recovery Factor (PNL>0 seulement)", by_rf, 40),
        (f"TOP 40 — PNL total", by_pnl, 40),
        (f"TOP 30 — Consistance (min losing_days, ≥50 trades)", by_cons, 30),
    ]

    for title, ranked, n in sections:
        lines.append(f"\n{'='*150}\n>>> {title}\n{'='*150}\n{hdr}{sep}")
        for i, r in enumerate(ranked[:n], 1):
            lines.append(fmt_row(i, r, days))

    # ── Analyse hedge ─────────────────────────────────────────────────────────
    # Meilleure config par edge_ratio → analyse détaillée
    if by_edge_ratio:
        best = by_edge_ratio[0]
        lines.append(f"\n{'='*150}\n>>> ANALYSE HEDGE — Meilleure config par edge_ratio\n{'='*150}\n")
        lines.append(fmt_row(1, best, days))
        lines.append(
            f"\n  Concept hedge : la stratégie lottery GAGNE quand il y a un REVERSAL en fin de contrat.\n"
            f"  C'est exactement le scénario où god_curve PERD (toxic fill).\n"
            f"  → Corrélation naturelle négative avec les pertes god_curve.\n\n"
            f"  Break-even : {best['price_max']:.2f} × position → win 1 sur {1/best['price_max']:.0f} suffit.\n"
            f"  Win rate observé : {best['wr']:.1f}%  (break-even : {best['break_even']:.1f}%)  "
            f"edge : {best['edge']:+.1f}%\n"
            f"  EV par trade    : {best['ev_trade']:+.2f}$ pour {args.size}$ misé  "
            f"(ratio : {best['wr']/best['break_even']:.2f}x le break-even)\n"
        )

    output = "".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n{'='*80}")
    print(f"  Terminé en {elapsed:.0f}s  |  {len(all_results)} configs évaluées")
    print(f"  Sortie : {out_path}")
    print(f"{'='*80}\n")

    # Affichage console : top 20 par edge_ratio
    print(f"\n>>> TOP 20 EDGE RATIO\n{hdr}{sep}")
    for i, r in enumerate(by_edge_ratio[:20], 1):
        print(fmt_row(i, r, days), end='')

    print(f"\n>>> TOP 20 EV/TRADE\n{hdr}{sep}")
    for i, r in enumerate(by_ev[:20], 1):
        print(fmt_row(i, r, days), end='')

    print(f"\n>>> TOP 20 RF\n{hdr}{sep}")
    for i, r in enumerate(by_rf[:20], 1):
        print(fmt_row(i, r, days), end='')


if __name__ == '__main__':
    main()
