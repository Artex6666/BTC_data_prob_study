"""
lottery_optimizer_v2.py — Optimiseur "Lottery Long" BTC M5 — Formule dynamique

Formule miroir de la God Equation :
  God curve  : buffer_requis  > slope × T + intercept   (rester LOIN du strike)
  Lottery v2 : |spot - open|  ≤ slope × T + intercept   (rester PROCHE du strike)

Condition d'entrée complète :
  cheap_bid <= price_max
  AND |spot - open| <= slope × secs_left + intercept    ← fenêtre qui se resserre
  AND secs_lo <= secs_left <= secs_hi

Plus T diminue → plafond de proximité se resserre → on n'entre que si on est vraiment
collé au strike sur la fin.

Grid :
  price_max  : 0.01, 0.02, 0.03, 0.05       (4)
  slope      : 0.25, 0.5, 1.0, 2.0, 5.0     (5)   $/seconde
  intercept  : 0, 5, 10, 15, 25             (5)   $ (plafond à T≈0)
  secs_lo/hi : 18 paires valides             (18)

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
DEFAULT_SIZE      = 10.0
VOL_WINDOW_S      = 30.0
SCAN_WINDOW_S     = 60

_CONTRACTS   = None
_TOTAL_HOURS = None


# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
def build_configs():
    configs = []
    price_grid   = [0.01, 0.02, 0.03, 0.05]
    slope_grid   = [0.25, 0.5, 1.0, 2.0, 5.0]
    intcp_grid   = [0.0, 5.0, 10.0, 15.0, 25.0]
    secs_lo_g    = [0, 5, 10, 20]
    secs_hi_g    = [15, 20, 30, 45, 60]

    for pmax in price_grid:
        for slope in slope_grid:
            for intercept in intcp_grid:
                for slo in secs_lo_g:
                    for shi in secs_hi_g:
                        if slo >= shi:
                            continue
                        configs.append(dict(
                            price_max  = pmax,
                            slope      = float(slope),
                            intercept  = float(intercept),
                            secs_lo    = float(slo),
                            secs_hi    = float(shi),
                        ))
    return configs  # 1800 configs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_settlement(asset='btc', tf='5min'):
    df = pd.read_csv(SETTLEMENT_CSV)
    df = df[(df['asset'] == asset) & (df['tf'] == tf)]
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


def load_dataset(csv_paths):
    global _CONTRACTS, _TOTAL_HOURS

    print("[*] Chargement CSV ...", flush=True)
    dfs = []
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            print(f"    [SKIP] {p}", flush=True)
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
            'ce_date':   ce.date(),
            'ce':        ce,
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
# Workers
# ─────────────────────────────────────────────────────────────────────────────
def _init_worker(pickle_path: str):
    global _CONTRACTS, _TOTAL_HOURS
    with open(pickle_path, 'rb') as f:
        payload = pickle.load(f)
    _CONTRACTS   = payload['contracts']
    _TOTAL_HOURS = payload['total_hours']


def _simulate_config(c, cfg):
    """
    Condition d'entrée : cheap_bid <= price_max ET |spot-open| <= slope*T + intercept.
    Retourne (won, entry_price) ou None.
    """
    pmax     = cfg['price_max']
    slope    = cfg['slope']
    intcp    = cfg['intercept']
    slo      = cfg['secs_lo']
    shi      = cfg['secs_hi']
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

        # Plafond dynamique : plus T est petit, plus la fenêtre est serrée
        prox_limit = slope * sl + intcp
        buf = abs(spots[j] - op)
        if buf > prox_limit:
            continue

        ub = up_bid[j]
        db = down_bid[j]
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
            price = max(price, 1e-4)

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
            'max_dd':      max_dd,
            'rf':          rf,
            'total_hours': float(_TOTAL_HOURS),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Formatage
# ─────────────────────────────────────────────────────────────────────────────
def fmt_row(i, r, days):
    # Exemple de plafond à T=60s, T=30s, T=10s, T=0s
    p60  = r['slope'] * 60 + r['intercept']
    p30  = r['slope'] * 30 + r['intercept']
    p10  = r['slope'] * 10 + r['intercept']
    p0   = r['intercept']
    edge_m = "★" if r['edge'] > 0 else " "
    return (
        f"{i:>4d} | "
        f"p≤{r['price_max']:.2f} "
        f"eq={r['slope']:.2f}xT+{r['intercept']:.0f} "
        f"[60s:{p60:>5.1f}$ 30s:{p30:>5.1f}$ 10s:{p10:>4.1f}$ 0s:{p0:>4.1f}$] "
        f"t=[{r['secs_lo']:>2.0f}s-{r['secs_hi']:>2.0f}s] | "
        f"T={r['trades']:>5d} ({r['trades']/days:>5.1f}/j) "
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
    ap = argparse.ArgumentParser(description="Lottery Optimizer v2 — formule dynamique |spot-open| ≤ slope×T+intercept")
    ap.add_argument('--csv',     nargs='+', default=DEFAULT_CSV_PATHS)
    ap.add_argument('--workers', type=int,  default=DEFAULT_N_WORKERS)
    ap.add_argument('--threads', action='store_true')
    ap.add_argument('--size',    type=float, default=10.0, help='$ par trade')
    args = ap.parse_args()

    global DEFAULT_SIZE, _CONTRACTS, _TOTAL_HOURS
    DEFAULT_SIZE = args.size

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / f"lottery_btc_v2_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"lottery_v2_results_{run_ts}.txt"

    t0 = time.time()
    configs = build_configs()
    n_workers = max(1, args.workers)

    print(f"\n{'='*90}")
    print(f"  Lottery Optimizer v2 — BTC M5 — Formule dynamique slope×T+intercept")
    print(f"  Run: {run_ts}  |  Configs: {len(configs)}  |  Workers: {n_workers}")
    print(f"  Entrée : cheap_bid ≤ price_max  ET  |spot-open| ≤ slope×T+intercept")
    print(f"  Size: {args.size}$/trade  |  Sortie: {run_dir.name}")
    print(f"{'='*90}\n")

    n_ct = load_dataset(args.csv)

    batch_size = max(1, len(configs) // n_workers + 1)
    batches = [configs[i:i+batch_size] for i in range(0, len(configs), batch_size)]
    print(f"\n[*] {len(batches)} batches × ~{batch_size} configs\n", flush=True)

    all_results = []

    if args.threads:
        print("  Backend: threads\n", flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker_fn, b) for b in batches]
            for i, fut in enumerate(as_completed(futures), 1):
                all_results.extend(fut.result())
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i}/{len(batches)} — {len(all_results)} résultats", flush=True)
    else:
        print("  Backend: processus (vrai multi-coeur)\n", flush=True)
        payload_path = run_dir / "_worker_data.pkl"
        with open(payload_path, 'wb') as f:
            pickle.dump({'contracts': _CONTRACTS, 'total_hours': _TOTAL_HOURS}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        sz_mb = payload_path.stat().st_size / (1024*1024)
        print(f"  Snapshot: {sz_mb:.1f} MiB  (×{n_workers})\n", flush=True)
        _CONTRACTS = None; _TOTAL_HOURS = None
        try:
            with mp.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(str(payload_path.resolve()),),
            ) as pool:
                for i, br in enumerate(pool.imap_unordered(worker_fn, batches), 1):
                    all_results.extend(br)
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i}/{len(batches)} — {len(all_results)} résultats", flush=True)
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
        [r for r in all_results if r['trades'] >= 100],
        key=lambda x: (x['losing_days'], -x['rf'])
    )

    hdr = (
        f"  {'#':>4}   {'p_max':>5} {'equation':>18} {'plafonds':>38} {'t_range':>10}   "
        f"{'trades':>6} {'T/j':>5}   "
        f"{'WR':>6} {'BE':>5} {'edge':>7}   "
        f"{'EV$/t':>8}   "
        f"{'PNL_tot':>11} {'PNL/j':>9}   "
        f"{'maxDD':>7} {'RF':>5}   "
        f"{'loseDays':>9} {'worstDay':>10}\n"
    )
    sep = "─" * 180 + "\n"

    lines = [
        f"Lottery Optimizer v2 — BTC M5 — |spot-open| ≤ slope×T+intercept  |  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Data: {days:.1f}j  |  Configs: {len(all_results)}/{len(configs)}  |  "
        f"Size: {args.size}$/trade  |  {elapsed:.0f}s\n\n",
    ]

    sections = [
        ("TOP 50 — Edge Ratio (WR/BE, edge>0)", by_edge_ratio, 50),
        ("TOP 50 — EV par trade ($/trade)",      by_ev,         50),
        ("TOP 50 — Recovery Factor",             by_rf,         50),
        ("TOP 40 — PNL total",                   by_pnl,        40),
        ("TOP 30 — Consistance (min loseDays, ≥100 trades)", by_cons, 30),
    ]

    for title, ranked, n in sections:
        lines.append(f"\n{'='*180}\n>>> {title}\n{'='*180}\n{hdr}{sep}")
        for i, r in enumerate(ranked[:n], 1):
            lines.append(fmt_row(i, r, days))

    # ── Comparaison v1 vs v2 ──────────────────────────────────────────────────
    # Meilleure config v2 par RF
    best_rf  = by_rf[0]  if by_rf  else None
    best_ev  = by_ev[0]  if by_ev  else None
    best_era = by_edge_ratio[0] if by_edge_ratio else None

    lines.append(f"\n{'='*180}\n>>> SYNTHÈSE v1 → v2  (formule statique vs dynamique)\n{'='*180}\n")
    lines.append(
        "  v1 (statique) : condition fixe  |spot-open| ≤ prox_max\n"
        "  v2 (dynamique): condition tightening  |spot-open| ≤ slope×T + intercept\n\n"
        "  Effet attendu : la v2 filtre mieux les entrées tardives éloignées du strike,\n"
        "  accepte les entrées précoces un peu plus loin, et se resserre naturellement\n"
        "  vers l'expiry pour ne garder que les vraies opportunités de crossing.\n\n"
    )
    if best_era:
        p60 = best_era['slope']*60 + best_era['intercept']
        p0  = best_era['intercept']
        lines.append(
            f"  Meilleure config v2 (edge ratio) :\n"
            f"    price_max={best_era['price_max']:.2f}  "
            f"slope={best_era['slope']:.2f}  intercept={best_era['intercept']:.0f}\n"
            f"    Plafond prox : {p60:.1f}$ à T=60s → {p0:.1f}$ à T=0s\n"
            f"    WR={best_era['wr']:.1f}% vs BE={best_era['break_even']:.1f}%  "
            f"edge={best_era['edge']:+.1f}%  edge_ratio={best_era['edge_ratio']:.2f}x\n"
            f"    EV={best_era['ev_trade']:+.2f}$/trade  RF={best_era['rf']:.1f}x  "
            f"loseDays={best_era['losing_days']}/{int(days)}\n"
        )

    output = "".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n{'='*90}")
    print(f"  Terminé en {elapsed:.0f}s  |  {len(all_results)} configs évaluées")
    print(f"  Sortie : {out_path}")
    print(f"{'='*90}\n")

    # Console : top 25 de chaque section
    for title, ranked, n in [
        ("TOP 25 EDGE RATIO", by_edge_ratio, 25),
        ("TOP 25 EV/TRADE",   by_ev,         25),
        ("TOP 25 RF",         by_rf,         25),
    ]:
        print(f"\n>>> {title}\n{hdr}{sep}")
        for i, r in enumerate(ranked[:n], 1):
            print(fmt_row(i, r, days), end='')


if __name__ == '__main__':
    main()
