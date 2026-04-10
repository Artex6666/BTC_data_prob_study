"""
lottery_optimizer_v4_fillsim.py — Fill maker réaliste : fill si ask ≤ entry_bid dans un tick futur.

Modes de fill :
  'instant'  : fill immédiat au bid dès le signal (v1-v3, optimiste)
  'ask_cross': fill si dans un tick futur (avant expiry) cheap_ask <= entry_bid
               → simule un vendeur qui tape ton bid
  'taker'    : fill immédiat au ask du tick signal (pessimiste)

Grid identique v2/v3 : 1800 configs × 3 modes = 5400 runs.
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
SCAN_WINDOW_S     = 60

_CONTRACTS   = None
_TOTAL_HOURS = None

FILL_MODES = ('instant', 'ask_cross', 'taker')


# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
def build_configs():
    configs = []
    for pmax in [0.01, 0.02, 0.03, 0.05]:
        for slope in [0.25, 0.5, 1.0, 2.0, 5.0]:
            for intercept in [0.0, 5.0, 10.0, 15.0, 25.0]:
                for slo in [0, 5, 10, 20]:
                    for shi in [15, 20, 30, 45, 60]:
                        if slo >= shi:
                            continue
                        for mode in FILL_MODES:
                            configs.append(dict(
                                price_max = pmax,
                                slope     = float(slope),
                                intercept = float(intercept),
                                secs_lo   = float(slo),
                                secs_hi   = float(shi),
                                fill_mode = mode,
                            ))
    return configs  # 5400 configs


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_settlement():
    df = pd.read_csv(SETTLEMENT_CSV)
    df = df[(df['asset'] == 'btc') & (df['tf'] == '5min')]
    return {int(r.open_ts): (float(r.outcome_up) > 0.5) for _, r in df.iterrows()}


def load_dataset(csv_paths):
    global _CONTRACTS, _TOTAL_HOURS
    print("[*] Chargement CSV ...", flush=True)
    dfs = []
    for p in csv_paths:
        path = Path(p)
        if not path.exists():
            continue
        d = pd.read_csv(path, usecols=[
            'timestamp', 'spot_price',
            'm5_up_bid', 'm5_down_bid',
            'm5_up_ask', 'm5_down_ask',
        ])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
        print(f"    {path.name}: {len(d):,} ticks", flush=True)

    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract_ce'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)
    outcomes = load_settlement()
    print(f"    Settlement: {len(outcomes):,} contrats", flush=True)

    contracts = []
    for ce, grp in df.groupby('contract_ce', sort=True):
        open_ts = int((ce - pd.Timedelta(minutes=5)).timestamp())
        if open_ts not in outcomes:
            continue
        op_spot = float(grp['spot_price'].iloc[0])
        wnd = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=SCAN_WINDOW_S)].reset_index(drop=True)
        if len(wnd) < 2:
            continue
        spots     = wnd['spot_price'].values.astype(np.float64)
        ts_ns     = wnd['timestamp'].values.astype(np.int64)
        secs_left = (ce.value - ts_ns) / 1e9

        contracts.append({
            'won_up':    outcomes[open_ts],
            'op':        op_spot,
            'spot':      spots,
            'secs_left': secs_left,
            'up_bid':    wnd['m5_up_bid'].values.astype(np.float64),
            'down_bid':  wnd['m5_down_bid'].values.astype(np.float64),
            'up_ask':    wnd['m5_up_ask'].values.astype(np.float64),
            'down_ask':  wnd['m5_down_ask'].values.astype(np.float64),
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
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
def _init_worker(pkl):
    global _CONTRACTS, _TOTAL_HOURS
    with open(pkl, 'rb') as f:
        p = pickle.load(f)
    _CONTRACTS, _TOTAL_HOURS = p['contracts'], p['total_hours']


def _simulate_contract(c, cfg):
    """
    Retourne (won, fill_price, filled) ou None si pas de signal.

    fill_mode='instant'  : fill au bid du tick signal, toujours filled
    fill_mode='ask_cross': signal au bid, fill si ask <= bid dans un tick futur
    fill_mode='taker'    : fill au ask du tick signal si ask > 0
    """
    pmax      = cfg['price_max']
    slope     = cfg['slope']
    intcp     = cfg['intercept']
    slo       = cfg['secs_lo']
    shi       = cfg['secs_hi']
    mode      = cfg['fill_mode']
    op        = c['op']
    spots     = c['spot']
    up_bid    = c['up_bid']
    dn_bid    = c['down_bid']
    up_ask    = c['up_ask']
    dn_ask    = c['down_ask']
    secs_left = c['secs_left']
    won_up    = c['won_up']
    n         = len(spots)

    for j in range(n):
        sl = secs_left[j]
        if sl < slo or sl > shi:
            continue
        if abs(spots[j] - op) > slope * sl + intcp:
            continue

        ub = up_bid[j]; db = dn_bid[j]
        ua = up_ask[j]; da = dn_ask[j]
        if ub <= 0.0 and db <= 0.0:
            continue

        if ub <= 0.0:
            cheap_bid, cheap_ask, cheap_is_up = db, da, False
        elif db <= 0.0:
            cheap_bid, cheap_ask, cheap_is_up = ub, ua, True
        elif ub <= db:
            cheap_bid, cheap_ask, cheap_is_up = ub, ua, True
        else:
            cheap_bid, cheap_ask, cheap_is_up = db, da, False

        if cheap_bid > pmax:
            continue

        won = won_up if cheap_is_up else (not won_up)

        if mode == 'instant':
            return won, max(cheap_bid, 1e-4), True

        elif mode == 'taker':
            if cheap_ask <= 0:
                return None  # pas d'ask → skip
            return won, max(cheap_ask, 1e-4), True

        else:  # ask_cross : fill si ask descend ≤ entry_bid dans les ticks suivants
            entry_bid = max(cheap_bid, 1e-4)
            # Chercher dans les ticks j+1 ... n-1 si cheap_ask <= entry_bid
            for k in range(j + 1, n):
                if cheap_is_up:
                    fut_ask = up_ask[k]
                else:
                    fut_ask = dn_ask[k]
                if fut_ask <= 0:
                    continue
                if fut_ask <= entry_bid:
                    return won, entry_bid, True
            # Pas de fill avant expiry
            return won, entry_bid, False  # signal mais pas fill

    return None


def worker_fn(config_batch):
    global _CONTRACTS, _TOTAL_HOURS
    results = []
    size    = DEFAULT_SIZE

    for cfg in config_batch:
        wins = 0; losses = 0; no_fills = 0
        total_pnl     = 0.0
        contract_pnls = []
        day_pnls      = {}

        for c in _CONTRACTS:
            res = _simulate_contract(c, cfg)
            if res is None:
                continue
            won, fill_price, filled = res

            if not filled:
                no_fills += 1
                continue  # ordre posé mais jamais fill

            pnl = size * (1.0 / fill_price - 1.0) if won else -size
            if won:
                wins += 1
            else:
                losses += 1
            total_pnl += pnl
            contract_pnls.append(pnl)
            day_pnls[c['ce_date']] = day_pnls.get(c['ce_date'], 0.0) + pnl

        trades = wins + losses
        if trades == 0:
            continue

        wr       = wins / trades * 100.0
        ev_trade = total_pnl / trades
        losing_d = sum(1 for v in day_pnls.values() if v < 0)
        worst_d  = min(day_pnls.values()) if day_pnls else 0.0

        cum = np.cumsum([0.0] + contract_pnls)
        pk = 0.0; mdd = 0.0
        for v in cum:
            if v > pk: pk = v
            d = pk - v
            if d > mdd: mdd = d

        rf = total_pnl / max(mdd, size) if mdd > 0 else 0.0

        results.append({**cfg,
            'trades':      trades,
            'no_fills':    no_fills,
            'wins':        wins,
            'wr':          wr,
            'edge':        wr - cfg['price_max'] * 100.0,
            'ev_trade':    ev_trade,
            'total_pnl':   total_pnl,
            'losing_days': losing_d,
            'worst_day':   worst_d,
            'max_dd':      mdd,
            'rf':          rf,
            'total_hours': float(_TOTAL_HOURS),
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Formatage
# ─────────────────────────────────────────────────────────────────────────────
def fmt_row(i, r, days):
    p60 = r['slope'] * 60 + r['intercept']
    p10 = r['slope'] * 10 + r['intercept']
    p0  = r['intercept']
    em  = "★" if r['edge'] > 0 else " "
    nf  = f"nofill={r['no_fills']}" if r['fill_mode'] == 'ask_cross' else ""
    return (
        f"{i:>4d} [{r['fill_mode']:>9s}] "
        f"p≤{r['price_max']:.2f} {r['slope']:.2f}xT+{r['intercept']:.0f} "
        f"[60s:{p60:>5.1f}$ 10s:{p10:>4.1f}$ 0s:{p0:>3.0f}$] "
        f"t=[{r['secs_lo']:>2.0f}s-{r['secs_hi']:>2.0f}s] | "
        f"T={r['trades']:>5d} ({r['trades']/days:>5.1f}/j) {nf:>12s} "
        f"WR={r['wr']:>5.1f}% edge={r['edge']:>+5.1f}% {em} "
        f"EV={r['ev_trade']:>+7.2f}$/t "
        f"PNL={r['total_pnl']:>+9.0f}$ ({r['total_pnl']/days:>+6.1f}$/j) "
        f"maxDD={r['max_dd']:>6.0f}$ RF={r['rf']:>5.1f}x "
        f"lose={r['losing_days']:>3d}/{int(days)} wrstDay={r['worst_day']:>+7.0f}$\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',     nargs='+', default=DEFAULT_CSV_PATHS)
    ap.add_argument('--workers', type=int,  default=DEFAULT_N_WORKERS)
    ap.add_argument('--threads', action='store_true')
    ap.add_argument('--size',    type=float, default=10.0)
    args = ap.parse_args()

    global DEFAULT_SIZE, _CONTRACTS, _TOTAL_HOURS
    DEFAULT_SIZE = args.size

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / f"lottery_btc_v4_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"lottery_v4_results_{run_ts}.txt"

    t0 = time.time()
    configs   = build_configs()
    n_workers = max(1, args.workers)

    print(f"\n{'='*90}")
    print(f"  Lottery v4 — 3 modes de fill : instant / ask_cross / taker")
    print(f"  {len(configs)} configs  |  Workers: {n_workers}")
    print(f"  ask_cross = fill maker réaliste : sell doit taper le bid dans un tick futur")
    print(f"{'='*90}\n")

    load_dataset(args.csv)

    batch_size = max(1, len(configs) // n_workers + 1)
    batches    = [configs[i:i+batch_size] for i in range(0, len(configs), batch_size)]
    print(f"[*] {len(batches)} batches × ~{batch_size} configs\n")

    all_results = []
    if args.threads:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker_fn, b) for b in batches]
            for i, fut in enumerate(as_completed(futures), 1):
                all_results.extend(fut.result())
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i}/{len(batches)} — {len(all_results)}", flush=True)
    else:
        pkl = run_dir / "_data.pkl"
        with open(pkl, 'wb') as f:
            pickle.dump({'contracts': _CONTRACTS, 'total_hours': _TOTAL_HOURS}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        sz = pkl.stat().st_size / 1024**2
        print(f"  Snapshot: {sz:.1f} MiB  (×{n_workers})\n", flush=True)
        _CONTRACTS = None; _TOTAL_HOURS = None
        try:
            with mp.Pool(n_workers, initializer=_init_worker, initargs=(str(pkl.resolve()),)) as pool:
                for i, br in enumerate(pool.imap_unordered(worker_fn, batches), 1):
                    all_results.extend(br)
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i}/{len(batches)} — {len(all_results)}", flush=True)
        finally:
            pkl.unlink(missing_ok=True)

    elapsed = time.time() - t0
    days    = all_results[0]['total_hours'] / 24

    by_mode = {m: [r for r in all_results if r['fill_mode'] == m] for m in FILL_MODES}

    sep = "─" * 170 + "\n"
    hdr = (
        f"  {'#':>4} {'fill_mode':>10}  {'config':>46}  "
        f"{'T':>6} {'T/j':>5}  {'WR':>6} {'edge':>6}  "
        f"{'EV$/t':>8}  {'PNL':>9} {'PNL/j':>7}  "
        f"{'maxDD':>7} {'RF':>5}  {'lose':>5}\n"
    )

    lines = [
        f"Lottery v4 — 3 modes fill  |  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Data: {days:.1f}j  |  {len(all_results)} configs  |  {elapsed:.0f}s\n\n"
    ]

    # Top 30 RF par mode
    for mode in FILL_MODES:
        res   = by_mode[mode]
        by_rf = sorted([r for r in res if r['total_pnl'] > 0], key=lambda x: -x['rf'])
        by_ev = sorted(res, key=lambda x: -x['ev_trade'])
        lines.append(f"\n{'='*170}\n>>> {mode.upper()} — TOP 30 RF\n{'='*170}\n{hdr}{sep}")
        for i, r in enumerate(by_rf[:30], 1):
            lines.append(fmt_row(i, r, days))
        lines.append(f"\n>>> {mode.upper()} — TOP 30 EV/trade\n{sep}")
        for i, r in enumerate(by_ev[:30], 1):
            lines.append(fmt_row(i, r, days))

    # Comparaison directe sur les top maker par RF
    instant_by_rf = sorted([r for r in by_mode['instant'] if r['total_pnl'] > 0], key=lambda x: -x['rf'])
    idx = {}
    for mode in ('ask_cross', 'taker'):
        idx[mode] = {
            (r['price_max'], r['slope'], r['intercept'], r['secs_lo'], r['secs_hi']): r
            for r in by_mode[mode]
        }

    lines.append(f"\n{'='*170}\n>>> COMPARAISON DIRECTE — même config, 3 modes (top 40 instant par RF)\n{'='*170}\n")
    lines.append(
        f"  {'Config':>52}  "
        f"{'inst_EV':>9} {'cross_EV':>9} {'take_EV':>9}  "
        f"{'inst_RF':>8} {'cross_RF':>8} {'take_RF':>8}  "
        f"{'inst_T':>7} {'cross_T':>7} {'take_T':>7}  "
        f"{'nofill%':>8}  winner\n"
    )
    lines.append("  " + "─"*160 + "\n")

    for r_i in instant_by_rf[:40]:
        key = (r_i['price_max'], r_i['slope'], r_i['intercept'], r_i['secs_lo'], r_i['secs_hi'])
        r_c = idx['ask_cross'].get(key)
        r_t = idx['taker'].get(key)
        if not r_c or not r_t:
            continue

        p60 = r_i['slope']*60 + r_i['intercept']
        p0  = r_i['intercept']
        cfg_s = (f"p≤{r_i['price_max']:.2f} {r_i['slope']:.2f}xT+{r_i['intercept']:.0f} "
                 f"[60s:{p60:.0f}$ 0s:{p0:.0f}$] t=[{r_i['secs_lo']:.0f}s-{r_i['secs_hi']:.0f}s]")

        total_signals = r_c['trades'] + r_c['no_fills']
        nofill_pct = r_c['no_fills'] / total_signals * 100 if total_signals > 0 else 0

        evs  = [r_i['ev_trade'], r_c['ev_trade'], r_t['ev_trade']]
        winner = ['instant','ask_cross','taker'][evs.index(max(evs))]

        lines.append(
            f"  {cfg_s:>52}  "
            f"{r_i['ev_trade']:>+9.2f}$ {r_c['ev_trade']:>+9.2f}$ {r_t['ev_trade']:>+9.2f}$  "
            f"{r_i['rf']:>8.1f}x {r_c['rf']:>8.1f}x {r_t['rf']:>8.1f}x  "
            f"{r_i['trades']:>7d} {r_c['trades']:>7d} {r_t['trades']:>7d}  "
            f"{nofill_pct:>7.1f}%  {winner}\n"
        )

    # Résumé global
    lines.append(f"\n{'='*170}\n>>> RÉSUMÉ GLOBAL\n{'='*170}\n")
    for mode in FILL_MODES:
        res     = by_mode[mode]
        n_pos   = sum(1 for r in res if r['ev_trade'] > 0)
        ev_mean = np.mean([r['ev_trade'] for r in res])
        rf_max  = max((r['rf'] for r in res if r['total_pnl'] > 0), default=0)
        nf_mean = (np.mean([r['no_fills'] for r in res if r['fill_mode']=='ask_cross'])
                   if mode == 'ask_cross' else 0)
        lines.append(
            f"  {mode:>10s} : EV>0={n_pos}/{len(res)}  "
            f"EV_moy={ev_mean:>+7.2f}$/t  RF_max={rf_max:>6.1f}x"
            + (f"  no_fill_moy={nf_mean:.1f}/config" if mode == 'ask_cross' else "")
            + "\n"
        )

    output = "".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)

    # ── Console ───────────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  Terminé : {elapsed:.0f}s  |  {len(all_results)} configs")
    print(f"  Sortie  : {out_path}")
    print(f"{'='*90}\n")

    for mode in FILL_MODES:
        res   = by_mode[mode]
        by_rf = sorted([r for r in res if r['total_pnl'] > 0], key=lambda x: -x['rf'])
        print(f"\n>>> TOP 15 {mode.upper()} (RF)\n{hdr}{sep}")
        for i, r in enumerate(by_rf[:15], 1):
            print(fmt_row(i, r, days), end='')

    print(f"\n>>> COMPARAISON DIRECTE — top 20 instant par RF\n")
    print(f"  {'Config':>52}  {'inst_EV':>8} {'cross_EV':>9} {'take_EV':>8}  {'inst_RF':>7} {'cross_RF':>8}  {'no_fill%':>9}  winner")
    print(f"  {'─'*140}")
    for r_i in instant_by_rf[:20]:
        key = (r_i['price_max'], r_i['slope'], r_i['intercept'], r_i['secs_lo'], r_i['secs_hi'])
        r_c = idx['ask_cross'].get(key)
        r_t = idx['taker'].get(key)
        if not r_c or not r_t: continue
        p60 = r_i['slope']*60 + r_i['intercept']
        p0  = r_i['intercept']
        cfg_s = f"p≤{r_i['price_max']:.2f} {r_i['slope']:.2f}xT+{r_i['intercept']:.0f} [60s:{p60:.0f}$ 0s:{p0:.0f}$] t=[{r_i['secs_lo']:.0f}s-{r_i['secs_hi']:.0f}s]"
        total_sig = r_c['trades'] + r_c['no_fills']
        nfpct = r_c['no_fills'] / total_sig * 100 if total_sig > 0 else 0
        evs   = [r_i['ev_trade'], r_c['ev_trade'], r_t['ev_trade']]
        w     = ['inst','cross','take'][evs.index(max(evs))]
        print(f"  {cfg_s:>52}  {r_i['ev_trade']:>+8.2f}$ {r_c['ev_trade']:>+9.2f}$ {r_t['ev_trade']:>+8.2f}$  {r_i['rf']:>7.1f}x {r_c['rf']:>8.1f}x  {nfpct:>8.1f}%  {w}")

    print(f"\n{'='*90}")
    for mode in FILL_MODES:
        res     = by_mode[mode]
        n_pos   = sum(1 for r in res if r['ev_trade'] > 0)
        ev_mean = np.mean([r['ev_trade'] for r in res])
        rf_max  = max((r['rf'] for r in res if r['total_pnl'] > 0), default=0)
        print(f"  {mode:>10s} : EV>0={n_pos}/{len(res)}  EV_moy={ev_mean:>+7.2f}$/t  RF_max={rf_max:.1f}x")
    print(f"{'='*90}\n")


if __name__ == '__main__':
    main()
