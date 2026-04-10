"""
lottery_optimizer_v3_fillmode.py — Comparaison Maker vs Taker sur toute la grid v2.

Pour chaque config on tourne 2 fois :
  - MAKER : fill au BID  (limit order, on attend qu'un vendeur tape notre bid)
  - TAKER : fill au ASK  (market order, on paye le ask immédiatement)

Si ask=0 (pas d'ask dispo) → on skip le trade dans le mode TAKER.

Grid identique à v2 : price_max × slope × intercept × secs_lo/hi = 1800 configs × 2 modes = 3600 runs.
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


# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
def build_configs():
    configs = []
    price_grid = [0.01, 0.02, 0.03, 0.05]
    slope_grid = [0.25, 0.5, 1.0, 2.0, 5.0]
    intcp_grid = [0.0, 5.0, 10.0, 15.0, 25.0]
    secs_lo_g  = [0, 5, 10, 20]
    secs_hi_g  = [15, 20, 30, 45, 60]

    for pmax in price_grid:
        for slope in slope_grid:
            for intercept in intcp_grid:
                for slo in secs_lo_g:
                    for shi in secs_hi_g:
                        if slo >= shi:
                            continue
                        for fill_mode in ('maker', 'taker'):
                            configs.append(dict(
                                price_max  = pmax,
                                slope      = float(slope),
                                intercept  = float(intercept),
                                secs_lo    = float(slo),
                                secs_hi    = float(shi),
                                fill_mode  = fill_mode,
                            ))
    return configs  # 3600 configs


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

    outcomes = load_settlement()
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

        spots     = wnd['spot_price'].values.astype(np.float64)
        ts_ns     = wnd['timestamp'].values.astype(np.int64)
        ce_ns     = ce.value
        secs_left = (ce_ns - ts_ns) / 1e9

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
    Retourne (won, fill_price) ou None.
    fill_mode='maker' → fill au cheap_bid
    fill_mode='taker' → fill au cheap_ask (skip si ask=0)
    """
    pmax      = cfg['price_max']
    slope     = cfg['slope']
    intcp     = cfg['intercept']
    slo       = cfg['secs_lo']
    shi       = cfg['secs_hi']
    fill_mode = cfg['fill_mode']
    op        = c['op']

    spots     = c['spot']
    up_bid    = c['up_bid']
    down_bid  = c['down_bid']
    up_ask    = c['up_ask']
    down_ask  = c['down_ask']
    secs_left = c['secs_left']
    won_up    = c['won_up']

    for j in range(len(spots)):
        sl = secs_left[j]
        if sl < slo or sl > shi:
            continue

        prox_limit = slope * sl + intcp
        buf = abs(spots[j] - op)
        if buf > prox_limit:
            continue

        ub = up_bid[j]; db = down_bid[j]
        ua = up_ask[j]; da = down_ask[j]
        if ub <= 0.0 and db <= 0.0:
            continue

        # Côté le moins cher (bid side)
        if ub <= 0.0:
            cheap_bid, cheap_ask, cheap_is_up = db, da, False
        elif db <= 0.0:
            cheap_bid, cheap_ask, cheap_is_up = ub, ua, True
        elif ub <= db:
            cheap_bid, cheap_ask, cheap_is_up = ub, ua, True
        else:
            cheap_bid, cheap_ask, cheap_is_up = db, da, False

        # Filtre sur le bid (signal de déclenchement dans les 2 modes)
        if cheap_bid > pmax:
            continue

        # Prix de fill selon le mode
        if fill_mode == 'maker':
            fill_price = cheap_bid
        else:  # taker
            fill_price = cheap_ask
            if fill_price <= 0:
                continue  # pas d'ask dispo → skip

        won = won_up if cheap_is_up else (not won_up)
        return won, max(fill_price, 1e-4)

    return None


def worker_fn(config_batch):
    global _CONTRACTS, _TOTAL_HOURS
    contracts = _CONTRACTS
    size      = DEFAULT_SIZE
    results   = []

    for cfg in config_batch:
        wins = 0; losses = 0; skipped_no_ask = 0
        total_pnl = 0.0
        contract_pnls = []
        day_pnls = {}

        for c in contracts:
            res = _simulate_config(c, cfg)
            if res is None:
                continue
            won, fill_price = res

            if won:
                pnl = size * (1.0 / fill_price - 1.0)
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
        # break-even dépend du fill_mode : au bid → BE=price_max, au ask c'est variable
        # on garde price_max comme référence du signal
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
            'wr':          wr,
            'break_even':  break_even,
            'edge':        edge,
            'ev_trade':    ev_trade,
            'total_pnl':   total_pnl,
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
    p60 = r['slope'] * 60 + r['intercept']
    p10 = r['slope'] * 10 + r['intercept']
    p0  = r['intercept']
    em  = "★" if r['edge'] > 0 else " "
    return (
        f"{i:>4d} [{r['fill_mode']:>5s}] "
        f"p≤{r['price_max']:.2f} {r['slope']:.2f}xT+{r['intercept']:.0f} "
        f"60s:{p60:>5.1f}$ 10s:{p10:>4.1f}$ 0s:{p0:>3.0f}$ "
        f"t=[{r['secs_lo']:>2.0f}s-{r['secs_hi']:>2.0f}s] | "
        f"T={r['trades']:>5d} ({r['trades']/days:>5.1f}/j) "
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
    run_dir = Path(__file__).resolve().parent / f"lottery_btc_v3_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"lottery_v3_results_{run_ts}.txt"

    t0      = time.time()
    configs = build_configs()
    n_workers = max(1, args.workers)

    print(f"\n{'='*90}")
    print(f"  Lottery v3 — Maker vs Taker — BTC M5")
    print(f"  {len(configs)} configs ({len(configs)//2} × 2 modes)  |  Workers: {n_workers}")
    print(f"{'='*90}\n")

    load_dataset(args.csv)

    batch_size = max(1, len(configs) // n_workers + 1)
    batches    = [configs[i:i+batch_size] for i in range(0, len(configs), batch_size)]
    print(f"\n[*] {len(batches)} batches × ~{batch_size} configs\n")

    all_results = []
    if args.threads:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker_fn, b) for b in batches]
            for i, fut in enumerate(as_completed(futures), 1):
                all_results.extend(fut.result())
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i}/{len(batches)} — {len(all_results)} résultats", flush=True)
    else:
        payload_path = run_dir / "_data.pkl"
        with open(payload_path, 'wb') as f:
            pickle.dump({'contracts': _CONTRACTS, 'total_hours': _TOTAL_HOURS}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        sz = payload_path.stat().st_size / 1024**2
        print(f"  Snapshot: {sz:.1f} MiB  (×{n_workers})\n", flush=True)
        _CONTRACTS = None; _TOTAL_HOURS = None
        try:
            with mp.Pool(n_workers, initializer=_init_worker,
                         initargs=(str(payload_path.resolve()),)) as pool:
                for i, br in enumerate(pool.imap_unordered(worker_fn, batches), 1):
                    all_results.extend(br)
                    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {i}/{len(batches)} — {len(all_results)} résultats", flush=True)
        finally:
            payload_path.unlink(missing_ok=True)

    if not all_results:
        print("[!] Aucun résultat."); return

    elapsed = time.time() - t0
    days    = all_results[0]['total_hours'] / 24

    maker_res = [r for r in all_results if r['fill_mode'] == 'maker']
    taker_res = [r for r in all_results if r['fill_mode'] == 'taker']

    sep = "─" * 170 + "\n"
    hdr = (
        f"  {'#':>4} {'mode':>7}  {'config':>40}  "
        f"{'T':>6} {'T/j':>5}  {'WR':>6} {'edge':>6}  "
        f"{'EV$/t':>7}  {'PNL':>9} {'PNL/j':>7}  "
        f"{'maxDD':>6} {'RF':>5}  {'lose':>5} {'wrstDay':>8}\n"
    )

    lines = [
        f"Lottery v3 — Maker vs Taker — BTC M5  |  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Data: {days:.1f}j  |  "
        f"Configs: {len(all_results)}/3600  |  Size: {args.size}$/t  |  {elapsed:.0f}s\n\n"
    ]

    # ── Top 40 par RF pour chaque mode ────────────────────────────────────────
    for mode_label, res in [("MAKER (fill au bid)", maker_res), ("TAKER (fill au ask)", taker_res)]:
        by_rf  = sorted([r for r in res if r['total_pnl'] > 0], key=lambda x: -x['rf'])
        by_ev  = sorted(res, key=lambda x: -x['ev_trade'])
        by_era = sorted([r for r in res if r['edge'] > 0], key=lambda x: (-x['ev_trade']))

        lines.append(f"\n{'='*170}\n>>> {mode_label} — TOP 30 RF\n{'='*170}\n{hdr}{sep}")
        for i, r in enumerate(by_rf[:30], 1):
            lines.append(fmt_row(i, r, days))

        lines.append(f"\n{'='*170}\n>>> {mode_label} — TOP 30 EV/trade\n{'='*170}\n{hdr}{sep}")
        for i, r in enumerate(by_ev[:30], 1):
            lines.append(fmt_row(i, r, days))

    # ── Comparaison directe : même config, maker vs taker ────────────────────
    lines.append(f"\n{'='*170}\n>>> COMPARAISON DIRECTE — même config, maker vs taker (top 40 maker par RF)\n{'='*170}\n")
    lines.append(
        f"  {'Config':>55}  "
        f"{'maker_EV':>9} {'taker_EV':>9}  "
        f"{'maker_RF':>9} {'taker_RF':>9}  "
        f"{'maker_T':>8} {'taker_T':>8}  "
        f"{'maker_WR':>9} {'taker_WR':>9}  "
        f"{'EV_delta':>10}\n"
    )
    lines.append("  " + "─"*160 + "\n")

    # Index taker par (pmax, slope, intcp, slo, shi)
    taker_idx = {
        (r['price_max'], r['slope'], r['intercept'], r['secs_lo'], r['secs_hi']): r
        for r in taker_res
    }

    maker_by_rf = sorted([r for r in maker_res if r['total_pnl'] > 0], key=lambda x: -x['rf'])
    for r_m in maker_by_rf[:40]:
        key = (r_m['price_max'], r_m['slope'], r_m['intercept'], r_m['secs_lo'], r_m['secs_hi'])
        r_t = taker_idx.get(key)
        if r_t is None:
            continue
        p60 = r_m['slope']*60 + r_m['intercept']
        p0  = r_m['intercept']
        cfg_s = (f"p≤{r_m['price_max']:.2f} {r_m['slope']:.2f}xT+{r_m['intercept']:.0f} "
                 f"[60s:{p60:.0f}$ 0s:{p0:.0f}$] t=[{r_m['secs_lo']:.0f}s-{r_m['secs_hi']:.0f}s]")
        winner = "MAKER" if r_m['ev_trade'] > r_t['ev_trade'] else "TAKER"
        delta  = r_m['ev_trade'] - r_t['ev_trade']
        lines.append(
            f"  {cfg_s:>55}  "
            f"{r_m['ev_trade']:>+9.2f}$ {r_t['ev_trade']:>+9.2f}$  "
            f"{r_m['rf']:>9.1f}x {r_t['rf']:>9.1f}x  "
            f"{r_m['trades']:>8d} {r_t['trades']:>8d}  "
            f"{r_m['wr']:>8.1f}% {r_t['wr']:>8.1f}%  "
            f"Δ={delta:>+9.2f}$ [{winner}]\n"
        )

    # ── Résumé global ─────────────────────────────────────────────────────────
    n_maker_pos = sum(1 for r in maker_res if r['ev_trade'] > 0)
    n_taker_pos = sum(1 for r in taker_res if r['ev_trade'] > 0)
    maker_ev_mean = np.mean([r['ev_trade'] for r in maker_res])
    taker_ev_mean = np.mean([r['ev_trade'] for r in taker_res])
    maker_rf_top  = max((r['rf'] for r in maker_res if r['total_pnl'] > 0), default=0)
    taker_rf_top  = max((r['rf'] for r in taker_res if r['total_pnl'] > 0), default=0)

    lines.append(f"\n{'='*170}\n>>> RÉSUMÉ GLOBAL Maker vs Taker\n{'='*170}\n")
    lines.append(
        f"  Configs EV>0   : maker={n_maker_pos}/{len(maker_res)}  taker={n_taker_pos}/{len(taker_res)}\n"
        f"  EV moyen/trade : maker={maker_ev_mean:+.2f}$   taker={taker_ev_mean:+.2f}$\n"
        f"  RF max         : maker={maker_rf_top:.1f}x        taker={taker_rf_top:.1f}x\n"
    )

    output = "".join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"\n{'='*90}")
    print(f"  Terminé en {elapsed:.0f}s  |  {len(all_results)} configs évaluées")
    print(f"  Sortie : {out_path}")
    print(f"{'='*90}\n")

    # Console
    maker_by_rf = sorted([r for r in maker_res if r['total_pnl'] > 0], key=lambda x: -x['rf'])
    taker_by_rf = sorted([r for r in taker_res if r['total_pnl'] > 0], key=lambda x: -x['rf'])

    print(f"\n>>> TOP 20 MAKER (RF)\n{hdr}{sep}")
    for i, r in enumerate(maker_by_rf[:20], 1):
        print(fmt_row(i, r, days), end='')

    print(f"\n>>> TOP 20 TAKER (RF)\n{hdr}{sep}")
    for i, r in enumerate(taker_by_rf[:20], 1):
        print(fmt_row(i, r, days), end='')

    # Résumé global
    print(f"\n{'='*90}")
    print(f"  Configs EV>0   : maker={n_maker_pos}/{len(maker_res)}  taker={n_taker_pos}/{len(taker_res)}")
    print(f"  EV moyen/trade : maker={maker_ev_mean:+.2f}$   taker={taker_ev_mean:+.2f}$")
    print(f"  RF max         : maker={maker_rf_top:.1f}x       taker={taker_rf_top:.1f}x")
    print(f"{'='*90}\n")

    # Comparaison directe top 20 maker
    print(f"\n>>> COMPARAISON DIRECTE top 20 maker par RF\n")
    print(f"  {'Config':>52}  {'mkr_EV':>8} {'tkr_EV':>8}  {'mkr_RF':>7} {'tkr_RF':>7}  {'delta_EV':>9}  winner")
    print(f"  {'─'*120}")
    for r_m in maker_by_rf[:20]:
        key = (r_m['price_max'], r_m['slope'], r_m['intercept'], r_m['secs_lo'], r_m['secs_hi'])
        r_t = taker_idx.get(key)
        if not r_t: continue
        p60 = r_m['slope']*60 + r_m['intercept']
        p0  = r_m['intercept']
        cfg_s = f"p≤{r_m['price_max']:.2f} {r_m['slope']:.2f}xT+{r_m['intercept']:.0f} [60s:{p60:.0f}$ 0s:{p0:.0f}$] t=[{r_m['secs_lo']:.0f}s-{r_m['secs_hi']:.0f}s]"
        delta = r_m['ev_trade'] - r_t['ev_trade']
        w = "MAKER" if delta > 0 else "TAKER"
        print(f"  {cfg_s:>52}  {r_m['ev_trade']:>+8.2f}$ {r_t['ev_trade']:>+8.2f}$  {r_m['rf']:>7.1f}x {r_t['rf']:>7.1f}x  {delta:>+9.2f}$  {w}")


if __name__ == '__main__':
    main()
