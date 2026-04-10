"""
xrp_curve_optimizer_v1.py — Optimizer M5 XRP

Dérivé de sol_curve_optimizer_v1.py.
Slopes/intercepts/vol seuils calibrés pour XRP (~$1.30, vol1h médiane ~$0.008).
  thresh = slope * remain_s + intercept  (en $)
  slope=0.0001, remain=60s → thresh=$0.006 (~0.46% du spot)
"""
import argparse, pickle, sys, time
from collections import deque
from datetime import datetime
from pathlib import Path

import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chart_utils import (
    simulate as simulate_contract,
    attach_settlement_outcomes,
    precompute_vol as cu_precompute_vol,
    generate_charts,
    VOL_LBS,
)

_CONTRACTS   = None
_TOTAL_HOURS = None

ASSET       = 'xrp'
MAX_ORDERS  = 2
BASE_SIZE   = 50.0
CB_WINDOW_S = 5400
CNET_N_LIST = [2, 5, 10]

OUT_ROOT = ROOT / "god_curve" / "xrp_curve_results_v1"

DEFAULT_CSV_PATHS = [
    str(ROOT / "reportLive" / "safeChase" / "XRP.csv"),
]

# Grilles calibrées XRP (~$1.30)
# thresh à remain=60s : slope*60. Ex: 0.0001*60=$0.006 (~0.46%)
SLOPE_GRID     = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003]
INTERCEPT_GRID = [0.0, 0.0005, 0.001, 0.003]
CAP_GRID       = [0.55, 0.65, 0.75, 0.85, 0.90]

# Vol range/net : médiane vol1h~$0.008-0.015, p90~$0.025
VOL_RANGE_GRID = [0.003, 0.006, 0.012, 0.020, 0.035]
VOL_NET_GRID   = [0.002, 0.004, 0.008, 0.015]
VOL_TREND_GRID = [0.30, 0.50, 0.70]
VOL_PCT_R_GRID = [0.30, 0.50, 0.80, 1.20, 2.00]  # % du prix
VOL_PCT_N_GRID = [0.20, 0.40, 0.60, 1.00]

VOL_LB_GRID    = [1, 2, 4]

# VRS base calibrée XRP
VRS_BASE_XRP   = {1.0: 0.006, 2.0: 0.012, 4.0: 0.020}
CNET_THRESH    = [0.0005, 0.001, 0.002, 0.005]


def build_configs():
    configs = []

    # A : linéaire, sans filtre
    for slope in SLOPE_GRID:
        for intercept in INTERCEPT_GRID:
            for cap in CAP_GRID:
                configs.append(dict(
                    label='linear', curve='linear',
                    slope=slope, intercept=intercept,
                    eq_cap=cap, max_losses_cb=None,
                    vol_lb_h=None, vol_thresh=None,
                ))

    # B : range $ + linéaire
    for slope in SLOPE_GRID:
        for intercept in [0.0, 0.001]:
            for cap in CAP_GRID:
                for lb in VOL_LB_GRID:
                    for thresh in VOL_RANGE_GRID:
                        configs.append(dict(
                            label='vol_rng_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh, vol_type='range',
                        ))

    # C : net $ + linéaire
    for slope in SLOPE_GRID:
        for intercept in [0.0, 0.001]:
            for cap in CAP_GRID:
                for lb in VOL_LB_GRID:
                    for thresh in VOL_NET_GRID:
                        configs.append(dict(
                            label='vol_net_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh, vol_type='net',
                        ))

    # D : trend (relatif, cross-asset)
    for slope in SLOPE_GRID:
        for intercept in [0.0]:
            for cap in CAP_GRID:
                for lb in VOL_LB_GRID:
                    for thresh in VOL_TREND_GRID:
                        configs.append(dict(
                            label='vol_trend_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh, vol_type='trend',
                        ))

    # E : pct_range % (relatif)
    for slope in SLOPE_GRID:
        for intercept in [0.0]:
            for cap in CAP_GRID:
                for lb in VOL_LB_GRID:
                    for thresh in VOL_PCT_R_GRID:
                        configs.append(dict(
                            label='vol_pctr_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh, vol_type='pct_range',
                        ))

    # F : pct_net % (relatif)
    for slope in SLOPE_GRID:
        for intercept in [0.0]:
            for cap in CAP_GRID:
                for lb in VOL_LB_GRID:
                    for thresh in VOL_PCT_N_GRID:
                        configs.append(dict(
                            label='vol_pctn_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh, vol_type='pct_net',
                        ))

    # G : VRS
    for lb, base in VRS_BASE_XRP.items():
        for slope in [0.00003, 0.0001, 0.0003]:
            for cap in CAP_GRID:
                for g_min in [0.5, 0.75]:
                    for g_max in [1.5, 2.5]:
                        configs.append(dict(
                            label='vrs_lin', curve='linear',
                            slope=slope, intercept=0.0,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=None, vol_thresh=None,
                            vrs_enabled=True,
                            vrs_lb=float(lb), vrs_base=float(base),
                            vrs_g_min=float(g_min), vrs_g_max=float(g_max),
                        ))

    # H : cnet
    for slope in [0.00003, 0.0001, 0.0003]:
        for intercept in [0.0, 0.0005]:
            for cap in CAP_GRID:
                for n in CNET_N_LIST:
                    for thresh in CNET_THRESH:
                        configs.append(dict(
                            label='cnet_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=None, vol_thresh=None,
                            cnet_n=n, cnet_thresh=float(thresh),
                        ))

    return configs


def load_contracts_dataset(csv_paths):
    global _CONTRACTS, _TOTAL_HOURS
    dfs = []
    for p in csv_paths:
        if not Path(p).exists():
            print(f"  [WARN] {p} introuvable, skip")
            continue
        d = pd.read_csv(p, usecols=[
            'timestamp', 'spot_price',
            'm5_up_ask', 'm5_up_bid', 'm5_down_ask', 'm5_down_bid',
        ])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)

    if not dfs:
        raise RuntimeError("Aucun CSV XRP chargé")

    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    contracts = []
    for ce, grp in df.groupby('contract'):
        op_spot = grp['spot_price'].iloc[0]
        t60 = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=60)]
        if len(t60) < 2: continue
        contracts.append({
            'ce':       ce,
            'ce_ts':    ce.timestamp(),
            'open_ts':  ce.timestamp() - 300,
            'op':       float(op_spot),
            'spot':     t60['spot_price'].values.astype(np.float64),
            'ts':       t60['timestamp'].values,
            'up_bid':   t60['m5_up_bid'].values.astype(np.float64),
            'down_bid': t60['m5_down_bid'].values.astype(np.float64),
            'up_ask':   t60['m5_up_ask'].values.astype(np.float64),
            'down_ask': t60['m5_down_ask'].values.astype(np.float64),
        })
    del df

    # Vol avec support pct (chart_utils.precompute_vol)
    cu_precompute_vol(contracts, VOL_LBS)

    # cnet
    bodies = [0.0] * len(contracts)
    for i in range(1, len(contracts)):
        bodies[i] = abs(contracts[i]['op'] - contracts[i - 1]['op'])
    for i, c in enumerate(contracts):
        for n in CNET_N_LIST:
            c[f'cnet_{n}c'] = 0.0 if i < n else min(bodies[i - n + 1: i + 1])

    total_hours = (
        (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 3600
        if len(contracts) > 1 else 1
    )
    _CONTRACTS   = contracts
    _TOTAL_HOURS = total_hours

    n_settled = attach_settlement_outcomes(contracts, ASSET, tf='5min')
    print(f"  Settlement : {n_settled}/{len(contracts)} contrats")
    return len(contracts)


def _init_worker_from_pickle(pickle_path):
    global _CONTRACTS, _TOTAL_HOURS
    with open(pickle_path, 'rb') as f:
        payload = pickle.load(f)
    _CONTRACTS   = payload['contracts']
    _TOTAL_HOURS = payload['total_hours']


def worker_fn(config_batch):
    from collections import deque
    results = []
    for cfg in config_batch:
        vol_thresh = cfg.get('vol_thresh')
        vol_key    = (f"vol_{cfg['vol_lb_h']:g}h_{cfg.get('vol_type','range')}"
                      if vol_thresh is not None and cfg.get('vol_lb_h') is not None else None)
        cnet_n     = cfg.get('cnet_n')
        cnet_thr   = cfg.get('cnet_thresh') or 0.0
        max_cb     = cfg.get('max_losses_cb')
        loss_times = deque()

        wins = losses = trades = 0
        total_pnl = 0.0
        day_pnls: dict = {}
        contract_pnls = []

        for c in _CONTRACTS:
            if vol_key and c.get(vol_key, 0.0) < vol_thresh: continue
            if cnet_n and c.get(f'cnet_{cnet_n}c', 0.0) < cnet_thr: continue
            if max_cb is not None:
                ce_ts = c['ce_ts']
                while loss_times and ce_ts - loss_times[0] > CB_WINDOW_S: loss_times.popleft()
                if len(loss_times) >= max_cb: continue

            won, pnl = simulate_contract(c, cfg)
            if won is None: continue

            trades += 1; total_pnl += pnl; contract_pnls.append(pnl)
            if won: wins += 1
            else:
                losses += 1
                if max_cb is not None: loss_times.append(c['ce_ts'])
            day_pnls[c['ce'].date()] = day_pnls.get(c['ce'].date(), 0.0) + pnl

        if trades == 0: continue

        wr          = wins / trades * 100
        losing_days = sum(1 for v in day_pnls.values() if v < 0)
        worst_day   = min(day_pnls.values())
        peak = max_dd = 0.0
        for v in [sum(contract_pnls[:i+1]) for i in range(len(contract_pnls))]:
            if v > peak: peak = v
            if peak - v > max_dd: max_dd = peak - v
        rf = total_pnl / max(max_dd, BASE_SIZE) if (losing_days > 0 and total_pnl > 0) else 0.0

        results.append({**cfg,
            'trades': trades, 'wr': wr, 'total_pnl': total_pnl,
            'losing_days': losing_days, 'worst_day': worst_day,
            'max_dd': max_dd, 'rf': rf, 'total_hours': float(_TOTAL_HOURS),
        })
    return results


def fmt_row(i, r, days):
    curve_s = f"sl={r['slope']:.6f} int={r['intercept']:.5f}"
    cb_s    = str(r['max_losses_cb']) if r['max_losses_cb'] else "-"
    if r.get('vol_thresh') is not None:
        vt = r.get('vol_type', 'range')
        if vt in ('pct_range', 'pct_net'):
            thr_s = f"{r['vol_thresh']:.2f}%"
        elif vt == 'trend':
            thr_s = f"{r['vol_thresh']:.2f}"
        else:
            thr_s = f"${r['vol_thresh']:.4f}"
        vol_s = f"{vt[:4]}{r['vol_lb_h']:g}h>{thr_s}"
    elif r.get('vrs_enabled'):
        vol_s = f"vrs{r['vrs_lb']:g}h b={r['vrs_base']:.4f} g={r['vrs_g_min']:.2f}-{r['vrs_g_max']:.2f}"
    elif r.get('cnet_n'):
        vol_s = f"cnet{r['cnet_n']}c>${r['cnet_thresh']:.4f}"
    else:
        vol_s = "              -"
    rf_s = f"{r['rf']:>6.1f}x" if r['max_dd'] > 0 else f"{r['rf']:>5.1f}x*"
    return (
        f"{i:>3d} {r['label']:>12s} {curve_s} cap={r['eq_cap']:.2f} cb={cb_s} {vol_s} | "
        f"T={r['trades']:>4d} ({r['trades']/days:>4.1f}/j) WR={r['wr']:>5.1f}% "
        f"${r['total_pnl']/days:>6.1f}/j | "
        f"loseDay={r['losing_days']:>2d}/{int(days):>2d} "
        f"wrstDay=${r['worst_day']:>7.4f} maxDD=${r['max_dd']:>6.0f} RF={rf_s}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="XRP Curve Optimizer v1")
    parser.add_argument('--csv',     nargs='+', default=DEFAULT_CSV_PATHS)
    parser.add_argument('--workers', type=int,  default=max(1, mp.cpu_count() - 5))
    parser.add_argument('--top',     type=int,  default=20)
    parser.add_argument('--batch',   type=int,  default=64)
    parser.add_argument('--no-charts', action='store_true')
    args = parser.parse_args()

    ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUT_ROOT / f"xrp_v1_{ts_str}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== XRP Curve Optimizer v1 ===\n")
    n = load_contracts_dataset(args.csv)
    days = _TOTAL_HOURS / 24
    print(f"  {n} contrats M5  ({days:.1f}j)\n")

    configs = build_configs()
    print(f"Grid : {len(configs)} configs\n")
    batches = [configs[i:i+args.batch] for i in range(0, len(configs), args.batch)]

    all_results = []
    t0 = time.monotonic()

    if args.workers > 1:
        pkl = str(out_dir / '_contracts.pkl')
        with open(pkl, 'wb') as f:
            pickle.dump({'contracts': _CONTRACTS, 'total_hours': _TOTAL_HOURS}, f)
        with mp.Pool(args.workers, _init_worker_from_pickle, (pkl,)) as pool:
            for i, res in enumerate(pool.imap_unordered(worker_fn, batches)):
                all_results.extend(res)
                pct = 100*(i+1)/len(batches)
                eta = (time.monotonic()-t0)/(i+1)*(len(batches)-i-1)
                print(f"\r  {pct:5.1f}%  {len(all_results):>5d} configs  ETA {eta/60:.1f}min",
                      end='', flush=True)
    else:
        for i, batch in enumerate(batches):
            all_results.extend(worker_fn(batch))
            print(f"\r  {100*(i+1)/len(batches):5.1f}%  {len(all_results)} configs",
                  end='', flush=True)

    elapsed = time.monotonic() - t0
    print(f"\n\nTerminé en {elapsed:.0f}s — {len(all_results)} configs avec trades\n")

    sep = "-" * 130 + "\n"
    lines = [
        f"=== XRP Curve Optimizer v1 ===\n",
        f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Data: {days:.1f}j  |  Contrats: {n}  |  Configs: {len(all_results)}/{len(configs)}  |  {elapsed:.0f}s\n\n",
    ]

    for sort_key, title in [
        (lambda x: -x['rf'],       f"TOP {args.top} RF"),
        (lambda x: -x['total_pnl'], f"TOP {args.top} PnL"),
        (lambda x: (x['losing_days'], -x['rf']), f"TOP {args.top} CONSISTANCE"),
    ]:
        all_results.sort(key=sort_key)
        lines.append(f">>> {title}\n{sep}")
        for i, r in enumerate(all_results[:args.top]):
            lines.append(fmt_row(i+1, r, days))
        lines.append("\n")

    txt = out_dir / f"xrp_curve_results_v1_{ts_str}.txt"
    txt.write_text("".join(lines), encoding='utf-8')
    print(f"Résultats -> {txt}")

    if not args.no_charts and all_results:
        all_results.sort(key=lambda x: -x['rf'])
        generate_charts(
            all_results[:12], args.csv, str(out_dir / "charts"),
            n_top=12, title_prefix="XRP Curve v1",
            vol_slope_ref_key="vol_0.5h_range", asset=ASSET,
        )

    print(f"\nDone -> {out_dir}/")


if __name__ == '__main__':
    mp.freeze_support()
    main()
