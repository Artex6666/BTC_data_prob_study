"""
ETH Curve Optimizer v1
Meme logique que god_curve_optimizer_v6 (BTC) mais adapte ETH :
  - Slopes ~20x plus petites (ETH spot ~$2000 vs BTC ~$70K)
  - Vol thresholds ~20x plus petites
  - CSV : Datas/csv/ETH.csv + reportLive/safeChase/60_lb0.5h/ETH.csv
  - Attention : seulement ~12 jours de donnees -> resultats a valider
"""

import argparse
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import deque
from pathlib import Path

import multiprocessing as mp
import numpy as np
import pandas as pd

from chart_utils import (
    simulate as simulate_contract,
    CAP_GRID,
    attach_settlement_outcomes,
)

_CONTRACTS = None
_TOTAL_HOURS = None

MAX_ORDERS  = 3
BASE_SIZE   = 50.0
DEFAULT_CSV_PATHS = [
    "Datas/csv/ETH.csv",
    "reportLive/safeChase/ETH.csv",
]
DEFAULT_N_WORKERS = max(1, mp.cpu_count() - 2)
CB_WINDOW_S = 5400
VOL_LBS     = [1, 2, 4, 8]
CNET_N_LIST = [2, 5, 10]


def build_configs():
    configs = []

    # Groupe A : lineaire, sans filtre vol
    for slope in [0.02, 0.05, 0.10, 0.20]:
        for intercept in [0.0, 0.5, 1.0, 2.0]:
            for cap in CAP_GRID:
                configs.append(dict(
                    label='linear', curve='linear',
                    slope=slope, intercept=intercept,
                    A_exp=None, tau=None,
                    eq_cap=cap, max_losses_cb=None,
                    vol_lb_h=None, vol_thresh=None,
                ))

    # Groupe D : filtre range + lineaire
    for slope in [0.02, 0.05, 0.10, 0.20]:
        for intercept in [0.0, 1.0]:
            for cap in CAP_GRID:
                for lb in [1, 2, 4, 8]:
                    for thresh in [15.0, 25.0, 40.0]:
                        configs.append(dict(
                            label='vol_linear', curve='linear',
                            slope=slope, intercept=intercept,
                            A_exp=None, tau=None,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh,
                            vol_type='range',
                        ))

    # Groupe D2 : filtre net move + lineaire
    for slope in [0.02, 0.05, 0.10, 0.20]:
        for intercept in [0.0, 1.0]:
            for cap in CAP_GRID:
                for lb in [1, 2, 4, 8]:
                    for thresh in [8.0, 18.0, 36.0]:
                        configs.append(dict(
                            label='vol_net_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            A_exp=None, tau=None,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh,
                            vol_type='net',
                        ))

    # Groupe D3 : trend ratio + lineaire
    for slope in [0.02, 0.05, 0.10, 0.20]:
        for intercept in [0.0, 1.0]:
            for cap in CAP_GRID:
                for lb in [1, 2, 4, 8]:
                    for thresh in [0.3, 0.5, 0.7]:
                        configs.append(dict(
                            label='vol_trend_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            A_exp=None, tau=None,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=float(lb), vol_thresh=thresh,
                            vol_type='trend',
                        ))

    # Groupe VRS : g = clip(base / vol_Nh, g_min, g_max) -> slope *= g
    # base hardcodee sur vol mediane ETH par lookback
    VRS_BASE_ETH = {1.0: 8.0, 2.0: 15.0, 4.0: 25.0}
    for lb, base in VRS_BASE_ETH.items():
        for slope in [0.05, 0.10]:
            for cap in CAP_GRID:
                for g_min in [0.5, 0.75]:
                    for g_max in [1.5, 2.5, 4.0]:
                        configs.append(dict(
                            label='vrs_lin', curve='linear',
                            slope=slope, intercept=0.0,
                            A_exp=None, tau=None,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=None, vol_thresh=None,
                            vrs_enabled=True,
                            vrs_lb=float(lb),
                            vrs_base=float(base),
                            vrs_g_min=float(g_min),
                            vrs_g_max=float(g_max),
                        ))

    # Groupe cnet : filtre candle-net (N bougies M5)
    CNET_THRESH_ETH = [0.3, 0.5, 1.0, 2.0]
    for slope in [0.05, 0.10, 0.20]:
        for intercept in [0.0, 1.0]:
            for cap in CAP_GRID:
                for n in CNET_N_LIST:
                    for thresh in CNET_THRESH_ETH:
                        configs.append(dict(
                            label='cnet_lin', curve='linear',
                            slope=slope, intercept=intercept,
                            A_exp=None, tau=None,
                            eq_cap=cap, max_losses_cb=None,
                            vol_lb_h=None, vol_thresh=None,
                            cnet_n=n, cnet_thresh=float(thresh),
                        ))

    return configs


def _resolve_csv_paths(cli_paths):
    base = Path(__file__).resolve().parent.parent
    out = []
    for p in cli_paths:
        path = Path(p)
        out.append(str(path.resolve() if path.is_absolute() else (base / path)))
    return out


def load_contracts_dataset(csv_paths):
    global _CONTRACTS, _TOTAL_HOURS
    dfs = []
    for p in csv_paths:
        d = pd.read_csv(p, usecols=[
            'timestamp', 'spot_price',
            'm5_up_ask', 'm5_up_bid',
            'm5_down_ask', 'm5_down_bid'
        ])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)

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

    # Precompute vol (range / net / trend) sans look-ahead
    for lb_h in VOL_LBS:
        lb_s = lb_h * 3600
        window = deque()
        for c in contracts:
            t = c['open_ts']
            while window and t - window[0][0] > lb_s: window.popleft()
            if window:
                spots_w = [x[1] for x in window]
                rng   = max(spots_w) - min(spots_w)
                net   = abs(c['op'] - window[0][1])
                trend = net / rng if rng > 0 else 0.0
            else:
                rng = net = trend = 0.0
            c[f'vol_{lb_h:g}h_range'] = rng
            c[f'vol_{lb_h:g}h_net']   = net
            c[f'vol_{lb_h:g}h_trend'] = trend
            window.append((t, c['op']))

    # Precompute candle-net : minimum body sur chaque bougie des N dernieres
    bodies = [0.0] * len(contracts)
    for i in range(1, len(contracts)):
        bodies[i] = abs(contracts[i]['op'] - contracts[i - 1]['op'])
    for i, c in enumerate(contracts):
        for n in CNET_N_LIST:
            if i < n:
                c[f'cnet_{n}c'] = 0.0
            else:
                c[f'cnet_{n}c'] = min(bodies[i - n + 1: i + 1])

    total_hours = (
        (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 3600
        if len(contracts) > 1 else 1
    )
    _CONTRACTS = contracts
    _TOTAL_HOURS = total_hours

    n_settled = attach_settlement_outcomes(contracts, 'eth')
    if n_settled:
        print(f"  Settlement Gamma attaché : {n_settled}/{len(contracts)} contrats")

    return len(contracts)


def _init_worker_from_pickle(pickle_path: str):
    global _CONTRACTS, _TOTAL_HOURS
    with open(pickle_path, "rb") as f:
        payload = pickle.load(f)
    _CONTRACTS = payload["contracts"]
    _TOTAL_HOURS = payload["total_hours"]


def worker_fn(config_batch):
    import numpy as np
    from collections import deque

    global _CONTRACTS, _TOTAL_HOURS
    if _CONTRACTS is None:
        raise RuntimeError("Dataset non charge. Appelez load_contracts_dataset() avant les workers.")

    contracts = _CONTRACTS

    results = []
    for cfg in config_batch:
        wins = 0; losses = 0
        total_pnl = 0.0; win_pnl = 0.0; loss_pnl = 0.0
        trades = 0; day_pnls = {}; contract_pnls = []
        max_losses_cb = cfg.get('max_losses_cb')
        vol_thresh    = cfg.get('vol_thresh')
        vol_key       = (f"vol_{cfg['vol_lb_h']:g}h_{cfg.get('vol_type','range')}"
                         if vol_thresh is not None and cfg.get('vol_lb_h') is not None else None)
        cnet_n        = cfg.get('cnet_n')
        cnet_thresh   = cfg.get('cnet_thresh') or 0.0
        loss_times    = deque()

        for c in contracts:
            if vol_key is not None and c[vol_key] < vol_thresh:
                continue
            if cnet_n is not None and c.get(f'cnet_{cnet_n}c', 0.0) < cnet_thresh:
                continue
            if max_losses_cb is not None:
                ce_ts = c['ce_ts']
                while loss_times and ce_ts - loss_times[0] > CB_WINDOW_S:
                    loss_times.popleft()
                if len(loss_times) >= max_losses_cb:
                    continue

            won, pnl = simulate_contract(c, cfg)
            if won is None: continue

            trades += 1; total_pnl += pnl
            contract_pnls.append(pnl)
            if won:
                wins += 1; win_pnl += pnl
            else:
                losses += 1; loss_pnl += pnl
                if max_losses_cb is not None:
                    loss_times.append(c['ce_ts'])

            day_pnls[c['ce'].date()] = day_pnls.get(c['ce'].date(), 0.0) + pnl

        if trades == 0: continue

        wr    = wins / trades * 100
        avg_w = win_pnl  / wins   if wins   else 0.0
        avg_l = loss_pnl / losses if losses else 0.0

        losing_days = sum(1 for v in day_pnls.values() if v < 0)
        worst_day   = min(day_pnls.values()) if day_pnls else 0.0

        cum_arr = np.cumsum([0.0] + contract_pnls)
        peak = 0.0; max_dd = 0.0
        for v in cum_arr:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd

        if losing_days == 0 or total_pnl <= 0:
            rf = 0.0
        else:
            rf = total_pnl / max(max_dd, BASE_SIZE)

        results.append({**cfg,
            'trades':      trades,
            'wr':          wr,
            'total_pnl':   total_pnl,
            'avg_w':       avg_w,
            'avg_l':       avg_l,
            'losing_days': losing_days,
            'worst_day':   worst_day,
            'max_dd':      max_dd,
            'rf':          rf,
            'total_hours': float(_TOTAL_HOURS),
        })

    return results


def fmt_row(i, r, days):
    if r['curve'] == 'expo':
        curve_s = f"A={r['A_exp']:.2f} t={r['tau']:.0f}"
    else:
        curve_s = f"sl={r['slope']:.3f} int={r['intercept']:.1f}"
    cb_s  = str(r['max_losses_cb']) if r['max_losses_cb'] else "-"
    if r.get('vol_thresh') is not None:
        vt = r.get('vol_type', 'range')
        thresh_s = f"{r['vol_thresh']:.2f}" if vt == 'trend' else f"${r['vol_thresh']:.0f}"
        lb_str = f"{r['vol_lb_h']:g}"
        vol_s = f"{vt[:3]}{lb_str}h>{thresh_s}"
    elif r.get('vrs_enabled'):
        vol_s = (
            f"vrs{r['vrs_lb']:g}h b={r['vrs_base']:.0f} "
            f"g={r['vrs_g_min']:.2f}-{r['vrs_g_max']:.2f}"
        )
    elif r.get('cnet_n') is not None:
        vol_s = f"cnet{r['cnet_n']}c>${r['cnet_thresh']:.0f}           "
    else:
        vol_s = "              -"
    rf_s  = f"{r['rf']:>6.1f}x" if r['max_dd'] > 0 else f"{r['rf']:>5.1f}x*"
    return (
        f"{i:>3d} {r['label']:>12s} {curve_s:>18s} cap={r['eq_cap']:.2f} "
        f"cb={cb_s} {vol_s} | "
        f"T={r['trades']:>4d} ({r['trades']/days:>4.1f}/j) WR={r['wr']:>5.1f}% ${r['total_pnl']/days:>6.1f}/j | "
        f"loseDay={r['losing_days']:>2d}/{int(days):>2d} "
        f"wrstDay=${r['worst_day']:>7.2f} maxDD=${r['max_dd']:>6.0f} RF={rf_s}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="ETH Curve Optimizer v1")
    parser.add_argument(
        "--csv",
        nargs="+",
        metavar="PATH",
        help="Fichier(s) CSV (colonnes m5_*). Par defaut: Datas/csv/ETH.csv + reportLive/safeChase/60_lb0.5h/ETH.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help=f"Nombre de workers (defaut: {DEFAULT_N_WORKERS}).",
    )
    parser.add_argument(
        "--threads",
        action="store_true",
        help="Threads (1 copie RAM, GIL). Defaut: processus multi-coeur + snapshot disque.",
    )
    args = parser.parse_args()
    csv_paths = _resolve_csv_paths(args.csv) if args.csv else DEFAULT_CSV_PATHS
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    global _CONTRACTS, _TOTAL_HOURS

    n_workers = max(1, int(args.workers))
    run_started = datetime.now()
    run_ts = run_started.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parent / f"god_curve_eth_v1_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"eth_curve_results_v1_{run_ts}.txt"

    t0 = time.time()
    configs = build_configs()
    print(f"[{run_started.strftime('%H:%M:%S')}] ETH Curve Optimizer v1")
    print(f"  Sortie: {run_dir.resolve()}")
    backend = "threads" if args.threads else "processus"
    print(
        f"  Workers={n_workers}  Backend={backend}  Configs={len(configs)}  Tri=RF  $50/ordre",
        flush=True,
    )
    print(f"  CSV: {csv_paths}\n")

    t_load = time.time()
    n_ct = load_contracts_dataset(csv_paths)
    print(f"  Contrats charges: {n_ct}  ({time.time() - t_load:.1f}s)", flush=True)

    batch_size = max(1, len(configs) // n_workers + 1)
    batches = [configs[i : i + batch_size] for i in range(0, len(configs), batch_size)]

    all_results = []
    if args.threads:
        print("  (threads: memoire partagee, CPU souvent faible)\n", flush=True)
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(worker_fn, b) for b in batches]
            for i, fut in enumerate(as_completed(futures), 1):
                all_results.extend(fut.result())
                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i}/{len(batches)} — {len(all_results)} configs",
                    flush=True,
                )
    else:
        payload_path = run_dir / "_worker_dataset.pkl"
        t_snap = time.time()
        with open(payload_path, "wb") as f:
            pickle.dump(
                {"contracts": _CONTRACTS, "total_hours": _TOTAL_HOURS},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        sz_mb = payload_path.stat().st_size / (1024 * 1024)
        print(
            f"  Snapshot workers: {sz_mb:.1f} MiB  ({time.time() - t_snap:.1f}s)",
            flush=True,
        )
        _CONTRACTS = None
        _TOTAL_HOURS = None
        print("  (processus: multi-coeur)\n", flush=True)
        try:
            with mp.Pool(
                processes=n_workers,
                initializer=_init_worker_from_pickle,
                initargs=(str(payload_path.resolve()),),
            ) as pool:
                for i, br in enumerate(pool.imap_unordered(worker_fn, batches)):
                    all_results.extend(br)
                    print(
                        f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i+1}/{len(batches)} — {len(all_results)} configs",
                        flush=True,
                    )
        finally:
            payload_path.unlink(missing_ok=True)

    if not all_results: print("Aucun resultat."); return

    days = all_results[0]['total_hours'] / 24
    all_results.sort(key=lambda x: -x['rf'])
    elapsed = time.time() - t0

    sep = "─" * 135 + "\n"
    lines = [
        f"ETH Curve Optimizer v1  —  Tri RF  |  "
        f"Run: {run_started.strftime('%Y-%m-%d %H:%M:%S')}  run_id={run_ts}  |  "
        f"Data: {days:.1f}j  |  Configs: {len(all_results)}/{len(configs)}  |  {elapsed:.0f}s\n"
        f"Sortie: {run_dir.resolve()}\n"
        f"Charts: {(run_dir / 'charts').resolve()}\n"
        f"CSV: {csv_paths}\n",
        f"ATTENTION : echantillon court ({days:.0f} j) — valider sur plus de donnees si besoin\n",
        sep,
    ]

    for group in ['linear', 'expo', 'cb_linear',
                  'vol_linear', 'vol_expo', 'vol_cb',
                  'vol_net_lin', 'vol_net_expo', 'vol_trend_lin',
                  'vol_slope_lin', 'vol_slope_expo']:
        grp = [r for r in all_results if r['label'] == group]
        if not grp: continue
        lines.append(f"\n>>> {group.upper()} — top 15 par RF\n{sep}")
        for i, r in enumerate(grp[:15]):
            lines.append(fmt_row(i+1, r, days))

    lines.append(f"\n>>> TOP 30 GLOBAL par RF\n{sep}")
    for i, r in enumerate(all_results[:30]):
        lines.append(fmt_row(i+1, r, days))

    by_consist = sorted(
        [r for r in all_results if r['trades'] >= 100],
        key=lambda x: (x['losing_days'], -x['rf'])
    )
    lines.append(f"\n>>> TOP 20 CONSISTANCE (min losing_days, min 100 trades)\n{sep}")
    for i, r in enumerate(by_consist[:20]):
        lines.append(fmt_row(i+1, r, days))

    output = "".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"\nSauvegarde -> {output_path}  ({elapsed:.0f}s)")
    print("\n" + output)

    from chart_utils import generate_charts
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Charts -> {charts_dir.resolve()}", flush=True)
    try:
        generate_charts(
            all_results,
            csv_paths,
            str(charts_dir),
            n_top=12,
            title_prefix="ETH God Curve v1",
            vol_slope_ref_key="vol_1h_range",
            asset='eth',
        )
    except Exception:
        import traceback
        print("\n[CHARTS] Echec generation (voir traceback ci-dessous).", flush=True)
        traceback.print_exc()


if __name__ == "__main__":
    main()
