#!/usr/bin/env python3
"""
alts_curve_optimizer_v1.py — Optimizer M5 multi-assets (XRP, SOL, BNB)

Différences vs BTC/ETH :
  - Pool les contrats des 3 assets dans un seul backtest
  - Vol seuils relatifs : pct_range / pct_net (% du prix spot) + trend (déjà relatif)
  - Settlement via settlement.csv par asset
  - Score = PnL total + métriques de consistance sur l'ensemble poolé

Usage :
    python alts_curve_optimizer_v1.py [--workers N] [--top K]
"""

import sys, time, argparse, pickle, multiprocessing, itertools
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import (
    load_contracts, precompute_vol, attach_settlement_outcomes,
    simulate as simulate_contract,
    generate_charts,
    CAP_GRID, VOL_LBS,
    make_config_label, make_config_name,
)

ROOT     = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "god_curve" / "alts_curve_results_v1"

ASSETS = ['xrp', 'sol', 'bnb']
ASSET_CSV = {
    'xrp': [str(ROOT / "reportLive" / "safeChase" / "XRP.csv")],
    'sol': [str(ROOT / "Datas" / "csv" / "SOL.csv"),
            str(ROOT / "reportLive" / "safeChase" / "SOL.csv")],
    'bnb': [str(ROOT / "reportLive" / "safeChase" / "BNB.csv")],
}

# ── Grille de recherche ───────────────────────────────────────────────────────
SLOPE_GRID     = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
INTERCEPT_GRID = [0.0]
_CAP_GRID      = [0.55, 0.65, 0.75, 0.85, 0.90]
CB_GRID        = [None]

# Vol filters : None (pas de filtre) + combinaisons (type, lb_h, thresh)
#   pct_range / pct_net : % du prix spot → fonctionne cross-asset
#   trend               : ratio net/range (0-1) → déjà relatif
def _build_vol_grid():
    entries = [None]
    for lb in [0.5, 1.0, 2.0]:
        for thr in [0.20, 0.30, 0.50]:
            entries.append(('trend',     lb, thr))
        for thr in [0.30, 0.50, 0.80, 1.20]:
            entries.append(('pct_range', lb, thr))
        for thr in [0.20, 0.40, 0.60, 1.00]:
            entries.append(('pct_net',   lb, thr))
    return entries

VOL_GRID = _build_vol_grid()

# ── Globals worker ────────────────────────────────────────────────────────────
_CONTRACTS   = []   # contrats poolés (tous assets)
_TOTAL_HOURS = 1.0

CB_WINDOW_S = 3600  # circuit-breaker : fenêtre 1h


def _build_configs():
    configs = []
    for slope, intercept, cap, cb, vol in itertools.product(
            SLOPE_GRID, INTERCEPT_GRID, _CAP_GRID, CB_GRID, VOL_GRID):
        cfg = dict(curve='linear', label='linear',
                   slope=slope, intercept=intercept,
                   eq_cap=cap, max_losses_cb=cb, max_orders=2)
        if vol is None:
            cfg.update(vol_thresh=None, vol_lb_h=None)
        else:
            vt, lb, thr = vol
            cfg.update(vol_type=vt, vol_lb_h=lb, vol_thresh=thr,
                       label='vol_trend_lin' if vt == 'trend'
                             else f'vol_{vt}_lin')
        configs.append(cfg)
    return configs


def _init_worker(contracts, total_hours):
    global _CONTRACTS, _TOTAL_HOURS
    _CONTRACTS   = contracts
    _TOTAL_HOURS = total_hours


def _init_worker_from_pickle(pickle_path):
    global _CONTRACTS, _TOTAL_HOURS
    with open(pickle_path, 'rb') as f:
        payload = pickle.load(f)
    _CONTRACTS   = payload['contracts']
    _TOTAL_HOURS = payload['total_hours']


def worker_fn(config_batch):
    import numpy as np
    from collections import deque
    results = []
    for cfg in config_batch:
        vol_key       = None
        vol_thresh    = cfg.get('vol_thresh')
        vol_type      = cfg.get('vol_type', 'range')
        if vol_thresh is not None and cfg.get('vol_lb_h') is not None:
            vol_key = f"vol_{cfg['vol_lb_h']:g}h_{vol_type}"
        max_losses_cb = cfg.get('max_losses_cb')

        trades = 0; total_pnl = 0.0; wins = 0; losses = 0
        day_pnls: dict = {}
        loss_times: deque = deque()

        for c in _CONTRACTS:
            if vol_key is not None and c.get(vol_key, 0.0) < vol_thresh:
                continue
            if max_losses_cb is not None:
                ce_ts = c['ce_ts']
                while loss_times and ce_ts - loss_times[0] > CB_WINDOW_S:
                    loss_times.popleft()
                if len(loss_times) >= max_losses_cb:
                    continue

            won, pnl = simulate_contract(c, cfg)
            if won is None:
                continue
            trades += 1; total_pnl += pnl
            if won:
                wins += 1
            else:
                losses += 1
                if max_losses_cb is not None:
                    loss_times.append(c['ce_ts'])
            day_pnls[c['ce'].date()] = day_pnls.get(c['ce'].date(), 0.0) + pnl

        if trades == 0:
            continue

        days         = _TOTAL_HOURS / 24
        wr           = wins / trades * 100
        losing_days  = sum(1 for v in day_pnls.values() if v < 0)
        worst_day    = min(day_pnls.values()) if day_pnls else 0.0
        pnl_arr      = np.array(list(day_pnls.values()))
        max_dd       = 0.0
        peak = 0.0
        running = 0.0
        for v in sorted(day_pnls):
            running += day_pnls[v]
            if running > peak:
                peak = running
            dd = peak - running
            if dd > max_dd:
                max_dd = dd
        rf = total_pnl / max_dd if max_dd > 0 else total_pnl

        results.append({**cfg,
            'trades': trades, 'wins': wins, 'losses': losses,
            'wr': wr, 'total_pnl': total_pnl,
            'losing_days': losing_days, 'worst_day': worst_day,
            'max_dd': max_dd, 'rf': rf,
            'total_hours': _TOTAL_HOURS,
        })
    return results


# ── Chargement des contrats poolés ────────────────────────────────────────────
def load_all_contracts():
    all_contracts = []
    for asset in ASSETS:
        csv_paths = [p for p in ASSET_CSV[asset] if Path(p).exists()]
        if not csv_paths:
            print(f"  [{asset.upper()}] CSV introuvable, skip")
            continue

        dfs = []
        for p in csv_paths:
            d = pd.read_csv(p, usecols=[
                'timestamp', 'spot_price',
                'm5_up_ask', 'm5_up_bid',
                'm5_down_ask', 'm5_down_bid',
            ])
            d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
            dfs.append(d)
        df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
        df['contract'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

        contracts = []
        for ce, grp in df.groupby('contract'):
            op_spot = grp['spot_price'].iloc[0]
            t60 = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=60)]
            if len(t60) < 2:
                continue
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
                '_asset':   asset,
            })
        del df

        precompute_vol(contracts, VOL_LBS)

        n_settled = attach_settlement_outcomes(contracts, asset, tf='5min')
        print(f"  [{asset.upper()}] {len(contracts)} contrats M5  settlement={n_settled}", flush=True)
        all_contracts.extend(contracts)

    all_contracts.sort(key=lambda c: c['ce_ts'])
    return all_contracts


def fmt_row(i, r, days):
    if r['curve'] == 'expo':
        curve_s = f"A={r['A_exp']:.2f} t={r['tau']:.0f}"
    else:
        curve_s = f"sl={r['slope']:.3f} int={r['intercept']:.1f}"
    cb_s = str(r['max_losses_cb']) if r['max_losses_cb'] else "-"
    if r.get('vol_thresh') is not None:
        vt       = r.get('vol_type', 'range')
        thresh_s = (f"{r['vol_thresh']:.2f}%" if vt in ('pct_range', 'pct_net')
                    else (f"{r['vol_thresh']:.2f}" if vt == 'trend'
                          else f"${r['vol_thresh']:.0f}"))
        vol_s = f"{vt[:4]}{r['vol_lb_h']:g}h>{thresh_s}"
    else:
        vol_s = "              -"
    rf_s = f"{r['rf']:>6.1f}x" if r['max_dd'] > 0 else f"{r['rf']:>5.1f}x*"
    return (
        f"{i:>3d} {r['label']:>14s} {curve_s:>18s} cap={r['eq_cap']:.2f} "
        f"cb={cb_s} {vol_s} | "
        f"T={r['trades']:>4d} ({r['trades']/days:>4.1f}/j) WR={r['wr']:>5.1f}% "
        f"${r['total_pnl']/days:>6.1f}/j | "
        f"loseDay={r['losing_days']:>2d}/{int(days):>2d} "
        f"wrstDay=${r['worst_day']:>7.2f} maxDD=${r['max_dd']:>6.0f} RF={rf_s}\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Alts God Curve Optimizer v1 (XRP+SOL+BNB M5)")
    parser.add_argument('--workers', type=int, default=max(1, multiprocessing.cpu_count() - 5))
    parser.add_argument('--top',     type=int, default=20)
    parser.add_argument('--batch',   type=int, default=64)
    parser.add_argument('--no-charts', action='store_true')
    args = parser.parse_args()

    ts_str   = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir  = OUT_ROOT / f"alts_v1_{ts_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"

    print("=== Alts Curve Optimizer v1 (XRP + SOL + BNB) ===\n")
    print("Chargement des contrats...", flush=True)
    contracts = load_all_contracts()

    if not contracts:
        print("Aucun contrat chargé. Vérifier les CSV.")
        return

    total_hours = (
        (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 3600
        if len(contracts) > 1 else 1
    )
    days = total_hours / 24
    print(f"\n  Total poolé : {len(contracts)} contrats M5  ({days:.1f}j)\n")

    configs = _build_configs()
    print(f"Grid : {len(configs)} configurations\n")

    batches = [configs[i:i+args.batch] for i in range(0, len(configs), args.batch)]
    all_results = []
    t0 = time.monotonic()

    if args.workers > 1:
        pickle_path = str(out_dir / "_contracts.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump({'contracts': contracts, 'total_hours': total_hours}, f)

        with multiprocessing.Pool(
            processes=args.workers,
            initializer=_init_worker_from_pickle,
            initargs=(pickle_path,),
        ) as pool:
            for i, batch_results in enumerate(pool.imap_unordered(worker_fn, batches)):
                all_results.extend(batch_results)
                pct = 100 * (i + 1) / len(batches)
                elapsed = time.monotonic() - t0
                eta = elapsed / (i + 1) * (len(batches) - i - 1)
                print(f"\r  {pct:5.1f}%  {len(all_results):>5d} configs  "
                      f"ETA {eta/60:.1f}min", end='', flush=True)
    else:
        _init_worker(contracts, total_hours)
        for i, batch in enumerate(batches):
            all_results.extend(worker_fn(batch))
            pct = 100 * (i + 1) / len(batches)
            print(f"\r  {pct:5.1f}%  {len(all_results):>5d} configs", end='', flush=True)

    elapsed = time.monotonic() - t0
    print(f"\n\nTerminé en {elapsed:.0f}s — {len(all_results)} configs avec trades\n")

    all_results.sort(key=lambda x: -x['rf'])
    sep = "-" * 120 + "\n"

    lines = [
        f"=== Alts Curve Optimizer v1 (XRP + SOL + BNB) ===\n",
        f"Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Data: {days:.1f}j  |  Contrats: {len(contracts)}  |  Configs: {len(all_results)}/{len(configs)}  |  {elapsed:.0f}s\n\n",
        f">>> TOP {args.top} RF (total PnL / maxDD)\n{sep}",
    ]
    for i, r in enumerate(all_results[:args.top]):
        lines.append(fmt_row(i + 1, r, days))

    all_results.sort(key=lambda x: -x['total_pnl'])
    lines.append(f"\n>>> TOP {args.top} PnL TOTAL\n{sep}")
    for i, r in enumerate(all_results[:args.top]):
        lines.append(fmt_row(i + 1, r, days))

    all_results.sort(key=lambda x: (x['losing_days'], -x['rf']))
    lines.append(f"\n>>> TOP {args.top} CONSISTANCE (losing_days ↑, RF ↓)\n{sep}")
    for i, r in enumerate(all_results[:args.top]):
        lines.append(fmt_row(i + 1, r, days))

    txt_path = out_dir / f"alts_curve_results_v1_{ts_str}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"Résultats -> {txt_path}")

    # Charts sur le top RF
    if not args.no_charts and all_results:
        all_results.sort(key=lambda x: -x['rf'])
        top_results = all_results[:12]
        print(f"\nGénération des charts (top {len(top_results)})...")
        csv_paths_by_asset = {a: ASSET_CSV[a] for a in ASSETS}
        # generate_charts ne supporte pas multi-asset nativement :
        # on génère par asset séparément avec les top configs
        for asset in ASSETS:
            asset_charts = charts_dir / asset
            generate_charts(
                top_results,
                [p for p in ASSET_CSV[asset] if Path(p).exists()],
                str(asset_charts),
                n_top=12,
                title_prefix=f"Alts v1 — {asset.upper()}",
                vol_slope_ref_key="vol_0.5h_range",
                asset=asset,
            )
        print(f"Charts -> {charts_dir}/")

    print(f"\nDone -> {out_dir}/")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
