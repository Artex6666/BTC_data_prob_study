"""
split_backtest.py — Comparaison per-config sur deux périodes
  - Période A : tout sauf les 5 derniers jours
  - Période B : les 5 derniers jours (mauvaise période)

Configs testées :
  - LIVE actuelle : vol_linear sl=2 cap=0.65 ran0.5h>60
  - Top nouvelles  : vol_linear sl=2 cap=0.75 ran2h>350/200, ran0.5h>100/60
  - expo A40 t20 cap=0.75

maxDD = peak-to-trough sur equity curve per-contrat (pas agrégé par jour).
$/j   = total_pnl / nb_jours_calendaires_avec_data
"""

import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path
from datetime import timedelta

BASE      = Path(__file__).resolve().parent.parent
CSV_PATHS = [
    str(BASE / "Datas/csv/BTC.csv"),
    str(BASE / "reportLive/safeChase/60_lb0.5h/BTC.csv"),
]
VOL_LBS = [0.25, 0.5, 1.0, 2.0, 8.0]
MAX_ORDERS = 3
BASE_SIZE  = 50.0
CB_WINDOW_S = 5400

CONFIGS = [
    dict(name='LIVE  sl=2 c65 ran0.5h>60',
         curve='linear', slope=2.0, intercept=0.0, eq_cap=0.65,
         max_losses_cb=None, vol_lb_h=0.5, vol_thresh=60.0, vol_type='range'),
    dict(name='      sl=2 c75 ran0.5h>60',
         curve='linear', slope=2.0, intercept=0.0, eq_cap=0.75,
         max_losses_cb=None, vol_lb_h=0.5, vol_thresh=60.0, vol_type='range'),
    dict(name='      sl=2 c75 ran0.5h>100',
         curve='linear', slope=2.0, intercept=0.0, eq_cap=0.75,
         max_losses_cb=None, vol_lb_h=0.5, vol_thresh=100.0, vol_type='range'),
    dict(name='      sl=2 c75 ran2h>200',
         curve='linear', slope=2.0, intercept=0.0, eq_cap=0.75,
         max_losses_cb=None, vol_lb_h=2.0, vol_thresh=200.0, vol_type='range'),
    dict(name='      sl=2 c75 ran2h>350',
         curve='linear', slope=2.0, intercept=0.0, eq_cap=0.75,
         max_losses_cb=None, vol_lb_h=2.0, vol_thresh=350.0, vol_type='range'),
    dict(name='      sl=2 c65 ran2h>350',
         curve='linear', slope=2.0, intercept=0.0, eq_cap=0.65,
         max_losses_cb=None, vol_lb_h=2.0, vol_thresh=350.0, vol_type='range'),
    dict(name='expo  A40 t20 c75',
         curve='expo', slope=None, intercept=None, A_exp=40.0, tau=20.0,
         eq_cap=0.75, max_losses_cb=None, vol_lb_h=None, vol_thresh=None),
    dict(name='      sl=3 c55 cb=1 int=30',
         curve='linear', slope=3.0, intercept=30.0, eq_cap=0.55,
         max_losses_cb=1, vol_lb_h=None, vol_thresh=None),
]


def load_contracts():
    dfs = []
    for p in CSV_PATHS:
        if not Path(p).exists():
            continue
        d = pd.read_csv(p, usecols=['timestamp','spot_price',
                                     'm5_up_bid','m5_down_bid',
                                     'm5_up_ask','m5_down_ask'])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    contracts = []
    for ce, grp in df.groupby('contract'):
        if len(grp) < 2:
            continue
        op = float(grp['spot_price'].iloc[0])
        contracts.append({
            'ce':     ce,
            'ce_ts':  ce.timestamp(),
            'op':     op,
            'spot':   grp['spot_price'].values.astype(np.float64),
            'ts':     grp['timestamp'].values,
            'up_bid': grp['m5_up_bid'].values.astype(np.float64),
            'dn_bid': grp['m5_down_bid'].values.astype(np.float64),
        })

    for lb_h in VOL_LBS:
        lb_s = lb_h * 3600
        window = deque()
        for c in contracts:
            t = c['ce_ts']
            while window and t - window[0][0] > lb_s:
                window.popleft()
            if window:
                ops = [x[1] for x in window]
                rng   = max(ops) - min(ops)
                net   = abs(c['op'] - window[0][1])
                trend = net / rng if rng > 0 else 0.0
            else:
                rng = net = trend = 0.0
            c[f'vol_{lb_h:g}h_range'] = rng
            c[f'vol_{lb_h:g}h_net']   = net
            c[f'vol_{lb_h:g}h_trend'] = trend
            window.append((c['ce_ts'], c['op']))

    return contracts


def simulate(c, cfg):
    spots  = c['spot']; ts_arr = c['ts']
    up_bid = c['up_bid']; dn_bid = c['dn_bid']
    ce_ns  = np.datetime64(c['ce']); op = c['op']
    cap    = cfg['eq_cap']; curve = cfg['curve']
    slope  = cfg.get('slope', 2.0); intercept = cfg.get('intercept', 0.0)
    A_exp  = cfg.get('A_exp'); tau = cfg.get('tau')

    activated_side = fill_side = None
    cost = shares = 0.0; fill_count = 0
    active_order = active_size = pending_place = pending_size = None
    pending_cancel = False

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or fill_count >= MAX_ORDERS: break
        ub = round(up_bid[j], 2); db = round(dn_bid[j], 2)

        if pending_cancel and active_order is not None:
            if activated_side:
                cb = ub if activated_side == 'UP' else db
                if cb <= round(active_order - 0.01, 2):
                    fill_count += 1; cost += active_size; shares += active_size / active_order
                    if fill_side is None: fill_side = activated_side
            active_order = active_size = None; pending_cancel = False

        if pending_place is not None:
            active_order = pending_place; active_size = pending_size
            pending_place = pending_size = None

        if active_order is not None and activated_side:
            cb = ub if activated_side == 'UP' else db
            if cb <= round(active_order - 0.01, 2):
                fill_count += 1; cost += active_size; shares += active_size / active_order
                if fill_side is None: fill_side = activated_side
                active_order = active_size = None
                if fill_count >= MAX_ORDERS: break

        if curve == 'linear':
            thresh = slope * remain + intercept
        else:
            thresh = A_exp * (np.exp(remain / tau) - 1.0)

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = None; activated_side = None; continue

        if not activated_side:
            if s - op >= thresh:   activated_side = 'UP'
            elif op - s >= thresh: activated_side = 'DOWN'
        if not activated_side: continue

        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = BASE_SIZE
            elif active_order is not None and proposed > active_order:
                pending_cancel = True; pending_place = proposed; pending_size = BASE_SIZE
            elif pending_place is not None and proposed > pending_place:
                pending_place = proposed; pending_size = BASE_SIZE

    if fill_side is None or shares == 0:
        return None, 0.0
    won_up = up_bid[-1] > dn_bid[-1]
    won = (won_up == (fill_side == 'UP'))
    return won, (shares if won else 0.0) - cost


def max_dd_from_equity(pnl_list):
    """Peak-to-trough sur equity curve per-contrat, départ à 0."""
    peak = 0.0; max_dd = 0.0; eq = 0.0
    for p in pnl_list:
        eq += p
        if eq > peak: peak = eq
        dd = peak - eq
        if dd > max_dd: max_dd = dd
    return max_dd


def run_config(contracts, cfg):
    vol_lb    = cfg.get('vol_lb_h')
    vol_thresh = cfg.get('vol_thresh')
    vol_type  = cfg.get('vol_type', 'range')
    max_losses_cb = cfg.get('max_losses_cb')
    vol_key = f"vol_{vol_lb:g}h_{vol_type}" if vol_lb is not None else None

    cb_window = deque()
    wins = losses = trades = 0
    total_pnl = 0.0; pnl_list = []; day_pnls = {}

    for c in contracts:
        if vol_key and c.get(vol_key, 0) < vol_thresh:
            continue
        if max_losses_cb is not None:
            t = c['ce_ts']
            while cb_window and t - cb_window[0][0] > CB_WINDOW_S:
                cb_window.popleft()
            if sum(1 for _, p in cb_window if p < 0) >= max_losses_cb:
                continue

        won, pnl = simulate(c, cfg)
        if won is None: continue

        trades += 1; total_pnl += pnl; pnl_list.append(pnl)
        if won: wins += 1
        else:   losses += 1
        if max_losses_cb is not None:
            cb_window.append((c['ce_ts'], pnl))
        day_pnls[c['ce'].date()] = day_pnls.get(c['ce'].date(), 0.0) + pnl

    if trades == 0:
        return None

    wr       = wins / trades * 100
    max_dd   = max_dd_from_equity(pnl_list)
    rf       = total_pnl / max_dd if max_dd > 0 else float('inf')
    n_days   = len(day_pnls)
    pnl_day  = total_pnl / n_days if n_days else 0
    lose_days = sum(1 for v in day_pnls.values() if v < 0)
    worst_day = min(day_pnls.values()) if day_pnls else 0.0

    return dict(trades=trades, wins=wins, losses=losses, wr=wr,
                total_pnl=total_pnl, pnl_day=pnl_day, max_dd=max_dd,
                rf=rf, lose_days=lose_days, worst_day=worst_day,
                n_days=n_days)


def print_period(label, contracts):
    if not contracts:
        print("  (aucun contrat)")
        return
    dates = sorted(set(c['ce'].date() for c in contracts))
    print(f"\n  {label}  [{dates[0]} -> {dates[-1]}]  ({len(dates)} jours, {len(contracts)} contrats)")
    print(f"  {'Config':<36}  {'T':>5}  {'WR':>6}  {'$/j':>8}  {'maxDD':>7}  {'RF':>8}  {'loseD':>6}  {'wrstD':>9}")
    print(f"  {'-'*95}")
    for cfg in CONFIGS:
        r = run_config(contracts, cfg)
        if r is None:
            print(f"  {cfg['name']:<36}  -- aucun trade --")
            continue
        rf_s = f"{r['rf']:.1f}x" if r['rf'] != float('inf') else "inf"
        print(f"  {cfg['name']:<36}  {r['trades']:>5}  {r['wr']:>5.1f}%"
              f"  ${r['pnl_day']:>7.0f}  ${r['max_dd']:>6.0f}"
              f"  {rf_s:>8}  {r['lose_days']:>4}/{r['n_days']}"
              f"  ${r['worst_day']:>8.2f}")


def main():
    print("Chargement CSV...", flush=True)
    contracts = load_contracts()
    print(f"  {len(contracts)} contrats totaux")

    all_dates = sorted(set(c['ce'].date() for c in contracts))
    split_date = all_dates[-1] - timedelta(days=5)

    before = [c for c in contracts if c['ce'].date() <= split_date]
    after  = [c for c in contracts if c['ce'].date() >  split_date]

    print(f"\n  Split : avant={split_date}  apres={split_date + timedelta(days=1)}")
    print(f"{'='*100}")

    print_period("PERIODE A — avant les 5 derniers jours", before)
    print(f"\n{'='*100}")
    print_period("PERIODE B — 5 derniers jours (mauvaise periode)", after)
    print(f"\n{'='*100}")
    print_period("PERIODE TOTALE", contracts)


if __name__ == "__main__":
    main()
