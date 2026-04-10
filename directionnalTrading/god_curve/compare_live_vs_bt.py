"""
Compare backtest vs live — période JSONL reportLive/safeChase/60_lb0.5h/btc/m5/
Objectif :
  1. Extraire PnL réel live depuis chaque fichier JSONL (expected_pnl @ window_ended)
  2. Lancer backtest sur la même période avec plusieurs configs
  3. Afficher comparaison contrat par contrat + cumulatif

Config live user : sl=2 int=0 cap=0.65 vol0.5h>$60 (le bot qui tournait)
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path

# ── Chemins ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parent.parent   # directionnalTrading/
REPORTS_DIR = BASE / "reportLive/safeChase/60_lb0.5h/btc/m5"
CSV_PATHS   = [
    str(BASE / "Datas/csv/BTC.csv"),
    str(BASE / "reportLive/safeChase/60_lb0.5h/BTC.csv"),
]

# ── Configs à comparer ───────────────────────────────────────────────────────
# Config actuellement en live (ce que le bot tournait)
LIVE_CONFIG = dict(
    name='LIVE (sl=2 c65 ran0.5h>60)',
    curve='linear', slope=2.0, intercept=0.0,
    eq_cap=0.65, max_losses_cb=None,
    vol_lb_h=0.5, vol_thresh=60.0, vol_type='range',
)

COMPARE_CONFIGS = [
    LIVE_CONFIG,
    dict(name='sl=2 c75 ran0.5h>60 (cap upgrade)',
         curve='linear', slope=2.0, intercept=0.0,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=0.5, vol_thresh=60.0, vol_type='range'),
    dict(name='sl=2 c75 ran0.5h>100',
         curve='linear', slope=2.0, intercept=0.0,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=0.5, vol_thresh=100.0, vol_type='range'),
    dict(name='sl=2 c75 ran2h>200 (RF=65x)',
         curve='linear', slope=2.0, intercept=0.0,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=200.0, vol_type='range'),
    dict(name='sl=2 c75 ran2h>350 (RF=193x)',
         curve='linear', slope=2.0, intercept=0.0,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=350.0, vol_type='range'),
    dict(name='expo A40 t20 c75 (RF=60x)',
         curve='expo', slope=None, intercept=None,
         A_exp=40.0, tau=20.0,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None),
    dict(name='sl=2 c65 ran2h>350',
         curve='linear', slope=2.0, intercept=0.0,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=350.0, vol_type='range'),
    dict(name='sl=3 c55 cb=1 int=30',
         curve='linear', slope=3.0, intercept=30.0,
         eq_cap=0.55, max_losses_cb=1,
         vol_lb_h=None, vol_thresh=None),
]

MAX_ORDERS = 3
BASE_SIZE  = 50.0
CB_WINDOW_S = 5400   # 90 min
VOL_LBS_NEEDED = [0.25, 0.5, 1.0, 2.0, 8.0]


# ── 1. Lecture JSONL live ─────────────────────────────────────────────────────
def load_live_jsonl():
    """Lit tous les JSONL et retourne une liste de dicts par contrat."""
    records = []
    for f in sorted(REPORTS_DIR.glob("*.jsonl")):
        events = {}
        fills_total_cost = 0.0
        fills_total_shares = 0.0
        fill_side = None
        for line in f.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            evt = ev.get('event', '')
            if evt == 'window_open':
                events['open'] = ev
            elif evt == 'window_start':
                events['start'] = ev
            elif evt == 'fill':
                if fill_side is None:
                    fill_side = ev.get('side')
                fills_total_cost   = ev.get('down_cost', 0) + ev.get('up_cost', 0)
                fills_total_shares = ev.get('down_shares', 0) + ev.get('up_shares', 0)
            elif evt == 'window_ended':
                events['ended'] = ev

        if 'open' not in events or 'ended' not in events:
            continue
        ended = events['ended']
        start = events.get('start', {})

        ts_utc = pd.Timestamp(events['open']['ts']).tz_localize(None)
        ce = ts_utc.floor('5min') + pd.Timedelta(minutes=5)
        up_s   = ended.get('up_shares', 0)
        dn_s   = ended.get('down_shares', 0)
        up_c   = ended.get('up_cost', 0)
        dn_c   = ended.get('down_cost', 0)
        expected_pnl = ended.get('expected_pnl', 0.0)
        traded = (up_s + dn_s) > 0

        records.append({
            'ce':             ce,
            'filename':       f.name,
            'vol_live':       start.get('vol_gate_range_usd', 0),
            'traded_live':    traded,
            'fill_side':      fill_side,
            'pnl_live':       expected_pnl if traded else 0.0,
            'up_shares':      up_s,
            'dn_shares':      dn_s,
        })
    records.sort(key=lambda x: x['ce'])
    return records


# ── 2. Chargement et precompute CSV backtest ──────────────────────────────────
def load_contracts():
    dfs = []
    for p in CSV_PATHS:
        if Path(p).exists():
            d = pd.read_csv(p, usecols=['timestamp','spot_price','m5_up_bid','m5_down_bid','m5_up_ask','m5_down_ask'])
            d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
            dfs.append(d)
    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    contracts = []
    for ce, grp in df.groupby('contract'):
        if len(grp) < 2:
            continue
        op = grp['spot_price'].iloc[0]
        contracts.append({
            'ce':     ce,
            'ce_ts':  ce.timestamp(),
            'op':     float(op),
            'spot':   grp['spot_price'].values.astype(np.float64),
            'ts':     grp['timestamp'].values,
            'up_bid': grp['m5_up_bid'].values.astype(np.float64),
            'dn_bid': grp['m5_down_bid'].values.astype(np.float64),
        })

    # Precompute vol rolling
    for lb_h in VOL_LBS_NEEDED:
        lb_s  = lb_h * 3600
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


# ── 3. Simulation d'un contrat ────────────────────────────────────────────────
def simulate_contract(c, cfg):
    spots   = c['spot']
    ts_arr  = c['ts']
    up_bid  = c['up_bid']
    dn_bid  = c['dn_bid']
    ce_ns   = np.datetime64(c['ce'])
    op      = c['op']
    cap     = cfg['eq_cap']
    curve   = cfg['curve']
    slope   = cfg.get('slope', 2.0)
    intercept = cfg.get('intercept', 0.0)
    A_exp   = cfg.get('A_exp')
    tau     = cfg.get('tau')

    activated_side = None; fill_side = None
    contract_cost = 0.0; contract_shares = 0.0; fill_count = 0
    active_order = None; active_size = None
    pending_place = None; pending_size = None; pending_cancel = False

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0:
            break
        if fill_count >= MAX_ORDERS:
            break
        ub = round(up_bid[j], 2)
        db = round(dn_bid[j], 2)

        if pending_cancel and active_order is not None:
            if activated_side:
                cb = ub if activated_side == 'UP' else db
                if cb <= round(active_order - 0.01, 2):
                    fill_count += 1; contract_cost += active_size
                    contract_shares += active_size / active_order
                    if fill_side is None: fill_side = activated_side
                    if fill_count >= MAX_ORDERS:
                        active_order = None; break
            active_order = None; active_size = None; pending_cancel = False

        if pending_place is not None:
            active_order = pending_place; active_size = pending_size
            pending_place = None; pending_size = None

        if active_order is not None and activated_side:
            cb = ub if activated_side == 'UP' else db
            if cb <= round(active_order - 0.01, 2):
                fill_count += 1; contract_cost += active_size
                contract_shares += active_size / active_order
                if fill_side is None: fill_side = activated_side
                active_order = None; active_size = None
                if fill_count >= MAX_ORDERS:
                    break

        if curve == 'linear':
            thresh = slope * remain + intercept
        else:
            thresh = A_exp * (np.exp(remain / tau) - 1.0)

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = None; pending_size = None
                activated_side = None; continue

        if not activated_side:
            if s - op >= thresh:   activated_side = 'UP'
            elif op - s >= thresh: activated_side = 'DOWN'
        if not activated_side:
            continue

        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = BASE_SIZE
            elif active_order is not None and proposed > active_order:
                pending_cancel = True; pending_place = proposed; pending_size = BASE_SIZE
            elif pending_place is not None and proposed > pending_place:
                pending_place = proposed; pending_size = BASE_SIZE

    won_up = up_bid[-1] > dn_bid[-1]
    if contract_shares > 0 and fill_side is not None:
        won = (won_up == (fill_side == 'UP'))
        return (contract_shares if won else 0.0) - contract_cost, fill_side
    return 0.0, None


# ── 4. Run backtest sur periode live pour une config ─────────────────────────
def run_config_on_contracts(contracts, live_ces, cfg):
    """Retourne dict{ce -> {'pnl': float, 'traded': bool}} pour la période live."""
    live_set = set(live_ces)
    vol_lb   = cfg.get('vol_lb_h')
    vol_thresh = cfg.get('vol_thresh')
    vol_type = cfg.get('vol_type', 'range')
    max_losses_cb = cfg.get('max_losses_cb')

    cb_window = deque()  # (ce_ts, pnl) pour circuit breaker
    results = {}

    for c in contracts:
        if c['ce'] not in live_set:
            continue

        # Vol filter
        if vol_lb is not None and vol_thresh is not None:
            vk = f"vol_{vol_lb:g}h_{vol_type}"
            if c.get(vk, 0) < vol_thresh:
                results[c['ce']] = {'pnl': 0.0, 'traded': False, 'reason': 'vol_filter'}
                continue

        # Circuit breaker
        if max_losses_cb is not None:
            t = c['ce_ts']
            while cb_window and t - cb_window[0][0] > CB_WINDOW_S:
                cb_window.popleft()
            losses_in_window = sum(1 for _, p in cb_window if p < 0)
            if losses_in_window >= max_losses_cb:
                results[c['ce']] = {'pnl': 0.0, 'traded': False, 'reason': 'cb'}
                continue

        pnl, fill_side = simulate_contract(c, cfg)

        if max_losses_cb is not None:
            cb_window.append((c['ce_ts'], pnl if fill_side else 0.0))

        if fill_side is not None:
            results[c['ce']] = {'pnl': pnl, 'traded': True}
        else:
            results[c['ce']] = {'pnl': 0.0, 'traded': False, 'reason': 'no_trigger'}

    return results


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():
    print("Chargement JSONL live...", flush=True)
    live_records = load_live_jsonl()
    print(f"  {len(live_records)} contrats live ({live_records[0]['ce']} -> {live_records[-1]['ce']})")

    print("Chargement CSV + precompute...", flush=True)
    contracts = load_contracts()
    by_ce = {c['ce']: c for c in contracts}

    # Filtrer les contrats live qui ont un match CSV
    live_records = [r for r in live_records if r['ce'] in by_ce]
    live_ces     = [r['ce'] for r in live_records]
    print(f"  {len(live_records)} contrats avec match CSV")

    # Stats live
    live_traded    = [r for r in live_records if r['traded_live']]
    live_pnl_total = sum(r['pnl_live'] for r in live_records)
    live_wins      = sum(1 for r in live_records if r['pnl_live'] > 0)
    live_losses    = sum(1 for r in live_records if r['pnl_live'] < 0)

    print(f"\n{'='*70}")
    print(f"LIVE RÉEL : {len(live_traded)} tradés / {len(live_records)} contrats")
    print(f"  Wins: {live_wins}  Losses: {live_losses}  "
          f"WR: {live_wins/(live_wins+live_losses)*100:.1f}%" if (live_wins+live_losses) > 0 else "  Aucun trade")
    print(f"  PnL total live = ${live_pnl_total:.2f}")
    print(f"{'='*70}")

    # Backtest par config
    print("\n\nRESULTATS BACKTEST vs LIVE :")
    print(f"{'Config':<38}  {'Tradés':>7}  {'WR':>6}  {'PnL/contrat':>12}  {'PnL total':>10}  {'vs live':>10}")
    print("-" * 90)

    config_results = {}
    for cfg in COMPARE_CONFIGS:
        bt_res = run_config_on_contracts(contracts, live_ces, cfg)
        bt_traded  = [(ce, r) for ce, r in bt_res.items() if r['traded']]
        bt_wins    = sum(1 for _, r in bt_traded if r['pnl'] > 0)
        bt_losses  = sum(1 for _, r in bt_traded if r['pnl'] < 0)
        bt_pnl     = sum(r['pnl'] for r in bt_res.values())
        wr         = bt_wins / (bt_wins + bt_losses) * 100 if (bt_wins + bt_losses) > 0 else 0
        pnl_per    = bt_pnl / len(bt_traded) if bt_traded else 0
        delta      = bt_pnl - live_pnl_total

        config_results[cfg['name']] = bt_res
        print(f"  {cfg['name']:<36}  {len(bt_traded):>7}  {wr:>5.1f}%  ${pnl_per:>10.2f}  ${bt_pnl:>9.2f}  {'+' if delta>=0 else ''}{delta:>8.2f}")

    # Détail par heure pour la config live vs backtest
    print(f"\n\n{'='*70}")
    print(f"DÉTAIL HORAIRE — config live vs backtest (sl=2 c65 ran0.5h>60)")
    print(f"{'Heure (UTC)':>16}  {'N live':>6}  {'PnL live':>9}  {'N bt':>5}  {'PnL bt':>9}  {'Diff':>8}")
    print("-" * 60)

    by_hour = {}
    for r in live_records:
        h = r['ce'].floor('h')
        if h not in by_hour:
            by_hour[h] = {'live_pnl': 0.0, 'live_n': 0, 'bt_pnl': 0.0, 'bt_n': 0}
        by_hour[h]['live_pnl'] += r['pnl_live']
        if r['traded_live']:
            by_hour[h]['live_n'] += 1

    bt_live_res = config_results.get(LIVE_CONFIG['name'], {})
    for ce, bt_r in bt_live_res.items():
        h = ce.floor('h')
        if h in by_hour:
            by_hour[h]['bt_pnl'] += bt_r['pnl']
            if bt_r['traded']:
                by_hour[h]['bt_n'] += 1

    for h in sorted(by_hour.keys()):
        d = by_hour[h]
        diff = d['bt_pnl'] - d['live_pnl']
        print(f"  {str(h):>16}  {d['live_n']:>6}  ${d['live_pnl']:>8.2f}  {d['bt_n']:>5}  ${d['bt_pnl']:>8.2f}  {'+' if diff>=0 else ''}{diff:>7.2f}")

    # Détail contrat par contrat pour les pertes live
    print(f"\n\n{'='*70}")
    print(f"CONTRATS PERDANTS LIVE vs BACKTEST (sl=2 c65 ran0.5h>60 et ran2h>350)")
    print(f"{'ce (UTC)':>20}  {'PnL live':>9}  {'bt c65 0.5h':>11}  {'bt c75 2h>350':>14}  {'vol0.5h':>8}  {'vol2h':>8}")
    print("-" * 80)

    bt_2h350_res = config_results.get('sl=2 c75 ran2h>350 (RF=193x)', {})
    for r in sorted(live_records, key=lambda x: x['ce']):
        pnl_live = r['pnl_live']
        if not r['traded_live'] and pnl_live == 0:
            continue  # skip non-traded
        bt_c65 = bt_live_res.get(r['ce'], {}).get('pnl', 0.0)
        bt_2h  = bt_2h350_res.get(r['ce'], {}).get('pnl', 0.0)
        traded_2h = bt_2h350_res.get(r['ce'], {}).get('traded', False)
        c = by_ce[r['ce']]
        vol_0h5 = c.get('vol_0.5h_range', 0)
        vol_2h  = c.get('vol_2h_range', 0)
        mark = ' <<<' if pnl_live < -5 else ''
        print(f"  {str(r['ce']):>20}  ${pnl_live:>8.2f}  ${bt_c65:>10.2f}  "
              f"{'SKIP' if not traded_2h else f'${bt_2h:.2f}':>14}  "
              f"${vol_0h5:>7.1f}  ${vol_2h:>7.1f}{mark}")


if __name__ == "__main__":
    main()
