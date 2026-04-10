"""
God Curve Optimizer v5 - BTC M5
Objectif  : maximiser RF (PNL/maxDD) et minimiser losing_days
Courbes   : lineaire + exponentielle
Filtres   : circuit breaker N pertes/90min + RSI temporel anti-spike
RSI       : fenetre glissante en secondes (event-driven, floor 50ms)
            skip activation si RSI extreme (overbought UP ou oversold DOWN)
Tri       : par Recovery Factor
Sizing    : $50/ordre, MAX_ORDERS=3
"""

import pandas as pd
import numpy as np
import multiprocessing as mp
import time
from datetime import datetime
from collections import deque

MAX_ORDERS  = 3
BASE_SIZE   = 50.0
CSV_PATHS   = ["BTC_BIG.csv", "BTC.csv"]
OUTPUT_FILE = "god_curve_results_v5.txt"
N_WORKERS   = max(1, mp.cpu_count() - 2)
CB_WINDOW_S = 5400  # 90 min circuit breaker


def build_configs():
    configs = []

    # Groupe A : lineaire int=0 uniquement (intercept>0 explore en D)
    # 4 slopes x 6 caps = 24 configs
    for slope in [1.0, 2.0, 3.0, 5.0]:
        for cap in [0.65, 0.70, 0.80, 0.85, 0.90, 0.95]:
            configs.append(dict(
                label='linear', curve='linear',
                slope=slope, intercept=0.0,
                A_exp=None, tau=None,
                eq_cap=cap, max_losses_cb=None,
                rsi_window_s=None, rsi_extreme=None,
            ))

    # Groupe B : exponentielle thresh = A*(exp(remain/tau)-1)
    # 3 A x 3 tau x 5 caps = 45 configs
    for A_exp in [10.0, 20.0, 40.0]:
        for tau in [20.0, 40.0, 60.0]:
            for cap in [0.65, 0.70, 0.80, 0.85, 0.90]:
                configs.append(dict(
                    label='expo', curve='expo',
                    slope=None, intercept=None,
                    A_exp=A_exp, tau=tau,
                    eq_cap=cap, max_losses_cb=None,
                    rsi_window_s=None, rsi_extreme=None,
                ))

    # Groupe C : circuit breaker (sur meilleures zones connues)
    # slopes [2,3] x intercepts [0,5] x caps [0.65,0.85,0.90] x cb [1,2] = 24 configs
    for slope in [2.0, 3.0]:
        for intercept in [0.0, 5.0]:
            for cap in [0.65, 0.85, 0.90]:
                for max_losses in [1, 2]:
                    configs.append(dict(
                        label='cb_linear', curve='linear',
                        slope=slope, intercept=intercept,
                        A_exp=None, tau=None,
                        eq_cap=cap, max_losses_cb=max_losses,
                        rsi_window_s=None, rsi_extreme=None,
                    ))

    # Groupe D : RSI anti-spike (fenetre temporelle, pas de periodes fixes)
    # RSI calcule sur tous les ticks des X dernieres secondes au moment d'activation
    # skip UP si RSI > rsi_extreme (overbought = spike) / skip DOWN si RSI < 100-rsi_extreme
    # slopes [2,3] x intercepts [0,5] x caps [0.65,0.85,0.90] x rsi_win [30,60,120] x rsi_ext [70,75,80]
    # = 2x2x3x3x3 = 108 configs
    for slope in [2.0, 3.0]:
        for intercept in [0.0, 5.0]:
            for cap in [0.65, 0.85, 0.90]:
                for rsi_win in [30, 60, 120]:
                    for rsi_ext in [70, 75, 80]:
                        configs.append(dict(
                            label='rsi_filter', curve='linear',
                            slope=slope, intercept=intercept,
                            A_exp=None, tau=None,
                            eq_cap=cap, max_losses_cb=None,
                            rsi_window_s=float(rsi_win),
                            rsi_extreme=float(rsi_ext),
                        ))

    return configs  # total: 201 configs


# ── RSI temporel (fenetre en secondes, event-driven) ──────────────────────────
def compute_rsi(spot_ctx, ts_ctx, current_ts_ns, window_s):
    """
    RSI sur tous les ticks dans [current_ts - window_s, current_ts].
    Retourne 50 si pas assez de ticks (neutre = pas de filtre).
    """
    cutoff = current_ts_ns - np.timedelta64(int(window_s * 1e9), 'ns')
    mask   = ts_ctx >= cutoff
    prices = spot_ctx[mask]
    if len(prices) < 3:
        return 50.0  # neutre, pas assez de donnees
    deltas = np.diff(prices)
    gains  = float(deltas[deltas > 0].sum())
    losses = float(-deltas[deltas < 0].sum())
    if losses == 0:
        return 100.0
    if gains == 0:
        return 0.0
    return 100.0 - (100.0 / (1.0 + gains / losses))


# ── Simulation d'un contrat ───────────────────────────────────────────────────
def simulate_contract(c, cfg):
    curve       = cfg['curve']
    cap         = cfg['eq_cap']
    rsi_win     = cfg['rsi_window_s']
    rsi_ext     = cfg['rsi_extreme']
    ce_ns       = np.datetime64(c['ce'])
    spots       = c['spot']; ts_arr = c['ts']; op = c['op']
    # Contexte etendu pour RSI (jusqu'a 120s)
    spot_ctx    = c['spot_ctx']; ts_ctx = c['ts_ctx']

    activated_side = None; fill_side = None
    contract_cost = 0.0; contract_shares = 0.0; fill_count = 0
    active_order = None; active_size = None
    pending_place = None; pending_size = None; pending_cancel = False

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0: break
        if fill_count >= MAX_ORDERS: break
        ub = round(c['up_bid'][j], 2)
        db = round(c['down_bid'][j], 2)

        if pending_cancel and active_order is not None:
            if activated_side:
                cb = ub if activated_side == 'UP' else db
                if cb <= round(active_order - 0.01, 2):
                    fill_count += 1; contract_cost += active_size
                    contract_shares += active_size / active_order
                    if fill_side is None: fill_side = activated_side
                    if fill_count >= MAX_ORDERS: active_order = None; break
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
                if fill_count >= MAX_ORDERS: break

        if curve == 'linear':
            thresh = cfg['slope'] * remain + cfg['intercept']
        else:
            thresh = cfg['A_exp'] * (np.exp(remain / cfg['tau']) - 1.0)

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = None; pending_size = None
                activated_side = None; continue

        if not activated_side:
            new_side = None
            if   s - op >= thresh: new_side = 'UP'
            elif op - s >= thresh: new_side = 'DOWN'

            if new_side is not None:
                # Filtre RSI anti-spike au moment de la premiere activation
                if rsi_win is not None:
                    rsi = compute_rsi(spot_ctx, ts_ctx, ts_arr[j], rsi_win)
                    if new_side == 'UP'   and rsi > rsi_ext:
                        continue  # overbought, spike probable
                    if new_side == 'DOWN' and rsi < (100.0 - rsi_ext):
                        continue  # oversold, spike probable
                activated_side = new_side

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

    won_up = c['up_bid'][-1] > c['down_bid'][-1]
    if contract_shares > 0:
        won = (won_up == (fill_side == 'UP'))
        pnl = (contract_shares if won else 0.0) - contract_cost
        return won, pnl
    return None, None


# ── Worker ────────────────────────────────────────────────────────────────────
def worker_fn(config_batch):
    import pandas as pd, numpy as np
    from collections import deque

    dfs = []
    for p in CSV_PATHS:
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
        # t60 : fenetre de trading (derniere minute)
        t60 = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=60)]
        if len(t60) < 2: continue
        # t120 : contexte etendu pour RSI (120s lookback depuis n'importe quel tick t60)
        t120 = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=120)]
        contracts.append({
            'ce':       ce,
            'ce_ts':    ce.timestamp(),
            'op':       float(op_spot),
            # Trading
            'spot':     t60['spot_price'].values.astype(np.float64),
            'ts':       t60['timestamp'].values,
            'up_bid':   t60['m5_up_bid'].values.astype(np.float64),
            'down_bid': t60['m5_down_bid'].values.astype(np.float64),
            'up_ask':   t60['m5_up_ask'].values.astype(np.float64),
            'down_ask': t60['m5_down_ask'].values.astype(np.float64),
            # Contexte RSI
            'spot_ctx': t120['spot_price'].values.astype(np.float64),
            'ts_ctx':   t120['timestamp'].values,
        })
    del df

    total_hours = (
        (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 3600
        if len(contracts) > 1 else 1
    )

    results = []
    for cfg in config_batch:
        wins = 0; losses = 0
        total_pnl = 0.0; win_pnl = 0.0; loss_pnl = 0.0
        trades = 0; day_pnls = {}
        max_losses_cb = cfg['max_losses_cb']
        loss_times = deque()

        for c in contracts:
            if max_losses_cb is not None:
                ce_ts = c['ce_ts']
                while loss_times and ce_ts - loss_times[0] > CB_WINDOW_S:
                    loss_times.popleft()
                if len(loss_times) >= max_losses_cb:
                    continue

            won, pnl = simulate_contract(c, cfg)
            if won is None: continue

            trades += 1; total_pnl += pnl
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

        pnl_list = [float(v) for _, v in sorted(day_pnls.items())]
        cum_arr  = np.cumsum(pnl_list)
        peak = cum_arr[0]; max_dd = 0.0
        for v in cum_arr:
            if v > peak: peak = v
            dd = peak - v
            if dd > max_dd: max_dd = dd

        rf = total_pnl / max_dd if max_dd > 0 else total_pnl

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
            'total_hours': total_hours,
        })

    return results


def fmt_row(i, r, days):
    if r['curve'] == 'expo':
        curve_s = f"A={r['A_exp']:.0f} t={r['tau']:.0f}"
    else:
        curve_s = f"sl={r['slope']:.0f} int={r['intercept']:.0f}"
    cb_s  = str(r['max_losses_cb']) if r['max_losses_cb'] else "-"
    rsi_s = (f"rsi{r['rsi_window_s']:.0f}>{r['rsi_extreme']:.0f}"
             if r['rsi_window_s'] else "      -      ")
    rf_s  = f"{r['rf']:>6.1f}x" if r['max_dd'] > 0 else "   inf "
    return (
        f"{i:>3d} {r['label']:>10s} {curve_s:>14s} cap={r['eq_cap']:.2f} "
        f"cb={cb_s} {rsi_s} | "
        f"T={r['trades']:>4d} WR={r['wr']:>5.1f}% ${r['total_pnl']/days:>6.1f}/j | "
        f"loseDay={r['losing_days']:>2d}/{int(days):>2d} "
        f"wrstDay=${r['worst_day']:>7.2f} maxDD=${r['max_dd']:>6.0f} RF={rf_s}\n"
    )


def main():
    t0 = time.time()
    configs = build_configs()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] God Curve Optimizer v5")
    print(f"  Workers={N_WORKERS}  Configs={len(configs)}  Tri=RF  $50/ordre\n")

    batch_size = max(1, len(configs) // N_WORKERS + 1)
    batches = [configs[i:i+batch_size] for i in range(0, len(configs), batch_size)]

    all_results = []
    with mp.Pool(processes=N_WORKERS) as pool:
        for i, br in enumerate(pool.imap_unordered(worker_fn, batches)):
            all_results.extend(br)
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] batch {i+1}/{len(batches)} — {len(all_results)} configs", flush=True)

    if not all_results: print("Aucun resultat."); return

    days = all_results[0]['total_hours'] / 24
    all_results.sort(key=lambda x: -x['rf'])
    elapsed = time.time() - t0

    sep = "─" * 130 + "\n"
    lines = [
        f"God Curve Optimizer v5  —  Tri RF  |  "
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
        f"Data: {days:.1f}j  |  Configs: {len(all_results)}/{len(configs)}  |  {elapsed:.0f}s\n",
        sep,
    ]

    for group in ['linear', 'expo', 'cb_linear', 'rsi_filter']:
        grp = [r for r in all_results if r['label'] == group]
        if not grp: continue
        lines.append(f"\n>>> {group.upper()} — top 15 par RF\n{sep}")
        for i, r in enumerate(grp[:15]):
            lines.append(fmt_row(i+1, r, days))

    lines.append(f"\n>>> TOP 20 GLOBAL par RF\n{sep}")
    for i, r in enumerate(all_results[:20]):
        lines.append(fmt_row(i+1, r, days))

    # Top par consistance (min losing_days, volume minimum)
    by_consist = sorted(
        [r for r in all_results if r['trades'] >= 400],
        key=lambda x: (x['losing_days'], -x['rf'])
    )
    lines.append(f"\n>>> TOP 15 CONSISTANCE (min losing_days, min 400 trades)\n{sep}")
    for i, r in enumerate(by_consist[:15]):
        lines.append(fmt_row(i+1, r, days))

    output = "".join(lines)
    print("\n" + output)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f"\nSauvegarde -> {OUTPUT_FILE}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
