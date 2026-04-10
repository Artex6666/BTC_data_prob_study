"""
Test filtre doji - body_avg ratio sur fenetre glissante
Metrique : mean(|close-open| / (high-low)) sur N dernieres bougies M5
Si body_avg < seuil -> marche choppy -> skip

Config de base : lin_sl2_c65 + vol0.5h_range > $60
"""
import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path
from itertools import groupby

CSV_PATHS     = ["../BTC_BIG.csv", "../BTC.csv"]
LIVE_CSV      = "../reportLive/safeChase/60_lb0.5h/BTC.csv"
MAX_ORDERS    = 3
BASE_SIZE     = 50.0
SLOPE         = 2.0
INTERCEPT     = 0.0
CAP           = 0.65
LB_H          = 0.5
RANGE_THRESH  = 60.0

BODY_WINDOWS  = [6, 12, 24]          # nb bougies (30min, 1h, 2h)
BODY_THRESHS  = [0.15, 0.20, 0.25]   # body_avg < seuil -> skip


def load_contracts(csv_paths):
    dfs = []
    for p in csv_paths:
        if Path(p).exists():
            d = pd.read_csv(p, usecols=['timestamp', 'spot_price', 'm5_up_bid', 'm5_down_bid'])
            d['ts'] = pd.to_datetime(d['timestamp'])
            dfs.append(d)
    df = pd.concat(dfs).sort_values('ts').drop_duplicates('timestamp').reset_index(drop=True)
    df['ce_raw'] = df['ts'].dt.floor('5min')

    contracts = []
    for ce, g in df.groupby('ce_raw'):
        g = g.sort_values('ts')
        ce_end = ce + pd.Timedelta(minutes=5)
        spots  = g['spot_price'].values
        op     = spots[0]
        cl     = spots[-1]
        hi     = spots.max()
        lo     = spots.min()
        rng    = hi - lo
        body   = abs(cl - op)
        body_ratio = body / rng if rng > 0 else 0.0

        contracts.append({
            'ce':         ce_end,
            'ce_ts':      ce_end.timestamp(),
            'op':         op,
            'spot':       spots,
            'ts':         g['ts'].values,
            'up_bid':     g['m5_up_bid'].values,
            'dn_bid':     g['m5_down_bid'].values,
            'body_ratio': body_ratio,
        })
    contracts.sort(key=lambda x: x['ce_ts'])

    # Precompute vol 0.5h range
    lb_s = LB_H * 3600
    window = deque()
    for c in contracts:
        t = c['ce_ts']
        while window and t - window[0][0] > lb_s:
            window.popleft()
        spots_w = [x[1] for x in window]
        c['vol_range'] = max(spots_w) - min(spots_w) if spots_w else 0.0
        window.append((t, c['op']))

    # Precompute rolling body_avg sur N bougies (fenetre AVANT ce contrat, pas de look-ahead)
    for win_n in BODY_WINDOWS:
        buf = deque()
        for c in contracts:
            c[f'body_avg_{win_n}'] = sum(buf) / len(buf) if buf else 1.0
            buf.append(c['body_ratio'])
            if len(buf) > win_n:
                buf.popleft()

    return contracts


def simulate_contract(c):
    spots=c['spot']; ts_arr=c['ts']; up_bid=c['up_bid']; dn_bid=c['dn_bid']
    ce_ns=np.datetime64(c['ce']); op=c['op']
    activated_side=None; fill_side=None
    contract_cost=0.0; contract_shares=0.0; fill_count=0
    active_order=None; active_size=None
    pending_place=None; pending_size=None; pending_cancel=False

    for j in range(len(spots)):
        s=spots[j]
        remain=(ce_ns-ts_arr[j])/np.timedelta64(1,'s')
        if remain<=0: break
        if fill_count>=MAX_ORDERS: break
        ub=round(up_bid[j],2); db=round(dn_bid[j],2)

        if pending_cancel and active_order is not None:
            if activated_side:
                cb=ub if activated_side=='UP' else db
                if cb<=round(active_order-0.01,2):
                    fill_count+=1; contract_cost+=active_size
                    contract_shares+=active_size/active_order
                    if fill_side is None: fill_side=activated_side
                    if fill_count>=MAX_ORDERS: active_order=None; break
            active_order=None; active_size=None; pending_cancel=False

        if pending_place is not None:
            active_order=pending_place; active_size=pending_size
            pending_place=None; pending_size=None

        if active_order is not None and activated_side:
            cb=ub if activated_side=='UP' else db
            if cb<=round(active_order-0.01,2):
                fill_count+=1; contract_cost+=active_size
                contract_shares+=active_size/active_order
                if fill_side is None: fill_side=activated_side
                active_order=None; active_size=None
                if fill_count>=MAX_ORDERS: break

        if activated_side:
            buf=(s-op) if activated_side=='UP' else (op-s)
            thresh=SLOPE*remain+INTERCEPT
            if buf<thresh:
                if active_order is not None: pending_cancel=True
                pending_place=None; pending_size=None
                activated_side=None; continue

        thresh=SLOPE*remain+INTERCEPT
        if not activated_side:
            if s-op>=thresh: activated_side='UP'
            elif op-s>=thresh: activated_side='DOWN'
        if not activated_side: continue

        bid=ub if activated_side=='UP' else db
        if bid>=CAP:
            proposed=min(bid,0.99)
            if active_order is None and pending_place is None:
                pending_place=proposed; pending_size=BASE_SIZE
            elif active_order is not None and proposed>active_order:
                pending_cancel=True; pending_place=proposed; pending_size=BASE_SIZE
            elif pending_place is not None and proposed>pending_place:
                pending_place=proposed; pending_size=BASE_SIZE

    won_up=up_bid[-1]>dn_bid[-1]
    if contract_shares>0 and fill_side is not None:
        won=(won_up==(fill_side=='UP'))
        return (contract_shares if won else 0.0)-contract_cost
    return 0.0


def run(contracts, win_n, body_thresh):
    pnl_total=0.0; max_dd=0.0; peak=0.0
    day_pnls={}; trades=0; skipped=0

    for c in contracts:
        if c['vol_range'] < RANGE_THRESH:
            continue
        if c[f'body_avg_{win_n}'] < body_thresh:
            skipped += 1
            continue
        pnl=simulate_contract(c)
        pnl_total+=pnl
        if pnl_total>peak: peak=pnl_total
        dd=peak-pnl_total
        if dd>max_dd: max_dd=dd
        day=c['ce'].date()
        day_pnls[day]=day_pnls.get(day,0.0)+pnl
        if pnl!=0.0: trades+=1

    days=len(day_pnls)
    pnl_j=pnl_total/days if days else 0
    lose_days=sum(1 for v in day_pnls.values() if v<0)
    worst_day=min(day_pnls.values()) if day_pnls else 0
    rf=pnl_total/max_dd if max_dd>0 else 0
    return {'skipped': skipped, 'trades': trades, 'pnl_total': pnl_total,
            'pnl_j': pnl_j, 'max_dd': max_dd, 'rf': rf,
            'lose_days': lose_days, 'worst_day': worst_day, 'days': days}


def main():
    print("=== BACKTEST PRINCIPAL (BTC_BIG + BTC) ===")
    contracts = load_contracts(CSV_PATHS)
    days_total = (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 86400
    print(f"{len(contracts)} contrats  |  {days_total:.1f} jours")

    ratios = [c['body_ratio'] for c in contracts]
    sr = sorted(ratios)
    n = len(sr)
    print(f"Body ratio p10={sr[n//10]:.2f}  p25={sr[n//4]:.2f}  p50={sr[n//2]:.2f}  p75={sr[3*n//4]:.2f}")

    r0 = run(contracts, 6, 0.0)
    print(f"\nBaseline: trades={r0['trades']}  PnL/j=${r0['pnl_j']:.1f}  maxDD=${r0['max_dd']:.0f}  RF={r0['rf']:.1f}x  loseDays={r0['lose_days']}/{r0['days']}")

    print(f"\n{'Window':>8}  {'body<':>7}  {'Skip':>5}  {'Trades':>7}  {'PnL/j':>8}  {'maxDD':>8}  {'RF':>7}  {'LoseDay':>9}  {'WorstDay':>10}")
    print("-" * 88)
    for win_n in BODY_WINDOWS:
        for bt in BODY_THRESHS:
            r = run(contracts, win_n, bt)
            print(f"  {win_n*5:>4}min  {bt:.2f}  {r['skipped']:>5}  {r['trades']:>7}  "
                  f"${r['pnl_j']:>7.1f}  ${r['max_dd']:>7.0f}  {r['rf']:>6.1f}x  "
                  f"{r['lose_days']}/{r['days']}  ${r['worst_day']:>8.2f}")

    # Validation sur la nuit du 30/31 mars
    print("\n\n=== VALIDATION NUIT 30/31 MARS ===")
    live_contracts = load_contracts(CSV_PATHS + [LIVE_CSV])
    live_start = pd.Timestamp('2026-03-30 22:39:00', tz='UTC')
    live_c = [c for c in live_contracts if c['ce'] >= live_start]
    print(f"{len(live_c)} contrats dans la periode live")

    # Evolution body_avg par heure
    print(f"\nEvolution body_avg pendant la nuit (bougies tradees = vol30m>$60):")
    print(f"  {'heure':>8}  {'avg_6(30m)':>10}  {'avg_12(1h)':>10}  {'avg_24(2h)':>10}  {'n':>4}")
    for hour, grp in groupby(live_c, key=lambda c: c['ce'].strftime('%m/%d %H:00')):
        grp = list(grp)
        eligible = [c for c in grp if c['vol_range'] >= RANGE_THRESH]
        if not eligible: continue
        a6  = sum(c['body_avg_6']  for c in eligible) / len(eligible)
        a12 = sum(c['body_avg_12'] for c in eligible) / len(eligible)
        a24 = sum(c['body_avg_24'] for c in eligible) / len(eligible)
        print(f"  {hour}      {a6:.3f}      {a12:.3f}      {a24:.3f}  {len(eligible):>4}")

    print(f"\nPnL live selon filtre body_avg:")
    print(f"{'Window':>8}  {'body<':>7}  {'Skip':>5}  {'PnL total':>10}")
    print("-" * 38)
    for win_n in BODY_WINDOWS:
        for bt in BODY_THRESHS:
            skip=0; pnl_tot=0.0
            for c in live_c:
                if c['vol_range'] < RANGE_THRESH: continue
                if c[f'body_avg_{win_n}'] < bt:
                    skip+=1; continue
                pnl_tot += simulate_contract(c)
            print(f"  {win_n*5:>4}min  {bt:.2f}  {skip:>5}  ${pnl_tot:>8.2f}")


if __name__ == "__main__":
    main()
