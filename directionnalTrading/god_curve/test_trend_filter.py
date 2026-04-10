"""
Test filtre trend ratio (net/range) sur lb 0.5h
Compare : sans filtre trend / trend > 0.3 / trend > 0.4 / trend > 0.5
Config de base : lin_sl2_c65 + vol0.5h_range > $60
"""
import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path

CSV_PATHS     = ["BTC_BIG.csv", "BTC.csv"]
MAX_ORDERS    = 3
BASE_SIZE     = 50.0
SLOPE         = 2.0
INTERCEPT     = 0.0
CAP           = 0.65
LB_H          = 0.5
RANGE_THRESH  = 60.0
TREND_THRESHS = [None, 0.3, 0.4, 0.5]  # None = pas de filtre trend


def load_contracts():
    dfs = []
    for p in CSV_PATHS:
        if Path(p).exists():
            d = pd.read_csv(p, usecols=['timestamp', 'spot_price', 'm5_up_bid', 'm5_down_bid'])
            d['ts'] = pd.to_datetime(d['timestamp'])
            dfs.append(d)
    df = pd.concat(dfs).sort_values('ts').reset_index(drop=True)
    df['ce_raw'] = df['ts'].dt.floor('5min')

    contracts = []
    for ce, g in df.groupby('ce_raw'):
        g = g.sort_values('ts')
        ce_end = ce + pd.Timedelta(minutes=5)
        contracts.append({
            'ce':     ce_end,
            'ce_ts':  ce_end.timestamp(),
            'op':     g['spot_price'].iloc[0],
            'spot':   g['spot_price'].values,
            'ts':     g['ts'].values,
            'up_bid': g['m5_up_bid'].values,
            'dn_bid': g['m5_down_bid'].values,
        })
    contracts.sort(key=lambda x: x['ce_ts'])

    # Precompute vol 0.5h : range + net + trend
    lb_s = LB_H * 3600
    window = deque()
    for c in contracts:
        t = c['ce_ts']
        while window and t - window[0][0] > lb_s:
            window.popleft()
        if window:
            spots_w = [x[1] for x in window]
            rng   = max(spots_w) - min(spots_w)
            net   = abs(c['op'] - window[0][1])
            trend = net / rng if rng > 0 else 0.0
        else:
            rng = net = trend = 0.0
        c['vol_range'] = rng
        c['vol_net']   = net
        c['vol_trend'] = trend
        window.append((t, c['op']))

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


def run(contracts, trend_thresh):
    pnl_total=0.0; max_dd=0.0; peak=0.0
    day_pnls={}; trades=0; skipped=0

    for c in contracts:
        if c['vol_range'] < RANGE_THRESH:
            continue
        if trend_thresh is not None and c['vol_trend'] < trend_thresh:
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
    return {'trend_thresh': trend_thresh, 'trades': trades, 'skipped': skipped,
            'pnl_total': pnl_total, 'pnl_j': pnl_j, 'max_dd': max_dd,
            'rf': rf, 'lose_days': lose_days, 'worst_day': worst_day, 'days': days}


def main():
    print("Chargement contrats...", flush=True)
    contracts = load_contracts()
    days_total = (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 86400
    print(f"{len(contracts)} contrats  |  {days_total:.1f} jours")

    # Distribution du trend ratio pour info
    trends = [c['vol_trend'] for c in contracts if c['vol_range'] >= RANGE_THRESH]
    import statistics
    print(f"\nDistribution trend ratio (sur contrats vol30m>$60, n={len(trends)}):")
    print(f"  p10={sorted(trends)[int(len(trends)*0.1)]:.2f}  p25={sorted(trends)[int(len(trends)*0.25)]:.2f}  "
          f"p50={sorted(trends)[int(len(trends)*0.5)]:.2f}  p75={sorted(trends)[int(len(trends)*0.75)]:.2f}  "
          f"p90={sorted(trends)[int(len(trends)*0.9)]:.2f}")

    print(f"\n{'Trend filter':>14}  {'Skip':>5}  {'Trades':>7}  {'PnL/j':>8}  {'PnL total':>10}  {'maxDD':>8}  {'RF':>7}  {'LoseDay':>9}  {'WorstDay':>10}")
    print("-" * 95)
    for tt in TREND_THRESHS:
        r = run(contracts, tt)
        label = f"trend>={tt:.1f}" if tt else "no trend filter"
        print(f"  {label:>12}  {r['skipped']:>5}  {r['trades']:>7}  ${r['pnl_j']:>7.1f}  "
              f"${r['pnl_total']:>9.0f}  ${r['max_dd']:>7.0f}  {r['rf']:>6.1f}x  "
              f"{r['lose_days']}/{r['days']}  ${r['worst_day']:>8.2f}")


if __name__ == "__main__":
    main()
