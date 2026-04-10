"""
Test doji slope boost : augmente la slope dynamiquement selon le taux de dojis recent
thresh = (slope + doji_boost * doji_rate_N) * remain + intercept

Config de base : lin_sl2_c65 + vol0.5h_range > $60
Corps doji = bougie dont |close-open| / (high-low) < 0.10
"""
import pandas as pd
import numpy as np
from collections import deque
from pathlib import Path
from itertools import groupby

CSV_PATHS    = ["../BTC_BIG.csv", "../BTC.csv"]
LIVE_CSV     = "../reportLive/safeChase/60_lb0.5h/BTC.csv"
MAX_ORDERS   = 3
BASE_SIZE    = 50.0
BASE_SLOPE   = 2.0
INTERCEPT    = 0.0
CAP          = 0.65
LB_H         = 0.5
RANGE_THRESH = 60.0
DOJI_BODY    = 0.10   # corps < 10% du range = doji

DOJI_WINDOWS = [6, 12, 24]       # nb bougies lookback
DOJI_BOOSTS  = [1.0, 2.0, 5.0, 10.0]  # boost slope par unite de doji_rate


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
        op = spots[0]; cl = spots[-1]
        rng = spots.max() - spots.min()
        body = abs(cl - op)
        is_doji = (body / rng < DOJI_BODY) if rng > 0 else True

        contracts.append({
            'ce':      ce_end,
            'ce_ts':   ce_end.timestamp(),
            'op':      op,
            'spot':    spots,
            'ts':      g['ts'].values,
            'up_bid':  g['m5_up_bid'].values,
            'dn_bid':  g['m5_down_bid'].values,
            'is_doji': is_doji,
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

    # Precompute doji_rate sur N bougies (fenetre AVANT ce contrat, no look-ahead)
    for win_n in DOJI_WINDOWS:
        buf = deque()
        for c in contracts:
            c[f'doji_rate_{win_n}'] = sum(buf) / len(buf) if buf else 0.0
            buf.append(float(c['is_doji']))
            if len(buf) > win_n:
                buf.popleft()

    return contracts


def simulate_contract(c, effective_slope):
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

        thresh = effective_slope * remain + INTERCEPT

        if activated_side:
            buf=(s-op) if activated_side=='UP' else (op-s)
            if buf<thresh:
                if active_order is not None: pending_cancel=True
                pending_place=None; pending_size=None
                activated_side=None; continue

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


def run(contracts, win_n, boost):
    pnl_total=0.0; max_dd=0.0; peak=0.0
    day_pnls={}; trades=0

    for c in contracts:
        if c['vol_range'] < RANGE_THRESH:
            continue
        eff_slope = BASE_SLOPE + boost * c[f'doji_rate_{win_n}']
        pnl = simulate_contract(c, eff_slope)
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
    return {'trades': trades, 'pnl_total': pnl_total, 'pnl_j': pnl_j,
            'max_dd': max_dd, 'rf': rf, 'lose_days': lose_days,
            'worst_day': worst_day, 'days': days}


def main():
    print("=== BACKTEST PRINCIPAL (BTC_BIG + BTC) ===")
    contracts = load_contracts(CSV_PATHS)
    days_total = (contracts[-1]['ce'] - contracts[0]['ce']).total_seconds() / 86400
    print(f"{len(contracts)} contrats  |  {days_total:.1f} jours")

    doji_pct = sum(c['is_doji'] for c in contracts) / len(contracts) * 100
    print(f"% dojis global: {doji_pct:.1f}%")

    # Baseline slope fixe
    r0 = run(contracts, 6, 0.0)
    print(f"\nBaseline (boost=0): trades={r0['trades']}  PnL/j=${r0['pnl_j']:.1f}  maxDD=${r0['max_dd']:.0f}  RF={r0['rf']:.1f}x  loseDays={r0['lose_days']}/{r0['days']}")

    print(f"\n{'Window':>8}  {'boost':>6}  {'eff_slope_range':>16}  {'Trades':>7}  {'PnL/j':>8}  {'maxDD':>8}  {'RF':>7}  {'LoseDay':>9}  {'WorstDay':>10}")
    print("-" * 100)
    for win_n in DOJI_WINDOWS:
        for boost in DOJI_BOOSTS:
            r = run(contracts, win_n, boost)
            max_doji = max(c[f'doji_rate_{win_n}'] for c in contracts)
            slope_range = f"{BASE_SLOPE:.1f} - {BASE_SLOPE + boost*max_doji:.1f}"
            print(f"  {win_n*5:>4}min  {boost:>5.1f}  {slope_range:>16}  {r['trades']:>7}  "
                  f"${r['pnl_j']:>7.1f}  ${r['max_dd']:>7.0f}  {r['rf']:>6.1f}x  "
                  f"{r['lose_days']}/{r['days']}  ${r['worst_day']:>8.2f}")

    # Validation nuit 30/31 mars
    print("\n\n=== VALIDATION NUIT 30/31 MARS ===")
    live_contracts = load_contracts(CSV_PATHS + [LIVE_CSV])
    live_start = pd.Timestamp('2026-03-30 22:39:00', tz='UTC')
    live_c = [c for c in live_contracts if c['ce'] >= live_start]
    eligible = [c for c in live_c if c['vol_range'] >= RANGE_THRESH]
    print(f"{len(eligible)} contrats eligibles (vol30m>$60)")

    # Evolution doji_rate horaire
    print(f"\n  {'heure':>12}  {'doji_rate6':>10}  {'doji_rate12':>11}  {'eff_slope(b=5,w6)':>18}")
    for hour, grp in groupby(eligible, key=lambda c: c['ce'].strftime('%m/%d %H:00')):
        grp = list(grp)
        dr6  = sum(c['doji_rate_6']  for c in grp) / len(grp)
        dr12 = sum(c['doji_rate_12'] for c in grp) / len(grp)
        eff  = BASE_SLOPE + 5.0 * dr6
        print(f"  {hour}      {dr6:.3f}      {dr12:.3f}      {eff:.2f}")

    print(f"\nPnL live selon boost:")
    print(f"{'Window':>8}  {'boost':>6}  {'Trades':>7}  {'PnL total':>10}")
    print("-" * 38)
    # Baseline
    pnl_base = sum(simulate_contract(c, BASE_SLOPE) for c in eligible)
    print(f"  baseline   0.0  {len([c for c in eligible if abs(simulate_contract(c, BASE_SLOPE))>0]):>6}  ${pnl_base:>8.2f}")
    for win_n in DOJI_WINDOWS:
        for boost in DOJI_BOOSTS:
            pnl_tot=0.0; trades=0
            for c in eligible:
                eff = BASE_SLOPE + boost * c[f'doji_rate_{win_n}']
                pnl = simulate_contract(c, eff)
                pnl_tot += pnl
                if pnl != 0.0: trades += 1
            print(f"  {win_n*5:>4}min  {boost:>5.1f}  {trades:>7}  ${pnl_tot:>8.2f}")


if __name__ == "__main__":
    main()
