"""
Debug : montre les ticks BT des 2 contrats ETH "live pas tradé"
pour voir exactement comment/quand le BT a fillé.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_utils as cu
from chart_utils import load_contracts, precompute_vol, simulate_trade_log, \
                        attach_settlement_outcomes, TIMEFRAMES, VOL_LBS

BASE     = Path(__file__).resolve().parent.parent
CSV_LIVE = BASE / "reportLive" / "safeChase"
CSV_HIST = BASE / "Datas" / "csv"

csv_paths = [str(CSV_HIST / "ETH.csv"), str(CSV_LIVE / "ETH.csv")]
ETH_CFG = dict(curve='linear', slope=0.2, intercept=0.0, eq_cap=0.85,
               vol_lb_h=2.0, vol_thresh=0.3, vol_type='trend',
               max_losses_cb=None, max_orders=2)

# Contrats à débugger
# 10/04 16:25 UTC → open_ts = ?  (Paris UTC+2 → 14:25 UTC)
# 13/04 03:00 UTC (h1) → open_ts = 1775833200 ? non, slug: ethereum-up-or-down-april-12-2026-9pm-et = 01:00 UTC 13 avr

# Les timestamps dans analyze_eth_losers sont affichés en UTC (pd.Timestamp.strftime UTC).
# "10/04 16:25" = UTC 16:25 → open_ts = 1775838300
# "13/04 03:00" = UTC 03:00 → open_ts = 1776049200
ts1_m5 = int(pd.Timestamp("2026-04-10 16:25:00", tz="UTC").timestamp())  # UTC 16:25
ts2_h1 = int(pd.Timestamp("2026-04-13 03:00:00", tz="UTC").timestamp())  # UTC 03:00

print(f"ts m5  10/04 16:25 Paris = {ts1_m5}  ({pd.Timestamp(ts1_m5, unit='s', tz='UTC')})")
print(f"ts h1  13/04 03:00 Paris = {ts2_h1}  ({pd.Timestamp(ts2_h1, unit='s', tz='UTC')})")

targets = {
    ('5min', ts1_m5): 'm5 10/04 16:25',
    ('1h',   ts2_h1): 'h1 13/04 03:00',
}

orig = cu.BASE_SIZE; cu.BASE_SIZE = 130.0
m5_ref = None

for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_name = ['m5', 'm15', 'h1'][idx]
    tf_off  = [300, 900, 3600][idx]
    cts = load_contracts(csv_paths, tf_floor, b1, b2, a1, a2)
    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS); m5_ref = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
    attach_settlement_outcomes(cts, 'eth', tf=tf_floor)

    for c in cts:
        key = (tf_floor, c.get('open_ts', 0))
        if key not in targets:
            continue
        lbl = targets[key]
        print(f"\n{'='*70}")
        print(f"  {lbl}  |  tf={tf_floor}  open_ts={c['open_ts']}")
        print(f"  open_price = {c['op']}  |  settlement_won_up = {c.get('_settlement_won_up', 'N/A')}")
        spots   = c['spot']
        ts_arr  = c['ts']
        up_bid  = c['up_bid']
        dn_bid  = c['down_bid']
        ce_ns   = np.datetime64(c['ce'])

        print(f"  {'sec_remain':>10} {'spot':>10} {'delta':>8} {'thresh':>8} {'up_bid':>7} {'dn_bid':>7}")
        for j in range(len(spots)):
            s      = spots[j]
            remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
            thresh = ETH_CFG['slope'] * remain
            delta  = s - c['op']
            ub     = round(float(up_bid[j]), 2)
            db     = round(float(dn_bid[j]), 2)
            if remain <= 0: break
            # Affiche seulement les ticks intéressants (near threshold ou fill)
            if abs(delta) >= thresh * 0.8 or db >= 0.85 or ub >= 0.85:
                print(f"  {remain:>10.1f} {s:>10.2f} {delta:>+8.2f} {thresh:>8.3f} {ub:>7.2f} {db:>7.2f}")

        print(f"\n  --- Simulation complète ---")
        won, pnl, fills = simulate_trade_log(c, ETH_CFG)
        if won is None:
            print(f"  won=None  pnl=None  fills=0  (aucun trade BT)")
        else:
            print(f"  won={won}  pnl={pnl:+.1f}  fills={len(fills)}")
        for fi in fills:
            j = fi['j']
            remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
            db_at_fill = round(float(dn_bid[j]), 2)
            ub_at_fill = round(float(up_bid[j]), 2)
            print(f"    Fill #{fi['side']} @ price={fi['price']}  "
                  f"(j={j}, remain={remain:.1f}s, dn_bid={db_at_fill}, up_bid={ub_at_fill}, spot={spots[j]:.2f})")

cu.BASE_SIZE = orig
