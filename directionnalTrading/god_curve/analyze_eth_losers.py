"""
Compare les contrats ETH perdants en BT avec leur résultat live.
Pour chaque contrat BT perdant (toutes TFs, Apr 10-16) :
  - pnl BT, sens tradé, settlement
  - si le live a tradé ce contrat : pnl live, sens, fill price
  - si le live n'a pas tradé : 'pas fillé live'
"""
import json, sys
from pathlib import Path
from collections import defaultdict
from datetime import timezone

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_utils as cu
from chart_utils import load_contracts, precompute_vol, simulate_trade_log, \
                        attach_settlement_outcomes, TIMEFRAMES, VOL_LBS

BASE      = Path(__file__).resolve().parent.parent
LIVE_BASE = BASE / "reportLive" / "safeChase" / "slope7cap75"
CSV_LIVE  = BASE / "reportLive" / "safeChase"
CSV_HIST  = BASE / "Datas" / "csv"

START_TS = pd.Timestamp("2026-04-10T00:00:00", tz="UTC").timestamp()
END_TS   = pd.Timestamp("2026-04-17T00:00:00", tz="UTC").timestamp()

ASSET = 'eth'
ETH_CFG = dict(curve='linear', slope=0.2, intercept=0.0, eq_cap=0.85,
               vol_lb_h=2.0, vol_thresh=0.3, vol_type='trend',
               max_losses_cb=None, max_orders=2)
SIZE = 130.0

_SLUG_TF_MAP = {'5m': '5min', '15m': '15min', '1h': '1h'}

def load_jsonl_settlements_eth():
    result = {}
    asset_dir = LIVE_BASE / ASSET
    if not asset_dir.exists():
        return result
    for tf_dir in asset_dir.iterdir():
        if not tf_dir.is_dir():
            continue
        for f in tf_dir.glob('*.jsonl'):
            for line in f.open(encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get('event') != 'window_settled':
                    continue
                slug = d.get('slug', '')
                parts = slug.rsplit('-', 1)
                if len(parts) != 2:
                    continue
                try:
                    ots = int(parts[1])
                except ValueError:
                    continue
                slug_tf = parts[0].rsplit('-', 1)[-1]
                tf_floor = _SLUG_TF_MAP.get(slug_tf, slug_tf)
                up_w = int(d.get('epnl_up_wins',  0))
                dn_w = int(d.get('epnl_down_wins', 0))
                if up_w + dn_w > 0:
                    result[(ots, tf_floor)] = (up_w == 1)
    return result


def parse_live_eth():
    """
    Retourne {(open_ts, tf_floor): dict} avec pnl, side traded, fill_price_avg, won.
    """
    settlement_all = {}
    for tf_csv, tf_off in [('5min', 300), ('15min', 900), ('1h', 3600)]:
        for ots, won in cu.load_settlement_outcomes(ASSET, tf=tf_csv).items():
            settlement_all[(ots, tf_off)] = won

    jsonl_sett = load_jsonl_settlements_eth()
    result = {}

    for tf_name, tf_off in [('m5', 300), ('m15', 900), ('h1', 3600)]:
        tf_floor = {300: '5min', 900: '15min', 3600: '1h'}[tf_off]
        tf_dir = LIVE_BASE / ASSET / tf_name
        if not tf_dir.exists():
            continue
        for f in sorted(tf_dir.glob('*.jsonl')):
            cur_ots = None
            up_s = dn_s = up_c = dn_c = 0.0
            for line in f.open(encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ev = d.get('event', '')
                if ev == 'window_open':
                    cur_ots = None
                    up_s = dn_s = up_c = dn_c = 0.0
                    ts = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc)
                    if ts.timestamp() < START_TS or ts.timestamp() >= END_TS:
                        continue
                    cur_ots = int(pd.Timestamp(ts).floor(
                        pd.Timedelta(seconds=tf_off)).timestamp())
                elif ev in ('window_ended', 'window_settled') and cur_ots is not None:
                    up_s  = float(d.get('up_shares',  0))
                    dn_s  = float(d.get('down_shares', 0))
                    up_c  = float(d.get('up_cost',    0))
                    dn_c  = float(d.get('down_cost',  0))
                    traded = (up_s + dn_s) > 1e-9

                    skey = (cur_ots, tf_off)
                    if skey in settlement_all:
                        up_wins = settlement_all[skey]
                    else:
                        k2 = (cur_ots, tf_floor)
                        if k2 in jsonl_sett:
                            up_wins = jsonl_sett[k2]
                        else:
                            s0 = float(d.get('start_spot', 0))
                            s1 = float(d.get('end_spot',   0))
                            up_wins = (s1 >= s0) if (s0 > 0 and s1 > 0) else True

                    if traded:
                        pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c)
                        side = 'UP' if up_s > dn_s else 'DOWN'
                        shares = up_s if up_s > dn_s else dn_s
                        cost   = up_c if up_s > dn_s else dn_c
                        avg_price = cost / shares if shares > 1e-6 else 0.0
                        result[(cur_ots, tf_floor)] = dict(
                            pnl=pnl, side=side, shares=shares,
                            avg_price=avg_price, won=(pnl > 0), up_wins=up_wins,
                        )
                    cur_ots = None
    return result


# ── BT : collecte les perdants ────────────────────────────────────────────────
print("=== Chargement CSV ETH ===", flush=True)
csv_paths = []
for p in [str(CSV_HIST / "ETH.csv"), str(CSV_LIVE / "ETH.csv")]:
    if Path(p).exists():
        csv_paths.append(p)

jsonl_sett = load_jsonl_settlements_eth()
live_data  = parse_live_eth()

orig = cu.BASE_SIZE; cu.BASE_SIZE = float(SIZE)
TF_MAP = [('5min', 300, 'm5'), ('15min', 900, 'm15'), ('1h', 3600, 'h1')]

bt_losers   = []  # (open_ts, tf_floor, tf_name, pnl_bt, side_bt, up_wins)
bt_all      = []  # tous les trades BT

m5_ref = None
for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_name = ['m5', 'm15', 'h1'][idx]
    tf_off  = [300, 900, 3600][idx]

    try:
        cts = load_contracts(csv_paths, tf_floor, b1, b2, a1, a2)
    except Exception as e:
        print(f"  {tf_floor}: {e}", flush=True); continue

    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS); m5_ref = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
    attach_settlement_outcomes(cts, ASSET, tf=tf_floor)

    for c in cts:
        if '_settlement_won_up' not in c:
            ots = c.get('open_ts', 0)
            k = (ots, tf_floor)
            if k in jsonl_sett:
                c['_settlement_won_up']  = jsonl_sett[k]
                c['_settlement_won_down'] = not jsonl_sett[k]

    vol_key = f"vol_{ETH_CFG['vol_lb_h']:g}h_{ETH_CFG.get('vol_type','range')}"

    for c in cts:
        ots = c.get('open_ts', 0)
        ce_ts = ots + tf_off
        if ce_ts < START_TS or ce_ts >= END_TS:
            continue
        if c.get(vol_key, 0.0) < ETH_CFG.get('vol_thresh', 0):
            continue
        if '_settlement_won_up' not in c:
            continue
        won, pnl, fills = simulate_trade_log(c, ETH_CFG)
        if won is None:
            continue
        up_wins = c['_settlement_won_up']
        side_bt = fills[0]['side'] if fills else ('UP' if not up_wins else 'DOWN')
        bt_all.append((ots, tf_floor, tf_name, pnl, side_bt, up_wins))
        if pnl < 0:
            bt_losers.append((ots, tf_floor, tf_name, pnl, side_bt, up_wins))

cu.BASE_SIZE = orig

print(f"BT total trades: {len(bt_all)}, perdants: {len(bt_losers)}", flush=True)

# ── Comparison losers BT vs live ──────────────────────────────────────────────
print(f"\n{'='*100}")
print(f"  {'Date/Heure (UTC)':<22} {'TF':<5} {'BT PnL':>8} {'BT Side':<7} {'Settlement':<11} "
      f"{'Live PnL':>9} {'Live Side':<10} {'Avg Px':>7} {'Commentaire'}")
print(f"{'='*100}")

n_live_same_lose = 0; n_live_win = 0; n_live_not_traded = 0; n_live_opp_side = 0
sum_bt_lose = 0.0; sum_live_where_bt_lose = 0.0

for ots, tf_floor, tf_name, pnl_bt, side_bt, up_wins in sorted(bt_losers, key=lambda x: x[0]):
    dt = pd.Timestamp(ots, unit='s', tz='UTC')
    dt_str = dt.strftime('%d/%m %H:%M')
    sett_str = 'UP won' if up_wins else 'DOWN won'

    lv = live_data.get((ots, tf_floor))
    sum_bt_lose += pnl_bt

    if lv is None:
        n_live_not_traded += 1
        comment = 'live: pas tradé'
        print(f"  {dt_str:<22} {tf_name:<5} {pnl_bt:>+8.1f} {side_bt:<7} {sett_str:<11} "
              f"{'—':>9} {'—':<10} {'—':>7} {comment}")
    else:
        sum_live_where_bt_lose += lv['pnl']
        if lv['side'] != side_bt:
            n_live_opp_side += 1
            comment = f"COTE DIFF live={lv['side']}"
        elif lv['won']:
            n_live_win += 1
            comment = 'live GAGNE'
        else:
            n_live_same_lose += 1
            comment = 'live PERD aussi'
        print(f"  {dt_str:<22} {tf_name:<5} {pnl_bt:>+8.1f} {side_bt:<7} {sett_str:<11} "
              f"{lv['pnl']:>+9.1f} {lv['side']:<10} {lv['avg_price']:>7.4f} {comment}")

print(f"\n{'='*100}")
print(f"  RÉSUMÉ ETH perdants BT ({len(bt_losers)} contrats) :")
print(f"    Live pas tradé      : {n_live_not_traded:>4}  (BT trade des contrats que le live skip)")
print(f"    Live tradé même côté, PERD aussi : {n_live_same_lose:>4}")
print(f"    Live tradé même côté, GAGNE      : {n_live_win:>4}  (settlement ok mais fill price différent?)")
print(f"    Live tradé côté OPPOSÉ           : {n_live_opp_side:>4}  (BT trigger inverse!)")
print(f"\n    Total PnL BT perdants    : {sum_bt_lose:>+.1f} USD")
n_common = len(bt_losers) - n_live_not_traded
if n_common > 0:
    print(f"    PnL live sur ces mêmes contrats : {sum_live_where_bt_lose:>+.1f} USD "
          f"({n_common} contrats communs)")
print(f"\n  BT global ETH : {sum(x[3] for x in bt_all):>+.1f} USD sur {len(bt_all)} trades")
