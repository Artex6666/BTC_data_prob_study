"""
Compare le timing des fills BT vs live sur les contrats perdants BT.
Pour chaque contrat BT perdant (ETH Apr 13-16) :
  - remain au moment du fill BT (depuis simulate_trade_log)
  - remain au moment du fill live (depuis event 'fill' dans JSONL)
  - côté BT vs côté live
"""
import json, sys
from pathlib import Path
from datetime import timezone

import numpy as np
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

START_TS = pd.Timestamp("2026-04-13T00:00:00", tz="UTC").timestamp()
END_TS   = pd.Timestamp("2026-04-17T00:00:00", tz="UTC").timestamp()

ASSET = 'eth'
ETH_CFG = dict(curve='linear', slope=0.2, intercept=0.0, eq_cap=0.85,
               vol_lb_h=2.0, vol_thresh=0.3, vol_type='trend',
               max_losses_cb=None, max_orders=2)
SIZE = 130.0

TF_OFF_MAP = {'5min': 300, '15min': 900, '1h': 3600}
TF_NAME_MAP = {'5min': 'm5', '15min': 'm15', '1h': 'h1'}
TF_SLUG_MAP = {'5min': '5m', '15min': '15m', '1h': '1h'}

_SLUG_TF_MAP = {'5m': '5min', '15m': '15min', '1h': '1h'}


def load_jsonl_settlements():
    result = {}
    asset_dir = LIVE_BASE / ASSET
    if not asset_dir.exists():
        return result
    for tf_dir in asset_dir.iterdir():
        if not tf_dir.is_dir(): continue
        for f in tf_dir.glob('*.jsonl'):
            for line in f.open(encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except: continue
                if d.get('event') != 'window_settled': continue
                slug = d.get('slug', '')
                parts = slug.rsplit('-', 1)
                if len(parts) != 2: continue
                try: ots = int(parts[1])
                except ValueError: continue
                slug_tf_raw = parts[0].rsplit('-', 1)[-1]
                tf_floor = _SLUG_TF_MAP.get(slug_tf_raw, slug_tf_raw)
                up_w = int(d.get('epnl_up_wins', 0))
                dn_w = int(d.get('epnl_down_wins', 0))
                if up_w + dn_w > 0:
                    result[(ots, tf_floor)] = (up_w == 1)
    return result


def load_live_fills():
    """
    Retourne {(open_ts, tf_floor): [(fill_ts, side, price, remain)]}
    """
    result = {}
    for tf_name, tf_off in [('m5', 300), ('m15', 900), ('h1', 3600)]:
        tf_floor = {300: '5min', 900: '15min', 3600: '1h'}[tf_off]
        tf_dir = LIVE_BASE / ASSET / tf_name
        if not tf_dir.exists(): continue
        for f in sorted(tf_dir.glob('*.jsonl')):
            cur_ots = None
            cur_fills = []
            for line in f.open(encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except: continue
                ev = d.get('event', '')
                if ev == 'window_open':
                    cur_ots = None; cur_fills = []
                    ts = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc)
                    if ts.timestamp() < START_TS or ts.timestamp() >= END_TS:
                        continue
                    cur_ots = int(pd.Timestamp(ts).floor(
                        pd.Timedelta(seconds=tf_off)).timestamp())
                elif ev == 'fill' and cur_ots is not None:
                    fill_ts = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc).timestamp()
                    ce_ts = cur_ots + tf_off
                    remain = ce_ts - fill_ts
                    side = d.get('side', '?')
                    price = float(d.get('price', 0))
                    cur_fills.append((fill_ts, side, price, remain))
                elif ev in ('window_ended', 'window_settled') and cur_ots is not None:
                    if cur_fills:
                        result[(cur_ots, tf_floor)] = cur_fills
                    cur_ots = None
    return result


# ── Chargement ────────────────────────────────────────────────────────────────
print("=== Chargement ===", flush=True)
csv_paths = [str(CSV_HIST / "ETH.csv"), str(CSV_LIVE / "ETH.csv")]
jsonl_sett = load_jsonl_settlements()
live_fills = load_live_fills()

orig = cu.BASE_SIZE; cu.BASE_SIZE = float(SIZE)
m5_ref = None

bt_losers = []   # (ots, tf_floor, pnl, fills_bt, c)

for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_off  = TF_OFF_MAP[tf_floor]
    cts = load_contracts(csv_paths, tf_floor, b1, b2, a1, a2)
    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS); m5_ref = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
    attach_settlement_outcomes(cts, ASSET, tf=tf_floor)
    for c in cts:
        if '_settlement_won_up' not in c:
            key = (c.get('open_ts', 0), tf_floor)
            if key in jsonl_sett:
                c['_settlement_won_up']  = jsonl_sett[key]
                c['_settlement_won_down'] = not jsonl_sett[key]

    vol_key = f"vol_{ETH_CFG['vol_lb_h']:g}h_{ETH_CFG.get('vol_type','range')}"

    for c in cts:
        ots = c.get('open_ts', 0)
        ce_ts = ots + tf_off
        if ce_ts < START_TS or ce_ts >= END_TS: continue
        if c.get(vol_key, 0.0) < ETH_CFG.get('vol_thresh', 0): continue
        if '_settlement_won_up' not in c: continue
        won, pnl, fills = simulate_trade_log(c, ETH_CFG)
        if won is None: continue
        if pnl < 0:
            bt_losers.append((ots, tf_floor, pnl, fills, c))

cu.BASE_SIZE = orig
print(f"BT perdants ETH Apr 13-16 : {len(bt_losers)}", flush=True)


# ── Comparaison timing ────────────────────────────────────────────────────────
print(f"\n{'='*110}")
print(f"  {'Date (UTC)':<17} {'TF':<5} {'BT PnL':>8}  "
      f"{'BT side':>7} {'BT remain':>10} {'BT bid':>7}  |  "
      f"{'LV side':>7} {'LV remain':>10} {'LV price':>8}  "
      f"{'Settlement':<10} {'Note'}")
print(f"{'='*110}")

for ots, tf_floor, pnl, fills_bt, c in sorted(bt_losers, key=lambda x: x[0]):
    tf_off  = TF_OFF_MAP[tf_floor]
    tf_name = TF_NAME_MAP[tf_floor]
    ce_ts   = ots + tf_off
    ce_ns   = np.datetime64(c['ce'])
    dt_str  = pd.Timestamp(ots, unit='s', tz='UTC').strftime('%d/%m %H:%M')
    up_wins = c['_settlement_won_up']
    sett_str = 'UP won' if up_wins else 'DN won'

    # BT fills
    bt_lines = []
    for fi in fills_bt:
        j = fi['j']
        remain_bt = (ce_ns - c['ts'][j]) / np.timedelta64(1, 's')
        bid_at_fill = round(float((c['up_bid'] if fi['side']=='UP' else c['down_bid'])[j]), 2)
        bt_lines.append((fi['side'], remain_bt, bid_at_fill, fi['price']))

    # Live fills
    lv_fills = live_fills.get((ots, tf_floor), [])

    # Check direction mismatch
    bt_sides = set(fi['side'] for fi in fills_bt)
    lv_sides = set(f[1] for f in lv_fills) if lv_fills else set()

    if lv_fills:
        if lv_sides != bt_sides:
            note = f"COTES DIFF bt={bt_sides} lv={lv_sides}"
        else:
            note = "meme cote"
    else:
        note = "live: pas trade"

    # Affichage ligne principale
    for i, (bt_s, bt_r, bt_b, bt_p) in enumerate(bt_lines):
        if i == 0:
            lv_str = ""
            if lv_fills:
                lv = lv_fills[0]
                lv_str = f"{lv[1]:>7} {lv[3]:>10.1f}s {lv[2]:>8.2f}"
            else:
                lv_str = f"{'—':>7} {'—':>10} {'—':>8}"
            print(f"  {dt_str:<17} {tf_name:<5} {pnl:>+8.1f}  "
                  f"{bt_s:>7} {bt_r:>9.1f}s {bt_b:>7.2f}  |  "
                  f"{lv_str}  {sett_str:<10} {note}")
        else:
            # 2ème fill BT
            lv_str = ""
            if len(lv_fills) > i:
                lv = lv_fills[i]
                lv_str = f"{lv[1]:>7} {lv[3]:>10.1f}s {lv[2]:>8.2f}"
            else:
                lv_str = f"{'':>7} {'':>10} {'':>8}"
            print(f"  {'':17} {'':5} {'':8}  "
                  f"{bt_s:>7} {bt_r:>9.1f}s {bt_b:>7.2f}  |  "
                  f"{lv_str}")

    # Fills live supplémentaires
    for i in range(len(fills_bt), len(lv_fills)):
        lv = lv_fills[i]
        print(f"  {'':17} {'':5} {'':8}  "
              f"{'':>7} {'':>10} {'':>7}  |  "
              f"{lv[1]:>7} {lv[3]:>10.1f}s {lv[2]:>8.2f}")

print(f"\n{'='*110}")

# ── Stats résumé ─────────────────────────────────────────────────────────────
same_side_same_time = same_side_diff_time = diff_side = not_traded_lv = 0
early_bt = late_bt = 0  # early = remain > 30s, late = remain <= 30s

for ots, tf_floor, pnl, fills_bt, c in bt_losers:
    tf_off  = TF_OFF_MAP[tf_floor]
    ce_ns   = np.datetime64(c['ce'])
    lv_fills = live_fills.get((ots, tf_floor), [])
    bt_sides = set(fi['side'] for fi in fills_bt)
    lv_sides = set(f[1] for f in lv_fills) if lv_fills else set()

    # Timing BT
    first_fill = fills_bt[0]
    j = first_fill['j']
    remain_bt = (ce_ns - c['ts'][j]) / np.timedelta64(1, 's')
    if remain_bt > 30: early_bt += 1
    else: late_bt += 1

    if not lv_fills:
        not_traded_lv += 1
    elif lv_sides != bt_sides:
        diff_side += 1
    else:
        lv_remain = lv_fills[0][3]
        if abs(remain_bt - lv_remain) < 10:
            same_side_same_time += 1
        else:
            same_side_diff_time += 1

print(f"\n  RÉSUMÉ ({len(bt_losers)} perdants BT) :")
print(f"    BT fill > 30s avant expiry : {early_bt}  (trigger early)")
print(f"    BT fill <= 30s avant expiry: {late_bt}  (trigger late)")
print(f"    Live pas tradé             : {not_traded_lv}")
print(f"    Meme cote, timing ~pareil  : {same_side_same_time}")
print(f"    Meme cote, timing DIFF     : {same_side_diff_time}")
print(f"    Cote DIFF live vs BT       : {diff_side}")
