"""
compare_trigger_timing.py
Compare le moment du trigger BT (remain en secondes) vs live (time_to_expiry_s)
pour chaque contrat. Si live trigger nettement plus tard que BT → synthétique.
"""
import sys, json, re, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_utils as cu
from chart_utils import load_contracts, precompute_vol, VOL_LBS, TIMEFRAMES

CEST      = timezone(timedelta(hours=2))
MONTH_MAP = {'janv':1,'fevr':2,'mars':3,'avr':4,'mai':5,'juin':6,
             'juil':7,'aout':8,'sept':9,'oct':10,'nov':11,'dec':12}

BASE     = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"
LIVE_DIR = BASE / "slope10int0VRS" / "btc"
CSV_PATH = str(BASE / "BTC.csv")
CUTOFF   = pd.Timestamp("2026-04-05T21:45:00", tz="UTC")
CUTOFF_TS = CUTOFF.timestamp()

CFG = dict(curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
           vol_lb_h=None, vol_thresh=None,
           vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
           vrs_g_min=0.5, vrs_g_max=1.5,
           max_losses_cb=None, max_orders=2)
BTC_SIZE = 100.0

def parse_ce_utc(fname, tf):
    m = re.match(r'(\d+)_(\w+)_(\d+)h(\d+)-(\d+)h(\d+)\.jsonl', os.path.basename(fname))
    if not m: return None
    day, mon_str = int(m.group(1)), m.group(2)
    h2, mn2 = int(m.group(5)), int(m.group(6))
    mon = MONTH_MAP.get(mon_str)
    if not mon: return None
    return datetime(2026, mon, day, h2, mn2, 0, tzinfo=CEST).astimezone(timezone.utc)


# ── BT : extrait le premier trigger (remain) pour chaque contrat ──────────────
def bt_first_trigger(c, cfg):
    """Retourne remain_s au premier trigger BT qui mène à un fill."""
    import numpy as np
    cap = cfg['eq_cap']
    ce_ns = np.datetime64(c['ce'])
    spots = c['spot']; ts_arr = c['ts']; op = c['op']
    up_bid = c['up_bid']; down_bid = c['down_bid']
    activated_side = None
    active_order = pending_place = pending_size = None
    pending_cancel = False; fill_count = 0
    max_orders = int(cfg.get('max_orders', 2))
    first_trigger_remain = None

    # VRS
    _vol = c.get('vol_1.0h_range', 0.0)
    _ratio = float(cfg['vrs_base']) / _vol if _vol > 1e-6 else float(cfg['vrs_g_max'])
    _g = float(np.clip(_ratio, cfg['vrs_g_min'], cfg['vrs_g_max']))

    for j in range(len(spots)):
        s = float(spots[j])
        remain = float((ce_ns - ts_arr[j]) / np.timedelta64(1, 's'))
        if remain <= 0 or fill_count >= max_orders: break
        ub = round(float(up_bid[j]), 2); db = round(float(down_bid[j]), 2)
        thresh = cfg['slope'] * _g * remain

        if pending_cancel and active_order is not None:
            cb2 = ub if activated_side == 'UP' else db
            if activated_side and cb2 <= round(active_order - 0.01, 2):
                fill_count += 1
            active_order = None; pending_cancel = False

        if pending_place is not None:
            active_order = pending_place; pending_place = pending_size = None

        if active_order is not None and activated_side:
            cb2 = ub if activated_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fill_count += 1; active_order = None
                if fill_count >= max_orders: break

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order: pending_cancel = True
                pending_place = pending_size = None; activated_side = None; continue

        if not activated_side:
            if   s - op >= thresh: activated_side = 'UP'
            elif op - s >= thresh: activated_side = 'DOWN'
            if activated_side and first_trigger_remain is None:
                first_trigger_remain = remain

        if activated_side:
            bid = ub if activated_side == 'UP' else db
            if bid >= cap:
                proposed = min(bid, 0.99)
                if active_order is None and pending_place is None:
                    pending_place = proposed; pending_size = BTC_SIZE
                elif active_order is not None and proposed > active_order:
                    pending_cancel = True; pending_place = proposed; pending_size = BTC_SIZE

    return first_trigger_remain, fill_count > 0


# ── BT simulation ─────────────────────────────────────────────────────────────
print("Chargement BT...")
cu.BASE_SIZE = BTC_SIZE
m5_ref = None
bt_trigger = {}  # (ce_utc, tf) -> first_trigger_remain

for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_name = ['m5', 'm15', 'h1'][idx]
    cts = load_contracts([CSV_PATH], tf_floor, b1, b2, a1, a2)
    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS); m5_ref = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
    for c in cts:
        if c.get('open_ts', 0) < CUTOFF_TS: continue
        tr, filled = bt_first_trigger(c, CFG)
        if tr is not None:
            ce_utc = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
            bt_trigger[(ce_utc, tf_name)] = {'trigger_remain': tr, 'filled': filled}

print(f"BT contrats avec trigger: {len(bt_trigger)}")


# ── Live : extrait safety_trigger time_to_expiry_s ────────────────────────────
live_trigger = {}  # (ce_utc, tf) -> tte_s (None si pas de trigger)

for tf in ['m5', 'm15', 'h1']:
    tf_dir = LIVE_DIR / tf
    if not tf_dir.exists(): continue
    for fn in sorted(tf_dir.glob("*.jsonl")):
        events = []
        for line in fn.open(encoding="utf-8", errors="replace"):
            try: events.append(json.loads(line.strip()))
            except: continue
        wo = next((e for e in events if e.get("event") == "window_open"), None)
        if wo is None or pd.to_datetime(wo["ts"]).timestamp() < CUTOFF_TS: continue
        trg = next((e for e in events if e.get("event") == "safety_trigger"), None)
        ce_utc = parse_ce_utc(str(fn), tf)
        if ce_utc is None: continue
        live_trigger[(ce_utc, tf)] = {
            'tte': float(trg["time_to_expiry_s"]) if trg else None,
            'triggered': trg is not None,
        }

print(f"Live contrats monitorés : {len(live_trigger)}")


# ── Comparaison ───────────────────────────────────────────────────────────────
rows = []
for key, bt in sorted(bt_trigger.items()):
    live = live_trigger.get(key)
    if live is None: continue  # pas monitoré live

    bt_tte  = bt['trigger_remain']
    live_tte = live['tte']

    # Gap : combien de secondes plus tard live a triggeré vs BT
    if live_tte is not None:
        gap = bt_tte - live_tte  # positif = live a triggeré plus tard
    else:
        gap = bt_tte  # live n'a jamais triggeré

    rows.append({
        "key": key,
        "bt_tte": round(bt_tte, 2),
        "live_tte": round(live_tte, 2) if live_tte else None,
        "live_triggered": live["triggered"],
        "bt_filled": bt["filled"],
        "gap": round(gap, 2),
        "synth_suspect": gap > 0.5,  # >500ms de retard → probablement synthétique
    })

# Tri par gap décroissant
rows.sort(key=lambda x: x["gap"], reverse=True)

print(f"\n{'='*70}")
print(f"COMPARAISON TIMING TRIGGER BT vs LIVE ({len(rows)} contrats communs)")
print(f"{'='*70}")
hdr = f"{'CE UTC':17s} {'TF':3s} {'BT tte':7s} {'Live tte':8s} {'Gap':7s} {'Suspect'}"
print(hdr); print("-"*len(hdr))
for r in rows:
    live_s = f"{r['live_tte']:7.2f}s" if r['live_tte'] else "  NO TRG"
    susp = "*** SYNTH ***" if r["synth_suspect"] else ""
    print(f"{r['key'][0].strftime('%m-%d %H:%M UTC'):17s} {r['key'][1]:3s} "
          f"{r['bt_tte']:7.2f}s {live_s:8s} {r['gap']:+7.2f}s  {susp}")

suspect = [r for r in rows if r["synth_suspect"]]
no_live_trigger = [r for r in rows if not r["live_triggered"]]
on_time = [r for r in rows if not r["synth_suspect"] and r["live_triggered"]]
# Absence pure : live ne trigger pas, BT a triggeré avec assez de temps (>0.5s restant)
no_trg_suspect = [r for r in no_live_trigger if r["gap"] > 0.5]
# Retard seul : les deux triggerent mais live >0.5s plus tard
delay_only = [r for r in suspect if r["live_triggered"]]

print(f"\n{'='*50}")
print(f"Gap > 0.5s (suspect synthétique) : {len(suspect)}")
print(f"  dont retard seul (les 2 ont triggeré) : {len(delay_only)}")
print(f"  dont absence totale (live no trigger)  : {len(no_trg_suspect)}")
print(f"Live n'a pas triggeré du tout    : {len(no_live_trigger)}")
print(f"Timing OK (<= 0.5s d'écart)      : {len(on_time)}")

if suspect:
    gaps = [r["gap"] for r in suspect]
    print(f"\nSur les {len(suspect)} suspects :")
    print(f"  Gap moyen  : {np.mean(gaps):.2f}s")
    print(f"  Gap médian : {np.median(gaps):.2f}s")
    print(f"  Gap max    : {np.max(gaps):.2f}s")

# ── PnL des suspects ──────────────────────────────────────────────────────────
# Charge les fills live et BT pour les suspects
from chart_utils import simulate_trade_log

print(f"\n{'='*70}")
print(f"PNL DETAIL — 33 SUSPECTS (retard synthétique)")
print(f"{'='*70}")

# Live fills pour les suspects
def get_live_pnl(ce_utc, tf, live_dir, month_map):
    tf_dir = live_dir / tf
    for fn in tf_dir.glob("*.jsonl"):
        fce = parse_ce_utc(str(fn), tf)
        if fce != ce_utc: continue
        events = []
        for line in fn.open(encoding="utf-8", errors="replace"):
            try: events.append(json.loads(line.strip()))
            except: continue
        we = next((e for e in events if e.get("event") in ("window_ended", "window_settled")), None)
        if we is None: return None, None, None
        up_s = float(we.get("up_shares", 0)); dn_s = float(we.get("down_shares", 0))
        up_c = float(we.get("up_cost", 0));   dn_c = float(we.get("down_cost", 0))
        shares = up_s + dn_s
        if shares < 1e-9: return None, None, None
        fill_evts = [e for e in events if e.get("event") == "fill"]
        fills = [float(e["price"]) for e in fill_evts if "price" in e]
        avg_fill = float(np.mean(fills)) if fills else (up_c + dn_c) / shares
        start_sp = float(we.get("start_spot", 0)); end_sp = float(we.get("end_spot", 0))
        up_wins = end_sp > start_sp if start_sp > 0 and end_sp > 0 else bool(we.get("epnl_up_wins", 1))
        pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c)
        return round(pnl, 2), round(avg_fill, 4), shares > 1e-9
    return None, None, None

# BT fills pour les suspects — charge les contrats une fois
print("Chargement BT pour PnL suspects...")
bt_pnl_map = {}
m5_ref2 = None
for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_name = ['m5', 'm15', 'h1'][idx]
    cts = load_contracts([CSV_PATH], tf_floor, b1, b2, a1, a2)
    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS); m5_ref2 = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref2)
    for c in cts:
        if c.get('open_ts', 0) < CUTOFF_TS: continue
        won, pnl, fills = simulate_trade_log(c, CFG)
        if fills is None: continue
        ce_utc = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
        avg_fill = float(np.mean([f['price'] for f in fills])) if fills else None
        bt_pnl_map[(ce_utc, tf_name)] = {'pnl': pnl, 'avg_fill': avg_fill, 'filled': pnl is not None}

hdr = f"{'CE UTC':17s} {'TF':3s} {'Gap':6s} {'Live fill':9s} {'BT fill':7s} {'Live PnL':8s} {'BT PnL':7s} {'Delta':8s}"
print(hdr); print("-"*len(hdr))

total_live_pnl = 0; total_bt_pnl = 0; count = 0
gains_perdus = 0; pertes_evitees = 0

for r in sorted(suspect, key=lambda x: x["key"][0]):
    key = r["key"]
    live_pnl, live_fill, live_traded = get_live_pnl(key[0], key[1], LIVE_DIR, MONTH_MAP)
    bt = bt_pnl_map.get(key, {})
    bt_pnl  = bt.get('pnl')
    bt_fill = bt.get('avg_fill')

    live_pnl_s = f"${live_pnl:7.2f}" if live_pnl is not None else "  NO FILL"
    bt_pnl_s   = f"${bt_pnl:6.2f}"  if bt_pnl  is not None else "  NO FILL"
    live_fill_s = f"{live_fill:.4f}" if live_fill else "  N/A "
    bt_fill_s   = f"{bt_fill:.4f}"   if bt_fill  else " N/A "
    delta = (live_pnl or 0) - (bt_pnl or 0) if (live_pnl is not None and bt_pnl is not None) else None
    delta_s = f"{delta:+8.2f}" if delta is not None else "     N/A"

    print(f"{key[0].strftime('%m-%d %H:%M UTC'):17s} {key[1]:3s} {r['gap']:6.2f}s "
          f"{live_fill_s:9s} {bt_fill_s:7s} {live_pnl_s:8s} {bt_pnl_s:7s} {delta_s}")

    if live_pnl is not None: total_live_pnl += live_pnl
    if bt_pnl is not None:   total_bt_pnl   += bt_pnl
    if live_pnl is not None and bt_pnl is not None:
        count += 1
        if bt_pnl > 0 and live_pnl is None: gains_perdus += bt_pnl
        if bt_pnl < 0 and live_pnl is None: pertes_evitees += abs(bt_pnl)

print()
print(f"Total live PnL (suspects, retard+absence) : ${total_live_pnl:.2f}")
print(f"Total BT PnL   (suspects, retard+absence) : ${total_bt_pnl:.2f}")
print(f"Delta live - BT                           : ${total_live_pnl - total_bt_pnl:.2f}")

# ── PnL des 55 NO LIVE TRIGGER (tous, pas seulement les gap>0.5s) ────────────
print(f"\n{'='*70}")
print(f"PNL DETAIL — {len(no_live_trigger)} CONTRATS SANS TRIGGER LIVE")
print(f"(BT a eu un signal, live n'a jamais déclenché le safety_trigger)")
print(f"{'='*70}")
hdr2 = f"{'CE UTC':17s} {'TF':3s} {'BT tte':6s} {'Gap':6s} {'BT fill':7s} {'BT PnL':8s} {'Suspect'}"
print(hdr2); print("-"*len(hdr2))

total_bt_notrig = 0; gains_perdus2 = 0; pertes_evitees2 = 0; filled_bt_notrig = 0
for r in sorted(no_live_trigger, key=lambda x: x["key"][0]):
    key = r["key"]
    bt = bt_pnl_map.get(key, {})
    bt_pnl  = bt.get('pnl')
    bt_fill = bt.get('avg_fill')
    bt_fill_s = f"{bt_fill:.4f}" if bt_fill else "  N/A "
    bt_pnl_s  = f"${bt_pnl:7.2f}" if bt_pnl is not None else "  NO FILL"
    susp = "SYNTH?" if r["gap"] > 0.5 else ""
    print(f"{key[0].strftime('%m-%d %H:%M UTC'):17s} {key[1]:3s} {r['bt_tte']:6.2f}s "
          f"{r['gap']:+6.2f}s {bt_fill_s:7s} {bt_pnl_s:8s}  {susp}")
    if bt_pnl is not None:
        total_bt_notrig += bt_pnl
        filled_bt_notrig += 1
        if bt_pnl > 0: gains_perdus2 += bt_pnl
        else:           pertes_evitees2 += abs(bt_pnl)

print()
print(f"BT fills sur les {len(no_live_trigger)} sans-trigger live : {filled_bt_notrig}")
print(f"Total BT PnL (si live avait aussi tradé)  : ${total_bt_notrig:.2f}")
print(f"  dont gains manqués (BT positif)         : +${gains_perdus2:.2f}")
print(f"  dont pertes évitées (BT négatif)        : +${pertes_evitees2:.2f}  (évitées par live)")
print(f"  Net impact filtre synthétique (absence) : ${gains_perdus2 - pertes_evitees2:.2f}")
