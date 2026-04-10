"""
trace_contract.py
Trace détaillée d'un contrat spécifique en BT : chaque tick, le signal, les ordres.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import load_contracts, precompute_vol, VOL_LBS

BASE     = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"
CSV_PATH = str(BASE / "BTC.csv")

# Contrat cible : ce = 2026-04-06 23:15 UTC
TARGET_CE = datetime(2026, 4, 7, 15, 15, 0, tzinfo=timezone.utc)
TARGET_TF = "15min"  # "5min", "15min", "1h"

CFG = dict(
    curve='linear', slope=0.2, intercept=0.0,
    eq_cap=0.60, max_orders=1,
    vol_lb_h=2.0, vol_thresh=0.5, vol_type='trend',
)

TF_MAP = {
    "5min":  ("m5_up_bid",  "m5_down_bid",  "m5_up_ask",  "m5_down_ask"),
    "15min": ("m15_up_bid", "m15_down_bid", "m15_up_ask", "m15_down_ask"),
    "1h":    ("h1_up_bid",  "h1_down_bid",  "h1_up_ask",  "h1_down_ask"),
}
tf_args = TF_MAP[TARGET_TF]

# BTC VRS config
CFG = dict(curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
           vol_lb_h=None, vol_thresh=None,
           vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
           vrs_g_min=0.5, vrs_g_max=1.5,
           max_losses_cb=None, max_orders=2)

import chart_utils as cu
cu.BASE_SIZE = 100.0

# Load m5 first for VRS ref
print("Chargement CSV...")
m5_ref = load_contracts([CSV_PATH], "5min", "m5_up_bid", "m5_down_bid", "m5_up_ask", "m5_down_ask")
precompute_vol(m5_ref, VOL_LBS)
contracts = load_contracts([CSV_PATH], TARGET_TF, *tf_args)
precompute_vol(contracts, VOL_LBS, ref_contracts=(m5_ref if TARGET_TF != "5min" else None))

# Cherche le contrat
c = None
for contract in contracts:
    ce = contract["ce"].to_pydatetime().replace(tzinfo=timezone.utc)
    if ce == TARGET_CE:
        c = contract
        break

if c is None:
    print(f"Contrat {TARGET_CE} introuvable dans le CSV.")
    sys.exit(1)

op = float(c["op"])
vol_trend = c.get("vol_2h_trend", 0.0)
print(f"\nContrat : {TARGET_CE}")
print(f"Open (BT op)     : {op:.4f}")
print(f"vol_2h_trend     : {vol_trend:.4f}  (seuil >= 0.50 requis)")
print(f"Filtre vol passé : {'OUI' if vol_trend >= 0.5 else 'NON — contrat exclu'}")
print()

if vol_trend < 0.5:
    # Montre quand même les ticks pour comprendre le marché
    pass

# VRS : calcul du g initial
if CFG.get("vrs_enabled"):
    _vrs_base = float(CFG["vrs_base"])
    _vrs_lb   = CFG["vrs_lb"]
    _vrs_g_min = float(CFG["vrs_g_min"])
    _vrs_g_max = float(CFG["vrs_g_max"])
    _vol = c.get(f"vol_{_vrs_lb:g}h_range", 0.0)
    _ratio = _vrs_base / _vol if _vol > 1e-6 else _vrs_g_max
    _vrs_g = float(np.clip(_ratio, _vrs_g_min, _vrs_g_max))
    print(f"VRS : vol_1h_range={_vol:.2f}  g={_vrs_g:.4f}  slope_eff={CFG['slope']*_vrs_g:.4f}")
else:
    _vrs_g = 1.0

# Trace tick par tick
spots    = c["spot"]
ts_arr   = c["ts"]
up_bid   = c["up_bid"]
down_bid = c["down_bid"]
ce_ns    = np.datetime64(c["ce"])

print(f"{'Tick':>5} {'Timestamp':22} {'Remain':>7} {'Spot':>10} {'Delta':>8} {'Thresh':>7} {'UBid':>6} {'DBid':>6} {'Signal'}")
print("-" * 95)

activated_side = None
active_order   = None
active_size    = None
pending_place  = None
pending_size   = None
pending_cancel = False
fill_count     = 0
MAX_ORDERS     = int(CFG.get("max_orders", 1))
BASE_SIZE      = 100.0
cap            = CFG["eq_cap"]

for j in range(len(spots)):
    s      = float(spots[j])
    remain = float((ce_ns - ts_arr[j]) / np.timedelta64(1, 's'))
    if remain <= 0 or fill_count >= MAX_ORDERS:
        break

    ub = round(float(up_bid[j]), 2)
    db = round(float(down_bid[j]), 2)
    slope_eff = CFG["slope"] * _vrs_g
    thresh = slope_eff * remain + CFG.get("intercept", 0.0)
    delta_up   = s - op
    delta_down = op - s

    # Logique de fill / cancel (reproduit simulate)
    filled_now = False
    cancelled_now = False

    if pending_cancel and active_order is not None:
        if activated_side:
            cb2 = ub if activated_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fill_count += 1; filled_now = True
        active_order = active_size = None; pending_cancel = False; cancelled_now = True

    if pending_place is not None:
        active_order = pending_place; active_size = pending_size
        pending_place = pending_size = None

    if active_order is not None and activated_side and not filled_now:
        cb2 = ub if activated_side == 'UP' else db
        if cb2 <= round(active_order - 0.01, 2):
            fill_count += 1; filled_now = True
            active_order = active_size = None

    # Déactivation
    deactivated = False
    if activated_side and not filled_now:
        buf = (s - op) if activated_side == 'UP' else (op - s)
        if buf < thresh:
            if active_order is not None:
                pending_cancel = True
            pending_place = pending_size = None
            activated_side = None
            deactivated = True

    # Activation
    new_activation = None
    if not activated_side and not deactivated:
        if   delta_up   >= thresh: activated_side = 'UP';   new_activation = 'UP'
        elif delta_down >= thresh: activated_side = 'DOWN'; new_activation = 'DOWN'

    # Placement d'ordre
    placed_price = None
    if activated_side and not filled_now:
        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = BASE_SIZE; placed_price = proposed
            elif active_order is not None and proposed > active_order:
                pending_cancel = True; pending_place = proposed; pending_size = BASE_SIZE; placed_price = proposed
            elif pending_place is not None and proposed > pending_place:
                pending_place = proposed; pending_size = BASE_SIZE; placed_price = proposed

    # Affichage
    status_parts = []
    if new_activation:    status_parts.append(f"TRIGGER {new_activation}")
    if deactivated:       status_parts.append("DEACTIVATE")
    if cancelled_now:     status_parts.append("CANCEL")
    if placed_price:      status_parts.append(f"ORDER@{placed_price:.2f}")
    if filled_now:        status_parts.append(f"FILL@{active_order or '?'}")
    if active_order and not filled_now and not placed_price:
        status_parts.append(f"ord={active_order:.2f}")
    if pending_cancel:    status_parts.append("pend_cancel")

    delta_str = f"{delta_up:+.4f}" if delta_up >= 0 else f"{delta_down:+.4f}dn"
    ts_str = str(ts_arr[j])[:22]

    print(f"{j:5d} {ts_str:22} {remain:7.2f} {s:10.4f} {delta_str:>8} {thresh:7.4f} {ub:6.2f} {db:6.2f}  {' | '.join(status_parts)}")

    if fill_count >= MAX_ORDERS:
        print(f"\n  -> FILL total : {fill_count} ordre(s)")
        break

# Expiry fill : ordre en attente au dernier tick, contrat résout dans notre direction
if fill_count < MAX_ORDERS and activated_side and (pending_place is not None or active_order is not None):
    best_order = pending_place if pending_place is not None else active_order
    best_size  = pending_size  if pending_place is not None else active_size
    # Résolution (fallback spot)
    _won_up = float(spots[-1]) > op
    if best_order and best_size and (activated_side == 'UP') == _won_up:
        fill_count += 1
        print(f"\n  -> EXPIRY FILL : ordre {activated_side} @{best_order:.2f} fill à l'expiry (contrat résout {activated_side})")

if fill_count == 0:
    print(f"\n  -> Aucun fill BT sur ce contrat.")
    print(f"     Raison : {'vol filter (vol_2h_trend=' + str(round(vol_trend,3)) + ' < 0.50)' if vol_trend < 0.5 else 'signal jamais atteint ou bid jamais retracé'}")

# Final
print(f"\nClose spot (dernier tick) : {float(spots[-1]):.4f}")
print(f"Direction réelle          : {'UP' if float(spots[-1]) > op else 'DOWN'} (close {'>' if float(spots[-1]) > op else '<'} open {op:.4f})")
