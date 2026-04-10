"""
analyze_synthetic_filter.py
Pour chaque contrat BT-only, vérifie si le delta_binance seul
aurait triggeré en live (sans le filtre synthétique).
Quantifie gains perdus / pertes évitées par le filtre synthétique.
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
from chart_utils import load_contracts, precompute_vol, simulate_trade_log, TIMEFRAMES, VOL_LBS

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
    dt_fr = datetime(2026, mon, day, h2, mn2, 0, tzinfo=CEST)
    return dt_fr.astimezone(timezone.utc)


# ── 1. Live : ce qui a été tradé ─────────────────────────────────────────────
live_traded = set()
for tf in ['m5', 'm15', 'h1']:
    tf_dir = LIVE_DIR / tf
    if not tf_dir.exists(): continue
    for fn in sorted(tf_dir.glob("*.jsonl")):
        events = []
        for line in fn.open(encoding="utf-8", errors="replace"):
            try: events.append(json.loads(line.strip()))
            except: continue
        wo = next((e for e in events if e.get("event") == "window_open"), None)
        if wo is None: continue
        if pd.to_datetime(wo["ts"]).timestamp() < CUTOFF_TS: continue
        we = next((e for e in events if e.get("event") == "window_ended"), None)
        if we is None: continue
        shares = float(we.get("up_shares", 0)) + float(we.get("down_shares", 0))
        if shares > 1e-9:
            ce_utc = parse_ce_utc(str(fn), tf)
            if ce_utc: live_traded.add((ce_utc, tf))

print(f"Live trades: {len(live_traded)}")


# ── 2. BT fills ───────────────────────────────────────────────────────────────
cu.BASE_SIZE = BTC_SIZE
m5_ref = None
bt_fills = {}
for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_name = ['m5', 'm15', 'h1'][idx]
    cts = load_contracts([CSV_PATH], tf_floor, b1, b2, a1, a2)
    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS); m5_ref = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
    for c in cts:
        if c.get('open_ts', 0) < CUTOFF_TS: continue
        won, pnl, fills = simulate_trade_log(c, CFG)
        if not fills: continue
        ce_utc = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
        bt_fills[(ce_utc, tf_name)] = {'pnl': pnl, 'won': won, 'fills': fills}

print(f"BT fills: {len(bt_fills)}")

# Contrats BT-only
bt_only_keys = [(k, v) for k, v in bt_fills.items() if k not in live_traded]
print(f"BT-only: {len(bt_only_keys)}")


# ── 3. Pour chaque BT-only, check dans les JSONL ─────────────────────────────
# Est-ce que delta_binance seul aurait triggeré ?
results = []

for (ce_utc, tf), bt in bt_only_keys:
    # Trouve le fichier JSONL correspondant
    tf_dir = LIVE_DIR / tf
    target_file = None
    for fn in tf_dir.glob("*.jsonl"):
        fce = parse_ce_utc(str(fn), tf)
        if fce == ce_utc:
            target_file = fn
            break

    synth_blocked = False
    binance_would_trigger = False
    max_delta_b = 0.0
    max_delta_s = 0.0
    min_required = float('inf')
    reason_found = None

    if target_file:
        events = []
        for line in target_file.open(encoding="utf-8", errors="replace"):
            try: events.append(json.loads(line.strip()))
            except: continue

        snapshots = [e for e in events if e.get("event") == "dashboard_snapshot"]
        for snap in snapshots:
            db = abs(float(snap.get("delta_binance", 0)))
            ds = abs(float(snap.get("delta_synthetic", snap.get("delta_binance", 0))))
            req = float(snap.get("required_usd", 9999))
            reason = snap.get("reason", "")
            max_delta_b = max(max_delta_b, db)
            max_delta_s = max(max_delta_s, ds)
            min_required = min(min_required, req)

            # Delta binance aurait suffi mais delta synthétique non
            if db >= req and ds < req:
                synth_blocked = True
                binance_would_trigger = True
                reason_found = f"db={db:.1f}>=req={req:.1f} but ds={ds:.1f}<req"

        # Est-ce que delta_binance seul aurait triggeré à un moment ?
        if not synth_blocked:
            for snap in snapshots:
                db = abs(float(snap.get("delta_binance", 0)))
                req = float(snap.get("required_usd", 9999))
                if db >= req:
                    binance_would_trigger = True
                    break
    else:
        # Pas de JSONL = contrat pas monitoré du tout
        pass

    results.append({
        "ce_utc": ce_utc, "tf": tf,
        "bt_pnl": bt["pnl"], "bt_won": bt["won"],
        "has_jsonl": target_file is not None,
        "synth_blocked": synth_blocked,
        "binance_would_trigger": binance_would_trigger,
        "max_delta_b": round(max_delta_b, 2),
        "max_delta_s": round(max_delta_s, 2),
        "min_required": round(min_required, 2) if min_required < 9999 else None,
        "reason_found": reason_found,
    })


# ── 4. Synthèse ───────────────────────────────────────────────────────────────
no_jsonl      = [r for r in results if not r["has_jsonl"]]
synth_blocked = [r for r in results if r["synth_blocked"]]
no_snapshot   = [r for r in results if r["has_jsonl"] and not r["synth_blocked"]]

print(f"\n{'='*60}")
print(f"Pas de JSONL (contrat pas du tout monitoré) : {len(no_jsonl)}")
print(f"Filtre synthétique clairement bloquant      : {len(synth_blocked)}")
print(f"JSONL présent mais synth pas identifié      : {len(no_snapshot)}")

print(f"\n--- FILTRE SYNTHÉTIQUE BLOQUANT ({len(synth_blocked)} contrats) ---")
gains_perdus = sum(r["bt_pnl"] for r in synth_blocked if r["bt_won"])
pertes_evitees = sum(abs(r["bt_pnl"]) for r in synth_blocked if not r["bt_won"])
print(f"Gains perdus (BT gagnant)  : ${gains_perdus:.2f}")
print(f"Pertes évitées (BT perdant): ${pertes_evitees:.2f}")
print(f"Net (gains - pertes)       : ${gains_perdus - pertes_evitees:.2f}")
print()
hdr = f"{'CE UTC':17s} {'TF':3s} {'BT PnL':7s} {'W':1s} {'MaxDB':7s} {'MaxDS':7s} {'MinReq':7s}"
print(hdr); print("-"*len(hdr))
for r in sorted(synth_blocked, key=lambda x: x["ce_utc"]):
    w = "W" if r["bt_won"] else "L"
    print(f"{r['ce_utc'].strftime('%m-%d %H:%M UTC'):17s} {r['tf']:3s} "
          f"${r['bt_pnl']:6.2f} {w} {r['max_delta_b']:7.2f} {r['max_delta_s']:7.2f} "
          f"{r['min_required'] or 0:7.2f}")

print(f"\n--- JSONL PRÉSENT MAIS SYNTH NON IDENTIFIÉ ({len(no_snapshot)} contrats) ---")
for r in sorted(no_snapshot, key=lambda x: x["ce_utc"]):
    w = "W" if r["bt_won"] else "L"
    print(f"{r['ce_utc'].strftime('%m-%d %H:%M UTC'):17s} {r['tf']:3s} "
          f"${r['bt_pnl']:6.2f} {w}  max_db={r['max_delta_b']}  max_ds={r['max_delta_s']}  min_req={r['min_required']}")

print(f"\n--- PAS DE JSONL ({len(no_jsonl)} contrats) ---")
for r in sorted(no_jsonl, key=lambda x: x["ce_utc"]):
    w = "W" if r["bt_won"] else "L"
    print(f"{r['ce_utc'].strftime('%m-%d %H:%M UTC'):17s} {r['tf']:3s} ${r['bt_pnl']:6.2f} {w}")
