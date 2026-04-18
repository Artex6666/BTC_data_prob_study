"""
analyze_synthetic_delay.py
Pour chaque contrat live, cherche dans les snapshots si delta_binance
aurait suffi à trigger AVANT que min(delta_b, delta_s) le fasse.
Quantifie le retard en secondes et l'impact sur le fill price.
"""
import sys, json, re, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

CEST      = timezone(timedelta(hours=2))
MONTH_MAP = {'janv':1,'fevr':2,'mars':3,'avr':4,'mai':5,'juin':6,
             'juil':7,'aout':8,'sept':9,'oct':10,'nov':11,'dec':12}

BASE     = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"
LIVE_DIR = BASE / "slope10int0VRS" / "btc"
CUTOFF   = pd.Timestamp("2026-04-05T21:45:00", tz="UTC")
CUTOFF_TS = CUTOFF.timestamp()

results = []

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

        we  = next((e for e in events if e.get("event") in ("window_ended", "window_settled")), None)
        trg = next((e for e in events if e.get("event") == "safety_trigger"), None)
        snaps = [e for e in events if e.get("event") == "dashboard_snapshot"]

        if not snaps: continue

        # Pour chaque snapshot : est-ce que delta_b aurait triggeré mais pas min() ?
        synth_delayed = False
        delay_seconds = 0.0
        first_binance_trigger_tte = None
        actual_trigger_tte = float(trg["time_to_expiry_s"]) if trg else None

        for snap in snaps:
            db  = abs(float(snap.get("delta_binance", 0)))
            ds  = abs(float(snap.get("delta_synthetic", snap.get("delta_binance", 0))))
            req = float(snap.get("required_usd", 9999))
            tte = float(snap.get("tte_s", 0))
            effective = min(db, ds)

            if db >= req and first_binance_trigger_tte is None:
                first_binance_trigger_tte = tte

            if db >= req and effective < req:
                synth_delayed = True

        if first_binance_trigger_tte is not None and actual_trigger_tte is not None:
            delay_seconds = first_binance_trigger_tte - actual_trigger_tte

        # Fill price live
        up_s = float(we.get("up_shares", 0)) if we else 0
        dn_s = float(we.get("down_shares", 0)) if we else 0
        up_c = float(we.get("up_cost", 0)) if we else 0
        dn_c = float(we.get("down_cost", 0)) if we else 0
        shares = up_s + dn_s
        cost   = up_c + dn_c
        traded = shares > 1e-9
        avg_fill = cost / shares if shares > 1e-9 else None

        fill_events = [e for e in events if e.get("event") == "fill"]
        fill_prices = [float(e["price"]) for e in fill_events if "price" in e]
        if fill_prices: avg_fill = float(np.mean(fill_prices))

        # PnL
        start_sp = float(we.get("start_spot", 0)) if we else 0
        end_sp   = float(we.get("end_spot", 0)) if we else 0
        ts_end   = pd.to_datetime(we["ts"]) if we else None
        LOSS_TS  = pd.Timestamp("2026-04-06T01:55:00", tz="UTC")
        is_real_loss = (ts_end and abs((ts_end - LOSS_TS).total_seconds()) < 5
                        and tf == 'm5' and up_s > 300)
        if is_real_loss: up_wins = False
        elif start_sp > 0 and end_sp > 0: up_wins = end_sp > start_sp
        else: up_wins = bool(we.get("epnl_up_wins", 1)) if we else True
        pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c) if traded else 0

        results.append({
            "file": os.path.basename(fn),
            "tf": tf,
            "traded": traded,
            "triggered": trg is not None,
            "synth_delayed": synth_delayed,
            "delay_s": round(delay_seconds, 1),
            "first_b_trigger_tte": first_binance_trigger_tte,
            "actual_trigger_tte": actual_trigger_tte,
            "avg_fill": round(avg_fill, 4) if avg_fill else None,
            "pnl": round(pnl, 2),
            "won": pnl > 0 if traded else None,
        })

# ── Synthèse ──────────────────────────────────────────────────────────────────
delayed     = [r for r in results if r["synth_delayed"]]
not_delayed = [r for r in results if r["triggered"] and not r["synth_delayed"]]
no_trigger  = [r for r in results if not r["triggered"]]

print(f"Total contrats monitorés : {len(results)}")
print(f"  Triggered (no delay)   : {len(not_delayed)}")
print(f"  Triggered (synth delay): {len(delayed)}")
print(f"  Pas de trigger         : {len(no_trigger)}")

print(f"\n{'='*65}")
print(f"CONTRATS OÙ LE SYNTHÉTIQUE A RETARDÉ LE TRIGGER ({len(delayed)})")
print(f"{'='*65}")
hdr = f"{'File':38s} {'TF':3s} {'Delay':6s} {'Fill':6s} {'PnL':8s} {'W'}"
print(hdr); print("-"*len(hdr))
for r in sorted(delayed, key=lambda x: x["delay_s"], reverse=True):
    w = "W" if r["won"] else ("L" if r["won"] is False else "-")
    fill_s = f"{r['avg_fill']:.4f}" if r["avg_fill"] else "  N/A"
    print(f"{r['file']:38s} {r['tf']:3s} {r['delay_s']:6.1f}s {fill_s:6s} "
          f"${r['pnl']:7.2f} {w}")

if delayed:
    traded_delayed = [r for r in delayed if r["traded"]]
    not_traded_delayed = [r for r in delayed if not r["traded"]]
    print(f"\nAvec trade  : {len(traded_delayed)} contrats")
    print(f"Sans trade  : {len(not_traded_delayed)} contrats (trigger trop tard pour fill)")
    if not_traded_delayed:
        print("  → Ces contrats BT aurait potentiellement tradé:")
        # On ne connait pas le PnL BT ici, juste qu'il y avait un signal
        for r in not_traded_delayed:
            print(f"    {r['file']} delay={r['delay_s']}s actual_trigger={r['actual_trigger_tte']}s")

    if traded_delayed:
        fills_delayed = [r["avg_fill"] for r in traded_delayed if r["avg_fill"]]
        pnls_delayed  = [r["pnl"] for r in traded_delayed]
        fills_normal  = [r["avg_fill"] for r in not_delayed if r["avg_fill"]]
        print(f"\nFill avg (avec delay) : {np.mean(fills_delayed):.4f}")
        print(f"Fill avg (sans delay) : {np.mean(fills_normal):.4f}")
        print(f"Impact fill           : {np.mean(fills_delayed) - np.mean(fills_normal):+.4f}")
        print(f"PnL total delayed     : ${sum(pnls_delayed):.2f}")
