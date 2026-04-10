"""
compare_fill_prices_btc.py
Même analyse que compare_fill_prices.py mais pour BTC VRS.
Compare fill prices live vs BT sur les contrats communs, et liste
les BT-only / live-only avec leurs PnL.
"""
import sys, json, re, os
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = dict(curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
           vol_lb_h=None, vol_thresh=None,
           vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
           vrs_g_min=0.5, vrs_g_max=1.5,
           max_losses_cb=None, max_orders=2)
BTC_SIZE = 100.0

def parse_ce_utc(fname, tf):
    """Extrait le timestamp de fin de contrat depuis le nom de fichier."""
    if tf == 'm5':
        m = re.match(r'(\d+)_(\w+)_(\d+)h(\d+)-(\d+)h(\d+)\.jsonl', os.path.basename(fname))
        if not m: return None
        day, mon_str = int(m.group(1)), m.group(2)
        h2, mn2 = int(m.group(5)), int(m.group(6))
        mon = MONTH_MAP.get(mon_str)
        if not mon: return None
        # h2:mn2 peut dépasser minuit -> gère le changement de jour
        dt_fr = datetime(2026, mon, day, h2, mn2, 0, tzinfo=CEST)
        return dt_fr.astimezone(timezone.utc)
    elif tf == 'm15':
        m = re.match(r'(\d+)_(\w+)_(\d+)h(\d+)-(\d+)h(\d+)\.jsonl', os.path.basename(fname))
        if not m: return None
        day, mon_str = int(m.group(1)), m.group(2)
        h2, mn2 = int(m.group(5)), int(m.group(6))
        mon = MONTH_MAP.get(mon_str)
        if not mon: return None
        dt_fr = datetime(2026, mon, day, h2, mn2, 0, tzinfo=CEST)
        return dt_fr.astimezone(timezone.utc)
    elif tf == 'h1':
        m = re.match(r'(\d+)_(\w+)_(\d+)h(\d+)-(\d+)h(\d+)\.jsonl', os.path.basename(fname))
        if not m: return None
        day, mon_str = int(m.group(1)), m.group(2)
        h2, mn2 = int(m.group(5)), int(m.group(6))
        mon = MONTH_MAP.get(mon_str)
        if not mon: return None
        dt_fr = datetime(2026, mon, day, h2, mn2, 0, tzinfo=CEST)
        return dt_fr.astimezone(timezone.utc)
    return None


# ── 1. Live fills ─────────────────────────────────────────────────────────────
live_fills = {}  # (ce_utc, tf) -> dict

for tf in ['m5', 'm15', 'h1']:
    tf_dir = LIVE_DIR / tf
    if not tf_dir.exists():
        continue
    for fn in sorted(tf_dir.glob("*.jsonl")):
        events = []
        for line in fn.open(encoding="utf-8", errors="replace"):
            try: events.append(json.loads(line.strip()))
            except: continue

        # Filtre cutoff depuis window_open
        wo = next((e for e in events if e.get("event") == "window_open"), None)
        if wo is None: continue
        ts_open = pd.to_datetime(wo["ts"])
        if ts_open.timestamp() < CUTOFF_TS: continue

        we = next((e for e in events if e.get("event") == "window_ended"), None)
        if we is None: continue

        up_s = float(we.get("up_shares", 0))
        dn_s = float(we.get("down_shares", 0))
        up_c = float(we.get("up_cost", 0))
        dn_c = float(we.get("down_cost", 0))
        shares = up_s + dn_s
        cost   = up_c + dn_c
        if shares < 1e-9: continue  # pas de fill

        # Fill prices depuis events fill
        fill_events = [e for e in events if e.get("event") == "fill"]
        fill_prices = [float(e["price"]) for e in fill_events if "price" in e]
        avg_fill = float(np.mean(fill_prices)) if fill_prices else (cost / shares)

        # Direction / PnL
        start_sp = float(we.get("start_spot", 0))
        end_sp   = float(we.get("end_spot", 0))
        # Exception connue: 01:55 UTC Apr6, m5 up_shares > 300 → DOWN gagne
        ts_end = pd.to_datetime(we["ts"])
        LOSS_TS = pd.Timestamp("2026-04-06T01:55:00", tz="UTC")
        is_real_loss = (abs((ts_end - LOSS_TS).total_seconds()) < 5 and tf == 'm5' and up_s > 300)
        if is_real_loss:
            up_wins = False
        elif start_sp > 0 and end_sp > 0:
            up_wins = end_sp > start_sp
        else:
            up_wins = bool(we.get("epnl_up_wins", 1))

        pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c)
        side = "UP" if up_s >= dn_s else "DOWN"

        ce_utc = parse_ce_utc(str(fn), tf)
        if ce_utc is None: continue
        key = (ce_utc, tf)

        live_fills[key] = {
            "side": side,
            "avg_fill": round(avg_fill, 4),
            "pnl": round(pnl, 2),
            "won": pnl > 0,
            "shares": round(shares, 2),
            "cost": round(cost, 2),
            "tf": tf,
            "file": os.path.basename(fn),
            "real_loss": is_real_loss,
        }

print(f"Live fills apres cutoff : {len(live_fills)}")


# ── 2. BT simulation ──────────────────────────────────────────────────────────
print("Chargement CSV BTC...")
m5_ref = None
bt_fills = {}  # (ce_utc, tf) -> dict

orig_size = cu.BASE_SIZE
cu.BASE_SIZE = BTC_SIZE

for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
    tf_name = ['m5', 'm15', 'h1'][idx]
    cts = load_contracts([CSV_PATH], tf_floor, b1, b2, a1, a2)
    if tf_floor == '5min':
        precompute_vol(cts, VOL_LBS)
        m5_ref = cts
    else:
        precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)

    print(f"  {tf_name}: {len(cts)} contrats charges")
    for c in cts:
        if c.get('open_ts', 0) < CUTOFF_TS:
            continue
        won, pnl, fills = simulate_trade_log(c, CFG)
        if not fills:
            continue
        avg_fill = float(np.mean([f['price'] for f in fills]))
        ce_utc = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
        key = (ce_utc, tf_name)
        bt_fills[key] = {
            "avg_fill": round(avg_fill, 4),
            "pnl": round(pnl, 2),
            "won": won,
            "tf": tf_name,
            "fills": fills,
        }

cu.BASE_SIZE = orig_size
print(f"BT fills apres cutoff   : {len(bt_fills)}")


# ── 3. Comparaison ────────────────────────────────────────────────────────────
shared    = []
live_only = []
bt_only_keys = set(bt_fills.keys()) - set(live_fills.keys())

for key, lf in sorted(live_fills.items()):
    bt = bt_fills.get(key)
    if bt:
        shared.append({"key": key, "lf": lf, "bt": bt})
    else:
        live_only.append({"key": key, "lf": lf})

bt_only = [(k, bt_fills[k]) for k in sorted(bt_only_keys)]

print(f"\nContrats communs     : {len(shared)}")
print(f"Live uniquement      : {len(live_only)}")
print(f"BT uniquement        : {len(bt_only)}")

# ── 4. Tableau communs ───────────────────────────────────────────────────────
print()
hdr = f"{'CE UTC':17s} {'TF':3s} {'Side':4s} {'Live fill':9s} {'BT fill':7s} {'Gap':7s} {'Live PnL':8s} {'BT PnL':7s}"
print(hdr); print("-" * len(hdr))
gaps = []
for r in shared:
    key = r["key"]; lf = r["lf"]; bt = r["bt"]
    gap = lf["avg_fill"] - bt["avg_fill"]
    gaps.append(gap)
    print(f"{key[0].strftime('%m-%d %H:%M UTC'):17s} {key[1]:3s} {lf['side']:4s} "
          f"{lf['avg_fill']:9.4f} {bt['avg_fill']:7.4f} {gap:+7.4f} "
          f"${lf['pnl']:7.2f} ${bt['pnl']:6.2f}")

if gaps:
    print()
    print(f"Gap moyen  (live - bt) : {np.mean(gaps):+.4f}")
    print(f"Gap median             : {np.median(gaps):+.4f}")
    print(f"live > bt (live + cher): {sum(1 for g in gaps if g > 0.001)} / {len(gaps)}")

# ── 5. Live-only ─────────────────────────────────────────────────────────────
lo_pnls   = [r["lf"]["pnl"] for r in live_only]
lo_fills  = [r["lf"]["avg_fill"] for r in live_only]
print(f"\nLive-only ({len(live_only)} contrats) :")
print(f"  fill avg moyen : {np.mean(lo_fills):.4f}")
print(f"  PnL total      : ${sum(lo_pnls):.2f}")
print(f"  PnL moyen/trade: ${np.mean(lo_pnls):.2f}")
wins = sum(1 for p in lo_pnls if p > 0)
print(f"  Win rate       : {wins}/{len(lo_pnls)} = {100*wins/len(lo_pnls):.0f}%")

# ── 6. BT-only ───────────────────────────────────────────────────────────────
bt_pnls  = [v["pnl"] for _, v in bt_only]
bt_fills_list = [v["avg_fill"] for _, v in bt_only]
print(f"\nBT-only ({len(bt_only)} contrats) :")
print(f"  fill avg moyen : {np.mean(bt_fills_list):.4f}")
print(f"  PnL total      : ${sum(bt_pnls):.2f}")
print(f"  PnL moyen/trade: ${np.mean(bt_pnls):.2f}")
bwins = sum(1 for p in bt_pnls if p > 0)
print(f"  Win rate       : {bwins}/{len(bt_pnls)} = {100*bwins/len(bt_pnls):.0f}%")
print()
hdr2 = f"{'CE UTC':17s} {'TF':3s} {'BT fill':7s} {'BT PnL':7s}"
print(hdr2); print("-" * len(hdr2))
for k, v in bt_only:
    print(f"{k[0].strftime('%m-%d %H:%M UTC'):17s} {k[1]:3s} {v['avg_fill']:7.4f} ${v['pnl']:6.2f}")

# ── 7. Synthèse ───────────────────────────────────────────────────────────────
print()
shared_live_pnl = sum(r["lf"]["pnl"] for r in shared)
shared_bt_pnl   = sum(r["bt"]["pnl"] for r in shared)
print(f"=== SYNTHESE ===")
print(f"Communs  : Live ${shared_live_pnl:.2f}  |  BT ${shared_bt_pnl:.2f}")
print(f"Live-only: Live ${sum(lo_pnls):.2f}")
print(f"BT-only  : BT   ${sum(bt_pnls):.2f}")
print(f"Total live (hors real_loss): ${shared_live_pnl + sum(lo_pnls):.2f}")
print(f"Total BT                  : ${shared_bt_pnl + sum(bt_pnls):.2f}")
