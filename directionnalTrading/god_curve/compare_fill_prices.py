"""
compare_fill_prices.py
Pour chaque contrat tradé en BT, extrait le prix de fill BT (= bid au moment du
placement de l'ordre) et le prix de fill live (= cost/shares depuis window_ended).
Compare les deux et identifie l'écart systématique.

CONFIG : slope=0.2, intercept=0, eq_cap=0.60, trend 2h>0.50
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
from chart_utils import load_contracts, precompute_vol, simulate_trade_log, VOL_LBS

CEST      = timezone(timedelta(hours=2))
MONTH_MAP = {'janv':1,'fevr':2,'mars':3,'avr':4,'mai':5,'juin':6,
             'juil':7,'aout':8,'sept':9,'oct':10,'nov':11,'dec':12}

BASE     = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"
LIVE_DIR = BASE / "slope0.2int0trend0.5" / "eth" / "m5"
CSV_PATH = str(BASE / "ETH.csv")
CUTOFF   = datetime(2026, 4, 5, 21, 45, 0, tzinfo=timezone.utc)
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CFG = dict(
    curve='linear', slope=0.2, intercept=0.0,
    eq_cap=0.60, max_orders=1,
    vol_lb_h=2.0, vol_thresh=0.5, vol_type='trend',
)


def parse_ce_utc(fname):
    m = re.match(r'(\d+)_(\w+)_\d+h\d+-(\d+)h(\d+)\.jsonl', os.path.basename(fname))
    if not m:
        return None
    day, mon_str, hour, minute = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
    mon = MONTH_MAP.get(mon_str)
    if not mon:
        return None
    dt_fr = datetime(2026, mon, day, hour, minute, 0, tzinfo=CEST)
    return dt_fr.astimezone(timezone.utc)


# ── 1. Charge les fills live ──────────────────────────────────────────────────
live_fills = {}  # ce_utc -> dict

for fn in sorted(LIVE_DIR.glob("*.jsonl")):
    ce_utc = parse_ce_utc(fn)
    if ce_utc is None or ce_utc < CUTOFF:
        continue

    events = []
    for line in fn.open(encoding="utf-8", errors="replace"):
        try: events.append(json.loads(line.strip()))
        except: continue

    # Prix de fill : cost total / shares totales depuis window_ended
    we = next((e for e in events if e.get("event") == "window_ended"), None)
    if we is None:
        continue
    up_s  = we.get("up_shares", 0)
    dn_s  = we.get("down_shares", 0)
    up_c  = we.get("up_cost", 0)
    dn_c  = we.get("down_cost", 0)

    if up_s > 0:
        side = "UP"; shares = up_s; cost = up_c
    elif dn_s > 0:
        side = "DOWN"; shares = dn_s; cost = dn_c
    else:
        continue  # pas de fill live

    avg_fill = cost / shares if shares > 0 else None

    # PnL live depuis window_ended
    start_spot = we.get("start_spot", 0)
    end_spot   = we.get("end_spot", 0)
    won = (end_spot > start_spot and side == "UP") or (end_spot < start_spot and side == "DOWN")
    live_pnl = round((shares - cost) if won else -cost, 4)

    # Premier chase_placed (prix de l'ordre initial)
    cp_events = [e for e in events if e.get("event") == "chase_placed"]
    first_chase_price = float(cp_events[0]["price"]) if cp_events else None

    live_fills[ce_utc] = {
        "side": side,
        "shares": round(shares, 4),
        "cost": round(cost, 4),
        "avg_fill_price": round(avg_fill, 4) if avg_fill else None,
        "first_chase_price": first_chase_price,
        "won": won,
        "live_pnl": live_pnl,
        "file": os.path.basename(fn),
    }

print(f"Live fills apres cutoff : {len(live_fills)}")


# ── 2. BT simulation ──────────────────────────────────────────────────────────
print("Chargement CSV BT...")
contracts = load_contracts([CSV_PATH], "5min", "m5_up_bid", "m5_down_bid", "m5_up_ask", "m5_down_ask")
precompute_vol(contracts, VOL_LBS)
print(f"  {len(contracts)} contrats M5 charges")


def passes_vol(c):
    return c.get("vol_2h_trend", 0.0) >= 0.5


bt_fills = {}  # ce_utc -> dict

for c in contracts:
    ce_utc = c["ce"].to_pydatetime().replace(tzinfo=timezone.utc)
    if ce_utc < CUTOFF:
        continue
    if not passes_vol(c):
        continue
    won, pnl, fills = simulate_trade_log(c, CFG)
    if not fills:
        continue
    f0 = fills[0]
    bt_fills[ce_utc] = {
        "side": f0["side"],
        "fill_price": f0["price"],  # bid au moment du placement de l'ordre
        "pnl": round(pnl, 4),
        "won": won,
        "op": float(c["op"]),
    }

print(f"BT fills apres cutoff   : {len(bt_fills)}")


# ── 3. Comparaison sur contrats communs ───────────────────────────────────────
rows = []
for ce, lf in sorted(live_fills.items()):
    bt = bt_fills.get(ce)
    rows.append({
        "ce_utc": ce,
        "file": lf["file"],
        "live_side": lf["side"],
        "live_avg_fill": lf["avg_fill_price"],
        "live_first_chase": lf["first_chase_price"],
        "bt_side": bt["side"] if bt else None,
        "bt_fill_price": bt["fill_price"] if bt else None,
        "bt_pnl": bt["pnl"] if bt else None,
        "in_bt": bt is not None,
        "same_side": (bt is not None and bt["side"] == lf["side"]),
    })

shared = [r for r in rows if r["in_bt"] and r["same_side"]]
live_only = [r for r in rows if not r["in_bt"]]
diff_side = [r for r in rows if r["in_bt"] and not r["same_side"]]

print(f"\nContrats communs (meme side) : {len(shared)}")
print(f"Live uniquement              : {len(live_only)}")
print(f"Cotes opposees               : {len(diff_side)}")

# ── 4. Tableau des contrats communs ───────────────────────────────────────────
print()
hdr = f"{'CE UTC':17s} {'File':38s} {'Side':4s} {'Live avg':8s} {'Live 1st':8s} {'BT fill':7s} {'Gap':7s} {'BT pnl':7s}"
print(hdr)
print("-" * len(hdr))

gaps = []
for r in shared:
    gap = r["live_avg_fill"] - r["bt_fill_price"] if (r["live_avg_fill"] and r["bt_fill_price"]) else None
    if gap is not None:
        gaps.append(gap)
    gap_s = f"{gap:+.4f}" if gap is not None else "  N/A "
    print(
        f"{r['ce_utc'].strftime('%m-%d %H:%M UTC'):17s} "
        f"{r['file']:38s} "
        f"{r['live_side']:4s} "
        f"{r['live_avg_fill'] or 0:8.4f} "
        f"{r['live_first_chase'] or 0:8.4f} "
        f"{r['bt_fill_price'] or 0:7.4f} "
        f"{gap_s:7s} "
        f"${r['bt_pnl'] or 0:6.2f}"
    )

if gaps:
    print()
    print(f"Gap moyen  (live_avg - bt_fill) : {np.mean(gaps):+.4f}")
    print(f"Gap median                      : {np.median(gaps):+.4f}")
    print(f"Gap std                         : {np.std(gaps):.4f}")
    print(f"live > bt (live plus cher)      : {sum(1 for g in gaps if g > 0.001)} / {len(gaps)}")
    print(f"live < bt (live moins cher)     : {sum(1 for g in gaps if g < -0.001)} / {len(gaps)}")
    print(f"similaire (|gap| <= 0.001)      : {sum(1 for g in gaps if abs(g) <= 0.001)} / {len(gaps)}")

# ── 5. Distribution des fill prices ──────────────────────────────────────────
if shared:
    live_prices = [r["live_avg_fill"] for r in shared if r["live_avg_fill"]]
    bt_prices   = [r["bt_fill_price"] for r in shared if r["bt_fill_price"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x = range(len(shared))
    ax.plot(x, live_prices, "o-", color="#FF5722", label=f"Live avg fill (moy={np.mean(live_prices):.4f})", lw=1.5, ms=5)
    ax.plot(x, bt_prices,   "s-", color="#2196F3", label=f"BT fill price (moy={np.mean(bt_prices):.4f})", lw=1.5, ms=5)
    for i, (lp, bp) in enumerate(zip(live_prices, bt_prices)):
        ax.plot([i, i], [lp, bp], color="gray", lw=0.7, alpha=0.5)
    ax.set_title("Fill price : Live avg vs BT — contrats communs (meme side)", fontsize=10)
    ax.set_xlabel("Contrats (chronologique)")
    ax.set_ylabel("Prix de fill (0-1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xticks(range(0, len(shared), max(1, len(shared)//15)))
    ax.set_xticklabels(
        [shared[i]["ce_utc"].strftime("%d/%m %H:%M") for i in range(0, len(shared), max(1, len(shared)//15))],
        fontsize=6, rotation=45
    )

    ax2 = axes[1]
    bins = np.linspace(0.55, 1.01, 30)
    ax2.hist(live_prices, bins=bins, alpha=0.6, color="#FF5722", label="Live avg fill")
    ax2.hist(bt_prices,   bins=bins, alpha=0.6, color="#2196F3", label="BT fill price")
    ax2.axvline(np.mean(live_prices), color="#FF5722", lw=2, ls="--", label=f"Live moy {np.mean(live_prices):.4f}")
    ax2.axvline(np.mean(bt_prices),   color="#2196F3", lw=2, ls="--", label=f"BT moy {np.mean(bt_prices):.4f}")
    ax2.set_title("Distribution des fill prices — contrats communs", fontsize=10)
    ax2.set_xlabel("Prix de fill")
    ax2.set_ylabel("Nombre de contrats")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    fig.suptitle(
        f"ETH M5 — Fill price comparison : {len(shared)} contrats communs\n"
        f"Gap moyen live-BT : {np.mean(gaps):+.4f}  |  std : {np.std(gaps):.4f}",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()

    out = OUT_DIR / "fill_price_gap_eth.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n-> {out}")

# ── 6. Live-only : fill prices distribution ───────────────────────────────────
if live_only:
    lo_prices = [r["live_avg_fill"] for r in live_only if r["live_avg_fill"]]
    lo_pnls   = [live_fills[r["ce_utc"]]["live_pnl"] for r in live_only]
    print(f"\nLive-only ({len(live_only)} contrats) :")
    print(f"  fill price moyen  : {np.mean(lo_prices):.4f}")
    print(f"  fill price median : {np.median(lo_prices):.4f}")
    print(f"  PnL total         : ${sum(lo_pnls):.2f}")
    print(f"  PnL moyen/trade   : ${np.mean(lo_pnls):.2f}")
    wins = sum(1 for p in lo_pnls if p > 0)
    print(f"  Win rate          : {wins}/{len(lo_pnls)} = {100*wins/len(lo_pnls):.0f}%")

# ── 7. BT-only : contrats que BT a tradés mais pas live ───────────────────────
live_ces = set(live_fills.keys())
bt_only = {ce: bt for ce, bt in bt_fills.items() if ce not in live_ces}

print(f"\nBT-only (BT fill, pas de live) : {len(bt_only)}")
if bt_only:
    print()
    hdr2 = f"{'CE UTC':17s} {'Side':4s} {'BT fill':7s} {'BT pnl':7s}"
    print(hdr2)
    print("-" * len(hdr2))
    bt_only_prices = []
    for ce, bt in sorted(bt_only.items()):
        print(f"{ce.strftime('%m-%d %H:%M UTC'):17s} {bt['side']:4s} {bt['fill_price']:7.4f} ${bt['pnl']:6.2f}")
        bt_only_prices.append(bt["fill_price"])
    print()
    print(f"BT-only fill price moyen  : {np.mean(bt_only_prices):.4f}")
    print(f"BT-only fill price median : {np.median(bt_only_prices):.4f}")
