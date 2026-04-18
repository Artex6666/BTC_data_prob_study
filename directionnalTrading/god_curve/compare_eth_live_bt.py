"""
Comparatif Live vs Backtest - ETH
Config: lin_sl0.20_int0.0_c70_tre2h050_ord1
"""
import sys, json, glob, os, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import load_contracts, precompute_vol, simulate, VOL_LBS

CEST      = timezone(timedelta(hours=2))
MONTH_MAP = {'janv':1,'fevr':2,'mars':3,'avr':4,'mai':5,'juin':6,
             'juil':7,'aout':8,'sept':9,'oct':10,'nov':11,'dec':12}

BASE      = Path(__file__).resolve().parent.parent
CSV       = str(BASE / "reportLive/safeChase/ETH.csv")
LIVE_DIR  = str(BASE / "reportLive/safeChase/slope3int0cap70/eth/m5") + os.sep
OUT       = BASE / "eth_live_vs_bt.png"
CUTOFF     = datetime(2026, 4, 3, 16,  5, 0, tzinfo=timezone.utc)
LIVE_START = datetime(2026, 4, 3,  0, 35, tzinfo=CEST).astimezone(timezone.utc)
LIVE_END   = datetime(2026, 4, 4,  8, 40, tzinfo=CEST).astimezone(timezone.utc)

CFG = dict(
    curve='linear', slope=0.2, intercept=0.0,
    eq_cap=0.70, max_orders=1,
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


# ── Live trades ───────────────────────────────────────────────────────────────
live_trades = {}
for fn in sorted(glob.glob(LIVE_DIR + "*.jsonl")):
    events = []
    with open(fn) as f:
        for line in f:
            try:
                events.append(json.loads(line.strip()))
            except Exception:
                pass
    fills = [e for e in events
             if e.get("event") in ("window_ended", "window_settled")
             and (e.get("up_shares", 0) > 0 or e.get("down_shares", 0) > 0)]
    if not fills:
        continue
    ce_utc = parse_ce_utc(fn)
    if ce_utc is None:
        continue
    for e in fills:
        up_s = e.get("up_shares", 0)
        dn_s = e.get("down_shares", 0)
        side = "UP" if up_s > 0 else "DOWN"
        shares = up_s if up_s > 0 else dn_s
        cost   = e.get("up_cost", 0) if up_s > 0 else e.get("down_cost", 0)
        start  = e.get("start_spot", 0)
        end    = e.get("end_spot", 0)
        won    = (end > start and side == "UP") or (end < start and side == "DOWN")
        pnl    = (shares - cost) if won else -cost
        live_trades[ce_utc] = {
            "ce_utc": ce_utc, "side": side, "shares": round(shares, 4),
            "cost": round(cost, 2), "won": won, "live_pnl": round(pnl, 2),
            "file": os.path.basename(fn),
        }

print(f"Live trades: {len(live_trades)}")


# ── Backtest ──────────────────────────────────────────────────────────────────
print("Loading BT contracts (M5)...")
contracts = load_contracts([CSV], "5min", "m5_up_bid", "m5_down_bid", "m5_up_ask", "m5_down_ask")
precompute_vol(contracts, VOL_LBS)
print(f"  {len(contracts)} M5 contracts loaded")


def passes_vol(c):
    val = c.get("vol_2h_trend", 0.0)
    return val >= 0.5


bt_results = []
for c in contracts:
    ce_utc = c["ce"].to_pydatetime().replace(tzinfo=timezone.utc)
    if ce_utc < LIVE_START or ce_utc > LIVE_END:
        continue
    if not passes_vol(c):
        continue
    won, pnl = simulate(c, CFG)
    if pnl is not None:
        bt_results.append({"ce_utc": ce_utc, "pnl": pnl, "won": won})

print(f"  BT trades (filled): {len(bt_results)}")


# ── Build cumulative series ───────────────────────────────────────────────────
bt_sorted   = sorted(bt_results, key=lambda x: x["ce_utc"])
live_sorted = sorted(live_trades.values(), key=lambda x: x["ce_utc"])

bt_dates    = [r["ce_utc"] for r in bt_sorted]
bt_pnl_cum  = np.cumsum([r["pnl"] for r in bt_sorted]) if bt_sorted else np.array([])

live_dates   = [t["ce_utc"] for t in live_sorted]
live_pnl_cum = np.cumsum([t["live_pnl"] for t in live_sorted]) if live_sorted else np.array([])


# ── Match table ───────────────────────────────────────────────────────────────
bt_by_ce = {r["ce_utc"]: r for r in bt_sorted}
match_rows = []
for lt in live_sorted:
    bt = bt_by_ce.get(lt["ce_utc"])
    match_rows.append({
        "ce_utc":    lt["ce_utc"],
        "file":      lt["file"],
        "side":      lt["side"],
        "live_cost": lt["cost"],
        "live_won":  lt["won"],
        "live_pnl":  lt["live_pnl"],
        "bt_pnl":    bt["pnl"] if bt else None,
        "valid":     lt["ce_utc"] >= CUTOFF,
    })


# ── Print table ───────────────────────────────────────────────────────────────
print()
hdr = f"{'CE UTC':17s} {'File':35s} {'Side':4s} {'L.Cost':7s} {'L.Won':5s} {'L.PnL':7s} {'BT.PnL':7s} {'Valid':5s}"
print(hdr)
print("-" * len(hdr))
for r in match_rows:
    bpnl = f"${r['bt_pnl']:6.2f}" if r["bt_pnl"] is not None else "  SKIP "
    print(
        f"{r['ce_utc'].strftime('%m-%d %H:%M UTC'):17s} "
        f"{r['file']:35s} {r['side']:4s} "
        f"${r['live_cost']:6.2f} {str(r['live_won']):5s} "
        f"${r['live_pnl']:6.2f} {bpnl:7s} {str(r['valid']):5s}"
    )

live_tot   = sum(r["live_pnl"] for r in match_rows)
bt_matched = [r for r in match_rows if r["bt_pnl"] is not None]
bt_tot     = sum(r["bt_pnl"] for r in bt_matched)
live_valid = sum(r["live_pnl"] for r in match_rows if r["valid"])
bt_valid   = sum(r["bt_pnl"] for r in match_rows if r["valid"] and r["bt_pnl"] is not None)
print()
print(f"TOTAL live  : ${live_tot:.2f}  ({len(match_rows)} trades)")
print(f"TOTAL BT    : ${bt_tot:.2f}  ({len(bt_matched)} matched fills)")
print(f"After cutoff: live=${live_valid:.2f}  BT=${bt_valid:.2f}")


# ── Chart ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Equity curves
if len(bt_dates) > 0:
    ax1.plot(bt_dates, bt_pnl_cum, color="#2196F3", lw=1.5, label=f"BT M5 ({len(bt_results)} fills)")
if len(live_dates) > 0:
    ax1.plot(live_dates, live_pnl_cum, color="#FF5722", lw=2, marker="o", ms=5,
             label=f"Live ({len(live_sorted)} trades)")
ax1.axvline(CUTOFF, color="gray", ls="--", lw=1, label="Cutoff vol_gate fix (3 avr 18h05 FR)")
ax1.axhline(0, color="gray", lw=0.5)
ax1.set_title("ETH Live vs Backtest — lin_sl0.20_int0.0_c70_tre2h050_ord1  (periode live uniquement: 3-4 avr)", fontsize=10)
ax1.set_ylabel("PnL cumulé ($)")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %Hh"))
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, fontsize=7)

# Per-trade bars
labels    = [r["ce_utc"].astimezone(CEST).strftime("%m/%d\n%Hh%M") for r in match_rows]
x         = np.arange(len(labels))
w         = 0.35
live_vals = [r["live_pnl"] for r in match_rows]
bt_vals   = [r["bt_pnl"] if r["bt_pnl"] is not None else 0.0 for r in match_rows]
c_live    = ["#4CAF50" if v > 0 else "#F44336" for v in live_vals]
c_bt      = ["#2196F3" if v > 0 else "#FF9800" for v in bt_vals]
ax2.bar(x - w/2, live_vals, w, color=c_live, label="Live PnL", alpha=0.85)
ax2.bar(x + w/2, bt_vals,   w, color=c_bt,   label="BT PnL (0=SKIP)", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=7)
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_title("PnL par contrat (Live vs BT)", fontsize=9)
ax2.set_ylabel("PnL ($)")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3, axis="y")

# Summary table
ax3.axis("off")
col_labels = ["CE (FR)", "Side", "L.Cost", "W", "L.PnL", "BT.PnL", "Valid"]
table_data = []
for r in match_rows:
    ce_fr = r["ce_utc"].astimezone(CEST)
    bpnl  = f"${r['bt_pnl']:.2f}" if r["bt_pnl"] is not None else "SKIP"
    table_data.append([
        ce_fr.strftime("%m/%d %Hh%M"),
        r["side"],
        f"${r['live_cost']:.0f}",
        "W" if r["live_won"] else "L",
        f"${r['live_pnl']:.2f}",
        bpnl,
        "V" if r["valid"] else "—",
    ])
table_data.append(["TOTAL (all)",  "", "", "", f"${live_tot:.2f}", f"${bt_tot:.2f}", ""])
table_data.append(["TOTAL valid",  "", "", "", f"${live_valid:.2f}", f"${bt_valid:.2f}", ""])

tbl = ax3.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.3)
for i in range(len(table_data)):
    for j in range(len(col_labels)):
        cell = tbl[i + 1, j]
        if i >= len(table_data) - 2:
            cell.set_facecolor("#E3F2FD")
        elif table_data[i][3] == "L":
            cell.set_facecolor("#FFEBEE")
ax3.set_title("Tableau comparatif ETH", fontsize=9)

plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\nChart saved: {OUT}")
