"""
compare_open_prices.py
Compare binance_kline (live start_price) vs premier tick CSV (BT op)
pour chaque contrat m5 ETH après le cutoff.

Sortie : god_curve/live_vs_bt/open_price_gap_eth.png
"""
import json, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE     = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"
LIVE_DIR = BASE / "slope0.2int0trend0.5" / "eth" / "m5"
CSV_PATH = BASE / "ETH.csv"
CUTOFF   = pd.Timestamp("2026-04-05T21:45:00", tz="UTC")
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Charge le CSV et calcule BT op (premier tick de chaque fenetre 5min) ──
df = pd.read_csv(CSV_PATH, usecols=["timestamp", "spot_price"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").drop_duplicates("timestamp")
df["contract"] = df["timestamp"].dt.floor("5min") + pd.Timedelta(minutes=5)

bt_op = {}  # ce_ts (UTC) -> premier tick spot_price
for ce, grp in df.groupby("contract"):
    ce_utc = ce.tz_localize("UTC") if ce.tzinfo is None else ce
    if ce_utc < CUTOFF: continue
    bt_op[ce_utc] = float(grp["spot_price"].iloc[0])

# ── 2. Parse les window_open des JSONL pour extraire binance_kline ────────────
live_open = {}  # ce_ts (UTC) -> binance_kline
for f in sorted(LIVE_DIR.glob("*.jsonl")):
    for line in f.open(encoding="utf-8", errors="replace"):
        try: d = json.loads(line.strip())
        except: continue
        if d.get("event") != "window_open": continue
        ts_raw = pd.to_datetime(d["ts"]).tz_localize(None).tz_localize("UTC")
        ce = (ts_raw + pd.Timedelta(minutes=5)).floor("5min")
        bk = d.get("binance_kline")
        if bk: live_open[ce] = float(bk)
        break  # un seul window_open par fichier

# ── 3. Join et calcule l'écart ────────────────────────────────────────────────
rows = []
for ce, bk in sorted(live_open.items()):
    bt = bt_op.get(ce)
    if bt is None: continue
    gap = bk - bt   # positif = live plus haut que BT
    rows.append(dict(ce=ce, binance_kline=bk, bt_op=bt, gap=gap, abs_gap=abs(gap)))

df_gap = pd.DataFrame(rows)

if df_gap.empty:
    print("Aucun contrat en commun.")
    sys.exit(0)

print(f"Contrats compares : {len(df_gap)}")
print(f"Gap moyen         : {df_gap['gap'].mean():+.4f}$")
print(f"Gap median        : {df_gap['gap'].median():+.4f}$")
print(f"Gap std           : {df_gap['gap'].std():.4f}$")
print(f"Gap max (abs)     : {df_gap['abs_gap'].max():.4f}$")
print(f"|gap| > 0.50$     : {(df_gap['abs_gap'] > 0.50).sum()} contrats")
print(f"|gap| > 0.25$     : {(df_gap['abs_gap'] > 0.25).sum()} contrats")
print(f"|gap| > 0.10$     : {(df_gap['abs_gap'] > 0.10).sum()} contrats")
print(f"|gap| <= 0.05$    : {(df_gap['abs_gap'] <= 0.05).sum()} contrats")

# ── 4. Graphique ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# Panel 1 : gap dans le temps
ax = axes[0]
colors = ["#F44336" if g > 0 else "#2196F3" for g in df_gap["gap"]]
ax.bar(range(len(df_gap)), df_gap["gap"], color=colors, alpha=0.8, width=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(df_gap["gap"].mean(), color="orange", linewidth=1.2,
           linestyle="--", label=f"Moyenne {df_gap['gap'].mean():+.3f} USD")
ax.set_title("Ecart open price : binance_kline (live) - premier tick CSV (BT)", fontsize=11, fontweight="bold")
ax.set_xlabel("Contrats M5 (chronologique)")
ax.set_ylabel("Gap (USD)  [rouge = live > BT]")
ax.legend(fontsize=9)
ax.grid(alpha=0.25)

# Xticks toutes les ~20 contrats
n = len(df_gap)
step = max(1, n // 20)
ax.set_xticks(range(0, n, step))
ax.set_xticklabels(
    [df_gap["ce"].iloc[i].strftime("%d/%m %H:%M") for i in range(0, n, step)],
    fontsize=6, rotation=45
)

# Panel 2 : histogramme du gap
ax2 = axes[1]
bins = np.linspace(df_gap["gap"].min() - 0.1, df_gap["gap"].max() + 0.1, 50)
ax2.hist(df_gap["gap"], bins=bins, color="#90CAF9", edgecolor="white", linewidth=0.4)
ax2.axvline(0, color="black", linewidth=1)
ax2.axvline(df_gap["gap"].mean(), color="orange", linewidth=1.5,
            linestyle="--", label=f"Moy {df_gap['gap'].mean():+.3f} USD")
ax2.axvline(df_gap["gap"].median(), color="green", linewidth=1.2,
            linestyle=":", label=f"Med {df_gap['gap'].median():+.3f} USD")

# Annotations stats
pct_large = 100 * (df_gap["abs_gap"] > 0.25).mean()
ax2.set_title(
    f"Distribution du gap  |  std={df_gap['gap'].std():.3f} USD  "
    f"|  {pct_large:.0f}% des contrats ont |gap| > 0.25 USD",
    fontsize=10
)
ax2.set_xlabel("Gap (binance_kline - bt_op)  en USD")
ax2.set_ylabel("Nombre de contrats")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)

fig.suptitle(
    f"ETH M5 — Open price : Binance kline (live) vs premier tick CSV (BT)\n"
    f"{len(df_gap)} contrats depuis {CUTOFF.strftime('%Y-%m-%d %H:%M')} UTC",
    fontsize=12, fontweight="bold"
)
fig.tight_layout()

out = OUT_DIR / "open_price_gap_eth.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n-> {out}")
