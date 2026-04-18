"""
Analyse des triggers des bots ordresMarkets sur Polymarket (BTC & ETH).

Questions principales :
1. Quand et pourquoi switchent-ils de côté (UP→DOWN ou DOWN→UP) ?
2. Quelles sont les conditions au premier achat ?
3. La position nette finale est-elle dans le bon sens plus souvent que le hasard ?
4. L'ask_c influence-t-il les entrées ?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
OUT  = Path(__file__).parent / "trigger_analysis"
OUT.mkdir(exist_ok=True)

TF_MAP = {"h1": "1h", "m15": "15min", "m5": "5min"}

# ─────────────────────────────────────────────────────────────
# 1. CHARGEMENT
# ─────────────────────────────────────────────────────────────
print("Chargement des données…")
dfs = []
for asset in ("btc", "eth"):
    path = BASE / "ordresmarkets_strategy" / f"features_{asset}.csv"
    d = pd.read_csv(path, parse_dates=["timestamp"])
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
print(f"  Features : {len(df):,} lignes — {df['asset'].value_counts().to_dict()}")

settlement = pd.read_csv(BASE / "Datas" / "csv" / "settlement.csv")
print(f"  Settlement : {len(settlement):,} lignes")

# Jointure : contract_open_ts (UTC Unix) + asset + tf
df["tf_set"] = df["tf_slug"].map(TF_MAP)
df = df.merge(
    settlement[["asset", "tf", "open_ts", "outcome_up", "outcome_down"]],
    left_on=["asset", "tf_set", "contract_open_ts"],
    right_on=["asset", "tf", "open_ts"],
    how="left"
)
matched = df["outcome_up"].notna().mean()
print(f"  Jointure settlement : {matched:.1%} des lignes matchées")

df["outcome_side"] = np.where(df["outcome_up"] == 1.0, "Up", "Down")

# Filtre : garder seulement les lignes avec achat
buys = df[df["buy_count"] > 0].copy()
print(f"  Lignes d'achat : {len(buys):,}")

# ─────────────────────────────────────────────────────────────
# 2. SÉQUENCE DES ACHATS PAR CONTRAT
# ─────────────────────────────────────────────────────────────
print("\nAnalyse des séquences par contrat…")

buys_sorted = buys.sort_values(["asset", "market_key", "side", "timestamp"])

# Résumé par (market_key, side)
mkt_side = (
    buys_sorted.groupby(["asset", "tf_slug", "market_key", "side", "outcome_side"])
    .agg(
        n_buys=("buy_count", "sum"),
        cost_usd=("buy_cost_usd", "sum"),
        shares=("buy_shares", "sum"),
        avg_ask=("ask_c", "mean"),
        first_elapsed=("elapsed_frac", "first"),
        first_ask=("ask_c", "first"),
        first_aligned=("aligned", "first"),
        first_recross=("recross", "first"),
        first_signed_open=("signed_open_pct", "first"),
        first_move30s=("signed_move_30s_pct", "first"),
    )
    .reset_index()
)
mkt_side["won"] = mkt_side["side"] == mkt_side["outcome_side"]

# Position nette par contrat (UP shares − DOWN shares en USD)
mkt_net = mkt_side.pivot_table(
    index=["asset", "tf_slug", "market_key", "outcome_side"],
    columns="side",
    values=["cost_usd", "shares", "avg_ask"],
    aggfunc="first"
).reset_index()
mkt_net.columns = ["_".join(c).strip("_") for c in mkt_net.columns]
mkt_net = mkt_net.rename(columns={
    "asset_": "asset", "tf_slug_": "tf_slug",
    "market_key_": "market_key", "outcome_side_": "outcome_side"
})

for col in ["cost_usd_Up", "cost_usd_Down", "shares_Up", "shares_Down"]:
    if col not in mkt_net.columns:
        mkt_net[col] = 0.0
mkt_net = mkt_net.fillna(0.0)

mkt_net["net_cost_usd"] = mkt_net["cost_usd_Up"] - mkt_net["cost_usd_Down"]
mkt_net["net_sign"] = np.sign(mkt_net["net_cost_usd"])
mkt_net["net_correct"] = (
    ((mkt_net["net_sign"] > 0) & (mkt_net["outcome_side"] == "Up")) |
    ((mkt_net["net_sign"] < 0) & (mkt_net["outcome_side"] == "Down"))
)
mkt_net["both_sides"] = (mkt_net["cost_usd_Up"] > 0) & (mkt_net["cost_usd_Down"] > 0)

# ─────────────────────────────────────────────────────────────
# 3. SWITCH EVENTS
# ─────────────────────────────────────────────────────────────
print("Analyse des switch events…")

# Pour chaque contrat, construire la séquence temporelle des achats
all_buys_seq = buys_sorted.sort_values(["asset", "market_key", "timestamp"])

def get_switch_events(group):
    events = []
    prev_side = None
    for _, row in group.iterrows():
        if prev_side is not None and row["side"] != prev_side:
            events.append({
                "asset": row["asset"],
                "tf_slug": row["tf_slug"],
                "market_key": row["market_key"],
                "from_side": prev_side,
                "to_side": row["side"],
                "elapsed_frac": row["elapsed_frac"],
                "recross": row["recross"],
                "aligned": row["aligned"],
                "signed_open_pct": row["signed_open_pct"],
                "signed_move_30s_pct": row["signed_move_30s_pct"],
                "ask_c": row["ask_c"],
            })
        prev_side = row["side"]
    return events

switch_list = []
for (asset, mkey), grp in all_buys_seq.groupby(["asset", "market_key"]):
    switch_list.extend(get_switch_events(grp))

switches = pd.DataFrame(switch_list)
print(f"  Switch events : {len(switches):,}")

# ─────────────────────────────────────────────────────────────
# 4. PREMIER ACHAT PAR CONTRAT
# ─────────────────────────────────────────────────────────────
first_buys = (
    all_buys_seq.groupby(["asset", "market_key"])
    .first()
    .reset_index()[["asset", "tf_slug", "market_key", "side", "elapsed_frac",
                     "ask_c", "aligned", "recross", "signed_open_pct",
                     "signed_move_30s_pct", "outcome_side"]]
)
first_buys["first_side_correct"] = first_buys["side"] == first_buys["outcome_side"]

# ─────────────────────────────────────────────────────────────
# 5. RAPPORT TEXTE
# ─────────────────────────────────────────────────────────────
lines = []

lines.append("=" * 70)
lines.append("ANALYSE DES TRIGGERS — ordresMarkets (BTC & ETH)")
lines.append("=" * 70)

for asset in ("btc", "eth"):
    for tf in ("h1", "m15", "m5"):
        sub = mkt_net[(mkt_net["asset"] == asset) & (mkt_net["tf_slug"] == tf)]
        if sub.empty:
            continue

        sub_buys = mkt_side[(mkt_side["asset"] == asset) & (mkt_side["tf_slug"] == tf)]
        sub_sw = switches[(switches["asset"] == asset) & (switches["tf_slug"] == tf)]
        sub_first = first_buys[(first_buys["asset"] == asset) & (first_buys["tf_slug"] == tf)]

        n_contracts = len(sub)
        both = sub["both_sides"].mean()
        net_correct = sub[sub["net_cost_usd"] != 0]["net_correct"].mean()

        avg_switches = len(sub_sw) / n_contracts if n_contracts else 0
        sw_recross = sub_sw["recross"].mean() if len(sub_sw) else float("nan")
        sw_aligned = sub_sw["aligned"].mean() if len(sub_sw) else float("nan")

        first_elapsed = sub_first["elapsed_frac"].median()
        first_ask = sub_first["ask_c"].median()
        first_aligned = sub_first["aligned"].mean()
        first_recross = sub_first["recross"].mean()
        first_correct = sub_first["first_side_correct"].mean()

        up_buys = sub_buys[sub_buys["side"] == "Up"]
        dn_buys = sub_buys[sub_buys["side"] == "Down"]
        avg_ask_up = up_buys["avg_ask"].mean() if len(up_buys) else float("nan")
        avg_ask_dn = dn_buys["avg_ask"].mean() if len(dn_buys) else float("nan")

        lines.append(f"\n{'─'*60}")
        lines.append(f"{asset.upper()} | {tf.upper()}  ({n_contracts} contrats)")
        lines.append(f"{'─'*60}")
        lines.append(f"  Contrats avec les 2 côtés  : {both:.1%}")
        lines.append(f"  Position nette finale OK   : {net_correct:.1%}  (aléatoire=50%)")
        lines.append(f"")
        lines.append(f"  SWITCHES DE CÔTÉ")
        lines.append(f"    Switches / contrat (moy) : {avg_switches:.1f}")
        lines.append(f"    % switches avec recross=1: {sw_recross:.1%}")
        lines.append(f"    % switches avec aligned=1: {sw_aligned:.1%}")
        lines.append(f"")
        lines.append(f"  PREMIER ACHAT")
        lines.append(f"    elapsed_frac médian      : {first_elapsed:.3f}  ({first_elapsed*100:.1f}% du contrat)")
        lines.append(f"    ask_c médian             : {first_ask:.0f}c")
        lines.append(f"    % aligned=1              : {first_aligned:.1%}")
        lines.append(f"    % recross=1              : {first_recross:.1%}")
        lines.append(f"    % 1er côté = bon côté    : {first_correct:.1%}")
        lines.append(f"")
        lines.append(f"  ASK_C MOYEN")
        lines.append(f"    Achats UP                : {avg_ask_up:.1f}c")
        lines.append(f"    Achats DOWN              : {avg_ask_dn:.1f}c")

# Ask_c global : buy vs non-buy
lines.append("\n" + "=" * 70)
lines.append("ASK_C : DISTRIBUTION BUY vs NON-BUY")
lines.append("=" * 70)
for asset in ("btc", "eth"):
    for tf in ("h1", "m15", "m5"):
        sub_all = df[(df["asset"] == asset) & (df["tf_slug"] == tf)]
        buy_ask = sub_all[sub_all["buy_count"] > 0]["ask_c"]
        no_ask  = sub_all[sub_all["buy_count"] == 0]["ask_c"]
        if len(buy_ask) == 0:
            continue
        lines.append(f"\n  {asset.upper()} | {tf.upper()}")
        lines.append(f"    Buy  ask_c : mean={buy_ask.mean():.1f}  med={buy_ask.median():.1f}  p25={buy_ask.quantile(0.25):.1f}  p75={buy_ask.quantile(0.75):.1f}")
        lines.append(f"    No-buy ask : mean={no_ask.mean():.1f}  med={no_ask.median():.1f}  p25={no_ask.quantile(0.25):.1f}  p75={no_ask.quantile(0.75):.1f}")

report_text = "\n".join(lines)
(OUT / "report_triggers.txt").write_text(report_text, encoding="utf-8")
print(report_text.encode("ascii", "replace").decode("ascii"))

# ─────────────────────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────────────────────
print("\nGénération des plots…")

ASSETS  = ["btc", "eth"]
TFS     = ["h1", "m15", "m5"]
COLORS  = {"Up": "#2196F3", "Down": "#F44336"}

# ── Plot 1 : Position nette finale vs outcome ──────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Position nette finale dans le bon sens (%)\n(>50% = edge directionnel réel)", fontsize=13)
for i, asset in enumerate(ASSETS):
    for j, tf in enumerate(TFS):
        ax = axes[i][j]
        sub = mkt_net[(mkt_net["asset"] == asset) & (mkt_net["tf_slug"] == tf)]
        sub = sub[sub["net_cost_usd"] != 0]
        if sub.empty:
            ax.set_visible(False)
            continue
        pct = sub["net_correct"].mean() * 100
        both_pct = sub[sub["both_sides"]]["net_correct"].mean() * 100 if sub["both_sides"].any() else 0

        ax.bar(["Global", "Deux côtés"], [pct, both_pct],
               color=["steelblue", "darkorange"], alpha=0.85)
        ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50% (aléatoire)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{asset.upper()} | {tf.upper()}\n(n={len(sub)})")
        ax.set_ylabel("% correct")
        for bar, val in zip(ax.patches, [pct, both_pct]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(OUT / "plot1_net_position_correct.png", dpi=120)
plt.close()

# ── Plot 2 : % switches avec recross=1 par asset/tf ──────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Conditions au moment du switch de côté (UP↔DOWN)", fontsize=13)
for i, asset in enumerate(ASSETS):
    for j, tf in enumerate(TFS):
        ax = axes[i][j]
        sub_sw = switches[(switches["asset"] == asset) & (switches["tf_slug"] == tf)]
        if sub_sw.empty:
            ax.set_visible(False)
            continue
        conditions = {
            "recross=1": sub_sw["recross"].mean(),
            "aligned=1": sub_sw["aligned"].mean(),
            "move30s>0": (sub_sw["signed_move_30s_pct"] > 0).mean(),
        }
        ax.bar(list(conditions.keys()), [v * 100 for v in conditions.values()],
               color=["#9C27B0", "#FF9800", "#4CAF50"], alpha=0.85)
        ax.axhline(50, color="red", linestyle="--", linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_title(f"{asset.upper()} | {tf.upper()}\n(n={len(sub_sw)} switches)")
        ax.set_ylabel("% vrai lors du switch")
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "plot2_switch_conditions.png", dpi=120)
plt.close()

# ── Plot 3 : Distribution ask_c buy vs non-buy ────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Distribution ask_c : moments d'achat vs non-achat", fontsize=13)
for i, asset in enumerate(ASSETS):
    for j, tf in enumerate(TFS):
        ax = axes[i][j]
        sub_all = df[(df["asset"] == asset) & (df["tf_slug"] == tf)]
        buy_ask = sub_all[sub_all["buy_count"] > 0]["ask_c"].dropna()
        no_ask  = sub_all[sub_all["buy_count"] == 0]["ask_c"].dropna()
        if buy_ask.empty:
            ax.set_visible(False)
            continue
        bins = np.arange(0, 102, 5)
        ax.hist(no_ask,  bins=bins, alpha=0.5, density=True, label="Non-achat", color="gray")
        ax.hist(buy_ask, bins=bins, alpha=0.7, density=True, label="Achat",     color="steelblue")
        ax.set_title(f"{asset.upper()} | {tf.upper()}")
        ax.set_xlabel("ask_c (cents)")
        ax.set_ylabel("densité")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "plot3_askc_distribution.png", dpi=120)
plt.close()

# ── Plot 4 : Timing du premier achat ──────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Timing du premier achat (elapsed_frac)", fontsize=13)
for i, asset in enumerate(ASSETS):
    for j, tf in enumerate(TFS):
        ax = axes[i][j]
        sub = first_buys[(first_buys["asset"] == asset) & (first_buys["tf_slug"] == tf)]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.hist(sub["elapsed_frac"], bins=20, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(sub["elapsed_frac"].median(), color="red", linestyle="--",
                   label=f"médiane={sub['elapsed_frac'].median():.2f}")
        ax.set_title(f"{asset.upper()} | {tf.upper()}\n(n={len(sub)})")
        ax.set_xlabel("elapsed_frac (0=début, 1=fin)")
        ax.set_ylabel("nb contrats")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "plot4_first_buy_timing.png", dpi=120)
plt.close()

# ── Plot 5 : ask_c des achats UP vs DOWN par outcome ──────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("ask_c moyen des achats : UP vs DOWN\n(groupé par issue réelle du contrat)", fontsize=13)
for i, asset in enumerate(ASSETS):
    for j, tf in enumerate(TFS):
        ax = axes[i][j]
        sub = mkt_side[(mkt_side["asset"] == asset) & (mkt_side["tf_slug"] == tf)]
        if sub.empty:
            ax.set_visible(False)
            continue
        outcomes = ["Up", "Down"]
        x = np.arange(len(outcomes))
        w = 0.35
        for k, side in enumerate(["Up", "Down"]):
            vals = [sub[(sub["outcome_side"] == o) & (sub["side"] == side)]["avg_ask"].mean()
                    for o in outcomes]
            bars = ax.bar(x + (k - 0.5) * w, vals, w, label=f"Achète {side}",
                          color=COLORS[side], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Résout {o}" for o in outcomes])
        ax.set_title(f"{asset.upper()} | {tf.upper()}")
        ax.set_ylabel("ask_c moyen (cents)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUT / "plot5_askc_by_side_and_outcome.png", dpi=120)
plt.close()

# ── Plot 6 : Évolution position nette (échantillon) ───────────
print("Plot 6 : évolution position nette…")
sample_markets = (
    mkt_net[(mkt_net["both_sides"]) & (mkt_net["tf_slug"] == "h1")]
    .sample(min(12, len(mkt_net)), random_state=42)["market_key"].tolist()
)
fig, axes = plt.subplots(3, 4, figsize=(18, 10))
fig.suptitle("Évolution cumulative de la position nette USD (UP − DOWN)\n(contrats H1 aléatoires)", fontsize=13)
axes_flat = axes.flatten()
for idx, mkey in enumerate(sample_markets[:12]):
    ax = axes_flat[idx]
    mkey_buys = all_buys_seq[all_buys_seq["market_key"] == mkey].sort_values("timestamp")
    if mkey_buys.empty:
        ax.set_visible(False)
        continue
    outcome = mkey_buys["outcome_side"].iloc[0] if "outcome_side" in mkey_buys.columns else "?"
    mkey_buys = mkey_buys.copy()
    mkey_buys["signed_cost"] = mkey_buys.apply(
        lambda r: r["buy_cost_usd"] if r["side"] == "Up" else -r["buy_cost_usd"], axis=1
    )
    mkey_buys["cum_net"] = mkey_buys["signed_cost"].cumsum()
    color = "#2196F3" if outcome == "Up" else "#F44336"
    ax.plot(mkey_buys["elapsed_frac"], mkey_buys["cum_net"], color=color, linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{mkey}\nRésout: {outcome}", fontsize=7)
    ax.set_xlabel("elapsed_frac", fontsize=7)
    ax.set_ylabel("net USD", fontsize=7)

plt.tight_layout()
plt.savefig(OUT / "plot6_net_position_evolution.png", dpi=120)
plt.close()

print(f"\nTous les fichiers sauvegardés dans : {OUT}")
print("Plots : plot1 à plot6 + report_triggers.txt")
