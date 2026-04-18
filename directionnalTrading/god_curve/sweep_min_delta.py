"""
Sweep de min_delta_usd sur la stratégie baseline slope=7 / BTC.

min_delta_usd : plancher du move |s - op| requis pour trigger,
indépendamment de la slope (qui tend vers 0 en fin de contrat).

Contrairement à l'intercept (qui s'ajoute dès le début), ce filtre
ne joue que quand slope*remain < min_delta, i.e. dans les dernières
secondes. Sur slope=7 BTC : min_delta=10$ → s'active uniquement si
remain < 10/7 ≈ 1.4s.

Sorties : god_curve/min_delta_sweep/
  - equity_curves.png   : courbes d'equity superposées
  - stats_table.png     : tableau récap (trades, WR, PnL, BE-WR)
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "min_delta_sweep"
OUT_DIR.mkdir(exist_ok=True)

CSV_FILES = [
    str(BASE / "Datas/csv/BTC.csv"),
    str(BASE / "reportLive/safeChase/BTC.csv"),
]

# ── Import chart_utils ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_utils as cu
from chart_utils import (
    load_contracts, precompute_vol,
    build_cumulative, build_hour_index, make_xlabels,
    attach_settlement_outcomes, run_config_tf,
)

# ── Config baseline ───────────────────────────────────────────────────────────
BASE_CFG = dict(
    curve='linear', slope=7.0, intercept=0.0,
    eq_cap=0.75,
    vol_lb_h=1, vol_thresh=60.0, vol_type='net',
    max_losses_cb=None, max_orders=2,
    # min_delta_usd injecté à chaque sweep
)

# Valeurs à tester (en USD delta depuis open)
MIN_DELTAS = [0, 2, 5, 10, 20, 30, 50]

COLORS_SWEEP = [
    '#888888',   # 0  (baseline gris)
    '#4fc3f7',   # 2
    '#29b6f6',   # 5
    '#0288d1',   # 10
    '#f4a261',   # 20
    '#e76f51',   # 30
    '#d62828',   # 50
]

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement CSV BTC...", flush=True)
contracts = load_contracts(CSV_FILES, '5min', 'm5_up_bid', 'm5_down_bid', 'm5_up_ask', 'm5_down_ask')
print(f"  {len(contracts)} contrats bruts", flush=True)

precompute_vol(contracts, cu.VOL_LBS)
n_set = attach_settlement_outcomes(contracts, 'btc', tf='5min')
print(f"  {n_set} contrats avec settlement connu (settlement-only mode)", flush=True)

# Filtre settlement-only (pessimiste)
contracts = [c for c in contracts if '_settlement_won_up' in c]
print(f"  {len(contracts)} contrats après filtre settlement", flush=True)

hour_index, n_hours = build_hour_index([contracts])
x_vals = np.arange(n_hours)

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = []   # list of dict per min_delta

for md in MIN_DELTAS:
    cfg = {**BASE_CFG, 'min_delta_usd': md}
    pnl_by_hour, wins, losses = run_config_tf(contracts, cfg, hour_index)

    n_trades = wins + losses
    wr       = 100.0 * wins / n_trades if n_trades else 0.0

    cum, _ = build_cumulative(pnl_by_hour, n_hours)

    pnl_total = float(cum[-1]) if len(cum) else 0.0

    results.append(dict(
        md=md,
        n_trades=n_trades,
        wins=wins,
        losses=losses,
        wr=wr,
        pnl_total=pnl_total,
        cum=cum,
    ))
    label = f"min_delta={md}$"
    print(f"  {label:<20}  trades={n_trades:4d}  wins={wins:4d}  WR={wr:.1f}%  PnL={pnl_total:+.1f}$", flush=True)

# ── Figure principale : equity curves ─────────────────────────────────────────
print("\nGénération equity_curves.png...", flush=True)

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

for i, r in enumerate(results):
    color = COLORS_SWEEP[i % len(COLORS_SWEEP)]
    lw    = 2.5 if r['md'] == 0 else 1.5
    alpha = 1.0 if r['md'] == 0 else 0.85
    label = f"min={r['md']}USD  |  {r['n_trades']} trades  WR={r['wr']:.1f}%  PnL={r['pnl_total']:+.0f}USD"
    ax.plot(x_vals[:len(r['cum'])], r['cum'], color=color, lw=lw, alpha=alpha, label=label)

ax.axhline(0, color='#555', lw=0.8, ls='--')
ax.set_xlabel('Temps', color='#aaa', fontsize=10)
ax.set_ylabel('PnL cumulé (USD)', color='#aaa', fontsize=10)
ax.set_title('Sweep min_delta_usd — slope=7 / BTC / settlement-only', color='white', fontsize=13, pad=12)
ax.tick_params(colors='#aaa')
for sp in ax.spines.values(): sp.set_edgecolor('#333')

# X labels (décimées)
step = max(1, n_hours // 12)
tick_pos, xlabs = make_xlabels(hour_index, step=step)
ax.set_xticks(tick_pos)
ax.set_xticklabels(xlabs, rotation=35, ha='right', fontsize=8, color='#aaa')

legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.25,
                   facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
plt.tight_layout()
fig.savefig(OUT_DIR / "equity_curves.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Sauvé : {OUT_DIR / 'equity_curves.png'}", flush=True)

# ── Figure : stats table ───────────────────────────────────────────────────────
print("Génération stats_table.png...", flush=True)

fig2, ax2 = plt.subplots(figsize=(10, 4))
fig2.patch.set_facecolor('#0e1117')
ax2.set_facecolor('#0e1117')
ax2.axis('off')

col_labels = ['min_delta ($)', 'Trades', 'Wins', 'Losses', 'WR (%)', 'PnL ($)']
table_data = []
for r in results:
    table_data.append([
        f"{r['md']}",
        f"{r['n_trades']}",
        f"{r['wins']}",
        f"{r['losses']}",
        f"{r['wr']:.1f}",
        f"{r['pnl_total']:+.1f}",
    ])

tbl = ax2.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.3, 1.8)

# Coloration
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#1a1a2e' if row % 2 == 0 else '#12122a')
    cell.set_text_props(color='white')
    cell.set_edgecolor('#333')
    if row == 0:
        cell.set_facecolor('#0d47a1')
        cell.set_text_props(color='white', fontweight='bold')

# Colore la ligne baseline (md=0) différemment
for col in range(len(col_labels)):
    tbl[1, col].set_facecolor('#2a2a2a')

ax2.set_title('Stats par min_delta_usd — slope=7 / BTC', color='white', fontsize=12, pad=10)
plt.tight_layout()
fig2.savefig(OUT_DIR / "stats_table.png", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Sauvé : {OUT_DIR / 'stats_table.png'}", flush=True)

print("\nTerminé.", flush=True)
