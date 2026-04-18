"""
Sweep de min_delta_usd pour ETH, SOL, XRP.
Baselines depuis baselines.txt.
Settlement-only (pessimiste). Deux CSV par asset.
Sorties : god_curve/min_delta_sweep/{asset}_equity_curves.png + {asset}_stats_table.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "min_delta_sweep"
OUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_utils as cu
from chart_utils import (
    load_contracts, precompute_vol,
    build_cumulative, build_hour_index, make_xlabels,
    attach_settlement_outcomes, run_config_tf,
)

# ── Configs par asset ─────────────────────────────────────────────────────────
ASSET_CONFIGS = {
    'ETH': dict(
        csv_files=[
            str(BASE / "Datas/csv/ETH.csv"),
            str(BASE / "reportLive/safeChase/ETH.csv"),
        ],
        base_cfg=dict(
            curve='linear', slope=0.2, intercept=0.0,
            eq_cap=0.85, max_losses_cb=None, max_orders=2,
            vol_lb_h=2.0, vol_thresh=0.3, vol_type='trend',
        ),
        # min_delta à tester — ETH ~2000$, slope*60s = 12$
        min_deltas=[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
    ),
    'SOL': dict(
        csv_files=[
            str(BASE / "Datas/csv/SOL.csv"),
            str(BASE / "reportLive/safeChase/SOL.csv"),
        ],
        base_cfg=dict(
            curve='linear', slope=0.005, intercept=0.0,
            eq_cap=0.75, max_losses_cb=None, max_orders=3,
            vol_lb_h=1.0, vol_thresh=0.50, vol_type='pct_range',
        ),
        # SOL ~120$, slope*60s = 0.3$
        min_deltas=[0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
    ),
    'XRP': dict(
        csv_files=[
            str(BASE / "Datas/csv/XRP.csv"),
            str(BASE / "reportLive/safeChase/XRP.csv"),
        ],
        base_cfg=dict(
            curve='linear', slope=0.0001, intercept=0.0,
            eq_cap=0.65, max_losses_cb=None, max_orders=3,
            vol_lb_h=2.0, vol_thresh=0.50, vol_type='trend',
        ),
        # XRP ~2$, slope*60s = 0.006$
        min_deltas=[0, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005],
    ),
}

COLORS_SWEEP = [
    '#888888',
    '#4fc3f7',
    '#29b6f6',
    '#0288d1',
    '#f4a261',
    '#e76f51',
    '#d62828',
]

# ── Boucle par asset ──────────────────────────────────────────────────────────
for asset, acfg in ASSET_CONFIGS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"  {asset}", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"Chargement CSV {asset}...", flush=True)
    contracts = load_contracts(
        acfg['csv_files'], '5min',
        'm5_up_bid', 'm5_down_bid', 'm5_up_ask', 'm5_down_ask'
    )
    print(f"  {len(contracts)} contrats bruts", flush=True)

    precompute_vol(contracts, cu.VOL_LBS)
    n_set = attach_settlement_outcomes(contracts, asset.lower(), tf='5min')
    print(f"  {n_set} contrats avec settlement connu", flush=True)

    contracts = [c for c in contracts if '_settlement_won_up' in c]
    print(f"  {len(contracts)} contrats après filtre settlement", flush=True)

    if not contracts:
        print(f"  Aucun contrat, skip.", flush=True)
        continue

    hour_index, n_hours = build_hour_index([contracts])
    x_vals = np.arange(n_hours)

    results = []
    for md in acfg['min_deltas']:
        cfg = {**acfg['base_cfg'], 'min_delta_usd': md}
        pnl_by_hour, wins, losses = run_config_tf(contracts, cfg, hour_index)

        n_trades = wins + losses
        wr = 100.0 * wins / n_trades if n_trades else 0.0
        cum, _ = build_cumulative(pnl_by_hour, n_hours)
        pnl_total = float(cum[-1]) if n_hours > 0 else 0.0

        results.append(dict(md=md, n_trades=n_trades, wins=wins,
                            losses=losses, wr=wr, pnl_total=pnl_total, cum=cum))

        print(f"  min_delta={md:<8}  trades={n_trades:4d}  wins={wins:4d}  "
              f"WR={wr:.1f}%  PnL={pnl_total:+.1f}USD", flush=True)

    # ── Equity curves ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    for i, r in enumerate(results):
        color = COLORS_SWEEP[i % len(COLORS_SWEEP)]
        lw    = 2.5 if r['md'] == 0 else 1.5
        label = (f"min={r['md']}USD  |  {r['n_trades']} trades  "
                 f"WR={r['wr']:.1f}%  PnL={r['pnl_total']:+.0f}USD")
        ax.plot(x_vals[:len(r['cum'])], r['cum'], color=color, lw=lw, alpha=0.85, label=label)

    ax.axhline(0, color='#555', lw=0.8, ls='--')
    ax.set_xlabel('Temps', color='#aaa', fontsize=10)
    ax.set_ylabel('PnL cumulé (USD)', color='#aaa', fontsize=10)
    ax.set_title(f'Sweep min_delta_usd — {asset} / slope={acfg["base_cfg"]["slope"]} / settlement-only',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#aaa')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

    step = max(1, n_hours // 12)
    tick_pos, xlabs = make_xlabels(hour_index, step=step)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(xlabs, rotation=35, ha='right', fontsize=8, color='#aaa')

    ax.legend(loc='upper left', fontsize=9, framealpha=0.25,
              facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
    plt.tight_layout()
    out_eq = OUT_DIR / f"{asset.lower()}_equity_curves.png"
    fig.savefig(out_eq, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sauvé : {out_eq}", flush=True)

    # ── Stats table ───────────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor('#0e1117')
    ax2.set_facecolor('#0e1117')
    ax2.axis('off')

    col_labels = ['min_delta', 'Trades', 'Wins', 'Losses', 'WR (%)', 'PnL (USD)']
    table_data = [[str(r['md']), str(r['n_trades']), str(r['wins']),
                   str(r['losses']), f"{r['wr']:.1f}", f"{r['pnl_total']:+.1f}"]
                  for r in results]

    tbl = ax2.table(cellText=table_data, colLabels=col_labels,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.3, 1.8)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a2e' if row % 2 == 0 else '#12122a')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333')
        if row == 0:
            cell.set_facecolor('#0d47a1')
            cell.set_text_props(color='white', fontweight='bold')
    for col in range(len(col_labels)):
        tbl[1, col].set_facecolor('#2a2a2a')

    ax2.set_title(f'Stats par min_delta_usd — {asset}', color='white', fontsize=12, pad=10)
    plt.tight_layout()
    out_tbl = OUT_DIR / f"{asset.lower()}_stats_table.png"
    fig2.savefig(out_tbl, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Sauvé : {out_tbl}", flush=True)

print("\nTerminé.", flush=True)
