"""
gen_custom_chart_alts.py — Charts manuels XRP + SOL + BNB

Une courbe par config (max_orders=2 fixe), scalée à maxDD=$500.

Chaque chart par stratégie contient :
  - Courbe equity totale (toutes cryptos, tous TF)
  - Courbes par TF (M5 / M15 / H1) en sous-tracé
  - Courbes par crypto (XRP / SOL / BNB) en sous-tracé
  - Hourly bars + drawdown

Overlay : toutes les configs sur un seul graphe.

CLI : --day YYYY-MM-DD → alts_custom_charts/day_YYYY-MM-DD/ (un jour UTC, ce).
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import (
    load_contracts, precompute_vol, run_config_tf, attach_settlement_outcomes,
    attach_cnet_data, build_cumulative, build_hour_index,
    make_config_label, make_config_name,
    chart_entry_and_loss_analytics,
    TIMEFRAMES, VOL_LBS, COLORS,
    make_xlabels, _rf_str_hourly_equity, _esc,
)
from gen_custom_chart_common import utc_day_bounds, filter_contracts_by_utc_day

# ── Sources CSV ───────────────────────────────────────────────────────────────
ASSETS = ['xrp', 'sol', 'bnb']
ASSET_CSV = {
    'xrp': [str(BASE / "reportLive" / "safeChase" / "XRP.csv")],
    'sol': [str(BASE / "Datas" / "csv" / "SOL.csv"),
            str(BASE / "reportLive" / "safeChase" / "SOL.csv")],
    'bnb': [str(BASE / "reportLive" / "safeChase" / "BNB.csv")],
}

ASSET_COLORS = {
    'xrp': '#2196F3',
    'sol': '#9C27B0',
    'bnb': '#FF9800',
}

OUT_DIR = BASE / "god_curve" / "alts_custom_charts"

# ── Configs manuelles ─────────────────────────────────────────────────────────
# Remplir après avoir lancé l'optimizer. Exemples :
BASE_CONFIGS = [
    # Pas de filtre vol
    dict(label='linear', curve='linear',
         slope=0.3, intercept=0.0,
         eq_cap=0.85, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None),

    # Trend 1h > 0.30 (filtre relatif, cross-asset)
    dict(label='vol_trend_lin', curve='linear',
         slope=0.2, intercept=0.0,
         eq_cap=0.85, max_losses_cb=None,
         vol_lb_h=1.0, vol_thresh=0.30, vol_type='trend'),

    # Trend 2h > 0.30
    dict(label='vol_trend_lin', curve='linear',
         slope=0.2, intercept=0.0,
         eq_cap=0.70, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=0.30, vol_type='trend'),

    # Pct range 1h > 0.5%
    dict(label='vol_pctr_lin', curve='linear',
         slope=0.3, intercept=0.0,
         eq_cap=0.85, max_losses_cb=None,
         vol_lb_h=1.0, vol_thresh=0.50, vol_type='pct_range'),

    # Pct range 2h > 0.8%
    dict(label='vol_pctr_lin', curve='linear',
         slope=0.2, intercept=0.0,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=0.80, vol_type='pct_range'),
]

MAX_ORDERS    = 2
MAX_DD_TARGET = 500.0
MAX_SCALE     = 5.0

TF_NAMES   = ['M5', 'M15', 'H1']
TF_COLORS  = {'M5': '#f44336', 'M15': '#FF9800', 'H1': '#4CAF50'}


# ── Helpers ───────────────────────────────────────────────────────────────────
def compute_maxdd(cum):
    return float((np.maximum.accumulate(cum) - cum).max())


def run_config_asset_tf(contracts_by_asset_tf, cfg, hour_index, n_hours):
    """
    Retourne (cum_total, cum_by_tf, cum_by_asset, wins, losses).
    contracts_by_asset_tf[asset][tf_idx] = liste de contrats
    """
    combined_pnl    = defaultdict(float)
    tf_pnls         = [defaultdict(float) for _ in TF_NAMES]
    asset_pnls      = {a: defaultdict(float) for a in ASSETS}
    wins = losses   = 0

    for asset in ASSETS:
        for tf_idx, contracts in enumerate(contracts_by_asset_tf[asset]):
            ph, w, l = run_config_tf(contracts, cfg, hour_index)
            for h_idx, pnl in ph.items():
                combined_pnl[h_idx]          += pnl
                tf_pnls[tf_idx][h_idx]       += pnl
                asset_pnls[asset][h_idx]     += pnl
            wins += w; losses += l

    cum_total, _ = build_cumulative(combined_pnl, n_hours)
    cum_by_tf    = {}
    for i, tf_name in enumerate(TF_NAMES):
        c, _ = build_cumulative(tf_pnls[i], n_hours)
        cum_by_tf[tf_name] = c
    cum_by_asset = {}
    for asset in ASSETS:
        c, _ = build_cumulative(asset_pnls[asset], n_hours)
        cum_by_asset[asset] = c

    return cum_total, cum_by_tf, cum_by_asset, wins, losses


# ── Chart individuel avec courbes par TF et par crypto ────────────────────────
def chart_equity_alts(cum, cfg_label, n_hours, hour_index,
                      wins, losses, out_path,
                      cum_by_tf=None, cum_by_asset=None):
    days = n_hours / 24
    x    = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)
    rf_s = _rf_str_hourly_equity(cum)
    wr   = wins / (wins + losses) * 100 if (wins + losses) else 0
    n_trades = wins + losses

    # Calculer hourly et drawdown depuis cum
    hourly = np.diff(np.concatenate([[0.0], cum]))
    dd     = np.maximum.accumulate(cum) - cum

    n_rows = 5  # total, par-TF, par-asset, hourly, drawdown
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 18),
                             gridspec_kw={"height_ratios": [2.5, 1.5, 1.5, 1, 1]})
    fig.subplots_adjust(hspace=0.5)

    # ── Total ──
    axes[0].plot(x, cum, color='#2196F3', linewidth=2.0, label='Total')
    axes[0].fill_between(x, cum, 0, where=cum >= 0, color='#2196F3', alpha=0.07)
    axes[0].fill_between(x, cum, 0, where=cum < 0,  color='#f44336', alpha=0.10)
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_title(
        _esc(f"{cfg_label}  (XRP+SOL+BNB, M5+M15+H1)  —  "
             f"{n_trades} trades ({n_trades/days:.1f}/j)  "
             f"WR={wr:.1f}%  ${cum[-1]/days:.0f}/j  RF={rf_s}"),
        fontsize=10)
    axes[0].set_ylabel("PnL cumulé ($)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(xticks); axes[0].set_xticklabels(xlabels, fontsize=8, rotation=30)
    axes[0].text(0.99, 0.03,
                 _esc(f"PnL total: ${cum[-1]:+.0f}\nMaxDD: ${dd.max():.0f}\nRF: {rf_s}"),
                 transform=axes[0].transAxes, ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='gray'))

    # ── Par TF ──
    if cum_by_tf:
        for tf_name, tf_cum in cum_by_tf.items():
            col = TF_COLORS.get(tf_name, '#888888')
            rf_tf = _rf_str_hourly_equity(tf_cum)
            axes[1].plot(x, tf_cum, color=col, linewidth=1.4,
                         label=_esc(f"{tf_name}  ${tf_cum[-1]:+.0f}  RF={rf_tf}"))
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].set_ylabel("PnL par TF ($)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(xticks); axes[1].set_xticklabels(xlabels, fontsize=8, rotation=30)

    # ── Par crypto ──
    if cum_by_asset:
        for asset, a_cum in cum_by_asset.items():
            col = ASSET_COLORS.get(asset, '#888888')
            rf_a = _rf_str_hourly_equity(a_cum)
            axes[2].plot(x, a_cum, color=col, linewidth=1.4,
                         label=_esc(f"{asset.upper()}  ${a_cum[-1]:+.0f}  RF={rf_a}"))
    axes[2].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[2].set_ylabel("PnL par crypto ($)")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    axes[2].set_xticks(xticks); axes[2].set_xticklabels(xlabels, fontsize=8, rotation=30)

    # ── Hourly ──
    bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in hourly]
    axes[3].bar(x, hourly, color=bar_colors, width=1.0, alpha=0.85)
    axes[3].axhline(0, color='gray', linewidth=0.5)
    axes[3].set_ylabel("PnL/heure ($)")
    axes[3].grid(alpha=0.3, axis='y')
    axes[3].set_xticks(xticks); axes[3].set_xticklabels(xlabels, fontsize=8, rotation=30)

    # ── Drawdown ──
    axes[4].fill_between(x, dd, 0, color='#f44336', alpha=0.4)
    axes[4].plot(x, dd, color='#f44336', linewidth=1)
    axes[4].set_ylabel("Drawdown ($)")
    axes[4].set_xlabel("Heures tradées (gaps exclus)")
    axes[4].grid(alpha=0.3); axes[4].invert_yaxis()
    axes[4].set_xticks(xticks); axes[4].set_xticklabels(xlabels, fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')


# ── Overlay ───────────────────────────────────────────────────────────────────
def custom_overlay(series_list, n_hours, hour_index, out_path, title):
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)
    days = n_hours / 24

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                   gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.4)

    for s in series_list:
        cum = s['cum']
        wins = s['wins']; losses = s['losses']
        wr   = wins / (wins + losses) * 100 if (wins + losses) else 0
        rf_s = _rf_str_hourly_equity(cum)
        n_trades = wins + losses
        dd_arr = np.maximum.accumulate(cum) - cum
        lbl = _esc(f"{s['label']}  T={n_trades}({n_trades/days:.1f}/j)  "
                   f"WR={wr:.0f}%  ${cum[-1]/days:.0f}/j  RF={rf_s}")
        ax1.plot(x, cum, color=s['color'], linewidth=s['linewidth'],
                 linestyle=s['linestyle'], alpha=s['alpha'], label=lbl)
        ax2.plot(x, dd_arr, color=s['color'], linewidth=s['linewidth'],
                 linestyle=s['linestyle'], alpha=s['alpha'] * 0.8)

    ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel(f"PnL cumulé ($) — sizing: maxDD→$500 (cap {MAX_SCALE:.0f}x base)")
    ax1.legend(loc='upper left', fontsize=7)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8, rotation=30)

    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Heures tradées (gaps exclus)")
    ax2.invert_yaxis(); ax2.grid(alpha=0.3)
    ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Overlay -> {out_path}")


# ── Moteur principal ──────────────────────────────────────────────────────────
def run_chart(out_dir, title_prefix, day=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day_bounds = utc_day_bounds(day) if day else None

    # Chargement par asset par TF
    contracts_by_asset_tf = {}
    m5_ref_by_asset = {}
    all_cts_flat = []

    for asset in ASSETS:
        csv_paths = [p for p in ASSET_CSV[asset] if Path(p).exists()]
        if not csv_paths:
            print(f"  [{asset.upper()}] CSV manquant, skip")
            contracts_by_asset_tf[asset] = [[] for _ in TIMEFRAMES]
            continue

        asset_contracts = []
        m5_ref = None
        for tf_floor, bid_up, bid_down, ask_up, ask_down in TIMEFRAMES:
            try:
                cts = load_contracts(csv_paths, tf_floor, bid_up, bid_down, ask_up, ask_down)
                if day_bounds:
                    cts = filter_contracts_by_utc_day(cts, day_bounds[0], day_bounds[1])
                if tf_floor == '5min':
                    precompute_vol(cts, VOL_LBS)
                    m5_ref = cts
                else:
                    precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
                attach_settlement_outcomes(cts, asset, tf=tf_floor)
                asset_contracts.append(cts)
                all_cts_flat.extend(cts)
                print(f"  [{asset.upper()}] {tf_floor}: {len(cts)} contrats", flush=True)
            except Exception as e:
                print(f"  [{asset.upper()}] {tf_floor}: skip ({e})")
                asset_contracts.append([])

        contracts_by_asset_tf[asset] = asset_contracts
        m5_ref_by_asset[asset] = m5_ref

    # hour_index global (union de tous les assets)
    hour_index, n_hours = build_hour_index([all_cts_flat])
    if n_hours == 0:
        print("\n  Aucun contrat sur cette période — rien à tracer.", flush=True)
        return
    days = n_hours / 24
    print(f"\n  {n_hours} heures (gaps exclus)  {days:.1f}j"
          + (f"  (jour UTC {day})" if day else "") + "\n",
          flush=True)

    series_list = []

    for i, base_cfg in enumerate(BASE_CONFIGS):
        color = COLORS[i % len(COLORS)]
        base_label = make_config_label(base_cfg)
        print(f"  [{i+1}/{len(BASE_CONFIGS)}] {base_label}", flush=True)

        cfg_var = {**base_cfg, 'max_orders': MAX_ORDERS}
        cum_raw, cum_by_tf_raw, cum_by_asset_raw, wins, losses = run_config_asset_tf(
            contracts_by_asset_tf, cfg_var, hour_index, n_hours)

        maxdd_raw = compute_maxdd(cum_raw)
        scale     = min(MAX_DD_TARGET / maxdd_raw, MAX_SCALE) if maxdd_raw > 1.0 else 1.0
        cum_scaled         = cum_raw * scale
        cum_by_tf_scaled   = {k: v * scale for k, v in cum_by_tf_raw.items()}
        cum_by_asset_scaled = {k: v * scale for k, v in cum_by_asset_raw.items()}

        n_trades = wins + losses
        wr       = wins / n_trades * 100 if n_trades else 0
        maxdd_s  = compute_maxdd(cum_scaled)
        rf_s     = _rf_str_hourly_equity(cum_scaled)
        print(f"    WR={wr:.1f}%  T={n_trades}({n_trades/days:.1f}/j)  "
              f"scale={scale:.2f}x  PnL/j=${cum_scaled[-1]/days:.0f}  "
              f"maxDD=${maxdd_s:.0f}  RF={rf_s}", flush=True)

        var_label = f"{base_label} [x{scale:.1f}]"
        cfg_dir = out_dir / make_config_name(base_cfg)
        cfg_dir.mkdir(parents=True, exist_ok=True)

        chart_equity_alts(
            cum_scaled, var_label, n_hours, hour_index,
            wins, losses, cfg_dir / "equity.png",
            cum_by_tf=cum_by_tf_scaled,
            cum_by_asset=cum_by_asset_scaled,
        )

        try:
            # Analytics sur M5 de chaque asset (on passe le premier asset non-vide)
            for asset in ASSETS:
                cts_list = contracts_by_asset_tf[asset]
                if any(cts_list):
                    chart_entry_and_loss_analytics(
                        cts_list, cfg_var, hour_index,
                        cfg_dir / f"analytics_{asset}", var_label)
                    break
        except Exception as e:
            print(f"    analytics skip ({e})")

        series_list.append(dict(
            name=f"cfg{i}",
            label=var_label,
            cum=cum_scaled,
            wins=wins, losses=losses,
            color=color,
            linestyle='solid', linewidth=1.5, alpha=1.0,
        ))

    custom_overlay(
        series_list, n_hours, hour_index,
        out_dir / "overlay_all.png",
        title=f"{title_prefix}  (XRP+SOL+BNB, M5+M15+H1, ord={MAX_ORDERS}, maxDD→$500)",
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom charts XRP+SOL+BNB")
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Un jour UTC (ce). Sortie alts_custom_charts/day_*/.",
    )
    args = parser.parse_args()

    print("=== Gen Custom Chart Alts (XRP + SOL + BNB) ===\n", flush=True)
    if args.day:
        d = args.day.strip()
        run_chart(OUT_DIR / f"day_{d}", f"Custom Alts — {d}", day=d)
        print(f"\nOverlay -> {OUT_DIR / f'day_{d}' / 'overlay_all.png'}")
    else:
        run_chart(OUT_DIR / "all", "Custom Alts — Full")
    print("\nDone.")
