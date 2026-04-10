"""
Génère deux overlay charts XRP avec configs manuelles :
  - overlay_all.png    : toute la période (safeChase XRP.csv)
  - overlay_last4d.png : période récente (reportLive XRP.csv)

Une seule courbe par config (max_orders=2 fixe), scalée à maxDD=$500.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
CSV_FULL = [str(BASE / "reportLive/safeChase/XRP.csv")]
OUT_DIR = Path(__file__).resolve().parent / "custom_chart_xrp"

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from chart_utils import (
    load_contracts, precompute_vol, run_config_tf, attach_settlement_outcomes,
    build_cumulative, build_hour_index, make_config_label, make_config_name,
    chart_equity, chart_entry_and_loss_analytics,
    TIMEFRAMES, VOL_LBS, COLORS, make_xlabels, _rf_str_hourly_equity, _esc,
)

# ── Configs XRP ───────────────────────────────────────────────────────────────
# A remplir après premiers résultats optimizer
# Calibration XRP (~$1.30) : slope*60 = thresh à 60s
# Ex: slope=0.0001 → thresh=$0.006 à remain=60s (~0.46% du spot)
BASE_CONFIGS = [
    # rank 1 RF — vol_trend_lin sl=0.0001 cap=0.55 trend1h>0.70
    dict(label='vol_trend_lin', curve='linear',
         slope=0.0001, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vol_lb_h=1.0, vol_thresh=0.70, vol_type='trend'),

    # rank 6 RF — vol_trend_lin sl=0.0001 cap=0.65 trend1h>0.50
    dict(label='vol_trend_lin', curve='linear',
         slope=0.0001, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=1.0, vol_thresh=0.50, vol_type='trend'),

    # rank 1 PnL — vol_pctr_lin sl=0.00001 cap=0.55 pct_range1h>0.30%
    dict(label='vol_pctr_lin', curve='linear',
         slope=0.00001, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vol_lb_h=1.0, vol_thresh=0.30, vol_type='pct_range'),

    # rank 1 consistance — vol_net_lin sl=0.00003 cap=0.55 net1h>$0.004
    dict(label='vol_net_lin', curve='linear',
         slope=0.00003, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vol_lb_h=1.0, vol_thresh=0.004, vol_type='net'),
]

ORDER_VARIANTS = [
    (2, 'solid',  1.5, 1.0),   # (max_orders, linestyle, linewidth, alpha)
    (5, 'dashed', 1.0, 0.7),
]
MAX_DD_TARGET = 500.0
MAX_SCALE     = 5.0

ASSET = 'xrp'

# ── Helpers ───────────────────────────────────────────────────────────────────
TF_NAMES = ['M5', 'M15', 'H1']

def run_base_config(contracts_by_tf, cfg, hour_index, n_hours):
    combined_pnl = defaultdict(float)
    tf_pnls = [defaultdict(float) for _ in contracts_by_tf]
    wins = losses = 0
    for i_tf, contracts in enumerate(contracts_by_tf):
        ph, w, l = run_config_tf(contracts, cfg, hour_index)
        for h_idx, pnl in ph.items():
            combined_pnl[h_idx]  += pnl
            tf_pnls[i_tf][h_idx] += pnl
        wins += w; losses += l
    cum, _ = build_cumulative(combined_pnl, n_hours)
    cum_by_tf = {}
    for i_tf, tf_name in enumerate(TF_NAMES[:len(contracts_by_tf)]):
        c, _ = build_cumulative(tf_pnls[i_tf], n_hours)
        cum_by_tf[tf_name] = c
    return cum, wins, losses, cum_by_tf


def compute_maxdd(cum):
    return float((np.maximum.accumulate(cum) - cum).max())


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
        dd_arr = np.maximum.accumulate(cum) - cum
        n_trades = wins + losses
        lbl = _esc(f"{s['label']}  T={n_trades}({n_trades/days:.1f}/j)  WR={wr:.0f}%  ${cum[-1]/days:.0f}/j  RF={rf_s}")
        ax1.plot(x, cum, color=s['color'], linewidth=s['linewidth'],
                 linestyle=s['linestyle'], alpha=s['alpha'], label=lbl)
        ax2.plot(x, dd_arr, color=s['color'], linewidth=s['linewidth'],
                 linestyle=s['linestyle'], alpha=s['alpha'] * 0.8)

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel(f"PnL cumulé ($) — sizing: maxDD→$500 (cap {MAX_SCALE:.0f}x base)")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8, rotation=30)

    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Heures tradées (gaps exclus)")
    ax2.invert_yaxis(); ax2.grid(alpha=0.3)
    ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Overlay -> {out_path}")


# ── Moteur principal ──────────────────────────────────────────────────────────
def run_chart(csv_paths, out_dir, title_prefix, last_days=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contracts_by_tf = []
    m5_ref = None
    for tf_floor, bid_up, bid_down, ask_up, ask_down in TIMEFRAMES:
        try:
            cts = load_contracts(csv_paths, tf_floor, bid_up, bid_down, ask_up, ask_down)
            if last_days is not None and cts:
                cutoff = cts[-1]['ce'] - np.timedelta64(int(last_days * 24 * 3600), 's')
                cts = [c for c in cts if c['ce'] >= cutoff]
            if tf_floor == '5min':
                precompute_vol(cts, VOL_LBS)
                m5_ref = cts
            else:
                precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
            contracts_by_tf.append(cts)
            print(f"  {tf_floor}: {len(cts)} contracts", flush=True)
        except Exception as e:
            print(f"  {tf_floor}: skipped ({e})", flush=True)
            contracts_by_tf.append([])

    for (tf_floor, *_), cts in zip(TIMEFRAMES, contracts_by_tf):
        if cts:
            attach_settlement_outcomes(cts, ASSET, tf=tf_floor)

    hour_index, n_hours = build_hour_index(contracts_by_tf)
    days = n_hours / 24
    print(f"  {n_hours} heures tradées ({days:.1f}j)", flush=True)

    series_list = []

    for i, base_cfg in enumerate(BASE_CONFIGS):
        color = COLORS[i % len(COLORS)]
        base_label = make_config_label(base_cfg)
        print(f"\n  [{i+1}/{len(BASE_CONFIGS)}] {base_label}", flush=True)

        for max_ord, lstyle, lw, alpha in ORDER_VARIANTS:
            cfg_var = {**base_cfg, 'max_orders': max_ord}
            cum_raw, wins, losses, cum_by_tf_raw = run_base_config(
                contracts_by_tf, cfg_var, hour_index, n_hours)
            maxdd_raw = compute_maxdd(cum_raw)
            scale = min(MAX_DD_TARGET / maxdd_raw, MAX_SCALE) if maxdd_raw > 1.0 else 1.0
            cum_scaled = cum_raw * scale
            cum_by_tf_scaled = {tf: c * scale for tf, c in cum_by_tf_raw.items()}

            var_label = f"{base_label} ord={max_ord} [x{scale:.1f}]"
            wr = wins / (wins + losses) * 100 if (wins + losses) else 0
            maxdd_scaled = compute_maxdd(cum_scaled)
            print(f"  ord={max_ord}  WR={wr:.1f}%  scale={scale:.2f}x  "
                  f"PnL/j=${cum_scaled[-1]/days:.0f}  maxDD=${maxdd_scaled:.0f}",
                  flush=True)

            # Equity chart + analytics uniquement pour max_orders=2
            if max_ord == 2:
                cfg_dir = out_dir / make_config_name(base_cfg)
                cfg_dir.mkdir(parents=True, exist_ok=True)
                hourly_scaled = np.diff(np.concatenate([[0.0], cum_scaled]))
                chart_equity(cum_scaled, hourly_scaled, var_label, n_hours, hour_index,
                             wins, losses, cfg_dir / "equity.png",
                             cum_by_tf=cum_by_tf_scaled)
                try:
                    chart_entry_and_loss_analytics(
                        contracts_by_tf, cfg_var, hour_index, cfg_dir, var_label)
                except Exception as e:
                    print(f"  analytics skip ({e})", flush=True)

            series_list.append(dict(
                name=f"cfg{i}_ord{max_ord}",
                label=var_label,
                cum=cum_scaled,
                wins=wins, losses=losses,
                color=color,
                linestyle=lstyle, linewidth=lw, alpha=alpha,
            ))

    custom_overlay(
        series_list, n_hours, hour_index,
        out_dir / "overlay_all.png",
        title=f"{title_prefix}  (M5+M15+H1, ord=2solid/5dash, sizing maxDD→$500 cap {MAX_SCALE:.0f}x)",
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== XRP Chart 1 : toute la période ===")
    run_chart(CSV_FULL, OUT_DIR / "all", "Custom XRP - Full")

    print("\n=== XRP Chart 2 : 4 derniers jours ===")
    run_chart(CSV_FULL, OUT_DIR / "last4d", "Custom XRP - Last 4d", last_days=4)

    print(f"\nOverlay full   -> {OUT_DIR / 'all' / 'overlay_all.png'}")
    print(f"Overlay last4d -> {OUT_DIR / 'last4d' / 'overlay_all.png'}")
