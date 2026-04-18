"""
Génère deux overlay charts ETH avec configs manuelles :
  - overlay_all.png    : toute la période (ETH.csv)
  - overlay_last4d.png : période récente (reportLive ETH.csv)

Une seule courbe par config (max_orders=2 fixe), scalée à maxDD=$500.
CLI : -4d / --last-4d ; --day YYYY-MM-DD (un jour UTC, dossier day_*/).
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
CSV_FULL   = [str(BASE / "Datas/csv/ETH.csv")]
CSV_RECENT = [str(BASE / "reportLive/safeChase/ETH.csv")]
OUT_DIR = Path(__file__).resolve().parent / "custom_chart_eth"

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from chart_utils import (
    load_contracts, precompute_vol, run_config_tf, attach_settlement_outcomes,
    attach_cnet_data,
    build_cumulative, build_hour_index, make_config_label, make_config_name,
    chart_equity, chart_entry_and_loss_analytics,
    TIMEFRAMES, VOL_LBS, COLORS, make_xlabels, _rf_str_hourly_equity, _esc,
)
from gen_custom_chart_common import utc_day_bounds, filter_contracts_by_utc_day

# ── Configs ETH ───────────────────────────────────────────────────────────────
BASE_CONFIGS = [
    # sl=0.2 int=0 cap=0.55 trend2h>0.5
    dict(label='vol_trend_lin', curve='linear',
         slope=0.2, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=0.5, vol_type='trend'),

    # sl=0.2 int=0 cap=0.70 trend2h>0.5
    dict(label='vol_trend_lin', curve='linear',
         slope=0.2, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.70, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=0.5, vol_type='trend'),

    # sl=0.05 int=1 cap=0.65 (pas de filtre vol)
    dict(label='linear', curve='linear',
         slope=0.05, intercept=1.0,
         A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None),

    # sl=0.3 int=0 cap=0.90 (pas de filtre vol)
    dict(label='linear', curve='linear',
         slope=0.3, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.90, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None),

    # optimizer rank 6 — vol_trend_lin sl=0.2 cap=0.85 tre2h>0.30
    dict(label='vol_trend_lin', curve='linear',
         slope=0.2, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.85, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=0.30, vol_type='trend'),

    # optimizer rank 4 — vol_net_lin sl=0.2 cap=0.55 net2h>$8
    dict(label='vol_net_lin', curve='linear',
         slope=0.2, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vol_lb_h=2.0, vol_thresh=8.0, vol_type='net'),

    # optimizer rank 3 — vrs_lin sl=0.300 int=0.1 cap=0.75 vrs1h b=8 g=0.75-1.50
    dict(label='vrs_lin', curve='linear',
         slope=0.3, intercept=0.1, intercept_mode='floor',
         A_exp=None, tau=None,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None,
         vrs_enabled=True, vrs_lb=1.0, vrs_base=8.0,
         vrs_g_min=0.75, vrs_g_max=1.50),

    # optimizer rank 11 — cnet_lin sl=0.200 int=1.0 cap=0.85 cnet2c>$1
    dict(label='cnet_lin', curve='linear',
         slope=0.2, intercept=1.0, intercept_mode='floor',
         A_exp=None, tau=None,
         eq_cap=0.85, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None,
         cnet_n=2, cnet_thresh=1.0),

    # optimizer rank 9 — cnet_lin sl=0.250 int=0.1 cap=0.65 cnet2c>$0 (thresh=0.3 ?)
    dict(label='cnet_lin', curve='linear',
         slope=0.25, intercept=0.1, intercept_mode='floor',
         A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None,
         cnet_n=2, cnet_thresh=0.3),

    # optimizer rank 9 — cnet_lin sl=0.250 int=0.1 cap=0.65 cnet2c>$0 (thresh=0.5 ?)
    dict(label='cnet_lin', curve='linear',
         slope=0.25, intercept=0.1, intercept_mode='floor',
         A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None,
         cnet_n=2, cnet_thresh=0.5),
]

MAX_ORDERS    = 2
MAX_DD_TARGET = 500.0
MAX_SCALE     = 5.0   # trade size plafonnée à 5x base ($50 → $250 max)


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
def run_chart(csv_paths, out_dir, title_prefix, day=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day_bounds = utc_day_bounds(day) if day else None

    contracts_by_tf = []
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
            contracts_by_tf.append(cts)
            print(f"  {tf_floor}: {len(cts)} contracts", flush=True)
        except Exception as e:
            print(f"  {tf_floor}: skipped ({e})", flush=True)
            contracts_by_tf.append([])

    for (tf_floor, *_), cts in zip(TIMEFRAMES, contracts_by_tf):
        if cts:
            attach_settlement_outcomes(cts, 'eth', tf=tf_floor)
            attach_cnet_data(cts, (2, 5, 10), ref_contracts=m5_ref)

    hour_index, n_hours = build_hour_index(contracts_by_tf)
    if n_hours == 0:
        print("  Aucun contrat sur cette période — rien à tracer.", flush=True)
        return
    days = n_hours / 24
    print(f"  {n_hours} heures tradées ({days:.1f}j)" + (f"  (jour UTC {day})" if day else ""), flush=True)

    series_list = []

    for i, base_cfg in enumerate(BASE_CONFIGS):
        color = COLORS[i % len(COLORS)]
        base_label = make_config_label(base_cfg)
        print(f"\n  [{i+1}/{len(BASE_CONFIGS)}] {base_label}", flush=True)

        cfg_var = {**base_cfg, 'max_orders': MAX_ORDERS}
        cum_raw, wins, losses, cum_by_tf_raw = run_base_config(
            contracts_by_tf, cfg_var, hour_index, n_hours)
        maxdd_raw = compute_maxdd(cum_raw)
        scale = min(MAX_DD_TARGET / maxdd_raw, MAX_SCALE) if maxdd_raw > 1.0 else 1.0
        cum_scaled = cum_raw * scale
        cum_by_tf_scaled = {tf: c * scale for tf, c in cum_by_tf_raw.items()}

        var_label = f"{base_label} [x{scale:.1f}]"
        wr = wins / (wins + losses) * 100 if (wins + losses) else 0
        maxdd_scaled = compute_maxdd(cum_scaled)
        print(f"  WR={wr:.1f}%  scale={scale:.2f}x  "
              f"PnL/j=${cum_scaled[-1]/days:.0f}  maxDD=${maxdd_scaled:.0f}",
              flush=True)

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
        title=f"{title_prefix}  (M5+M15+H1, ord={MAX_ORDERS}, sizing maxDD→$500 cap {MAX_SCALE:.0f}x)",
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom god_curve charts (ETH)")
    parser.add_argument(
        "-4d",
        "--last-4d",
        action="store_true",
        dest="last_4d_only",
        help="Générer uniquement la chart last4d (CSV récent), pas la période full.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Un jour UTC (clôture contrat). Sortie custom_chart_eth/day_*/.",
    )
    args = parser.parse_args()

    if args.day:
        d = args.day.strip()
        sub = OUT_DIR / f"day_{d}"
        print(f"\n=== ETH jour UTC {d} ===", flush=True)
        run_chart(CSV_FULL, sub, f"Custom ETH - {d}", day=d)
        print(f"\nOverlay -> {sub / 'overlay_all.png'}")
    else:
        if not args.last_4d_only:
            print("\n=== ETH Chart 1 : toute la période ===")
            run_chart(CSV_FULL, OUT_DIR / "all", "Custom ETH - Full")

        print("\n=== ETH Chart 2 : période récente ===")
        run_chart(CSV_RECENT, OUT_DIR / "last4d", "Custom ETH - Recent")

        if not args.last_4d_only:
            print(f"\nOverlay full   -> {OUT_DIR / 'all' / 'overlay_all.png'}")
        print(f"Overlay recent -> {OUT_DIR / 'last4d' / 'overlay_all.png'}")
