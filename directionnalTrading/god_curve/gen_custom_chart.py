"""
Génère deux overlay charts avec des configs manuelles (sans relancer l'optimizer) :
  - overlay_all.png   : toute la période
  - overlay_last4d.png: les 4 derniers jours uniquement

Une seule courbe par config (pas de variantes max fill). Par défaut max_orders=2
(nombre max de fills par contrat). Options CLI : --max-orders N ; -4d / --last-4d ; --day YYYY-MM-DD
(backtest d'un seul jour UTC, sortie dans custom_chart/day_YYYY-MM-DD/).

Sizing : scale = $500 / maxDD (sur ce run unique).
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
CSV_FULL   = [str(BASE / "Datas/csv/BTC.csv"),
              str(BASE / "reportLive/safeChase/BTC.csv")]
CSV_RECENT = [str(BASE / "reportLive/safeChase/BTC.csv")]
OUT_DIR = Path(__file__).resolve().parent / "custom_chart"

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

from chart_utils import (
    load_contracts, precompute_vol, attach_cnet_data, run_config_tf,
    build_cumulative, build_hour_index, make_config_label, make_config_name,
    chart_equity, chart_entry_and_loss_analytics, attach_settlement_outcomes,
    TIMEFRAMES, VOL_LBS, COLORS, make_xlabels, _rf_str_hourly_equity, _esc,
)
from gen_custom_chart_common import utc_day_bounds, filter_contracts_by_utc_day

DEFAULT_MAX_ORDERS = 2

# ── Base configs (max_orders injecté au run, défaut DEFAULT_MAX_ORDERS) ───────
BASE_CONFIGS = [
    dict(label='vol_net_lin', curve='linear',
         slope=7.0, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=1, vol_thresh=60.0, vol_type='net'),

    dict(label='vrs_lin', curve='linear',
         slope=10.0, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vol_lb_h=None, vol_thresh=None,
         vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
         vrs_g_min=0.5, vrs_g_max=1.5),

    # god_curve_btc_v6_20260404_172554 (top lignes demandees)
    dict(label='vol_net_lin', curve='linear',
         slope=3.0, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=0.25, vol_thresh=40.0, vol_type='net'),

    # # linear sl=3 int=30 cap=0.85
    # dict(label='linear', curve='linear',
    #      slope=3.0, intercept=30.0,
    #      A_exp=None, tau=None,
    #      eq_cap=0.85, max_losses_cb=None,
    #      vol_lb_h=None, vol_thresh=None),

    # vol_linear sl=2 int=0 cap=0.65 ran0.5h>60(ancienne baseline)
    dict(label='vol_linear', curve='linear',
         slope=2.0, intercept=0.0,
         A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=0.5, vol_thresh=60.0, vol_type='range'),



    # # linear sl=2 int=15 cap=0.85
    # dict(label='linear', curve='linear',
    #      slope=2.0, intercept=15.0,
    #      A_exp=None, tau=None,
    #      eq_cap=0.85, max_losses_cb=None,
    #      vol_lb_h=None, vol_thresh=None),
]

# (linestyle, linewidth, alpha) — une courbe par config
OVERLAY_LINE_STYLE = ('solid', 1.6, 1.0)
MAX_DD_TARGET = 500.0   # $ cible pour le sizing dynamique


# ── Helpers ───────────────────────────────────────────────────────────────────
TF_NAMES = ['M5', 'M15', 'H1']

def run_base_config(contracts_by_tf, cfg, hour_index, n_hours):
    """Retourne (cum, wins, losses, cum_by_tf) pour un cfg donné."""
    combined_pnl = defaultdict(float)
    tf_pnls = [defaultdict(float) for _ in contracts_by_tf]
    wins = losses = 0
    for i_tf, contracts in enumerate(contracts_by_tf):
        ph, w, l = run_config_tf(contracts, cfg, hour_index)
        for h_idx, pnl in ph.items():
            combined_pnl[h_idx]   += pnl
            tf_pnls[i_tf][h_idx]  += pnl
        wins += w; losses += l
    cum, _ = build_cumulative(combined_pnl, n_hours)
    cum_by_tf = {}
    for i_tf, tf_name in enumerate(TF_NAMES[:len(contracts_by_tf)]):
        c, _ = build_cumulative(tf_pnls[i_tf], n_hours)
        cum_by_tf[tf_name] = c
    return cum, wins, losses, cum_by_tf


def compute_maxdd(cum):
    dd = (np.maximum.accumulate(cum) - cum).max()
    return float(dd)


def custom_overlay(series_list, n_hours, hour_index, out_path, title):
    """
    series_list : liste de dicts avec clés :
        name, label, cum, wins, losses, color, linestyle, linewidth, alpha
    """
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)
    days = n_hours / 24

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.4)

    for s in series_list:
        cum = s['cum']
        wins = s['wins']; losses = s['losses']
        wr  = wins / (wins + losses) * 100 if (wins + losses) else 0
        rf_s = _rf_str_hourly_equity(cum)
        dd_arr = np.maximum.accumulate(cum) - cum
        lbl = _esc(f"{s['label']}  WR={wr:.0f}%  ${cum[-1]/days:.0f}/j  RF={rf_s}")
        ax1.plot(x, cum, color=s['color'], linewidth=s['linewidth'],
                 linestyle=s['linestyle'], alpha=s['alpha'], label=lbl)
        ax2.plot(x, dd_arr, color=s['color'], linewidth=s['linewidth'],
                 linestyle=s['linestyle'], alpha=s['alpha'] * 0.8)

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title(title, fontsize=12)
    ax1.set_ylabel("PnL cumulé ($) — sizing: maxDD baseline → $500")
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
def run_chart(csv_paths, out_dir, title_prefix, max_orders=None, day=None):
    if max_orders is None:
        max_orders = DEFAULT_MAX_ORDERS
    max_orders = max(1, int(max_orders))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    day_bounds = utc_day_bounds(day) if day else None

    # Charger contrats — M5 sert de référence vol pour M15 et H1
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
            attach_settlement_outcomes(cts, 'btc', tf=tf_floor)
            attach_cnet_data(cts, (2, 5, 10), ref_contracts=m5_ref)

    hour_index, n_hours = build_hour_index(contracts_by_tf)
    if n_hours == 0:
        print("  Aucun contrat sur cette période — rien à tracer.", flush=True)
        return
    days = n_hours / 24
    print(f"  {n_hours} heures tradées ({days:.1f}j)  max_orders={max_orders}"
          + (f"  (jour UTC {day})" if day else ""),
          flush=True)

    series_list = []
    ls, lw, alpha = OVERLAY_LINE_STYLE

    for i, base_cfg in enumerate(BASE_CONFIGS):
        color = COLORS[i % len(COLORS)]
        base_label = make_config_label(base_cfg)
        print(f"\n  [{i+1}/{len(BASE_CONFIGS)}] {base_label}", flush=True)

        cfg_var = {**base_cfg, 'max_orders': max_orders}
        cum_raw, wins, losses, cum_by_tf_raw = run_base_config(
            contracts_by_tf, cfg_var, hour_index, n_hours)
        maxdd_raw = compute_maxdd(cum_raw)
        scale = MAX_DD_TARGET / maxdd_raw if maxdd_raw > 1.0 else 1.0
        cum_scaled = cum_raw * scale
        cum_by_tf_scaled = {tf: c * scale for tf, c in cum_by_tf_raw.items()}

        var_label = f"{base_label} [x{scale:.1f} ord={max_orders}]"
        wr = wins / (wins + losses) * 100 if (wins + losses) else 0
        print(f"    WR={wr:.1f}%  scale={scale:.2f}x  PnL/j=${cum_scaled[-1]/days:.0f}  maxDD→${MAX_DD_TARGET:.0f}",
              flush=True)

        cfg_dir = out_dir / f"{make_config_name(base_cfg)}_ord{max_orders}"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        hourly_scaled = np.diff(np.concatenate([[0.0], cum_scaled]))
        chart_equity(cum_scaled, hourly_scaled, var_label, n_hours, hour_index,
                     wins, losses, cfg_dir / "equity.png",
                     cum_by_tf=cum_by_tf_scaled)

        try:
            chart_entry_and_loss_analytics(
                contracts_by_tf, cfg_var, hour_index, cfg_dir, var_label)
        except Exception as e:
            print(f"    analytics skip ({e})", flush=True)

        series_list.append(dict(
            name=f"cfg{i}_ord{max_orders}",
            label=var_label,
            cum=cum_scaled,
            wins=wins, losses=losses,
            color=color,
            linestyle=ls, linewidth=lw, alpha=alpha,
        ))

    custom_overlay(
        series_list, n_hours, hour_index,
        out_dir / "overlay_all.png",
        title=f"{title_prefix}  (M5+M15+H1, ord={max_orders}, sizing maxDD→${MAX_DD_TARGET:.0f})",
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom god_curve charts (BTC)")
    parser.add_argument(
        "--max-orders",
        type=int,
        default=DEFAULT_MAX_ORDERS,
        help=f"Max fills par contrat (defaut: {DEFAULT_MAX_ORDERS}).",
    )
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
        help="Backtest uniquement ce jour (UTC, filtre sur ce=clôture). "
        "Sortie : custom_chart/day_YYYY-MM-DD/. Ignore all/last4d.",
    )
    args = parser.parse_args()
    mo = max(1, int(args.max_orders))

    if args.day:
        d = args.day.strip()
        sub = OUT_DIR / f"day_{d}"
        print(f"\n=== Jour UTC {d} (CSV full filtré) ===", flush=True)
        run_chart(CSV_FULL, sub, f"Custom BTC - {d}", max_orders=mo, day=d)
        print(f"\nOverlay -> {sub / 'overlay_all.png'}")
    else:
        if not args.last_4d_only:
            print("\n=== Chart 1 : toute la période ===")
            run_chart(CSV_FULL, OUT_DIR / "all", "Custom BTC - Full", max_orders=mo)

        print("\n=== Chart 2 : 4 derniers jours ===")
        run_chart(CSV_RECENT, OUT_DIR / "last4d", "Custom BTC - Last 4 days", max_orders=mo)

        if not args.last_4d_only:
            print(f"\nOverlay full   -> {OUT_DIR / 'all' / 'overlay_all.png'}")
        print(f"Overlay last4d -> {OUT_DIR / 'last4d' / 'overlay_all.png'}")
