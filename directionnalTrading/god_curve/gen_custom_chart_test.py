"""
Comparatif VRS statique vs VRS dynamique (recalcul toutes les 5min pour M15/H1).

Bug identifié : pour les contrats M15 et H1, _g_vrs est calculé une seule
fois à l'open du contrat (vol figée). La variante dynamique le recalcule à
chaque bucket 5min en lisant le vol M5 courant.

Config testée : vrs_lin sl=10 cap=0.55 vrs1h b=150 g=0.50-1.50 (baseline BTC live)
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
CSV_FULL   = [str(BASE / "Datas/csv/BTC.csv")]
CSV_RECENT = [str(BASE / "reportLive/safeChase/BTC.csv")]
OUT_DIR = Path(__file__).resolve().parent / "custom_chart" / "test"

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

import chart_utils as cu
from chart_utils import (
    load_contracts, precompute_vol, attach_cnet_data, run_config_tf,
    build_cumulative, build_hour_index,
    TIMEFRAMES, VOL_LBS, COLORS, make_xlabels, _rf_str_hourly_equity, _esc,
)

# ── Config testée ─────────────────────────────────────────────────────────────
CFG = dict(
    label='vrs_lin', curve='linear',
    slope=10.0, intercept=0.0, A_exp=None, tau=None,
    eq_cap=0.55, max_losses_cb=None,
    vol_lb_h=None, vol_thresh=None,
    vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
    vrs_g_min=0.5, vrs_g_max=1.5,
    max_orders=2,
)

MAX_DD_TARGET = 500.0
TF_NAMES = ['M5', 'M15', 'H1']


# ── Vol lookup depuis contrats M5 ─────────────────────────────────────────────
def build_vol_lookup(m5_ref, lb_h):
    """
    Construit {bucket_ts_300s: vol_range} depuis les contrats M5.
    bucket_ts = open_ts floored à la minute 5 la plus proche.
    Permet de retrouver la vol M5 courante à n'importe quel instant.
    """
    lookup = {}
    for c in m5_ref:
        ts = int(c.get('open_ts', c['ce_ts']))
        bucket = (ts // 300) * 300
        lookup[bucket] = c.get(f'vol_{lb_h:g}h_range', 0.0)
    return lookup


# ── Simulation VRS dynamique ──────────────────────────────────────────────────
def simulate_vrs_dynamic(c, cfg, vol_lookup):
    """
    Identique à chart_utils.simulate() mais recalcule _g_vrs à chaque
    nouveau bucket 5min (utile pour M15/H1 où la vol change en cours de contrat).
    vol_lookup : {bucket_ts_300s: vol_range} construit depuis M5 ref.
    """
    curve = cfg['curve']; cap = cfg['eq_cap']
    ce_ns = np.datetime64(c['ce'])
    spots = c['spot']; ts_arr = c['ts']; op = c['op']
    activated_side = fill_side = None
    contract_cost = contract_shares = 0.0; fill_count = 0
    active_order = active_size = pending_place = pending_size = None
    pending_cancel = False
    max_orders = int(cfg.get('max_orders', cu.MAX_ORDERS))

    lb_h    = cfg['vrs_lb']
    vrs_base = float(cfg['vrs_base'])
    g_min   = float(cfg['vrs_g_min'])
    g_max   = float(cfg['vrs_g_max'])
    is_inv  = bool(cfg.get('vrs_inv', False))

    cur_bucket = None
    _g_vrs = 1.0  # sera écrasé dès le premier bucket

    def compute_g(vol):
        if vol > 1e-6:
            ratio = vol / vrs_base if is_inv else vrs_base / vol
        else:
            ratio = g_min if is_inv else g_max
        return float(np.clip(ratio, g_min, g_max))

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or fill_count >= max_orders:
            break
        ub = round(c['up_bid'][j], 2)
        db = round(c['down_bid'][j], 2)

        # Mise à jour vol / _g_vrs à chaque nouveau bucket 5min
        ts_sec = int(ts_arr[j].astype('datetime64[s]').astype(np.int64))
        bucket = (ts_sec // 300) * 300
        if bucket != cur_bucket:
            cur_bucket = bucket
            vol = vol_lookup.get(bucket)
            if vol is None:
                # Fallback : vol attachée à l'open du contrat (comportement statique)
                vol = c.get(f'vol_{lb_h:g}h_range', 0.0)
            _g_vrs = compute_g(vol)

        # ── Logique de remplissage identique à simulate() ────────────────────
        if pending_cancel and active_order is not None:
            if activated_side:
                cb2 = ub if activated_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    fill_count += 1
                    contract_cost   += active_size
                    contract_shares += active_size / active_order
                    if fill_side is None: fill_side = activated_side
                    if fill_count >= max_orders:
                        active_order = None; break
            active_order = active_size = None; pending_cancel = False

        if pending_place is not None:
            active_order = pending_place; active_size = pending_size
            pending_place = pending_size = None

        if active_order is not None and activated_side:
            cb2 = ub if activated_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fill_count += 1
                contract_cost   += active_size
                contract_shares += active_size / active_order
                if fill_side is None: fill_side = activated_side
                active_order = active_size = None
                if fill_count >= max_orders: break

        # ── Calcul seuil avec g dynamique ────────────────────────────────────
        g = _g_vrs
        if curve == 'linear':
            thresh = (cfg['slope'] * g) * remain + cfg['intercept']
        else:
            thresh = cfg['A_exp'] * (np.exp(remain / cfg['tau']) - 1.0) * g

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = None
                activated_side = None
                continue

        if not activated_side:
            if   s - op >= thresh: activated_side = 'UP'
            elif op - s >= thresh: activated_side = 'DOWN'
        if not activated_side:
            continue

        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = cu.BASE_SIZE
            elif active_order is not None and proposed > active_order:
                pending_cancel = True; pending_place = proposed; pending_size = cu.BASE_SIZE
            elif pending_place is not None and proposed > pending_place:
                pending_place = proposed; pending_size = cu.BASE_SIZE

    if 'up_ask' in c and len(c['up_ask']):
        won_up = float(c['up_ask'][-1]) > float(c['down_ask'][-1])
    else:
        won_up = c['up_bid'][-1] > c['down_bid'][-1]

    if contract_shares > 0:
        won = (won_up == (fill_side == 'UP'))
        return won, (contract_shares if won else 0.0) - contract_cost
    return None, None


# ── Run config tf ─────────────────────────────────────────────────────────────
def run_config_tf_dynamic(contracts, cfg, hour_index, vol_lookup):
    """run_config_tf mais avec simulate_vrs_dynamic."""
    pnl_by_hour = defaultdict(float)
    wins = losses = 0
    for c in contracts:
        won, pnl = simulate_vrs_dynamic(c, cfg, vol_lookup)
        if won is None: continue
        h = c['ce'].replace(minute=0, second=0, microsecond=0)
        if h not in hour_index: continue
        pnl_by_hour[hour_index[h]] += pnl
        if won: wins += 1
        else:   losses += 1
    return pnl_by_hour, wins, losses


def run_variant(contracts_by_tf, cfg, hour_index, n_hours, dynamic=False, vol_lookup=None):
    """
    Run config sur les 3 TF.
    dynamic=True → simulate_vrs_dynamic pour M15/H1 (M5 inchangé car bucket unique).
    dynamic=False → run_config_tf standard (comportement actuel).
    """
    combined_pnl = defaultdict(float)
    tf_pnls      = [defaultdict(float) for _ in contracts_by_tf]
    wins = losses = 0

    for i_tf, contracts in enumerate(contracts_by_tf):
        if dynamic and i_tf > 0:
            # M15 / H1 : recalcul vol toutes les 5min
            ph, w, l = run_config_tf_dynamic(contracts, cfg, hour_index, vol_lookup)
        else:
            # M5 ou variante statique : comportement actuel
            ph, w, l = run_config_tf(contracts, cfg, hour_index)

        for h_idx, pnl in ph.items():
            combined_pnl[h_idx]   += pnl
            tf_pnls[i_tf][h_idx]  += pnl
        wins += w; losses += l

    cum, _ = build_cumulative(combined_pnl, n_hours)
    cum_by_tf = {}
    for i_tf, tf_name in enumerate(TF_NAMES[:len(contracts_by_tf)]):
        c_arr, _ = build_cumulative(tf_pnls[i_tf], n_hours)
        cum_by_tf[tf_name] = c_arr
    return cum, wins, losses, cum_by_tf


# ── Overlay ───────────────────────────────────────────────────────────────────
def custom_overlay(series_list, n_hours, hour_index, out_path, title):
    """
    Layout 4 panels :
      Row 0 : M15 + H1 combinés statique vs dynamique (equity)
      Row 1 : M15 seul statique vs dynamique
      Row 2 : H1 seul statique vs dynamique
      Row 3 : Delta (dynamique - statique) sur M15+H1
    M5 exclu : identique dans les deux variantes.
    """
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)
    days = n_hours / 24

    fig, axes = plt.subplots(4, 1, figsize=(16, 18),
                              gridspec_kw={"height_ratios": [3, 2, 2, 1.5]})
    ax_comb, ax_m15, ax_h1, ax_delta = axes
    fig.subplots_adjust(hspace=0.5)

    palette = {'static': '#FFC107', 'dynamic': '#2196F3'}

    for s in series_list:
        color = palette.get(s['variant'], '#888888')
        ls    = '--' if s['variant'] == 'dynamic' else '-'

        # Cumul M15+H1 (exclut M5)
        cum_m15 = s['cum_by_tf'].get('M15', np.zeros(n_hours))
        cum_h1  = s['cum_by_tf'].get('H1',  np.zeros(n_hours))
        cum_m15h1 = cum_m15 + cum_h1

        wr  = s['wins'] / (s['wins'] + s['losses']) * 100 if (s['wins'] + s['losses']) else 0
        rf_s = _rf_str_hourly_equity(cum_m15h1)
        dd   = np.maximum.accumulate(cum_m15h1) - cum_m15h1
        lbl  = _esc(f"{s['label'][:40]}  WR={wr:.1f}%  T={s['wins']+s['losses']}"
                    f"  {cum_m15h1[-1]/days:+.0f}/j [M15+H1]  maxDD={dd.max():.0f}")
        ax_comb.plot(x, cum_m15h1, color=color, linewidth=1.8, linestyle=ls, label=lbl)

        # M15
        dd_m15 = np.maximum.accumulate(cum_m15) - cum_m15
        lbl_m15 = _esc(f"{s['variant']}  {cum_m15[-1]/days:+.0f}/j  maxDD={dd_m15.max():.0f}")
        ax_m15.plot(x, cum_m15, color=color, linewidth=1.4, linestyle=ls, label=lbl_m15)

        # H1
        dd_h1 = np.maximum.accumulate(cum_h1) - cum_h1
        lbl_h1 = _esc(f"{s['variant']}  {cum_h1[-1]/days:+.0f}/j  maxDD={dd_h1.max():.0f}")
        ax_h1.plot(x, cum_h1, color=color, linewidth=1.4, linestyle=ls, label=lbl_h1)

    # Delta M15+H1 (dynamique - statique)
    if len(series_list) == 2:
        cum_m15h1_st = (series_list[0]['cum_by_tf'].get('M15', np.zeros(n_hours)) +
                        series_list[0]['cum_by_tf'].get('H1',  np.zeros(n_hours)))
        cum_m15h1_dy = (series_list[1]['cum_by_tf'].get('M15', np.zeros(n_hours)) +
                        series_list[1]['cum_by_tf'].get('H1',  np.zeros(n_hours)))
        diff = cum_m15h1_dy - cum_m15h1_st
        ax_delta.fill_between(x, 0, diff, where=diff >= 0, color='#4CAF50', alpha=0.6,
                              label='Dynamique > Statique')
        ax_delta.fill_between(x, 0, diff, where=diff < 0,  color='#f44336', alpha=0.6,
                              label='Statique > Dynamique')
        ax_delta.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        final = diff[-1]
        ax_delta.set_ylabel(_esc(f'Delta M15+H1 ($)  final={final:+.0f}$'))
        ax_delta.legend(loc='upper left', fontsize=8)
        ax_delta.grid(alpha=0.3)
        ax_delta.set_xticks(xticks); ax_delta.set_xticklabels(xlabels, fontsize=8, rotation=30)

    for ax, lbl in [(ax_comb, 'M15+H1 cumulé ($) — sizing maxDD→500$'),
                    (ax_m15,  'M15 cumulé ($)'),
                    (ax_h1,   'H1 cumulé ($)')]:
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel(lbl); ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.3)
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=8, rotation=30)

    ax_comb.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  Overlay -> {out_path}", flush=True)


def print_tf_breakdown(cum_by_tf, label, days):
    """Breakdown PnL par TF."""
    parts = []
    for tf in TF_NAMES:
        if tf in cum_by_tf:
            c = cum_by_tf[tf]
            dd = float((np.maximum.accumulate(c) - c).max())
            parts.append(f"{tf}: {c[-1]/days:+.0f}/j  maxDD={dd:.0f}")
    print(f"    [{label}] " + "  |  ".join(parts), flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_chart(csv_paths, out_dir, title_prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    contracts_by_tf = []
    m5_ref = None
    for tf_floor, bid_up, bid_down, ask_up, ask_down in TIMEFRAMES:
        try:
            cts = load_contracts(csv_paths, tf_floor, bid_up, bid_down, ask_up, ask_down)
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

    hour_index, n_hours = build_hour_index(contracts_by_tf)
    days = n_hours / 24
    print(f"  {n_hours}h ({days:.1f}j)\n", flush=True)

    # Lookup vol M5 pour la variante dynamique
    lb_h = CFG['vrs_lb']
    vol_lookup = build_vol_lookup(m5_ref, lb_h) if m5_ref else {}
    print(f"  vol_lookup: {len(vol_lookup)} buckets 5min\n", flush=True)

    series_list = []
    variants = [
        ('static',  False, 'VRS statique  (vol figée à l\'open du contrat — comportement actuel)'),
        ('dynamic', True,  'VRS dynamique (vol recalculée toutes les 5min pour M15/H1)'),
    ]

    for variant, dynamic, desc in variants:
        print(f"  [{variant}] {desc}", flush=True)
        cum_raw, wins, losses, cum_by_tf_raw = run_variant(
            contracts_by_tf, CFG, hour_index, n_hours,
            dynamic=dynamic, vol_lookup=vol_lookup,
        )
        maxdd_raw = float((np.maximum.accumulate(cum_raw) - cum_raw).max())
        scale = MAX_DD_TARGET / maxdd_raw if maxdd_raw > 1.0 else 1.0
        cum_sc = cum_raw * scale
        cum_by_tf_sc = {tf: c * scale for tf, c in cum_by_tf_raw.items()}
        wr = wins / (wins + losses) * 100 if (wins + losses) else 0
        dd_sc = float((np.maximum.accumulate(cum_sc) - cum_sc).max())
        print(f"    T={wins+losses}  WR={wr:.1f}%  scale={scale:.2f}x  "
              f"PnL/j={cum_sc[-1]/days:+.0f}$  maxDD={dd_sc:.0f}$", flush=True)
        print_tf_breakdown(cum_by_tf_sc, variant, days)

        series_list.append(dict(
            label=desc, variant=variant,
            cum=cum_sc, wins=wins, losses=losses,
            scale=scale, cum_by_tf=cum_by_tf_sc,
        ))

    # Résumé delta
    if len(series_list) == 2:
        s_st = series_list[0]; s_dy = series_list[1]
        delta_pnl = (s_dy['cum'][-1] - s_st['cum'][-1]) / days
        wr_st = s_st['wins'] / (s_st['wins'] + s_st['losses']) * 100
        wr_dy = s_dy['wins'] / (s_dy['wins'] + s_dy['losses']) * 100
        dt_st = (s_st['wins'] + s_st['losses'])
        dt_dy = (s_dy['wins'] + s_dy['losses'])
        print(f"\n  === DELTA dynamique - statique ===", flush=True)
        print(f"  Trades : {dt_st} -> {dt_dy}  ({dt_dy-dt_st:+d})", flush=True)
        print(f"  WR     : {wr_st:.1f}% -> {wr_dy:.1f}%  ({wr_dy-wr_st:+.1f}pp)", flush=True)
        print(f"  PnL/j  : {delta_pnl:+.0f}$/j\n", flush=True)

    custom_overlay(
        series_list, n_hours, hour_index,
        out_dir / "overlay_vrs_static_vs_dynamic.png",
        title=f"{title_prefix}  vrs_lin sl=10 cap=0.55 vrs1h b=150 g=0.5-1.5  —  VRS statique vs dynamique",
    )


if __name__ == "__main__":
    print("\n=== Chart 1 : toute la période ===")
    run_chart(CSV_FULL, OUT_DIR / "all", "BTC — Full")

    print("\n=== Chart 2 : 4 derniers jours ===")
    run_chart(CSV_RECENT, OUT_DIR / "last4d", "BTC — Last 4d")

    print(f"\nOverlay full   -> {OUT_DIR / 'all' / 'overlay_vrs_static_vs_dynamic.png'}")
    print(f"Overlay last4d -> {OUT_DIR / 'last4d' / 'overlay_vrs_static_vs_dynamic.png'}")
