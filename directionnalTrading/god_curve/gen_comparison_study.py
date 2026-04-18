"""
Comparison study — Ancienne stratégie (optimizer) vs Nouvelle (god_curve BTC).

Les deux stratégies sont re-backtestedées from scratch sur le même dataset :
  - OLD : tri_02 et shield_05, meilleurs configs par TF depuis summary.txt,
          via bt_momentum_single.run_strategy + dataset_runtime cache
  - NEW : god_curve BTC configs via chart_utils (maker fill ask_cross)

Output : comparison_study_YYYYMMDD/
  report.txt
  metrics_comparison.png
  composite_equity.png          — equity curves sur axe temps commun
  old_strategy/per_tf/equity.png
  new_strategy/overlay_all.png + per_config/equity.png
"""
import sys
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
CACHE_DIR   = BASE / "optimizer" / ".dataset_cache"
SUMMARY     = BASE / "optimizer" / "demo" / "final" / "summary.txt"
OLD_DIR_SRC = BASE / "optimizer" / "demo"
OUT_BASE    = Path(__file__).resolve().parent / f"comparison_study_{datetime.now():%Y%m%d}"

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

# ── Imports god_curve ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import (
    load_contracts, precompute_vol, attach_cnet_data, run_config_tf,
    build_cumulative, build_hour_index, make_config_label, make_config_name,
    chart_equity, chart_entry_and_loss_analytics, attach_settlement_outcomes,
    TIMEFRAMES as GC_TIMEFRAMES, VOL_LBS, COLORS, make_xlabels,
    _rf_str_hourly_equity, _esc,
)

CSV_FULL = [str(BASE / "Datas/csv/BTC.csv"),
            str(BASE / "reportLive/safeChase/BTC.csv")]

# ── Imports optimizer (moteur original + dataset) ─────────────────────────────
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "old"))
from optimizer.dataset_runtime import TIMEFRAME_NAMES
import bt_momentum_single as bt

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DD_TARGET   = 500.0
MAX_ORDERS_NEW  = 2
SHARES_PER_FILL = 5          # old strategy fixed sizing
TF_NAMES_OLD    = ["m5", "m15", "h1"]

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COL = "#e0e0e0"
GRID_COL = "#2a2a4a"
OLD_COLORS  = ["#ff6b35", "#ff9f1c", "#e63946"]
NEW_COLORS_ = ["#00d4ff", "#00ff88", "#ffcc02", "#cc66ff"]

# ── god_curve configs (gen_custom_chart.py) ───────────────────────────────────
NEW_CONFIGS = [
    dict(label='vol_net_lin', curve='linear',
         slope=7.0, intercept=0.0, A_exp=None, tau=None,
         eq_cap=0.75, max_losses_cb=None,
         vol_lb_h=1, vol_thresh=60.0, vol_type='net'),
    dict(label='vrs_lin', curve='linear',
         slope=10.0, intercept=0.0, A_exp=None, tau=None,
         eq_cap=0.55, max_losses_cb=None,
         vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
         vrs_g_min=0.5, vrs_g_max=1.5),
    dict(label='vol_net_lin_025h', curve='linear',
         slope=3.0, intercept=0.0, A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=0.25, vol_thresh=40.0, vol_type='net'),
    dict(label='vol_range_lin', curve='linear',
         slope=2.0, intercept=0.0, A_exp=None, tau=None,
         eq_cap=0.65, max_losses_cb=None,
         vol_lb_h=0.5, vol_thresh=60.0, vol_type='range'),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. ANCIENNE STRATÉGIE — bt_momentum_single engine
# ═══════════════════════════════════════════════════════════════════════════════

def _load_old_dataset():
    """
    Charge les données sur les mêmes deux CSV que la new strategy.
    Si nécessaire, concatène les deux CSV et reconstruit le cache.
    """
    import tempfile, json, csv as csv_mod
    from optimizer.dataset_runtime import load_shared_dataset

    csv_paths = [Path(p) for p in CSV_FULL if Path(p).exists()]
    if not csv_paths:
        raise FileNotFoundError(f"Aucun CSV BTC trouvé dans {CSV_FULL}")

    if len(csv_paths) == 1:
        p = csv_paths[0]
        print(f"  [old] Dataset depuis {p.name}", flush=True)
        return load_shared_dataset(str(p), str(BASE), 30.0, verbose=True)

    # Deux CSV — concatène dans un fichier temporaire
    tmp_csv   = BASE / "optimizer" / ".dataset_cache" / "_merged_btc.csv"
    meta_path = BASE / "optimizer" / ".dataset_cache" / "metadata_merged.json"
    src_mtimes = {str(p): p.stat().st_mtime_ns for p in csv_paths}

    rebuild = True
    if meta_path.exists() and tmp_csv.exists():
        try:
            if json.loads(meta_path.read_text()).get("src_mtimes") == src_mtimes:
                rebuild = False
        except Exception:
            pass

    if rebuild:
        print(f"  [old] Concaténation {' + '.join(p.name for p in csv_paths)} → {tmp_csv.name}", flush=True)
        tmp_csv.parent.mkdir(parents=True, exist_ok=True)
        header_written = False
        with tmp_csv.open("w", encoding="utf-8", newline="") as out_f:
            writer = None
            for csv_p in csv_paths:
                with csv_p.open("r", encoding="utf-8", newline="") as in_f:
                    reader = csv_mod.DictReader(in_f)
                    if not header_written:
                        writer = csv_mod.DictWriter(out_f, fieldnames=reader.fieldnames)
                        writer.writeheader()
                        header_written = True
                    for row in reader:
                        writer.writerow(row)
        meta_path.write_text(json.dumps({"src_mtimes": src_mtimes}))
        print(f"  [old] Mergé : {tmp_csv.stat().st_size // 1024} KB", flush=True)
        # Invalider le cache existant pour forcer sa reconstruction
        old_meta = CACHE_DIR / "metadata.json"
        if old_meta.exists():
            old_meta.unlink()

    print(f"  [old] Chargement dataset...", flush=True)
    # Pointe bt vers le CSV mergé et charge son dataset interne
    bt.CSV_PATH = str(tmp_csv)
    return bt.ensure_dataset_loaded(verbose=True)


# ── Configs old strategy via bt.make_config (moteur original) ─────────────────
def _make_old_cfgs():
    """Construit les meilleurs configs par TF via le vrai bt.make_config."""
    return {
        # M5 rank1: tri_02 | dec=60s | conf=3s+1s | decvol=1.5 | confvol=0.5 | floor=15 | imb=2
        "m5": bt.make_config(
            "tri_02_m5", "momentum",
            decision_sec=60, decision_vol_mult=1.5,
            confirmations=[{"sec": 3}, {"sec": 1}],
            confirmation_vol_mult=0.5,
            price_floor_c=15, max_imbalance=2,
        ),
        # M15 rank1: shield_05 | decvol=1.0 | confvol=2.5 | floor=40 | imb=10
        "m15": bt.make_config(
            "shield_05_m15", "momentum",
            decision_sec=30, decision_vol_mult=1.0,
            confirmations=[{"sec": 10}],
            confirmation_vol_mult=2.5,
            price_floor_c=40, max_imbalance=10,
            counter_guards=[{"sec": 2, "max_counter": 0.10}],
            bid_guards=[{"sec": 1, "min_delta": 0.0}],
        ),
        # H1 rank1: tri_02 | dec=60s | conf=3s+1s | decvol=1.5 | confvol=0.75 | floor=5 | imb=4
        "h1": bt.make_config(
            "tri_02_h1", "momentum",
            decision_sec=60, decision_vol_mult=1.5,
            confirmations=[{"sec": 3}, {"sec": 1}],
            confirmation_vol_mult=0.75,
            price_floor_c=5, max_imbalance=4,
        ),
    }


def run_old_all_tfs(_dataset=None) -> dict:
    """
    Run l'ancienne stratégie via bt.run_strategy (moteur original, pas de re-implémentation).
    Le dataset est chargé dans bt via bt.ensure_dataset_loaded().
    """
    # Pointer bt vers le bon dataset (déjà chargé en mémoire via cache)
    # bt.ensure_dataset_loaded() a été appelé en amont dans _load_old_dataset()
    cfgs = _make_old_cfgs()
    EPOCH = datetime(1970, 1, 1)
    from datetime import timedelta

    results = {}
    for tf in TF_NAMES_OLD:
        cfg     = cfgs[tf]
        windows = bt.get_all_windows(tf)
        print(f"  [old] {tf.upper()}: {len(windows)} fenêtres", flush=True)
        pnl_series = []
        ts_series  = []
        n_fills    = 0
        for w in windows:
            cs, ce, btc_up = w["cs"], w["ce"], w["btc_up"]
            ts_start = EPOCH + timedelta(milliseconds=int(bt._TS_MS[cs]))
            sim  = bt.run_strategy(cs, ce, tf, cfg, btc_up=btc_up)
            fills = sim["buys"]
            sells = sim["sells"]
            pnl_c = sum(bt.SHARES_PER_FILL * (float(s["price"]) - float(s["entry_price"]))
                        for s in sells)
            pnl_series.append(pnl_c)
            ts_series.append(ts_start)
            n_fills += len(fills)

        total_pnl = sum(pnl_series) / 100.0
        cum  = np.cumsum([p / 100.0 for p in pnl_series])
        dd   = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
        rf   = total_pnl / dd if dd > 0 else float("inf")
        n_trades = sum(1 for p in pnl_series if p != 0)
        wins     = sum(1 for p in pnl_series if p > 0)
        wr       = wins / n_trades * 100 if n_trades else 0.0
        print(f"    PnL=${total_pnl:+.2f}  MaxDD=${dd:.2f}  RF={rf:.2f}  WR={wr:.1f}%  fills={n_fills}", flush=True)
        results[tf] = {
            "pnl_cents_series": pnl_series,
            "ts_series": ts_series,
            "cum_raw":  cum,
            "pnl_total": total_pnl,
            "max_dd_raw": dd,
            "rf_raw": rf,
            "n_fills": n_fills,
            "n_trades": n_trades,
            "wins": wins,
            "wr": wr,
            "cfg": cfg,
        }
    return results


def scale_old(results: dict) -> dict:
    """Scale chaque TF séparément à maxDD=$500."""
    out = {}
    for tf, r in results.items():
        dd = r["max_dd_raw"]
        scale = (MAX_DD_TARGET / dd) if dd > 1.0 else 1.0
        out[tf] = {**r,
                   "scale": scale,
                   "cum_scaled": r["cum_raw"] * scale,
                   "pnl_scaled": r["pnl_total"] * scale,
                   "dd_scaled":  dd * scale}
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NOUVELLE STRATÉGIE — god_curve BTC
# ═══════════════════════════════════════════════════════════════════════════════

def run_new_strategy(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts_by_tf = []
    m5_ref = None
    for tf_floor, bid_up, bid_down, ask_up, ask_down in GC_TIMEFRAMES:
        try:
            cts = load_contracts(CSV_FULL, tf_floor, bid_up, bid_down, ask_up, ask_down)
            if tf_floor == "5min":
                precompute_vol(cts, VOL_LBS)
                m5_ref = cts
            else:
                precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
            contracts_by_tf.append(cts)
            print(f"  [new] {tf_floor}: {len(cts)} contrats", flush=True)
        except Exception as e:
            print(f"  [new] {tf_floor}: skipped ({e})", flush=True)
            contracts_by_tf.append([])
    for (tf_floor, *_), cts in zip(GC_TIMEFRAMES, contracts_by_tf):
        if cts:
            attach_settlement_outcomes(cts, "btc", tf=tf_floor)
            attach_cnet_data(cts, (2, 5, 10), ref_contracts=m5_ref)

    hour_index, n_hours = build_hour_index(contracts_by_tf)
    days = n_hours / 24
    print(f"  [new] {n_hours}h ({days:.1f}j)", flush=True)

    series = []
    for i, base_cfg in enumerate(NEW_CONFIGS):
        color = NEW_COLORS_[i % len(NEW_COLORS_)]
        lbl   = make_config_label(base_cfg)
        cfg   = {**base_cfg, "max_orders": MAX_ORDERS_NEW}
        combined = defaultdict(float)
        tf_pnls  = [defaultdict(float) for _ in contracts_by_tf]
        wins = losses = 0
        for i_tf, cts in enumerate(contracts_by_tf):
            ph, w, l = run_config_tf(cts, cfg, hour_index)
            for h, p in ph.items():
                combined[h] += p
                tf_pnls[i_tf][h] += p
            wins += w; losses += l

        cum_raw, _ = build_cumulative(combined, n_hours)
        cum_by_tf  = {}
        for i_tf, tfn in enumerate(["M5","M15","H1"][:len(contracts_by_tf)]):
            c, _ = build_cumulative(tf_pnls[i_tf], n_hours)
            cum_by_tf[tfn] = c

        dd    = float((np.maximum.accumulate(cum_raw) - cum_raw).max()) if len(cum_raw) else 1.0
        scale = (MAX_DD_TARGET / dd) if dd > 1.0 else 1.0
        cum_s = cum_raw * scale
        cum_by_tf_s = {tf: c * scale for tf, c in cum_by_tf.items()}

        n_trades = wins + losses
        wr = wins / n_trades * 100 if n_trades else 0
        rf = float(cum_s[-1]) / (dd * scale) if (dd * scale) > 0 else 0
        var_lbl = f"{lbl} [x{scale:.1f}]"
        print(f"  [new] {lbl}: WR={wr:.1f}%  PnL=${cum_s[-1]:.0f}  RF={rf:.2f}", flush=True)

        cfg_dir = out_dir / make_config_name(base_cfg)
        cfg_dir.mkdir(exist_ok=True)
        hourly_s = np.diff(np.concatenate([[0.0], cum_s]))
        chart_equity(cum_s, hourly_s, var_lbl, n_hours, hour_index,
                     wins, losses, cfg_dir / "equity.png", cum_by_tf=cum_by_tf_s)
        try:
            chart_entry_and_loss_analytics(contracts_by_tf, cfg, hour_index, cfg_dir, var_lbl)
        except Exception as e:
            print(f"    analytics skip ({e})", flush=True)

        series.append(dict(
            label=var_lbl, lbl_short=lbl, cum=cum_s, wins=wins, losses=losses,
            n_trades=n_trades, wr=wr, rf=rf, pnl=float(cum_s[-1]),
            dd=dd * scale, days=days, color=color, ts_series=None,
        ))

    # Overlay new
    _overlay(series, n_hours, hour_index, out_dir / "overlay_all.png",
             "God Curve BTC — Nouvelle stratégie")
    return series, hour_index, n_hours


def _overlay(series, n_hours, hour_index, out_path, title):
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)
    days = n_hours / 24
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
        ax.tick_params(colors=TEXT_COL)
    for s in series:
        cum = s["cum"]
        dd_arr = np.maximum.accumulate(cum) - cum
        lbl = _esc(f"{s['label']}  T={s['n_trades']}({s['n_trades']/days:.1f}/j) "
                   f"WR={s['wr']:.0f}%  ${s['pnl']/days:.0f}/j  RF={s['rf']:.2f}")
        ax1.plot(x, cum, color=s["color"], linewidth=1.8, label=lbl)
        ax2.plot(x, dd_arr, color=s["color"], linewidth=1.2, alpha=0.7)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title(title, fontsize=12, color=TEXT_COL)
    ax1.set_ylabel(f"PnL ($) — maxDD→${MAX_DD_TARGET:.0f}", color=TEXT_COL)
    ax1.legend(loc="upper left", fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL)
    ax1.grid(alpha=0.25, color=GRID_COL)
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8, rotation=30, color=TEXT_COL)
    ax2.set_ylabel("Drawdown ($)", color=TEXT_COL)
    ax2.invert_yaxis(); ax2.grid(alpha=0.25, color=GRID_COL)
    ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, fontsize=8, rotation=30, color=TEXT_COL)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close("all")
    print(f"  -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHARTS D'EQUITY OLD (par TF, axe = contrat #)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_old_equity(old_scaled: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i_tf, (tf, r) in enumerate(old_scaled.items()):
        cum = r["cum_scaled"]
        x   = np.arange(len(cum))
        dd_arr = np.maximum.accumulate(cum) - cum
        cfg  = r["cfg"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})
        fig.patch.set_facecolor(DARK_BG)
        for ax in (ax1, ax2):
            ax.set_facecolor(PANEL_BG)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
            ax.tick_params(colors=TEXT_COL)

        color = OLD_COLORS[i_tf % len(OLD_COLORS)]
        ax1.plot(x, cum, color=color, linewidth=1.8)
        ax1.fill_between(x, cum, 0, where=cum >= 0, color=color, alpha=0.12)
        ax1.fill_between(x, cum, 0, where=cum < 0, color="#f44336", alpha=0.12)
        ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax1.set_title(
            f"OLD — {cfg['name']} ({tf.upper()})  "
            f"PnL=${r['pnl_scaled']:+.0f}  DD=${r['dd_scaled']:.0f}  "
            f"RF={r['rf_raw']:.2f}  WR={r['wr']:.1f}%  scale=x{r['scale']:.2f}",
            fontsize=11, color=TEXT_COL)
        ax1.set_ylabel("PnL ($) — scalé maxDD→$500", color=TEXT_COL)
        ax1.set_xlabel("Contrat #", color=TEXT_COL)
        ax1.grid(alpha=0.25, color=GRID_COL)

        ax2.fill_between(x, dd_arr, 0, color="#f44336", alpha=0.4)
        ax2.plot(x, dd_arr, color="#f44336", linewidth=1)
        ax2.set_ylabel("Drawdown ($)", color=TEXT_COL)
        ax2.set_xlabel("Contrat #", color=TEXT_COL)
        ax2.invert_yaxis(); ax2.grid(alpha=0.25, color=GRID_COL)

        plt.tight_layout()
        out_path = out_dir / f"{tf}_equity.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close("all")
        print(f"  [old] equity {tf.upper()} -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPOSITE — equity old+new sur axe temps commun (ts_start par contrat)
# ═══════════════════════════════════════════════════════════════════════════════

def chart_composite(old_scaled: dict, new_series: list, hour_index: dict,
                    n_hours: int, out_path: Path):
    """
    Old strategy : x = timestamp de chaque contrat (ts_series) → equity par contrat
    New strategy : x = heure indexée (hour_index) → equity horaire
    Les deux sont affichés sur un axe date commun.
    """
    # Construire l'axe date commun : toutes les heures de hour_index
    sorted_hours = sorted(hour_index.keys())
    n = len(sorted_hours)
    hour_to_x = {h: i for i, h in enumerate(sorted_hours)}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
        ax.tick_params(colors=TEXT_COL)

    # NEW strategy — sur hour_index
    x_new = np.arange(n_hours)
    for i, s in enumerate(new_series):
        cum = s["cum"]
        dd_arr = np.maximum.accumulate(cum) - cum
        lbl = _esc(f"[NEW] {s['lbl_short']}  WR={s['wr']:.0f}%  "
                   f"PnL=${s['pnl']:+.0f}  RF={s['rf']:.2f}")
        color = s["color"]
        # Interpoler pour aligner sur l'axe commun (approximation linéaire)
        orig_x = np.linspace(0, n - 1, n_hours)
        xc = np.arange(n)
        cum_i  = np.interp(xc, orig_x, cum)
        dd_i   = np.maximum.accumulate(cum_i) - cum_i
        ax1.plot(xc, cum_i, color=color, linewidth=1.8, alpha=0.9, label=lbl)
        ax2.plot(xc, dd_i,  color=color, linewidth=1.0, alpha=0.6)

    # OLD strategy — on mappe chaque contrat sur l'axe date
    for i_tf, (tf, r) in enumerate(old_scaled.items()):
        ts_list = r["ts_series"]
        cum_raw = r["cum_raw"]
        scale   = r["scale"]
        color   = OLD_COLORS[i_tf % len(OLD_COLORS)]
        cfg     = r["cfg"]

        # Trouver l'heure de chaque contrat dans l'axe commun
        xs = []
        ys = []
        for j, (ts, pnl_cum) in enumerate(zip(ts_list, cum_raw * scale)):
            # Heure tronquée
            h_key = ts.replace(minute=0, second=0, microsecond=0)
            if h_key in hour_to_x:
                xs.append(hour_to_x[h_key])
                ys.append(pnl_cum)
        if not xs:
            continue

        # Trier par x
        pairs = sorted(zip(xs, ys))
        xs_s = [p[0] for p in pairs]
        ys_s = [p[1] for p in pairs]
        dd_s = list((np.maximum.accumulate(ys_s) - ys_s))
        lbl_old = _esc(f"[OLD] {cfg['name']} ({tf.upper()})  "
                       f"WR={r['wr']:.0f}%  PnL=${r['pnl_scaled']:+.0f}  "
                       f"RF={r['rf_raw']:.2f}")
        ax1.plot(xs_s, ys_s, color=color, linewidth=1.6, linestyle="dashed", alpha=0.85, label=lbl_old)
        ax2.plot(xs_s, dd_s, color=color, linewidth=1.0, linestyle="dashed", alpha=0.5)

    # X-ticks sur dates
    xticks, xlabels = make_xlabels(hour_index, step=24)
    # On projette les xticks sur l'axe [0, n-1] via hour_index
    sorted_hi = sorted(hour_index.keys())
    xt2 = []
    xl2 = []
    step = max(1, len(sorted_hi) // 12)
    for k in range(0, len(sorted_hi), step):
        h = sorted_hi[k]
        xi = hour_to_x.get(h)
        if xi is not None:
            xt2.append(xi)
            xl2.append(str(h)[5:10])  # MM-DD

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title(
        f"Comparaison OLD vs NEW — même axe temporel  |  sizing: maxDD→${MAX_DD_TARGET:.0f}\n"
        f"OLD=dashed (contrats projetés sur date heure)  |  NEW=solid (equity horaire)",
        fontsize=11, color=TEXT_COL, pad=6)
    ax1.set_ylabel("PnL cumulé ($)", color=TEXT_COL)
    ax1.legend(loc="upper left", fontsize=7, facecolor=PANEL_BG, labelcolor=TEXT_COL, framealpha=0.8)
    ax1.grid(alpha=0.25, color=GRID_COL)
    ax1.set_xticks(xt2); ax1.set_xticklabels(xl2, fontsize=8, rotation=30, color=TEXT_COL)

    ax2.set_ylabel("Drawdown ($)", color=TEXT_COL)
    ax2.invert_yaxis(); ax2.grid(alpha=0.25, color=GRID_COL)
    ax2.set_xticks(xt2); ax2.set_xticklabels(xl2, fontsize=8, rotation=30, color=TEXT_COL)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close("all")
    print(f"  Composite -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TABLE MÉTRIQUES
# ═══════════════════════════════════════════════════════════════════════════════

def chart_metrics(old_scaled: dict, new_series: list, out_path: Path):
    rows = []
    for tf, r in old_scaled.items():
        rows.append({
            "label": f"OLD – {r['cfg']['name']} ({tf.upper()})",
            "pnl": r["pnl_scaled"], "dd": r["dd_scaled"],
            "rf": r["rf_raw"], "wr": r["wr"],
            "fills": r["n_fills"], "scale": r["scale"],
            "is_old": True,
        })
    for s in new_series:
        rows.append({
            "label": f"NEW – {s['lbl_short']}",
            "pnl": s["pnl"], "dd": s["dd"],
            "rf": s["rf"], "wr": s["wr"],
            "fills": s["n_trades"], "scale": None,
            "is_old": False,
        })

    col_labels = ["Stratégie", "PnL ($)", "MaxDD ($)", "RF", "WR (%)", "Trades", "Scale"]
    data = []
    for r in rows:
        data.append([
            r["label"],
            f"{r['pnl']:+.0f}",
            f"{r['dd']:.0f}",
            f"{r['rf']:.2f}",
            f"{r['wr']:.1f}",
            f"{r['fills']}",
            f"x{r['scale']:.2f}" if r["scale"] else "—",
        ])

    n_rows = len(data)
    fig, ax = plt.subplots(figsize=(16, max(4, n_rows * 0.8 + 2)))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG); ax.axis("off")

    cell_colors = [["#2a1a0e" if r["is_old"] else "#0e1a2a"] * len(col_labels) for r in rows]
    tbl = ax.table(cellText=data, colLabels=col_labels,
                   cellLoc="center", loc="center", cellColours=cell_colors)
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COL)
        if row == 0:
            cell.set_facecolor("#0a0a1e")
            cell.set_text_props(color=TEXT_COL, fontweight="bold", fontsize=11)
        else:
            is_old = rows[row - 1]["is_old"]
            cell.set_text_props(color="#ff9955" if is_old else "#55ddff")
    ax.set_title(
        f"Métriques comparatives — sizing : maxDD→${MAX_DD_TARGET:.0f}  |  OLD=Optimizer / NEW=God Curve BTC (ord={MAX_ORDERS_NEW})",
        fontsize=13, color=TEXT_COL, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close("all")
    print(f"  Metrics -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPORT TEXTE
# ═══════════════════════════════════════════════════════════════════════════════

def write_report(old_scaled: dict, new_series: list, out_path: Path):
    lines = [
        "=" * 70,
        "COMPARISON STUDY — Optimizer (OLD) vs God Curve BTC (NEW)",
        f"Généré le : {datetime.now():%Y-%m-%d %H:%M}",
        f"Sizing : maxDD→${MAX_DD_TARGET:.0f} pour les deux stratégies",
        "=" * 70, "",
        "──── ANCIENNE STRATÉGIE (optimizer) — meilleur par TF ────────────────",
    ]
    for tf, r in old_scaled.items():
        cfg = r["cfg"]
        lines += [
            f"  {tf.upper()} — {cfg['name']}",
            f"    decision_sec={cfg['decision_sec']}s  dec_vol={cfg['decision_vol_mult']}x  "
            f"conf={[c['sec'] for c in cfg['confirmations']]}  "
            f"conf_vol={cfg['confirmation_vol_mult']}x  floor={cfg['price_floor_c']}  "
            f"imb={cfg['max_imbalance']}",
            f"    PnL brut   : ${r['pnl_total']:+.2f}  MaxDD brut : ${r['max_dd_raw']:.2f}",
            f"    Scale      : x{r['scale']:.2f}  →  PnL=${r['pnl_scaled']:+.0f}  DD=${r['dd_scaled']:.0f}",
            f"    RF         : {r['rf_raw']:.2f}   WR={r['wr']:.1f}%   fills={r['n_fills']}", "",
        ]
    lines += ["──── NOUVELLE STRATÉGIE (god_curve BTC) ──────────────────────────────"]
    for s in new_series:
        lines += [
            f"  {s['lbl_short']}",
            f"    PnL=${s['pnl']:+.0f}  DD=${s['dd']:.0f}  RF={s['rf']:.2f}  "
            f"WR={s['wr']:.1f}%  trades={s['n_trades']}  ({s['n_trades']/s['days']:.1f}/j)", "",
        ]
    lines += ["=" * 70]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report -> {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n=== Comparison Study — OLD (Optimizer) vs NEW (God Curve BTC) ===")
    print(f"Output : {OUT_BASE}\n")
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    old_dir = OUT_BASE / "old_strategy"
    new_dir = OUT_BASE / "new_strategy"

    # 1. Load dataset & run old strategy
    print("=== [1/4] Chargement dataset & backtest ancienne stratégie ===")
    dataset = _load_old_dataset()
    print(f"  Dataset: {dataset.row_count} rows", flush=True)
    old_results = run_old_all_tfs(dataset)
    old_scaled  = scale_old(old_results)

    # 2. Run new strategy
    print("\n=== [2/4] Backtest nouvelle stratégie (god_curve BTC) ===")
    new_series, hour_index, n_hours = run_new_strategy(new_dir)

    # 3. Charts
    print("\n=== [3/4] Génération charts ===")
    chart_old_equity(old_scaled, old_dir)
    chart_composite(old_scaled, new_series, hour_index, n_hours,
                    OUT_BASE / "composite_equity.png")
    chart_metrics(old_scaled, new_series, OUT_BASE / "metrics_comparison.png")

    # 4. Report
    print("\n=== [4/4] Report texte ===")
    write_report(old_scaled, new_series, OUT_BASE / "report.txt")

    print(f"\n{'='*60}")
    print(f"Dossier : {OUT_BASE}")
    print(f"  composite_equity.png   — equity OLD+NEW sur axe date commun")
    print(f"  metrics_comparison.png — table de métriques")
    print(f"  old_strategy/          — equity par TF ancienne strat")
    print(f"  new_strategy/          — equity god_curve + overlay")
    print(f"  report.txt")


if __name__ == "__main__":
    main()
