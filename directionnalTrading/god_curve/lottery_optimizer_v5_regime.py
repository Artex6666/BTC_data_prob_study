"""
lottery_optimizer_v5_regime.py — Ask-cross (maker pessimiste) + filtre régime excursion.
Sorties : dossier par config (equity.png, daily_pnl.png) + overlay_all.png + résultats .txt

Insight (god_curve) :
  Les meilleures configs côté gagnant exigent net > X$ (excursion significative).
  => Les contrats à faible excursion ont un losing side à vraie proba ~10%
     pricé à 5 cents => edge lottery même en "marché calme" (hedge 5%).

Fill model : ask_cross uniquement (maker pessimiste).
  => Fill si ask ≤ bid dans un tick futur avant expiry.
"""

import sys, argparse, time, itertools, multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Datas" / "csv"
SETTLEMENT_CSV = DATA_DIR / "settlement.csv"
CSV_PATHS = [
    DATA_DIR / "BTC.csv",
    BASE_DIR / "reportLive" / "safeChase" / "BTC.csv",
]

SIZE          = 10.0
SCAN_WINDOW_S = 60
TOP_CHARTS    = 25   # nombre de configs pour lesquelles on génère des charts
TOP_N         = 30   # lignes dans le .txt

# ── Grille ────────────────────────────────────────────────────────────────────
PRICE_MAX_GRID = [0.02, 0.03, 0.05]
SLOPE_GRID     = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
INTERCEPT_GRID = [0, 5, 10, 15, 25]
SECS_PAIRS = [
    (0,  60), (5,  60), (10, 60), (20, 60),
    (5,  45), (10, 45), (20, 45),
    (5,  30), (10, 30), (20, 30),
    (0,  30), (0,  45),
]
# (exc_lo, exc_hi) — max |spot-open| atteint depuis le début du contrat
EXCURSION_BINS = [
    (0,    9999),
    (0,     200),
    (0,     400),
    (50,   9999),
    (100,  9999),
    (200,  9999),
    (50,    400),
    (100,   600),
]

# Fréquence minimum pour l'affichage (courbe non en escalier)
MIN_TRADES_PER_DAY = 3.0

COLORS = [
    "#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336",
    "#00BCD4","#FF5722","#8BC34A","#3F51B5","#FFC107",
    "#E91E63","#607D8B","#795548","#009688","#CDDC39",
]

# ─────────────────────────────────────────────────────────────────────────────
# Worker (multiprocess)
# ─────────────────────────────────────────────────────────────────────────────
_SHARED = {}

def _init_worker(snap):
    _SHARED['contracts'] = snap


def _simulate_config(cfg):
    pmax, slope, intcp, slo, shi, exc_lo, exc_hi = cfg
    contracts = _SHARED['contracts']

    wins = 0; losses = 0; total_pnl = 0.0
    no_fill_total = 0
    daily = defaultdict(float)

    for c in contracts:
        op        = c['op']
        spots     = c['spot']
        up_bid    = c['up_bid']
        dn_bid    = c['dn_bid']
        up_ask    = c['up_ask']
        dn_ask    = c['dn_ask']
        secs_left = c['secs_left']
        won_up    = c['won_up']
        max_exc   = c['max_exc']
        day       = c['day']
        n         = len(spots)

        for j in range(n):
            sl = secs_left[j]
            if sl < slo or sl > shi:
                continue
            exc = max_exc[j]
            if exc < exc_lo or exc > exc_hi:
                continue
            prox_limit = slope * sl + intcp
            if abs(spots[j] - op) > prox_limit:
                continue
            ub = up_bid[j]; db = dn_bid[j]
            if ub <= 0.0 and db <= 0.0:
                continue
            if ub <= 0.0:
                cheap_bid, cheap_is_up = db, False
            elif db <= 0.0:
                cheap_bid, cheap_is_up = ub, True
            elif ub <= db:
                cheap_bid, cheap_is_up = ub, True
            else:
                cheap_bid, cheap_is_up = db, False
            if cheap_bid > pmax:
                continue

            won       = won_up if cheap_is_up else (not won_up)
            entry_bid = max(cheap_bid, 1e-4)

            filled = False
            for k in range(j + 1, n):
                fut_ask = up_ask[k] if cheap_is_up else dn_ask[k]
                if fut_ask <= 0:
                    continue
                if fut_ask <= entry_bid:
                    filled = True
                    break
            if not filled:
                no_fill_total += 1
                break

            pnl = SIZE * (1.0 / entry_bid - 1.0) if won else -SIZE
            if won: wins += 1
            else:   losses += 1
            total_pnl += pnl
            daily[day] += pnl
            break

    n_trades = wins + losses
    if n_trades == 0:
        return cfg, None

    wr = wins / n_trades
    ev = total_pnl / n_trades

    days_sorted = sorted(daily.keys())
    cum = 0.0; peak = 0.0; maxdd = 0.0
    for d in days_sorted:
        cum += daily[d]
        if cum > peak: peak = cum
        dd = peak - cum
        if dd > maxdd: maxdd = dd

    rf        = total_pnl / maxdd if maxdd > 0 else float('inf')
    lose_days = sum(1 for v in daily.values() if v < 0)
    worst_day = min(daily.values()) if daily else 0.0

    return cfg, {
        'n': n_trades, 'wins': wins, 'wr': wr, 'ev': ev,
        'pnl': total_pnl, 'maxdd': maxdd, 'rf': rf,
        'lose_days': lose_days, 'nofill': no_fill_total,
        'worst_day': worst_day,
        'daily': dict(daily),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Re-simulation détaillée pour charts (process principal)
# ─────────────────────────────────────────────────────────────────────────────
def _simulate_detailed(contracts, cfg):
    """Re-simule une config et retourne trade_pnls + daily_pnl pour les charts."""
    pmax, slope, intcp, slo, shi, exc_lo, exc_hi = cfg

    trade_pnls  = []      # PNL par trade, ordre chronologique
    trade_dates = []
    daily       = defaultdict(float)

    for c in contracts:
        op        = c['op']
        spots     = c['spot']
        up_bid    = c['up_bid']
        dn_bid    = c['dn_bid']
        up_ask    = c['up_ask']
        dn_ask    = c['dn_ask']
        secs_left = c['secs_left']
        won_up    = c['won_up']
        max_exc   = c['max_exc']
        day       = c['day']
        n         = len(spots)

        for j in range(n):
            sl = secs_left[j]
            if sl < slo or sl > shi:
                continue
            exc = max_exc[j]
            if exc < exc_lo or exc > exc_hi:
                continue
            prox_limit = slope * sl + intcp
            if abs(spots[j] - op) > prox_limit:
                continue
            ub = up_bid[j]; db = dn_bid[j]
            if ub <= 0.0 and db <= 0.0:
                continue
            if ub <= 0.0:
                cheap_bid, cheap_is_up = db, False
            elif db <= 0.0:
                cheap_bid, cheap_is_up = ub, True
            elif ub <= db:
                cheap_bid, cheap_is_up = ub, True
            else:
                cheap_bid, cheap_is_up = db, False
            if cheap_bid > pmax:
                continue

            won       = won_up if cheap_is_up else (not won_up)
            entry_bid = max(cheap_bid, 1e-4)

            filled = False
            for k in range(j + 1, n):
                fut_ask = up_ask[k] if cheap_is_up else dn_ask[k]
                if fut_ask <= 0:
                    continue
                if fut_ask <= entry_bid:
                    filled = True
                    break
            if not filled:
                break

            pnl = SIZE * (1.0 / entry_bid - 1.0) if won else -SIZE
            trade_pnls.append(pnl)
            trade_dates.append(day)
            daily[day] += pnl
            break

    return trade_pnls, trade_dates, dict(daily)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def fmt_cfg(cfg):
    pmax, slope, intcp, slo, shi, exc_lo, exc_hi = cfg
    exc_hi_s = '∞' if exc_hi >= 9999 else f"{exc_hi}$"
    return (
        f"p≤{pmax:.2f} {slope:.2f}xT+{intcp:.0f} "
        f"[60s:{slope*60+intcp:.0f}$ 10s:{slope*10+intcp:.0f}$ 0s:{intcp:.0f}$] "
        f"t=[{slo:.0f}s-{shi:.0f}s] exc=[{exc_lo}$-{exc_hi_s}]"
    )


def cfg_dirname(cfg):
    pmax, slope, intcp, slo, shi, exc_lo, exc_hi = cfg
    exc_hi_s = 'inf' if exc_hi >= 9999 else str(int(exc_hi))
    return (
        f"p{pmax:.2f}_sl{slope:.2f}_int{intcp:.0f}"
        f"_t{slo:.0f}-{shi:.0f}"
        f"_exc{exc_lo:.0f}-{exc_hi_s}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────
def _date_xticks(ax, dates, max_ticks=12):
    """Pose des xticks avec labels de dates lisibles, espacés régulièrement."""
    n = len(dates)
    step = max(1, n // max_ticks)
    idxs = list(range(0, n, step))
    ax.set_xticks(idxs)
    ax.set_xticklabels(
        [dates[i][5:] for i in idxs],   # "MM-DD"
        rotation=40, ha='right', fontsize=7, color='white'
    )


def chart_equity(trade_pnls, trade_dates, daily, cfg_label, n_days, out_path):
    """3 panneaux avec dates sur l'axe X :
       - Equity cumulée par trade (x = date du trade)
       - PNL journalier (barres, x = date)
       - Drawdown par trade (x = date du trade)
    """
    if not trade_pnls:
        return

    # ── Séries journalières ────────────────────────────────────────────────────
    day_keys   = sorted(daily.keys())
    day_vals   = [daily[d] for d in day_keys]
    day_colors = ['#4CAF50' if v >= 0 else '#F44336' for v in day_vals]

    # ── Equity et drawdown sur axe calendaire (pas par trade index) ───────────
    daily_arr = np.array(day_vals)
    cum       = np.cumsum(np.concatenate([[0.0], daily_arr]))   # len = n_days+1
    cum_dates = [''] + day_keys                                  # aligner
    peak      = np.maximum.accumulate(cum)
    dd        = peak - cum

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_pnl = float(cum[-1])
    n_trades  = len(trade_pnls)
    wins      = sum(1 for p in trade_pnls if p > 0)
    wr        = wins / n_trades * 100 if n_trades else 0
    ev        = total_pnl / n_trades  if n_trades else 0
    maxdd     = float(dd.max())
    rf        = total_pnl / maxdd if maxdd > 0 else float('inf')
    rf_s      = f"{rf:.1f}x" if rf < 1e6 else "∞"
    lose_d    = sum(1 for v in daily.values() if v < 0)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55,
                           height_ratios=[3, 2, 1.5])

    # ── Panneau 1 : Equity curve (par trade, x = date) ───────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#16213e')
    x1 = range(len(cum))
    ax1.plot(x1, cum, color='#2196F3', linewidth=1.5)
    ax1.fill_between(x1, 0, cum, where=cum >= 0, alpha=0.15, color='#4CAF50')
    ax1.fill_between(x1, 0, cum, where=cum <  0, alpha=0.15, color='#F44336')
    ax1.axhline(0, color='#555', linewidth=0.7, linestyle='--')
    ax1.set_ylabel('PNL cumulé ($)', color='white')
    ax1.set_title(cfg_label, color='white', fontsize=8.5, pad=6)
    ax1.text(0.01, 0.97,
             f"T={n_trades}  WR={wr:.1f}%  EV={ev:+.2f}$/t  PNL={total_pnl:+.0f}$  "
             f"maxDD={maxdd:.0f}$  RF={rf_s}  loseDays={lose_d}/{n_days}",
             transform=ax1.transAxes, color='#ccc', fontsize=7.5,
             verticalalignment='top')
    _date_xticks(ax1, cum_dates)
    ax1.tick_params(colors='white'); ax1.spines[:].set_color('#444')

    # ── Panneau 2 : PNL journalier (barres, x = date) ────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#16213e')
    x2 = range(len(day_keys))
    ax2.bar(x2, day_vals, color=day_colors, alpha=0.85, width=0.7)
    ax2.axhline(0, color='#555', linewidth=0.7)
    ax2.set_ylabel('PNL journalier ($)', color='white')
    _date_xticks(ax2, day_keys)
    ax2.tick_params(colors='white'); ax2.spines[:].set_color('#444')

    # ── Panneau 3 : Drawdown (par trade, x = date) ───────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#16213e')
    ax3.fill_between(x1, 0, -dd, color='#F44336', alpha=0.6)
    ax3.plot(x1, -dd, color='#F44336', linewidth=0.8)
    ax3.set_ylabel('Drawdown ($)', color='white')
    ax3.set_xlabel('Date', color='white')
    _date_xticks(ax3, cum_dates)
    ax3.tick_params(colors='white'); ax3.spines[:].set_color('#444')

    for ax in [ax1, ax2, ax3]:
        ax.yaxis.label.set_color('white')

    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close('all')


def chart_winrate_by_price(trade_pnls, _trade_dates, cfg, out_path):
    """Analyse WR et EV par prix d'entrée (0.01 / 0.02) + par excursion bin."""
    if not trade_pnls:
        return

    pmax = cfg[0]
    buckets = defaultdict(lambda: {'wins': 0, 'n': 0, 'pnl': 0.0})
    for p in trade_pnls:
        won = p > 0
        price = round(pmax, 2)  # tous au même prix_max pour cette config
        buckets[price]['n']   += 1
        buckets[price]['pnl'] += p
        if won: buckets[price]['wins'] += 1

    # Courbe WR mobile (fenêtre 20 trades)
    wins_arr = np.array([1.0 if p > 0 else 0.0 for p in trade_pnls])
    window   = min(20, len(wins_arr))
    wr_roll  = np.convolve(wins_arr, np.ones(window)/window, mode='valid')

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # WR glissant
    ax = axes[0]
    ax.set_facecolor('#16213e')
    ax.plot(wr_roll * 100, color='#2196F3', linewidth=1.2, label=f'WR roll {window}t')
    ax.axhline(pmax * 100, color='#FF9800', linewidth=1.0, linestyle='--',
               label=f'Break-even {pmax*100:.0f}%')
    ax.set_ylabel('Win Rate (%)', color='white')
    ax.set_title('WR glissant (fenêtre 20 trades)', color='white', fontsize=9)
    ax.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')

    # PNL cumulé par jour (bar)
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')
    cum_arr = np.cumsum(trade_pnls)
    ax2.plot(cum_arr, color='#4CAF50', linewidth=1.2)
    ax2.fill_between(range(len(cum_arr)), 0, cum_arr, alpha=0.15, color='#4CAF50')
    ax2.axhline(0, color='#555', linewidth=0.7)
    ax2.set_xlabel('Trades', color='white')
    ax2.set_ylabel('PNL cumulé ($)', color='white')
    ax2.set_title('Equity curve (par trade)', color='white', fontsize=9)
    ax2.tick_params(colors='white'); ax2.spines[:].set_color('#444')

    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close('all')


def chart_overlay(all_series_with_dates, cfg_labels, out_path):
    """Overlay avec axe X = dates calendaires communes.
    Toutes les courbes couvrent la même période — les jours sans trade restent flat.
    """
    if not all_series_with_dates:
        return

    # ── Axe X commun : toutes les dates de toutes les séries ─────────────────
    all_dates_set = set()
    for _, trade_dates in all_series_with_dates:
        all_dates_set.update(trade_dates)
    all_dates = sorted(all_dates_set)   # liste de "YYYY-MM-DD" triée
    date_idx  = {d: i for i, d in enumerate(all_dates)}
    n_dates   = len(all_dates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#16213e')
        ax.axhline(0, color='#444', linewidth=0.7)
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#444')

    ax1.set_title('Overlay — top configs (PNL cumulé, axe = date)', color='white', fontsize=10)
    ax1.set_ylabel('PNL cumulé ($)', color='white')
    ax2.set_title('Drawdown comparatif', color='white', fontsize=10)
    ax2.set_ylabel('Drawdown ($)', color='white')
    ax2.set_xlabel('Date', color='white')

    for i, (trade_pnls, trade_dates) in enumerate(all_series_with_dates):
        color = COLORS[i % len(COLORS)]
        label = cfg_labels[i]

        # PNL journalier agrégé → cumul sur l'axe commun
        daily_pnl = defaultdict(float)
        for pnl, d in zip(trade_pnls, trade_dates):
            daily_pnl[d] += pnl

        cum_by_date = np.zeros(n_dates)
        for d, pnl in daily_pnl.items():
            cum_by_date[date_idx[d]] = pnl
        cum_by_date = np.cumsum(cum_by_date)   # ffill implicite via cumsum

        peak  = np.maximum.accumulate(cum_by_date)
        dd    = peak - cum_by_date
        total = float(cum_by_date[-1])
        maxdd = float(dd.max())
        rf    = total / maxdd if maxdd > 0 else float('inf')
        rf_s  = f"{rf:.1f}x" if rf < 1e6 else "∞"

        lbl = f"#{i+1} RF={rf_s} {label[:55]}"
        x   = range(n_dates)
        ax1.plot(x, cum_by_date, color=color, linewidth=1.2, alpha=0.85, label=lbl)
        ax2.fill_between(x, 0, -dd, color=color, alpha=0.20)
        ax2.plot(x, -dd, color=color, linewidth=0.8, alpha=0.7)

    _date_xticks(ax1, all_dates)
    _date_xticks(ax2, all_dates)

    ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=6.5,
               loc='upper left', ncol=2)
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close('all')


def chart_excursion_summary(results, out_path):
    """Résumé RF/EV par bin d'excursion — 1 bar chart."""
    bin_data = defaultdict(list)
    for cfg, r in results:
        key = (cfg[5], cfg[6])
        bin_data[key].append(r['rf'])

    bins   = sorted(bin_data.keys(), key=lambda x: x[0])
    labels = [f"[{lo}$-{'∞' if hi>=9999 else str(hi)+'$'}]" for lo, hi in bins]
    rf_med = [float(np.median(bin_data[b])) for b in bins]
    rf_max = [float(np.max(bin_data[b]))    for b in bins]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')
    x = np.arange(len(bins))
    ax.bar(x - 0.2, rf_med, 0.35, label='RF médian', color='#2196F3', alpha=0.85)
    ax.bar(x + 0.2, rf_max, 0.35, label='RF max',    color='#FF9800', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, color='white', fontsize=8)
    ax.set_ylabel('Recovery Factor', color='white')
    ax.set_title('RF par bin d\'excursion — quel régime est le plus profiteux?',
                 color='white', fontsize=10)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.tick_params(colors='white'); ax.spines[:].set_color('#444')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=20)
    args = ap.parse_args()
    n_workers = args.workers

    ts_run  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(__file__).parent / f"lottery_btc_v5_{ts_run}"
    out_dir.mkdir(exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"lottery_v5_results_{ts_run}.txt"

    print(f"\n{'='*80}")
    print(f"  Lottery Optimizer v5 — BTC M5 — Ask-cross + Régime Excursion")
    print(f"  Run: {ts_run}  |  Workers: {n_workers}")
    print(f"{'='*80}\n")

    # ── Chargement données ────────────────────────────────────────────────────
    print("[*] Chargement CSV ...", flush=True)
    dfs = []
    for p in CSV_PATHS:
        if not p.exists(): continue
        d = pd.read_csv(p, usecols=[
            'timestamp', 'spot_price',
            'm5_up_bid', 'm5_down_bid',
            'm5_up_ask', 'm5_down_ask',
        ])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
        print(f"    {p.name}: {len(d):,} ticks")

    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    df['contract_ce'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    sdf = pd.read_csv(SETTLEMENT_CSV)
    sdf = sdf[(sdf['asset'] == 'btc') & (sdf['tf'] == '5min')]
    outcomes = {int(r.open_ts): (float(r.outcome_up) > 0.5) for _, r in sdf.iterrows()}
    print(f"    Settlement: {len(outcomes):,} contrats\n")

    # ── Contrats ──────────────────────────────────────────────────────────────
    print("[*] Construction des contrats ...", flush=True)
    contracts  = []
    n_days_set = set()

    for ce, grp in df.groupby('contract_ce', sort=True):
        open_ts = int((ce - pd.Timedelta(minutes=5)).timestamp())
        if open_ts not in outcomes:
            continue
        op_spot = float(grp['spot_price'].iloc[0])

        all_spots       = grp['spot_price'].values.astype(np.float64)
        running_max_exc = np.maximum.accumulate(np.abs(all_spots - op_spot))

        wnd_mask  = grp['timestamp'] >= ce - pd.Timedelta(seconds=SCAN_WINDOW_S)
        wnd       = grp[wnd_mask].reset_index(drop=True)
        if len(wnd) < 2:
            continue

        max_exc   = running_max_exc[wnd_mask.values]
        spots     = wnd['spot_price'].values.astype(np.float64)
        ts_ns     = wnd['timestamp'].values.astype(np.int64)
        ce_ns     = ce.value
        secs_left = (ce_ns - ts_ns) / 1e9
        day       = str(ce.date())
        n_days_set.add(day)

        contracts.append({
            'won_up':    outcomes[open_ts],
            'op':        op_spot,
            'spot':      spots,
            'secs_left': secs_left,
            'up_bid':    wnd['m5_up_bid'].values.astype(np.float64),
            'dn_bid':    wnd['m5_down_bid'].values.astype(np.float64),
            'up_ask':    wnd['m5_up_ask'].values.astype(np.float64),
            'dn_ask':    wnd['m5_down_ask'].values.astype(np.float64),
            'max_exc':   max_exc,
            'day':       day,
        })

    n_days = len(n_days_set)
    print(f"    {len(contracts):,} contrats  |  {n_days} jours\n")
    del df

    # ── Grille ────────────────────────────────────────────────────────────────
    configs = [
        (p, sl, ic, slo, shi, el, eh)
        for p, sl, ic, (slo, shi), (el, eh) in itertools.product(
            PRICE_MAX_GRID, SLOPE_GRID, INTERCEPT_GRID, SECS_PAIRS, EXCURSION_BINS
        )
        if slo < shi
    ]
    print(f"[*] {len(configs):,} configs  |  {n_workers} workers\n", flush=True)

    # ── Run multiprocess ──────────────────────────────────────────────────────
    t0 = time.time()
    results = []
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(contracts,)) as pool:
        for i, (cfg, res) in enumerate(
            pool.imap_unordered(_simulate_config, configs, chunksize=15), 1
        ):
            if res is not None:
                results.append((cfg, res))
            if i % 1000 == 0:
                print(f"  {i}/{len(configs)} ({i/len(configs)*100:.0f}%)  {time.time()-t0:.0f}s",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\n[*] Done in {elapsed:.0f}s  |  {len(results):,} configs avec trades\n")

    if not results:
        print("[!] Aucun résultat."); return

    # ── Tri ───────────────────────────────────────────────────────────────────
    results_rf = sorted(results, key=lambda x: -x[1]['rf'])
    results_ev = sorted(results, key=lambda x: -x[1]['ev'])

    # ── Texte .txt ────────────────────────────────────────────────────────────
    COL = 120
    lines = []
    lines.append(
        f"Lottery v5 — ask_cross + régime excursion  |  "
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Data: {n_days}j  |  {len(configs)} configs  |  {elapsed:.0f}s"
    )
    lines.append("")

    # Deux vues : tous résultats + filtré fréquence ≥ MIN_TRADES_PER_DAY
    results_rf_freq = [(c, r) for c, r in results_rf if r['n'] / n_days >= MIN_TRADES_PER_DAY]
    results_ev_freq = [(c, r) for c, r in results_ev if r['n'] / n_days >= MIN_TRADES_PER_DAY]

    sections = [
        (f"RF  — T/j ≥ {MIN_TRADES_PER_DAY:.0f} (courbe lisse)",   results_rf_freq),
        (f"EV  — T/j ≥ {MIN_TRADES_PER_DAY:.0f} (courbe lisse)",   results_ev_freq),
        ("RF  — tous résultats",                                     results_rf),
        ("EV  — tous résultats",                                     results_ev),
    ]

    for sort_label, sorted_res in sections:
        lines.append("=" * COL)
        lines.append(f">>> TOP {TOP_N} — trié par {sort_label}")
        lines.append("=" * COL)
        lines.append(
            f"  {'#':>3}  {'config':<90}  {'T':>5}  {'T/j':>4}"
            f"  {'WR':>6}  {'edge':>6}  {'EV$/t':>8}"
            f"  {'PNL':>9}  {'PNL/j':>7}  {'maxDD':>6}  {'RF':>5}  {'nofill':>7}  lose  wrstDay"
        )
        lines.append("─" * COL)
        for rank, (cfg, r) in enumerate(sorted_res[:TOP_N], 1):
            pmax = cfg[0]
            edge = r['wr'] * 100 - pmax * 100
            star = '★' if edge > 0 else ' '
            lines.append(
                f"  {rank:>3}  {fmt_cfg(cfg):<90}"
                f"  {r['n']:>5}  {r['n']/n_days:>4.1f}"
                f"  {r['wr']*100:>5.1f}%  {edge:>+5.1f}%{star}"
                f"  {r['ev']:>+8.2f}$"
                f"  {r['pnl']:>+9.0f}$  {r['pnl']/n_days:>+7.1f}$/j"
                f"  {r['maxdd']:>6.0f}$  {r['rf']:>5.1f}x"
                f"  {r['nofill']:>7d}"
                f"  {r['lose_days']}/{n_days}  {r['worst_day']:>+8.0f}$"
            )
        lines.append("")

    # Synthèse par bin excursion
    lines.append("=" * COL)
    lines.append(">>> ANALYSE PAR BIN EXCURSION (meilleure config par bin, triées par RF)")
    lines.append("=" * COL)
    bin_best = {}
    for cfg, r in results_rf:
        key = (cfg[5], cfg[6])
        if key not in bin_best:
            bin_best[key] = (cfg, r)
    lines.append(f"  {'exc_bin':<22}  {'T':>5}  {'WR':>6}  {'EV$/t':>8}  {'RF':>6}  config")
    lines.append("─" * COL)
    for (el, eh), (cfg, r) in sorted(bin_best.items(), key=lambda x: -x[1][1]['rf']):
        eh_s = '∞' if eh >= 9999 else str(eh)
        lines.append(
            f"  [{el}$-{eh_s}]{'':>12}"
            f"  {r['n']:>5}  {r['wr']*100:>5.1f}%"
            f"  {r['ev']:>+8.2f}$  {r['rf']:>6.1f}x  {fmt_cfg(cfg)}"
        )
    lines.append("")

    if results:
        ev_pos = sum(1 for _, r in results if r['ev'] > 0)
        rf_pos = sum(1 for _, r in results if r['rf'] > 1)
        ev_moy = float(np.mean([r['ev'] for _, r in results]))
        rf_vals = [r['rf'] for _, r in results if r['rf'] < 1e9]
        rf_max  = max(rf_vals) if rf_vals else 0.0
        lines.append("=" * COL)
        lines.append(
            f"SUMMARY  ask_cross+régime  "
            f"EV>0={ev_pos}/{len(results)}  RF>1x={rf_pos}  "
            f"EV_moy={ev_moy:+.2f}$/t  RF_max={rf_max:.1f}x"
        )
        lines.append("=" * COL)

    out_text = "\n".join(lines)
    print(out_text)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(out_text)
    print(f"\n[*] Résultats → {out_path}")

    # ── Génération des charts ─────────────────────────────────────────────────
    # Priorité aux configs avec fréquence suffisante pour une courbe lisse
    top_for_charts = (results_rf_freq[:TOP_CHARTS]
                      if len(results_rf_freq) >= 5
                      else results_rf[:TOP_CHARTS])
    print(f"\n[*] Génération des charts ({len(top_for_charts)} configs, "
          f"T/j≥{MIN_TRADES_PER_DAY:.0f} ou top RF) ...", flush=True)

    all_series  = []
    all_labels  = []

    for rank, (cfg, r) in enumerate(top_for_charts, 1):
        label    = fmt_cfg(cfg)
        dir_name = f"{rank:02d}_{cfg_dirname(cfg)}"
        cfg_dir  = charts_dir / dir_name
        cfg_dir.mkdir(exist_ok=True)

        # Re-simulation détaillée
        trade_pnls, trade_dates, daily = _simulate_detailed(contracts, cfg)
        if not trade_pnls:
            continue

        # equity.png
        chart_equity(
            trade_pnls, trade_dates, daily, label, n_days,
            cfg_dir / "equity.png"
        )

        # winrate_analysis.png
        chart_winrate_by_price(
            trade_pnls, trade_dates, cfg,
            cfg_dir / "winrate_analysis.png"
        )

        all_series.append((trade_pnls, trade_dates))
        all_labels.append(label)
        print(f"  [{rank:>2}/{TOP_CHARTS}] {label[:70]}", flush=True)

    # overlay_all.png
    if all_series:
        print("[*] Génération overlay_all.png ...", flush=True)
        chart_overlay(
            all_series, all_labels,
            charts_dir / "overlay_all.png"
        )

    # excursion_summary.png
    print("[*] Génération excursion_summary.png ...", flush=True)
    chart_excursion_summary(results, charts_dir / "excursion_summary.png")

    print(f"\n[*] Charts → {charts_dir}")
    print(f"[*] Terminé  ({time.time()-t0:.0f}s total)\n")


if __name__ == '__main__':
    mp.freeze_support()
    main()
