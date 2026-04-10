"""
analyze_live_vs_bt.py — Comparaison exhaustive Live vs Backtest (BTC + ETH)

Sources live : reportLive/safeChase/slope3int0cap70/{btc,eth}/{m5,m15,h1}/
Sources BT   : reportLive/safeChase/{BTC,ETH}.csv (même CSV que le live)

Configs :
  BTC : vol_net_lin  sl=3.0  int=0  cap=0.70  net0.5h>60   max_orders=2  size=$100/ordre
  ETH : vol_trend_lin sl=0.20 int=0  cap=0.55  tre2h>0.50   max_orders=2  size=$150/ordre

Sorties : live_vs_bt/
  equity_btc.png         equity_eth.png         equity_combined.png
  fill_prices.png        pnl_per_trade.png      stats_comparison.png
  trigger_match.png      pnl_by_fill_bucket.png stats.txt
"""
import sys, json, glob, os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE     = Path(__file__).resolve().parent.parent
LIVE_DIR = BASE / "reportLive" / "safeChase" / "slope3int0cap70"
CSV_DIR  = BASE / "reportLive" / "safeChase"
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import (
    load_contracts, precompute_vol, run_config_tf, simulate,
    build_cumulative, build_hour_index, make_xlabels,
    TIMEFRAMES, VOL_LBS,
)
import chart_utils as cu

CFG_BTC = dict(curve='linear', slope=3.0, intercept=0.0, eq_cap=0.70,
               vol_lb_h=0.5, vol_thresh=60.0, vol_type='net',
               max_losses_cb=None, max_orders=2)
CFG_ETH = dict(curve='linear', slope=0.20, intercept=0.0, eq_cap=0.55,
               vol_lb_h=2.0, vol_thresh=0.50, vol_type='trend',
               max_losses_cb=None, max_orders=2)
BTC_SIZE = 100.0
ETH_SIZE = 150.0

# Cutoff : BTC → 2x$100 et ETH → 1x$150 cap=0.55 trend2h>0.50
CUTOFF_TS = pd.Timestamp("2026-04-04T13:10:00", tz="UTC").timestamp()

# ── Parse live ────────────────────────────────────────────────────────────────
def parse_live(asset_dir):
    trades, fills_all = [], []
    windows = []   # toutes les windows (pour trigger analysis)
    for tf in ['m5', 'm15', 'h1']:
        path = asset_dir / tf
        cur = None
        for f in sorted(path.glob('*.jsonl')):
            cur_fills = []
            for line in f.open():
                try:
                    d = json.loads(line); ev = d.get('event', '')
                    if ev == 'window_open':
                        ts = pd.to_datetime(d['ts']).to_pydatetime().replace(tzinfo=timezone.utc)
                        if ts.timestamp() < CUTOFF_TS:
                            cur = None; cur_fills = []; break  # skip file entier
                        cur = dict(ts=ts,
                                   tf=tf, skip=False, vol=0, chase=False, fill=False,
                                   pnl=0, cost=0, shares=0, avg_fill=0)
                        cur_fills = []
                    elif ev == 'window_start' and cur:
                        cur['vol'] = float(d.get('vol_gate_net_usd', 0))
                    elif ev == 'vol_gate_skip' and cur:
                        cur['skip'] = True
                    elif ev == 'chase_placed' and cur:
                        cur['chase'] = True
                    elif ev == 'fill' and cur:
                        cur['fill'] = True
                        cur_fills.append(float(d.get('price', 0)))
                    elif ev == 'window_ended' and cur:
                        shares = float(d.get('up_shares', 0)) + float(d.get('down_shares', 0))
                        cost   = float(d.get('up_cost', 0))   + float(d.get('down_cost', 0))
                        pnl    = float(d.get('expected_pnl', 0))
                        cur['shares'] = shares; cur['cost'] = cost; cur['pnl'] = pnl
                        cur['avg_fill'] = cost / shares if shares > 0 else 0
                        cur['won'] = pnl > 0
                        windows.append(dict(cur))
                        if shares > 0:
                            trades.append(dict(cur))
                            fills_all.extend(cur_fills)
                        cur = None; cur_fills = []
                except:
                    cur = None; cur_fills = []
    trades.sort(key=lambda x: x['ts'])
    windows.sort(key=lambda x: x['ts'])
    return trades, fills_all, windows


# ── Load BT contracts ─────────────────────────────────────────────────────────
def load_bt(asset, cfg, size):
    csv = [str(CSV_DIR / f"{asset}.csv")]
    m5_ref = None; ctf = []
    for tf_floor, b1, b2, a1, a2 in TIMEFRAMES:
        cts = load_contracts(csv, tf_floor, b1, b2, a1, a2)
        if tf_floor == '5min':
            precompute_vol(cts, VOL_LBS); m5_ref = cts
        else:
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
        ctf.append(cts)
    return ctf


def run_bt_trades(ctf, cfg, size):
    """Run simulate sur chaque contrat, retourne liste de trades avec fill price."""
    orig = cu.BASE_SIZE; cu.BASE_SIZE = float(size)
    from chart_utils import simulate_trade_log
    vol_key = f"vol_{cfg['vol_lb_h']:g}h_{cfg.get('vol_type','range')}" if cfg.get('vol_lb_h') else None
    trades = []
    tf_names = ['m5', 'm15', 'h1']
    for i_tf, cts in enumerate(ctf):
        for c in cts:
            if c.get('open_ts', 0) < CUTOFF_TS: continue
            if vol_key and c.get(vol_key, 0.0) < cfg.get('vol_thresh', 0): continue
            won, pnl, fills = simulate_trade_log(c, cfg)
            if won is None: continue
            avg_fill = np.mean([f['price'] for f in fills]) if fills else 0
            trades.append(dict(ts=c['ce'], open_ts=c['open_ts'],
                               tf=tf_names[i_tf], pnl=pnl, won=won,
                               cost=size * cfg['max_orders'],
                               avg_fill=avg_fill, fills=fills))
    cu.BASE_SIZE = orig
    trades.sort(key=lambda x: x['ts'])
    return trades


def run_bt_equity(ctf, cfg, size):
    orig = cu.BASE_SIZE; cu.BASE_SIZE = float(size)
    # Filtre cutoff : on patch temporairement les listes
    ctf_filtered = []
    for cts in ctf:
        ctf_filtered.append([c for c in cts if c.get('open_ts', 0) >= CUTOFF_TS])
    hour_index, n_hours = build_hour_index(ctf_filtered)
    combined = defaultdict(float); wins = losses = 0
    for cts in ctf_filtered:
        ph, w, l = run_config_tf(cts, cfg, hour_index)
        for h, p in ph.items(): combined[h] += p
        wins += w; losses += l
    cum, _ = build_cumulative(combined, n_hours)
    cu.BASE_SIZE = orig
    return cum, wins, losses, hour_index, n_hours


# ── Stats ─────────────────────────────────────────────────────────────────────
def compute_stats(trades, label, n_days):
    if not trades:
        return dict(label=label, n=0, wr=0, pnl=0, pnl_j=0, avg_win=0, avg_loss=0,
                    avg_fill=0, maxdd=0, rf=0, n_days=n_days)
    wins   = [t for t in trades if t.get('won', t['pnl'] > 0)]
    losses = [t for t in trades if not t.get('won', t['pnl'] > 0)]
    pnl = sum(t['pnl'] for t in trades)
    cum = 0; peak = 0; dds = []
    for t in trades:
        cum += t['pnl']; peak = max(peak, cum); dds.append(peak - cum)
    maxdd = max(dds) if dds else 0
    return dict(label=label, n=len(trades), wr=100*len(wins)/len(trades),
                pnl=pnl, pnl_j=pnl/max(n_days, 0.01),
                avg_win=float(np.mean([t['pnl'] for t in wins])) if wins else 0,
                avg_loss=float(np.mean([t['pnl'] for t in losses])) if losses else 0,
                avg_fill=float(np.mean([t.get('avg_fill', 0) for t in trades])),
                maxdd=maxdd, rf=pnl/maxdd if maxdd > 0.01 else float('inf'),
                n_days=n_days)


# ── Graphique equity ──────────────────────────────────────────────────────────
def plot_equity_comparison(live_trades, bt_cum, bt_hi, bt_nh, bt_days,
                           title, out_path, color='#2196F3'):
    x = np.arange(bt_nh)
    xticks, xlabels = make_xlabels(bt_hi, step=24)
    dd_bt = np.maximum.accumulate(bt_cum) - bt_cum

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [3, 1]})
    rf_bt = bt_cum[-1] / max(dd_bt.max(), 0.01)
    axes[0].plot(x, bt_cum, color=color, linewidth=1.6,
                 label=f'BT  ${bt_cum[-1]:,.0f}  (${bt_cum[-1]/bt_days:.0f}/j)  RF={rf_bt:.1f}x')

    if live_trades:
        t0 = live_trades[0]['ts'].timestamp()
        t1 = live_trades[-1]['ts'].timestamp()
        live_h = (t1 - t0) / 3600
        scale = bt_nh / max(live_h, 1)
        lc = np.cumsum([t['pnl'] for t in live_trades])
        lx = [(t['ts'].timestamp() - t0) / 3600 * scale for t in live_trades]
        dd_live = np.maximum.accumulate(lc) - lc
        live_days = live_h / 24
        rf_live = lc[-1] / max(dd_live.max(), 0.01)
        axes[0].plot(lx, lc, color='#F44336', linewidth=1.8, linestyle='--',
                     label=f'LIVE  ${lc[-1]:,.2f}  (${lc[-1]/live_days:.0f}/j)  RF={rf_live:.1f}x')
        axes[1].plot(lx, dd_live, color='#F44336', linewidth=1.2, linestyle='--', label='DD Live')

    axes[1].plot(x, dd_bt, color=color, linewidth=1.2, alpha=0.8, label='DD BT')
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_title(title, fontsize=11)
    axes[0].set_ylabel('PnL cumulé ($)')
    axes[0].legend(loc='upper left', fontsize=9); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(xticks); axes[0].set_xticklabels(xlabels, fontsize=8, rotation=30)
    axes[1].invert_yaxis(); axes[1].set_ylabel('Drawdown ($)')
    axes[1].legend(loc='lower left', fontsize=8); axes[1].grid(alpha=0.3)
    axes[1].set_xticks(xticks); axes[1].set_xticklabels(xlabels, fontsize=8, rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Graphique equity combiné ──────────────────────────────────────────────────
def plot_combined(btc_live, eth_live, bt_btc, bt_eth, bt_hi, bt_nh, bt_days, out_path):
    x = np.arange(bt_nh)
    xticks, xlabels = make_xlabels(bt_hi, step=24)
    fig, ax = plt.subplots(figsize=(16, 7))
    bt_comb = bt_btc + bt_eth
    ax.plot(x, bt_btc,  color='#FFC107', linewidth=1.2, linestyle='--',
            label=f'BT BTC  ${bt_btc[-1]:,.0f}  (${bt_btc[-1]/bt_days:.0f}/j)')
    ax.plot(x, bt_eth,  color='#4CAF50', linewidth=1.2, linestyle='--',
            label=f'BT ETH  ${bt_eth[-1]:,.0f}  (${bt_eth[-1]/bt_days:.0f}/j)')
    ax.plot(x, bt_comb, color='#2196F3', linewidth=1.8, linestyle='--',
            label=f'BT BTC+ETH  ${bt_comb[-1]:,.0f}  (${bt_comb[-1]/bt_days:.0f}/j)')
    all_live = sorted(btc_live + eth_live, key=lambda x: x['ts'])
    if all_live:
        t0 = all_live[0]['ts'].timestamp(); t1 = all_live[-1]['ts'].timestamp()
        live_h = (t1 - t0) / 3600; scale = bt_nh / max(live_h, 1)
        lc = np.cumsum([t['pnl'] for t in all_live])
        lx = [(t['ts'].timestamp() - t0) / 3600 * scale for t in all_live]
        ax.plot(lx, lc, color='#F44336', linewidth=2.2,
                label=f'LIVE BTC+ETH  ${lc[-1]:.2f}  (${lc[-1]/live_h*24:.0f}/j)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=8, rotation=30)
    ax.set_title('Portfolio — LIVE (rouge) vs BT (tirets bleus)\n'
                 'BT normalisé sur meme CSV — courbes comparables en $/j', fontsize=11)
    ax.set_ylabel('PnL cumulé ($)'); ax.legend(loc='upper left', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Graphique fill prices ─────────────────────────────────────────────────────
def plot_fill_prices(btc_fills, eth_fills, btc_bt_trades, eth_bt_trades, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Prix de fill : Live vs BT', fontsize=12)

    for col, (asset, live_f, bt_t, cap, color) in enumerate([
        ('BTC', btc_fills, btc_bt_trades, 0.70, '#F44336'),
        ('ETH', eth_fills, eth_bt_trades, 0.55, '#FF9800'),
    ]):
        bt_fills = [t['avg_fill'] for t in bt_t if t['avg_fill'] > 0]

        # Histogram live
        ax = axes[0, col]
        if live_f:
            ax.hist(live_f, bins=40, color=color, alpha=0.7, edgecolor='white')
            ax.axvline(cap, color='black', linestyle='--', linewidth=2, label=f'Cap BT={cap}')
            ax.axvline(np.median(live_f), color='navy', linestyle='-', linewidth=1.5,
                       label=f'Med live={np.median(live_f):.3f}')
            pct_low = 100 * sum(1 for f in live_f if f <= cap + 0.05) / len(live_f)
            ax.text(0.02, 0.95, f'{pct_low:.0f}% fills <= cap+0.05', transform=ax.transAxes,
                    fontsize=9, color='darkgreen', va='top')
        ax.set_title(f'{asset} — Fills live (n={len(live_f)})')
        ax.set_xlabel('Prix fill'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Overlay live vs BT histogram
        ax = axes[1, col]
        all_vals = live_f + bt_fills
        if all_vals:
            bins = np.linspace(min(all_vals), max(all_vals), 50)
            ax.hist(live_f, bins=bins, alpha=0.6, color=color, label=f'Live n={len(live_f)}')
            ax.hist(bt_fills, bins=bins, alpha=0.5, color='#2196F3', label=f'BT n={len(bt_fills)}')
            ax.axvline(cap, color='black', linestyle='--', linewidth=1.5, label=f'Cap={cap}')
            ax.set_title(f'{asset} — Live vs BT fill prices overlay')
            ax.set_xlabel('Prix fill'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Graphique PnL par trade ───────────────────────────────────────────────────
def plot_pnl_per_trade(btc_live, btc_bt, eth_live, eth_bt, out_path):
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle('PnL par trade — Live vs BT', fontsize=12)

    # BTC distribution
    ax = fig.add_subplot(gs[0, 0])
    lp = [t['pnl'] for t in btc_live]; bp = [t['pnl'] for t in btc_bt]
    if lp or bp:
        lo = min((lp or [0]) + (bp or [0])); hi = max((lp or [0]) + (bp or [0]))
        bins = np.linspace(lo, hi, 50)
        if lp: ax.hist(lp, bins=bins, alpha=0.6, color='#F44336', label=f'Live n={len(lp)}')
        if bp: ax.hist(bp, bins=bins, alpha=0.5, color='#2196F3', label=f'BT n={len(bp)}')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title('BTC — PnL/trade'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ETH distribution
    ax = fig.add_subplot(gs[0, 1])
    lp = [t['pnl'] for t in eth_live]; bp = [t['pnl'] for t in eth_bt]
    if lp or bp:
        lo = min((lp or [0]) + (bp or [0])); hi = max((lp or [0]) + (bp or [0]))
        bins = np.linspace(lo, hi, 50)
        if lp: ax.hist(lp, bins=bins, alpha=0.6, color='#FF9800', label=f'Live n={len(lp)}')
        if bp: ax.hist(bp, bins=bins, alpha=0.5, color='#4CAF50', label=f'BT n={len(bp)}')
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title('ETH — PnL/trade'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # PnL par TF - BTC
    ax = fig.add_subplot(gs[0, 2])
    for tf, color in [('m5','#F44336'),('m15','#FF9800'),('h1','#4CAF50')]:
        tl = [t['pnl'] for t in btc_live if t['tf']==tf]
        tb = [t['pnl'] for t in btc_bt   if t['tf']==tf]
        x = np.arange(2); w = 0.25
        idx = ['m5','m15','h1'].index(tf)
        ax.bar(0 + idx*w, sum(tl) if tl else 0, w, color=color, alpha=0.8, label=tf)
        ax.bar(1 + idx*w, sum(tb) if tb else 0, w, color=color, alpha=0.4)
    ax.set_xticks([0.25, 1.25]); ax.set_xticklabels(['Live','BT'])
    ax.set_title('BTC — PnL par TF (live vs BT)'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

    # Equity chronologique BTC live
    ax = fig.add_subplot(gs[1, 0])
    if btc_live:
        ts = sorted(btc_live, key=lambda x: x['ts'])
        cum = np.cumsum([t['pnl'] for t in ts])
        th  = [(t['ts'].timestamp() - ts[0]['ts'].timestamp())/3600 for t in ts]
        ax.plot(th, cum, color='#F44336', linewidth=1.6)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        colors_c = ['green' if t['pnl']>0 else 'red' for t in ts]
        ax.scatter(th, cum, c=colors_c, s=25, zorder=5)
        ax.fill_between(th, cum, 0, where=cum>=0, alpha=0.1, color='green')
        ax.fill_between(th, cum, 0, where=cum<0,  alpha=0.1, color='red')
    ax.set_title(f'BTC Live — cumul chronologique (${sum(t["pnl"] for t in btc_live):.2f})')
    ax.set_xlabel('Heures depuis start'); ax.grid(alpha=0.3)

    # Equity chronologique ETH live
    ax = fig.add_subplot(gs[1, 1])
    if eth_live:
        ts = sorted(eth_live, key=lambda x: x['ts'])
        cum = np.cumsum([t['pnl'] for t in ts])
        th  = [(t['ts'].timestamp() - ts[0]['ts'].timestamp())/3600 for t in ts]
        ax.plot(th, cum, color='#FF9800', linewidth=1.6)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        colors_c = ['green' if t['pnl']>0 else 'red' for t in ts]
        ax.scatter(th, cum, c=colors_c, s=25, zorder=5)
        ax.fill_between(th, cum, 0, where=cum>=0, alpha=0.1, color='green')
        ax.fill_between(th, cum, 0, where=cum<0,  alpha=0.1, color='red')
    ax.set_title(f'ETH Live — cumul chronologique (${sum(t["pnl"] for t in eth_live):.2f})')
    ax.set_xlabel('Heures depuis start'); ax.grid(alpha=0.3)

    # PnL journalier live (BTC+ETH)
    ax = fig.add_subplot(gs[1, 2])
    all_live = sorted(btc_live + eth_live, key=lambda x: x['ts'])
    if all_live:
        by_day = defaultdict(float)
        for t in all_live:
            d = t['ts'].strftime('%d-%m')
            by_day[d] += t['pnl']
        days_sorted = sorted(by_day.keys())
        vals = [by_day[d] for d in days_sorted]
        colors_d = ['green' if v >= 0 else 'red' for v in vals]
        ax.bar(days_sorted, vals, color=colors_d, alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.5)
        for i, (d, v) in enumerate(zip(days_sorted, vals)):
            ax.text(i, v + (2 if v >= 0 else -5), f'${v:.0f}', ha='center', fontsize=9)
    ax.set_title('PnL journalier LIVE (BTC+ETH)')
    ax.set_ylabel('PnL ($)'); ax.grid(alpha=0.3, axis='y')

    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Graphique stats comparison (barres) ───────────────────────────────────────
def plot_stats_comparison(s_bl, s_bb, s_el, s_eb, out_path):
    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)
    fig.suptitle('Statistiques Live vs BT — BTC & ETH', fontsize=13)

    assets = ['BTC', 'ETH']
    lives  = [s_bl, s_el]
    bts    = [s_bb, s_eb]
    cl = ['#F44336', '#FF9800']
    cb = ['#2196F3', '#4CAF50']

    def bar2(ax, vl, vb, title, ylabel, fmt='{:.1f}', ylim=None):
        x = np.arange(2); w = 0.35
        bars_l = ax.bar(x - w/2, vl, w, color=cl, alpha=0.85)
        bars_b = ax.bar(x + w/2, vb, w, color=cb, alpha=0.85)
        for i, (a, v1, v2) in enumerate(zip(assets, vl, vb)):
            ax.text(i-w/2, v1*(1.03 if v1>=0 else 0.97), fmt.format(v1), ha='center', va='bottom' if v1>=0 else 'top', fontsize=8)
            ax.text(i+w/2, v2*(1.03 if v2>=0 else 0.97), fmt.format(v2), ha='center', va='bottom' if v2>=0 else 'top', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(assets)
        ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)
        ax.axhline(0, color='gray', linewidth=0.5); ax.grid(alpha=0.3, axis='y')
        if ylim: ax.set_ylim(*ylim)
        if not ax.get_legend_handles_labels()[1]:
            from matplotlib.patches import Patch
            ax.legend([Patch(color=cl[0],alpha=0.85), Patch(color=cb[0],alpha=0.85)],
                      ['Live','BT'], fontsize=8)

    bar2(fig.add_subplot(gs[0,0]), [s['n']/s['n_days'] for s in lives], [s['n']/s['n_days'] for s in bts],
         'Trades/jour', 'n/j', '{:.1f}')
    bar2(fig.add_subplot(gs[0,1]), [s['wr'] for s in lives], [s['wr'] for s in bts],
         'Win Rate (%)', '%', '{:.1f}%', ylim=(85,100))
    bar2(fig.add_subplot(gs[0,2]), [s['pnl_j'] for s in lives], [s['pnl_j'] for s in bts],
         'PnL/jour ($)', '$', '${:.0f}')
    bar2(fig.add_subplot(gs[1,0]), [s['avg_win'] for s in lives], [s['avg_win'] for s in bts],
         'Avg win ($)', '$', '${:.2f}')
    bar2(fig.add_subplot(gs[1,1]), [abs(s['avg_loss']) for s in lives], [abs(s['avg_loss']) for s in bts],
         'Avg loss abs ($)', '$', '${:.2f}')
    # Ratio win/loss
    rl = [abs(s['avg_win']/s['avg_loss']) if s['avg_loss']!=0 else 0 for s in lives]
    rb = [abs(s['avg_win']/s['avg_loss']) if s['avg_loss']!=0 else 0 for s in bts]
    bar2(fig.add_subplot(gs[1,2]), rl, rb, 'Ratio avg_win/avg_loss', 'x', '{:.2f}x')

    ax_fill = fig.add_subplot(gs[2,0])
    bar2(ax_fill, [s['avg_fill'] for s in lives], [s['avg_fill'] for s in bts],
         'Prix fill moyen', 'price', '{:.3f}', ylim=(0,1.05))
    ax_fill.axhline(0.70, color='#FFC107', linestyle=':', linewidth=1.5, label='cap BTC=0.70')
    ax_fill.axhline(0.55, color='#9C27B0', linestyle=':', linewidth=1.5, label='cap ETH=0.55')
    ax_fill.legend(fontsize=7)

    bar2(fig.add_subplot(gs[2,1]), [s['maxdd'] for s in lives], [s['maxdd'] for s in bts],
         'MaxDD ($)', '$', '${:.0f}')

    rf_l = [min(s['rf'],500) if s['rf']!=float('inf') else 500 for s in lives]
    rf_b = [min(s['rf'],500) if s['rf']!=float('inf') else 500 for s in bts]
    bar2(fig.add_subplot(gs[2,2]), rf_l, rf_b, 'RF (PnL/maxDD)', 'x', '{:.1f}x')

    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Graphique trigger match ───────────────────────────────────────────────────
def plot_trigger_match(btc_windows, out_path, csv_path):
    """Montre accord vol gate BT vs live sur les contrats M5 BTC."""
    cts = load_contracts([csv_path], '5min', 'm5_up_bid','m5_down_bid','m5_up_ask','m5_down_ask')
    precompute_vol(cts, VOL_LBS)
    t0 = btc_windows[0]['ts'].timestamp()
    t1 = btc_windows[-1]['ts'].timestamp()
    cts = [c for c in cts if t0-300 <= c['open_ts'] <= t1+300]

    bt_map = {int(c['open_ts']): c.get('vol_0.5h_net', 0) for c in cts}
    live_map = {int(w['ts'].timestamp()): w for w in btc_windows if w['tf']=='m5'}

    matched = sorted(set(bt_map) & set(live_map))
    if not matched:
        print("  Aucun match trouvé pour trigger_match.png"); return

    bt_vols   = [bt_map[ts] for ts in matched]
    live_vols = [live_map[ts]['vol'] for ts in matched]
    live_pass = [not live_map[ts]['skip'] for ts in matched]
    bt_pass   = [bt_map[ts] >= 60 for ts in matched]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Accord trigger vol gate BT vs Live (M5 BTC)', fontsize=12)

    # Scatter vol BT vs vol live
    ax = axes[0]
    ax.scatter(bt_vols, live_vols, s=8, alpha=0.5, color='#2196F3')
    both_pass = [b and l for b, l in zip(bt_pass, live_pass)]
    only_bt   = [b and not l for b, l in zip(bt_pass, live_pass)]
    only_live = [not b and l for b, l in zip(bt_pass, live_pass)]
    ax.scatter([bt_vols[i] for i,v in enumerate(both_pass) if v],
               [live_vols[i] for i,v in enumerate(both_pass) if v], s=20, color='green', zorder=5, label='Les deux passent')
    ax.scatter([bt_vols[i] for i,v in enumerate(only_bt) if v],
               [live_vols[i] for i,v in enumerate(only_bt) if v], s=20, color='orange', zorder=5, label='BT seul')
    ax.scatter([bt_vols[i] for i,v in enumerate(only_live) if v],
               [live_vols[i] for i,v in enumerate(only_live) if v], s=20, color='purple', zorder=5, label='Live seul')
    m = max(max(bt_vols), max(live_vols))
    ax.plot([0,m],[0,m], 'k--', linewidth=0.8, alpha=0.5)
    ax.axvline(60, color='#2196F3', linestyle=':', linewidth=1)
    ax.axhline(60, color='#F44336', linestyle=':', linewidth=1)
    ax.set_xlabel('Vol 0.5h net BT ($)'); ax.set_ylabel('Vol 0.5h net Live ($)')
    ax.set_title(f'BT vs Live vol\n{sum(both_pass)} accord, {sum(only_bt)} BT-seul, {sum(only_live)} live-seul')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Diff distribution
    ax = axes[1]
    diffs = [lv - bt for bt, lv in zip(bt_vols, live_vols)]
    ax.hist(diffs, bins=40, color='#9C27B0', alpha=0.7, edgecolor='white')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.axvline(float(np.median(diffs)), color='red', linewidth=1.5,
               label=f'med={np.median(diffs):.1f}$')
    ax.set_title('Diff vol (live - BT)\n(precision M5-open vs tick-level)')
    ax.set_xlabel('Vol live - Vol BT ($)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Pie accord
    ax = axes[2]
    n_both  = sum(both_pass); n_obt = sum(only_bt); n_olv = sum(only_live)
    n_none  = len(matched) - n_both - n_obt - n_olv
    sizes = [n_both, n_obt, n_olv, n_none]
    labels = [f'Les deux pass: {n_both}', f'BT seul: {n_obt}', f'Live seul: {n_olv}', f'Aucun: {n_none}']
    colors = ['green','orange','purple','#cccccc']
    ax.pie([s for s in sizes if s > 0],
           labels=[l for s,l in zip(sizes,labels) if s > 0],
           colors=[c for s,c in zip(sizes,colors) if s > 0],
           autopct='%1.0f%%', startangle=90)
    ax.set_title(f'Accord vol gate\nsur {len(matched)} contrats matchés')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Graphique PnL par bucket de fill price ───────────────────────────────────
def plot_pnl_by_fill_bucket(btc_live, btc_bt, eth_live, eth_bt, out_path):
    """Segmente les trades par prix d'entrée et compare WR/PnL moyen."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Performance par prix de fill (bucketing)', fontsize=12)

    BUCKETS = [(0.50, 0.75), (0.75, 0.85), (0.85, 0.92), (0.92, 0.96), (0.96, 1.00)]
    labels = [f'{a:.2f}-{b:.2f}' for a,b in BUCKETS]

    for row, (asset, live_t, bt_t, color_l, color_b) in enumerate([
        ('BTC', btc_live, btc_bt, '#F44336', '#2196F3'),
        ('ETH', eth_live, eth_bt, '#FF9800', '#4CAF50'),
    ]):
        for col, (metric, ylabel) in enumerate([('pnl_avg', 'PnL moyen/trade ($)'), ('wr', 'Win Rate (%)')]):
            ax = axes[row, col]
            x = np.arange(len(BUCKETS)); w = 0.35

            live_vals = []; bt_vals = []; live_n = []; bt_n = []
            for lo, hi in BUCKETS:
                lt = [t for t in live_t if lo <= t.get('avg_fill', 0) < hi]
                bt_t2 = [t for t in bt_t   if lo <= t.get('avg_fill', 0) < hi]
                if metric == 'pnl_avg':
                    live_vals.append(np.mean([t['pnl'] for t in lt]) if lt else 0)
                    bt_vals.append(np.mean([t['pnl'] for t in bt_t2]) if bt_t2 else 0)
                else:
                    live_vals.append(100*sum(1 for t in lt if t.get('won', t['pnl']>0))/max(len(lt),1))
                    bt_vals.append(100*sum(1 for t in bt_t2 if t.get('won', t['pnl']>0))/max(len(bt_t2),1))
                live_n.append(len(lt)); bt_n.append(len(bt_t2))

            bl = ax.bar(x - w/2, live_vals, w, color=color_l, alpha=0.85, label='Live')
            bb = ax.bar(x + w/2, bt_vals,   w, color=color_b, alpha=0.85, label='BT')
            # Annotations (nombre de trades)
            for i, (vl, vb, nl, nb) in enumerate(zip(live_vals, bt_vals, live_n, bt_n)):
                if nl: ax.text(i-w/2, vl + (0.5 if vl>=0 else -2), f'n={nl}', ha='center', fontsize=7, color='darkred')
                if nb: ax.text(i+w/2, vb + (0.5 if vb>=0 else -2), f'n={nb}', ha='center', fontsize=7, color='darkblue')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=8)
            ax.set_title(f'{asset} — {ylabel}'); ax.set_ylabel(ylabel)
            ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
            if metric == 'wr': ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close('all')
    print(f"  -> {out_path}", flush=True)


# ── Matched-trade comparison ──────────────────────────────────────────────────
def matched_comparison(live_trades, bt_trades, asset, out_txt):
    """
    Compare live vs BT uniquement sur les contrats M5 qui ont un match timestamp.
    M15/H1 exclus car le live re-check la vol gate à chaque boundary M5 (≠ BT qui
    check une seule fois à l'open du contrat).
    """
    # M5 seulement : la vol gate est synchrone entre live et BT
    bt_m5   = [t for t in bt_trades   if t['tf'] == 'm5']
    live_m5 = [t for t in live_trades if t['tf'] == 'm5']

    bt_by_open = {int(t['open_ts']): t for t in bt_m5}
    # Floor live ts au bucket 5min pour matcher avec BT open_ts
    live_by_open = {int(pd.Timestamp(t['ts']).floor('5min').timestamp()): t
                    for t in live_m5}

    all_keys     = sorted(set(bt_by_open) | set(live_by_open))
    matched_keys = sorted(set(bt_by_open) & set(live_by_open))
    live_only    = [k for k in all_keys if k in live_by_open and k not in bt_by_open]
    bt_only      = [k for k in all_keys if k in bt_by_open   and k not in live_by_open]

    matched_live = [live_by_open[k] for k in matched_keys]
    matched_bt   = [bt_by_open[k]   for k in matched_keys]

    lines = []
    lines.append(f"\n=== MATCHED TRADES {asset} (M5 uniquement) ===")
    lines.append(f"  BT M5 trades            : {len(bt_m5)}  (total tous TF : {len(bt_trades)})")
    lines.append(f"  Live M5 trades          : {len(live_m5)}  (total tous TF : {len(live_trades)})")
    lines.append(f"  Contrats M5 matchés     : {len(matched_keys)}")
    lines.append(f"  Live-only M5            : {len(live_only)}  "
                 f"(vol gate ok live mais pas en BT, ou bot speed)")
    lines.append(f"  BT-only M5              : {len(bt_only)}  "
                 f"(vol gate ok BT mais bot hors-ligne / timing off)")
    lines.append(f"  NOTE: M15/H1 exclus — le live re-check vol gate chaque 5min "
                 f"(BT check seulement à l'open du contrat)")

    if matched_live:
        live_pnl   = sum(t['pnl'] for t in matched_live)
        bt_pnl     = sum(t['pnl'] for t in matched_bt)
        live_wr    = 100 * sum(1 for t in matched_live if t['pnl'] > 0) / len(matched_live)
        bt_wr      = 100 * sum(1 for t in matched_bt   if t['pnl'] > 0) / len(matched_bt)
        live_fill  = float(np.mean([t['avg_fill'] for t in matched_live]))
        bt_fill    = float(np.mean([t['avg_fill'] for t in matched_bt]))
        agree_sign = sum(1 for l, b in zip(matched_live, matched_bt)
                         if (l['pnl'] > 0) == (b['pnl'] > 0))
        lines.append(f"\n  Sur les {len(matched_keys)} contrats M5 matchés :")
        lines.append(f"    PnL live : ${live_pnl:.2f}   WR live : {live_wr:.1f}%   fill moy : {live_fill:.3f}")
        lines.append(f"    PnL BT   : ${bt_pnl:.2f}   WR BT   : {bt_wr:.1f}%   fill moy : {bt_fill:.3f}")
        lines.append(f"    Même signe W/L : {agree_sign}/{len(matched_keys)} ({100*agree_sign/len(matched_keys):.0f}%)")
        if abs(bt_pnl) > 0.01:
            lines.append(f"    PnL gap live vs BT : ${live_pnl - bt_pnl:.2f}  ({(live_pnl/bt_pnl - 1)*100:+.0f}%)")
        else:
            lines.append(f"    PnL gap live vs BT : ${live_pnl - bt_pnl:.2f}")
    for l in lines:
        print(l, flush=True)
        out_txt.write(l + "\n")
    return matched_live, matched_bt


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Parse live ===", flush=True)
    btc_live, btc_fills, btc_windows = parse_live(LIVE_DIR / "btc")
    eth_live, eth_fills, eth_windows = parse_live(LIVE_DIR / "eth")
    btc_days = (btc_live[-1]['ts'].timestamp() - btc_live[0]['ts'].timestamp()) / 86400 if btc_live else 1
    eth_days = (eth_live[-1]['ts'].timestamp() - eth_live[0]['ts'].timestamp()) / 86400 if eth_live else 1
    print(f"  BTC: {len(btc_live)} trades sur {btc_days:.1f}j", flush=True)
    print(f"  ETH: {len(eth_live)} trades sur {eth_days:.1f}j", flush=True)

    print("\n=== Load BT ===", flush=True)
    ctf_btc = load_bt("BTC", CFG_BTC, BTC_SIZE)
    ctf_eth = load_bt("ETH", CFG_ETH, ETH_SIZE)

    print("  BT trades BTC...", flush=True)
    btc_bt = run_bt_trades(ctf_btc, CFG_BTC, BTC_SIZE)
    print("  BT trades ETH...", flush=True)
    eth_bt = run_bt_trades(ctf_eth, CFG_ETH, ETH_SIZE)
    print(f"  BT BTC: {len(btc_bt)} trades  ETH: {len(eth_bt)} trades", flush=True)

    print("  BT equity...", flush=True)
    bt_btc_cum, bw, bl, bt_hi, bt_nh = run_bt_equity(ctf_btc, CFG_BTC, BTC_SIZE)
    bt_eth_cum, ew, el, _, _         = run_bt_equity(ctf_eth, CFG_ETH, ETH_SIZE)
    bt_days = bt_nh / 24

    s_bl = compute_stats(btc_live, 'BTC Live',    btc_days)
    s_bb = compute_stats(btc_bt,   'BTC BT',      bt_days)
    s_el = compute_stats(eth_live, 'ETH Live',    eth_days)
    s_eb = compute_stats(eth_bt,   'ETH BT',      bt_days)

    # Stats + diagnostic
    print("\n=== STATS ===", flush=True)
    with open(OUT_DIR / "stats.txt", "w", encoding="utf-8") as f:
        def pr(l): print(l, flush=True); f.write(l+"\n")
        pr(f"=== LIVE vs BACKTEST  (depuis {pd.Timestamp(CUTOFF_TS, unit='s', tz='UTC')}) ===\n")
        for s in [s_bl, s_bb, s_el, s_eb]:
            rf_s = f"{s['rf']:.1f}x" if s['rf']!=float('inf') else "inf"
            pr(f"  {s['label']}: T={s['n']}  WR={s['wr']:.1f}%  PnL/j=${s['pnl_j']:.1f}  "
               f"AvgWin=${s['avg_win']:.2f}  AvgLoss=${s['avg_loss']:.2f}  "
               f"AvgFill={s['avg_fill']:.3f}  maxDD=${s['maxdd']:.0f}  RF={rf_s}")
        pr("")
        pr("=== DIAGNOSTIC FILL PRICE ===")
        af_btc = s_bl['avg_fill'] or 0.96
        af_eth = s_el['avg_fill'] or 0.96
        pr(f"  BTC : fill live moy={af_btc:.3f}  vs cap BT=0.70")
        pr(f"  ETH : fill live moy={af_eth:.3f}  vs cap ETH=0.55")
        pr(f"  BTC au cap 0.70 : win/ordre = $100*(0.99/0.70-1) = ${100*(0.99/0.70-1):.1f}")
        pr(f"  BTC au fill {af_btc:.2f} : win/ordre = $100*(0.99/{af_btc:.2f}-1) = ${100*(0.99/af_btc-1):.2f}")
        pr(f"  ETH au cap 0.55 : win/ordre = $150*(0.99/0.55-1) = ${150*(0.99/0.55-1):.1f}")
        pr(f"  ETH au fill {af_eth:.2f} : win/ordre = $150*(0.99/{af_eth:.2f}-1) = ${150*(0.99/af_eth-1):.2f}")
        pr(f"  -> Voir pnl_by_fill_bucket.png")

        # Matched comparison
        matched_comparison(btc_live, btc_bt, 'BTC', f)
        matched_comparison(eth_live, eth_bt, 'ETH', f)

    print("\n=== Graphiques ===", flush=True)
    plot_equity_comparison(btc_live, bt_btc_cum, bt_hi, bt_nh, bt_days,
                           f'BTC — Live vs BT  (size $100x2)', OUT_DIR/"equity_btc.png", '#2196F3')
    plot_equity_comparison(eth_live, bt_eth_cum, bt_hi, bt_nh, bt_days,
                           f'ETH — Live vs BT  (size $150x2)', OUT_DIR/"equity_eth.png", '#4CAF50')
    plot_combined(btc_live, eth_live, bt_btc_cum, bt_eth_cum, bt_hi, bt_nh, bt_days,
                  OUT_DIR/"equity_combined.png")
    plot_fill_prices(btc_fills, eth_fills, btc_bt, eth_bt, OUT_DIR/"fill_prices.png")
    plot_pnl_per_trade(btc_live, btc_bt, eth_live, eth_bt, OUT_DIR/"pnl_per_trade.png")
    plot_stats_comparison(s_bl, s_bb, s_el, s_eb, OUT_DIR/"stats_comparison.png")
    plot_trigger_match(btc_windows, OUT_DIR/"trigger_match.png",
                       str(CSV_DIR/"BTC.csv"))
    plot_pnl_by_fill_bucket(btc_live, btc_bt, eth_live, eth_bt,
                            OUT_DIR/"pnl_by_fill_bucket.png")
    print(f"\n  -> {OUT_DIR}", flush=True)
