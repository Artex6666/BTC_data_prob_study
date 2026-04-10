"""
gen_live_vs_bt_recap_eth.py — Live vs BT pour ETH
Config : slope0.2int0trend0.5  (slope=0.2, trend_2h>0.5, min_bid=0.60)
Perte manuelle : 2026-04-06T23:15 UTC — DOWN fill @ 0.64 non logué, résolu UP → -$149.76

Sortie : live_vs_bt/live_vs_bt_recap_eth.png
"""
import json, sys
from pathlib import Path
from datetime import timezone
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chart_utils as cu
from chart_utils import load_contracts, precompute_vol, simulate_trade_log, TIMEFRAMES, VOL_LBS

BASE     = Path(__file__).resolve().parent.parent
LIVE_DIR = BASE / "reportLive" / "safeChase" / "slope0.2int0trend0.5" / "eth"
CSV_DIR  = BASE / "reportLive" / "safeChase"
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF    = pd.Timestamp("2026-04-05T21:45:00", tz="UTC")
CUTOFF_TS = CUTOFF.timestamp()

# Config ETH live : slope=0.2, no VRS, trend filter 2h>0.50, min_bid=0.60
CFG_ETH = dict(
    curve='linear', slope=0.20, intercept=0.0,
    eq_cap=0.60,          # correspond au min_bid=0.60 du live
    vol_lb_h=2.0, vol_thresh=0.50, vol_type='trend',
    max_losses_cb=None, max_orders=1,
)
ETH_SIZE = 150.0

# Perte manuelle non loguée : 2026-04-06T23:10-23:15 UTC
# Chase DOWN @ 0.64 (234 shares = $149.76) placé à 23:14:59.972
# Fill confirmé en exchange mais pas dans window_ended (race condition)
# Contrat résolu UP → pnl = -149.76
MANUAL_LOSS_TS  = pd.Timestamp("2026-04-06T23:15:00", tz="UTC").to_pydatetime()
MANUAL_LOSS_PNL = -149.76   # DOWN fill @ 0.64 × 234 shares, UP wins


# ── Parse live JSONL ──────────────────────────────────────────────────────────
def parse_live(asset_dir):
    windows = []
    for tf in ['m5', 'm15', 'h1']:
        tf_dir = asset_dir / tf
        if not tf_dir.exists():
            continue
        for f in sorted(tf_dir.glob('*.jsonl')):
            cur = None
            fills_prices = []
            tradable = False
            for line in f.open(encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except: continue
                ev = d.get('event', '')

                if ev == 'window_open':
                    ts = pd.to_datetime(d['ts']).to_pydatetime().replace(tzinfo=timezone.utc)
                    if ts.timestamp() < CUTOFF_TS:
                        cur = None; fills_prices = []; tradable = False; break
                    cur = dict(ts=ts, tf=tf)
                    fills_prices = []; tradable = False

                elif ev == 'window_start' and cur:
                    tradable = True

                elif ev in ('vol_gate_skip', 'eth_trend_gate_skip') and cur:
                    tradable = False

                elif ev == 'fill' and cur:
                    p = float(d.get('price', 0))
                    if p > 0: fills_prices.append(p)

                elif ev == 'window_ended' and cur:
                    up_s  = float(d.get('up_shares',  0))
                    dn_s  = float(d.get('down_shares', 0))
                    up_c  = float(d.get('up_cost',    0))
                    dn_c  = float(d.get('down_cost',  0))
                    shares = up_s + dn_s
                    cost   = up_c + dn_c
                    traded = shares > 1e-9

                    ts_end   = pd.to_datetime(d['ts']).to_pydatetime().replace(tzinfo=timezone.utc)
                    start_sp = float(d.get('start_spot', 0))
                    end_sp   = float(d.get('end_spot',   0))

                    # Perte manuelle 23:15 UTC : fill non logué, résolu UP
                    is_manual_loss = (
                        abs((ts_end - MANUAL_LOSS_TS).total_seconds()) < 5
                        and tf == 'm5'
                    )
                    if is_manual_loss and not traded:
                        # Injecter le fill manquant
                        up_s, dn_s = 0.0, 234.0
                        up_c, dn_c = 0.0, 149.76
                        shares = dn_s; cost = dn_c
                        traded = True
                        up_wins = True   # résolu UP → DOWN perd
                    elif start_sp > 0 and end_sp > 0:
                        up_wins = end_sp > start_sp
                    else:
                        up_wins = bool(d.get('epnl_up_wins', 1))

                    if traded:
                        pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c)
                        won = pnl > 0
                    else:
                        pnl = 0.0; won = None

                    avg_fill = float(np.mean(fills_prices)) if fills_prices else (
                        cost / shares if shares > 1e-9 else None
                    )

                    windows.append(dict(
                        ts=cur['ts'], tf=tf,
                        tradable=tradable,
                        traded=traded,
                        won=won,
                        pnl=pnl,
                        avg_fill=avg_fill,
                        shares=shares,
                        cost=cost,
                        manual_loss=is_manual_loss if 'is_manual_loss' in dir() else False,
                    ))
                    cur = None; fills_prices = []; tradable = False

    windows.sort(key=lambda x: x['ts'])
    trades = [w for w in windows if w['traded']]
    return windows, trades


# ── Run BT ────────────────────────────────────────────────────────────────────
def run_bt(cfg, size):
    csv = [str(CSV_DIR / "ETH.csv")]
    m5_ref = None
    all_trades = []

    for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
        cts = load_contracts(csv, tf_floor, b1, b2, a1, a2)
        if tf_floor == '5min':
            precompute_vol(cts, VOL_LBS)
            m5_ref = cts
        else:
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)

        tf_name = ['m5', 'm15', 'h1'][idx]
        vol_key = (f"vol_{cfg['vol_lb_h']:g}h_{cfg.get('vol_type','range')}"
                   if cfg.get('vol_lb_h') else None)

        orig = cu.BASE_SIZE; cu.BASE_SIZE = float(size)
        for c in cts:
            if c.get('open_ts', 0) < CUTOFF_TS: continue
            if vol_key and c.get(vol_key, 0.0) < cfg.get('vol_thresh', 0): continue
            won, pnl, fills = simulate_trade_log(c, cfg)
            if won is None: continue
            avg_fill = float(np.mean([f['price'] for f in fills])) if fills else 0.0
            ts = c['ce']
            all_trades.append(dict(ts=ts, tf=tf_name, pnl=pnl, won=won, avg_fill=avg_fill))
        cu.BASE_SIZE = orig

    all_trades.sort(key=lambda x: x['ts'])
    return all_trades


# ── Equity helpers ────────────────────────────────────────────────────────────
def trades_to_equity(trades):
    if not trades: return [], []
    t0 = trades[0]['ts']
    t0_ts = t0.timestamp() if hasattr(t0, 'timestamp') else float(t0)
    xs = [(t['ts'].timestamp() - t0_ts) / 3600 if hasattr(t['ts'], 'timestamp')
          else (float(t['ts']) - t0_ts) / 3600 for t in trades]
    ys = list(np.cumsum([t['pnl'] for t in trades]))
    return xs, ys

def bt_to_equity_hours(trades, ref_start_ts):
    if not trades: return [], []
    xs, ys = [], []; cum = 0
    for t in trades:
        ts = t['ts'].timestamp() if hasattr(t['ts'], 'timestamp') else float(t['ts'])
        cum += t['pnl']
        xs.append((ts - ref_start_ts) / 3600)
        ys.append(cum)
    return xs, ys

def compute_stats(windows, trades, label, days):
    tradable = sum(1 for w in windows if w['tradable'])
    n = len(trades)
    wins   = [t for t in trades if t.get('won')]
    losses = [t for t in trades if not t.get('won') and t.get('won') is not None]
    pnl    = sum(t['pnl'] for t in trades)
    fills  = [t['avg_fill'] for t in trades if t.get('avg_fill')]
    cum  = np.cumsum([t['pnl'] for t in trades]) if trades else np.array([0.0])
    peak = np.maximum.accumulate(cum)
    maxdd = float((peak - cum).max()) if len(cum) else 0.0
    return dict(label=label, days=days, tradable=tradable, n=n,
                wins=len(wins), losses=len(losses),
                wr=100*len(wins)/n if n else 0,
                pnl=pnl, pnl_j=pnl/max(days,0.01),
                avg_fill=float(np.mean(fills)) if fills else 0, maxdd=maxdd)


# ── Main ──────────────────────────────────────────────────────────────────────
print("=== Parse live ETH ===", flush=True)
eth_windows, eth_live = parse_live(LIVE_DIR)
tradables = sum(1 for w in eth_windows if w['tradable'])
print(f"  ETH live: {len(eth_live)} trades / {tradables} tradables", flush=True)
print(f"  Pertes : {len([t for t in eth_live if not t.get('won') and t.get('won') is not None])}", flush=True)

print("=== Run BT ETH ===", flush=True)
eth_bt = run_bt(CFG_ETH, ETH_SIZE)
print(f"  BT ETH: {len(eth_bt)} trades depuis cutoff", flush=True)

now_ts   = pd.Timestamp.now(tz='UTC')
eth_days = (now_ts - CUTOFF).total_seconds() / 86400
live_end_ts = (max(t['ts'].timestamp() for t in eth_live)
               if eth_live else now_ts.timestamp())
live_span_h = (live_end_ts - CUTOFF_TS) / 3600
live_span_d = live_span_h / 24

s_el = compute_stats(eth_windows, eth_live, 'ETH Live', live_span_d)
s_eb = compute_stats([], eth_bt,   'ETH BT',  live_span_d)

ref_start = CUTOFF_TS
lx, ly = trades_to_equity(eth_live)
bx, by = bt_to_equity_hours(eth_bt, ref_start)

# ── Graphique ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50, height_ratios=[3, 1.5, 1.2])

COLOR_LIVE = '#F44336'
COLOR_BT   = '#90CAF9'

ax_eq = fig.add_subplot(gs[0])
if bx:
    ax_eq.plot(bx, by, color=COLOR_BT, linewidth=1.8, linestyle='--',
               label=f'BT  {by[-1]:+.1f}$  WR={s_eb["wr"]:.1f}%  T={s_eb["n"]}', zorder=2)
if lx:
    ax_eq.plot(lx, ly, color=COLOR_LIVE, linewidth=2.2,
               label=f'Live  {ly[-1]:+.1f}$  WR={s_el["wr"]:.1f}%  T={s_el["n"]}', zorder=3)
    ax_eq.scatter(lx, ly, color=COLOR_LIVE, s=20, zorder=4, alpha=0.7)
ax_eq.axhline(0, color='gray', linewidth=0.6, linestyle='--')
ax_eq.set_title(f'ETH — Live vs BT  ($150×1, slope=0.2 int=0 trend2h>0.5 min_bid=0.60)', fontsize=11, fontweight='bold')
ax_eq.set_xlabel(f'Heures depuis {CUTOFF.strftime("%Y-%m-%d %H:%M")} UTC', fontsize=9)
ax_eq.set_ylabel('PnL cumulé ($)', fontsize=9)
ax_eq.legend(fontsize=9); ax_eq.grid(alpha=0.25)
xticks = np.arange(0, live_span_h + 2, 4)
ax_eq.set_xticks(xticks)
ax_eq.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
ax_eq.set_xlim(-0.2, live_span_h + 0.5)

ax_dd = fig.add_subplot(gs[1])
if ly:
    cum_l = np.array(ly); dd_l = np.maximum.accumulate(cum_l) - cum_l
    ax_dd.fill_between(lx, 0, -dd_l, color=COLOR_LIVE, alpha=0.4, label='DD Live')
if by:
    cum_b = np.array(by); dd_b = np.maximum.accumulate(cum_b) - cum_b
    ax_dd.fill_between(bx, 0, -dd_b, color=COLOR_BT, alpha=0.4, label='DD BT')
ax_dd.axhline(0, color='gray', linewidth=0.5)
ax_dd.set_ylabel('Drawdown ($)'); ax_dd.legend(fontsize=8); ax_dd.grid(alpha=0.25)
ax_dd.set_xticks(xticks)
ax_dd.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
ax_dd.set_xlim(-0.2, live_span_h + 0.5)

ax_tbl = fig.add_subplot(gs[2])
ax_tbl.axis('off')
cols = ['', 'Tradables', 'Tradés', 'Wins', 'Losses', 'WR', 'Fill avg', 'PnL total', 'PnL/h', 'MaxDD']
rows_data = []
for s in [s_el, s_eb]:
    pnl_h = s['pnl'] / max(live_span_h, 0.01)
    rows_data.append([
        s['label'],
        str(s['tradable']) if s['tradable'] else '—',
        str(s['n']), str(s['wins']), str(s['losses']),
        f"{s['wr']:.1f}%",
        f"{s['avg_fill']:.3f}" if s['avg_fill'] else '—',
        f"${s['pnl']:+.1f}",
        f"${pnl_h:+.2f}/h",
        f"${s['maxdd']:.0f}",
    ])

tbl = ax_tbl.table(cellText=rows_data, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.4)
for j in range(len(cols)):
    tbl[0, j].set_facecolor('#37474F')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
tbl[1, 0].set_facecolor('#FFEBEE'); tbl[2, 0].set_facecolor('#E3F2FD')
for j in range(1, len(cols)):
    tbl[1, j].set_facecolor('#FFEBEE'); tbl[2, j].set_facecolor('#E3F2FD')

ax_tbl.set_title(
    f"Stats  {CUTOFF.strftime('%Y-%m-%d %H:%M')} UTC  →  {live_span_h:.1f}h"
    f"  [perte 23:15 UTC Apr6 injectée manuellement -$149.76]",
    fontsize=9, pad=6, fontweight='bold'
)
fig.suptitle('ETH Live vs Backtest — slope=0.2 int=0 trend2h>0.5  ($150×1)',
             fontsize=13, fontweight='bold', y=0.99)

out = OUT_DIR / "live_vs_bt_recap_eth.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"\n-> {out}", flush=True)

print(f"\n=== RECAP ETH ({live_span_h:.1f}h depuis cutoff) ===")
for s in [s_el, s_eb]:
    pnl_h = s['pnl'] / max(live_span_h, 0.01)
    print(f"  {s['label']:<12} T={s['n']:>3}  W={s['wins']} L={s['losses']}  "
          f"WR={s['wr']:.1f}%  fill={s['avg_fill']:.3f}  "
          f"PnL={s['pnl']:+.1f}$  ({pnl_h:+.2f}$/h)  maxDD=${s['maxdd']:.0f}")

losses_live = [t for t in eth_live if not t.get('won') and t.get('won') is not None]
if losses_live:
    print(f"\n  Pertes live ({len(losses_live)}) :")
    for t in losses_live:
        flag = "[MANUAL]" if t.get('manual_loss') else ""
        print(f"    {t['ts'].strftime('%Y-%m-%d %H:%M')} UTC  tf={t['tf']}  "
              f"pnl={t['pnl']:+.2f}$  fill={t.get('avg_fill',0):.3f}  {flag}")
