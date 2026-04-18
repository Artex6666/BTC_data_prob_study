"""
Comparatif Live vs BT sur les contrats EXACTEMENT vus par le live (Apr 13-16).
On reconstruit chaque contrat depuis les JSONL (dashboard_snapshots) et on
relance simulate_trade_log dessus → BT "apple-to-apple".

Seuls logs dispo pour Apr 13-16 : slope7cap75/btc/h1
Sorties : live_vs_bt/apr13_16_btc.png
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
from chart_utils import simulate_trade_log

BASE      = Path(__file__).resolve().parent.parent
BTC_DIR   = BASE / "reportLive" / "safeChase" / "slope7cap75" / "btc"
OUT_DIR   = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_UTC = pd.Timestamp("2026-04-13T00:00:00", tz="UTC")
END_UTC   = pd.Timestamp("2026-04-17T00:00:00", tz="UTC")
START_TS  = START_UTC.timestamp()
END_TS    = END_UTC.timestamp()

DAYS_STR = ['2026-04-13','2026-04-14','2026-04-15','2026-04-16']
DAYS_LBL = ['13 avr','14 avr','15 avr','16 avr']

CFG_BTC = dict(curve='linear', slope=7.0, intercept=0.0, eq_cap=0.75,
               vol_lb_h=1, vol_thresh=60.0, vol_type='net',
               max_losses_cb=None, max_orders=2)
BTC_SIZE = 120.0

# Settlement (toutes TFs)
settlement = {}
for tf_csv, tf_off in [('5min', 300), ('15min', 900), ('1h', 3600)]:
    for open_ts, won_up in cu.load_settlement_outcomes('btc', tf=tf_csv).items():
        settlement[(open_ts, tf_off)] = won_up


TF_INFO = {
    'm5':  ('5min',  300),
    'm15': ('15min', 900),
    'h1':  ('1h',   3600),
}

def parse_jsonl_dir(tf_dir: Path, tf_name: str):
    """
    Pour chaque contrat dans les JSONL :
    - Reconstruit le contrat depuis les dashboard_snapshots
    - Calcule live PnL depuis fills réels + settlement
    - Calcule BT PnL via simulate_trade_log sur les mêmes ticks

    Retourne list of dict par contrat.
    """
    _, tf_offset = TF_INFO[tf_name]
    results = []

    for f in sorted(tf_dir.glob('*.jsonl')):
        # State machine par window
        cur_open_ts  = None
        cur_start_ts = None
        open_price   = None
        tradable     = False
        ticks_ts     = []   # pd.Timestamp
        ticks_spot   = []
        ticks_up_bid = []
        ticks_dn_bid = []
        fills_live   = []   # (side, price, shares, cost)
        up_shares = dn_shares = up_cost = dn_cost = 0.0
        settlement_up = None
        window_ts_obj = None   # pd.Timestamp de l'open

        for line in f.open(encoding='utf-8', errors='replace'):
            line = line.strip()
            if not line: continue
            try: d = json.loads(line)
            except: continue
            ev = d.get('event', '')

            if ev == 'window_open':
                # Reset
                cur_open_ts  = None; cur_start_ts = None
                open_price   = None; tradable = False
                ticks_ts     = []; ticks_spot = []; ticks_up_bid = []; ticks_dn_bid = []
                fills_live   = []
                up_shares = dn_shares = up_cost = dn_cost = 0.0
                settlement_up = None
                ts = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc)
                if ts.timestamp() < START_TS or ts.timestamp() >= END_TS:
                    cur_open_ts = None; continue
                # open_ts = floor au timeframe
                floor_td = pd.Timedelta(seconds=tf_offset)
                cur_open_ts = int(pd.Timestamp(ts).floor(floor_td).timestamp())
                window_ts_obj = ts

            elif ev == 'window_start' and cur_open_ts is not None:
                tradable   = True
                open_price = float(d.get('start_price', 0))
                cur_start_ts = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc)

            elif ev in ('vol_gate_skip', 'eth_trend_gate_skip') and cur_open_ts is not None:
                tradable = False

            elif ev == 'dashboard_snapshot' and cur_open_ts is not None and tradable:
                ts_tick = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc)
                spot    = float(d.get('price_binance', 0) or d.get('price_synthetic', 0) or 0)
                ub      = float(d.get('up_bid',   0) or 0)
                db      = float(d.get('down_bid', 0) or 0)
                if spot > 0:
                    ticks_ts.append(np.datetime64(ts_tick.replace(tzinfo=None)))
                    ticks_spot.append(spot)
                    ticks_up_bid.append(ub)
                    ticks_dn_bid.append(db)

            elif ev == 'fill' and cur_open_ts is not None:
                side  = d.get('side', '')
                price = float(d.get('price', 0))
                size  = float(d.get('size', 0))    # shares
                cost  = float(d.get('cost', price * size / (1/price) if price > 0 else 0))
                # cost = notional (USD) ≈ size * price? ou d.get('reserved_notional')
                fills_live.append(dict(side=side, price=price))

            elif ev in ('window_ended', 'window_settled') and cur_open_ts is not None:
                up_shares = float(d.get('up_shares',  0))
                dn_shares = float(d.get('down_shares', 0))
                up_cost   = float(d.get('up_cost',    0))
                dn_cost   = float(d.get('down_cost',  0))

                # Settlement
                sett_key = (cur_open_ts, tf_offset)
                if sett_key in settlement:
                    settlement_up = settlement[sett_key]
                else:
                    s_sp = float(d.get('start_spot', 0))
                    e_sp = float(d.get('end_spot',   0))
                    settlement_up = (e_sp >= s_sp) if (s_sp > 0 and e_sp > 0) else \
                                    bool(d.get('epnl_up_wins', 1))

                # PnL live
                shares_total = up_shares + dn_shares
                traded = shares_total > 1e-9
                if traded:
                    pnl_live = (up_shares - up_cost - dn_cost) if settlement_up \
                               else (dn_shares - dn_cost - up_cost)
                    avg_fill_live = (up_cost + dn_cost) / shares_total
                else:
                    pnl_live = 0.0; avg_fill_live = 0.0

                # BT sur les ticks du live
                pnl_bt = None; avg_fill_bt = None; won_bt = None
                if tradable and len(ticks_ts) >= 3 and open_price:
                    # Expiry = open_ts + 3600s
                    ce_np = np.datetime64(
                        pd.Timestamp(cur_open_ts + tf_offset, unit='s').to_pydatetime().replace(tzinfo=None)
                    )
                    c = dict(
                        ce     = ce_np,
                        ts     = np.array(ticks_ts),
                        spot   = np.array(ticks_spot, dtype=float),
                        op     = open_price,
                        up_bid = np.array(ticks_up_bid, dtype=float),
                        down_bid = np.array(ticks_dn_bid, dtype=float),
                        _settlement_won_up = settlement_up,
                    )
                    orig = cu.BASE_SIZE; cu.BASE_SIZE = BTC_SIZE
                    try:
                        won_bt, pnl_bt, bt_fills = simulate_trade_log(c, CFG_BTC)
                    except Exception as e:
                        won_bt = pnl_bt = None; bt_fills = []
                    cu.BASE_SIZE = orig

                    if won_bt is not None and bt_fills:
                        avg_fill_bt = float(np.mean([f['price'] for f in bt_fills]))

                if cur_open_ts is None: continue
                day = pd.Timestamp(cur_open_ts + tf_offset, unit='s', tz='UTC').strftime('%Y-%m-%d')

                results.append(dict(
                    day          = day,
                    open_ts      = cur_open_ts,
                    window_ts    = window_ts_obj,
                    tradable     = tradable,
                    traded_live  = traded,
                    pnl_live     = pnl_live,
                    avg_fill_live= avg_fill_live,
                    won_live     = (pnl_live > 0) if traded else None,
                    pnl_bt       = pnl_bt if pnl_bt is not None else 0.0,
                    avg_fill_bt  = avg_fill_bt,
                    won_bt       = won_bt,
                    settlement_up= settlement_up,
                    n_ticks      = len(ticks_ts),
                ))
                cur_open_ts = None

    return results


# ── Run ───────────────────────────────────────────────────────────────────────
contracts = []
for tf_name in ['m5', 'm15', 'h1']:
    tf_dir = BTC_DIR / tf_name
    if not tf_dir.exists():
        print(f"  {tf_name}: dossier introuvable, skip", flush=True); continue
    print(f"Parsing JSONL {tf_name} ...", flush=True)
    cts = parse_jsonl_dir(tf_dir, tf_name)
    print(f"  {len(cts)} contrats parsés", flush=True)
    contracts.extend(cts)
print(f"  Total : {len(contracts)} contrats parsés (m5+m15+h1)", flush=True)

# Agréger par jour
by_day_live = defaultdict(float)
by_day_bt   = defaultdict(float)
n_traded_live = defaultdict(int)
n_traded_bt   = defaultdict(int)
n_tradable    = defaultdict(int)

for r in contracts:
    d = r['day']
    if d not in DAYS_STR: continue
    if r['tradable']:
        n_tradable[d] += 1
    if r['traded_live']:
        by_day_live[d]    += r['pnl_live']
        n_traded_live[d]  += 1
    if r['won_bt'] is not None:
        by_day_bt[d]   += r['pnl_bt']
        n_traded_bt[d] += 1

print("\n=== SYNTHESE PAR JOUR ===")
print(f"{'Jour':<12} {'Tradable':>9} {'Live trades':>12} {'Live PnL':>10} {'BT trades':>10} {'BT PnL':>10}")
for d, lbl in zip(DAYS_STR, DAYS_LBL):
    print(f"  {lbl:<10} {n_tradable.get(d,0):>9} {n_traded_live.get(d,0):>12} "
          f"{by_day_live.get(d,0):>+10.1f} {n_traded_bt.get(d,0):>10} "
          f"{by_day_bt.get(d,0):>+10.1f}")

# Détail des contrats tradés live
print("\n=== DETAIL CONTRATS TRADES (live) ===")
traded = [r for r in contracts if r['traded_live'] and r['day'] in DAYS_STR]
for r in sorted(traded, key=lambda x: x['open_ts']):
    ts_str = pd.Timestamp(r['open_ts'], unit='s', tz='UTC').strftime('%m-%d %H:%M')
    bt_str = f"{r['pnl_bt']:+.1f}" if r['won_bt'] is not None else "no_trig"
    sett   = "UP" if r['settlement_up'] else "DOWN"
    print(f"  {ts_str}  live={r['pnl_live']:+.1f}  BT={bt_str}  "
          f"fill_live={r['avg_fill_live']:.3f}  fill_bt={r['avg_fill_bt'] or 0:.3f}  settle={sett}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#0e1117')
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                       height_ratios=[2.5, 2.0, 1.3])

COLOR_BT   = '#90CAF9'
COLOR_LIVE = '#F44336'
COLOR_ACC  = '#FFD700'

x = np.arange(len(DAYS_STR))

# ── Panel 1 : equity cumulée (par trade chronologique) ───────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#0e1117')

# Construire equity depuis les contrats dans l'ordre chronologique
traded_sorted = sorted([r for r in contracts if r['day'] in DAYS_STR],
                       key=lambda r: r['open_ts'])

# Live equity (seulement contrats tradés)
live_eq_xs, live_eq_ys = [], []
bt_eq_xs,   bt_eq_ys   = [], []
cum_live = cum_bt = 0.0
t0 = START_TS

for r in traded_sorted:
    rel_h = (r['open_ts'] + 3600 - t0) / 3600
    if r['traded_live']:
        cum_live += r['pnl_live']
        live_eq_xs.append(rel_h); live_eq_ys.append(cum_live)
    if r['won_bt'] is not None:
        cum_bt += r['pnl_bt']
        bt_eq_xs.append(rel_h); bt_eq_ys.append(cum_bt)

if bt_eq_xs:
    ax1.plot(bt_eq_xs, bt_eq_ys, color=COLOR_BT, lw=2.0, ls='--',
             label=f'BT (mêmes contrats)  {cum_bt:+.0f}USD  ({sum(n_traded_bt.values())} trades)', zorder=3)
if live_eq_xs:
    ax1.plot(live_eq_xs, live_eq_ys, color=COLOR_LIVE, lw=2.2,
             label=f'Live BTC h1  {cum_live:+.0f}USD  ({sum(n_traded_live.values())} trades)', zorder=4)
    ax1.scatter(live_eq_xs, live_eq_ys, color=COLOR_LIVE, s=30, zorder=5)

# Lignes minuit
for i in range(1, 4):
    h_off = (pd.Timestamp(DAYS_STR[i], tz='UTC').timestamp() - t0) / 3600
    ax1.axvline(h_off, color='#555', lw=0.9, ls=':')
    ax1.text(h_off + 0.3, min(bt_eq_ys + live_eq_ys + [0]) if (bt_eq_ys or live_eq_ys) else 0,
             DAYS_LBL[i], color='#888', fontsize=8)

ax1.axhline(0, color='#555', lw=0.6, ls='--')
ax1.set_xlabel('Heures depuis 13 avr 00h UTC', color='#aaa', fontsize=9)
ax1.set_ylabel('PnL cumulé (USD)', color='#aaa', fontsize=9)
ax1.set_title('BTC (m5+m15+h1) — Live vs BT (mêmes contrats, mêmes ticks)', color='white', fontsize=12, pad=8)
ax1.legend(fontsize=9, framealpha=0.25, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
ax1.tick_params(colors='#aaa'); ax1.grid(alpha=0.2)
for sp in ax1.spines.values(): sp.set_edgecolor('#333')

# ── Panel 2 : barres PnL/jour Live vs BT ─────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#0e1117')
bt_vals = [by_day_bt.get(d, 0.0)   for d in DAYS_STR]
lv_vals = [by_day_live.get(d, 0.0) for d in DAYS_STR]
ax2.bar(x - 0.2, bt_vals, 0.38, color=COLOR_BT,   alpha=0.85, label='BT h1 (mêmes contrats)', zorder=3)
ax2.bar(x + 0.2, lv_vals, 0.38, color=COLOR_LIVE,  alpha=0.85, label='Live BTC h1', zorder=3)
for i, (bv, lv) in enumerate(zip(bt_vals, lv_vals)):
    if abs(bv) > 0.5: ax2.text(i-0.2, bv + (3 if bv >= 0 else -12), f'{bv:+.0f}',
                                ha='center', fontsize=8, color='white')
    if abs(lv) > 0.5: ax2.text(i+0.2, lv + (3 if lv >= 0 else -12), f'{lv:+.0f}',
                                ha='center', fontsize=8, color='white')
ax2.axhline(0, color='#555', lw=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(DAYS_LBL, color='#ccc', fontsize=10)
ax2.set_title('PnL/jour — BT vs Live (BTC m5+m15+h1)', color='white', fontsize=11)
ax2.legend(fontsize=8, framealpha=0.25, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
ax2.tick_params(colors='#aaa'); ax2.grid(axis='y', alpha=0.2)
for sp in ax2.spines.values(): sp.set_edgecolor('#333')

# ── Panel 3 : BT BTC vs Compte Polymarket total ───────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#0e1117')
live_user = [1140.0, 1200.0, 462.49, 155.90]
ax3.bar(x - 0.2, bt_vals, 0.38, color=COLOR_BT,   alpha=0.85, label='BT BTC h1', zorder=3)
ax3.bar(x + 0.2, live_user, 0.38, color=COLOR_ACC, alpha=0.85, label='Compte Polymarket (total assets)', zorder=3)
for i, (bv, uv) in enumerate(zip(bt_vals, live_user)):
    if abs(bv) > 0.5: ax3.text(i-0.2, bv+5, f'{bv:+.0f}', ha='center', fontsize=8, color='white')
    ax3.text(i+0.2, uv+5, f'+{uv:.0f}', ha='center', fontsize=8, color='white')
ax3.axhline(0, color='#555', lw=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(DAYS_LBL, color='#ccc', fontsize=10)
ax3.set_title('BT BTC h1 vs Compte total (contexte)', color='white', fontsize=11)
ax3.legend(fontsize=8, framealpha=0.25, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
ax3.tick_params(colors='#aaa'); ax3.grid(axis='y', alpha=0.2)
for sp in ax3.spines.values(): sp.set_edgecolor('#333')

# ── Panel 4 : tableau ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, :])
ax4.set_facecolor('#0e1117'); ax4.axis('off')

col_labels = ['Jour', 'Tradable', 'Live trades', 'Live PnL', 'BT trades', 'BT PnL',
              'Diff (Live-BT)', 'Compte Poly']
rows = []
for di, d in enumerate(DAYS_STR):
    lv = by_day_live.get(d, 0.0)
    bt = by_day_bt.get(d, 0.0)
    diff = lv - bt
    rows.append([
        DAYS_LBL[di],
        str(n_tradable.get(d,0)),
        str(n_traded_live.get(d,0)),
        f'{lv:+.1f}',
        str(n_traded_bt.get(d,0)),
        f'{bt:+.1f}',
        f'{diff:+.1f}',
        f'+{live_user[di]:.1f}',
    ])

tbl = ax4.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.1, 2.1)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#0d47a1')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#1a1a2e' if row % 2 == 0 else '#12122a')
        cell.set_text_props(color='white')
    cell.set_edgecolor('#333')

ax4.set_title('Récap — BTC h1 Live vs BT (mêmes contrats) + Compte Polymarket',
              color='white', fontsize=10, pad=8)

fig.suptitle('BTC (m5+m15+h1) — Live vs BT apple-to-apple — Apr 13-16  (slope7, cap0.75)',
             color='white', fontsize=13, fontweight='bold', y=0.99)

out = OUT_DIR / "apr13_16_btc.png"
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0e1117')
plt.close('all')
print(f"\n-> {out}", flush=True)
