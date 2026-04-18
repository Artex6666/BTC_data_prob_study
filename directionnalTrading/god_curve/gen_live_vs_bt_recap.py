"""
gen_live_vs_bt_recap.py — Graphique recap Live vs BT (BTC)
depuis 2026-04-09T13:12:20Z  (slope7cap75)

Stats par asset :
  - Contrats tradables (vol gate OK)
  - Contrats tradés (avec fill)
  - Gagnants / Perdants / WR
  - Prix fill moyen
  - PnL total et PnL/jour

Sorties : live_vs_bt/live_vs_bt_recap.png
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
from chart_utils import (
    load_contracts, precompute_vol, simulate_trade_log,
    attach_settlement_outcomes,
    TIMEFRAMES, VOL_LBS,
)

BASE     = Path(__file__).resolve().parent.parent
LIVE_DIR = BASE / "reportLive" / "safeChase" / "slope7cap75" / "btc"
CSV_DIR  = BASE / "reportLive" / "safeChase"
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Config slope7cap75 actif depuis le 9 avr 13:12 UTC
CUTOFF    = pd.Timestamp("2026-04-09T13:12:20", tz="UTC")
CUTOFF_TS = CUTOFF.timestamp()

CFG_BTC = dict(curve='linear', slope=7.0, intercept=0.0, eq_cap=0.75,
               vol_lb_h=1, vol_thresh=60.0, vol_type='net',
               max_losses_cb=None, max_orders=2)
BTC_SIZE = 100.0


# ── Settlement live : charge une fois, keyed par (open_ts_int, tf_live) ──────
_TF_CSV  = {'m5': '5min', 'm15': '15min', 'h1': '1h'}
_TF_OFF  = {'m5': pd.Timedelta(minutes=5), 'm15': pd.Timedelta(minutes=15), 'h1': pd.Timedelta(hours=1)}

def _load_live_settlement(asset: str) -> dict:
    """Retourne {(open_ts_int, tf_live): won_up} depuis settlement.csv."""
    out = {}
    for tf_live, tf_csv in _TF_CSV.items():
        for open_ts, won_up in cu.load_settlement_outcomes(asset, tf=tf_csv).items():
            out[(open_ts, tf_live)] = won_up
    return out


# ── Parse live JSONL ──────────────────────────────────────────────────────────
def parse_live(asset_dir, settlement: dict):
    """
    Retourne (windows, trades, fills_all) depuis CUTOFF.

    Résolution du PnL via settlement.csv (Gamma/Polymarket officiel).
    Fallback sur end_spot vs start_spot si contrat absent du CSV (trop récent).

    Formule PnL :
    - UP gagne : pnl = up_shares - up_cost - down_cost
    - DOWN gagne : pnl = down_shares - down_cost - up_cost
    """
    windows = []
    fills_all = []
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
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except:
                    continue
                ev = d.get('event', '')

                if ev == 'window_open':
                    ts = pd.to_datetime(d['ts']).to_pydatetime().replace(tzinfo=timezone.utc)
                    if ts.timestamp() < CUTOFF_TS:
                        cur = None; fills_prices = []; tradable = False; break
                    cur = dict(ts=ts, tf=tf)
                    fills_prices = []; tradable = False

                elif ev == 'window_start' and cur:
                    tradable = True
                    cur['vol'] = float(d.get('vol_gate_net_usd', 0))

                elif ev in ('vol_gate_skip', 'eth_trend_gate_skip') and cur:
                    tradable = False

                elif ev == 'fill' and cur:
                    p = float(d.get('price', 0))
                    if p > 0:
                        fills_prices.append(p)
                        fills_all.append(p)

                elif ev in ('window_ended', 'window_settled') and cur:
                    up_s  = float(d.get('up_shares',  0))
                    dn_s  = float(d.get('down_shares', 0))
                    up_c  = float(d.get('up_cost',    0))
                    dn_c  = float(d.get('down_cost',  0))
                    shares = up_s + dn_s
                    cost   = up_c + dn_c
                    traded = shares > 1e-9

                    # open_ts du contrat = floor(window_open, tf)
                    ts_pd    = pd.Timestamp(cur['ts'])
                    open_ts  = int(ts_pd.floor(_TF_OFF[tf]).timestamp())
                    sett_key = (open_ts, tf)

                    if sett_key in settlement:
                        up_wins   = settlement[sett_key]
                        src       = 'settlement'
                    else:
                        # Fallback : prix Binance (contrat trop récent pour le CSV)
                        start_sp = float(d.get('start_spot', 0))
                        end_sp   = float(d.get('end_spot',   0))
                        up_wins  = (end_sp > start_sp) if (start_sp > 0 and end_sp > 0) \
                                   else bool(d.get('epnl_up_wins', 1))
                        src      = 'price'

                    if traded:
                        pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c)
                        won = pnl > 0
                    else:
                        pnl = 0.0; won = None

                    avg_fill = cost / shares if shares > 1e-9 else None
                    if fills_prices:
                        avg_fill = float(np.mean(fills_prices))

                    windows.append(dict(
                        ts=cur['ts'], tf=tf,
                        tradable=tradable,
                        skip=not tradable,
                        vol=float(cur.get('vol', 0)),
                        traded=traded,
                        won=won,
                        pnl=pnl,
                        avg_fill=avg_fill,
                        shares=shares,
                        cost=cost,
                        src=src,
                    ))
                    cur = None; fills_prices = []; tradable = False

    windows.sort(key=lambda x: x['ts'])
    trades = [w for w in windows if w['traded']]
    return windows, trades, fills_all


# ── Run BT ────────────────────────────────────────────────────────────────────
def run_bt(asset, cfg, size):
    csv = [str(CSV_DIR / f"{asset}.csv")]
    m5_ref = None
    all_trades = []
    pnl_by_hour = defaultdict(float)

    for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
        cts = load_contracts(csv, tf_floor, b1, b2, a1, a2)
        if tf_floor == '5min':
            precompute_vol(cts, VOL_LBS)
            m5_ref = cts
        else:
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)

        attach_settlement_outcomes(cts, asset.lower(), tf=tf_floor)

        tf_name = ['m5', 'm15', 'h1'][idx]
        # VRS n'a pas de vol_key (filtre de vol désactivé — la slope s'adapte)
        vol_key = (f"vol_{cfg['vol_lb_h']:g}h_{cfg.get('vol_type','range')}"
                   if cfg.get('vol_lb_h') else None)

        orig = cu.BASE_SIZE; cu.BASE_SIZE = float(size)
        for c in cts:
            if c.get('open_ts', 0) < CUTOFF_TS:
                continue
            if vol_key and c.get(vol_key, 0.0) < cfg.get('vol_thresh', 0):
                continue
            won, pnl, fills = simulate_trade_log(c, cfg)
            if won is None:
                continue
            avg_fill = float(np.mean([f['price'] for f in fills])) if fills else 0.0
            ts = c['ce']
            all_trades.append(dict(ts=ts, tf=tf_name, pnl=pnl, won=won, avg_fill=avg_fill))
            h = int(c.get('open_ts', 0)) // 3600
            pnl_by_hour[h] += pnl
        cu.BASE_SIZE = orig

    all_trades.sort(key=lambda x: x['ts'])
    return all_trades, pnl_by_hour


# ── Compute equity curve depuis trades ────────────────────────────────────────
def trades_to_equity(trades):
    if not trades:
        return [], []
    t0 = trades[0]['ts']
    if hasattr(t0, 'timestamp'):
        t0_ts = t0.timestamp()
        xs = [(t['ts'].timestamp() - t0_ts) / 3600 for t in trades]
    else:
        t0_ts = float(t0)
        xs = [(float(t['ts']) - t0_ts) / 3600 for t in trades]
    ys = list(np.cumsum([t['pnl'] for t in trades]))
    return xs, ys


def bt_to_equity_hours(trades, ref_start_ts, ref_end_ts):
    """BT equity alignée sur la meme echelle horaire que le live."""
    if not trades:
        return [], []
    t0 = ref_start_ts
    xs, ys = [], []
    cum = 0
    for t in trades:
        ts = t['ts'].timestamp() if hasattr(t['ts'], 'timestamp') else float(t['ts'])
        cum += t['pnl']
        xs.append((ts - t0) / 3600)
        ys.append(cum)
    return xs, ys


# ── Stats dict ────────────────────────────────────────────────────────────────
def compute_stats(windows, trades, label, days):
    tradable = sum(1 for w in windows if w['tradable'])
    n        = len(trades)
    wins     = [t for t in trades if t.get('won')]
    losses   = [t for t in trades if not t.get('won') and t.get('won') is not None]
    pnl      = sum(t['pnl'] for t in trades)
    fills    = [t['avg_fill'] for t in trades if t.get('avg_fill')]
    cum = np.cumsum([t['pnl'] for t in trades]) if trades else np.array([0.0])
    peak = np.maximum.accumulate(cum)
    maxdd = float((peak - cum).max()) if len(cum) else 0.0
    return dict(
        label=label, days=days,
        tradable=tradable, n=n,
        wins=len(wins), losses=len(losses),
        wr=100*len(wins)/n if n else 0,
        pnl=pnl, pnl_j=pnl/max(days, 0.01),
        avg_fill=float(np.mean(fills)) if fills else 0,
        maxdd=maxdd,
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Chargement settlement.csv ===", flush=True)
    btc_settlement = _load_live_settlement('btc')
    print(f"  {len(btc_settlement)} entrées (m5+m15+h1)", flush=True)

    print("=== Parse live BTC ===", flush=True)
    btc_windows, btc_live, _btc_fills = parse_live(LIVE_DIR, btc_settlement)
    tradables = sum(1 for w in btc_windows if w['tradable'])
    price_fallback = sum(1 for t in btc_live if t.get('src') == 'price')
    print(f"  BTC live: {len(btc_live)} trades / {tradables} tradables", flush=True)
    print(f"  Résolution : settlement={len(btc_live)-price_fallback}  fallback_price={price_fallback}", flush=True)

    print("=== Run BT BTC (slope7cap75, vol_net) ===", flush=True)
    btc_bt, _ = run_bt("BTC", CFG_BTC, BTC_SIZE)
    print(f"  BT BTC: {len(btc_bt)} trades depuis cutoff", flush=True)

    # Durée
    now_ts  = pd.Timestamp.now(tz='UTC')
    # BT limité à la même fenêtre temporelle (cutoff → dernier trade live)
    if btc_live:
        live_end_ts = max(t['ts'].timestamp() for t in btc_live)
    else:
        live_end_ts = now_ts.timestamp()
    live_span_h = (live_end_ts - CUTOFF_TS) / 3600
    live_span_d = live_span_h / 24

    s_bl = compute_stats(btc_windows, btc_live, 'BTC Live',          live_span_d)
    s_bb = compute_stats([], btc_bt,            'BT slope7cap75',    live_span_d)

    # Equity curves (abscisse = heures depuis CUTOFF)
    ref_start = CUTOFF_TS
    total_h   = live_span_h

    lx_btc, ly_btc = trades_to_equity(btc_live)
    bx_btc, by_btc = bt_to_equity_hours(btc_bt, ref_start, live_end_ts)

    # ── Graphique ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50,
                            height_ratios=[3, 1.5, 1.2])

    COLOR_LIVE = '#F44336'
    COLOR_BT   = '#90CAF9'

    ax_eq = fig.add_subplot(gs[0])
    if bx_btc:
        ax_eq.plot(bx_btc, by_btc, color=COLOR_BT, linewidth=1.8, linestyle='--',
                   label=f'BT slope7cap75  {by_btc[-1]:+.1f}$  WR={s_bb["wr"]:.1f}%  T={s_bb["n"]}', zorder=3)
    if lx_btc:
        ax_eq.plot(lx_btc, ly_btc, color=COLOR_LIVE, linewidth=2.2,
                   label=f'Live  {ly_btc[-1]:+.1f}$  WR={s_bl["wr"]:.1f}%  T={s_bl["n"]}', zorder=4)
        ax_eq.scatter(lx_btc, ly_btc, color=COLOR_LIVE, s=20, zorder=5, alpha=0.7)
    ax_eq.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax_eq.set_title('BTC — Live vs BT  ($100x2, slope7 cap0.75 vol_net_1h≥60)',
                    fontsize=10, fontweight='bold')
    ax_eq.set_xlabel(f'Heures depuis {CUTOFF.strftime("%Y-%m-%d %H:%M")} UTC', fontsize=9)
    ax_eq.set_ylabel('PnL cumulé ($)', fontsize=9)
    ax_eq.legend(fontsize=9); ax_eq.grid(alpha=0.25)
    xticks = np.arange(0, total_h + 2, 2)
    ax_eq.set_xticks(xticks)
    ax_eq.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
    ax_eq.set_xlim(-0.2, total_h + 0.5)

    # ── Drawdown ──────────────────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[1])
    if ly_btc:
        cum_l = np.array(ly_btc)
        dd_l  = np.maximum.accumulate(cum_l) - cum_l
        ax_dd.fill_between(lx_btc, 0, -dd_l, color=COLOR_LIVE, alpha=0.4, label='DD Live')
    if by_btc:
        cum_b = np.array(by_btc)
        dd_b  = np.maximum.accumulate(cum_b) - cum_b
        ax_dd.fill_between(bx_btc, 0, -dd_b, color=COLOR_BT, alpha=0.4, label='DD BT')
    ax_dd.axhline(0, color='gray', linewidth=0.5)
    ax_dd.set_ylabel('Drawdown ($)'); ax_dd.legend(fontsize=8); ax_dd.grid(alpha=0.25)
    xticks2 = np.arange(0, total_h + 2, 2)
    ax_dd.set_xticks(xticks2)
    ax_dd.set_xticklabels([f'{int(x)}h' for x in xticks2], fontsize=7, rotation=45)
    ax_dd.set_xlim(-0.2, total_h + 0.5)

    # ── Tableau stats ─────────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2])
    ax_tbl.axis('off')

    cols = ['', 'Tradables', 'Tradés', 'Wins', 'Losses',
            'WR', 'Fill avg', 'PnL total', 'PnL/h', 'MaxDD']
    rows_data = []
    for s in [s_bl, s_bb]:
        pnl_h = s['pnl'] / max(live_span_h, 0.01)
        rows_data.append([
            s['label'],
            str(s['tradable']) if s['tradable'] else '—',
            str(s['n']),
            str(s['wins']),
            str(s['losses']),
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
    tbl[1, 0].set_facecolor('#FFEBEE')
    tbl[2, 0].set_facecolor('#E3F2FD')
    for j in range(1, len(cols)):
        tbl[1, j].set_facecolor('#FFEBEE')
        tbl[2, j].set_facecolor('#E3F2FD')

    ax_tbl.set_title(
        f"Stats  {CUTOFF.strftime('%Y-%m-%d %H:%M')} UTC  ->  {live_span_h:.1f}h  "
        f"(resolution via settlement.csv)",
        fontsize=9, pad=6, fontweight='bold'
    )

    fig.suptitle(
        f'BTC Live vs Backtest — slope7 cap0.75 vol_net_1h≥60',
        fontsize=13, fontweight='bold', y=0.99
    )

    out = OUT_DIR / "live_vs_bt_recap.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"\n-> {out}", flush=True)

    # Console recap
    print(f"\n=== RECAP ({live_span_h:.1f}h depuis cutoff) ===")
    for s in [s_bl, s_bb]:
        pnl_h = s['pnl'] / max(live_span_h, 0.01)
        print(f"  {s['label']:<18} T={s['n']:>3}  tradable={s['tradable']:>4}  "
              f"W={s['wins']} L={s['losses']}  WR={s['wr']:.1f}%  "
              f"fill={s['avg_fill']:.3f}  PnL={s['pnl']:+.1f}$  "
              f"({pnl_h:+.2f}$/h)  maxDD=${s['maxdd']:.0f}")

    # Détail pertes live
    losses_live = [t for t in btc_live if not t.get('won') and t.get('won') is not None]
    if losses_live:
        print(f"\n  Pertes live ({len(losses_live)}) :")
        for t in losses_live:
            src = t.get('src', '?')
            print(f"    {t['ts'].strftime('%Y-%m-%d %H:%M')} UTC  tf={t['tf']}  "
                  f"pnl={t['pnl']:+.2f}$  cost={t['cost']:.2f}$  fill={t.get('avg_fill',0):.3f}  [{src}]")

