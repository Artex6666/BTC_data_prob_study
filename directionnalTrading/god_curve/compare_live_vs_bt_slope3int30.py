"""
Comparatif live vs backtest pour la config slope=3, intercept=30, cap=0.85
Live  : reportLive/safeChase/slope3_intercept30/btc/{m5,m15,h1}/
Backtest : CSV BTC, même config, BASE_SIZE=50$ (live utilise ~100$+)
"""
import sys, json, re
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from collections import deque

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import simulate, CAP_GRID

BASE_DIR = Path(__file__).resolve().parent.parent
LIVE_ROOT = BASE_DIR / "reportLive/safeChase/slope3_intercept30/btc"
CSV_PATHS = [
    BASE_DIR / "Datas/csv/BTC.csv",
    BASE_DIR / "reportLive/safeChase/60_lb0.5h/BTC.csv",
]

CFG = dict(
    label='linear', curve='linear',
    slope=3.0, intercept=30.0,
    A_exp=None, tau=None,
    eq_cap=0.85, max_losses_cb=None,
    vol_lb_h=None, vol_thresh=None,
)
BASE_SIZE_BT = 50.0


# ── Parsing JSONL live ────────────────────────────────────────────────────────
def parse_live_tf(tf_dir):
    """
    Retourne liste de dicts {ts, tf, won, pnl, cost, invested, side}
    pour chaque contrat avec au moins 1 fill.
    """
    results = []
    for f in sorted(tf_dir.glob("*.jsonl")):
        lines = [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
        ws = next((x for x in lines if x.get('event') == 'window_start'), None)
        we = next((x for x in lines if x.get('event') in ('window_ended', 'window_settled')), None)
        if ws is None or we is None:
            continue

        up_cost   = float(we.get('up_cost',   0) or 0)
        down_cost = float(we.get('down_cost', 0) or 0)
        up_sh     = float(we.get('up_shares',   0) or 0)
        down_sh   = float(we.get('down_shares', 0) or 0)
        invested  = up_cost + down_cost
        if invested < 1e-6:
            continue  # pas de fill

        final_up_bid   = float(we.get('final_up_bid',   0) or 0)
        final_down_bid = float(we.get('final_down_bid', 0) or 0)

        # Résolution : UP gagne si final_up_bid > 0.5
        if final_up_bid > final_down_bid:
            won = (up_sh > 0)
        else:
            won = (down_sh > 0)

        if won:
            pnl = float(we.get('expected_pnl', 0) or 0)
        else:
            pnl = -invested

        ts_str = ws.get('ts', '')
        try:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except Exception:
            ts = None

        side = 'UP' if up_sh > down_sh else 'DOWN'
        tf   = ws.get('timeframe', f.parent.name)
        results.append(dict(ts=ts, tf=tf, won=won, pnl=pnl,
                            cost=invested, side=side))
    return results


def stats(trades):
    if not trades:
        return dict(n=0, wr=0, total=0, avg_w=0, avg_l=0,
                    losing_days=0, worst_day=0, max_dd=0, rf=0)
    wins   = [t for t in trades if t['won']]
    losses = [t for t in trades if not t['won']]
    total  = sum(t['pnl'] for t in trades)
    avg_w  = sum(t['pnl'] for t in wins)   / len(wins)   if wins   else 0
    avg_l  = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
    wr     = len(wins) / len(trades) * 100

    # maxDD peak-to-trough
    pnls   = [t['pnl'] for t in trades]
    cum    = np.cumsum([0.0] + pnls)
    peak   = 0.0; max_dd = 0.0
    for v in cum:
        if v > peak: peak = v
        dd = peak - v
        if dd > max_dd: max_dd = dd

    # per-day PnL (UTC date)
    day_pnl = {}
    for t in trades:
        if t.get('ts'):
            d = t['ts'].date()
        else:
            d = None
        day_pnl[d] = day_pnl.get(d, 0.0) + t['pnl']
    losing_days = sum(1 for v in day_pnl.values() if v < 0)
    worst_day   = min(day_pnl.values()) if day_pnl else 0

    rf = total / max_dd if max_dd > 0 else float('inf') if total > 0 else 0
    return dict(n=len(trades), wr=wr, total=total,
                avg_w=avg_w, avg_l=avg_l,
                losing_days=losing_days, worst_day=worst_day,
                max_dd=max_dd, rf=rf)


# ── Backtest sur la même période ──────────────────────────────────────────────
def run_backtest(live_trades, cfg, base_size=50.0):
    # Déterminer la période live (UTC timestamps)
    ts_list = [t['ts'] for t in live_trades if t.get('ts')]
    if not ts_list:
        return []
    t_min = min(ts_list).replace(tzinfo=None)
    t_max = max(ts_list).replace(tzinfo=None)
    t_min_floor = t_min.replace(minute=0, second=0, microsecond=0)

    # Charger CSV
    dfs = []
    for p in CSV_PATHS:
        try:
            d = pd.read_csv(p, usecols=['timestamp','spot_price',
                'm5_up_ask','m5_up_bid','m5_down_ask','m5_down_bid'])
            d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
            dfs.append(d)
        except Exception as e:
            print(f"  CSV skip {p}: {e}")
    if not dfs:
        return []
    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    # Filtrer sur la période live
    df = df[(df['timestamp'] >= t_min_floor) & (df['timestamp'] <= t_max + pd.Timedelta(minutes=5))]
    df['contract'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)

    contracts = []
    for ce, grp in df.groupby('contract'):
        op_spot = grp['spot_price'].iloc[0]
        t60 = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=60)]
        if len(t60) < 2: continue
        contracts.append({
            'ce':       ce,
            'ce_ts':    ce.timestamp(),
            'op':       float(op_spot),
            'spot':     t60['spot_price'].values.astype(np.float64),
            'ts':       t60['timestamp'].values,
            'up_bid':   t60['m5_up_bid'].values.astype(np.float64),
            'down_bid': t60['m5_down_bid'].values.astype(np.float64),
            'up_ask':   t60['m5_up_ask'].values.astype(np.float64),
            'down_ask': t60['m5_down_ask'].values.astype(np.float64),
        })

    # Precompute vol (requis pour les éventuels filtres, ici aucun)
    from chart_utils import precompute_vol, VOL_LBS
    precompute_vol(contracts, VOL_LBS)

    bt_cfg = {**cfg, 'eq_cap': cfg['eq_cap']}
    results = []
    for c in contracts:
        won, pnl = simulate(c, bt_cfg)
        if won is None: continue
        # Normaliser à base_size ($50 = 1 ordre)
        # simulate interne utilise BASE_SIZE=50$
        results.append(dict(
            ts=c['ce'].to_pydatetime().replace(tzinfo=timezone.utc),
            tf='m5', won=won, pnl=pnl, cost=base_size, side=None
        ))
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass

    # Parse live
    live_all = []
    for tf in ['m5', 'm15', 'h1']:
        tf_dir = LIVE_ROOT / tf
        if tf_dir.exists():
            trades = parse_live_tf(tf_dir)
            live_all.extend(trades)
            print(f"  Live {tf}: {len(trades)} trades fillés", flush=True)

    live_all.sort(key=lambda x: x['ts'] or datetime.min.replace(tzinfo=timezone.utc))

    # Période live
    ts_list = [t['ts'] for t in live_all if t['ts']]
    period_start = min(ts_list) if ts_list else None
    period_end   = max(ts_list) if ts_list else None
    n_hours = (period_end - period_start).total_seconds() / 3600 if period_start and period_end else 1
    n_days  = n_hours / 24

    # Backtest (m5 seulement, sur la même période)
    print("\n  Backtest m5 en cours...", flush=True)
    bt_trades = run_backtest(live_all, CFG, BASE_SIZE_BT)
    print(f"  Backtest: {len(bt_trades)} trades")

    # Stats
    s_live = stats(live_all)
    s_bt   = stats(bt_trades)

    # Backtest normalisé à live size (avg live cost per trade)
    avg_live_cost = (sum(t['cost'] for t in live_all) / len(live_all)) if live_all else 1
    scale = avg_live_cost / BASE_SIZE_BT if BASE_SIZE_BT > 0 else 1

    sep = '-' * 70
    print(f"\n{'='*70}")
    print(f"  CONFIG : slope=3  intercept=30$  cap=0.85")
    print(f"  PERIODE: {period_start.strftime('%d/%m %H:%M') if period_start else '?'} -> "
          f"{period_end.strftime('%d/%m %H:%M') if period_end else '?'}  ({n_hours:.1f}h / {n_days:.1f}j)")
    print(f"{'='*70}")
    print(f"\n{'':30s} {'LIVE':>20s}  {'BACKTEST m5':>20s}")
    print(sep)

    def fmt(label, live_val, bt_val, fmt_str='{:>20.1f}'):
        lv = fmt_str.format(live_val)
        bv = fmt_str.format(bt_val)
        print(f"  {label:<28s} {lv}  {bv}")

    print(f"  {'Trade size (avg $)':28s} {'~'+str(round(avg_live_cost)):>20s}  {'$'+str(int(BASE_SIZE_BT)):>20s}")
    fmt("Nb trades",          s_live['n'],         s_bt['n'],         '{:>20.0f}')
    fmt("Win rate (%)",        s_live['wr'],         s_bt['wr'],         '{:>20.1f}')
    fmt("Total PnL ($)",       s_live['total'],      s_bt['total'],      '{:>20.2f}')
    fmt("PnL/j ($)",           s_live['total']/n_days if n_days else 0,
                               s_bt['total']/n_days if n_days else 0,   '{:>20.2f}')
    fmt("Avg win ($)",         s_live['avg_w'],      s_bt['avg_w'],      '{:>20.2f}')
    fmt("Avg loss ($)",        s_live['avg_l'],      s_bt['avg_l'],      '{:>20.2f}')
    fmt("Losing days",         s_live['losing_days'],s_bt['losing_days'],'{:>20.0f}')
    fmt("Worst day ($)",       s_live['worst_day'],  s_bt['worst_day'],  '{:>20.2f}')
    fmt("Max DD ($)",          s_live['max_dd'],     s_bt['max_dd'],     '{:>20.2f}')
    rf_live = s_live['rf']
    rf_bt   = s_bt['rf']
    rf_live_s = f"{'inf':>20s}" if rf_live == float('inf') else f"{rf_live:>20.1f}"
    rf_bt_s   = f"{'inf':>20s}" if rf_bt   == float('inf') else f"{rf_bt:>20.1f}"
    print(f"  {'RF (PnL/maxDD)':28s} {rf_live_s}  {rf_bt_s}")

    # Breakdown par TF (live)
    print(f"\n  --- Breakdown live par TF ---")
    for tf in ['m5', 'm15', 'h1']:
        tf_trades = [t for t in live_all if t['tf'] == tf]
        if not tf_trades: continue
        s = stats(tf_trades)
        print(f"  {tf:4s}: {s['n']:3d} trades  WR={s['wr']:.1f}%  "
              f"PnL={s['total']:+.2f}$  maxDD={s['max_dd']:.2f}$  RF={s['rf']:.1f}x")

    # Breakdown live par jour
    print(f"\n  --- PnL live par jour ---")
    day_pnl = {}
    for t in live_all:
        if t.get('ts'):
            d = t['ts'].date()
            day_pnl[d] = day_pnl.get(d, 0.0) + t['pnl']
    for d, pnl in sorted(day_pnl.items()):
        mark = ' <<' if pnl < 0 else ''
        print(f"  {d}  {pnl:+8.2f}${mark}")

    # Breakdown backtest par jour
    print(f"\n  --- PnL backtest par jour ---")
    day_pnl_bt = {}
    for t in bt_trades:
        if t.get('ts'):
            d = t['ts'].date()
            day_pnl_bt[d] = day_pnl_bt.get(d, 0.0) + t['pnl']
    for d, pnl in sorted(day_pnl_bt.items()):
        mark = ' <<' if pnl < 0 else ''
        print(f"  {d}  {pnl:+8.2f}${mark}")

    print(f"\n  Note: backtest = m5 spot delta, live = synthetic delta + fills partiels multiples")
    print(f"  Scale live/bt par taille: x{scale:.1f}  (avg_live={avg_live_cost:.0f}$ vs bt={BASE_SIZE_BT:.0f}$)")


if __name__ == '__main__':
    main()
