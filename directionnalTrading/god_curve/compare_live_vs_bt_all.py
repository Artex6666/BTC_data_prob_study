"""
Comparatif Live vs BT — BTC, ETH, SOL, XRP — Apr 10 au 16.
Live  : JSONL slope7cap75/{asset}/{tf}/
BT    : Datas/csv/{ASSET}.csv + reportLive/safeChase/{ASSET}.csv + settlement.csv
Fill% : live (traded/tradable) vs BT (triggered/tradable_bt)
Snipe : PnL isolé via cumuls fill avant/après snipe_fired
"""
import json, sys
from pathlib import Path
from collections import defaultdict
from datetime import timezone

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
from chart_utils import load_contracts, precompute_vol, simulate_trade_log, \
                        attach_settlement_outcomes, TIMEFRAMES, VOL_LBS

BASE      = Path(__file__).resolve().parent.parent
LIVE_BASE = BASE / "reportLive" / "safeChase" / "slope7cap75"
CSV_LIVE  = BASE / "reportLive" / "safeChase"   # CSV live (Apr 12+)
CSV_HIST  = BASE / "Datas" / "csv"              # CSV historique complet
OUT_DIR        = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR_EQUITY = Path(__file__).resolve().parent / "live_vs_bt" / "equity"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR_EQUITY.mkdir(parents=True, exist_ok=True)

START_UTC = pd.Timestamp("2026-04-13T00:00:00", tz="UTC")
END_UTC   = pd.Timestamp("2026-04-17T00:00:00", tz="UTC")
START_TS  = START_UTC.timestamp()
END_TS    = END_UTC.timestamp()

DAYS_STR = [f"2026-04-{d:02d}" for d in range(13, 17)]
DAYS_LBL = [f"{d} avr" for d in range(13, 17)]
TF_INFO  = {'m5': ('5min', 300), 'm15': ('15min', 900), 'h1': ('1h', 3600)}

ASSET_CFGS = {
    'btc': dict(
        cfg=dict(curve='linear', slope=7.0, intercept=0.0, eq_cap=0.75,
                 vol_lb_h=1, vol_thresh=60.0, vol_type='net',
                 max_losses_cb=None, max_orders=2),
        size=120.0, colors=('#F7931A', '#FFCA6B'),
    ),
    'eth': dict(
        cfg=dict(curve='linear', slope=0.2, intercept=0.0, eq_cap=0.85,
                 vol_lb_h=2.0, vol_thresh=0.3, vol_type='trend',
                 max_losses_cb=None, max_orders=2),
        size=130.0, colors=('#627EEA', '#A8B8F8'),
    ),
    'sol': dict(
        cfg=dict(curve='linear', slope=0.005, intercept=0.0, eq_cap=0.75,
                 vol_lb_h=1.0, vol_thresh=0.50, vol_type='pct_range',
                 max_losses_cb=None, max_orders=3),
        size=50.0, colors=('#9945FF', '#C8A0FF'),
    ),
    'xrp': dict(
        cfg=dict(curve='linear', slope=0.0001, intercept=0.0, eq_cap=0.65,
                 vol_lb_h=2.0, vol_thresh=0.50, vol_type='trend',
                 max_losses_cb=None, max_orders=3),
        size=40.0, colors=('#00AAE4', '#80D5F5'),
    ),
}


# ── Parse JSONL live — sépare chase vs snipe ──────────────────────────────────
def parse_live(asset: str):
    """
    Retourne deux dicts { (day, tf): {...} } : stats et snipe_stats.
    Séparation snipe via cumuls fill avant/après snipe_fired.
    """
    settlement_all = {}
    for tf_csv, tf_off in [('5min', 300), ('15min', 900), ('1h', 3600)]:
        for ots, won in cu.load_settlement_outcomes(asset, tf=tf_csv).items():
            settlement_all[(ots, tf_off)] = won

    stats = defaultdict(lambda: dict(
        tradable=0, traded=0, pnl=0.0, wins=0, losses=0,
        pnl_chase=0.0, pnl_snipe=0.0, trades_snipe=0,
    ))
    trades_list = []   # [(close_ts, pnl)]

    for tf_name, (tf_csv, tf_off) in TF_INFO.items():
        tf_dir = LIVE_BASE / asset / tf_name
        if not tf_dir.exists():
            continue

        for f in sorted(tf_dir.glob('*.jsonl')):
            cur_ots = None; tradable = False
            snipe_fired_side = None
            # cumuls fills avant snipe
            pre_snipe = {'UP': {'shares': 0.0, 'cost': 0.0},
                         'DOWN': {'shares': 0.0, 'cost': 0.0}}

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
                    cur_ots = None; tradable = False
                    snipe_fired_side = None
                    pre_snipe = {'UP': {'shares': 0.0, 'cost': 0.0},
                                 'DOWN': {'shares': 0.0, 'cost': 0.0}}
                    ts = pd.to_datetime(d['ts']).replace(tzinfo=timezone.utc)
                    if ts.timestamp() < START_TS or ts.timestamp() >= END_TS:
                        continue
                    cur_ots = int(pd.Timestamp(ts).floor(
                        pd.Timedelta(seconds=tf_off)).timestamp())

                elif ev == 'window_start' and cur_ots is not None:
                    tradable = True

                elif ev in ('vol_gate_skip', 'eth_trend_gate_skip') and cur_ots is not None:
                    tradable = False

                elif ev == 'snipe_fired' and cur_ots is not None:
                    snipe_fired_side = d.get('side', '')
                    # snapshot des cumuls actuels = chase total avant snipe
                    # les fill events suivants à 0.99 seront snipe fills

                elif ev == 'fill' and cur_ots is not None and snipe_fired_side is None:
                    # fill AVANT snipe_fired : on mémorise le cumul (les fill events
                    # contiennent le cumul running up_shares/down_shares)
                    side = d.get('side', '')
                    if side in pre_snipe:
                        pre_snipe[side]['shares'] = float(d.get('up_shares' if side == 'UP'
                                                                 else 'down_shares', 0))
                        pre_snipe[side]['cost'] = float(d.get('up_cost' if side == 'UP'
                                                               else 'down_cost', 0))

                elif ev in ('window_ended', 'window_settled') and cur_ots is not None:
                    up_s  = float(d.get('up_shares',  0))
                    dn_s  = float(d.get('down_shares', 0))
                    up_c  = float(d.get('up_cost',    0))
                    dn_c  = float(d.get('down_cost',  0))
                    total_shares = up_s + dn_s
                    traded = total_shares > 1e-9

                    skey = (cur_ots, tf_off)
                    if skey in settlement_all:
                        up_wins = settlement_all[skey]
                    else:
                        up_w = int(d.get('epnl_up_wins',  -1))
                        dn_w = int(d.get('epnl_down_wins', -1))
                        if up_w >= 0 and dn_w >= 0 and up_w + dn_w > 0:
                            up_wins = (up_w == 1)
                        else:
                            s0 = float(d.get('start_spot', 0))
                            s1 = float(d.get('end_spot',   0))
                            up_wins = (s1 >= s0) if (s0 > 0 and s1 > 0) else True

                    pnl_total = 0.0; won = None
                    if traded:
                        pnl_total = (up_s - up_c - dn_c) if up_wins \
                                    else (dn_s - dn_c - up_c)
                        won = pnl_total > 0

                    # Séparation chase / snipe
                    pnl_chase = 0.0; pnl_snipe = 0.0; snipe_traded = False
                    if traded and snipe_fired_side:
                        side = snipe_fired_side
                        pre_sh = pre_snipe[side]['shares']
                        pre_co = pre_snipe[side]['cost']
                        tot_sh = up_s if side == 'UP' else dn_s
                        tot_co = up_c if side == 'UP' else dn_c
                        snipe_sh = tot_sh - pre_sh
                        snipe_co = tot_co - pre_co
                        opp_co = dn_c if side == 'UP' else up_c  # cost côté opposé

                        if snipe_sh > 1e-6:
                            snipe_traded = True
                            # snipe pnl = snipe_sh si win, -snipe_co si lose - opp_co proportionnel
                            # on attribue l'opp_co entièrement au chase (snipe ne fait qu'un seul côté)
                            pnl_snipe = snipe_sh - snipe_co if up_wins == (side == 'UP') \
                                        else -snipe_co
                            pnl_chase = pnl_total - pnl_snipe
                        else:
                            pnl_chase = pnl_total

                    elif traded:
                        pnl_chase = pnl_total

                    day = pd.Timestamp(cur_ots + tf_off, unit='s',
                                       tz='UTC').strftime('%Y-%m-%d')
                    k = (day, tf_name)
                    if tradable:
                        stats[k]['tradable'] += 1
                    if traded:
                        stats[k]['traded']      += 1
                        stats[k]['pnl']         += pnl_total
                        stats[k]['pnl_chase']   += pnl_chase
                        if snipe_traded:
                            stats[k]['pnl_snipe']   += pnl_snipe
                            stats[k]['trades_snipe'] += 1
                        if won:  stats[k]['wins']   += 1
                        else:    stats[k]['losses'] += 1
                        trades_list.append((cur_ots + tf_off, pnl_total))
                    cur_ots = None

    trades_list.sort(key=lambda x: x[0])
    return stats, trades_list


# ── Fallback settlements depuis JSONL window_settled ─────────────────────────
_SLUG_TF_MAP = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}

def load_jsonl_settlements(asset: str) -> dict:
    """
    Retourne {(open_ts, tf_floor): bool (True=UP won)} depuis epnl_up_wins des window_settled.
    Keyed par (open_ts, tf_floor) pour éviter collision entre TFs au même timestamp.
    """
    result = {}
    asset_dir = LIVE_BASE / asset
    if not asset_dir.exists():
        return result
    for tf_dir in asset_dir.iterdir():
        if not tf_dir.is_dir():
            continue
        for f in tf_dir.glob('*.jsonl'):
            for line in f.open(encoding='utf-8', errors='replace'):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get('event') != 'window_settled':
                    continue
                slug = d.get('slug', '')
                # format: btc-updown-15m-1776345300
                parts = slug.rsplit('-', 1)
                if len(parts) != 2:
                    continue
                try:
                    ots = int(parts[1])
                except ValueError:
                    continue
                # extraire le TF depuis le slug  (e.g. "btc-updown-15m")
                slug_tf_raw = parts[0].rsplit('-', 1)[-1]   # "15m", "5m", "1h"...
                tf_floor = _SLUG_TF_MAP.get(slug_tf_raw, slug_tf_raw)
                up_w = int(d.get('epnl_up_wins',  0))
                dn_w = int(d.get('epnl_down_wins', 0))
                if up_w + dn_w > 0:
                    result[(ots, tf_floor)] = (up_w == 1)
    return result


# ── BT sur CSV complet ────────────────────────────────────────────────────────
def run_bt(asset: str):
    """
    Retourne { (day, tf): dict(tradable, trades, pnl, wins) }
    CSV : Datas/csv + safeChase (merge pour couvrir toute la période)
    """
    hist_csv = str(CSV_HIST / f"{asset.upper()}.csv")
    live_csv = str(CSV_LIVE  / f"{asset.upper()}.csv")
    # On prend les deux, load_contracts déduplique par timestamp
    csv_paths = []
    for p in [hist_csv, live_csv]:
        if Path(p).exists():
            csv_paths.append(p)

    cfg  = ASSET_CFGS[asset]['cfg']
    size = ASSET_CFGS[asset]['size']
    result = defaultdict(lambda: dict(tradable=0, trades=0, pnl=0.0, wins=0))

    m5_ref = None
    orig = cu.BASE_SIZE; cu.BASE_SIZE = float(size)
    jsonl_sett = load_jsonl_settlements(asset)
    bt_trades = []   # [(close_ts, pnl)]

    for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
        try:
            cts = load_contracts(csv_paths, tf_floor, b1, b2, a1, a2)
        except Exception as e:
            print(f"  {asset} {tf_floor}: {e}", flush=True); continue

        if tf_floor == '5min':
            precompute_vol(cts, VOL_LBS); m5_ref = cts
        else:
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
        attach_settlement_outcomes(cts, asset.lower(), tf=tf_floor)

        # Fallback JSONL pour les contrats sans settlement.csv
        for c in cts:
            if '_settlement_won_up' not in c:
                ots = c.get('open_ts', 0)
                key = (ots, tf_floor)
                if key in jsonl_sett:
                    c['_settlement_won_up']   = jsonl_sett[key]
                    c['_settlement_won_down']  = not jsonl_sett[key]

        tf_name = ['m5', 'm15', 'h1'][idx]
        tf_off  = [300, 900, 3600][idx]
        vol_key = (f"vol_{cfg['vol_lb_h']:g}h_{cfg.get('vol_type','range')}"
                   if cfg.get('vol_lb_h') else None)

        for c in cts:
            ce_ts = c.get('open_ts', 0) + tf_off
            if ce_ts < START_TS or ce_ts >= END_TS:
                continue
            day = pd.Timestamp(ce_ts, unit='s', tz='UTC').strftime('%Y-%m-%d')
            k   = (day, tf_name)

            # Tradable BT = passe le vol gate
            if vol_key and c.get(vol_key, 0.0) < cfg.get('vol_thresh', 0):
                continue
            result[k]['tradable'] += 1

            if '_settlement_won_up' not in c:
                continue
            won, pnl, fills = simulate_trade_log(c, cfg)
            if won is None:
                continue
            result[k]['trades'] += 1
            result[k]['pnl']    += pnl
            if won: result[k]['wins'] += 1
            bt_trades.append((int(ce_ts), pnl))

    cu.BASE_SIZE = orig
    bt_trades.sort(key=lambda x: x[0])
    return result, bt_trades


# ── Run ───────────────────────────────────────────────────────────────────────
print("=== Parsing live + BT ===", flush=True)
all_live = {}; all_bt = {}
all_live_trades = {}; all_bt_trades = {}
for asset in ['btc', 'eth', 'sol', 'xrp']:
    print(f"  {asset.upper()}...", flush=True)
    all_live[asset], all_live_trades[asset] = parse_live(asset)
    all_bt[asset],   all_bt_trades[asset]   = run_bt(asset)

# ── Console ───────────────────────────────────────────────────────────────────
print("\n=== RECAP PAR ASSET ===", flush=True)
for asset in ['btc', 'eth', 'sol', 'xrp']:
    print(f"\n{'='*80}\n  {asset.upper()}\n{'='*80}")
    print(f"  {'Jour':<10} {'TF':<5} {'L.Trad':>7} {'L.Fill':>7} "
          f"{'L.Fill%':>8} {'BT.Trad':>8} {'BT.Fill':>8} {'BT.Fill%':>9} "
          f"{'L.PnL':>8} {'BT.PnL':>8} {'L.Snipe':>8}")
    for d, dlbl in zip(DAYS_STR, DAYS_LBL):
        for tf in ['m5', 'm15', 'h1']:
            k  = (d, tf)
            lv = all_live[asset].get(k, {})
            bt = all_bt[asset].get(k, {})
            lt = lv.get('tradable', 0); ld = lv.get('traded', 0)
            btr= bt.get('tradable', 0); bfill = bt.get('trades', 0)
            lf = 100*ld/lt   if lt   else 0.0
            bf = 100*bfill/btr if btr else 0.0
            if lt or btr:
                print(f"  {dlbl:<10} {tf:<5} {lt:>7} {ld:>7} {lf:>7.1f}% "
                      f"{btr:>8} {bfill:>8} {bf:>8.1f}% "
                      f"{lv.get('pnl',0):>+8.1f} {bt.get('pnl',0):>+8.1f} "
                      f"{lv.get('pnl_snipe',0):>+8.1f}")

# Snipe global
print("\n=== SNIPE GLOBAL (toute période, tous assets) ===")
total_chase = total_snipe = 0.0
total_t_chase = total_t_snipe = 0
for asset in ['btc','eth','sol','xrp']:
    a_chase = a_snipe = 0.0; t_c = t_s = 0
    for v in all_live[asset].values():
        a_chase += v.get('pnl_chase', 0)
        a_snipe += v.get('pnl_snipe', 0)
        t_c += v.get('traded', 0) - v.get('trades_snipe', 0)
        t_s += v.get('trades_snipe', 0)
    total_chase += a_chase; total_snipe += a_snipe
    total_t_chase += t_c; total_t_snipe += t_s
    print(f"  {asset.upper():<5}  Chase: {t_c:>4} trades  {a_chase:>+8.1f} USD   "
          f"Snipe: {t_s:>4} trades  {a_snipe:>+8.1f} USD")
print(f"  {'TOTAL':<5}  Chase: {total_t_chase:>4} trades  {total_chase:>+8.1f} USD   "
      f"Snipe: {total_t_snipe:>4} trades  {total_snipe:>+8.1f} USD")


# ── Figures ───────────────────────────────────────────────────────────────────
def make_asset_fig(asset):
    col_live, col_bt = ASSET_CFGS[asset]['colors']
    fig = plt.figure(figsize=(20, 15))
    fig.patch.set_facecolor('#0e1117')
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.35,
                            height_ratios=[2, 2, 1.5])
    tfs = ['m5', 'm15', 'h1']
    x   = np.arange(len(DAYS_STR)); w = 0.3

    for ti, tf in enumerate(tfs):
        # Row 0 : PnL/jour
        ax0 = fig.add_subplot(gs[0, ti])
        ax0.set_facecolor('#0e1117')
        lv_pnl = [all_live[asset].get((d, tf), {}).get('pnl', 0.) for d in DAYS_STR]
        bt_pnl = [all_bt[asset].get((d, tf), {}).get('pnl', 0.)   for d in DAYS_STR]
        ax0.bar(x - w/2, lv_pnl, w, color=col_live, alpha=0.85, label='Live', zorder=3)
        ax0.bar(x + w/2, bt_pnl, w, color=col_bt,   alpha=0.85, label='BT',   zorder=3)
        ax0.axhline(0, color='#555', lw=0.7)
        ax0.set_xticks(x); ax0.set_xticklabels(DAYS_LBL, rotation=40, ha='right', fontsize=7, color='#ccc')
        ax0.set_title(f'{tf.upper()} — PnL/jour', color='white', fontsize=10)
        ax0.set_ylabel('USD', color='#aaa', fontsize=8)
        ax0.legend(fontsize=7, framealpha=0.2, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
        ax0.tick_params(colors='#aaa'); ax0.grid(axis='y', alpha=0.2)
        for sp in ax0.spines.values(): sp.set_edgecolor('#333')

        # Row 1 : Fill rate live vs BT
        ax1 = fig.add_subplot(gs[1, ti])
        ax1.set_facecolor('#0e1117')
        lv_fr, bt_fr = [], []
        for d in DAYS_STR:
            lv = all_live[asset].get((d, tf), {})
            bt = all_bt[asset].get((d, tf), {})
            lt = lv.get('tradable', 0); ld = lv.get('traded', 0)
            btr = bt.get('tradable', 0); bfill = bt.get('trades', 0)
            lv_fr.append(100*ld/lt   if lt   else np.nan)
            bt_fr.append(100*bfill/btr if btr else np.nan)
        ax1.plot(x, lv_fr, 'o-', color=col_live, lw=1.8, ms=5, label='Live fill%', zorder=4)
        ax1.plot(x, bt_fr, 's--', color=col_bt,  lw=1.5, ms=4, label='BT fill%',  zorder=3)
        ax1.set_xticks(x); ax1.set_xticklabels(DAYS_LBL, rotation=40, ha='right', fontsize=7, color='#ccc')
        ax1.set_title(f'{tf.upper()} — Fill rate %', color='white', fontsize=10)
        ax1.set_ylabel('%', color='#aaa', fontsize=8); ax1.set_ylim(bottom=0)
        ax1.legend(fontsize=7, framealpha=0.2, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
        ax1.tick_params(colors='#aaa'); ax1.grid(alpha=0.2)
        for sp in ax1.spines.values(): sp.set_edgecolor('#333')

    # Row 2 : Tableau récap
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.set_facecolor('#0e1117'); ax_t.axis('off')
    cols = ['Jour', 'L.Tradable', 'L.Traded', 'L.Fill%',
            'BT.Tradable', 'BT.Traded', 'BT.Fill%',
            'L.PnL', 'BT.PnL', 'L.Snipe PnL']
    rows = []
    for d, dlbl in zip(DAYS_STR, DAYS_LBL):
        lt = ld = btr = bfill = 0
        lpnl = btpnl = lsnipe = 0.0
        for tf in tfs:
            lv = all_live[asset].get((d, tf), {})
            bt = all_bt[asset].get((d, tf), {})
            lt    += lv.get('tradable', 0); ld    += lv.get('traded',  0)
            btr   += bt.get('tradable', 0); bfill += bt.get('trades',  0)
            lpnl  += lv.get('pnl', 0.); btpnl += bt.get('pnl', 0.)
            lsnipe+= lv.get('pnl_snipe', 0.)
        rows.append([dlbl, str(lt), str(ld),
                     f'{100*ld/lt:.1f}%' if lt else '—',
                     str(btr), str(bfill),
                     f'{100*bfill/btr:.1f}%' if btr else '—',
                     f'{lpnl:+.1f}', f'{btpnl:+.1f}', f'{lsnipe:+.1f}'])
    tbl = ax_t.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.05, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#333')
        cell.set_facecolor('#0d47a1' if r == 0 else ('#1a1a2e' if r%2==0 else '#12122a'))
        cell.set_text_props(color='white', fontweight='bold' if r==0 else 'normal')
    ax_t.set_title(f'{asset.upper()} — récap toutes TFs (Apr 10-16)', color='white', fontsize=10, pad=6)

    slope = ASSET_CFGS[asset]['cfg']['slope']
    cap   = ASSET_CFGS[asset]['cfg']['eq_cap']
    fig.suptitle(f'{asset.upper()} — Live vs BT — slope={slope} cap={cap}',
                 color='white', fontsize=13, fontweight='bold', y=0.99)
    out = OUT_DIR / f"live_vs_bt_{asset}_apr10_16.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close(fig)
    print(f"  -> {out}", flush=True)


def _trades_to_equity(trades_list):
    """Convertit [(ts, pnl)] en (ts_arr, equity_arr) pour une courbe continue."""
    if not trades_list:
        return np.array([START_TS, END_TS]), np.array([0.0, 0.0])
    ts_arr = np.array([t for t, _ in trades_list], dtype=float)
    pnl_arr = np.array([p for _, p in trades_list], dtype=float)
    equity = np.concatenate([[0.0], np.cumsum(pnl_arr)])
    ts_plot = np.concatenate([[START_TS], ts_arr])
    return ts_plot, equity


def _ts_to_mpl(ts_arr):
    """Convertit timestamps Unix en floats matplotlib (dates)."""
    import matplotlib.dates as mdates
    import datetime
    dts = [datetime.datetime.utcfromtimestamp(float(t)) for t in ts_arr]
    return mdates.date2num(dts)


def _day_ticks():
    """Retourne (positions_mpl, labels) pour un tick par jour."""
    import matplotlib.dates as mdates
    import datetime
    positions = [mdates.date2num(datetime.datetime.utcfromtimestamp(
                     pd.Timestamp(d, tz='UTC').timestamp())) for d in DAYS_STR]
    return positions, DAYS_LBL


def _style_ax(ax):
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='#aaa')
    ax.grid(alpha=0.18)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')


def make_equity_fig(asset):
    """Courbe d'équité intraday (trade par trade, tous TFs confondus) + stats table."""
    import matplotlib.dates as mdates
    col_live, col_bt = ASSET_CFGS[asset]['colors']
    tick_pos, tick_lbl = _day_ticks()

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor('#0e1117')
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, height_ratios=[2.8, 1.4])

    # ── Courbe d'équité ────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    _style_ax(ax0)
    lv_ts, lv_eq = _trades_to_equity(all_live_trades[asset])
    bt_ts, bt_eq = _trades_to_equity(all_bt_trades[asset])
    lv_x = _ts_to_mpl(lv_ts); bt_x = _ts_to_mpl(bt_ts)
    ax0.step(lv_x, lv_eq, where='post', color=col_live, lw=2.2,
             label=f'Live  {lv_eq[-1]:+.0f}$ ({len(all_live_trades[asset])} trades)', zorder=4)
    ax0.step(bt_x, bt_eq, where='post', color=col_bt, lw=1.8, ls='--',
             label=f'BT    {bt_eq[-1]:+.0f}$ ({len(all_bt_trades[asset])} trades)',   zorder=3)
    ax0.axhline(0, color='#555', lw=0.8, ls='--')
    ax0.fill_between(lv_x, lv_eq, step='post', alpha=0.10, color=col_live)
    ax0.fill_between(bt_x, bt_eq, step='post', alpha=0.08, color=col_bt)
    for tp in tick_pos:
        ax0.axvline(tp, color='#334', lw=0.7, ls=':')
    ax0.set_xticks(tick_pos)
    ax0.set_xticklabels(tick_lbl, rotation=20, ha='right', fontsize=10, color='#ccc')
    ax0.set_title('Equité cumulée — tous TFs (trade par trade)', color='white', fontsize=12, pad=7)
    ax0.set_ylabel('PnL cumulé (USD)', color='#aaa', fontsize=10)
    ax0.legend(fontsize=10, framealpha=0.28, facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')

    # ── Stats table (par jour, toutes TFs agrégées) ───────────────────────────
    ax_t = fig.add_subplot(gs[1])
    ax_t.set_facecolor('#0e1117'); ax_t.axis('off')
    cols = ['Jour', 'L.Tradable', 'L.Traded', 'L.Fill%', 'L.WR%', 'L.PnL', 'L.Snipe',
            'BT.Tradable', 'BT.Traded', 'BT.Fill%', 'BT.WR%', 'BT.PnL']
    rows = []
    for d, dlbl in zip(DAYS_STR, DAYS_LBL):
        lt = ld = lw = btr = bfill = bw = 0
        lpnl = btpnl = lsnipe = 0.0
        for tf in ['m5', 'm15', 'h1']:
            lv = all_live[asset].get((d, tf), {})
            bt = all_bt[asset].get((d, tf), {})
            lt    += lv.get('tradable', 0); ld    += lv.get('traded',  0)
            lw    += lv.get('wins', 0)
            btr   += bt.get('tradable', 0); bfill += bt.get('trades',  0)
            bw    += bt.get('wins', 0)
            lpnl  += lv.get('pnl', 0.);    btpnl += bt.get('pnl', 0.)
            lsnipe+= lv.get('pnl_snipe', 0.)
        rows.append([dlbl,
                     str(lt), str(ld),
                     f'{100*ld/lt:.1f}%'    if lt    else '—',
                     f'{100*lw/ld:.1f}%'    if ld    else '—',
                     f'{lpnl:+.1f}',        f'{lsnipe:+.1f}',
                     str(btr), str(bfill),
                     f'{100*bfill/btr:.1f}%' if btr  else '—',
                     f'{100*bw/bfill:.1f}%' if bfill else '—',
                     f'{btpnl:+.1f}'])
    tbl = ax_t.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#333')
        cell.set_facecolor('#0d47a1' if r == 0 else ('#1a1a2e' if r%2==0 else '#12122a'))
        cell.set_text_props(color='white', fontweight='bold' if r==0 else 'normal')
    ax_t.set_title(f'{asset.upper()} — stats par jour (Apr 13-16)', color='white', fontsize=10, pad=6)

    slope = ASSET_CFGS[asset]['cfg']['slope']
    cap   = ASSET_CFGS[asset]['cfg']['eq_cap']
    fig.suptitle(f'{asset.upper()} — Live vs BT — Equité + Stats — slope={slope} cap={cap}',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    out = OUT_DIR_EQUITY / f"equity_{asset}_apr13_16.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close(fig)
    print(f"  -> {out}", flush=True)


def make_all_summary_fig():
    """Un seul graph avec les 4 courbes d'équité intraday + tableau global."""
    import matplotlib.dates as mdates
    assets = ['btc', 'eth', 'sol', 'xrp']
    tick_pos, tick_lbl = _day_ticks()

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#0e1117')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30,
                           height_ratios=[2.5, 2.5, 1.8])

    for ai, asset in enumerate(assets):
        col_live, col_bt = ASSET_CFGS[asset]['colors']
        row, col = divmod(ai, 2)
        ax = fig.add_subplot(gs[row, col])
        _style_ax(ax)
        lv_ts, lv_eq = _trades_to_equity(all_live_trades[asset])
        bt_ts, bt_eq = _trades_to_equity(all_bt_trades[asset])
        lv_x = _ts_to_mpl(lv_ts); bt_x = _ts_to_mpl(bt_ts)
        ax.step(lv_x, lv_eq, where='post', color=col_live, lw=2.0,
                label=f'Live  {lv_eq[-1]:+.0f}$', zorder=4)
        ax.step(bt_x, bt_eq, where='post', color=col_bt,   lw=1.6, ls='--',
                label=f'BT    {bt_eq[-1]:+.0f}$',  zorder=3)
        ax.axhline(0, color='#555', lw=0.7, ls='--')
        ax.fill_between(lv_x, lv_eq, step='post', alpha=0.09, color=col_live)
        ax.fill_between(bt_x, bt_eq, step='post', alpha=0.07, color=col_bt)
        for tp in tick_pos:
            ax.axvline(tp, color='#333', lw=0.5, ls=':')
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=30, ha='right', fontsize=8, color='#ccc')
        ax.set_title(f'{asset.upper()} — Equité cumulée (trade par trade)', color='white', fontsize=11)
        ax.set_ylabel('PnL cumulé (USD)', color='#aaa', fontsize=9)
        ax.legend(fontsize=9, framealpha=0.25, facecolor='#1a1a2e',
                  edgecolor='#444', labelcolor='white')

    # Tableau récap global
    ax_t = fig.add_subplot(gs[2, :])
    ax_t.set_facecolor('#0e1117'); ax_t.axis('off')
    cols = ['Jour'] + [f'{a.upper()} L.PnL' for a in assets] + \
           [f'{a.upper()} BT.PnL' for a in assets] + ['TOTAL Live', 'TOTAL BT']
    rows = []
    for d, dlbl in zip(DAYS_STR, DAYS_LBL):
        row_data = [dlbl]
        l_tot = bt_tot = 0.0
        l_pnls = []
        bt_pnls = []
        for asset in assets:
            lp = sum(all_live[asset].get((d, tf), {}).get('pnl', 0.) for tf in ['m5','m15','h1'])
            bp = sum(all_bt[asset].get((d, tf), {}).get('pnl', 0.)   for tf in ['m5','m15','h1'])
            l_pnls.append(f'{lp:+.0f}'); bt_pnls.append(f'{bp:+.0f}')
            l_tot += lp; bt_tot += bp
        row_data += l_pnls + bt_pnls + [f'{l_tot:+.0f}', f'{bt_tot:+.0f}']
        rows.append(row_data)
    tbl = ax_t.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.0, 1.9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#333')
        cell.set_facecolor('#0d47a1' if r == 0 else ('#1a1a2e' if r%2==0 else '#12122a'))
        cell.set_text_props(color='white', fontweight='bold' if r==0 else 'normal')
    ax_t.set_title('Tous assets — PnL quotidien Live vs BT (Apr 13-16)',
                   color='white', fontsize=10, pad=6)

    fig.suptitle('Live vs BT — BTC / ETH / SOL / XRP — Apr 13-16',
                 color='white', fontsize=14, fontweight='bold', y=1.00)
    out = OUT_DIR_EQUITY / "equity_all_apr13_16.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0e1117')
    plt.close(fig)
    print(f"  -> {out}", flush=True)


print("\n=== Génération figures ===", flush=True)
for asset in ['btc', 'eth', 'sol', 'xrp']:
    make_asset_fig(asset)
    make_equity_fig(asset)
make_all_summary_fig()
print("Terminé.", flush=True)
