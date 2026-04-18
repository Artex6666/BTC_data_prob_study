"""
compare_settlement_vs_old.py
Comparaison : ancienne méthode de résolution vs settlement Gamma CSV.

Ancienne méthode :
  BT   -> up_ask[-1] > down_ask[-1]
  Live -> end_spot (Binance) > start_spot

Nouvelle méthode :
  BT   -> settlement.csv (Gamma API)
  Live -> settlement.csv (Gamma API)

Génère un graphique 2-panneaux côte à côte + tableau stats.
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
    attach_settlement_outcomes, load_settlement_outcomes,
    TIMEFRAMES, VOL_LBS,
)

BASE     = Path(__file__).resolve().parent.parent
LIVE_DIR = BASE / "reportLive" / "safeChase" / "slope10int0VRS" / "btc"
CSV_DIR  = BASE / "reportLive" / "safeChase"
SETTLE   = BASE / "Datas" / "csv" / "settlement.csv"
OUT_DIR  = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF    = pd.Timestamp("2026-04-05T21:45:00", tz="UTC")
CUTOFF_TS = CUTOFF.timestamp()

CFG_BTC  = dict(curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
                vol_lb_h=None, vol_thresh=None,
                vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
                vrs_g_min=0.5, vrs_g_max=1.5,
                max_losses_cb=None, max_orders=2)
BTC_SIZE = 100.0

TF_MAP = {'m5': '5min', 'm15': '15min', 'h1': '1h'}


# ── Charge le settlement en dict {(asset, tf, open_ts): won_up} ───────────────
def load_settlement_dict() -> dict:
    if not SETTLE.exists():
        return {}
    out = {}
    df = pd.read_csv(SETTLE)
    df = df[df['outcome_up'].notna() & (df['outcome_up'] != '')]
    for _, row in df.iterrows():
        key = (str(row['asset']).lower(), str(row['tf']), int(row['open_ts']))
        out[key] = float(row['outcome_up']) > 0.5
    return out


# ── Parse live JSONL ──────────────────────────────────────────────────────────
def parse_live(asset_dir: Path, settlement: dict):
    """
    Retourne (windows_old, windows_new) :
      old : résolution Binance (end_spot > start_spot)
      new : résolution settlement Gamma CSV
    """
    asset = 'btc'
    windows_old = []
    windows_new = []

    for tf_slug in ['m5', 'm15', 'h1']:
        tf_key = TF_MAP[tf_slug]
        tf_dir = asset_dir / tf_slug
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
                    cur = dict(ts=ts, tf=tf_slug, open_ts=int(ts.timestamp()))
                    fills_prices = []; tradable = False

                elif ev == 'window_start' and cur:
                    tradable = True

                elif ev in ('vol_gate_skip', 'eth_trend_gate_skip') and cur:
                    tradable = False

                elif ev == 'fill' and cur:
                    p = float(d.get('price', 0))
                    if p > 0:
                        fills_prices.append(p)

                elif ev in ('window_ended', 'window_settled') and cur:
                    up_s  = float(d.get('up_shares',  0))
                    dn_s  = float(d.get('down_shares', 0))
                    up_c  = float(d.get('up_cost',    0))
                    dn_c  = float(d.get('down_cost',  0))
                    shares = up_s + dn_s
                    cost   = up_c + dn_c
                    traded = shares > 1e-9

                    start_sp = float(d.get('start_spot', 0))
                    end_sp   = float(d.get('end_spot',   0))

                    # ── Résolution ANCIENNE (Binance) ──
                    if start_sp > 0 and end_sp > 0:
                        up_wins_old = end_sp > start_sp
                    else:
                        up_wins_old = bool(d.get('epnl_up_wins', 1))

                    # ── Résolution NOUVELLE (settlement Gamma) ──
                    settle_key = (asset, tf_key, cur['open_ts'])
                    if settle_key in settlement:
                        up_wins_new = settlement[settle_key]
                    else:
                        up_wins_new = up_wins_old  # fallback si absent

                    avg_fill = float(np.mean(fills_prices)) if fills_prices else (
                        cost / shares if shares > 1e-9 else None)

                    def make_window(up_wins):
                        if traded:
                            pnl = (up_s - up_c - dn_c) if up_wins else (dn_s - dn_c - up_c)
                            won = pnl > 0
                        else:
                            pnl = 0.0; won = None
                        return dict(
                            ts=cur['ts'], tf=tf_slug,
                            tradable=tradable, traded=traded,
                            won=won, pnl=pnl, avg_fill=avg_fill,
                            shares=shares, cost=cost,
                            open_ts=cur['open_ts'],
                        )

                    windows_old.append(make_window(up_wins_old))
                    windows_new.append(make_window(up_wins_new))
                    cur = None; fills_prices = []; tradable = False

    windows_old.sort(key=lambda x: x['ts'])
    windows_new.sort(key=lambda x: x['ts'])
    return windows_old, windows_new


# ── Run BT (ancienne ou nouvelle méthode) ────────────────────────────────────
def run_bt(use_settlement: bool) -> list:
    csv = [str(CSV_DIR / "BTC.csv")]
    m5_ref = None
    all_trades = []

    for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
        cts = load_contracts(csv, tf_floor, b1, b2, a1, a2)
        if tf_floor == '5min':
            precompute_vol(cts, VOL_LBS)
            m5_ref = cts
        else:
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)

        if use_settlement:
            n = attach_settlement_outcomes(cts, 'btc', tf=tf_floor)
        else:
            # S'assurer qu'aucun settlement n'est attaché
            for c in cts:
                c.pop('_settlement_won_up', None)

        tf_name = ['m5', 'm15', 'h1'][idx]
        orig = cu.BASE_SIZE; cu.BASE_SIZE = float(BTC_SIZE)

        for c in cts:
            if c.get('open_ts', 0) < CUTOFF_TS:
                continue
            won, pnl, fills = simulate_trade_log(c, CFG_BTC)
            if won is None:
                continue
            avg_fill = float(np.mean([f['price'] for f in fills])) if fills else 0.0
            all_trades.append(dict(ts=c['ce'], tf=tf_name, pnl=pnl, won=won, avg_fill=avg_fill))

        cu.BASE_SIZE = orig

    all_trades.sort(key=lambda x: x['ts'])
    return all_trades


# ── Equity & stats ────────────────────────────────────────────────────────────
def trades_to_equity(trades, ref_ts):
    if not trades:
        return [], []
    xs, ys, cum = [], [], 0.0
    for t in trades:
        ts = t['ts'].timestamp() if hasattr(t['ts'], 'timestamp') else float(t['ts'])
        cum += t['pnl']
        xs.append((ts - ref_ts) / 3600)
        ys.append(cum)
    return xs, ys


def compute_stats(windows, trades, label):
    tradable = sum(1 for w in windows if w.get('tradable'))
    n        = len(trades)
    wins     = sum(1 for t in trades if t.get('won'))
    losses   = sum(1 for t in trades if t.get('won') is False)
    pnl      = sum(t['pnl'] for t in trades)
    fills    = [t['avg_fill'] for t in trades if t.get('avg_fill')]
    cum      = np.cumsum([t['pnl'] for t in trades]) if trades else np.array([0.0])
    peak     = np.maximum.accumulate(cum)
    maxdd    = float((peak - cum).max()) if len(cum) else 0.0
    return dict(label=label, tradable=tradable, n=n, wins=wins, losses=losses,
                wr=100*wins/n if n else 0, pnl=pnl, maxdd=maxdd,
                avg_fill=float(np.mean(fills)) if fills else 0)


# ── Main ─────────────────────────────────────────────────────────────────────
print("Chargement settlement...", flush=True)
settlement = load_settlement_dict()
print(f"  {len(settlement)} outcomes charges", flush=True)

print("\nParse live BTC...", flush=True)
live_old_w, live_new_w = parse_live(LIVE_DIR, settlement)
live_old = [w for w in live_old_w if w['traded']]
live_new = [w for w in live_new_w if w['traded']]
print(f"  {len(live_old)} trades live", flush=True)

# Vérif : combien de contrats ont une résolution différente entre les deux méthodes
diffs = sum(1 for a, b in zip(live_old, live_new) if a['won'] != b['won'])
print(f"  Divergences live old vs new : {diffs}/{len(live_old)}", flush=True)

print("\nRun BT ancienne methode...", flush=True)
bt_old = run_bt(use_settlement=False)
print(f"  {len(bt_old)} trades BT old", flush=True)

print("Run BT nouvelle methode...", flush=True)
bt_new = run_bt(use_settlement=True)
print(f"  {len(bt_new)} trades BT new", flush=True)

# Période
live_end_ts = max((t['ts'].timestamp() for t in live_old), default=CUTOFF_TS)
live_span_h = (live_end_ts - CUTOFF_TS) / 3600
live_span_d = live_span_h / 24

# Stats
s = {
    'live_old': compute_stats(live_old_w, live_old, 'Live (Binance)'),
    'live_new': compute_stats(live_new_w, live_new, 'Live (Gamma)'),
    'bt_old':   compute_stats([], bt_old, 'BT (ask-based)'),
    'bt_new':   compute_stats([], bt_new, 'BT (Gamma)'),
}

# Equity curves
ref = CUTOFF_TS
lox, loy = trades_to_equity(live_old, ref)
lnx, lny = trades_to_equity(live_new, ref)
box, boy = trades_to_equity(bt_old, ref)
bnx, bny = trades_to_equity(bt_new, ref)

# ── Graphique ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35,
                        height_ratios=[3, 2, 1.3])

C_LIVE_OLD = '#EF9A9A'   # rouge clair
C_LIVE_NEW = '#F44336'   # rouge vif
C_BT_OLD   = '#B3E5FC'   # bleu clair
C_BT_NEW   = '#1565C0'   # bleu foncé

total_h = live_span_h
xticks  = np.arange(0, total_h + 2, 6)

def make_xticks(ax):
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
    ax.set_xlim(-0.5, total_h + 0.5)
    ax.grid(alpha=0.25)


# ── Panneau gauche : ANCIENNE METHODE ─────────────────────────────────────────
ax_l = fig.add_subplot(gs[0, 0])
if boy:
    ax_l.plot(box, boy, color=C_BT_OLD, linewidth=2.0, linestyle='--',
              label=f'BT ask  {boy[-1]:+.0f}$  WR={s["bt_old"]["wr"]:.1f}%  T={s["bt_old"]["n"]}')
if loy:
    ax_l.plot(lox, loy, color=C_LIVE_OLD, linewidth=2.2,
              label=f'Live Binance  {loy[-1]:+.0f}$  WR={s["live_old"]["wr"]:.1f}%  T={s["live_old"]["n"]}')
    ax_l.scatter(lox, loy, color=C_LIVE_OLD, s=15, zorder=4, alpha=0.6)
ax_l.axhline(0, color='gray', linewidth=0.6, linestyle='--')
ax_l.set_title('Ancienne méthode\nBT: ask-based  |  Live: Binance spot', fontsize=10, fontweight='bold')
ax_l.set_ylabel('PnL cumulé ($)')
ax_l.legend(fontsize=8)
make_xticks(ax_l)

# ── Panneau droit : NOUVELLE METHODE ──────────────────────────────────────────
ax_r = fig.add_subplot(gs[0, 1])
if bny:
    ax_r.plot(bnx, bny, color=C_BT_NEW, linewidth=2.0, linestyle='--',
              label=f'BT Gamma  {bny[-1]:+.0f}$  WR={s["bt_new"]["wr"]:.1f}%  T={s["bt_new"]["n"]}')
if lny:
    ax_r.plot(lnx, lny, color=C_LIVE_NEW, linewidth=2.2,
              label=f'Live Gamma  {lny[-1]:+.0f}$  WR={s["live_new"]["wr"]:.1f}%  T={s["live_new"]["n"]}')
    ax_r.scatter(lnx, lny, color=C_LIVE_NEW, s=15, zorder=4, alpha=0.6)
ax_r.axhline(0, color='gray', linewidth=0.6, linestyle='--')
ax_r.set_title('Nouvelle méthode\nBT: Gamma CSV  |  Live: Gamma CSV', fontsize=10, fontweight='bold')
ax_r.set_ylabel('PnL cumulé ($)')
ax_r.legend(fontsize=8)
make_xticks(ax_r)

# ── Drawdowns ─────────────────────────────────────────────────────────────────
ax_ddl = fig.add_subplot(gs[1, 0])
if boy:
    cum = np.array(boy); ax_ddl.fill_between(box, 0, -(np.maximum.accumulate(cum)-cum),
                                              color=C_BT_OLD, alpha=0.5, label='DD BT ask')
if loy:
    cum = np.array(loy); ax_ddl.fill_between(lox, 0, -(np.maximum.accumulate(cum)-cum),
                                              color=C_LIVE_OLD, alpha=0.5, label='DD Live Binance')
ax_ddl.axhline(0, color='gray', linewidth=0.5)
ax_ddl.set_ylabel('Drawdown ($)'); ax_ddl.legend(fontsize=8)
make_xticks(ax_ddl)

ax_ddr = fig.add_subplot(gs[1, 1])
if bny:
    cum = np.array(bny); ax_ddr.fill_between(bnx, 0, -(np.maximum.accumulate(cum)-cum),
                                              color=C_BT_NEW, alpha=0.5, label='DD BT Gamma')
if lny:
    cum = np.array(lny); ax_ddr.fill_between(lnx, 0, -(np.maximum.accumulate(cum)-cum),
                                              color=C_LIVE_NEW, alpha=0.5, label='DD Live Gamma')
ax_ddr.axhline(0, color='gray', linewidth=0.5)
ax_ddr.set_ylabel('Drawdown ($)'); ax_ddr.legend(fontsize=8)
make_xticks(ax_ddr)

# ── Tableau stats (toute la largeur) ──────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.axis('off')

pnl_h = lambda st: st['pnl'] / max(live_span_h, 0.01)
cols = ['', 'Tradés', 'Wins', 'Losses', 'WR%', 'Fill moy', 'PnL total', 'PnL/h', 'MaxDD']
rows_data = []
colors_row = ['#FFCDD2', '#FFCDD2', '#BBDEFB', '#BBDEFB']
for key, lbl in [('live_old','Live (Binance)'), ('live_new','Live (Gamma)'),
                 ('bt_old','BT (ask-based)'),  ('bt_new','BT (Gamma)')]:
    st = s[key]
    rows_data.append([
        lbl,
        str(st['n']),
        str(st['wins']),
        str(st['losses']),
        f"{st['wr']:.1f}%",
        f"{st['avg_fill']:.3f}" if st['avg_fill'] else '—',
        f"${st['pnl']:+.1f}",
        f"${pnl_h(st):+.2f}/h",
        f"${st['maxdd']:.0f}",
    ])

tbl = ax_tbl.table(cellText=rows_data, colLabels=cols, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
for j in range(len(cols)):
    tbl[0, j].set_facecolor('#37474F')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
for i, col in enumerate(colors_row):
    for j in range(len(cols)):
        tbl[i+1, j].set_facecolor(col)

ax_tbl.set_title(
    f"Stats  {CUTOFF.strftime('%Y-%m-%d %H:%M')} UTC -> +{live_span_h:.1f}h  "
    f"|  Divergences live old/new : {diffs}/{len(live_old)}",
    fontsize=9, pad=6, fontweight='bold'
)

fig.suptitle(
    f'BTC — Résolution ancienne (ask/Binance) vs nouvelle (Gamma CSV)\n'
    f'slope=10 cap=0.55 VRS1h b=150 g=0.5-1.5  $100x2',
    fontsize=13, fontweight='bold', y=1.01
)

out = OUT_DIR / "settlement_vs_old_method.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"\n-> {out}", flush=True)

# Console
print(f"\n=== RECAP ({live_span_h:.1f}h depuis cutoff) ===")
for key, lbl in [('live_old','Live Binance'), ('live_new','Live Gamma'),
                 ('bt_old','BT ask-based'),   ('bt_new','BT Gamma')]:
    st = s[key]
    print(f"  {lbl:<18}  T={st['n']:>3}  W={st['wins']} L={st['losses']}  "
          f"WR={st['wr']:.1f}%  PnL={st['pnl']:+.1f}$  ({pnl_h(st):+.2f}$/h)  maxDD=${st['maxdd']:.0f}")

print(f"\n  Divergences live old vs new : {diffs}/{len(live_old)}")
if diffs > 0:
    print("  Contrats avec resolution differente :")
    for a, b in zip(live_old, live_new):
        if a['won'] != b['won']:
            ts_str = a['ts'].strftime('%Y-%m-%d %H:%M')
            print(f"    {ts_str} UTC  tf={a['tf']}  "
                  f"Binance={'WIN' if a['won'] else 'LOSS'}  "
                  f"Gamma={'WIN' if b['won'] else 'LOSS'}  "
                  f"pnl_old={a['pnl']:+.2f}$  pnl_new={b['pnl']:+.2f}$")
