"""
Comparaison M5 uniquement :
  Config A : lin sl=2 int=0 cap=0.65  (pas de filtre vol)
  Config B : lin sl=2 int=0 cap=0.65  vol8h>$500 (range)

+ Calcul du rebate Polymarket par fill :
    fee_equivalent = C × p × 0.072 × (p × (1-p))^1
    Avec C = 50/p (shares pour un ordre de $50) → fee = 3.6 × p × (1-p)

Sorties :
  god_curve_charts/lin_sl2_c065/equity_A_no_filter.png
  god_curve_charts/lin_sl2_c065/equity_B_vol8h500.png
  god_curve_charts/lin_sl2_c065/comparison.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict

CSV_PATHS  = ["BTC_BIG.csv", "BTC.csv"]
OUT_DIR    = Path("god_curve_charts/lin_sl2_c065")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_ORDERS = 3
BASE_SIZE  = 50.0
FEE_RATE   = 0.072
FEE_EXP    = 1.0
VOL_LB_S   = 8 * 3600

CONFIGS = [
    dict(name="A_no_filter", label="lin sl=2 cap=0.65  (sans filtre)",
         slope=2.0, intercept=0.0, eq_cap=0.65, vol_thresh=None),
    dict(name="B_vol8h500",  label="lin sl=2 cap=0.65  vol8h>$500",
         slope=2.0, intercept=0.0, eq_cap=0.65, vol_thresh=500.0),
]

COLORS = ["#2196F3", "#FF5722"]


# -- Chargement M5 uniquement --------------------------------------------------
def load():
    dfs = []
    for p in CSV_PATHS:
        d = pd.read_csv(p, usecols=['timestamp','spot_price','m5_up_bid','m5_down_bid'])
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
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
        })
    del df
    # Precompute vol range 8h (pas de look-ahead)
    window = deque()
    for c in contracts:
        t = c['ce_ts']
        while window and t - window[0][0] > VOL_LB_S: window.popleft()
        sw = [x[1] for x in window]
        c['vol8h'] = (max(sw) - min(sw)) if sw else 0.0
        window.append((t, c['op']))
    return contracts


# -- Simulation avec tracking des fills et rebate -----------------------------
def simulate_with_fills(c, cfg):
    """
    Retourne (won, pnl, fills_info) ou (None, None, []).
    fills_info : liste de (fill_price,) par fill exécuté.
    """
    cap    = cfg['eq_cap']
    slope  = cfg['slope']
    interc = cfg['intercept']
    ce_ns  = np.datetime64(c['ce'])
    spots  = c['spot']; ts_arr = c['ts']; op = c['op']

    activated_side = fill_side = None
    contract_cost = contract_shares = 0.0; fill_count = 0
    active_order = active_size = pending_place = pending_size = None
    pending_cancel = False
    fills_prices = []  # prix de chaque fill

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or fill_count >= MAX_ORDERS: break
        ub = round(c['up_bid'][j], 2); db = round(c['down_bid'][j], 2)

        if pending_cancel and active_order is not None:
            if activated_side:
                cb2 = ub if activated_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    fill_count += 1; contract_cost += active_size
                    contract_shares += active_size / active_order
                    fills_prices.append(active_order)
                    if fill_side is None: fill_side = activated_side
                    if fill_count >= MAX_ORDERS: active_order = None; break
            active_order = active_size = None; pending_cancel = False

        if pending_place is not None:
            active_order = pending_place; active_size = pending_size
            pending_place = pending_size = None

        if active_order is not None and activated_side:
            cb2 = ub if activated_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fill_count += 1; contract_cost += active_size
                contract_shares += active_size / active_order
                fills_prices.append(active_order)
                if fill_side is None: fill_side = activated_side
                active_order = active_size = None
                if fill_count >= MAX_ORDERS: break

        thresh = slope * remain + interc
        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = None; activated_side = None; continue

        if not activated_side:
            if   s - op >= thresh: activated_side = 'UP'
            elif op - s >= thresh: activated_side = 'DOWN'
        if not activated_side: continue

        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = BASE_SIZE
            elif active_order is not None and proposed > active_order:
                pending_cancel = True; pending_place = proposed; pending_size = BASE_SIZE
            elif pending_place is not None and proposed > pending_place:
                pending_place = proposed; pending_size = BASE_SIZE

    won_up = c['up_bid'][-1] > c['down_bid'][-1]
    if contract_shares > 0:
        won = (won_up == (fill_side == 'UP'))
        pnl = (contract_shares if won else 0.0) - contract_cost
        return won, pnl, fills_prices
    return None, None, []


# -- Index horaire -------------------------------------------------------------
def build_hour_index(contracts):
    all_hours = sorted({c['ce'].replace(minute=0, second=0, microsecond=0)
                        for c in contracts})
    return {h: i for i, h in enumerate(all_hours)}, len(all_hours)


def make_xlabels(hour_index, step=24):
    inv = {v: k for k, v in hour_index.items()}
    n   = max(hour_index.values()) + 1
    pos = list(range(0, n, step))
    lbl = [inv[p].strftime('%m/%d') if p in inv else '' for p in pos]
    return pos, lbl


def esc(s): return str(s).replace('$', r'\$')


# -- Run -----------------------------------------------------------------------
print("Loading M5 contracts...", flush=True)
contracts = load()
hour_index, n_hours = build_hour_index(contracts)
print(f"  {len(contracts)} contracts  |  {n_hours} heures tradees\n")

x    = np.arange(n_hours)
days = n_hours / 24
xticks, xlabels = make_xlabels(hour_index, step=24)

series = []

for cfg in CONFIGS:
    pnl_by_hour    = defaultdict(float)
    rebate_by_hour = defaultdict(float)
    wins = losses  = 0
    total_fills    = 0
    total_rebate   = 0.0
    prices_all     = []

    for c in contracts:
        if cfg['vol_thresh'] is not None and c['vol8h'] < cfg['vol_thresh']:
            continue
        won, pnl, fill_prices = simulate_with_fills(c, cfg)
        if won is None: continue

        h     = c['ce'].replace(minute=0, second=0, microsecond=0)
        hidx  = hour_index.get(h)
        if hidx is None: continue

        pnl_by_hour[hidx] += pnl
        if won: wins += 1
        else:   losses += 1

        # Rebate par fill : fee = 3.6 × p × (1-p)
        for p in fill_prices:
            fee = BASE_SIZE * FEE_RATE * p * (1.0 - p)  # = 3.6 × p × (1-p)
            rebate_by_hour[hidx] += fee
            total_rebate          += fee
            total_fills           += 1
            prices_all.append(p)

    # Séries cumulées
    arr_pnl    = np.zeros(n_hours)
    arr_rebate = np.zeros(n_hours)
    for i, v in pnl_by_hour.items():
        arr_pnl[i]    += v
    for i, v in rebate_by_hour.items():
        arr_rebate[i] += v

    cum_pnl    = np.cumsum(arr_pnl)
    cum_rebate = np.cumsum(arr_rebate)
    cum_total  = cum_pnl + cum_rebate

    dd_pnl   = (np.maximum.accumulate(cum_pnl)   - cum_pnl).max()
    dd_total = (np.maximum.accumulate(cum_total)  - cum_total).max()
    rf_pnl   = cum_pnl[-1]   / dd_pnl   if dd_pnl   > 0 else float('inf')
    rf_total = cum_total[-1]  / dd_total if dd_total  > 0 else float('inf')
    wr       = wins / (wins+losses) * 100 if (wins+losses) else 0

    print(f"{'-'*65}")
    print(f"CONFIG : {cfg['label']}")
    print(f"  Contracts   : {wins+losses}  WR={wr:.1f}%  Total fills={total_fills}")
    print(f"  Fill avg px : ${np.mean(prices_all):.3f}  "
          f"min=${min(prices_all):.2f}  max=${max(prices_all):.2f}")
    print(f"  PnL/j       : ${cum_pnl[-1]/days:+.1f}   total=${cum_pnl[-1]:+.0f}")
    print(f"  Rebate/j    : ${total_rebate/days:+.2f}  total=${total_rebate:+.0f}")
    print(f"  PnL+Rebate/j: ${cum_total[-1]/days:+.1f}   total=${cum_total[-1]:+.0f}")
    print(f"  RF (pnl)    : {rf_pnl:.1f}x  |  RF (pnl+rebate) : {rf_total:.1f}x")

    series.append(dict(
        cfg=cfg, cum_pnl=cum_pnl, cum_rebate=cum_rebate, cum_total=cum_total,
        arr_pnl=arr_pnl, wins=wins, losses=losses,
    ))

print(f"{'-'*65}\n")


# -- Chart individuel (pnl + pnl+rebate) --------------------------------------
for s in series:
    cfg = s['cfg']
    cum_p = s['cum_pnl']; cum_t = s['cum_total']
    dd    = (np.maximum.accumulate(cum_t) - cum_t)
    wr    = s['wins'] / (s['wins']+s['losses']) * 100

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})
    fig.subplots_adjust(hspace=0.45)

    axes[0].plot(x, cum_p, color="#2196F3", linewidth=1.5, label="PnL seul")
    axes[0].plot(x, cum_t, color="#4CAF50", linewidth=1.5, linestyle="--",
                 label=esc(f"PnL + rebate (+${s['cum_rebate'][-1]:.0f})"))
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title(esc(f"{cfg['label']}  (M5)  —  {s['wins']+s['losses']} trades  "
                          f"WR={wr:.1f}%  ${cum_t[-1]/days:.0f}/j"), fontsize=10)
    axes[0].set_ylabel("PnL cumule ($)"); axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_xticks(xticks); axes[0].set_xticklabels(xlabels, fontsize=8, rotation=30)

    bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in s['arr_pnl']]
    axes[1].bar(x, s['arr_pnl'], color=bar_colors, width=1.0, alpha=0.85)
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].set_ylabel("PnL/heure ($)"); axes[1].grid(alpha=0.3, axis='y')
    axes[1].set_xticks(xticks); axes[1].set_xticklabels(xlabels, fontsize=8, rotation=30)

    axes[2].fill_between(x, dd, 0, color="#f44336", alpha=0.4)
    axes[2].plot(x, dd, color="#f44336", linewidth=1)
    axes[2].set_ylabel("Drawdown ($)")
    axes[2].set_xlabel("Heures tradees (gaps exclus)")
    axes[2].grid(alpha=0.3); axes[2].invert_yaxis()
    axes[2].set_xticks(xticks); axes[2].set_xticklabels(xlabels, fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"equity_{cfg['name']}.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# -- Chart de comparaison overlay ----------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                          gridspec_kw={"height_ratios": [3, 1.2, 1]})
fig.subplots_adjust(hspace=0.45)
fig.suptitle(esc("lin sl=2 cap=0.65 — sans filtre vs vol8h>$500  (M5 seulement)"), fontsize=13)

for i, s in enumerate(series):
    col  = COLORS[i]
    cum_p = s['cum_pnl']; cum_t = s['cum_total']
    dd_t  = np.maximum.accumulate(cum_t) - cum_t
    rf_t  = cum_t[-1] / dd_t.max() if dd_t.max() > 0 else float('inf')
    wr    = s['wins'] / (s['wins']+s['losses']) * 100
    lbl_p = esc(f"{s['cfg']['label']}  WR={wr:.0f}%  ${cum_p[-1]/days:.0f}/j")
    lbl_t = esc(f"  + rebate  ${cum_t[-1]/days:.0f}/j  RF={rf_t:.0f}x")

    axes[0].plot(x, cum_p, color=col, linewidth=1.8, label=lbl_p)
    axes[0].plot(x, cum_t, color=col, linewidth=1.2, linestyle='--', label=lbl_t, alpha=0.75)
    axes[1].bar(x + i * 0.45 - 0.22, s['arr_pnl'], width=0.45,
                color=col, alpha=0.6, label=s['cfg']['label'])
    axes[2].plot(x, dd_t, color=col, linewidth=1.5, label=s['cfg']['label'])

axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
axes[0].set_ylabel("PnL cumule ($)")
axes[0].legend(loc="upper left", fontsize=8); axes[0].grid(alpha=0.3)
axes[0].set_xticks(xticks); axes[0].set_xticklabels(xlabels, fontsize=8, rotation=30)

axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_ylabel("PnL/heure ($)")
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis='y')
axes[1].set_xticks(xticks); axes[1].set_xticklabels(xlabels, fontsize=8, rotation=30)

axes[2].set_ylabel("Drawdown ($) — PnL+rebate")
axes[2].set_xlabel("Heures tradees (gaps exclus)")
axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3); axes[2].invert_yaxis()
axes[2].set_xticks(xticks); axes[2].set_xticklabels(xlabels, fontsize=8, rotation=30)

plt.tight_layout()
plt.savefig(OUT_DIR / "comparison.png", dpi=150, bbox_inches="tight")
plt.close("all")
print(f"Charts -> {OUT_DIR}/")
