"""
compare_limit_vs_market.py
==========================
Comparaison LIMIT maker vs MARKET taker — slope=7 cap=0.75 vol_net_1h>=60

Fees Polymarket (formules distinctes) :
  Maker rebate : cost × 0.072 × p × (1-p)        [ajouté au PnL]
  Taker fee    : cost × 0.072 × (1-p)             [soustrait au PnL]
    (taker fee = C×0.072×p×(1-p) avec C=cost/p → cost×0.072×(1-p))

Règles identiques pour les deux modes :
  - slope=7, intercept=0, cap=0.75, vol_net_1h>=60, max_orders=2/side
  - settlement UNIQUEMENT via settlement.csv (skip si absent)
  - même logique trigger / deactivation

Sorties :
  live_vs_bt/limit_vs_market_last4d.png
  live_vs_bt/limit_vs_market_all.png
"""
import sys
from pathlib import Path

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
    load_contracts, precompute_vol, attach_settlement_outcomes,
    TIMEFRAMES, VOL_LBS,
)

BASE    = Path(__file__).resolve().parent.parent
CSV_DIR = BASE / "reportLive" / "safeChase"
OUT_DIR = Path(__file__).resolve().parent / "live_vs_bt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOW_UTC   = pd.Timestamp.now(tz="UTC")
CUTOFF_4D = (NOW_UTC - pd.Timedelta(days=4)).replace(tzinfo=None)
CUTOFF_4D = pd.Timestamp(CUTOFF_4D).tz_localize("UTC")

CFG = dict(
    curve='linear', slope=7.0, intercept=0.0, eq_cap=0.75,
    vol_lb_h=1, vol_thresh=60.0, vol_type='net',
    max_orders=2,
)

ORDER_SIZE    = 100.0
FEE_RATE      = 0.072
FILL_DELAY_S  = 0.300
MARKET_TARGET = 0.99


# ── Formules fees ─────────────────────────────────────────────────────────────

def maker_rebate(cost, price):
    """Rebate maker Polymarket : cost × 0.072 × p × (1-p)"""
    return cost * FEE_RATE * price * (1.0 - price)

def taker_fee(cost, price):
    """Fee taker Polymarket : C×0.072×p×(1-p) avec C=cost/p → cost×0.072×(1-p)"""
    return cost * FEE_RATE * (1.0 - price)


# ── Settlement CSV only ───────────────────────────────────────────────────────

def get_settlement(c):
    if '_settlement_won_up' in c:
        return bool(c['_settlement_won_up'])
    return None


def make_result(cost_up, cost_down, shares_up, shares_down, fills, won_up):
    total_cost = cost_up + cost_down
    if total_cost <= 0:
        return None, None, []
    payout = (shares_up if won_up else 0.0) + (shares_down if not won_up else 0.0)
    pnl    = payout - total_cost
    won    = pnl > 0
    return won, pnl, fills


# ── Simulation LIMIT maker ────────────────────────────────────────────────────

def simulate_limit(c, cfg):
    won_up = get_settlement(c)
    if won_up is None:
        return None, None, None, []

    cap    = cfg['eq_cap']
    ce_ns  = np.datetime64(c['ce'])
    spots  = c['spot']; ts_arr = c['ts']; op = c['op']
    up_bid = c['up_bid']; down_bid = c['down_bid']

    activated_side = None
    fc_up = fc_down = 0
    shares_up = shares_down = 0.0
    cost_up   = cost_down   = 0.0
    active_order = active_size = active_ord_side = None
    pending_place = pending_size = pending_ord_side = None
    pending_cancel = False
    fills      = []
    max_orders = int(cfg.get('max_orders', 2))

    for j in range(len(spots)):
        s      = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders):
            break

        ub = round(up_bid[j], 2); db = round(down_bid[j], 2)

        # Cancel-fill
        if pending_cancel and active_order is not None:
            if active_ord_side:
                cb2 = ub if active_ord_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    fills.append({'j': j, 'side': active_ord_side, 'price': float(active_order)})
                    if active_ord_side == 'UP':
                        fc_up += 1; cost_up += active_size; shares_up += active_size / active_order
                    else:
                        fc_down += 1; cost_down += active_size; shares_down += active_size / active_order
                    if fc_up >= max_orders and fc_down >= max_orders:
                        active_order = active_ord_side = None; break
            active_order = active_size = active_ord_side = None; pending_cancel = False

        # Pending → active
        if pending_place is not None:
            fc_pend = fc_up if pending_ord_side == 'UP' else fc_down
            if fc_pend < max_orders:
                active_order = pending_place; active_size = pending_size; active_ord_side = pending_ord_side
            pending_place = pending_size = pending_ord_side = None

        # Fill actif
        if active_order is not None and active_ord_side:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fills.append({'j': j, 'side': active_ord_side, 'price': float(active_order)})
                if active_ord_side == 'UP':
                    fc_up += 1; cost_up += active_size; shares_up += active_size / active_order
                else:
                    fc_down += 1; cost_down += active_size; shares_down += active_size / active_order
                active_order = active_size = active_ord_side = None
                if fc_up >= max_orders and fc_down >= max_orders: break

        thresh = cfg['slope'] * remain + cfg['intercept']

        # Deactivation
        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = pending_ord_side = None
                activated_side = None; continue

        # Trigger
        if not activated_side:
            if   fc_up   < max_orders and (s - op) >= thresh: activated_side = 'UP'
            elif fc_down < max_orders and (op - s) >= thresh: activated_side = 'DOWN'
        if not activated_side: continue

        if (activated_side == 'UP' and fc_up >= max_orders) or \
           (activated_side == 'DOWN' and fc_down >= max_orders):
            activated_side = None; continue

        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = ORDER_SIZE; pending_ord_side = activated_side
            elif active_order is not None and proposed > active_order and active_ord_side == activated_side:
                pending_cancel = True
                pending_place = proposed; pending_size = ORDER_SIZE; pending_ord_side = activated_side
            elif pending_place is not None and proposed > pending_place and pending_ord_side == activated_side:
                pending_place = proposed; pending_size = ORDER_SIZE; pending_ord_side = activated_side

    # Expiry fill côté perdant
    exp_side = pending_ord_side if pending_place is not None else active_ord_side
    exp_ord  = pending_place    if pending_place is not None else active_order
    exp_sz   = pending_size     if pending_place is not None else active_size
    exp_fc   = (fc_up if exp_side == 'UP' else fc_down) if exp_side else max_orders
    if exp_fc < max_orders and exp_side and exp_ord and exp_sz:
        if (exp_side == 'UP') != won_up:
            fills.append({'j': len(spots)-1, 'side': exp_side, 'price': float(exp_ord)})
            if exp_side == 'UP': fc_up += 1; cost_up += exp_sz; shares_up += exp_sz / exp_ord
            else:                 fc_down += 1; cost_down += exp_sz; shares_down += exp_sz / exp_ord

    won, pnl, fills = make_result(cost_up, cost_down, shares_up, shares_down, fills, won_up)
    if won is None:
        return None, None, None, []

    rebate = sum(maker_rebate(ORDER_SIZE, f['price']) for f in fills)
    return won, pnl, -rebate, fills   # fee_net < 0 → gain


# ── Simulation MARKET taker ───────────────────────────────────────────────────

def simulate_market(c, cfg):
    won_up = get_settlement(c)
    if won_up is None:
        return None, None, None, []

    cap      = cfg['eq_cap']
    ce_ns    = np.datetime64(c['ce'])
    spots    = c['spot']; ts_arr = c['ts']; op = c['op']
    up_ask   = c['up_ask']; down_ask = c['down_ask']
    up_bid   = c['up_bid']; down_bid = c['down_bid']

    activated_side = None
    fc_up = fc_down = 0
    shares_up = shares_down = 0.0
    cost_up   = cost_down   = 0.0
    max_orders = int(cfg.get('max_orders', 2))
    fills = []
    ts_f  = ts_arr.astype(np.int64) / 1e9

    for j in range(len(spots)):
        s      = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders):
            break

        thresh = cfg['slope'] * remain + cfg['intercept']

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                activated_side = None; continue

        if not activated_side:
            if   fc_up   < max_orders and (s - op) >= thresh: activated_side = 'UP'
            elif fc_down < max_orders and (op - s) >= thresh: activated_side = 'DOWN'
        if not activated_side: continue

        if (activated_side == 'UP' and fc_up >= max_orders) or \
           (activated_side == 'DOWN' and fc_down >= max_orders):
            activated_side = None; continue

        # Fill market : ask à t+300ms, cap <= ask <= 0.99
        t_fill = ts_f[j] + FILL_DELAY_S
        k = j + 1
        while k < len(spots) and ts_f[k] < t_fill:
            k += 1
        k = min(k, len(spots) - 1)

        ask_arr  = up_ask if activated_side == 'UP' else down_ask
        ask_fill = float(ask_arr[k])
        filled_side    = activated_side
        activated_side = None

        if cap <= ask_fill <= MARKET_TARGET:
            price = round(ask_fill, 4)
            if filled_side == 'UP':
                fc_up    += 1; cost_up   += ORDER_SIZE; shares_up   += ORDER_SIZE / price
            else:
                fc_down  += 1; cost_down += ORDER_SIZE; shares_down += ORDER_SIZE / price
            fills.append({'j': k, 'side': filled_side, 'price': price})

    won, pnl, fills = make_result(cost_up, cost_down, shares_up, shares_down, fills, won_up)
    if won is None:
        return None, None, None, []

    fee = sum(taker_fee(ORDER_SIZE, f['price']) for f in fills)
    return won, pnl, +fee, fills   # fee_net > 0 → coût


# ── Simulation HYBRID : market + limit en parallèle ──────────────────────────
#
# À chaque trigger sur un side :
#   1. On envoie un ordre market → fill garanti à ask[t+300ms] (taker fee)
#   2. On continue à chasser le limit → fill bonus si bid baisse de 0.01 (maker rebate)
# Les deux fills sont indépendants et s'accumulent vers max_orders.
# Deactivation = annule le limit (le market est déjà soumis et fire quoi qu'il arrive).

def simulate_hybrid(c, cfg):
    won_up = get_settlement(c)
    if won_up is None:
        return None, None, None, []

    cap      = cfg['eq_cap']
    ce_ns    = np.datetime64(c['ce'])
    spots    = c['spot']; ts_arr = c['ts']; op = c['op']
    up_ask   = c['up_ask']; down_ask  = c['down_ask']
    up_bid   = c['up_bid']; down_bid  = c['down_bid']

    activated_side = None
    fc_up = fc_down = 0
    shares_up = shares_down = 0.0
    cost_up   = cost_down   = 0.0
    max_orders = int(cfg.get('max_orders', 2))
    fills = []
    ts_f  = ts_arr.astype(np.int64) / 1e9

    # État de l'ordre limit en cours (peut coexister avec le market)
    active_order = active_size = active_ord_side = None
    pending_place = pending_size = pending_ord_side = None
    pending_cancel = False

    # Market orders déjà soumis : {(tick_fill, side, price)} — on les fire quand on arrive
    pending_market = []   # list of (ts_fill_f, side, ask_price)

    for j in range(len(spots)):
        s      = spots[j]
        t_now  = ts_f[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders):
            break

        ub = round(up_bid[j], 2); db = round(down_bid[j], 2)

        # ── Fire les market orders arrivés à maturité ─────────────────────────
        still_pending = []
        for (t_mkt, mkt_side, mkt_price) in pending_market:
            if t_now >= t_mkt:
                fc_s = fc_up if mkt_side == 'UP' else fc_down
                if fc_s < max_orders and cap <= mkt_price <= MARKET_TARGET:
                    if mkt_side == 'UP':
                        fc_up  += 1; cost_up  += ORDER_SIZE; shares_up  += ORDER_SIZE / mkt_price
                    else:
                        fc_down+= 1; cost_down+= ORDER_SIZE; shares_down+= ORDER_SIZE / mkt_price
                    fills.append({'j': j, 'side': mkt_side, 'price': mkt_price, 'kind': 'market'})
                    # Si le limit actif était du même côté et qu'on est maintenant saturé → cancel
                    if active_ord_side == mkt_side and \
                       (fc_up >= max_orders if mkt_side == 'UP' else fc_down >= max_orders):
                        active_order = active_size = active_ord_side = None; pending_cancel = False
            else:
                still_pending.append((t_mkt, mkt_side, mkt_price))
        pending_market = still_pending

        # ── Cancel-fill du limit ──────────────────────────────────────────────
        if pending_cancel and active_order is not None:
            if active_ord_side:
                cb2 = ub if active_ord_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    fc_s = fc_up if active_ord_side == 'UP' else fc_down
                    if fc_s < max_orders:
                        fills.append({'j': j, 'side': active_ord_side,
                                      'price': float(active_order), 'kind': 'limit'})
                        if active_ord_side == 'UP':
                            fc_up  += 1; cost_up  += active_size; shares_up  += active_size / active_order
                        else:
                            fc_down+= 1; cost_down+= active_size; shares_down+= active_size / active_order
            active_order = active_size = active_ord_side = None; pending_cancel = False

        # ── Pending → actif ───────────────────────────────────────────────────
        if pending_place is not None:
            fc_pend = fc_up if pending_ord_side == 'UP' else fc_down
            if fc_pend < max_orders:
                active_order = pending_place; active_size = pending_size
                active_ord_side = pending_ord_side
            pending_place = pending_size = pending_ord_side = None

        # ── Fill limit actif ──────────────────────────────────────────────────
        if active_order is not None and active_ord_side:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fc_s = fc_up if active_ord_side == 'UP' else fc_down
                if fc_s < max_orders:
                    fills.append({'j': j, 'side': active_ord_side,
                                  'price': float(active_order), 'kind': 'limit'})
                    if active_ord_side == 'UP':
                        fc_up  += 1; cost_up  += active_size; shares_up  += active_size / active_order
                    else:
                        fc_down+= 1; cost_down+= active_size; shares_down+= active_size / active_order
                active_order = active_size = active_ord_side = None
                if fc_up >= max_orders and fc_down >= max_orders: break

        # ── Threshold + trigger ───────────────────────────────────────────────
        thresh = cfg['slope'] * remain + cfg['intercept']

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                # Deactivation : annule le limit, le market déjà soumis reste
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = pending_ord_side = None
                activated_side = None; continue

        if not activated_side:
            if   fc_up   < max_orders and (s - op) >= thresh: activated_side = 'UP'
            elif fc_down < max_orders and (op - s) >= thresh: activated_side = 'DOWN'
        if not activated_side: continue

        if (activated_side == 'UP' and fc_up >= max_orders) or \
           (activated_side == 'DOWN' and fc_down >= max_orders):
            activated_side = None; continue

        # ── Soumettre market + limit simultanément ────────────────────────────
        bid = ub if activated_side == 'UP' else db
        ask_arr  = up_ask if activated_side == 'UP' else down_ask

        # Market : fire à t+300ms
        t_mkt_fire = t_now + FILL_DELAY_S
        k = j + 1
        while k < len(spots) and ts_f[k] < t_mkt_fire:
            k += 1
        k = min(k, len(spots) - 1)
        ask_val = float(ask_arr[k])
        if cap <= ask_val <= MARKET_TARGET:
            pending_market.append((t_mkt_fire, activated_side, round(ask_val, 4)))

        # Limit : ordre au bid (chase si bid monte)
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = ORDER_SIZE; pending_ord_side = activated_side
            elif active_order is not None and proposed > active_order and active_ord_side == activated_side:
                pending_cancel = True
                pending_place = proposed; pending_size = ORDER_SIZE; pending_ord_side = activated_side
            elif pending_place is not None and proposed > pending_place and pending_ord_side == activated_side:
                pending_place = proposed; pending_size = ORDER_SIZE; pending_ord_side = activated_side

    # Fire les market orders restants (pas encore atteint leur tick)
    for (t_mkt, mkt_side, mkt_price) in pending_market:
        fc_s = fc_up if mkt_side == 'UP' else fc_down
        if fc_s < max_orders and cap <= mkt_price <= MARKET_TARGET:
            if mkt_side == 'UP':
                fc_up  += 1; cost_up  += ORDER_SIZE; shares_up  += ORDER_SIZE / mkt_price
            else:
                fc_down+= 1; cost_down+= ORDER_SIZE; shares_down+= ORDER_SIZE / mkt_price
            fills.append({'j': len(spots)-1, 'side': mkt_side, 'price': mkt_price, 'kind': 'market'})

    won, pnl, fills = make_result(cost_up, cost_down, shares_up, shares_down, fills, won_up)
    if won is None:
        return None, None, None, []

    rebate = sum(maker_rebate(ORDER_SIZE, f['price']) for f in fills if f.get('kind') == 'limit')
    fee    = sum(taker_fee(ORDER_SIZE,   f['price']) for f in fills if f.get('kind') == 'market')
    fee_net = fee - rebate   # positif = coût net, négatif = gain net
    return won, pnl, fee_net, fills


# ── Chargement des contrats (une fois) ───────────────────────────────────────

def load_all_contracts():
    csv    = [str(CSV_DIR / "BTC.csv")]
    result = []
    m5_ref = None
    for idx, (tf_floor, b1, b2, a1, a2) in enumerate(TIMEFRAMES):
        cts = load_contracts(csv, tf_floor, b1, b2, a1, a2)
        if tf_floor == '5min':
            precompute_vol(cts, VOL_LBS)
            m5_ref = cts
        else:
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
        attach_settlement_outcomes(cts, 'btc', tf=tf_floor)
        tf_name = ['m5', 'm15', 'h1'][idx]
        result.append((tf_name, cts))
    return result


# ── Run BT sur une liste de contrats ─────────────────────────────────────────

def run_bt(contracts_by_tf, sim_fn, cutoff_ts):
    cu.BASE_SIZE = ORDER_SIZE
    vol_key = 'vol_1h_net'
    trades  = []
    for tf_name, cts in contracts_by_tf:
        for c in cts:
            if c.get('open_ts', 0) < cutoff_ts:
                continue
            if c.get(vol_key, 0.0) < CFG.get('vol_thresh', 0):
                continue
            won, pnl, fee_net, fills = sim_fn(c, CFG)
            if won is None:
                continue
            avg_fill = float(np.mean([f['price'] for f in fills])) if fills else 0.0
            trades.append(dict(
                ts=c['ce'], tf=tf_name,
                pnl=pnl,
                pnl_net=pnl - fee_net,
                fee_net=fee_net,
                won=won, avg_fill=avg_fill, n_fills=len(fills),
            ))
    trades.sort(key=lambda x: x['ts'])
    return trades


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(trades, label, span_h):
    n        = len(trades)
    wins     = sum(1 for t in trades if t['won'])
    pnl      = sum(t['pnl'] for t in trades)
    pnl_net  = sum(t['pnl_net'] for t in trades)
    fee_tot  = sum(t['fee_net'] for t in trades)
    fills_p  = [t['avg_fill'] for t in trades if t.get('avg_fill')]
    avg_fill = float(np.mean(fills_p)) if fills_p else 0.
    avg_nf   = float(np.mean([t['n_fills'] for t in trades])) if trades else 0.
    cum      = np.cumsum([t['pnl_net'] for t in trades]) if trades else np.array([0.])
    peak     = np.maximum.accumulate(cum)
    maxdd    = float((peak - cum).max()) if len(cum) else 0.
    be_wr    = 0.0
    if avg_fill > 0:
        wpH = 100 / avg_fill - 100
        be_wr = 100 * 100 / (100 + wpH) if wpH > 0 else 100.
    return dict(
        label=label, n=n, wins=wins, losses=n-wins,
        wr=100*wins/n if n else 0,
        pnl=pnl, pnl_net=pnl_net, fee_tot=fee_tot,
        pnl_h=pnl_net/max(span_h, 0.01),
        avg_fill=avg_fill, avg_nf=avg_nf,
        maxdd=maxdd, be_wr=be_wr,
    )


# ── Graphique (style gen_custom_chart) ───────────────────────────────────────

def make_chart(trades_lim, trades_mkt, trades_hyb, cutoff_ts, label_period, out_path):
    if not trades_lim and not trades_mkt and not trades_hyb:
        print(f"  (pas de trades pour {label_period})")
        return

    all_t = trades_lim + trades_mkt + trades_hyb
    last_ts = max(
        (t['ts'].timestamp() for t in all_t),
        default=pd.Timestamp.now(tz='UTC').timestamp()
    )
    span_h = (last_ts - cutoff_ts) / 3600

    s_lim = compute_stats(trades_lim, 'LIMIT maker',         span_h)
    s_mkt = compute_stats(trades_mkt, 'MARKET taker',        span_h)
    s_hyb = compute_stats(trades_hyb, 'HYBRID (lim+mkt)',    span_h)

    def equity_hours(trades):
        if not trades: return [], []
        xs = [(t['ts'].timestamp() - cutoff_ts) / 3600 for t in trades]
        ys = list(np.cumsum([t['pnl_net'] for t in trades]))
        return xs, ys

    def hourly_pnl(trades, n_hours):
        arr = np.zeros(int(n_hours) + 2)
        for t in trades:
            h = int((t['ts'].timestamp() - cutoff_ts) / 3600)
            if 0 <= h < len(arr):
                arr[h] += t['pnl_net']
        return arr

    lx, ly = equity_hours(trades_lim)
    mx, my = equity_hours(trades_mkt)
    hx, hy = equity_hours(trades_hyb)
    n_h = int(span_h) + 2
    lh  = hourly_pnl(trades_lim, n_h)
    mh  = hourly_pnl(trades_mkt, n_h)
    hh  = hourly_pnl(trades_hyb, n_h)

    COLOR_LIM = '#2196F3'
    COLOR_MKT = '#FF9800'
    COLOR_HYB = '#4CAF50'

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50,
                            height_ratios=[3, 1.5, 1.5])
    xticks = np.arange(0, span_h + 2, max(2, int(span_h / 40)))

    # ── Equity cumulée ────────────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0])
    if mx:
        ax_eq.plot(mx, my, color=COLOR_MKT, linewidth=1.6, linestyle='--', zorder=2,
                   label=(f"MARKET taker  {my[-1]:+.0f}USD  WR={s_mkt['wr']:.1f}%"
                          f"  T={s_mkt['n']}  fee=-{s_mkt['fee_tot']:.0f}USD"
                          f"  BE={s_mkt['be_wr']:.1f}%"))
    if lx:
        ax_eq.plot(lx, ly, color=COLOR_LIM, linewidth=1.6, linestyle=':', zorder=3,
                   label=(f"LIMIT maker  {ly[-1]:+.0f}USD  WR={s_lim['wr']:.1f}%"
                          f"  T={s_lim['n']}  rebate=+{abs(s_lim['fee_tot']):.0f}USD"
                          f"  BE={s_lim['be_wr']:.1f}%"))
    if hx:
        ax_eq.plot(hx, hy, color=COLOR_HYB, linewidth=2.4, zorder=4,
                   label=(f"HYBRID (lim+mkt)  {hy[-1]:+.0f}USD  WR={s_hyb['wr']:.1f}%"
                          f"  T={s_hyb['n']}  BE={s_hyb['be_wr']:.1f}%"))
        ax_eq.scatter(hx, hy, color=COLOR_HYB, s=14, zorder=5, alpha=0.5)
    ax_eq.axhline(0, color='gray', linewidth=0.6, linestyle='--')
    ax_eq.set_title(
        f'LIMIT maker vs MARKET taker — {label_period}'
        '  (slope=7 int=0 cap=0.75 vol_net_1h>=60 settlement CSV only)',
        fontsize=10, fontweight='bold'
    )
    ax_eq.set_ylabel('PnL net cumulé (USD)', fontsize=9)
    ax_eq.legend(fontsize=9); ax_eq.grid(alpha=0.25)
    ax_eq.set_xticks(xticks)
    ax_eq.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
    ax_eq.set_xlim(-0.5, span_h + 0.5)

    # ── PnL horaire ───────────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[1])
    n_plot = min(len(lh), len(mh), len(hh))
    x_h = np.arange(n_plot)
    w = 0.27
    ax_h.bar(x_h - w, lh[:n_plot], width=w, color=COLOR_LIM, alpha=0.7, label='LIMIT/h')
    ax_h.bar(x_h,     mh[:n_plot], width=w, color=COLOR_MKT, alpha=0.7, label='MARKET/h')
    ax_h.bar(x_h + w, hh[:n_plot], width=w, color=COLOR_HYB, alpha=0.7, label='HYBRID/h')
    ax_h.axhline(0, color='gray', linewidth=0.5)
    ax_h.set_ylabel('PnL net/heure (USD)', fontsize=9)
    ax_h.legend(fontsize=8); ax_h.grid(alpha=0.2, axis='y')
    ax_h.set_xticks(xticks)
    ax_h.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
    ax_h.set_xlim(-0.5, n_h + 0.5)

    # ── Drawdown ──────────────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[2])
    for xs, ys, color, lbl in [(mx, my, COLOR_MKT, 'DD MARKET'), (lx, ly, COLOR_LIM, 'DD LIMIT'),
                                (hx, hy, COLOR_HYB, 'DD HYBRID')]:
        if ys:
            cum = np.array(ys); dd = np.maximum.accumulate(cum) - cum
            ax_dd.fill_between(xs, 0, -dd, color=color, alpha=0.4, label=lbl)
    ax_dd.axhline(0, color='gray', linewidth=0.5)
    ax_dd.set_ylabel('Drawdown (USD)', fontsize=9); ax_dd.legend(fontsize=8); ax_dd.grid(alpha=0.25)
    ax_dd.set_xticks(xticks)
    ax_dd.set_xticklabels([f'{int(x)}h' for x in xticks], fontsize=7, rotation=45)
    ax_dd.set_xlim(-0.5, span_h + 0.5)

    # Tableau résumé
    cols = ['', 'T', 'W', 'L', 'WR', 'BE-WR', 'avg fill', 'fills/T',
            'PnL brut', 'fee/rebate', 'PnL net', 'PnL/h', 'MaxDD']
    rows = []
    for s in [s_lim, s_mkt, s_hyb]:
        is_limit = 'maker' in s['label']
        fee_str  = (f"+{abs(s['fee_tot']):.1f}" if is_limit else f"-{s['fee_tot']:.1f}")
        rows.append([
            s['label'], str(s['n']), str(s['wins']), str(s['losses']),
            f"{s['wr']:.1f}%", f"{s['be_wr']:.1f}%",
            f"{s['avg_fill']:.4f}", f"{s['avg_nf']:.2f}",
            f"{s['pnl']:+.1f}", fee_str,
            f"{s['pnl_net']:+.1f}", f"{s['pnl_h']:+.2f}/h",
            f"{s['maxdd']:.0f}",
        ])

    fig.subplots_adjust(bottom=0.18)
    ax_tbl = fig.add_axes([0.02, 0.01, 0.96, 0.14])
    ax_tbl.axis('off')
    tbl = ax_tbl.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2.0)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#37474F')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(len(cols)):
        tbl[1, j].set_facecolor('#E3F2FD')
        tbl[2, j].set_facecolor('#FFF3E0')
        tbl[3, j].set_facecolor('#E8F5E9')

    fig.suptitle(
        f'LIMIT maker (rebate=cost×0.072×p×(1-p))  vs  MARKET taker (fee=cost×0.072×(1-p))'
        f'  —  {label_period}  —  100 USD/ordre',
        fontsize=11, fontweight='bold', y=0.99
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"  -> {out_path}", flush=True)

    # Console
    print(f"\n  {label_period} ({span_h:.1f}h) :")
    hdr = (f"  {'':20} {'T':>4} {'W':>4} {'L':>4} {'WR':>6} {'BE':>6} "
           f"{'fill':>6} {'f/T':>4} {'brut':>8} {'fee':>9} {'net':>8} {'net/h':>8} {'MaxDD':>7}")
    print(hdr)
    for s in [s_lim, s_mkt, s_hyb]:
        is_lim = 'maker' in s['label']
        ft = f"+{abs(s['fee_tot']):.1f}" if is_lim else f"-{s['fee_tot']:.1f}"
        print(
            f"  {s['label']:<20} {s['n']:>4} {s['wins']:>4} {s['losses']:>4} "
            f"{s['wr']:>5.1f}% {s['be_wr']:>5.1f}% "
            f"{s['avg_fill']:>6.4f} {s['avg_nf']:>4.2f} "
            f"{s['pnl']:>+8.1f} {ft:>9} {s['pnl_net']:>+8.1f} "
            f"{s['pnl_h']:>+7.2f}/h ${s['maxdd']:>6.0f}"
        )


# ── Main ─────────────────────────────────────────────────────────────────────
print("=== Chargement des contrats ===", flush=True)
contracts_by_tf = load_all_contracts()
total = sum(len(cts) for _, cts in contracts_by_tf)
print(f"  {total} contrats chargés (m5+m15+h1)", flush=True)

# Bornes temporelles
all_open_ts = [c.get('open_ts', 0) for _, cts in contracts_by_tf for c in cts]
ts_min = min(all_open_ts) if all_open_ts else 0
ts_max = max(all_open_ts) if all_open_ts else 0
CUTOFF_ALL_TS = ts_min

CUTOFF_4D_TS  = CUTOFF_4D.timestamp()

print(f"  Données : {pd.Timestamp(ts_min, unit='s', tz='UTC')} → "
      f"{pd.Timestamp(ts_max, unit='s', tz='UTC')}", flush=True)
print(f"  Last 4d depuis : {CUTOFF_4D.strftime('%Y-%m-%d %H:%M')} UTC", flush=True)

for period_label, cutoff_ts, out_name in [
    ("Last 4 days",  CUTOFF_4D_TS,  "limit_vs_market_last4d.png"),
    ("All CSV data", CUTOFF_ALL_TS, "limit_vs_market_all.png"),
]:
    print(f"\n=== {period_label} ===", flush=True)

    print("  Run LIMIT maker...", flush=True)
    tl = run_bt(contracts_by_tf, simulate_limit,   cutoff_ts)
    print(f"  {len(tl)} trades LIMIT", flush=True)

    print("  Run MARKET taker...", flush=True)
    tm = run_bt(contracts_by_tf, simulate_market,  cutoff_ts)
    print(f"  {len(tm)} trades MARKET", flush=True)

    print("  Run HYBRID (limit+market simultané)...", flush=True)
    th = run_bt(contracts_by_tf, simulate_hybrid,  cutoff_ts)
    print(f"  {len(th)} trades HYBRID", flush=True)

    make_chart(tl, tm, th, cutoff_ts, period_label, OUT_DIR / out_name)

print("\nDone.", flush=True)
