"""
chart_utils.py - Module partagé de génération de charts (BTC + ETH)
Génère equity curves M5+M15+H1 combinés pour les top configs d'un optimizer.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict
import bisect, csv as _csv

MAX_ORDERS   = 3
BASE_SIZE    = 50.0
CB_WINDOW_S  = 5400
# Union BTC [0.25..8] et ETH [1,2,4,8] — sinon vol_4h_* manque et les filtres ETH
# voient c.get(..., 0) < seuil -> tous les contrats exclus sur les charts.
VOL_LBS      = [0.25, 0.5, 1, 2, 4, 8]

TIMEFRAMES = [
    ('5min',  'm5_up_bid',  'm5_down_bid', 'm5_up_ask', 'm5_down_ask'),
    ('15min', 'm15_up_bid', 'm15_down_bid', 'm15_up_ask', 'm15_down_ask'),
    ('1h',    'h1_up_bid',  'h1_down_bid', 'h1_up_ask', 'h1_down_ask'),
]

# Grilles partagées (optimizers)
CAP_GRID = (0.55, 0.65, 0.75, 0.85, 0.90, 0.95)
INTERCEPT_GRID_BTC = (0.0, 15.0, 30.0)
VOL_SLOPE_WINDOWS = (60, 120, 300, 900)
# 3 profils : vol bas / neutre / vol élevé (clip sur ratio vol_inst / vol_ref)
VOL_SLOPE_G_TRIPLETS = ((0.35, 1.0), (0.55, 1.35), (0.75, 2.2))

COLORS = [
    "#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336",
    "#00BCD4","#FF5722","#8BC34A","#3F51B5","#FFC107",
    "#E91E63","#607D8B",
]


# ── Settlement outcomes (vrais résultats Gamma) ───────────────────────────────
_SETTLEMENT_CSV = Path(__file__).resolve().parent.parent / "Datas" / "csv" / "settlement.csv"

def load_settlement_outcomes(asset: str, tf: str | None = None,
                             settlement_csv: Path | None = None) -> dict:
    """
    Charge settlement.csv et retourne {open_ts (int): won_up (bool)}.
    won_up=True  -> UP a gagne selon Gamma.
    tf : '5min', '15min', '1h' — filtre obligatoire pour eviter les collisions
         (m5 et m15 peuvent avoir le meme open_ts aux heures rondes).
    Retourne {} si le fichier n'existe pas.
    """
    path = settlement_csv or _SETTLEMENT_CSV
    if not path.exists():
        return {}
    outcomes: dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            if row.get('asset', '').lower() != asset.lower():
                continue
            if tf is not None and row.get('tf', '') != tf:
                continue
            ou = row.get('outcome_up', '')
            if ou == '':
                continue
            try:
                open_ts = int(row['open_ts'])
                outcomes[open_ts] = float(ou) > 0.5
            except (ValueError, KeyError):
                continue
    return outcomes


def attach_settlement_outcomes(contracts: list, asset: str, tf: str = '5min',
                               settlement_csv: Path | None = None) -> int:
    """
    Attache `_settlement_won_up` (bool) a chaque contrat dont on connait le vrai outcome.
    simulate() l'utilise a la place de up_ask[-1] > down_ask[-1].
    Retourne le nombre de contrats mis a jour.
    """
    outcomes = load_settlement_outcomes(asset, tf=tf, settlement_csv=settlement_csv)
    if not outcomes:
        return 0
    n = 0
    for c in contracts:
        open_ts = int(c.get('open_ts', c['ce_ts'] - 300))
        if open_ts in outcomes:
            c['_settlement_won_up'] = outcomes[open_ts]
            n += 1
    return n


# ── Chargement ───────────────────────────────────────────────────────────────
def load_contracts(
    csv_paths,
    tf_floor,
    bid_up_col,
    bid_down_col,
    ask_up_col,
    ask_down_col,
    window_s=60,
):
    dfs = []
    for p in csv_paths:
        d = pd.read_csv(
            p,
            usecols=['timestamp', 'spot_price', bid_up_col, bid_down_col, ask_up_col, ask_down_col],
        )
        d['timestamp'] = pd.to_datetime(d['timestamp']).dt.tz_localize(None)
        dfs.append(d)
    df = pd.concat(dfs).dropna().sort_values('timestamp').drop_duplicates('timestamp')
    offset = pd.tseries.frequencies.to_offset(tf_floor)
    df['contract'] = df['timestamp'].dt.floor(tf_floor) + offset
    contracts = []
    for ce, grp in df.groupby('contract'):
        op_spot = grp['spot_price'].iloc[0]
        if window_s is None:
            ticks = grp
        else:
            ticks = grp[grp['timestamp'] >= ce - pd.Timedelta(seconds=int(window_s))]
        if len(ticks) < 2:
            continue
        open_time = ce - offset
        contracts.append({
            'ce':       ce,
            'ce_ts':    ce.timestamp(),
            'open_ts':  open_time.timestamp(),
            'op':       float(op_spot),
            'spot':     ticks['spot_price'].values.astype(np.float64),
            'ts':       ticks['timestamp'].values,
            'up_bid':   ticks[bid_up_col].values.astype(np.float64),
            'down_bid': ticks[bid_down_col].values.astype(np.float64),
            'up_ask':   ticks[ask_up_col].values.astype(np.float64),
            'down_ask': ticks[ask_down_col].values.astype(np.float64),
        })
    del df
    return contracts


# ── Vol instantanée (range spot sur fenêtre glissante, $) ─────────────────────
def compute_inst_vol_usd(spots, ts_arr, window_s):
    """Range max-min du spot sur les `window_s` dernières secondes (tick par tick)."""
    n = len(spots)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    ts_sec = (ts_arr - ts_arr[0]) / np.timedelta64(1, "s")
    dq = deque()
    ws = float(window_s)
    for j in range(n):
        t = float(ts_sec[j])
        while dq and dq[0][0] < t - ws:
            dq.popleft()
        dq.append((t, float(spots[j])))
        vals = [x[1] for x in dq]
        out[j] = max(vals) - min(vals) if vals else 0.0
    return out


def attach_vol_slope_data(contracts, vol_ref_key="vol_0.5h_range"):
    """
    Précalcule vol_inst_W (W en secondes) et vol_inst_ref (vol régime du contrat, $).
    ratio = vol_inst_W[j] / vol_inst_ref -> multiplicateur de pente (dans simulate).
    """
    for W in VOL_SLOPE_WINDOWS:
        for c in contracts:
            c[f"vol_inst_{W}"] = compute_inst_vol_usd(c["spot"], c["ts"], W)
    for c in contracts:
        ref = float(c.get(vol_ref_key, 0.0) or 0.0)
        c["vol_inst_ref"] = max(ref, 1e-6)


# ── Precompute vol (range / net / trend) sans look-ahead ─────────────────────
def precompute_vol(contracts, lb_hours_list, ref_contracts=None):
    """
    ref_contracts : si fourni (M5), la fenêtre glissante est construite depuis ces
    contrats (granularité 5min) et les résultats sont attachés aux contrats cibles
    par lookup temporel. Corrige le bug M15/H1 où la fenêtre était vide faute de
    contrats du même timeframe dans la période de lookback.
    """
    source = sorted(ref_contracts if ref_contracts is not None else contracts,
                    key=lambda c: c['ce_ts'])

    for lb_h in lb_hours_list:
        lb_s = lb_h * 3600
        src_ts_list = []
        vol_rng_list = []
        vol_net_list = []
        vol_trend_list = []

        window = deque()  # (open_ts, op)
        for c in source:
            t = c.get('open_ts', c['ce_ts'])
            while window and t - window[0][0] > lb_s:
                window.popleft()
            if window:
                spots_w = [x[1] for x in window]
                rng     = max(spots_w) - min(spots_w)
                net     = abs(c['op'] - window[0][1])
                trend   = net / rng if rng > 0 else 0.0
            else:
                rng = net = trend = 0.0
            src_ts_list.append(t)
            vol_rng_list.append(rng)
            vol_net_list.append(net)
            vol_trend_list.append(trend)
            window.append((t, c['op']))

        # Attacher aux contrats cibles par lookup (dernier src_ts <= open_ts du contrat)
        for c in contracts:
            idx = bisect.bisect_right(src_ts_list, c.get('open_ts', c['ce_ts'])) - 1
            op  = c.get('op', 1.0) or 1.0
            if idx >= 0:
                rng_v = vol_rng_list[idx]
                net_v = vol_net_list[idx]
                c[f'vol_{lb_h:g}h_range']     = rng_v
                c[f'vol_{lb_h:g}h_net']       = net_v
                c[f'vol_{lb_h:g}h_trend']     = vol_trend_list[idx]
                c[f'vol_{lb_h:g}h_pct_range'] = rng_v / op * 100.0
                c[f'vol_{lb_h:g}h_pct_net']   = net_v / op * 100.0
            else:
                c[f'vol_{lb_h:g}h_range']     = 0.0
                c[f'vol_{lb_h:g}h_net']       = 0.0
                c[f'vol_{lb_h:g}h_trend']     = 0.0
                c[f'vol_{lb_h:g}h_pct_range'] = 0.0
                c[f'vol_{lb_h:g}h_pct_net']   = 0.0

        # Pour M15/H1 (ref_contracts fourni) : attacher un lookup {ts_5min: vol_range}
        # afin de permettre le recalcul dynamique de _g_vrs dans simulate().
        # Pour M5 (ref_contracts is None) : non nécessaire (contrat = 1 bucket 5min).
        if ref_contracts is not None:
            vrs_lookup = {int(src_ts_list[i]): vol_rng_list[i]
                          for i in range(len(src_ts_list))}
            key = f'_vrs_vol_lookup_{lb_h:g}'
            for c in contracts:
                c[key] = vrs_lookup  # référence partagée


def attach_cnet_data(contracts, n_list=(2, 5, 10), ref_contracts=None):
    """Minimum M5 candle body sur les N dernières bougies M5 avant chaque contrat.
    ref_contracts : contrats M5 de référence (obligatoire pour M15/H1).
    Si None, utilise contracts lui-même (cas M5).
    """
    if not contracts:
        return
    source = sorted(ref_contracts if ref_contracts is not None else contracts,
                    key=lambda c: c['ce_ts'])
    # bodies[i] = |op[i] - op[i-1]| pour les contrats source (M5)
    src_ce  = [c['ce_ts'] for c in source]
    bodies  = [0.0] * len(source)
    for i in range(1, len(source)):
        bodies[i] = abs(source[i]['op'] - source[i - 1]['op'])

    for c in contracts:
        t = c.get('open_ts', c['ce_ts'])
        # dernier indice source dont ce_ts <= open_ts du contrat cible
        idx = bisect.bisect_right(src_ce, t) - 1
        for n in n_list:
            if idx < n:
                c[f'cnet_{n}c'] = 0.0
            else:
                c[f'cnet_{n}c'] = min(bodies[idx - n + 1: idx + 1])


# ── Index horaire sans trous ──────────────────────────────────────────────────
def build_hour_index(all_contracts_by_tf):
    all_hours = set()
    for contracts in all_contracts_by_tf:
        for c in contracts:
            all_hours.add(c['ce'].replace(minute=0, second=0, microsecond=0))
    valid_hours = sorted(all_hours)
    return {h: i for i, h in enumerate(valid_hours)}, len(valid_hours)


# ── Simulation ────────────────────────────────────────────────────────────────
def compute_trigger_threshold(cfg, remain, g=1.0):
    if cfg['curve'] == 'linear':
        linear_part = float(cfg["slope"]) * float(g) * float(remain)
        intercept = float(cfg.get("intercept", 0.0) or 0.0)
        if cfg.get("intercept_mode") == "floor":
            return max(linear_part, intercept)
        return linear_part + intercept
    return cfg["A_exp"] * (np.exp(remain / cfg["tau"]) - 1.0) * g


def simulate(c, cfg):
    curve = cfg['curve']; cap = cfg['eq_cap']
    ce_ns = np.datetime64(c['ce'])
    spots = c['spot']; ts_arr = c['ts']; op = c['op']
    activated_side = None
    # Per-side fill tracking (max_orders est PAR SIDE, pas total)
    fc_up = fc_down = 0
    shares_up = shares_down = 0.0
    cost_up = cost_down = 0.0
    active_order = active_size = active_ord_side = None
    pending_place = pending_size = pending_ord_side = None
    pending_cancel = False
    max_orders = int(cfg.get('max_orders', MAX_ORDERS))

    # VRS setup
    if cfg.get("vrs_enabled") or cfg.get("vrs_inv"):
        _vrs_lb      = cfg['vrs_lb']
        _vrs_base    = float(cfg['vrs_base'])
        _vrs_g_min   = float(cfg['vrs_g_min'])
        _vrs_g_max   = float(cfg['vrs_g_max'])
        _vrs_is_inv  = bool(cfg.get('vrs_inv', False))
        _vrs_lookup  = c.get(f'_vrs_vol_lookup_{_vrs_lb:g}')
        _vrs_bucket  = None
        _vol = c.get(f"vol_{_vrs_lb:g}h_range", 0.0)
        if _vrs_is_inv:
            _ratio = _vol / _vrs_base if _vol > 1e-6 else _vrs_g_min
        else:
            _ratio = _vrs_base / _vol if _vol > 1e-6 else _vrs_g_max
        _g_vrs = float(np.clip(_ratio, _vrs_g_min, _vrs_g_max))
    else:
        _vrs_lookup = None
        _g_vrs = None

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders): break
        ub = round(c['up_bid'][j], 2); db = round(c['down_bid'][j], 2)

        if _vrs_lookup is not None:
            _ts_sec = int(ts_arr[j].astype('datetime64[s]').astype(np.int64))
            _bucket = (_ts_sec // 300) * 300
            if _bucket != _vrs_bucket:
                _vrs_bucket = _bucket
                _dv = _vrs_lookup.get(_bucket)
                if _dv is not None:
                    _r = (_dv / _vrs_base if _vrs_is_inv else _vrs_base / _dv) if _dv > 1e-6 else (
                        _vrs_g_min if _vrs_is_inv else _vrs_g_max)
                    _g_vrs = float(np.clip(_r, _vrs_g_min, _vrs_g_max))

        if pending_cancel and active_order is not None:
            if active_ord_side:
                cb2 = ub if active_ord_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    if active_ord_side == 'UP': fc_up += 1; cost_up += active_size; shares_up += active_size / active_order
                    else:                        fc_down += 1; cost_down += active_size; shares_down += active_size / active_order
                    if fc_up >= max_orders and fc_down >= max_orders: active_order = active_ord_side = None; break
            active_order = active_size = active_ord_side = None; pending_cancel = False

        if pending_place is not None:
            fc_pend = fc_up if pending_ord_side == 'UP' else fc_down
            if fc_pend < max_orders:
                active_order = pending_place; active_size = pending_size; active_ord_side = pending_ord_side
            pending_place = pending_size = pending_ord_side = None

        if active_order is not None and active_ord_side:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                if active_ord_side == 'UP': fc_up += 1; cost_up += active_size; shares_up += active_size / active_order
                else:                        fc_down += 1; cost_down += active_size; shares_down += active_size / active_order
                active_order = active_size = active_ord_side = None
                if fc_up >= max_orders and fc_down >= max_orders: break

        if _g_vrs is not None:
            g = _g_vrs
        elif cfg.get("vol_slope_enabled"):
            W = int(cfg.get("vol_slope_window_s", 120))
            arr = c.get(f"vol_inst_{W}")
            if arr is None or j >= len(arr):
                g = 1.0
            else:
                ratio = float(arr[j]) / float(c.get("vol_inst_ref", 1.0))
                g = float(np.clip(ratio, float(cfg["vol_slope_g_min"]), float(cfg["vol_slope_g_max"])))
        else:
            g = 1.0

        thresh = compute_trigger_threshold(cfg, remain, g)

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = pending_ord_side = None; activated_side = None; continue

        min_delta = cfg.get('min_delta_usd', 0.0)
        if not activated_side:
            if   fc_up   < max_orders and s - op >= thresh and s - op >= min_delta: activated_side = 'UP'
            elif fc_down < max_orders and op - s >= thresh and op - s >= min_delta: activated_side = 'DOWN'
        if not activated_side: continue

        # Si ce côté est déjà maxé, on ignore
        if (activated_side == 'UP' and fc_up >= max_orders) or \
           (activated_side == 'DOWN' and fc_down >= max_orders):
            activated_side = None; continue

        if cfg.get('use_ask'):
            ua = round(c['up_ask'][j], 2); da = round(c['down_ask'][j], 2)
            ask = ua if activated_side == 'UP' else da
            if ask >= cap:
                price = min(ask, 0.99)
                if activated_side == 'UP': fc_up += 1; cost_up += BASE_SIZE; shares_up += BASE_SIZE / price
                else:                       fc_down += 1; cost_down += BASE_SIZE; shares_down += BASE_SIZE / price
                active_order = active_size = active_ord_side = None
                pending_place = pending_size = pending_ord_side = None
                pending_cancel = False
                activated_side = None
                if fc_up >= max_orders and fc_down >= max_orders: break
        else:
            bid = ub if activated_side == 'UP' else db
            if bid >= cap:
                proposed = min(bid, 0.99)
                if active_order is None and pending_place is None:
                    pending_place = proposed; pending_size = BASE_SIZE; pending_ord_side = activated_side
                elif active_order is not None and proposed > active_order and active_ord_side == activated_side:
                    pending_cancel = True; pending_place = proposed; pending_size = BASE_SIZE; pending_ord_side = activated_side
                elif pending_place is not None and proposed > pending_place and pending_ord_side == activated_side:
                    pending_place = proposed; pending_size = BASE_SIZE; pending_ord_side = activated_side

    if '_settlement_won_up' in c:
        won_up = c['_settlement_won_up']
    elif 'up_ask' in c and len(c['up_ask']):
        won_up = float(c['up_ask'][-1]) > float(c['down_ask'][-1])
    else:
        won_up = c['up_bid'][-1] > c['down_bid'][-1]

    # Expiry fill : ordre pending sur le côté PERDANT uniquement
    exp_side = pending_ord_side if pending_place is not None else active_ord_side
    exp_ord  = pending_place    if pending_place is not None else active_order
    exp_sz   = pending_size     if pending_place is not None else active_size
    exp_fc   = (fc_up if exp_side == 'UP' else fc_down) if exp_side else max_orders
    if exp_fc < max_orders and exp_side and exp_ord is not None and exp_sz is not None:
        if (exp_side == 'UP') != won_up:  # côté perdant uniquement
            if exp_side == 'UP': fc_up += 1; cost_up += exp_sz; shares_up += exp_sz / exp_ord
            else:                 fc_down += 1; cost_down += exp_sz; shares_down += exp_sz / exp_ord

    total_cost = cost_up + cost_down
    if total_cost == 0:
        return None, None
    payout = (shares_up if won_up else 0.0) + (shares_down if not won_up else 0.0)
    pnl = payout - total_cost
    won = pnl > 0
    return won, pnl


def simulate_trade_log(c, cfg):
    """
    Même logique que simulate ; retourne en plus la liste des fills avec max bid après fill.
    """
    curve = cfg['curve']; cap = cfg['eq_cap']
    ce_ns = np.datetime64(c['ce'])
    spots = c['spot']; ts_arr = c['ts']; op = c['op']
    up_bid = c['up_bid']; down_bid = c['down_bid']
    activated_side = None
    fc_up = fc_down = 0
    shares_up = shares_down = 0.0
    cost_up = cost_down = 0.0
    active_order = active_size = active_ord_side = None
    pending_place = pending_size = pending_ord_side = None
    pending_cancel = False
    fills = []
    max_orders = int(cfg.get('max_orders', MAX_ORDERS))

    if cfg.get("vrs_enabled") or cfg.get("vrs_inv"):
        _vrs_lb      = cfg['vrs_lb']
        _vrs_base    = float(cfg['vrs_base'])
        _vrs_g_min   = float(cfg['vrs_g_min'])
        _vrs_g_max   = float(cfg['vrs_g_max'])
        _vrs_is_inv  = bool(cfg.get('vrs_inv', False))
        _vrs_lookup  = c.get(f'_vrs_vol_lookup_{_vrs_lb:g}')
        _vrs_bucket  = None
        _vol = c.get(f"vol_{_vrs_lb:g}h_range", 0.0)
        if _vrs_is_inv:
            _ratio = _vol / _vrs_base if _vol > 1e-6 else _vrs_g_min
        else:
            _ratio = _vrs_base / _vol if _vol > 1e-6 else _vrs_g_max
        _g_vrs = float(np.clip(_ratio, _vrs_g_min, _vrs_g_max))
    else:
        _vrs_lookup = None
        _g_vrs = None

    for j in range(len(spots)):
        s = spots[j]
        remain = (ce_ns - ts_arr[j]) / np.timedelta64(1, 's')
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders): break
        ub = round(up_bid[j], 2); db = round(down_bid[j], 2)

        if _vrs_lookup is not None:
            _ts_sec = int(ts_arr[j].astype('datetime64[s]').astype(np.int64))
            _bucket = (_ts_sec // 300) * 300
            if _bucket != _vrs_bucket:
                _vrs_bucket = _bucket
                _dv = _vrs_lookup.get(_bucket)
                if _dv is not None:
                    _r = (_dv / _vrs_base if _vrs_is_inv else _vrs_base / _dv) if _dv > 1e-6 else (
                        _vrs_g_min if _vrs_is_inv else _vrs_g_max)
                    _g_vrs = float(np.clip(_r, _vrs_g_min, _vrs_g_max))

        if pending_cancel and active_order is not None:
            if active_ord_side:
                cb2 = ub if active_ord_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    fills.append({'j': j, 'side': active_ord_side, 'price': float(active_order)})
                    if active_ord_side == 'UP': fc_up += 1; cost_up += active_size; shares_up += active_size / active_order
                    else:                        fc_down += 1; cost_down += active_size; shares_down += active_size / active_order
                    if fc_up >= max_orders and fc_down >= max_orders: active_order = active_ord_side = None; break
            active_order = active_size = active_ord_side = None; pending_cancel = False

        if pending_place is not None:
            fc_pend = fc_up if pending_ord_side == 'UP' else fc_down
            if fc_pend < max_orders:
                active_order = pending_place; active_size = pending_size; active_ord_side = pending_ord_side
            pending_place = pending_size = pending_ord_side = None

        if active_order is not None and active_ord_side:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                fills.append({'j': j, 'side': active_ord_side, 'price': float(active_order)})
                if active_ord_side == 'UP': fc_up += 1; cost_up += active_size; shares_up += active_size / active_order
                else:                        fc_down += 1; cost_down += active_size; shares_down += active_size / active_order
                active_order = active_size = active_ord_side = None
                if fc_up >= max_orders and fc_down >= max_orders: break

        if _g_vrs is not None:
            g = _g_vrs
        elif cfg.get("vol_slope_enabled"):
            W = int(cfg.get("vol_slope_window_s", 120))
            arr = c.get(f"vol_inst_{W}")
            if arr is None or j >= len(arr):
                g = 1.0
            else:
                ratio = float(arr[j]) / float(c.get("vol_inst_ref", 1.0))
                g = float(np.clip(ratio, float(cfg["vol_slope_g_min"]), float(cfg["vol_slope_g_max"])))
        else:
            g = 1.0

        thresh = compute_trigger_threshold(cfg, remain, g)

        if activated_side:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None: pending_cancel = True
                pending_place = pending_size = pending_ord_side = None; activated_side = None; continue

        min_delta = cfg.get('min_delta_usd', 0.0)
        if not activated_side:
            if   fc_up   < max_orders and s - op >= thresh and s - op >= min_delta: activated_side = 'UP'
            elif fc_down < max_orders and op - s >= thresh and op - s >= min_delta: activated_side = 'DOWN'
        if not activated_side: continue

        if (activated_side == 'UP' and fc_up >= max_orders) or \
           (activated_side == 'DOWN' and fc_down >= max_orders):
            activated_side = None; continue

        bid = ub if activated_side == 'UP' else db
        if bid >= cap:
            proposed = min(bid, 0.99)
            if active_order is None and pending_place is None:
                pending_place = proposed; pending_size = BASE_SIZE; pending_ord_side = activated_side
            elif active_order is not None and proposed > active_order and active_ord_side == activated_side:
                pending_cancel = True; pending_place = proposed; pending_size = BASE_SIZE; pending_ord_side = activated_side
            elif pending_place is not None and proposed > pending_place and pending_ord_side == activated_side:
                pending_place = proposed; pending_size = BASE_SIZE; pending_ord_side = activated_side

    if '_settlement_won_up' in c:
        won_up = c['_settlement_won_up']
    elif 'up_ask' in c and len(c['up_ask']):
        won_up = float(c['up_ask'][-1]) > float(c['down_ask'][-1])
    else:
        won_up = up_bid[-1] > down_bid[-1]

    # Expiry fill : côté perdant uniquement
    exp_side = pending_ord_side if pending_place is not None else active_ord_side
    exp_ord  = pending_place    if pending_place is not None else active_order
    exp_sz   = pending_size     if pending_place is not None else active_size
    exp_fc   = (fc_up if exp_side == 'UP' else fc_down) if exp_side else max_orders
    if exp_fc < max_orders and exp_side and exp_ord is not None and exp_sz is not None:
        if (exp_side == 'UP') != won_up:
            if exp_side == 'UP': fc_up += 1; cost_up += exp_sz; shares_up += exp_sz / exp_ord
            else:                 fc_down += 1; cost_down += exp_sz; shares_down += exp_sz / exp_ord
            fills.append(dict(side=exp_side, price=exp_ord, size=exp_sz,
                               j=len(spots) - 1, expiry_fill=True))

    total_cost = cost_up + cost_down
    if total_cost <= 0:
        return None, None, []

    payout = (shares_up if won_up else 0.0) + (shares_down if not won_up else 0.0)
    pnl = payout - total_cost
    won = pnl > 0

    for f in fills:
        side = f['side']
        jj = f['j']
        bid_arr = up_bid if side == 'UP' else down_bid
        mxb = float(np.max(bid_arr[jj:])) if jj < len(bid_arr) else f['price']
        f['max_bid_after'] = mxb
        f['mfe'] = mxb - f['price']
        f['contract_won'] = won
        f['contract_pnl'] = pnl

    return won, pnl, fills


def simulate_market_trade_with_exit(
    c,
    side,
    entry_j,
    size_usd,
    take_profit_c=None,
    stop_loss_c=None,
    max_hold_s=None,
):
    """
    Achat market au meilleur ask du côté `side` à l'index `entry_j`.
    Sortie optionnelle avant settlement au meilleur bid du même côté.

    take_profit_c / stop_loss_c sont exprimés en cents de prix de contrat.
    max_hold_s : durée max après l'entrée ; si atteinte, sortie au bid courant.
    """
    side = side.upper()
    if side not in ("UP", "DOWN"):
        raise ValueError(f"Unsupported side: {side}")

    asks = c['up_ask'] if side == 'UP' else c['down_ask']
    bids = c['up_bid'] if side == 'UP' else c['down_bid']
    ts_arr = c['ts']
    if entry_j < 0 or entry_j >= len(ts_arr):
        return None

    entry_price = float(asks[entry_j])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    size_usd = float(size_usd)
    if size_usd <= 0:
        return None

    shares = size_usd / entry_price
    entry_ts = ts_arr[entry_j]
    tp_d = None if take_profit_c is None else float(take_profit_c) / 100.0
    sl_d = None if stop_loss_c is None else float(stop_loss_c) / 100.0
    max_hold_s = None if max_hold_s in (None, 0) else float(max_hold_s)

    exit_j = None
    exit_price = None
    exit_reason = "settlement"

    for j in range(entry_j + 1, len(ts_arr)):
        bid = float(bids[j])
        if not np.isfinite(bid) or bid <= 0:
            continue
        elapsed_s = float((ts_arr[j] - entry_ts) / np.timedelta64(1, 's'))

        if tp_d is not None and bid >= entry_price + tp_d:
            exit_j = j
            exit_price = bid
            exit_reason = "take_profit"
            break
        if sl_d is not None and bid <= max(0.01, entry_price - sl_d):
            exit_j = j
            exit_price = bid
            exit_reason = "stop_loss"
            break
        if max_hold_s is not None and elapsed_s >= max_hold_s:
            exit_j = j
            exit_price = bid
            exit_reason = "time_exit"
            break

    if '_settlement_won_up' in c:
        won_up = c['_settlement_won_up']
    elif 'up_ask' in c and len(c['up_ask']):
        won_up = float(c['up_ask'][-1]) > float(c['down_ask'][-1])
    else:
        won_up = c['up_bid'][-1] > c['down_bid'][-1]

    if exit_j is not None and exit_price is not None:
        proceeds = shares * exit_price
        pnl = proceeds - size_usd
        exit_ts = ts_arr[exit_j]
    else:
        payout = shares if ((side == 'UP') == won_up) else 0.0
        pnl = payout - size_usd
        exit_ts = ts_arr[-1]

    return {
        'side': side,
        'entry_j': int(entry_j),
        'entry_ts': entry_ts,
        'entry_price': entry_price,
        'size_usd': size_usd,
        'shares': shares,
        'exit_j': None if exit_j is None else int(exit_j),
        'exit_ts': exit_ts,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'won_up': bool(won_up),
        'won_side': bool(((side == 'UP') == won_up)),
        'pnl': float(pnl),
        'return_pct': float(pnl / size_usd * 100.0),
    }


# ── Run un config sur un timeframe ───────────────────────────────────────────
def run_config_tf(contracts, cfg, hour_index):
    vol_thresh    = cfg.get('vol_thresh')
    vol_type      = cfg.get('vol_type', 'range')
    vol_key       = (f"vol_{cfg['vol_lb_h']:g}h_{vol_type}"
                     if vol_thresh is not None and cfg.get("vol_lb_h") is not None else None)
    cnet_n        = cfg.get('cnet_n')
    cnet_thresh   = float(cfg.get('cnet_thresh') or 0.0)
    max_losses_cb = cfg.get('max_losses_cb')
    loss_times    = deque()
    pnl_by_hour   = defaultdict(float)
    wins = losses = 0

    for c in contracts:
        if vol_key is not None and c.get(vol_key, 0.0) < vol_thresh:
            continue
        if cnet_n is not None and c.get(f'cnet_{cnet_n}c', 0.0) < cnet_thresh:
            continue
        if max_losses_cb is not None:
            ce_ts = c['ce_ts']
            while loss_times and ce_ts - loss_times[0] > CB_WINDOW_S:
                loss_times.popleft()
            if len(loss_times) >= max_losses_cb:
                continue

        won, pnl = simulate(c, cfg)
        if won is None: continue

        h = c['ce'].replace(minute=0, second=0, microsecond=0)
        if h not in hour_index: continue
        pnl_by_hour[hour_index[h]] += pnl
        if won: wins += 1
        else:
            losses += 1
            if max_losses_cb is not None: loss_times.append(c['ce_ts'])

    return pnl_by_hour, wins, losses


def accumulate_entry_analytics(contracts, cfg, hour_index):
    """
    Même filtres que run_config_tf ; agrège PnL / winrate par prix du premier fill.
    """
    vol_thresh    = cfg.get('vol_thresh')
    vol_type      = cfg.get('vol_type', 'range')
    vol_key       = (f"vol_{cfg['vol_lb_h']:g}h_{vol_type}"
                     if vol_thresh is not None and cfg.get("vol_lb_h") is not None else None)
    cnet_n        = cfg.get('cnet_n')
    cnet_thresh   = float(cfg.get('cnet_thresh') or 0.0)
    max_losses_cb = cfg.get('max_losses_cb')
    loss_times    = deque()

    bucket_pnl  = defaultdict(float)
    bucket_n    = defaultdict(int)
    bucket_wins = defaultdict(int)
    loss_fills  = []

    for c in contracts:
        if vol_key is not None and c.get(vol_key, 0.0) < vol_thresh:
            continue
        if cnet_n is not None and c.get(f'cnet_{cnet_n}c', 0.0) < cnet_thresh:
            continue
        if max_losses_cb is not None:
            ce_ts = c['ce_ts']
            while loss_times and ce_ts - loss_times[0] > CB_WINDOW_S:
                loss_times.popleft()
            if len(loss_times) >= max_losses_cb:
                continue

        h = c['ce'].replace(minute=0, second=0, microsecond=0)
        if h not in hour_index:
            continue

        won, pnl, fills = simulate_trade_log(c, cfg)
        if won is None or not fills:
            continue

        first_price = round(fills[0]['price'], 2)
        bucket_pnl[first_price] += pnl
        bucket_n[first_price] += 1
        if won:
            bucket_wins[first_price] += 1
        else:
            if max_losses_cb is not None:
                loss_times.append(c['ce_ts'])
            for f in fills:
                loss_fills.append(f)

    return bucket_pnl, bucket_n, bucket_wins, loss_fills


def chart_entry_and_loss_analytics(contracts_by_tf, cfg, hour_index, cfg_dir, cfg_label):
    """Graphiques distribution entrée + analyse pertes (MFE bid après fill)."""
    cfg_dir = Path(cfg_dir)
    all_pnl = defaultdict(float)
    all_n = defaultdict(int)
    all_w = defaultdict(int)
    all_loss = []
    for contracts in contracts_by_tf:
        bp, bn, bw, lf = accumulate_entry_analytics(contracts, cfg, hour_index)
        for k, v in bp.items():
            all_pnl[k] += v
        for k, v in bn.items():
            all_n[k] += v
        for k, v in bw.items():
            all_w[k] += v
        all_loss.extend(lf)

    keys = sorted(all_n.keys())
    if not keys:
        return

    pnls = [all_pnl[k] for k in keys]
    wrs  = [100.0 * all_w[k] / all_n[k] if all_n[k] else 0.0 for k in keys]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    x = np.arange(len(keys))
    axes[0].bar(x, pnls, color="#2196F3", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{k:.2f}" for k in keys], rotation=45, ha='right')
    axes[0].set_ylabel("PnL total ($)")
    axes[0].set_title(
        _esc(f"{cfg_label[:80]}\nPnL par prix premier fill (attribution contrat)"),
        fontsize=10,
    )
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, wrs, color="#4CAF50", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{k:.2f}" for k in keys], rotation=45, ha='right')
    axes[1].set_ylabel("Winrate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(cfg_dir / "entry_pnl_winrate.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    if not all_loss:
        return

    mxb = np.array([f['max_bid_after'] for f in all_loss], dtype=np.float64)
    improved = np.sum(mxb > np.array([f['price'] for f in all_loss], dtype=np.float64))
    n_loss = len(all_loss)
    pct_improved = 100.0 * improved / n_loss if n_loss else 0.0
    mean_mxb = float(np.mean(mxb))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(mxb, bins=min(40, max(10, n_loss // 5)), color="#f44336", alpha=0.75, edgecolor="white")
    ax.axvline(mean_mxb, color="navy", linewidth=1.5, label=f"Moy. max bid après fill={mean_mxb:.3f}")
    ax.set_title(
        _esc(
            f"Pertes : max bid après fill (n={n_loss} fills) — "
            f"{pct_improved:.1f}% avec max bid > prix fill"
        ),
        fontsize=10,
    )
    ax.set_xlabel("Max bid (position) après fill jusqu'à résolution")
    ax.set_ylabel("Nombre de fills")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(cfg_dir / "loss_mfe_after_fill.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# ── Série cumulative ──────────────────────────────────────────────────────────
def build_cumulative(pnl_by_hour, n_hours):
    arr = np.zeros(n_hours)
    for h_idx, pnl in pnl_by_hour.items():
        if 0 <= h_idx < n_hours:
            arr[h_idx] += pnl
    return np.cumsum(arr), arr


# ── Labels axe X ─────────────────────────────────────────────────────────────
def make_xlabels(hour_index, step=24):
    inv = {v: k for k, v in hour_index.items()}
    n   = max(hour_index.values()) + 1
    positions = list(range(0, n, step))
    labels    = [inv[p].strftime('%m/%d') if p in inv else '' for p in positions]
    return positions, labels


def _esc(s):
    """Echappe les $ pour matplotlib (evite le mode math LaTeX)."""
    return str(s).replace('$', r'\$')


def _rf_str_hourly_equity(cum):
    """
    RF depuis la courbe cumulée horaire (charts M5+M15+H1).
    Si maxDD=0 et PnL final=0 -> 'n/a' (pas inf : ce n'est pas un bon RF).
    Si maxDD=0 et PnL>0 -> 'inf' (equity strictement croissante sur l'index horaire).
    """
    if cum is None or len(cum) == 0:
        return "n/a"
    dd_max = float((np.maximum.accumulate(cum) - cum).max())
    pnl = float(cum[-1])
    if dd_max > 1e-9:
        return f"{pnl / dd_max:.1f}x"
    if pnl > 1e-9:
        return "inf"
    return "n/a"


# ── Chart individuel ──────────────────────────────────────────────────────────
TF_COLORS = {'M5': '#f44336', 'M15': '#FF9800', 'H1': '#4CAF50'}

def chart_equity(cum, hourly, cfg_label, n_hours, hour_index, wins, losses, out_path,
                 cum_by_tf=None):
    """
    cum_by_tf : dict optionnel {'M5': array, 'M15': array, 'H1': array}
                Si fourni, trace les courbes par TF en plus du total.
    """
    peak = np.maximum.accumulate(cum)
    dd   = peak - cum
    rf_s = _rf_str_hourly_equity(cum)
    wr   = wins / (wins + losses) * 100 if (wins + losses) else 0
    days = n_hours / 24
    x    = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1.2, 1]})
    fig.subplots_adjust(hspace=0.45)

    # Courbes par TF (derrière le total)
    if cum_by_tf:
        for tf_name, tf_cum in cum_by_tf.items():
            color = TF_COLORS.get(tf_name, '#888888')
            axes[0].plot(x, tf_cum, color=color, linewidth=0.9,
                         linestyle='--', alpha=0.7, label=tf_name)

    # Courbe totale (au-dessus)
    axes[0].plot(x, cum, color="#2196F3", linewidth=1.8, label='All TF')
    axes[0].fill_between(x, cum, 0, where=cum >= 0, color="#2196F3", alpha=0.08)
    axes[0].fill_between(x, cum, 0, where=cum <  0, color="#f44336", alpha=0.12)
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title(
        _esc(f"{cfg_label}  (M5+M15+H1)  —  {wins+losses} trades ({(wins+losses)/days:.1f}/j)  "
             f"WR={wr:.1f}%  ${cum[-1]/days:.0f}/j  RF={rf_s}"),
        fontsize=10)
    axes[0].set_ylabel("PnL cumule ($)"); axes[0].grid(alpha=0.3)
    axes[0].set_xticks(xticks); axes[0].set_xticklabels(xlabels, fontsize=8, rotation=30)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].text(0.99, 0.03,
                 _esc(f"PnL total: ${cum[-1]:+.0f}\nMaxDD: ${dd.max():.0f}\nRF: {rf_s}"),
                 transform=axes[0].transAxes, ha='right', va='bottom', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.9, ec='gray'))

    bar_colors = ["#4CAF50" if v >= 0 else "#f44336" for v in hourly]
    axes[1].bar(x, hourly, color=bar_colors, width=1.0, alpha=0.85)
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
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close("all")


# ── Overlay ───────────────────────────────────────────────────────────────────
def chart_overlay(all_series, n_hours, hour_index, out_path,
                  title="Equity Curves", baseline_names=None):
    x = np.arange(n_hours)
    xticks, xlabels = make_xlabels(hour_index, step=24)
    days = n_hours / 24
    baseline_names = baseline_names or set()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.4)

    for i, (name, label, cum, wins, losses) in enumerate(all_series):
        dd_arr = np.maximum.accumulate(cum) - cum
        rf_s = _rf_str_hourly_equity(cum)
        wr  = wins / (wins + losses) * 100 if (wins+losses) else 0
        col = COLORS[i % len(COLORS)]
        is_baseline = name in baseline_names
        ls  = '--' if is_baseline else '-'
        lw  = 1.2 if is_baseline else 1.5
        n_trades = wins + losses
        lbl = _esc(f"{label}  T={n_trades}({n_trades/days:.1f}/j)  WR={wr:.0f}%  ${cum[-1]/days:.0f}/j  RF={rf_s}")
        ax1.plot(x, cum,    color=col, linewidth=lw, linestyle=ls, label=lbl,
                 alpha=0.85 if is_baseline else 1.0)
        ax2.plot(x, dd_arr, color=col, linewidth=lw, linestyle=ls,
                 alpha=0.7  if is_baseline else 0.8)

    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_title(title, fontsize=13)
    ax1.set_ylabel("PnL cumule ($)")
    ax1.legend(loc="upper left", fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels, fontsize=8, rotation=30)

    ax2.set_ylabel("Drawdown ($)")
    ax2.set_xlabel("Heures tradees (gaps exclus)")
    ax2.invert_yaxis(); ax2.grid(alpha=0.3)
    ax2.set_xticks(xticks); ax2.set_xticklabels(xlabels, fontsize=8, rotation=30)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close("all")
    print(f"  Overlay -> {out_path}")


# ── Helpers label / name ──────────────────────────────────────────────────────
def _gfmt(v):
    """Format a float with enough sig figs; filesystem-safe (no dot/minus)."""
    s = f"{v:.5g}"
    return s.replace('.', 'p').replace('-', 'm').replace('+', '')


def make_config_name(r):
    if r['curve'] == 'expo':
        s = f"expo_A{r['A_exp']:.0f}t{r['tau']:.0f}"
    else:
        if r.get("intercept_mode") == "floor":
            s = f"lin_sl{_gfmt(r['slope'])}_flo{_gfmt(r['intercept'])}"
        else:
            s = f"lin_sl{_gfmt(r['slope'])}_int{_gfmt(r['intercept'])}"
    s += f"_c{int(r['eq_cap']*100)}"
    if r.get('vol_thresh') is not None:
        vt  = r.get('vol_type', 'range')
        thr = r['vol_thresh']
        _VT_SHORT = {'range': 'rng', 'net': 'net', 'trend': 'tre', 'pct_range': 'pctr', 'pct_net': 'pctn'}
        vt_s = _VT_SHORT.get(vt, vt[:3])
        t_s  = f"{thr:.2f}".replace('.', '') if vt in ('trend', 'pct_range', 'pct_net') else _gfmt(thr)
        s += f"_{vt_s}{r['vol_lb_h']:g}h{t_s}"
    if r.get("vrs_inv"):
        s += f"_vrsinv{r['vrs_lb']:g}h_b{_gfmt(r['vrs_base'])}_{r['vrs_g_min']:.2f}-{r['vrs_g_max']:.2f}"
    elif r.get("vrs_enabled"):
        s += f"_vrs{r['vrs_lb']:g}h_b{_gfmt(r['vrs_base'])}_{r['vrs_g_min']:.2f}-{r['vrs_g_max']:.2f}"
    elif r.get("vol_slope_enabled"):
        s += f"_vs{int(r['vol_slope_window_s'])}_{r['vol_slope_g_min']:.2f}-{r['vol_slope_g_max']:.2f}"
    if r.get("cnet_n") is not None:
        s += f"_cnet{int(r['cnet_n'])}c_{_gfmt(float(r['cnet_thresh']))}"
    if r.get('max_losses_cb'):
        s += f"_cb{r['max_losses_cb']}"
    return s


def make_config_label(r):
    if r['curve'] == 'expo':
        curve_s = f"expo A={r['A_exp']:.0f} t={r['tau']:.0f}"
    else:
        if r.get("intercept_mode") == "floor":
            curve_s = f"sl={r['slope']:.5g} floor={r['intercept']:.4g}"
        else:
            curve_s = f"sl={r['slope']:.5g} int={r['intercept']:.4g}"
    cap_s = f"cap={r['eq_cap']:.2f}"
    vol_s = ""
    if r.get('vol_thresh') is not None:
        vt  = r.get('vol_type', 'range')
        thr = r['vol_thresh']
        if vt == 'trend':
            t_s = f"{thr:.2f}"
        elif vt in ('pct_range', 'pct_net'):
            t_s = f"{thr:.2f}%"
        else:
            t_s = f"${thr:.5g}"
        vol_s = f" {vt}{r['vol_lb_h']:g}h>{t_s}"
    elif r.get("vrs_inv"):
        vol_s = f" vrs_inv{r['vrs_lb']:g}h base={r['vrs_base']:.5g} g={r['vrs_g_min']:.2f}-{r['vrs_g_max']:.2f}"
    elif r.get("vrs_enabled"):
        vol_s = f" vrs{r['vrs_lb']:g}h base={r['vrs_base']:.5g} g={r['vrs_g_min']:.2f}-{r['vrs_g_max']:.2f}"
    elif r.get("cnet_n") is not None:
        vol_s = f" cnet{r['cnet_n']}c>${float(r['cnet_thresh']):.4g}          "
    elif r.get("vol_slope_enabled"):
        vol_s = (
            f" vslope{int(r['vol_slope_window_s'])}s "
            f"{r['vol_slope_g_min']:.2f}-{r['vol_slope_g_max']:.2f}"
        )
    cb_s = f" cb={r['max_losses_cb']}" if r.get('max_losses_cb') else ""
    # Optimizer: si max_dd=0, rf stocké = total_pnl ($) — ne pas l'afficher comme "xxxx x"
    md = r.get("max_dd")
    tr = r.get("trades") or 0
    if md is not None and md > 0 and r.get("rf") is not None:
        opt_rf = f" RF={r['rf']:.0f}x"
    elif tr > 0 and md == 0:
        opt_rf = " RF=inf"
    else:
        opt_rf = ""
    return f"{curve_s} {cap_s}{vol_s}{cb_s}{opt_rf}"


# ── Point d'entrée principal ──────────────────────────────────────────────────
def generate_charts(
    top_results,
    csv_paths,
    out_dir,
    n_top=12,
    title_prefix="God Curve",
    forced_configs=None,
    vol_slope_ref_key="vol_0.5h_range",
    skip_analytics=False,
    asset='btc',
):
    """
    Genere les equity charts pour les top configs depuis un optimizer.
    top_results     : liste de dicts (output optimizer), tries par RF decroissant.
    csv_paths       : CSVs a charger (BTC ou ETH).
    out_dir         : dossier de sortie (ex: 'god_curve_charts/btc').
    forced_configs  : configs toujours incluses dans le chart overlay (ex: ref 8h/500).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_n = top_results[:n_top]
    # Ajouter les configs forcees si pas deja dans le top
    top_ids = {id(r) for r in top_n}
    extras = [r for r in (forced_configs or []) if r is not None and id(r) not in top_ids]
    configs = top_n + extras
    print(f"\nGenerating charts -> {out_dir}  ({len(configs)} configs)...", flush=True)

    # Charger contrats pour chaque timeframe + precompute vol
    # M5 sert de référence pour les fenêtres vol de M15 et H1
    contracts_by_tf = []
    m5_ref = None
    for tf_floor, bid_up, bid_down, ask_up, ask_down in TIMEFRAMES:
        try:
            print(f"  {tf_floor}: chargement CSV...", flush=True)
            cts = load_contracts(csv_paths, tf_floor, bid_up, bid_down, ask_up, ask_down)
            print(f"  {tf_floor}: precompute vol...", flush=True)
            if tf_floor == '5min':
                precompute_vol(cts, VOL_LBS)
                m5_ref = cts
            else:
                precompute_vol(cts, VOL_LBS, ref_contracts=m5_ref)
            contracts_by_tf.append(cts)
            print(f"  {tf_floor}: {len(cts)} contracts", flush=True)
        except Exception as e:
            print(f"  {tf_floor}: skipped ({e})", flush=True)
            contracts_by_tf.append([])

    for (tf_floor, *_), cts in zip(TIMEFRAMES, contracts_by_tf):
        if cts:
            attach_settlement_outcomes(cts, asset, tf=tf_floor)
            attach_cnet_data(cts, (2, 5, 10), ref_contracts=m5_ref)

    hour_index, n_hours = build_hour_index(contracts_by_tf)
    print(f"  {n_hours} heures tradees")

    all_series = []
    for i, r in enumerate(configs):
        name  = make_config_name(r)
        label = make_config_label(r)
        print(f"  [{i + 1}/{len(configs)}] {name}  simulation + PNG...", flush=True)

        combined_pnl = defaultdict(float)
        tf_pnls      = [defaultdict(float) for _ in contracts_by_tf]
        total_wins = total_losses = 0
        for i_tf, contracts in enumerate(contracts_by_tf):
            ph, w, l = run_config_tf(contracts, r, hour_index)
            for h_idx, pnl in ph.items():
                combined_pnl[h_idx]    += pnl
                tf_pnls[i_tf][h_idx]   += pnl
            total_wins   += w
            total_losses += l

        cum, hourly = build_cumulative(combined_pnl, n_hours)
        tf_names = ['M5', 'M15', 'H1']
        cum_by_tf = {}
        for i_tf, tf_name in enumerate(tf_names[:len(contracts_by_tf)]):
            c, _ = build_cumulative(tf_pnls[i_tf], n_hours)
            cum_by_tf[tf_name] = c

        # Scale pour maxDD = $500 (référentiel commun entre configs)
        TARGET_DD = 500.0
        raw_dd = float((np.maximum.accumulate(cum) - cum).max())
        scale  = TARGET_DD / raw_dd if raw_dd > 1.0 else 1.0
        cum        = cum * scale
        hourly     = hourly * scale
        cum_by_tf  = {k: v * scale for k, v in cum_by_tf.items()}

        all_series.append((name, label, cum, total_wins, total_losses))

        cfg_dir = out_dir / name
        cfg_dir.mkdir(exist_ok=True)
        chart_equity(cum, hourly, label, n_hours, hour_index,
                     total_wins, total_losses, cfg_dir / "equity.png",
                     cum_by_tf=cum_by_tf)
        if not skip_analytics:
            try:
                chart_entry_and_loss_analytics(contracts_by_tf, r, hour_index, cfg_dir, label)
            except Exception as e:
                print(f"  analytics skip ({e})", flush=True)

        dd   = (np.maximum.accumulate(cum) - cum).max()
        rf_s = _rf_str_hourly_equity(cum)
        days = n_hours / 24
        print(f"  {label[:65]}  PnL/j=${cum[-1]/days:.0f}  RF={rf_s}  scale={scale:.2f}x")

    chart_overlay(
        all_series, n_hours, hour_index,
        out_dir / "overlay_all.png",
        title=f"{title_prefix}  (M5+M15+H1, gaps exclus)"
    )
    print(f"Done -> {out_dir}/\n")
