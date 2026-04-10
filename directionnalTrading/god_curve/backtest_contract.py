#!/usr/bin/env python3
"""
backtest_contract.py — Backtest détaillé d'un contrat spécifique par slug Polymarket.

Usage:
    python backtest_contract.py <slug> [options]

Formats de slug supportés:
    btc-updown-5m-1775631300
    eth-updown-15m-1775631600
    btc-updown-1h-1775635200
    sol-updown-5m-1775631300
    bitcoin-up-or-down-april-8-2026-1am-et
    ethereum-up-or-down-march-15-2026-10pm-et

Options:
    --all-ticks     Affiche tous les ticks (par défaut : 60s avant expiry)
    --cfg <crypto>  Force la config (btc/eth/sol/xrp/bnb)
    --size <float>  Override la taille de position (USD)
"""

import sys, re, argparse, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except: pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import load_contracts, precompute_vol, VOL_LBS
import chart_utils as cu

BASE_DIR = Path(__file__).resolve().parent.parent / "reportLive" / "safeChase"

# ── Configs par crypto ─────────────────────────────────────────────────────────
CRYPTO_CONFIGS = {
    'btc': {
        'cfg': dict(
            curve='linear', slope=7.0, intercept=0.0, eq_cap=0.75,
            vol_lb_h=1, vol_thresh=60.0, vol_type='net',
            max_losses_cb=None, max_orders=2,
        ),
        'size': 100.0,
        'csv':  'BTC.csv',
        'label': 'BTC — vol_net_lin slope=7 cap=0.75 net1h>$60',
        'live_dir': 'slope7cap75/btc',
    },
    'eth': {
        'cfg': dict(
            curve='linear', slope=0.20, intercept=0.0,
            eq_cap=0.60,
            vol_lb_h=2.0, vol_thresh=0.50, vol_type='trend',
            max_losses_cb=None, max_orders=1,
        ),
        'size': 150.0,
        'csv':  'ETH.csv',
        'label': 'ETH — slope=0.2 trend>0.5 cap=0.60',
        'live_dir': 'slope0.2int0trend0.5/eth',
    },
    'sol': {
        'cfg': dict(
            curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
            vol_lb_h=None, vol_thresh=None,
            vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
            vrs_g_min=0.5, vrs_g_max=1.5,
            max_losses_cb=None, max_orders=2,
        ),
        'size': 100.0,
        'csv':  'SOL.csv',
        'label': 'SOL — VRS slope=10 cap=0.55',
    },
    'xrp': {
        'cfg': dict(
            curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
            vol_lb_h=None, vol_thresh=None,
            vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
            vrs_g_min=0.5, vrs_g_max=1.5,
            max_losses_cb=None, max_orders=2,
        ),
        'size': 100.0,
        'csv':  'XRP.csv',
        'label': 'XRP — VRS slope=10 cap=0.55',
    },
    'bnb': {
        'cfg': dict(
            curve='linear', slope=10.0, intercept=0.0, eq_cap=0.55,
            vol_lb_h=None, vol_thresh=None,
            vrs_enabled=True, vrs_lb=1.0, vrs_base=150.0,
            vrs_g_min=0.5, vrs_g_max=1.5,
            max_losses_cb=None, max_orders=2,
        ),
        'size': 100.0,
        'csv':  'BNB.csv',
        'label': 'BNB — VRS slope=10 cap=0.55',
    },
}

TF_KEY_MAP = {
    '5m': '5min', '15m': '15min', '1h': '1h',
    'm5': '5min', 'm15': '15min', 'h1': '1h',
}

TF_COLS = {
    '5min':  ('m5_up_bid',  'm5_down_bid',  'm5_up_ask',  'm5_down_ask'),
    '15min': ('m15_up_bid', 'm15_down_bid', 'm15_up_ask', 'm15_down_ask'),
    '1h':    ('h1_up_bid',  'h1_down_bid',  'h1_up_ask',  'h1_down_ask'),
}

CRYPTO_ALIASES = {
    'btc': 'btc', 'bitcoin': 'btc',
    'eth': 'eth', 'ethereum': 'eth',
    'sol': 'sol', 'solana': 'sol',
    'xrp': 'xrp', 'ripple': 'xrp',
    'bnb': 'bnb', 'binancecoin': 'bnb',
}

MONTH_STR = {
    'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
    'apr': 4, 'april': 4,   'may': 5, 'jun': 6, 'june': 6,
    'jul': 7, 'july': 7,    'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
    'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
}

TF_DURATIONS = {'5min': 300, '15min': 900, '1h': 3600}
ALL_TF_KEYS = ['5min', '15min', '1h']


# ── Parsing du slug ────────────────────────────────────────────────────────────
def parse_slug(slug: str):
    """
    Retourne (crypto_key, tf_floor, ce_ts) ou lève ValueError.

    Formats supportés:
      1) {crypto}-updown-{tf}-{timestamp}     → btc-updown-5m-1775631300
      2) {crypto}-up-or-down-{month}-{day}-{year}-{hour}{am/pm}-{tz}
         → bitcoin-up-or-down-april-8-2026-1am-et
    """
    slug = slug.strip().lower()
    # Retire le préfixe de domaine Polymarket si présent
    slug = re.sub(r'^.*polymarket\.com/event/', '', slug)

    # Format 1 : {crypto}-updown-{tf}-{unix_ts}
    m1 = re.match(
        r'^([a-z]+(?:-[a-z]+)*)-updown-([0-9a-z]+)-(\d{9,11})$', slug
    )
    if m1:
        raw_crypto, raw_tf, raw_ts = m1.group(1), m1.group(2), m1.group(3)
        crypto = _resolve_crypto(raw_crypto)
        tf = TF_KEY_MAP.get(raw_tf)
        if tf is None:
            raise ValueError(f"Timeframe inconnu : '{raw_tf}'. Attendu : 5m, 15m, 1h")
        return crypto, tf, int(raw_ts)

    # Format 2 : {crypto}-up-or-down-{month}-{day}-{year}-{H}am/pm-{tz}
    m2 = re.match(
        r'^([a-z]+(?:-[a-z]+)*)-up-or-down-([a-z]+)-(\d{1,2})-(\d{4})-(\d{1,2})(am|pm)-([a-z]+)$',
        slug,
    )
    if m2:
        raw_crypto = m2.group(1)
        month_str  = m2.group(2)
        day        = int(m2.group(3))
        year       = int(m2.group(4))
        hour       = int(m2.group(5))
        ampm       = m2.group(6)
        tz_str     = m2.group(7)

        crypto = _resolve_crypto(raw_crypto)
        month = MONTH_STR.get(month_str)
        if not month:
            raise ValueError(f"Mois inconnu : '{month_str}'")

        if ampm == 'pm' and hour != 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0

        # Décalage fuseau horaire
        tz_offsets = {'et': -4, 'est': -5, 'edt': -4, 'ct': -5, 'cst': -6, 'cdt': -5,
                      'pt': -7, 'pst': -8, 'pdt': -7, 'utc': 0}
        tz_off = tz_offsets.get(tz_str, 0)
        dt_local = datetime(year, month, day, hour, 0, 0)
        dt_utc = dt_local - timedelta(hours=tz_off)
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        ce_ts = int(dt_utc.timestamp())

        # Pour ce format on ne connaît pas le TF → retourner None, cherche le meilleur
        return crypto, None, ce_ts

    raise ValueError(
        f"Slug non reconnu : '{slug}'\n"
        "Formats supportés :\n"
        "  btc-updown-5m-1775631300\n"
        "  bitcoin-up-or-down-april-8-2026-1am-et"
    )


def _resolve_crypto(raw: str) -> str:
    key = CRYPTO_ALIASES.get(raw)
    if key is None:
        raise ValueError(
            f"Crypto inconnue : '{raw}'. Supportées : {list(CRYPTO_ALIASES.keys())}"
        )
    return key


# ── Chargement et recherche du contrat ────────────────────────────────────────
def find_contract(crypto, tf_floor, ce_ts, cfg_override=None):
    """
    Charge les CSV et retourne (contract, tf_floor, cfg, base_size).
    Si tf_floor est None, cherche parmi tous les TF et prend le plus proche.
    """
    conf = CRYPTO_CONFIGS[crypto]
    csv_path = str(BASE_DIR / conf['csv'])
    cfg      = cfg_override or conf['cfg']
    size     = conf['size']
    cu.BASE_SIZE = size

    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    # Charge M5 en premier (vol ref pour M15/H1, et VRS lookup)
    print(f"Chargement {conf['csv']}...")
    m5_cts = load_contracts([csv_path], '5min', *TF_COLS['5min'])
    precompute_vol(m5_cts, VOL_LBS)

    tfs_to_try = [tf_floor] if tf_floor else ALL_TF_KEYS

    best = None
    best_diff = float('inf')
    best_tf = None

    for tf in tfs_to_try:
        if tf == '5min':
            cts = m5_cts
        else:
            cts = load_contracts([csv_path], tf, *TF_COLS[tf])
            precompute_vol(cts, VOL_LBS, ref_contracts=m5_cts)

        for c in cts:
            # Le slug Polymarket utilise l'open_ts (début du contrat), pas le CE
            diff = abs(c['open_ts'] - ce_ts)
            if diff < best_diff:
                best_diff = diff
                best = c
                best_tf = tf
                best_cts = cts

    if best is None:
        raise ValueError("Aucun contrat trouvé dans le CSV.")

    if best_diff > TF_DURATIONS.get(best_tf, 300):
        raise ValueError(
            f"Contrat le plus proche est à {best_diff:.0f}s du timestamp demandé "
            f"(open attendu : {datetime.fromtimestamp(ce_ts, tz=timezone.utc)} UTC).\n"
            f"Contrat trouvé : open={pd.Timestamp(best['open_ts'], unit='s', tz='UTC')}"
        )

    if best_diff > 0:
        print(
            f"[AVERTISSEMENT] Timestamp exact non trouvé. "
            f"Contrat le plus proche : delta={best_diff:.0f}s, TF={best_tf}"
        )

    # Recharge ce TF si nécessaire pour avoir la ref M5
    return best, best_tf, cfg, size


# ── Simulation tick-à-tick avec logging ───────────────────────────────────────
def run_trace(c, cfg, base_size, show_all_ticks=False):
    """
    Rejoue le contrat tick par tick.
    Retourne la liste des events pour les stats finales.
    """
    curve  = cfg['curve']
    cap    = cfg['eq_cap']
    ce_ns  = np.datetime64(c['ce'])
    spots  = c['spot']
    ts_arr = c['ts']
    op     = float(c['op'])

    up_bid   = c['up_bid']
    down_bid = c['down_bid']
    up_ask   = c['up_ask']
    down_ask = c['down_ask']

    max_orders = int(cfg.get('max_orders', 2))
    intercept  = float(cfg.get('intercept', 0.0))

    # ── VRS ──
    if cfg.get('vrs_enabled') or cfg.get('vrs_inv'):
        _vrs_lb      = cfg['vrs_lb']
        _vrs_base    = float(cfg['vrs_base'])
        _vrs_g_min   = float(cfg['vrs_g_min'])
        _vrs_g_max   = float(cfg['vrs_g_max'])
        _vrs_is_inv  = bool(cfg.get('vrs_inv', False))
        _vrs_lookup  = c.get(f'_vrs_vol_lookup_{_vrs_lb:g}')
        _vrs_bucket  = None
        _vol = float(c.get(f"vol_{_vrs_lb:g}h_range", 0.0))
        if _vrs_is_inv:
            _ratio = _vol / _vrs_base if _vol > 1e-6 else _vrs_g_min
        else:
            _ratio = _vrs_base / _vol if _vol > 1e-6 else _vrs_g_max
        _g_vrs = float(np.clip(_ratio, _vrs_g_min, _vrs_g_max))
    else:
        _g_vrs = 1.0
        _vrs_lookup = None
        _vrs_bucket = None
        _vrs_g_min = _vrs_g_max = 1.0

    # État de la simulation
    activated_side = None
    active_order = active_size = active_ord_side = None
    pending_place = pending_size = pending_ord_side = None
    pending_cancel = False
    # max_orders est PAR SIDE (pas total)
    fc_up = fc_down = 0
    fills = []
    events = []  # log complet
    g_history = []

    # Données accumulées par side
    cost_up = cost_down = 0.0
    shares_up = shares_down = 0.0
    g_current = _g_vrs

    # Tracking premier trigger BT
    first_trigger_remain = None
    first_trigger_side   = None
    first_trigger_ts     = None

    # Tracking premier trigger BT
    first_trigger_remain = None
    first_trigger_side   = None
    first_trigger_ts     = None

    print()
    hdr = (f"{'#':>4} {'Timestamp':22} {'Remain':>7} {'Spot':>12} "
           f"{'Delta':>9} {'Thresh':>9} {'UBid':>6} {'DBid':>6}  Events")
    print(hdr)
    print("─" * len(hdr) + "──────────────────────────────────")

    total_ticks = len(spots)
    show_from_remain = 9999 if show_all_ticks else 62.0  # tout si all_ticks

    for j in range(total_ticks):
        s      = float(spots[j])
        remain = float((ce_ns - ts_arr[j]) / np.timedelta64(1, 's'))
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders):
            break

        ub = round(float(up_bid[j]),   2)
        db = round(float(down_bid[j]), 2)
        ua = round(float(up_ask[j]),   2)
        da = round(float(down_ask[j]), 2)

        # Recalcul VRS g si nouveau bucket 5min (M15/H1)
        g_changed = False
        if _vrs_lookup is not None:
            _ts_sec = int(ts_arr[j].astype('datetime64[s]').astype(np.int64))
            _bucket = (_ts_sec // 300) * 300
            if _bucket != _vrs_bucket:
                _vrs_bucket = _bucket
                _dv = _vrs_lookup.get(_bucket)
                if _dv is not None:
                    _r = (_dv / _vrs_base if _vrs_is_inv else _vrs_base / _dv) if _dv > 1e-6 else (
                        _vrs_g_min if _vrs_is_inv else _vrs_g_max)
                    new_g = float(np.clip(_r, _vrs_g_min, _vrs_g_max))
                    if abs(new_g - g_current) > 1e-6:
                        g_current = new_g
                        g_changed = True
                        g_history.append((j, remain, g_current, _dv))

        g = g_current
        if curve == 'linear':
            thresh = cfg['slope'] * g * remain + intercept
        else:
            thresh = cfg['A_exp'] * (np.exp(remain / cfg['tau']) - 1.0) * g

        status_parts = []
        if g_changed:
            status_parts.append(f"[g={g:.3f}]")

        # ── Logique fill / cancel (reproduit simulate_trade_log exactement) ──
        filled_now   = False
        cancelled_now = False

        if pending_cancel and active_order is not None:
            if active_ord_side:
                cb2 = ub if active_ord_side == 'UP' else db
                if cb2 <= round(active_order - 0.01, 2):
                    sh = active_size / active_order
                    fills.append({
                        'j': j, 'remain': remain, 'ts': str(ts_arr[j])[:22],
                        'side': active_ord_side, 'price': float(active_order),
                        'cost': float(active_size), 'shares': float(sh),
                    })
                    if active_ord_side == 'UP': fc_up += 1; cost_up += active_size; shares_up += sh
                    else:                        fc_down += 1; cost_down += active_size; shares_down += sh
                    filled_now = True
                    status_parts.append(
                        f"FILL@{active_order:.2f} (cancel-fill, shares={sh:.2f})"
                    )
            active_order = active_size = active_ord_side = None
            pending_cancel = False
            cancelled_now = True
            if not filled_now:
                status_parts.append("CANCEL")

        if pending_place is not None:
            # Si un cancel-fill vient de saturer le side → ne pas activer le nouvel ordre
            fc_pend = fc_up if pending_ord_side == 'UP' else fc_down
            if fc_pend < max_orders:
                active_order = pending_place; active_size = pending_size; active_ord_side = pending_ord_side
            else:
                status_parts.append(f"NEW_ORD_CANCELLED (cancel-fill a saturé fc)")
            pending_place = pending_size = pending_ord_side = None

        if active_order is not None and active_ord_side and not filled_now:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                sh = active_size / active_order
                fills.append({
                    'j': j, 'remain': remain, 'ts': str(ts_arr[j])[:22],
                    'side': active_ord_side, 'price': float(active_order),
                    'cost': float(active_size), 'shares': float(sh),
                })
                if active_ord_side == 'UP': fc_up += 1; cost_up += active_size; shares_up += sh
                else:                        fc_down += 1; cost_down += active_size; shares_down += sh
                active_order = active_size = active_ord_side = None
                filled_now = True
                status_parts.append(
                    f"FILL@{fills[-1]['price']:.2f} "
                    f"(shares={fills[-1]['shares']:.2f}, cost=${fills[-1]['cost']:.2f})"
                )
                if fc_up >= max_orders and fc_down >= max_orders:
                    _print_tick(j, ts_arr[j], remain, s, op, thresh, ub, db, status_parts, show_from_remain)
                    break

        # ── Déactivation ──
        deactivated = False
        if activated_side and not filled_now:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None:
                    pending_cancel = True
                    status_parts.append(f"pend_cancel (buf={buf:.4f}<{thresh:.4f})")
                pending_place = pending_size = pending_ord_side = None
                activated_side = None
                deactivated = True
                status_parts.append(f"DEACTIVATE")

        # ── Activation (par side, si ce côté n'est pas maxé) ──
        if not activated_side and not deactivated:
            delta_up   = s - op
            delta_down = op - s
            if fc_up < max_orders and delta_up >= thresh:
                activated_side = 'UP'
                status_parts.append(f"TRIGGER UP  (delta={delta_up:+.4f} >= {thresh:.4f})")
                if first_trigger_remain is None:
                    first_trigger_remain = remain
                    first_trigger_side   = 'UP'
                    first_trigger_ts     = str(ts_arr[j])[:22]
            elif fc_down < max_orders and delta_down >= thresh:
                activated_side = 'DOWN'
                status_parts.append(f"TRIGGER DOWN (delta={delta_down:+.4f} >= {thresh:.4f})")
                if first_trigger_remain is None:
                    first_trigger_remain = remain
                    first_trigger_side   = 'DOWN'
                    first_trigger_ts     = str(ts_arr[j])[:22]

        # ── Si ce côté est maxé, on ignore ──
        if activated_side:
            fc_cur = fc_up if activated_side == 'UP' else fc_down
            if fc_cur >= max_orders:
                activated_side = None

        # ── Placement d'ordre ──
        if activated_side and not filled_now:
            bid = ub if activated_side == 'UP' else db
            if bid >= cap:
                proposed = min(bid, 0.99)
                if active_order is None and pending_place is None:
                    pending_place = proposed; pending_size = base_size; pending_ord_side = activated_side
                    status_parts.append(f"ORDER@{proposed:.2f} (bid={bid:.2f})")
                elif active_order is not None and proposed > active_order and active_ord_side == activated_side:
                    pending_cancel = True; pending_place = proposed; pending_size = base_size; pending_ord_side = activated_side
                    status_parts.append(f"UPGRADE {active_order:.2f}→{proposed:.2f} (pend_cancel+order)")
                elif pending_place is not None and proposed > pending_place and pending_ord_side == activated_side:
                    pending_place = proposed; pending_size = base_size; pending_ord_side = activated_side
                    status_parts.append(f"UPGRADE_PEND {proposed:.2f}")
            else:
                if activated_side:
                    status_parts.append(f"bid_too_low ({bid:.2f} < cap {cap:.2f})")

        # ── Affichage tick ──
        if active_order and not filled_now and not status_parts:
            status_parts.append(f"ord={active_order:.2f}")

        _print_tick(j, ts_arr[j], remain, s, op, thresh, ub, db, status_parts, show_from_remain)

        if fc_up >= max_orders and fc_down >= max_orders:
            break

    # ── Résolution direction ──
    won_up = float(up_ask[-1]) > float(down_ask[-1]) if len(up_ask) else up_bid[-1] > down_bid[-1]

    # ── Expiry fill : côté perdant uniquement ──
    expiry_fill = False
    exp_side = pending_ord_side if pending_place is not None else active_ord_side
    exp_ord  = pending_place    if pending_place is not None else active_order
    exp_sz   = pending_size     if pending_place is not None else active_size
    exp_fc   = (fc_up if exp_side == 'UP' else fc_down) if exp_side else max_orders
    if exp_fc < max_orders and exp_side and exp_ord is not None and exp_sz is not None:
        if (exp_side == 'UP') != won_up:
            sh = exp_sz / exp_ord
            if exp_side == 'UP': fc_up += 1; cost_up += exp_sz; shares_up += sh
            else:                 fc_down += 1; cost_down += exp_sz; shares_down += sh
            fills.append(dict(
                side=exp_side, price=exp_ord, size=exp_sz,
                j=total_ticks - 1, expiry_fill=True,
                max_bid_after=exp_ord, mfe=0.0,
                ts=ts_arr[-1] if len(ts_arr) else None,
                remain=0.0, cost=exp_sz, shares=sh,
            ))
            expiry_fill = True
            _opp = 'DOWN' if exp_side == 'UP' else 'UP'
            print(f"\n  [EXPIRY FILL] ordre {exp_side} @{exp_ord:.2f} fill à l'expiry "
                  f"(contrat résout {_opp} → bid {exp_side} s'effondre → fill garanti)")

    total_cost = cost_up + cost_down
    total_shares_up = shares_up
    total_shares_down = shares_down
    has_fills = total_cost > 0

    if has_fills:
        payout = (shares_up if won_up else 0.0) + (shares_down if not won_up else 0.0)
        pnl = payout - total_cost
        won = pnl > 0
        fill_side = 'UP' if shares_up > shares_down else 'DOWN'

        # MFE par fill
        for f in fills:
            if f.get('expiry_fill'):
                continue
            jj = f['j']
            arr = up_bid if f['side'] == 'UP' else down_bid
            f['max_bid_after'] = float(np.max(arr[jj:])) if jj < len(arr) else f['price']
            f['mfe'] = f['max_bid_after'] - f['price']
    else:
        won = None
        pnl = None
        fill_side = None

    fill_count = len(fills)
    contract_cost   = total_cost
    contract_shares = shares_up + shares_down

    return {
        'op':               op,
        'close_spot':       float(spots[-1]),
        'total_ticks':      total_ticks,
        'fills':            fills,
        'fill_count':       fill_count,
        'fill_side':        fill_side,
        'contract_cost':    contract_cost,
        'contract_shares':  contract_shares,
        'avg_fill_price':   (sum(f['price'] for f in fills) / len(fills)) if fills else None,
        'won_up':           won_up,
        'direction_real':   'UP' if won_up else 'DOWN',
        'direction_spot':   'UP' if float(spots[-1]) > op else 'DOWN',
        'won':              won,
        'pnl':              pnl,
        'g_history':        g_history,
        'g_final':          g_current,
        'active_order_at_expiry':  active_order,
        'expiry_fill':             expiry_fill,
        'first_trigger_remain':    first_trigger_remain,
        'first_trigger_side':      first_trigger_side,
        'first_trigger_ts':        first_trigger_ts,
    }


def _print_tick(j, ts, remain, spot, op, thresh, ub, db, status_parts, show_from_remain):
    if remain > show_from_remain and not status_parts:
        return
    ts_str = str(ts)[:22]
    delta  = spot - op
    delta_s = f"{delta:+.4f}"
    ev = ' | '.join(status_parts) if status_parts else ''
    print(
        f"{j:4d} {ts_str:22} {remain:7.2f}s {spot:12.4f} "
        f"{delta_s:>9} {thresh:9.4f} {ub:6.2f} {db:6.2f}  {ev}"
    )


# ── Affichage des stats finales ────────────────────────────────────────────────
def print_stats(c, result, cfg, base_size, tf_floor, crypto):
    conf = CRYPTO_CONFIGS[crypto]
    op   = result['op']
    cl   = result['close_spot']
    ce   = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
    open_ts = ce - timedelta(seconds=TF_DURATIONS[tf_floor])

    print()
    print("═" * 68)
    print(f"  BACKTEST — {ce.strftime('%Y-%m-%d %H:%M:%S UTC')}  [{tf_floor}]")
    print(f"  Config : {conf['label']}")
    print("═" * 68)

    # ── Infos contrat ──
    print(f"\n{'─'*35} CONTRAT {'─'*25}")
    print(f"  Ouverture       : {open_ts.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Clôture (CE)    : {ce.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Timeframe       : {tf_floor}")
    print(f"  Open spot (BT)  : {op:.4f}")
    print(f"  Close spot      : {cl:.4f}")
    print(f"  Mouvement net   : {cl - op:+.4f}  ({(cl-op)/op*100:+.3f}%)")
    print(f"  Direction spot  : {result['direction_spot']}")
    print(f"  Résolution marché: {result['direction_real']}  (via last up_ask vs down_ask)")
    if result['direction_real'] != result['direction_spot']:
        print(f"  [!] Divergence spot/marché : spot dit {result['direction_spot']} "
              f"mais marché résout {result['direction_real']}")
    print(f"  Ticks CSV       : {result['total_ticks']}")

    # ── Vol ──
    print(f"\n{'─'*35} VOLATILITÉ {'─'*22}")
    for lb in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        rng = c.get(f'vol_{lb:g}h_range')
        net = c.get(f'vol_{lb:g}h_net')
        trd = c.get(f'vol_{lb:g}h_trend')
        if rng is not None:
            print(f"  vol_{lb:g}h   range={rng:.4f}  net={net:.4f}  trend={trd:.3f}")

    # ── VRS ──
    if cfg.get('vrs_enabled'):
        lb   = cfg['vrs_lb']
        vol  = float(c.get(f"vol_{lb:g}h_range", 0.0))
        base = cfg['vrs_base']
        g    = result['g_final']
        slope_eff = cfg['slope'] * g
        print(f"\n{'─'*35} VRS {'─'*29}")
        print(f"  vol_{lb:g}h_range : {vol:.4f}")
        print(f"  vrs_base        : {base}")
        print(f"  g = clip({base}/{vol:.2f}, {cfg['vrs_g_min']}, {cfg['vrs_g_max']}) = {g:.4f}")
        print(f"  slope_eff       : {cfg['slope']} × {g:.4f} = {slope_eff:.4f}")
        if result['g_history']:
            print(f"  Changements g pendant le contrat :")
            for tick_j, rem, gv, dv in result['g_history']:
                print(f"    tick={tick_j:3d}  remain={rem:6.2f}s  vol={dv:.4f}  g={gv:.4f}")

    # ── Filtre vol ETH ──
    vol_thresh = cfg.get('vol_thresh')
    if vol_thresh is not None:
        lb   = cfg['vol_lb_h']
        vtype = cfg.get('vol_type', 'range')
        val  = c.get(f'vol_{lb:g}h_{vtype}', 0.0)
        gate = "PASSE" if float(val) >= vol_thresh else "BLOQUE"
        print(f"\n{'─'*35} FILTRE VOL {'─'*22}")
        print(f"  vol_{lb:g}h_{vtype} = {val:.4f}  (seuil >= {vol_thresh})  → {gate}")

    # ── Fills ──
    print(f"\n{'─'*35} FILLS {'─'*27}")
    fills = result['fills']
    if not fills:
        aoe = result['active_order_at_expiry']
        print(f"  Aucun fill BT.")
        if aoe:
            print(f"  Ordre actif à expiry : {aoe:.2f} (bid jamais retracé sous {aoe-0.01:.2f})")
        else:
            if vol_thresh and float(c.get(f'vol_{cfg["vol_lb_h"]:g}h_{cfg.get("vol_type","range")}', 0.0)) < vol_thresh:
                print(f"  Raison : filtre vol bloqué")
            else:
                print(f"  Raison : signal jamais atteint OU bid jamais sous ordre-0.01")
    else:
        print(f"  Nombre de fills   : {result['fill_count']}")
        print(f"  Côté tradé        : {result['fill_side']}")
        print(f"  Taille/fill (USD) : ${base_size:.2f}")
        print()
        for i, f in enumerate(fills, 1):
            tag = "  [EXPIRY FILL]" if f.get('expiry_fill') else ""
            print(f"  Fill #{i}:{tag}")
            print(f"    Timestamp     : {f.get('ts', 'expiry')}")
            print(f"    Remain        : {f.get('remain', 0.0):.2f}s")
            print(f"    Prix fill     : {f['price']:.4f}")
            # Certains fills (legacy / import) n'ont pas toujours les mêmes champs.
            cost = f.get('cost')
            if cost is None:
                if f.get('size') is not None:
                    cost = f['size']
                elif f.get('shares') is not None:
                    cost = float(f['shares']) * float(f['price'])
                else:
                    cost = float(base_size)

            shares = f.get('shares')
            if shares is None:
                if f.get('size') is not None:
                    shares = float(f['size']) / float(f['price'])
                else:
                    shares = float(cost) / float(f['price'])

            print(f"    Coût          : ${float(cost):.2f}")
            print(f"    Shares        : {float(shares):.4f}")
            print(f"    Max bid after : {f.get('max_bid_after', f['price']):.4f}  (MFE={f.get('mfe', 0):.4f})")

        avg_fill = result['contract_cost'] / result['fill_count'] / base_size
        # Recalcul propre du prix moyen pondéré
        avg_price = result['contract_cost'] / result['contract_shares']
        print()
        print(f"  Prix moyen fill   : {avg_price:.4f}")
        print(f"  Coût total        : ${result['contract_cost']:.2f}")
        print(f"  Shares totales    : {result['contract_shares']:.4f}")

    # ── PnL ──
    print(f"\n{'─'*35} PNL {'─'*29}")
    if result['won'] is not None:
        won      = result['won']
        pnl      = result['pnl']
        shares   = result['contract_shares']
        cost     = result['contract_cost']
        dir_real  = result['direction_real']   # résolution marché (last ask)
        dir_spot  = result['direction_spot']    # mouvement spot
        dir_trad  = result['fill_side']

        print(f"  Direction tradée   : {dir_trad}")
        print(f"  Direction spot     : {dir_spot}  (close {cl:.2f} vs open {op:.2f})")
        print(f"  Résolution marché  : {dir_real}  (via last ask CSV)")
        print(f"  Résultat           : {'WIN ✓' if won else 'LOSS ✗'}  (tradé {dir_trad}, marché={dir_real})")
        if dir_real != dir_spot:
            print(f"  [!] Spot et résolution divergent — le last tick CSV n'est pas "
                  f"le vrai prix de clôture Chainlink")
        print()
        if won:
            print(f"  Shares encaissées : {shares:.4f}")
            print(f"  Coût dépensé      : ${cost:.2f}")
            print(f"  PnL brut          : ${shares:.4f} - ${cost:.2f} = ${pnl:.2f}")
        else:
            print(f"  Shares perdues    : {shares:.4f}")
            print(f"  Coût dépensé      : ${cost:.2f}")
            print(f"  PnL               : -${cost:.2f} (perte totale)")

        print(f"\n  PNL NET           : ${pnl:.2f}")

        # Expected PnL si win garanti
        for f in fills:
            p = f['price']
            exp_win = (base_size / p) - base_size
            exp_loss = -base_size
            mfill = f.get('max_bid_after', p)
            mfe_pnl = (base_size / p) * mfill - base_size
            print(f"\n  Fill @ {p:.4f}  →  si WIN: +${exp_win:.2f} | si LOSS: -${base_size:.2f}")
            print(f"    Max bid after fill : {mfill:.4f} → PnL potentiel max : ${mfe_pnl:.2f}")
    else:
        print(f"  Aucun fill → PnL = $0.00")

    print()
    print("═" * 68)


# ── Live : recherche & parsing JSONL ──────────────────────────────────────────
CEST_OFFSET = timedelta(hours=2)  # April 2026 = CEST

MONTH_FR = {
    1: 'janv', 2: 'fevr', 3: 'mars', 4: 'avr',  5: 'mai',  6: 'juin',
    7: 'juil', 8: 'aout', 9: 'sept', 10: 'oct', 11: 'nov', 12: 'dec',
}

TF_SLUG = {'5min': 'm5', '15min': 'm15', '1h': 'h1'}


def find_live_jsonl(ce_utc: datetime, tf_floor: str, crypto: str):
    """
    Cherche le fichier JSONL live correspondant au contrat.
    Retourne Path ou None si introuvable.
    """
    live_dir_rel = CRYPTO_CONFIGS[crypto].get('live_dir')
    if not live_dir_rel:
        return None

    tf_slug = TF_SLUG.get(tf_floor, 'm5')
    live_dir = BASE_DIR / live_dir_rel / tf_slug
    if not live_dir.exists():
        return None

    dur     = timedelta(seconds=TF_DURATIONS[tf_floor])
    open_cest = (ce_utc - dur + CEST_OFFSET)
    ce_cest   = ce_utc + CEST_OFFSET

    day   = open_cest.day
    month = MONTH_FR[open_cest.month]

    open_s = f"{open_cest.hour:02d}h{open_cest.minute:02d}"
    ce_s   = f"{ce_cest.hour:02d}h{ce_cest.minute:02d}"
    fname  = f"{day}_{month}_{open_s}-{ce_s}.jsonl"

    path = live_dir / fname
    if path.exists():
        return path

    # Fallback : glob sur le pattern du jour/mois
    pattern = f"{day}_{month}_*-{ce_s}.jsonl"
    matches = list(live_dir.glob(pattern))
    return matches[0] if matches else None


def parse_live_jsonl(path: Path):
    """Parse un fichier JSONL live et retourne un dict structuré."""
    events = []
    for line in path.open(encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    def get(ev_type):
        return next((e for e in events if e.get('event') == ev_type), None)

    def get_all(ev_type):
        return [e for e in events if e.get('event') == ev_type]

    wo  = get('window_open')
    ws  = get('window_start')
    trg = get('safety_trigger')
    we  = get('window_ended')
    placements  = get_all('chase_placed')
    snapshots   = get_all('dashboard_snapshot')
    deactivates = get_all('safety_deactivate') or []

    # PnL live
    live_pnl   = None
    live_shares = 0.0
    live_cost   = 0.0
    live_filled = False
    live_side   = None
    if we:
        up_s  = float(we.get('up_shares',  0) or 0)
        dn_s  = float(we.get('down_shares', 0) or 0)
        up_c  = float(we.get('up_cost',    0) or 0)
        dn_c  = float(we.get('down_cost',  0) or 0)
        live_shares = up_s + dn_s
        live_cost   = up_c + dn_c
        live_filled = live_shares > 1e-9
        if live_filled:
            live_side = 'UP' if up_s > dn_s else 'DOWN'
            ep = we.get('expected_pnl')
            if ep is not None:
                live_pnl = float(ep)

    # Snapshot le plus proche du trigger
    snap_at_trigger = None
    if trg and snapshots:
        trg_ts = pd.to_datetime(trg['ts'])
        closest = min(snapshots, key=lambda s: abs(pd.to_datetime(s['ts']) - trg_ts))
        snap_at_trigger = closest

    return {
        'path':             path,
        'window_open':      wo,
        'window_start':     ws,
        'trigger':          trg,
        'placements':       placements,
        'window_ended':     we,
        'snapshots':        snapshots,
        'snap_at_trigger':  snap_at_trigger,
        'deactivates':      deactivates,
        'live_filled':      live_filled,
        'live_side':        live_side,
        'live_shares':      live_shares,
        'live_cost':        live_cost,
        'live_pnl':         live_pnl,
    }


def print_live_comparison(bt_result, live, c, cfg, base_size, tf_floor, crypto):
    """Affiche la comparaison détaillée Live vs BT."""
    print()
    print("═" * 68)
    print("  COMPARAISON LIVE vs BT")
    print("═" * 68)

    if live is None:
        print("  Aucun fichier JSONL live trouvé pour ce contrat.")
        print("  (le contrat n'était peut-être pas monitoré, ou hors période live)")
        print("═" * 68)
        return

    wo  = live['window_open']
    ws  = live['window_start']
    trg = live['trigger']
    we  = live['window_ended']

    # ── Open price ──
    print(f"\n{'─'*35} OPEN PRICE {'─'*22}")
    bt_open = bt_result['op']
    live_open = float(wo.get('binance_kline', 0)) if wo else None
    if live_open:
        gap = live_open - bt_open
        print(f"  BT open  (1er tick CSV)  : {bt_open:.4f}")
        print(f"  Live open (binance_kline): {live_open:.4f}")
        print(f"  Ecart                    : {gap:+.4f}  "
              f"({'live > BT' if gap > 0 else 'live < BT'})")
        if wo:
            print(f"  Chainlink à open         : {float(wo.get('chainlink', 0)):.4f}")
            print(f"  Synthetic à open         : {float(wo.get('synthetic', 0)):.4f}")
    else:
        print("  window_open manquant dans le JSONL")

    # ── VRS live ──
    if ws and cfg.get('vrs_enabled'):
        print(f"\n{'─'*35} VRS LIVE {'─'*24}")
        print(f"  btc_vrs_range_usd : {ws.get('btc_vrs_range_usd', '?')}")
        print(f"  btc_vrs_g         : {ws.get('btc_vrs_g', '?')}")
        print(f"  slope_effective   : {ws.get('slope_effective', '?')}")
        print(f"  max_chase_price   : {ws.get('max_chase_price', '?')}")
        print(f"  min_bid           : {ws.get('min_bid', '?')}")
        bt_g = bt_result['g_final']
        live_g = float(ws.get('btc_vrs_g', bt_g))
        if abs(live_g - bt_g) > 0.001:
            print(f"  [!] g divergence : live={live_g:.4f} vs BT={bt_g:.4f}")
        else:
            print(f"  [OK] g coherent  : live={live_g:.4f} == BT={bt_g:.4f}")

    # ── Trigger ──
    print(f"\n{'─'*35} TRIGGER {'─'*25}")
    bt_trg_tte  = bt_result['first_trigger_remain']
    bt_trg_side = bt_result['first_trigger_side']
    bt_trg_ts   = bt_result['first_trigger_ts']

    if bt_trg_tte is not None:
        print(f"  BT trigger  : {bt_trg_side:4s}  remain={bt_trg_tte:.2f}s  ts={bt_trg_ts}")
    else:
        print(f"  BT trigger  : AUCUN")

    if trg:
        live_trg_tte  = float(trg.get('time_to_expiry_s', 0))
        live_trg_side = trg.get('side', '?')
        live_trg_ts   = trg.get('ts', '?')[:22]
        live_trg_spot = float(trg.get('btc_price', trg.get('eth_price', 0)) or 0)
        binance_p     = float(trg.get('binance_price', 0) or 0)
        chainlink_p   = float(trg.get('chainlink_price', 0) or 0)

        print(f"  Live trigger: {live_trg_side:4s}  remain={live_trg_tte:.2f}s  ts={live_trg_ts}")

        if bt_trg_tte is not None:
            gap = bt_trg_tte - live_trg_tte
            print(f"\n  Ecart timing trigger : {gap:+.2f}s  "
                  f"({'BT plus tôt' if gap > 0 else 'Live plus tôt' if gap < 0 else 'identique'})")
            if abs(gap) > 0.5:
                print(f"  [!] Gap > 0.5s → probable retard synthétique")

        print(f"\n  Prix au trigger live:")
        print(f"    synthetic  : {live_trg_spot:.4f}")
        print(f"    binance    : {binance_p:.4f}")
        print(f"    chainlink  : {chainlink_p:.4f}")
        if binance_p and live_trg_spot:
            synth_lag = binance_p - live_trg_spot
            print(f"    synth lag  : {synth_lag:+.4f}  "
                  f"({'binance > synth → synth retard' if synth_lag > 0 else 'OK'})")

        # Snapshot le plus proche du trigger
        snap = live['snap_at_trigger']
        if snap:
            print(f"\n  Dashboard snapshot au trigger :")
            print(f"    tte_s         : {snap.get('tte_s', '?')}s")
            print(f"    delta_binance : {snap.get('delta_binance', '?')}")
            print(f"    delta_synth   : {snap.get('delta_synthetic', '?')}")
            print(f"    required_usd  : {snap.get('required_usd', '?')}")
            print(f"    up_bid        : {snap.get('up_bid', '?')}  down_bid: {snap.get('down_bid', '?')}")

        # Côtés concordent ?
        if bt_trg_side and live_trg_side:
            if bt_trg_side != live_trg_side:
                print(f"\n  [!!] DIRECTION DIVERGE : BT={bt_trg_side} vs Live={live_trg_side}")
            else:
                print(f"\n  [OK] Direction identique : {bt_trg_side}")
    else:
        print(f"  Live trigger: AUCUN (safety_trigger jamais tiré)")
        if bt_trg_tte is not None:
            print(f"  [!] BT avait triggeré à {bt_trg_tte:.2f}s → filtre synthétique probable")

    # ── Placements d'ordres live ──
    print(f"\n{'─'*35} ORDRES LIVE {'─'*21}")
    placements = live['placements']
    if placements:
        for i, p in enumerate(placements, 1):
            print(f"  Ordre #{i}: side={p.get('side','?'):4s}  price={p.get('price','?'):.2f}  "
                  f"size=${float(p.get('size', 0)):.2f}  rtt={p.get('rtt_ms','?')}ms  "
                  f"ts={p.get('ts','?')[:22]}")
    else:
        print("  Aucun ordre live placé")

    # Déactivations
    deacts = live['deactivates']
    if deacts:
        for d in deacts:
            print(f"  Déactivation live : ts={d.get('ts','?')[:22]}")

    # ── Fills comparaison ──
    print(f"\n{'─'*35} FILLS — BT vs Live {'─'*14}")

    bt_fills = bt_result['fills']
    bt_avg   = (bt_result['contract_cost'] / bt_result['contract_shares']
                if bt_result['contract_shares'] > 0 else None)

    live_filled  = live['live_filled']
    live_shares  = live['live_shares']
    live_cost    = live['live_cost']
    live_avg     = live_cost / live_shares if live_shares > 1e-9 else None

    if we:
        live_avg_display = f"{live_avg:.4f}" if live_avg else "N/A"
        print(f"  Live : {'filé' if live_filled else 'NON filé'}  "
              f"shares={live_shares:.4f}  cost=${live_cost:.2f}  avg_price={live_avg_display}")
    print(f"  BT   : {'filé' if bt_fills else 'NON filé'}  "
          f"shares={bt_result['contract_shares']:.4f}  cost=${bt_result['contract_cost']:.2f}  "
          f"avg_price={f'{bt_avg:.4f}' if bt_avg else 'N/A'}")

    if live_avg and bt_avg:
        fill_gap = live_avg - bt_avg
        print(f"\n  Ecart prix fill : live={live_avg:.4f}  BT={bt_avg:.4f}  "
              f"delta={fill_gap:+.4f}  "
              f"({'live plus cher' if fill_gap > 0 else 'live moins cher'})")
        if trg and bt_trg_tte is not None:
            gap_s = bt_trg_tte - float(trg.get('time_to_expiry_s', 0))
            if gap_s > 0.1:
                print(f"  → probable cause : BT a triggeré {gap_s:.2f}s plus tôt "
                      f"quand bid était plus bas")

    # ── PnL comparaison ──
    print(f"\n{'─'*35} PNL — BT vs Live {'─'*15}")
    bt_pnl   = bt_result['pnl']
    live_pnl = live['live_pnl']

    bt_won   = bt_result['won']
    dir_real = bt_result['direction_real']

    if live_filled and we:
        # Résolution live via final bids
        final_up_bid  = float(we.get('final_up_bid',  0) or 0)
        final_dn_bid  = float(we.get('final_down_bid', 0) or 0)
        live_won_up   = final_up_bid > final_dn_bid
        live_won      = (live_won_up == (live['live_side'] == 'UP'))
        live_pnl_calc = (live_shares if live_won else 0.0) - live_cost

        print(f"  Live : {'WIN ✓' if live_won else 'LOSS ✗'}  "
              f"PnL=${live_pnl_calc:.2f}  "
              f"(via final bids up={final_up_bid} dn={final_dn_bid})")
        if live_pnl is not None:
            print(f"         expected_pnl JSONL: ${live_pnl:.2f}")
    elif not live_filled:
        print(f"  Live : NON filé → PnL=$0.00")

    if bt_pnl is not None:
        print(f"  BT   : {'WIN ✓' if bt_won else 'LOSS ✗'}  PnL=${bt_pnl:.2f}")
    else:
        print(f"  BT   : NON filé → PnL=$0.00")

    # Delta
    bt_pnl_v   = bt_pnl or 0.0
    live_pnl_v  = (live_shares if live_filled and live_won else 0.0) - live_cost if live_filled else 0.0
    if 'live_won' not in dir():
        live_pnl_v = 0.0
    delta = live_pnl_v - bt_pnl_v
    print(f"\n  DELTA PnL (live - BT) : ${delta:+.2f}")
    if delta < -5:
        print(f"  → BT surperforme live de ${abs(delta):.2f} sur ce contrat")
    elif delta > 5:
        print(f"  → Live surperforme BT de ${abs(delta):.2f} sur ce contrat")
    else:
        print(f"  → Résultats proches")

    # ── Résolution direction ──
    if we:
        print(f"\n{'─'*35} RÉSOLUTION {'─'*22}")
        end_spot = float(we.get('end_spot', 0) or 0)
        start    = float(we.get('start_spot', 0) or 0)
        cl_price = float(we.get('chainlink_price', 0) or 0)
        sy_price = float(we.get('synthetic_price', 0) or 0)
        print(f"  Spot open  : {start:.4f}")
        print(f"  Spot close : {end_spot:.4f}  ({'+' if end_spot > start else ''}{end_spot-start:.4f})")
        print(f"  Chainlink  : {cl_price:.4f}  → {'UP' if cl_price > start else 'DOWN'}")
        print(f"  Synthetic  : {sy_price:.4f}  → {'UP' if sy_price > start else 'DOWN'}")
        bt_dir_ask  = dir_real   # basé sur last ask CSV
        bt_dir_spot = bt_result['direction_spot']
        print(f"  BT résolution (last ask)  : {bt_dir_ask}")
        print(f"  BT direction spot         : {bt_dir_spot}")
        if bt_dir_ask != bt_dir_spot:
            print(f"  [!] BT spot/ask divergent — last CSV tick ≠ vraie clôture")
        live_dir_final = 'UP' if cl_price > start else 'DOWN'
        if live_dir_final != bt_dir_ask:
            print(f"  [!!] DIVERGENCE RÉSOLUTION : live={live_dir_final} vs BT={dir_real}")
            print(f"       Chainlink vs Binance divergence → possible cause de loss/win flip")
        else:
            print(f"  [OK] Résolution identique : {dir_real}")

    print()
    print("═" * 68)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Backtest détaillé d'un contrat Polymarket par slug."
    )
    parser.add_argument('slug', help="Slug du contrat (ex: btc-updown-5m-1775631300)")
    parser.add_argument('--all-ticks', action='store_true',
                        help="Affiche tous les ticks CSV (par défaut: 62s avant expiry)")
    parser.add_argument('--cfg', choices=list(CRYPTO_CONFIGS.keys()), default=None,
                        help="Force la config crypto (ex: --cfg btc)")
    parser.add_argument('--size', type=float, default=None,
                        help="Override taille position USD (ex: --size 200)")
    args = parser.parse_args()

    # Parse slug
    try:
        crypto, tf_floor, ce_ts = parse_slug(args.slug)
    except ValueError as e:
        print(f"ERREUR slug : {e}")
        sys.exit(1)

    if args.cfg:
        crypto = args.cfg

    print(f"Slug parsé → crypto={crypto.upper()}, tf={tf_floor or 'auto'}, "
          f"CE={datetime.fromtimestamp(ce_ts, tz=timezone.utc)}")

    # Charge et trouve le contrat
    try:
        c, tf_floor, cfg, base_size = find_contract(crypto, tf_floor, ce_ts)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERREUR : {e}")
        sys.exit(1)

    if args.size:
        base_size = args.size
        cu.BASE_SIZE = base_size

    ce_dt = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
    print(f"Contrat trouvé : CE={ce_dt}  TF={tf_floor}  CE_TS={int(c['ce_ts'])}")

    # Trace
    result = run_trace(c, cfg, base_size, show_all_ticks=args.all_ticks)

    # Stats BT
    print_stats(c, result, cfg, base_size, tf_floor, crypto)

    # Recherche JSONL live
    ce_dt = c['ce'].to_pydatetime().replace(tzinfo=timezone.utc)
    jsonl_path = find_live_jsonl(ce_dt, tf_floor, crypto)
    if jsonl_path:
        print(f"\n  [LIVE] Fichier trouvé : {jsonl_path.name}")
        live_data = parse_live_jsonl(jsonl_path)
    else:
        print(f"\n  [LIVE] Aucun fichier JSONL trouvé (contrat non monitoré en live)")
        live_data = None

    # Comparaison Live vs BT
    print_live_comparison(result, live_data, c, cfg, base_size, tf_floor, crypto)


if __name__ == '__main__':
    main()
