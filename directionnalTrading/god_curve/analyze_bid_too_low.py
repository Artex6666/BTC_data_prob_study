#!/usr/bin/env python3
"""
analyze_bid_too_low.py — Analyse les cas où bid_too_low a bloqué un ordre
et croise avec settlement.csv pour voir si le côté triggeré avait raison.

Question : quand bid < cap (0.75) mais trigger actif, à quelle fréquence
le côté triggeré gagne-t-il quand même ? → calibrer un seuil plus bas ?
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_utils import load_contracts, precompute_vol, VOL_LBS

ROOT = Path(__file__).resolve().parent.parent
BTC_CSV = [str(ROOT / 'Datas' / 'csv' / 'BTC.csv'),
           str(ROOT / 'reportLive' / 'safeChase' / 'BTC.csv')]
BTC_CSV = [p for p in BTC_CSV if Path(p).exists()]

SETT_CSV = ROOT / 'Datas' / 'csv' / 'settlement.csv'

CFG = dict(
    curve='linear', slope=7.0, intercept=0.0, eq_cap=0.75,
    vol_lb_h=1, vol_thresh=60.0, vol_type='net',
    max_losses_cb=None, max_orders=2,
)
BASE_SIZE = 120.0
TF = '5min'
TF_COLS = ('m5_up_bid', 'm5_down_bid', 'm5_up_ask', 'm5_down_ask')

# ── Buckets de bid pour segmenter l'analyse ─────────────────────────────
BID_BUCKETS = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60),
               (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]


def bid_bucket_label(bid):
    for lo, hi in BID_BUCKETS:
        if lo <= bid < hi:
            return f"[{lo:.2f},{hi:.2f})"
    return "other"


def simulate_contract_bid_events(c, cfg, cap=0.75):
    """
    Rejoue le contrat tick à tick (version simplifiée).
    Retourne une liste d'événements bid_too_low avec le contexte.
    """
    slope     = cfg['slope']
    intercept = cfg.get('intercept', 0.0)
    max_orders = cfg.get('max_orders', 2)

    ce_ns  = np.datetime64(c['ce'])
    spots  = c['spot']
    ts_arr = c['ts']
    op     = float(c['op'])
    up_bid   = c['up_bid']
    down_bid = c['down_bid']
    up_ask   = c['up_ask']
    down_ask = c['down_ask']

    activated_side = None
    active_order = active_size = active_ord_side = None
    pending_place = pending_size = pending_ord_side = None
    pending_cancel = False
    fc_up = fc_down = 0

    bid_too_low_events = []  # (remain, bid, side, delta, thresh)
    triggered = False

    for j in range(len(spots)):
        s      = float(spots[j])
        remain = float((ce_ns - ts_arr[j]) / np.timedelta64(1, 's'))
        if remain <= 0 or (fc_up >= max_orders and fc_down >= max_orders):
            break

        ub = round(float(up_bid[j]),   2)
        db = round(float(down_bid[j]), 2)
        ua = round(float(up_ask[j]),   2)
        da = round(float(down_ask[j]), 2)

        thresh = slope * remain + intercept

        # Fill / cancel logic (simplifié, idem backtest_contract)
        filled_now = False
        if pending_cancel and active_order is not None:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                sh = active_size / active_order
                if active_ord_side == 'UP': fc_up += 1
                else:                        fc_down += 1
                filled_now = True
            active_order = active_size = active_ord_side = None
            pending_cancel = False

        if pending_place is not None:
            fc_pend = fc_up if pending_ord_side == 'UP' else fc_down
            if fc_pend < max_orders:
                active_order, active_size, active_ord_side = pending_place, pending_size, pending_ord_side
            pending_place = pending_size = pending_ord_side = None

        if active_order is not None and active_ord_side and not filled_now:
            cb2 = ub if active_ord_side == 'UP' else db
            if cb2 <= round(active_order - 0.01, 2):
                sh = active_size / active_order
                if active_ord_side == 'UP': fc_up += 1
                else:                        fc_down += 1
                active_order = active_size = active_ord_side = None
                filled_now = True

        deactivated = False
        if activated_side and not filled_now:
            buf = (s - op) if activated_side == 'UP' else (op - s)
            if buf < thresh:
                if active_order is not None:
                    pending_cancel = True
                pending_place = pending_size = pending_ord_side = None
                activated_side = None
                deactivated = True

        if not activated_side and not deactivated:
            delta_up   = s - op
            delta_down = op - s
            if fc_up < max_orders and delta_up >= thresh:
                activated_side = 'UP'
                triggered = True
            elif fc_down < max_orders and delta_down >= thresh:
                activated_side = 'DOWN'
                triggered = True

        if activated_side:
            fc_cur = fc_up if activated_side == 'UP' else fc_down
            if fc_cur >= max_orders:
                activated_side = None

        if activated_side and not filled_now:
            bid = ub if activated_side == 'UP' else db
            if bid < cap:
                bid_too_low_events.append({
                    'remain': remain,
                    'bid': bid,
                    'side': activated_side,
                    'delta': (s - op) if activated_side == 'UP' else (op - s),
                    'thresh': thresh,
                    'j': j,
                    # bid opposé (pour checker la divergence Chainlink/Binance)
                    'opp_bid': db if activated_side == 'UP' else ub,
                })

    return bid_too_low_events, triggered


def main():
    print("Chargement BTC.csv…")
    cts = load_contracts(BTC_CSV, TF, *TF_COLS)
    precompute_vol(cts, VOL_LBS)
    print(f"  {len(cts)} contrats 5min BTC chargés")

    print("Chargement settlement.csv…")
    sett = pd.read_csv(SETT_CSV)
    sett_btc = sett[(sett['asset'] == 'btc') & (sett['tf'] == '5min')].copy()
    sett_btc['ce_ts'] = sett_btc['ce_ts'].astype(int)
    sett_map = dict(zip(sett_btc['ce_ts'], sett_btc['outcome_up']))
    print(f"  {len(sett_map)} settlements BTC 5min")

    # Filtre vol (comme la config live)
    vol_lb   = CFG['vol_lb_h']
    vol_type = CFG['vol_type']
    vol_thr  = CFG['vol_thresh']
    cap      = CFG['eq_cap']

    records = []
    n_blocked_vol = 0
    n_no_trigger  = 0
    n_no_sett     = 0
    n_analyzed    = 0

    for c in cts:
        ce_ts = int(c['ce_ts'])
        outcome_up = sett_map.get(ce_ts)
        if outcome_up is None:
            n_no_sett += 1
            continue

        # Filtre vol
        vol_val = float(c.get(f'vol_{vol_lb:g}h_{vol_type}', 0.0) or 0.0)
        if vol_val < vol_thr:
            n_blocked_vol += 1
            continue

        events, triggered = simulate_contract_bid_events(c, CFG, cap=cap)

        if not triggered:
            n_no_trigger += 1
            continue

        n_analyzed += 1
        if not events:
            continue

        # Vrai outcome Chainlink
        won_up = bool(outcome_up > 0.5)

        for ev in events:
            ev['ce_ts']    = ce_ts
            ev['won_up']   = won_up
            ev['triggered_side_won'] = (
                (ev['side'] == 'UP' and won_up) or
                (ev['side'] == 'DOWN' and not won_up)
            )
            ev['bid_bucket'] = bid_bucket_label(ev['bid'])
            ev['opp_bid_bucket'] = bid_bucket_label(ev['opp_bid'])
            records.append(ev)

    print(f"\n  Contrats analysés (vol ok + trigger) : {n_analyzed}")
    print(f"  Contrats filtrés vol                 : {n_blocked_vol}")
    print(f"  Contrats sans settlement             : {n_no_sett}")
    print(f"  Contrats sans trigger                : {n_no_trigger}")
    print(f"  Événements bid_too_low total         : {len(records)}")

    if not records:
        print("\nAucun événement bid_too_low — cap peut-être mal configurée.")
        return

    df = pd.DataFrame(records)

    # ── 1. Winrate global par bucket de bid ─────────────────────────────
    print("\n" + "═" * 70)
    print("  WINRATE DU CÔTÉ TRIGGERÉ selon le niveau de bid (settlement Chainlink)")
    print("═" * 70)
    print(f"  {'Bucket bid':18} {'N events':>9} {'N contrats':>10} {'Winrate':>9} {'Bid moyen':>10} {'Remain moyen':>13}")
    print("  " + "─" * 68)

    by_bucket = df.groupby('bid_bucket')
    for label, grp in sorted(by_bucket, key=lambda x: x[0]):
        n_ev      = len(grp)
        n_cts     = grp['ce_ts'].nunique()
        winrate   = grp['triggered_side_won'].mean()
        bid_mean  = grp['bid'].mean()
        rem_mean  = grp['remain'].mean()
        mark = " ← CAP" if label == f"[{cap:.2f},1.01)" else ""
        print(f"  {label:18}  {n_ev:9d}  {n_cts:10d}  {winrate:8.1%}  {bid_mean:10.3f}  {rem_mean:12.2f}s{mark}")

    # ── 2. Quand bid est entre 0.50 et 0.75, quelle est la divergence opp_bid ? ──
    print("\n" + "═" * 70)
    print("  DIVERGENCE : bid_side vs bid_opposé (quand bid_side est entre 0.50–0.75)")
    print("  → bid_opposé élevé = marché pense que c'est l'autre sens (vrai risque Chainlink)")
    print("═" * 70)
    mask_zone = (df['bid'] >= 0.50) & (df['bid'] < 0.75)
    zone = df[mask_zone].copy()
    if len(zone):
        zone['opp_high'] = zone['opp_bid'] >= 0.80
        print(f"  N événements dans la zone 0.50–0.75 : {len(zone)}")
        print(f"  bid_opposé >= 0.80 (fort risque Chainlink diverge) : "
              f"{zone['opp_high'].sum()} ({zone['opp_high'].mean():.1%})")
        print(f"  bid_opposé <  0.80 (faible divergence)              : "
              f"{(~zone['opp_high']).sum()} ({(~zone['opp_high']).mean():.1%})")
        print()
        for lbl, grp in zone.groupby('opp_high'):
            tag = "opp >= 0.80" if lbl else "opp < 0.80"
            print(f"  [{tag}]  N={len(grp):5d}  winrate={grp['triggered_side_won'].mean():.1%}  "
                  f"bid_side_moy={grp['bid'].mean():.3f}  remain_moy={grp['remain'].mean():.2f}s")

    # ── 3. Winrate selon remain au moment du blocage ─────────────────────
    print("\n" + "═" * 70)
    print("  WINRATE selon le temps restant (remain) au moment du bid_too_low")
    print("═" * 70)
    bins = [0, 5, 15, 30, 60, 120, 300, 9999]
    labels_r = ['0-5s', '5-15s', '15-30s', '30-60s', '60-120s', '120-300s', '>300s']
    df['remain_bucket'] = pd.cut(df['remain'], bins=bins, labels=labels_r, right=False)
    for lbl, grp in df.groupby('remain_bucket', observed=True):
        n_ev    = len(grp)
        winrate = grp['triggered_side_won'].mean()
        bid_m   = grp['bid'].mean()
        print(f"  {str(lbl):12}  N={n_ev:7d}  winrate={winrate:.1%}  bid_moy={bid_m:.3f}")

    # ── 4. Proposition : seuil conditionnel ─────────────────────────────
    print("\n" + "═" * 70)
    print("  PROPOSITION SEUIL CONDITIONNEL")
    print("  → Chase si (bid >= cap) OU (bid >= low_cap ET opp_bid < 0.70)")
    print("═" * 70)
    LOW_CAP = 0.55
    OPP_THRESH = 0.70

    df['would_chase_new'] = (
        (df['bid'] >= cap) |
        ((df['bid'] >= LOW_CAP) & (df['opp_bid'] < OPP_THRESH))
    )
    # Événements actuellement bloqués mais qui passeraient avec la nouvelle règle
    newly_allowed = df[~(df['bid'] >= cap) & df['would_chase_new']]
    still_blocked = df[~df['would_chase_new']]

    if len(newly_allowed):
        wr_new = newly_allowed['triggered_side_won'].mean()
        print(f"  Événements débloqués par la nouvelle règle : {len(newly_allowed)}")
        print(f"    → Winrate Chainlink : {wr_new:.1%}")
        print(f"    → bid moyen         : {newly_allowed['bid'].mean():.3f}")
        print(f"    → opp_bid moyen     : {newly_allowed['opp_bid'].mean():.3f}")
    if len(still_blocked):
        wr_sb = still_blocked['triggered_side_won'].mean()
        print(f"  Événements encore bloqués                  : {len(still_blocked)}")
        print(f"    → Winrate Chainlink : {wr_sb:.1%}  (ces blocages sont justifiés)")

    # ── 5. Scan de différentes combinaisons low_cap / opp_thresh ────────
    print("\n" + "═" * 70)
    print("  SCAN : winrate des cas débloqués selon (low_cap, opp_thresh)")
    print(f"  {'low_cap':>8} {'opp_thresh':>10} {'N_débloq':>10} {'Winrate':>9}")
    print("  " + "─" * 42)
    for lc in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        for ot in [0.60, 0.65, 0.70, 0.75, 0.80]:
            mask = (df['bid'] < cap) & (df['bid'] >= lc) & (df['opp_bid'] < ot)
            sub = df[mask]
            if len(sub) == 0:
                continue
            wr = sub['triggered_side_won'].mean()
            print(f"  {lc:8.2f}  {ot:10.2f}  {len(sub):10d}  {wr:8.1%}")

    print("\n  Note : 'Winrate' = % de fois où le côté triggeré (Binance) gagne")
    print("         selon settlement.csv (résolution Chainlink réelle).")
    print()


if __name__ == '__main__':
    main()
