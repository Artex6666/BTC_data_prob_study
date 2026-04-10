"""
Analyse du lag ask → bid sur les données BTC M5.

Deux angles :
  1. Lag par contrat : pour chaque contrat M5, quand ask_up franchit le cap,
     combien de temps avant que bid_up le franchisse aussi ?
  2. Lag instantané tick-à-tick : sur les moves rapides (spot varie > X$ en 1 tick),
     mesure l'écart ask-bid et sa durée avant que bid rattrape.

Paramètres :
  CAP_LIST   : caps à tester (0.55, 0.65, 0.70, 0.75, 0.85)
  FAST_MOVE  : seuil de move rapide du spot en $/tick (défaut 5$)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass

BASE     = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "Datas" / "csv" / "BTC.csv"
CAP_LIST  = [0.55, 0.65, 0.70, 0.75, 0.85]
FAST_MOVE = 5.0   # $/tick pour qualifier un move "rapide"

# ─────────────────────────────────────────────────────────────────────────────
print("Chargement CSV...", flush=True)
df = pd.read_csv(CSV_PATH,
                 usecols=['timestamp','spot_price',
                          'm5_up_ask','m5_up_bid','m5_down_ask','m5_down_bid'])
df['ts'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
df = df.sort_values('ts').reset_index(drop=True)
print(f"  {len(df):,} ticks  |  {df['ts'].min()} → {df['ts'].max()}", flush=True)

# ─── Partie 1 : lag ask→bid par contrat M5 ───────────────────────────────────
print("\n=== 1. Lag ask→bid par contrat (cap franchi) ===", flush=True)

offset = pd.tseries.frequencies.to_offset('5min')
df['contract'] = df['ts'].dt.floor('5min') + offset

for cap in CAP_LIST:
    lags_s = []          # délais ask→bid en secondes (>0 = ask en avance)
    n_ask_only = 0       # ask franchit cap mais bid jamais
    n_bid_first = 0      # bid franchit avant ask (rare)
    n_both = 0

    for ce, grp in df.groupby('contract'):
        # Fenêtre : 60s avant la clôture
        t60 = grp[grp['ts'] >= ce - pd.Timedelta(seconds=60)]
        if len(t60) < 2:
            continue

        ts_arr  = t60['ts'].values
        ask_up  = t60['m5_up_ask'].values
        bid_up  = t60['m5_up_bid'].values

        # Premier franchissement de cap
        ask_cross = np.where(ask_up >= cap)[0]
        bid_cross = np.where(bid_up >= cap)[0]

        if len(ask_cross) == 0 and len(bid_cross) == 0:
            continue

        if len(ask_cross) > 0 and len(bid_cross) == 0:
            n_ask_only += 1
            continue

        if len(ask_cross) == 0 and len(bid_cross) > 0:
            n_bid_first += 1
            continue

        n_both += 1
        t_ask = ts_arr[ask_cross[0]]
        t_bid = ts_arr[bid_cross[0]]
        lag = (t_bid - t_ask) / np.timedelta64(1, 's')
        lags_s.append(lag)

    if not lags_s:
        print(f"  cap={cap:.2f} : aucune donnée")
        continue

    lags = np.array(lags_s)
    pos  = lags[lags > 0]   # ask devant bid
    neg  = lags[lags < 0]   # bid devant ask
    zero = lags[lags == 0]

    print(f"\n  cap={cap:.2f}  |  contrats : both={n_both}  ask_seul={n_ask_only}  bid_1er={n_bid_first}")
    print(f"    ask avant bid : {len(pos)} ({100*len(pos)/len(lags):.0f}%)")
    print(f"    simultané     : {len(zero)} ({100*len(zero)/len(lags):.0f}%)")
    print(f"    bid avant ask : {len(neg)} ({100*len(neg)/len(lags):.0f}%)")
    if len(pos) > 0:
        print(f"    lag ask→bid (ask en avance) : "
              f"p25={np.percentile(pos,25):.2f}s  "
              f"med={np.median(pos):.2f}s  "
              f"p75={np.percentile(pos,75):.2f}s  "
              f"p90={np.percentile(pos,90):.2f}s  "
              f"p99={np.percentile(pos,99):.2f}s  "
              f"max={pos.max():.2f}s")
    if len(neg) > 0:
        print(f"    lag bid→ask (bid en avance) : "
              f"med={np.median(-neg):.2f}s  max={(-neg).max():.2f}s")

# ─── Partie 2 : lag tick-à-tick sur les moves rapides ────────────────────────
print("\n\n=== 2. Lag tick-à-tick sur moves rapides (spot delta > {:.0f}$) ===".format(FAST_MOVE), flush=True)

spot   = df['spot_price'].values
ts_ns  = df['ts'].values
ask_up = df['m5_up_ask'].values
bid_up = df['m5_up_bid'].values

dspot = np.abs(np.diff(spot))

# Ticks avec move rapide UP (spot monte > FAST_MOVE)
fast_up = np.where(np.diff(spot) >= FAST_MOVE)[0]   # index i → move entre i et i+1
print(f"  Moves rapides UP (>={FAST_MOVE}$) : {len(fast_up)}", flush=True)

# Pour chaque move rapide : mesure spread ask_up - bid_up pendant les 30 ticks suivants
WINDOW = 50  # ticks après le move

spreads_at_move = []
ticks_to_close  = []   # nb de ticks pour que spread revienne < 0.02

for i in fast_up:
    if i + 1 >= len(ask_up):
        continue
    end = min(i + WINDOW + 1, len(ask_up))
    a = ask_up[i+1:end]
    b = bid_up[i+1:end]
    sp = a - b

    spreads_at_move.append(float(sp[0]))

    # Temps (en secondes) pour que bid rattrape ask (spread < 0.02)
    closed = np.where(sp < 0.02)[0]
    if len(closed) > 0:
        dt = (ts_ns[i + 1 + closed[0]] - ts_ns[i + 1]) / np.timedelta64(1, 's')
        ticks_to_close.append(float(dt))
    # sinon: bid n'a pas rattrapé dans la fenêtre

sp_arr  = np.array(spreads_at_move)
tc_arr  = np.array(ticks_to_close)

print(f"  Spread ask-bid AU moment du move rapide :")
print(f"    p25={np.percentile(sp_arr,25):.3f}  med={np.median(sp_arr):.3f}  "
      f"p75={np.percentile(sp_arr,75):.3f}  p90={np.percentile(sp_arr,90):.3f}  "
      f"max={sp_arr.max():.3f}")

print(f"\n  Temps (s) pour que bid rattrape ask (spread < 0.02) après move rapide :")
print(f"    Rattrapés dans la fenêtre : {len(tc_arr)}/{len(fast_up)} "
      f"({100*len(tc_arr)/max(len(fast_up),1):.0f}%)")
if len(tc_arr) > 0:
    print(f"    p25={np.percentile(tc_arr,25):.2f}s  med={np.median(tc_arr):.2f}s  "
          f"p75={np.percentile(tc_arr,75):.2f}s  p90={np.percentile(tc_arr,90):.2f}s  "
          f"max={tc_arr.max():.2f}s")

# Breakdown par amplitude du move
print("\n  Breakdown par amplitude du move spot :")
buckets = [(5,10),(10,20),(20,50),(50,200)]
for lo, hi in buckets:
    idx = np.where((np.diff(spot) >= lo) & (np.diff(spot) < hi))[0]
    if len(idx) == 0:
        continue
    sps = [float(ask_up[i+1] - bid_up[i+1]) for i in idx if i+1 < len(ask_up)]
    print(f"    move ${lo}-${hi} : n={len(sps)}  spread_med={np.median(sps):.3f}  spread_p90={np.percentile(sps,90):.3f}")

# ─── Partie 3 : gap ask-bid en fonction du spread actuel ─────────────────────
print("\n\n=== 3. Distribution du spread ask-bid (tous ticks) ===", flush=True)
all_spread = df['m5_up_ask'].values - df['m5_up_bid'].values
all_spread = all_spread[~np.isnan(all_spread)]
print(f"  p1={np.percentile(all_spread,1):.3f}  p25={np.percentile(all_spread,25):.3f}  "
      f"med={np.median(all_spread):.3f}  p75={np.percentile(all_spread,75):.3f}  "
      f"p90={np.percentile(all_spread,90):.3f}  p99={np.percentile(all_spread,99):.3f}")
print(f"  Spread=0.01 : {100*(all_spread<=0.01).mean():.1f}%  "
      f"Spread=0.02 : {100*(all_spread<=0.02).mean():.1f}%  "
      f"Spread>=0.05 : {100*(all_spread>=0.05).mean():.1f}%")
