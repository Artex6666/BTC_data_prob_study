import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

BTC_FILE = r"c:\Users\Artex\Desktop\BtcUpDownStudy\directionnalTrading\BTC_BIG.csv"
SHARES = 1000

def run_sweep(ts, spot, up_bid, dn_bid, ce_arr, opens_d, res_d, tr_d, unique_ces, configs):
    """
    configs: list of (label, pct_multiplier, min_dollar, max_time_window)
    pct_multiplier: scale factor on the base God Curve % thresholds
    """
    base_thresholds = {
        1: 8.0, 5: 21.0, 10: 23.0, 15: 27.0, 20: 29.0, 30: 29.0, 45: 40.0, 60: 42.0
    }
    
    for label, pct_mult, min_dollar, max_tw in configs:
        all_trades = []
        
        for ce in unique_ces:
            if ce not in res_d: continue
            op = opens_d[ce]; tr = tr_d[ce]
            if np.isnan(tr) or tr < 1: continue
            up_won = res_d[ce]
            
            ce_ns = ce
            t_ns = ce_ns - np.timedelta64(max_tw, 's')
            mask = (ce_arr == ce_ns) & (ts >= t_ns)
            idxs = np.where(mask)[0]
            if len(idxs) == 0: continue
            
            for side_idx, (side, bid_src, won_val) in enumerate([
                ('UP', up_bid, up_won), ('DOWN', dn_bid, not up_won)
            ]):
                lim = None; fills = 0; cost = 0.0; act = False
                
                for i in idxs:
                    bid = round(float(bid_src[i]), 2)
                    s = float(spot[i])
                    secs = (ce_ns - ts[i]) / np.timedelta64(1, 's')
                    if secs <= 0: break
                    
                    buf_usd = (s - op) if side == 'UP' else (op - s)
                    buf_pct = (buf_usd / tr) * 100.0
                    
                    if not act:
                        # Find applicable threshold
                        req = 100.0  # default: impossible
                        for t_key in sorted(base_thresholds.keys()):
                            if secs <= t_key:
                                req = base_thresholds[t_key] * pct_mult
                                break
                        if secs > 60:
                            req = 42.0 * pct_mult
                        
                        if buf_pct >= req and buf_usd >= min_dollar:
                            act = True
                    if not act: continue
                    
                    if lim is not None and bid <= round(lim - 0.01, 2):
                        fills += 1; cost += SHARES * lim
                        if fills >= 3: break
                        lim = None
                    if bid >= 0.90:
                        p = min(bid, 0.99)
                        if lim is None or p > lim: lim = p
                        
                if fills > 0:
                    pay = SHARES * fills if won_val else 0.0
                    all_trades.append({'won': won_val, 'pnl': pay - cost, 'fills': fills, 'side': side})
        
        if not all_trades:
            print(f"| {label:<35} |    0 tr |  N/A     |       $0 |", flush=True)
            continue
        
        d = pd.DataFrame(all_trades)
        w = int(d['won'].sum()); l = len(all_trades) - w
        wr = (w/len(all_trades))*100; pnl = d['pnl'].sum()
        fills = d['fills'].sum()
        tag = "*** 100% ***" if l == 0 else f"{l} LOSSES"
        up_n = len([t for t in all_trades if t['side']=='UP'])
        dn_n = len([t for t in all_trades if t['side']=='DOWN'])
        print(f"| {label:<35} | {len(all_trades):>4} tr (U:{up_n}/D:{dn_n}) | {w}W/{l}L ({wr:>5.1f}%) | ${pnl:>7,.0f} | fills:{fills:>4} | {tag}", flush=True)

def main():
    print("="*130, flush=True)
    print("GOD CURVE PARAMETRIC SWEEP — BOTH SIDES — FIND 100% WR", flush=True)
    print("="*130, flush=True)
    
    print("\nLoading BTC_BIG.csv...", flush=True)
    df = pd.read_csv(BTC_FILE, usecols=['timestamp','spot_price','m5_up_bid','m5_down_bid'], nrows=4500000)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna(subset=['spot_price','m5_up_bid','m5_down_bid']).sort_values('timestamp')
    print(f"Loaded {len(df)} rows.", flush=True)
    
    # Rolling 1H Range (more conservative)
    df.set_index('timestamp', inplace=True)
    df['tr'] = df['spot_price'].rolling('1h').max() - df['spot_price'].rolling('1h').min()
    df.reset_index(inplace=True)
    
    df['ce'] = df['timestamp'].dt.floor('5min') + timedelta(minutes=5)
    unique_ces = df['ce'].unique()
    
    ts = df['timestamp'].values
    spot = df['spot_price'].values
    up = df['m5_up_bid'].values
    dn = df['m5_down_bid'].values
    tr_arr = df['tr'].values
    ce_np = df['ce'].values
    
    opens_d = {}; res_d = {}; tr_d = {}
    for ce in unique_ces:
        idxs = np.where(ce_np == ce)[0]
        if len(idxs) == 0: continue
        opens_d[ce] = float(spot[idxs[0]])
        res_d[ce] = float(up[idxs[-1]]) > float(dn[idxs[-1]])
        tr_d[ce] = float(tr_arr[idxs[0]])
    
    # (label, pct_multiplier, min_dollar, max_time_window_seconds)
    configs = [
        # Baseline (what we had before, causes losses)
        ("Base 1.0x, $0,  T60", 1.0, 0, 60),
        ("Base 1.0x, $30, T60", 1.0, 30, 60),
        ("Base 1.0x, $50, T60", 1.0, 50, 60),
        ("Base 1.0x, $75, T60", 1.0, 75, 60),
        ("Base 1.0x, $100,T60", 1.0, 100, 60),
        ("Base 1.0x, $150,T60", 1.0, 150, 60),
        # Higher % thresholds
        ("1.25x pct, $0,  T60", 1.25, 0, 60),
        ("1.25x pct, $30, T60", 1.25, 30, 60),
        ("1.25x pct, $50, T60", 1.25, 50, 60),
        ("1.25x pct, $75, T60", 1.25, 75, 60),
        ("1.5x pct,  $0,  T60", 1.5, 0, 60),
        ("1.5x pct,  $30, T60", 1.5, 30, 60),
        ("1.5x pct,  $50, T60", 1.5, 50, 60),
        ("1.5x pct,  $75, T60", 1.5, 75, 60),
        ("1.75x pct, $0,  T60", 1.75, 0, 60),
        ("1.75x pct, $30, T60", 1.75, 30, 60),
        ("1.75x pct, $50, T60", 1.75, 50, 60),
        ("2.0x pct,  $0,  T60", 2.0, 0, 60),
        ("2.0x pct,  $30, T60", 2.0, 30, 60),
        ("2.0x pct,  $50, T60", 2.0, 50, 60),
        # Shorter time windows
        ("Base 1.0x, $0,  T30", 1.0, 0, 30),
        ("Base 1.0x, $30, T30", 1.0, 30, 30),
        ("Base 1.0x, $50, T30", 1.0, 50, 30),
        ("1.25x pct, $0,  T30", 1.25, 0, 30),
        ("1.25x pct, $30, T30", 1.25, 30, 30),
        ("1.25x pct, $50, T30", 1.25, 50, 30),
        ("1.5x pct,  $0,  T30", 1.5, 0, 30),
        ("1.5x pct,  $30, T30", 1.5, 30, 30),
        ("1.5x pct,  $50, T30", 1.5, 50, 30),
        # Very conservative
        ("2.0x pct,  $0,  T30", 2.0, 0, 30),
        ("2.0x pct,  $50, T30", 2.0, 50, 30),
        ("2.5x pct,  $0,  T60", 2.5, 0, 60),
        ("2.5x pct,  $0,  T30", 2.5, 0, 30),
        ("3.0x pct,  $0,  T60", 3.0, 0, 60),
    ]
    
    run_sweep(ts, spot, up, dn, ce_np, opens_d, res_d, tr_d, unique_ces, configs)

if __name__ == '__main__':
    main()
