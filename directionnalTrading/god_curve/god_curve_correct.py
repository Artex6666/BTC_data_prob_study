import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

BTC_FILE = r"c:\Users\Artex\Desktop\BtcUpDownStudy\directionnalTrading\BTC_BIG.csv"
SHARES = 1000

def get_req(secs):
    if secs <= 1: return 8.0
    if secs <= 5: return 21.0
    if secs <= 10: return 23.0
    if secs <= 15: return 27.0
    if secs <= 20: return 29.0
    if secs <= 30: return 29.0
    if secs <= 45: return 40.0
    return 42.0

def run_god(ts_arr, spot_arr, up_arr, dn_arr, tr_arr, ce_arr, opens_d, res_d, tr_d, unique_ces, label, max_fills=3):
    all_trades = []
    
    for ce in unique_ces:
        if ce not in res_d: continue
        op = opens_d[ce]; tr = tr_d[ce]
        if np.isnan(tr) or tr < 1: continue
        up_won = res_d[ce]
        
        # Get indices for this contract in the last 60 seconds
        ce_ns = ce
        t60_ns = ce_ns - np.timedelta64(60, 's')
        
        mask = (ce_arr == ce_ns) & (ts_arr >= t60_ns)
        idxs = np.where(mask)[0]
        if len(idxs) == 0: continue
        
        for side in ['UP', 'DOWN']:
            bid_arr = up_arr if side == 'UP' else dn_arr
            won = up_won if side == 'UP' else (not up_won)
            
            lim = None; fills = 0; cost = 0.0; act = False
            for i in idxs:
                bid = round(float(bid_arr[i]), 2)
                spot = float(spot_arr[i])
                secs = (ce_ns - ts_arr[i]) / np.timedelta64(1, 's')
                if secs <= 0: break
                
                buf = (spot - op) if side == 'UP' else (op - spot)
                bpct = (buf / tr) * 100.0
                
                if not act:
                    if bpct >= get_req(secs): act = True
                if not act: continue
                
                if lim is not None and bid <= round(lim - 0.01, 2):
                    fills += 1; cost += SHARES * lim
                    if fills >= max_fills: break
                    lim = None
                if bid >= 0.90:
                    p = min(bid, 0.99)
                    if lim is None or p > lim: lim = p
                    
            if fills > 0:
                pay = SHARES * fills if won else 0.0
                all_trades.append({'won': won, 'pnl': pay - cost, 'fills': fills, 'side': side, 'ce': str(ce)})
    
    for side in ['UP', 'DOWN', 'BOTH']:
        subset = [t for t in all_trades if (side == 'BOTH' or t['side'] == side)]
        if not subset:
            print(f"| {label:<18} | {side:<5} |    0 contracts |", flush=True)
            continue
        d = pd.DataFrame(subset)
        w = int(d['won'].sum()); l = len(subset) - w
        wr = (w/len(subset))*100; pnl = d['pnl'].sum()
        fills = d['fills'].sum(); worst = d['pnl'].min()
        print(f"| {label:<18} | {side:<5} | {len(subset):>4} contracts | {w}W/{l}L ({wr:>5.1f}%) | ${pnl:>8,.0f} | fills:{fills:>5} | worst: ${worst:>7,.0f} |", flush=True)
        
        losses = d[d['won'] == False]
        for _, lo in losses.iterrows():
            print(f"    *** LOSS: {lo['ce']} | {lo['side']} | fills={lo['fills']} | pnl=${lo['pnl']:,.0f}", flush=True)

def main():
    print("="*120, flush=True)
    print("GOD CURVE — UP+DOWN — TR METHOD COMPARISON (Cap 3)", flush=True)
    print("="*120, flush=True)
    
    print("\nLoading BTC_BIG.csv...", flush=True)
    cols = ['timestamp','spot_price','m5_up_bid','m5_down_bid']
    df = pd.read_csv(BTC_FILE, usecols=cols, nrows=4500000)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna(subset=['spot_price','m5_up_bid','m5_down_bid']).sort_values('timestamp')
    print(f"Loaded {len(df)} clean rows.", flush=True)
    
    # Rolling 1H Range
    print("Computing Rolling 1H Range...", flush=True)
    df.set_index('timestamp', inplace=True)
    df['tr_rolling'] = df['spot_price'].rolling('1h').max() - df['spot_price'].rolling('1h').min()
    df.reset_index(inplace=True)
    
    # Previous H1 candle HL
    print("Computing H1 Candle TR...", flush=True)
    df['h1f'] = df['timestamp'].dt.floor('h')
    h1 = df.groupby('h1f')['spot_price'].agg(['max','min']).reset_index()
    h1['tr_c'] = h1['max'] - h1['min']
    h1['tr_candle_prev'] = h1['tr_c'].shift(1)
    h1 = h1[['h1f','tr_candle_prev']].dropna()
    df = df.merge(h1, on='h1f', how='left')
    df = df.drop(columns=['h1f'])
    
    # Floor to 5min
    df['ce'] = df['timestamp'].dt.floor('5min') + timedelta(minutes=5)
    
    # Pre-compute per-contract data
    unique_ces = df['ce'].unique()
    print(f"Total M5 contracts: {len(unique_ces)}", flush=True)
    
    # Convert to numpy for fast access
    ts_np = df['timestamp'].values
    spot_np = df['spot_price'].values
    up_np = df['m5_up_bid'].values
    dn_np = df['m5_down_bid'].values
    tr_roll_np = df['tr_rolling'].values
    tr_cand_np = df['tr_candle_prev'].values
    ce_np = df['ce'].values
    
    # Pre-compute opens, resolution, TR per contract
    opens_d = {}; res_d = {}; tr_roll_d = {}; tr_cand_d = {}
    for ce in unique_ces:
        mask = ce_np == ce
        idxs = np.where(mask)[0]
        if len(idxs) == 0: continue
        first = idxs[0]; last = idxs[-1]
        opens_d[ce] = float(spot_np[first])
        res_d[ce] = float(up_np[last]) > float(dn_np[last])
        tr_roll_d[ce] = float(tr_roll_np[first])
        tr_cand_d[ce] = float(tr_cand_np[first]) if not np.isnan(tr_cand_np[first]) else 0.0
    
    print(f"\nRolling TR: mean=${np.nanmean(tr_roll_np):,.0f}, median=${np.nanmedian(tr_roll_np):,.0f}", flush=True)
    print(f"Candle TR:  mean=${np.nanmean(tr_cand_np):,.0f}, median=${np.nanmedian(tr_cand_np):,.0f}", flush=True)
    
    print("\n--- METHOD 1: Rolling 1H Range ---", flush=True)
    run_god(ts_np, spot_np, up_np, dn_np, tr_roll_np, ce_np, opens_d, res_d, tr_roll_d, unique_ces, 'Rolling 1H')
    
    print("\n--- METHOD 2: Prev H1 Candle HL ---", flush=True)
    run_god(ts_np, spot_np, up_np, dn_np, tr_cand_np, ce_np, opens_d, res_d, tr_cand_d, unique_ces, 'H1 Candle Prev')

if __name__ == '__main__':
    main()
