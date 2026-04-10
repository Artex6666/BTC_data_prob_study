import pandas as pd
import numpy as np

BTC_FILE = r"c:\Users\Artex\Desktop\BtcUpDownStudy\directionnalTrading\BTC_BIG.csv"

def sweep_intercepts():
    file_path = "BTC.csv"
    print(f"Loading {file_path} (M5)...", flush=True)
    df = pd.read_csv(file_path, usecols=['timestamp', 'spot_price', 'm5_up_bid', 'm5_down_bid'])
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna().sort_values('timestamp')
    
    df['ce'] = df['timestamp'].dt.floor('5min') + pd.Timedelta(minutes=5)
    
    spot = df['spot_price'].values
    ts = df['timestamp'].values
    up = df['m5_up_bid'].values
    dn = df['m5_down_bid'].values
    ce_arr = df['ce'].values
    
    unique_ces = np.unique(ce_arr)
    n_contracts = len(unique_ces)
    
    intercepts = [15, 20, 25, 30, 40, 50]
    
    # Pre-extract necessary contract slices to make the sweep fast
    contracts = []
    for ce in unique_ces:
        idxs = np.where(ce_arr == ce)[0]
        if len(idxs) < 5: continue
        
        first, last = idxs[0], idxs[-1]
        op = spot[first]
        won_up = up[last] > dn[last]
        
        # Look back 60 seconds for God Curve
        t_cutoff = ce - np.timedelta64(60, 's')
        lb_mask = (ce_arr == ce) & (ts >= t_cutoff)
        lb_idxs = np.where(lb_mask)[0]
        if len(lb_idxs) == 0: continue
        
        contracts.append({
            'ce': ce,
            'op': op,
            'won_up': won_up,
            'spots': spot[lb_idxs].tolist(),
            'ups': up[lb_idxs].tolist(),
            'dns': dn[lb_idxs].tolist(),
            'secs': ((ce - ts[lb_idxs]) / np.timedelta64(1, 's')).tolist()
        })
        
    print(f"Extracted {len(contracts)} valid M5 contracts. Running PNL Sweep for 3.0x + $15.00...", flush=True)
    print("="*90)
    print(f"{'Strategy Variant':<25} | {'WR':<7} | {'Fills':<6} | {'Trades':<8} | {'PNL (Scale 50)':<15}")
    print("-" * 90)
    
    variants = [("NO CANCEL (Old Logic)", False), ("DYNAMIC CANCEL (New Logic)", True)]
    
    for variant_name, use_cancel in variants:
        wins, losses, fills = 0, 0, 0
        total_cost, total_payout = 0.0, 0.0
        trades_executed = 0
        
        for c in contracts:
            activated_side = None
            our_limit = None
            fill_count = 0
            contract_cost = 0.0
            
            t_spots = c['spots']
            t_ups = c['ups']
            t_dns = c['dns']
            t_secs = c['secs']
            op = c['op']
            
            for j in range(len(t_secs)):
                s = t_spots[j]
                secs = t_secs[j]
                if secs <= 0: break
                
                threshold = (3.0 * secs) + 15.0
                
                # Dynamic Cancellation Logic
                if activated_side and use_cancel:
                    current_buf = (s - op) if activated_side == 'UP' else (op - s)
                    if current_buf < threshold:
                        our_limit = None
                        activated_side = None
                
                if not activated_side:
                    if s - op >= threshold:
                        activated_side = 'UP'
                    elif op - s >= threshold:
                        activated_side = 'DOWN'
                
                if not activated_side: continue
                
                # Maker Chaser Fill Simulator
                current_bid = round(float(t_ups[j] if activated_side == 'UP' else t_dns[j]), 2)
                
                if our_limit is not None and current_bid <= round(our_limit - 0.01, 2):
                    fill_count += 1
                    contract_cost += 50.0 * our_limit  # Scale 50 shares
                    our_limit = None
                    if fill_count >= 3: break
                
                if current_bid >= 0.90:
                    proposed = min(current_bid, 0.97) # STRICT CAP 97c
                    if our_limit is None or proposed > our_limit:
                        our_limit = proposed
            
            if fill_count > 0:
                trades_executed += 1
                fills += fill_count
                won = c['won_up'] if activated_side == 'UP' else (not c['won_up'])
                
                total_cost += contract_cost
                if won:
                    wins += 1
                    total_payout += (50.0 * fill_count) * 1.00 # $1 payout per share
                else:
                    losses += 1
                    
        wr = (wins / trades_executed * 100.0) if trades_executed > 0 else 0.0
        pnl = total_payout - total_cost
        
        res = f"{variant_name:<25} | {wr:>6.2f}% | {fills:<6} | {trades_executed} (W:{wins} L:{losses}) | ${pnl:,.2f}\n"
        print(res, end="")
        with open("sweep_god_out.txt", "a") as f:
            f.write(res)
        
if __name__ == '__main__':
    sweep_intercepts()
