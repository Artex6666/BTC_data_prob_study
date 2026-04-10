import pandas as pd
import numpy as np

def run_tf_backtest(df, tf_name, up_col, dn_col, freq_mins):
    print(f"--- ANALYZING {tf_name} ({freq_mins}min) ---", flush=True)
    
    # Define contract boundaries
    df['ce'] = df['timestamp'].dt.floor(f'{freq_mins}min') + pd.Timedelta(minutes=freq_mins)
    
    spot = df['spot_price'].values
    ts = df['timestamp'].values
    up = df[up_col].values
    dn = df[dn_col].values
    ce_arr = df['ce'].values
    
    unique_ces = np.unique(ce_arr)
    contracts = []
    
    for ce in unique_ces:
        idxs = np.where(ce_arr == ce)[0]
        if len(idxs) < 5: continue
        
        first, last = idxs[0], idxs[-1]
        op = spot[first]
        won_up = up[last] > dn[last]
        
        # Look back 60 seconds (or more if you want, but sticking to 60s for God Curve consistency)
        t_cutoff = ce - np.timedelta64(60, 's')
        lb_mask = (ce_arr == ce) & (ts >= t_cutoff)
        lb_idxs = np.where(lb_mask)[0]
        if len(lb_idxs) == 0: continue
        
        contracts.append({
            'op': op,
            'won_up': won_up,
            'spots': spot[lb_idxs],
            'ups': up[lb_idxs],
            'dns': dn[lb_idxs],
            'secs': (ce - ts[lb_idxs]) / np.timedelta64(1, 's')
        })
    
    # Simulation Parameters
    use_cancel = True
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
            
            threshold = (3.0 * secs) + 15.0 # UNIVERSAL GOD EQUATION
            
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
                contract_cost += 50.0 * our_limit 
                our_limit = None
                if fill_count >= 3: break
            
            if current_bid >= 0.85: # USER SPECIFIED CAP 85+
                proposed = min(current_bid, 0.97) # MAX CHASE 97c
                if our_limit is None or proposed > our_limit:
                    our_limit = proposed
        
        if fill_count > 0:
            trades_executed += 1
            fills += fill_count
            won = c['won_up'] if activated_side == 'UP' else (not c['won_up'])
            total_cost += contract_cost
            if won:
                wins += 1
                total_payout += (50.0 * fill_count) * 1.00 
            else:
                losses += 1
                
    wr = (wins / trades_executed * 100.0) if trades_executed > 0 else 0.0
    pnl = total_payout - total_cost
    
    res = f"{tf_name:<10} | {wr:>6.2f}% | {fills:<6} | {trades_executed} (W:{wins} L:{losses}) | ${pnl:,.2f}"
    print(res, flush=True)
    return res

def main():
    file_path = "BTC.csv"
    print(f"Loading {file_path} for 3-TF Sweep...", flush=True)
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna().sort_values('timestamp')
    
    results = []
    results.append(run_tf_backtest(df.copy(), "M5", "m5_up_bid", "m5_down_bid", 5))
    results.append(run_tf_backtest(df.copy(), "M15", "m15_up_bid", "m15_down_bid", 15))
    results.append(run_tf_backtest(df.copy(), "H1", "h1_up_bid", "h1_down_bid", 60))
    
    print("\n" + "="*80)
    print(f"{'TF':<10} | {'WR':<7} | {'Fills':<6} | {'Trades':<8} | {'PNL (Scale 50)'}")
    print("-" * 80)
    for r in results:
        print(r)
    print("="*80)

if __name__ == '__main__':
    main()
