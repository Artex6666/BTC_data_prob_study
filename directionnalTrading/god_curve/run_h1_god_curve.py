import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = r"c:\Users\Artex\Desktop\BtcUpDownStudy\directionnalTrading\BTC_BIG.csv"
TRADE_BASE_SHARES = 1000

def get_required_buffer_pct_h1(secs_left):
    if secs_left <= 1: return 19.5
    if secs_left <= 5: return 19.5
    if secs_left <= 15: return 19.5
    if secs_left <= 30: return 19.5
    if secs_left <= 60: return 63.3
    return 999.0

def backtest_linear_chaser_h1(df):
    trades = []
    tf_col = 'h1_up_bid'
    
    df_temp = df.dropna(subset=[tf_col]).copy()
    if df_temp.empty: return
    
    df_temp['contract_start'] = df_temp['timestamp'].dt.floor('h')
    df_temp['contract_end'] = df_temp['contract_start'] + timedelta(minutes=60)
    
    print("\nSimulating H1 Linear God Curve Chaser (T-60s to T-0s)...")
    
    unique_contracts = df_temp['contract_end'].unique()
    for contract_end in unique_contracts:
        group = df_temp[df_temp['contract_end'] == contract_end]
        if group.empty: continue
        
        open_price = group.iloc[0]['spot_price']
        target_timestamp = contract_end - timedelta(seconds=60)
        late_game = group[group['timestamp'] >= target_timestamp]
        
        if late_game.empty: continue
        
        our_limit = None
        fill_price = None
        activated = False
        
        for _, row in late_game.iterrows():
            current_bid = round(row[tf_col], 2)
            current_spot = row['spot_price']
            secs_left = (contract_end - row['timestamp']).total_seconds()
            if secs_left <= 0: break
            
            vol_range = row['spot_1h_range']
            if pd.isna(vol_range) or vol_range < 50:
                vol_range = 300.0
                
            actual_buffer = current_spot - open_price
            actual_pct = (actual_buffer / vol_range) * 100.0
            
            if not activated:
                req_pct = get_required_buffer_pct_h1(secs_left)
                if actual_pct >= req_pct:
                    activated = True
                    
            if not activated:
                continue
                
            if our_limit is not None and current_bid <= round(our_limit - 0.01, 2):
                fill_price = our_limit
                break
                
            if current_bid >= 0.90:
                proposed_limit = min(current_bid, 0.99)
                if our_limit is None or proposed_limit > our_limit:
                    our_limit = proposed_limit
                    
        if fill_price is not None:
            won = group.iloc[-1][tf_col] > 0.50
            cost_dollars = TRADE_BASE_SHARES * fill_price
            payout = TRADE_BASE_SHARES if won else 0.0
            pnl = payout - cost_dollars
            trades.append({'won': won, 'pnl': pnl, 'cost': fill_price})
            
    if not trades:
        print("\n[H1] No trades triggered.")
        return
        
    df_res = pd.DataFrame(trades)
    wins = df_res['won'].sum()
    losses = len(trades) - wins
    win_rate = (wins / len(trades)) * 100
    pnl = df_res['pnl'].sum()
    
    print("\n================= LINEAR GOD CURVE CHASER (H1) =================")
    print(f"Total Trades : {len(trades)}")
    print(f"Wins / Losses: {wins}W / {losses}L")
    print(f"Win Rate     : {win_rate:.2f}%")
    print(f"Absolute PNL : ${pnl:,.2f}")
    print("=================================================================\n")

def main():
    print("Loading BTC_BIG.csv for H1...")
    try:
        df = pd.read_csv(DATA_FILE, usecols=['timestamp', 'spot_price', 'h1_up_bid'], nrows=8000000)
    except Exception as e:
        print(f"Loading error: {e}")
        return
        
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
    df = df.dropna(subset=['spot_price'])
    df = df.sort_values('timestamp')
    
    print("Computing 1-Hour Rolling Volatility...")
    df.set_index('timestamp', inplace=True)
    df['spot_1h_range'] = df['spot_price'].rolling('1h').max() - df['spot_price'].rolling('1h').min()
    df.reset_index(inplace=True)
    
    backtest_linear_chaser_h1(df)

if __name__ == '__main__':
    main()
