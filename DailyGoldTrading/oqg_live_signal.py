import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

PORTFOLIO_FILE = "oqg_virtual_portfolio.csv"
STARTING_CAPITAL = 100000.0
SLIPPAGE_FEE = 0.001  # 0.1% cost to switch positions (slippage + commissions)

def fetch_market_data():
    """Fetches live futures data and calculates indicators."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=500)
    data = yf.download(['GC=F', 'SI=F', 'HG=F'], start=start_date, end=end_date, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'].copy() if 'Close' in data.columns.levels[0] else data['Adj Close'].copy()
    else:
        df = data.copy()

    df.rename(columns={'GC=F': 'Gold', 'HG=F': 'Copper', 'SI=F': 'Silver'}, inplace=True)
    df.ffill(inplace=True)

    df['GCR'] = df['Gold'] / df['Copper']
    df['GSR'] = df['Gold'] / df['Silver']
    df['GCR_Delta'] = df['GCR'].diff()

    window = 252
    df['GCR_Z_Score'] = (df['GCR'] - df['GCR'].rolling(window).mean()) / df['GCR'].rolling(window).std()
    df['GSR_Z_Score'] = (df['GSR'] - df['GSR'].rolling(window).mean()) / df['GSR'].rolling(window).std()
    
    return df.dropna()

def run_portfolio_engine():
    today_str = datetime.now().strftime('%Y-%m-%d')
    df = fetch_market_data()
    
    # 1. Get today's market reality
    current_gold = df['Gold'].iloc[-1]
    current_silver = df['Silver'].iloc[-1]
    current_gsr_z = df['GSR_Z_Score'].iloc[-1]
    current_gcr_delta = df['GCR_Delta'].iloc[-1]
    
    # 2. Calculate the Target Signal
    target_asset = "SILVER" if (current_gsr_z > 1.5 and current_gcr_delta < 0) else "GOLD"
    current_price = current_silver if target_asset == "SILVER" else current_gold

    # 3. Load or Initialize Portfolio
    if os.path.exists(PORTFOLIO_FILE):
        port_df = pd.read_csv(PORTFOLIO_FILE)
        if today_str in port_df['Date'].values:
            print(f"✅ Portfolio already updated for {today_str}. Current Equity: ${port_df['Total_Equity'].iloc[-1]:,.2f}")
            return
        
        last_state = port_df.iloc[-1]
        cash = last_state['Cash']
        units_held = last_state['Units_Held']
        current_holding = last_state['Current_Position']
    else:
        # Day 1: Initialize with $100k
        print("Initializing new $100k Virtual Portfolio...")
        cash = STARTING_CAPITAL
        units_held = 0
        current_holding = "CASH"
        port_df = pd.DataFrame()

    # 4. Execute Portfolio Logic (The Trade Engine)
    trade_executed = False
    fee_paid = 0.0

    # If the model tells us to change positions, we must SELL the old and BUY the new
    if current_holding != target_asset:
        if current_holding != "CASH":
            # SELL OLD ASSET
            sell_price = df['Silver'].iloc[-1] if current_holding == "SILVER" else df['Gold'].iloc[-1]
            cash += units_held * sell_price
            print(f"🔄 ROTATION TRIGGERED: Liquidating {current_holding}...")
            
        # Deduct Slippage/Fees for making a trade
        fee_paid = cash * SLIPPAGE_FEE
        cash -= fee_paid
        
        # BUY NEW ASSET
        units_held = cash / current_price
        cash = 0  # Fully invested
        current_holding = target_asset
        trade_executed = True
        print(f"📈 EXECUTED: Bought {units_held:.2f} units of {target_asset} at ${current_price:,.2f} (Fee: ${fee_paid:,.2f})")

    # 5. Calculate Mark-to-Market Total Equity
    asset_value = units_held * current_price
    total_equity = cash + asset_value

    # 6. Log the Data
    new_row = pd.DataFrame({
        'Date': [today_str],
        'Current_Position': [current_holding],
        'Target_Signal': [target_asset],
        'Asset_Price': [round(current_price, 2)],
        'Units_Held': [round(units_held, 4)],
        'Cash': [round(cash, 2)],
        'Total_Equity': [round(total_equity, 2)],
        'Turnover_Fee': [round(fee_paid, 2)]
    })

    if port_df.empty:
        new_row.to_csv(PORTFOLIO_FILE, index=False)
    else:
        new_row.to_csv(PORTFOLIO_FILE, mode='a', header=False, index=False)

    # 7. Print the Dashboard
    print("\n" + "=" * 55)
    print(f" OQG VIRTUAL PORTFOLIO DASHBOARD: {today_str}")
    print("=" * 55)
    print(f"Total Portfolio Value:  ${total_equity:,.2f}")
    print(f"Active Position:        {current_holding} ({units_held:.2f} units)")
    print(f"Last Traded Price:      ${current_price:,.2f}")
    if trade_executed:
        print(f"*** REBALANCED TODAY: Paid ${fee_paid:,.2f} in execution costs ***")
    print("=" * 55)

if __name__ == "__main__":
    run_portfolio_engine()