import pandas as pd
import numpy as np

# 1. Load the Preprocessed Data
# Updated filename and added 'header=0' to bypass the "Ticker" label
try:
    df = pd.read_csv('essential_metals_data_corrected.csv', index_col='Date', parse_dates=True, header=0)
    print("--- OQG Backtest Engine Initialized ---")
except FileNotFoundError:
    print("Error: Could not find 'essential_metals_data_corrected.csv'.")
    exit()

# If the CSV saved with a multi-level header because of yfinance, this flattens it automatically
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 2. Preprocessing: Calculate Daily and Forward Returns
df['Gold_Daily_Ret'] = df['Gold'].pct_change()
df['Silver_Daily_Ret'] = df['Silver'].pct_change()

# We want to know what happens TOMORROW based on TODAY'S signal
df['Gold_Fwd_Ret'] = df['Gold_Daily_Ret'].shift(-1)
df['Silver_Fwd_Ret'] = df['Silver_Daily_Ret'].shift(-1)

# Calculate GCR Delta (Momentum) for the Silver Signal
df['GCR_Delta'] = df['GCR'].diff()

# 3. Apply Signal Hypotheses

# Hypothesis 1: GCR Crisis Filter (Long Gold when GCR Z-Score > 1.5)
df['Signal_Gold'] = np.where(df['GCR_Z_Score'] > 1.5, 1, 0)

# Hypothesis 2: Silver Slingshot (Long Silver when GSR Z > 1.5 AND GCR is falling)
df['Signal_Silver'] = np.where((df['GSR_Z_Score'] > 1.5) & (df['GCR_Delta'] < 0), 1, 0)

# 4. Calculate Strategy Returns
# Strategy return = The signal generated today * the return of the asset tomorrow
df['Strat_Gold_Ret'] = df['Signal_Gold'] * df['Gold_Fwd_Ret']
df['Strat_Silver_Ret'] = df['Signal_Silver'] * df['Silver_Fwd_Ret']

# 5. Evaluate Performance (Cumulative Returns)
# Drop NaNs created by shifting and percentage changes
bt_df = df.dropna().copy()

# Calculate cumulative growth of $1
bt_df['Cum_Hold_Gold'] = (1 + bt_df['Gold_Daily_Ret']).cumprod()
bt_df['Cum_Strat_Gold'] = (1 + bt_df['Strat_Gold_Ret']).cumprod()

bt_df['Cum_Hold_Silver'] = (1 + bt_df['Silver_Daily_Ret']).cumprod()
bt_df['Cum_Strat_Silver'] = (1 + bt_df['Strat_Silver_Ret']).cumprod()

# 6. Output Results
print("\n--- Backtest Results (Cumulative Return Multiplier) ---")
print(f"Buy & Hold Gold:   {bt_df['Cum_Hold_Gold'].iloc[-1]:.2f}x")
print(f"Strategy Gold:     {bt_df['Cum_Strat_Gold'].iloc[-1]:.2f}x")
print(f"Buy & Hold Silver: {bt_df['Cum_Hold_Silver'].iloc[-1]:.2f}x")
print(f"Strategy Silver:   {bt_df['Cum_Strat_Silver'].iloc[-1]:.2f}x")

# Calculate Hit Rate (Win/Loss ratio of trades)
gold_trades = bt_df[bt_df['Signal_Gold'] == 1]
gold_win_rate = (gold_trades['Gold_Fwd_Ret'] > 0).mean() * 100 if not gold_trades.empty else 0

silver_trades = bt_df[bt_df['Signal_Silver'] == 1]
silver_win_rate = (silver_trades['Silver_Fwd_Ret'] > 0).mean() * 100 if not silver_trades.empty else 0

print("\n--- Signal Accuracy (Hit Rate) ---")
print(f"Gold Trades Executed:   {len(gold_trades)}")
print(f"Gold Signal Win Rate:   {gold_win_rate:.1f}%")
print(f"Silver Trades Executed: {len(silver_trades)}")
print(f"Silver Signal Win Rate: {silver_win_rate:.1f}%")