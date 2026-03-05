import pandas as pd
import numpy as np

print("(80% Train / 20% Test)")

# 1. Load Data Safely
try:
    df = pd.read_csv('essential_metals_data_corrected.csv', index_col='Date', parse_dates=True, header=0)
except FileNotFoundError:
    try:
        df = pd.read_csv('essential_metals_data.csv', index_col='Date', parse_dates=True, header=0)
    except FileNotFoundError:
        print("Error: Could not find the CSV data file.")
        exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Preprocessing
df.ffill(inplace=True)
df['Gold_Fwd_Ret'] = df['Gold'].pct_change().shift(-1)
df['Silver_Fwd_Ret'] = df['Silver'].pct_change().shift(-1)
df['GCR_Delta'] = df['GCR'].diff()

# Drop rows where indicators are still warming up or shifting created NaNs
df.dropna(subset=['GCR_Z_Score', 'GSR_Z_Score', 'Gold_Fwd_Ret'], inplace=True)

# 2. Split the Data (Exact 80/20 Split)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

print(f"Training Data (In-Sample):    {len(train_df)} trading days ({train_df.index.min().date()} to {train_df.index.max().date()})")
print(f"Testing Data  (Out-of-Sample): {len(test_df)} trading days ({test_df.index.min().date()} to {test_df.index.max().date()})\n")

# 3. Step 1: Optimize on Training Data ONLY
print("--- STEP 1: IN-SAMPLE OPTIMIZATION ---")
best_threshold = 0
best_sharpe = -99

# Testing thresholds to find the highest Sharpe Ratio
for z in [1.0, 1.5, 2.0]:
    # Strategy: Confirmed Slingshot (GSR High AND Copper Outperforming)
    strat_ret = np.where((train_df['GSR_Z_Score'] > z) & (train_df['GCR_Delta'] < 0), 
                         train_df['Silver_Fwd_Ret'], 
                         train_df['Gold_Fwd_Ret'])
    
    # Calculate IS Sharpe
    sharpe = (pd.Series(strat_ret).mean() / pd.Series(strat_ret).std()) * np.sqrt(252)
    
    # Check if this is the best so far
    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_threshold = z

print(f"Optimal Threshold Found: {best_threshold} (Sharpe: {best_sharpe:.2f})")
print("Locking parameters for blind forward test...\n")

# 4. Step 2: Blind Test on Out-of-Sample Data
print("--- STEP 2: OUT-OF-SAMPLE TEST ---")

# Apply the LOCKED threshold to the unseen data
test_df['Strat_Ret'] = np.where((test_df['GSR_Z_Score'] > best_threshold) & (test_df['GCR_Delta'] < 0), 
                                test_df['Silver_Fwd_Ret'], 
                                test_df['Gold_Fwd_Ret'])

# Calculate OOS Benchmarks
bh_gold_cum = (1 + test_df['Gold_Fwd_Ret']).prod()
bh_gold_sharpe = (test_df['Gold_Fwd_Ret'].mean() / test_df['Gold_Fwd_Ret'].std()) * np.sqrt(252)
bh_gold_dd = (((1 + test_df['Gold_Fwd_Ret']).cumprod() / (1 + test_df['Gold_Fwd_Ret']).cumprod().cummax()) - 1).min() * 100

bh_silver_cum = (1 + test_df['Silver_Fwd_Ret']).prod()
bh_silver_sharpe = (test_df['Silver_Fwd_Ret'].mean() / test_df['Silver_Fwd_Ret'].std()) * np.sqrt(252)
bh_silver_dd = (((1 + test_df['Silver_Fwd_Ret']).cumprod() / (1 + test_df['Silver_Fwd_Ret']).cumprod().cummax()) - 1).min() * 100

# Calculate OOS Strategy Performance
strat_cum = (1 + test_df['Strat_Ret']).prod()
strat_sharpe = (test_df['Strat_Ret'].mean() / test_df['Strat_Ret'].std()) * np.sqrt(252)
strat_max_dd = (((1 + test_df['Strat_Ret']).cumprod() / (1 + test_df['Strat_Ret']).cumprod().cummax()) - 1).min() * 100

print(f"B&H Gold   -> Return: {bh_gold_cum:.2f}x | Sharpe: {bh_gold_sharpe:.2f} | Max DD: {bh_gold_dd:.1f}%")
print(f"B&H Silver -> Return: {bh_silver_cum:.2f}x | Sharpe: {bh_silver_sharpe:.2f} | Max DD: {bh_silver_dd:.1f}%")
print("-" * 55)
print(f"STRATEGY   -> Return: {strat_cum:.2f}x | Sharpe: {strat_sharpe:.2f} | Max DD: {strat_max_dd:.1f}%")
print("========================================")

# Automatic Pass/Fail logic
if strat_sharpe > bh_gold_sharpe:
    print("✅ CONCLUSION: Strategy successfully generated Out-of-Sample Alpha. Not Overfit.")
else:
    print("❌ CONCLUSION: Strategy failed to beat benchmark OOS. The model was overfit.")