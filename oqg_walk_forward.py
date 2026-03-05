import pandas as pd
import numpy as np

print("WALK-FORWARD VALIDATOR         ")

# 1. Load Data
try:
    df = pd.read_csv('essential_metals_data.csv', index_col='Date', parse_dates=True, header=0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
except FileNotFoundError:
    print("Error: Could not find data file.")
    exit()

df.ffill(inplace=True)
df['Gold_Fwd_Ret'] = df['Gold'].pct_change().shift(-1)
df['Silver_Fwd_Ret'] = df['Silver'].pct_change().shift(-1)
df['GCR_Delta'] = df['GCR'].diff()
df.dropna(subset=['GSR_Z_Score', 'Gold_Fwd_Ret'], inplace=True)

# 2. Configuration for Walk-Forward
# We will use a 4-year training window and a 1-year testing window
train_days = 252 * 4
test_days = 252
total_days = len(df)

results = []
start_idx = 0

print(f"Running sliding window analysis across {total_days} days...")

# 3. The Walk-Forward Loop
while start_idx + train_days + test_days <= total_days:
    # Define current windows
    train_window = df.iloc[start_idx : start_idx + train_days]
    test_window = df.iloc[start_idx + train_days : start_idx + train_days + test_days]
    
    test_start_date = test_window.index.min().date()
    test_end_date = test_window.index.max().date()

    # Step A: Optimize Threshold on the Training Window
    best_z = 0
    best_sharpe = -99
    for z in [1.0, 1.25, 1.5, 1.75, 2.0]:
        strat = np.where((train_window['GSR_Z_Score'] > z) & (train_window['GCR_Delta'] < 0), 
                         train_window['Silver_Fwd_Ret'], train_window['Gold_Fwd_Ret'])
        sharpe = (pd.Series(strat).mean() / pd.Series(strat).std()) * np.sqrt(252)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_z = z

    # Step B: Apply THAT threshold to the UNSEEN Test Window
    oos_strat = np.where((test_window['GSR_Z_Score'] > best_z) & (test_window['GCR_Delta'] < 0), 
                         test_window['Silver_Fwd_Ret'], test_window['Gold_Fwd_Ret'])
    
    oos_ret = (1 + pd.Series(oos_strat)).prod()
    oos_sharpe = (pd.Series(oos_strat).mean() / pd.Series(oos_strat).std()) * np.sqrt(252)
    
    # Benchmark: Gold Buy & Hold for that same period
    gold_bh = (1 + test_window['Gold_Fwd_Ret']).prod()

    results.append({
        'Window': f"{test_start_date} to {test_end_date}",
        'Optimum_Z': best_z,
        'OOS_Return': round(oos_ret, 2),
        'Gold_BH': round(gold_bh, 2),
        'Alpha': round(oos_ret - gold_bh, 2)
    })

    # Slide the window forward by 1 year
    start_idx += test_days

# 4. Display Results
wf_results = pd.DataFrame(results)
print("\n--- Walk-Forward Window Results ---")
print(wf_results.to_string(index=False))

print("\n--- Summary Statistics ---")
print(f"Average Alpha per Window: {wf_results['Alpha'].mean():.2f}")
print(f"Win Rate (Strategy > Gold): {(wf_results['Alpha'] > 0).mean()*100:.1f}%")

if (wf_results['Alpha'] > 0).mean() > 0.5:
    print("\n✅ CONCLUSION: The model is ROBUST. It consistently finds Alpha across moving windows.")
else:
    print("\n❌ CONCLUSION: The model is UNSTABLE. Performance varies too much by window.")