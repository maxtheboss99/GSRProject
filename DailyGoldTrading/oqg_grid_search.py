import pandas as pd
import numpy as np

# 1. Load the Preprocessed Data
print("--- Initializing OQG Regime Rotation Backtester ---")
try:
    df = pd.read_csv('essential_metals_data_corrected.csv', index_col='Date', parse_dates=True, header=0)
except FileNotFoundError:
    try:
        df = pd.read_csv('essential_metals_data.csv', index_col='Date', parse_dates=True, header=0)
    except FileNotFoundError:
        print("Error: Could not find the CSV data file.")
        exit()

# Flatten headers if yfinance created a multi-index
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 2. Preprocessing: Calculate Returns & Momentum
df.ffill(inplace=True)

df['Gold_Ret'] = df['Gold'].pct_change()
df['Silver_Ret'] = df['Silver'].pct_change()
df['Copper_Ret'] = df['Copper'].pct_change()

# Tomorrow's return (Target variable)
df['Gold_Fwd_Ret'] = df['Gold_Ret'].shift(-1)
df['Silver_Fwd_Ret'] = df['Silver_Ret'].shift(-1)

# Momentum features
df['GCR_Delta'] = df['GCR'].diff()

# Drop rows with NaNs caused by the rolling/shifting
df.dropna(subset=['GCR_Z_Score', 'GSR_Z_Score', 'Gold_Fwd_Ret'], inplace=True)

# 3. Define the Grid Search Function for Rotation
def run_rotation_backtest(df, signal_series, name, threshold):
    """Calculates risk-adjusted metrics for a strategy"""
    
    # Calculate strategy returns
    strat_ret = signal_series
    
    # Cumulative Return Multiplier
    cum_ret = (1 + strat_ret).prod()
    
    # Maximum Drawdown (Risk)
    roll_max = (1 + strat_ret).cumprod().cummax()
    drawdown = ((1 + strat_ret).cumprod() / roll_max) - 1
    max_dd = drawdown.min() * 100
    
    # Annualized Sharpe Ratio (Risk-Adjusted Return)
    # Assumes 0% risk-free rate for simplicity of comparison
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252)
    
    return {'Method': name, 'Threshold': threshold, 
            'Return (x)': cum_ret, 'Max DD %': max_dd, 'Sharpe': sharpe}

# 4. Execute the Grid Search across different methods
results = []
thresholds = [1.0, 1.5, 2.0]

for z in thresholds:
    
    # -----------------------------------------------------------------
    # STRATEGY 1: Silver Slingshot Rotation
    # Base = Gold. Rotate 100% to Silver when GSR is statistically high (cheap silver)
    # -----------------------------------------------------------------
    ret_m1 = np.where(df['GSR_Z_Score'] > z, df['Silver_Fwd_Ret'], df['Gold_Fwd_Ret'])
    results.append(run_rotation_backtest(df, pd.Series(ret_m1), 'Silver Slingshot (Base Gold)', z))
    
    # -----------------------------------------------------------------
    # STRATEGY 2: Confirmed Silver Rotation
    # Base = Gold. Rotate to Silver ONLY when GSR is high AND GCR is falling (Industrial demand)
    # -----------------------------------------------------------------
    ret_m2 = np.where((df['GSR_Z_Score'] > z) & (df['GCR_Delta'] < 0), df['Silver_Fwd_Ret'], df['Gold_Fwd_Ret'])
    results.append(run_rotation_backtest(df, pd.Series(ret_m2), 'Confirmed Slingshot', z))

    # -----------------------------------------------------------------
    # STRATEGY 3: Gold Crisis Rotation
    # Base = Silver. Rotate 100% to Gold when GCR hits Crisis levels
    # -----------------------------------------------------------------
    ret_m3 = np.where(df['GCR_Z_Score'] > z, df['Gold_Fwd_Ret'], df['Silver_Fwd_Ret'])
    results.append(run_rotation_backtest(df, pd.Series(ret_m3), 'Crisis Rotation (Base Silver)', z))

# 5. Calculate Benchmarks
bh_gold_cum = (1 + df['Gold_Fwd_Ret']).prod()
bh_gold_dd = (((1 + df['Gold_Fwd_Ret']).cumprod() / (1 + df['Gold_Fwd_Ret']).cumprod().cummax()) - 1).min() * 100
bh_gold_sharpe = (df['Gold_Fwd_Ret'].mean() / df['Gold_Fwd_Ret'].std()) * np.sqrt(252)

bh_silver_cum = (1 + df['Silver_Fwd_Ret']).prod()
bh_silver_dd = (((1 + df['Silver_Fwd_Ret']).cumprod() / (1 + df['Silver_Fwd_Ret']).cumprod().cummax()) - 1).min() * 100
bh_silver_sharpe = (df['Silver_Fwd_Ret'].mean() / df['Silver_Fwd_Ret'].std()) * np.sqrt(252)

# 6. Format and Display Results
results_df = pd.DataFrame(results)
results_df['Return (x)'] = results_df['Return (x)'].round(2)
results_df['Max DD %'] = results_df['Max DD %'].round(1)
results_df['Sharpe'] = results_df['Sharpe'].round(2)

# Sort by Sharpe Ratio (The Quant Standard)
best_strategies = results_df.sort_values(by='Sharpe', ascending=False)

print("\n=== THE BENCHMARKS (10-Year Buy & Hold) ===")
print(f"B&H Gold   -> Return: {bh_gold_cum:.2f}x | Max Drawdown: {bh_gold_dd:.1f}% | Sharpe: {bh_gold_sharpe:.2f}")
print(f"B&H Silver -> Return: {bh_silver_cum:.2f}x | Max Drawdown: {bh_silver_dd:.1f}% | Sharpe: {bh_silver_sharpe:.2f}\n")

print("=== TOP PERFORMING REGIME ROTATION STRATEGIES ===")
print("(Ranked by Risk-Adjusted Return / Sharpe Ratio)")
print(best_strategies.to_string(index=False))

# Optional: Save results to CSV for your white paper
best_strategies.to_csv("rotation_strategy_summary.csv", index=False)