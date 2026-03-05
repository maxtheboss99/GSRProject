import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# 1. Define Timeframe (10 Years of Daily Data + 1 extra year to account for the 252-day rolling window)
end_date = datetime.now()
start_date = end_date - timedelta(days=365*11) # Pull 11 years so we get a full 10 years of clean output

print("--- Fetching Daily Futures Data ---")

# 2. Download Data
tickers = ['GC=F', 'HG=F', 'SI=F']
data = yf.download(tickers, start=start_date, end=end_date)

# 3. Handle the yfinance Multi-Index structure securely
if 'Close' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else 'Close' in data.columns:
    df = data['Close'].copy()
elif 'Adj Close' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else 'Adj Close' in data.columns:
    df = data['Adj Close'].copy()
else:
    print("Error: Could not locate Close prices.")
    exit()

# Rename columns to drop the ticker symbols for a cleaner CSV
df.rename(columns={'GC=F': 'Gold', 'HG=F': 'Copper', 'SI=F': 'Silver'}, inplace=True)

# Drop any days where markets were closed to keep correlations accurate
df.dropna(inplace=True)

# 4. Calculate Ratios
df['GCR'] = df['Gold'] / df['Copper']
df['GSR'] = df['Gold'] / df['Silver']

# 5. Feature Engineering: Z-Scores for EACH ONE (252 trading days = ~1 Year)
window_z = 252
for col in ['Gold', 'Copper', 'Silver', 'GCR', 'GSR']:
    mean = df[col].rolling(window=window_z).mean()
    std = df[col].rolling(window=window_z).std()
    df[f'{col}_Z_Score'] = (df[col] - mean) / std

# 6. Feature Engineering: Rolling Correlations for EACH ONE (60 trading days = ~1 Quarter)
# Correlating Copper, Silver, GCR, and GSR against Gold's movement
window_corr = 60
for col in ['Copper', 'Silver', 'GCR', 'GSR']:
    df[f'Corr_Gold_{col}'] = df['Gold'].rolling(window=window_corr).corr(df[col])

# 7. Format Date for easy Excel/Code ingestion (YYYY-MM-DD)
df.index = df.index.strftime('%Y-%m-%d')
df.index.name = 'Date'

# 8. Clean up the data for Excel (Drop NAs and Round)
# Dropping NAs removes the first 252 days that were blank due to the rolling window calculations
df.dropna(inplace=True)

# Round everything to 4 decimal places for clean "simple data"
df = df.round(4)

# 9. Output to CSV
filename = "essential_metals_data_corrected.csv"
df.to_csv(filename)

print(f"--- Success! Clean daily data saved to {filename} ---")
print(df.head()) # Show the clean starting rows!