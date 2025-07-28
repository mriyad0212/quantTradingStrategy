import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for better display
plt.style.use('default')
# plt.ion()  # Turn on interactive mode - commented out to prevent hanging

# Download data - DO NOT call to_frame()
data = yf.download('GDX', start='2013-01-01', end='2023-01-01')

# Fix the MultiIndex columns issue - flatten the column names
data.columns = data.columns.droplevel(1)  # Remove the ticker level

# Make sure it's a DataFrame and only keep 'Close'
data = data[['Close']]

# Log returns
data['return'] = np.log(data['Close'] / data['Close'].shift(1))

# SMA
SMA = 25
data['SMA'] = data['Close'].rolling(SMA).mean()

# Distance from SMA
data['distance'] = data['Close'] - data['SMA']

# Plot
plt.figure(figsize=(10, 6))
data['distance'].dropna().plot(figsize=(10, 6))
plt.axhline(3.5, color='r', linestyle='--', label='Upper threshold')
plt.axhline(-3.5, color='r', linestyle='--', label='Lower threshold')
plt.axhline(0, color='black', linestyle='-', alpha=0.7, label='SMA line')
plt.title('Distance from SMA with Thresholds')
plt.ylabel('Distance from SMA')
plt.xlabel('Date')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Also print some basic statistics
print("Data shape:", data.shape)
print("\nDistance statistics:")
print(data['distance'].describe())
print(f"\nNumber of times above +3.5 threshold: {(data['distance'] > 3.5).sum()}")
print(f"Number of times below -3.5 threshold: {(data['distance'] < -3.5).sum()}")

# Define threshold
threshold = 3.5

## when trend line is greater than upper threshold --> short 
data['position'] = np.where(data['distance'] > threshold, -1, np.nan)

## when trend line is less than lower threshold --> long
data['position'] = np.where(data['distance'] < -threshold, 1, data['position'])

## when trend crosses back into the threshold, it is a good time to close the position (go neatral)
data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0, 0, data['position'])

## fill NAs with 0 
data['position'] = data['position'].fillna(0)

## plot 
data['position'].iloc[SMA:].plot(ylim=[-1.1,1.1], figsize=(10, 6))
plt.title('Trading Positions Over Time')
plt.ylabel('Position (-1: Short, 0: Neutral, 1: Long)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

data['strategy'] = data['position'].shift(1) * data['return']

# Plot cumulative returns comparison
plt.figure(figsize=(10, 6))
data[['return','strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('Cumulative Returns: Buy & Hold vs Mean Reversion Strategy')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(['Buy & Hold', 'Mean Reversion Strategy'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

