import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting mean reversion analysis...")

# Configure matplotlib for better display
plt.style.use('default')
# Remove interactive mode to prevent hanging
# plt.ion()  # Turn on interactive mode

print("Downloading GDX data from 2013-2023...")
# Download data - DO NOT call to_frame()
data = yf.download('GDX', start='2013-01-01', end='2023-01-01')

print(f"Raw data shape: {data.shape}")
print(f"Raw data columns: {list(data.columns)}")

# Fix the MultiIndex columns issue - flatten the column names
data.columns = data.columns.droplevel(1)  # Remove the ticker level

# Make sure it's a DataFrame and only keep 'Close'
data = data[['Close']]

print(f"After selecting Close column: {data.shape}")

# Log returns
data['return'] = np.log(data['Close'] / data['Close'].shift(1))

# SMA
SMA = 25
data['SMA'] = data['Close'].rolling(SMA).mean()

# Distance from SMA
data['distance'] = data['Close'] - data['SMA']

print("Calculations completed. Creating first plot...")

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
plt.savefig('distance_plot.png', dpi=300, bbox_inches='tight')
print("First plot saved as 'distance_plot.png'")
plt.close()  # Close the plot to free memory

# Also print some basic statistics
print("Data shape:", data.shape)
print("\nDistance statistics:")
print(data['distance'].describe())
print(f"\nNumber of times above +3.5 threshold: {(data['distance'] > 3.5).sum()}")
print(f"Number of times below -3.5 threshold: {(data['distance'] < -3.5).sum()}")

# Define threshold
threshold = 3.5

print("Calculating trading positions...")

## when trend line is greater than upper threshold --> short 
data['position'] = np.where(data['distance'] > threshold, -1, np.nan)

## when trend line is less than lower threshold --> long
data['position'] = np.where(data['distance'] < -threshold, 1, data['position'])

## when trend crosses back into the threshold, it is a good time to close the position (go neutral)
data['position'] = np.where(data['distance'] * data['distance'].shift(1) < 0, 0, data['position'])

## fill NAs with 0 
data['position'] = data['position'].fillna(0)

print("Creating position plot...")

## plot 
plt.figure(figsize=(10, 6))
data['position'].iloc[SMA:].plot(ylim=[-1.1,1.1], figsize=(10, 6))
plt.title('Trading Positions Over Time')
plt.ylabel('Position (-1: Short, 0: Neutral, 1: Long)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('positions_plot.png', dpi=300, bbox_inches='tight')
print("Position plot saved as 'positions_plot.png'")
plt.close()

print("Calculating strategy returns...")

data['strategy'] = data['position'].shift(1) * data['return']

print("Creating cumulative returns plot...")

# Plot cumulative returns comparison
plt.figure(figsize=(10, 6))
data[['return','strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title('Cumulative Returns: Buy & Hold vs Mean Reversion Strategy')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(['Buy & Hold', 'Mean Reversion Strategy'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_plot.png', dpi=300, bbox_inches='tight')
print("Returns plot saved as 'returns_plot.png'")
plt.close()

# Print final statistics
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

# Calculate total returns
total_return_buy_hold = data['return'].dropna().cumsum().apply(np.exp).iloc[-1]
total_return_strategy = data['strategy'].dropna().cumsum().apply(np.exp).iloc[-1]

print(f"Buy & Hold Total Return: {total_return_buy_hold:.4f} ({(total_return_buy_hold-1)*100:.2f}%)")
print(f"Mean Reversion Strategy Total Return: {total_return_strategy:.4f} ({(total_return_strategy-1)*100:.2f}%)")

# Position statistics
positions = data['position'].value_counts()
print(f"\nPosition Statistics:")
print(f"Neutral (0): {positions.get(0, 0)} periods")
print(f"Long (1): {positions.get(1, 0)} periods") 
print(f"Short (-1): {positions.get(-1, 0)} periods")

print("\nAnalysis completed successfully!")
print("Check the generated PNG files for the plots.")
