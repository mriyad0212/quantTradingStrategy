import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting script...")

try:
    # Configure matplotlib for better display
    plt.style.use('default')
    print("Matplotlib configured")
    
    # Download data - DO NOT call to_frame()
    print("Downloading data...")
    data = yf.download('GDX', start='2013-01-01', end='2023-01-01')
    print(f"Downloaded data shape: {data.shape}")
    print(f"Downloaded data columns: {list(data.columns)}")
    
    # Fix the MultiIndex columns issue - flatten the column names
    data.columns = data.columns.droplevel(1)  # Remove the ticker level
    print(f"After flattening columns: {list(data.columns)}")
    
    # Make sure it's a DataFrame and only keep 'Close'
    data = data[['Close']]
    print(f"After selecting Close column: {data.shape}")
    
    # Log returns
    data['return'] = np.log(data['Close'] / data['Close'].shift(1))
    print("Log returns calculated")
    
    # SMA
    SMA = 25
    data['SMA'] = data['Close'].rolling(SMA).mean()
    print("SMA calculated")
    
    # Distance from SMA
    data['distance'] = data['Close'] - data['SMA']
    print("Distance from SMA calculated")
    
    # Print basic statistics first
    print("Data shape:", data.shape)
    print("\nDistance statistics:")
    print(data['distance'].describe())
    print(f"\nNumber of times above +3.5 threshold: {(data['distance'] > 3.5).sum()}")
    print(f"Number of times below -3.5 threshold: {(data['distance'] < -3.5).sum()}")
    
    print("Creating plot...")
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
    
    # Save the plot instead of just showing it
    plt.savefig('mean_reversion_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'mean_reversion_plot.png'")
    
    plt.show()
    print("Plot displayed")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

print("Script completed!")
