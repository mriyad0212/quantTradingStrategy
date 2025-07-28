print("Starting test...")

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("All imports successful")
    
    # Download a small amount of data
    print("Downloading data...")
    data = yf.download('GDX', start='2022-01-01', end='2022-12-31')
    print(f"Data downloaded successfully. Shape: {data.shape}")
    
    # Check columns
    print(f"Columns: {list(data.columns)}")
    
    # Fix columns
    data.columns = data.columns.droplevel(1)
    print(f"After fixing columns: {list(data.columns)}")
    
    # Test basic operations
    data = data[['Close']]
    print(f"Data shape after selecting Close: {data.shape}")
    print(f"First few Close prices:\n{data['Close'].head()}")
    
    print("Test completed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
