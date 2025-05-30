import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from quantlab.utils import load_pickle, save_pickle
from quantlab.quant_stats import permute_multi_bars
from copy import deepcopy
import time

def load_original_data():
    print("Loading original dataset...")
    tickers, ticker_dfs = load_pickle("dataset.obj")
    print(f"Loaded data for {len(tickers)} tickers")
    return tickers, ticker_dfs

def create_permuted_dataset(tickers, ticker_dfs, permutation_id):
    print(f"Creating permuted dataset {permutation_id}...")
    
    bars = []
    valid_tickers = []
    
    for ticker in tickers[:50]:
        if ticker in ticker_dfs:
            df = ticker_dfs[ticker]
            if len(df) > 100 and all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
                bars.append(df[["open", "high", "low", "close", "volume"]].copy())
                valid_tickers.append(ticker)
    
    print(f"Permuting {len(bars)} valid ticker datasets...")
    permuted_bars = permute_multi_bars(bars)
    
    permuted_ticker_dfs = {}
    for ticker, permuted_bar in zip(valid_tickers, permuted_bars):
        permuted_ticker_dfs[ticker] = permuted_bar
    
    return permuted_ticker_dfs

def generate_permuted_datasets(n_permutations):
    tickers, ticker_dfs = load_original_data()
    
    permuted_datasets = []
    
    start_time = time.time()
    
    for i in range(n_permutations):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            estimated_total = elapsed * n_permutations / max(1, i)
            remaining = estimated_total - elapsed
            print(f"Progress: {i}/{n_permutations} ({i/n_permutations*100:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
        
        try:
            permuted_dataset = create_permuted_dataset(tickers, ticker_dfs, i)
            permuted_datasets.append(permuted_dataset)
        except Exception as e:
            print(f"Error creating permuted dataset {i}: {e}")
            continue
    
    print(f"\nGenerated {len(permuted_datasets)} permuted datasets")
    return permuted_datasets

def main():
    print("="*60)
    print("PERMUTED DATASET GENERATOR")
    print("="*60)
    
    n_permutations = 5
    print(f"Generating {n_permutations} permuted datasets...")
    
    try:
        permuted_datasets = generate_permuted_datasets(n_permutations)
        
        print(f"\nSaving {len(permuted_datasets)} permuted datasets to permute.obj...")
        save_pickle("permute.obj", permuted_datasets)
        
        print("="*60)
        print("PERMUTATION COMPLETE")
        print("="*60)
        print(f"Successfully generated and saved {len(permuted_datasets)} permuted datasets")
        print("File saved as: permute.obj")
        
        print("\nDataset Statistics:")
        if permuted_datasets:
            sample_dataset = permuted_datasets[0]
            print(f"- Number of tickers per dataset: {len(sample_dataset)}")
            if sample_dataset:
                sample_ticker = list(sample_dataset.keys())[0]
                sample_data = sample_dataset[sample_ticker]
                print(f"- Data points per ticker: {len(sample_data)}")
                print(f"- Date range: {sample_data.index.min()} to {sample_data.index.max()}")
                print(f"- Columns: {list(sample_data.columns)}")
        
    except Exception as e:
        print(f"Error during permutation generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 