import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from quantlab.utils import load_pickle, save_pickle
from quantlab.quant_stats import permute_multi_bars
from copy import deepcopy
import time

def load_original_data():
    tickers, ticker_dfs = load_pickle("dataset.obj")
    print(f"Loaded data for {len(tickers)} tickers")
    return tickers, ticker_dfs

def create_permuted_dataset(tickers, ticker_dfs, permutation_id):
    bars = []
    valid_tickers = []
    
    for ticker in tickers[:50]:
        if ticker in ticker_dfs:
            df = ticker_dfs[ticker]
            if len(df) > 100 and all(col in df.columns for col in ["open", "high", "low", "close", "volume"]):
                bars.append(df[["open", "high", "low", "close", "volume"]].copy())
                valid_tickers.append(ticker)
    
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
    n_permutations = 5
    
    try:
        permuted_datasets = generate_permuted_datasets(n_permutations)
        
        save_pickle("permute.obj", permuted_datasets)
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