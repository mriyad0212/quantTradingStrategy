#!/usr/bin/env python3
"""
Strategy Testing on Permuted Data

This script loads the existing permute.obj file and tests the same 3 strategies 
from main.py on each permuted dataset using the exact same methodology.

Expected runtime: 30-60 minutes for 1000 permutations
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pytz
from quantlab.utils import load_pickle, save_pickle
from quantlab.gene import *

def check_requirements():    
    required_files = ['permute.obj', 'quantlab/']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
    
    try:
        import numpy
        import pandas
        from quantlab.utils import load_pickle, save_pickle
        from quantlab.gene import Gene, GeneticAlpha
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False
    
    return True

def test_strategies_on_permuted_data(permuted_ticker_dfs, tickers, period_start, period_end):
    for ticker in tickers:
        permuted_ticker_dfs.update({ticker+"_"+k: v for k, v in permuted_ticker_dfs[ticker].to_dict(orient="series").items()})
    strategies = [
        {
            'name': 'Gene 1: Volume-based Long-Short',
            'gene_str': "ls_25/75(neg(mean_12(cszscre(div(mult(volume,minus(minus(close,low),minus(high,close))),minus(high,low))))))",
            'gene_factor': 1
        },
        {
            'name': 'Gene 2: Open-Close Reversal', 
            'gene_str': "neg(mean_12(minus(const_1,div(open,close))))",
            'gene_factor': 2
        },
        {
            'name': 'Gene 3: Multi-Timeframe Momentum',
            'gene_str': "plus(ite(gt(mean_10(close),mean_50(close)),const_1,const_0),ite(gt(mean_20(close),mean_100(close)),const_1,const_0),ite(gt(mean_50(close),mean_200(close)),const_1,const_0))",
            'gene_factor': 3
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        try:
            gene = Gene.str_to_gene(strategy['gene_str'])
            alpha = GeneticAlpha(insts=tickers, dfs=permuted_ticker_dfs, start=period_start, end=period_end, genome=gene)
            df = alpha.run_simulation()
            perf = alpha.get_perf_stats(plot=False, gene_factor=strategy['gene_factor'])
            capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
            annual_return = capital_ret.mean() * 252
            annual_vol = capital_ret.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
            cumulative = (1 + capital_ret).cumprod()
            max_dd = (cumulative / cumulative.cummax() - 1).min()
            final_cumulative_return = cumulative.iloc[-1] - 1
            hyp_tests = alpha.get_hypothesis_tests()
            
            results[strategy['name']] = {
                'annual_return': annual_return,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_dd,
                'final_cumulative_return': final_cumulative_return,
                'simulation_df': df,
                'performance_stats': perf,
                'hypothesis_tests': hyp_tests,
                'capital_ret': capital_ret
            }
            
        except Exception as e:
            print(f"Error testing {strategy['name']} on permuted data: {e}")
            results[strategy['name']] = {
                'annual_return': np.nan,
                'annual_volatility': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'final_cumulative_return': np.nan,
                'simulation_df': None,
                'performance_stats': None,
                'hypothesis_tests': None,
                'capital_ret': None
            }
    
    return results

def test_strategies_on_all_permuted_datasets():
    try:
        permuted_datasets = load_pickle("permute.obj")
        print(f"Loaded {len(permuted_datasets)} permuted datasets")
    except Exception as e:
        print(f"Error loading permute.obj: {e}")
        return
    
    period_start = datetime(2000, 1, 1, tzinfo=pytz.utc)
    period_end = datetime(2023, 1, 1, tzinfo=pytz.utc)
    
    first_permuted = permuted_datasets[0]
    tickers = list(first_permuted.keys())[:50]
    
    permuted_results = []
    
    for i, permuted_ticker_dfs in enumerate(permuted_datasets):
        if (i + 1) % 50 == 0 or i < 10:
            print(f"Processing permuted dataset {i+1}/{len(permuted_datasets)}")
        
        strategy_results = test_strategies_on_permuted_data(
            permuted_ticker_dfs, tickers, period_start, period_end
        )
        
        permuted_results.append(strategy_results)
    
    save_pickle("permute_results.obj", permuted_results)
    
    print(f"Successfully tested strategies on {len(permuted_results)} permuted datasets")
    
    print("\n" + "="*80)
    print("PERMUTED RESULTS SUMMARY")
    print("="*80)
    
    for strategy_name in ['Gene 1: Volume-based Long-Short', 'Gene 2: Open-Close Reversal', 'Gene 3: Multi-Timeframe Momentum']:
        print(f"\n{strategy_name}:")
        strategy_data = [result[strategy_name] for result in permuted_results if not pd.isna(result[strategy_name]['sharpe_ratio'])]
        
        if strategy_data:
            sharpe_ratios = [data['sharpe_ratio'] for data in strategy_data]
            annual_returns = [data['annual_return'] for data in strategy_data]
            max_drawdowns = [data['max_drawdown'] for data in strategy_data]
            final_returns = [data['final_cumulative_return'] for data in strategy_data]
            
            print(f"  Sharpe Ratio - Mean: {np.mean(sharpe_ratios):.4f}, Std: {np.std(sharpe_ratios):.4f}")
            print(f"  Annual Return - Mean: {np.mean(annual_returns):.4f}, Std: {np.std(annual_returns):.4f}")
            print(f"  Max Drawdown - Mean: {np.mean(max_drawdowns):.4f}, Std: {np.std(max_drawdowns):.4f}")
            print(f"  Final Return - Mean: {np.mean(final_returns):.4f}, Std: {np.std(final_returns):.4f}")
            print(f"  Valid results: {len(strategy_data)}/{len(permuted_results)}")

def print_results():
    try:
        results = load_pickle("permute_results.obj")
        for strategy in ['Gene 1: Volume-based Long-Short', 'Gene 2: Open-Close Reversal', 'Gene 3: Multi-Timeframe Momentum']:
            data = [r[strategy] for r in results if not pd.isna(r[strategy]['sharpe_ratio'])]
            if data:
                sharpe = [d['sharpe_ratio'] for d in data]
                returns = [d['annual_return'] for d in data]
                drawdowns = [d['max_drawdown'] for d in data]
                print(f"{strategy}:")
                print(f"  Sharpe: {np.mean(sharpe):.4f} ± {np.std(sharpe):.4f}")
                print(f"  Return: {np.mean(returns):.4f} ± {np.std(returns):.4f}")
                print(f"  Drawdown: {np.mean(drawdowns):.4f} ± {np.std(drawdowns):.4f}")
    except:
        print("No results file found")

def main():
    if not check_requirements():
        return

    if Path("permute_results.obj").exists():
        response = input("Do you want to regenerate it? (y/n): ").lower().strip()
        if response not in ['y', 'yes']:
            return
    
    # Confirm before starting
    response = input("Do you want to proceed with testing strategies on permuted datasets? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Exiting.")
        return
    
    try:
        test_strategies_on_all_permuted_datasets()
        print_results()
    except Exception as e:
        print(f"Error testing strategies on permuted datasets: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 