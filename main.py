import pytz
import yfinance
import requests
import threading
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from quantlab.utils import timeme
from quantlab.utils import save_pickle, load_pickle
from quantlab.utils import Portfolio
import warnings
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from quantlab.utils import *
from quantlab.gene import *
from quantlab.quant_stats import *
from copy import deepcopy
from scipy import stats
import yfinance as yf

warnings.filterwarnings("ignore")

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.content,'html')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table))
    tickers = list(df[0].Symbol)
    return tickers

def get_history(ticker, period_start, period_end, granularity="1d", tries=0):
    try:
        df = yfinance.Ticker(ticker).history(
            start=period_start,
            end=period_end,
            interval=granularity,
            auto_adjust=True
        ).reset_index()
    except Exception as err:
        if tries < 5:
            return get_history(ticker, period_start, period_end, granularity, tries+1)
        return pd.DataFrame()
    
    df = df.rename(columns={
        "Date":"datetime",
        "Open":"open",
        "High":"high",
        "Low":"low",
        "Close":"close",
        "Volume":"volume"
    })
    if df.empty:
        return pd.DataFrame()
    df.datetime = pd.DatetimeIndex(df.datetime.dt.date).tz_localize(pytz.utc)
    df = df.drop(columns=["Dividends", "Stock Splits"])
    df = df.set_index("datetime",drop=True)
    return df

def get_histories(tickers, period_starts,period_ends, granularity="1d"):
    dfs = [None]*len(tickers)
    def _helper(i):
        print(tickers[i])
        df = get_history(
            tickers[i],
            period_starts[i], 
            period_ends[i], 
            granularity=granularity
        )
        dfs[i] = df
    threads = [threading.Thread(target=_helper,args=(i,)) for i in range(len(tickers))]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    tickers = [tickers[i] for i in range(len(tickers)) if not dfs[i].empty]
    dfs = [df for df in dfs if not df.empty]
    return tickers, dfs

def get_ticker_dfs(start,end):
    from quantlab.utils import load_pickle,save_pickle
    try:
        tickers, ticker_dfs = load_pickle("dataset.obj")
    except Exception as err:
        tickers = get_sp500_tickers()
        starts=[start]*len(tickers)
        ends=[end]*len(tickers)
        tickers,dfs = get_histories(tickers,starts,ends,granularity="1d")
        ticker_dfs = {ticker:df for ticker,df in zip(tickers,dfs)}
        save_pickle("dataset.obj", (tickers,ticker_dfs))
    return tickers, ticker_dfs 

def get_sp500_benchmark(start_date, end_date):
    """Get S&P 500 benchmark data using Ticker.history()"""
    try:
        print("Downloading S&P 500 benchmark data")
        ticker = yfinance.Ticker("SPY")
        spy_data = ticker.history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )
        
        if spy_data.empty:
            raise ValueError("No data returned for SPY")
        
        # Use Close price for returns calculation
        spy_returns = spy_data['Close'].pct_change().dropna()
        
        # Ensure proper timezone handling
        if spy_returns.index.tz is None:
            spy_returns.index = spy_returns.index.tz_localize('UTC')
        else:
            spy_returns.index = spy_returns.index.tz_convert('UTC')
            
        return spy_returns
        
    except Exception as e:
        print(f"Warning: Could not download S&P 500 data: {e}")
        print("Continuing analysis without S&P 500 benchmark")
        # Return empty series with proper index
        empty_series = pd.Series(dtype=float, name='SPY_returns')
        return empty_series

def calculate_sp500_volatility_regime(spy_returns, window=63):
    """Calculate S&P 500 volatility regime for transaction costs"""
    rolling_vol = spy_returns.rolling(window).std() * np.sqrt(252)
    return rolling_vol

def get_transaction_cost_multiplier(volatility):
    """
    Get transaction cost multiplier based on S&P 500 volatility regime
    <0.15: low vol (0.2% transaction cost)
    0.15-0.25: normal (0.35% transaction cost) 
    0.25-0.35: walking on ice (0.4% transaction cost)
    >0.35: crisis (0.5% transaction cost)
    """
    if pd.isna(volatility):
        return 0.0035  # Default to normal regime
    elif volatility < 0.15:
        return 0.002  # 0.2%
    elif volatility < 0.25:
        return 0.0035  # 0.35%
    elif volatility < 0.35:
        return 0.004   # 0.4%
    else:
        return 0.005   # 0.5%

def apply_transaction_costs_to_alpha(alpha, spy_volatility):
    """Apply transaction costs to GeneticAlpha based on volatility regime"""
    # Get the original stats
    zfs = alpha.get_zero_filtered_stats()
    
    # Try to get position data - check different possible keys
    if "capital_weights" in zfs:
        positions = zfs["capital_weights"]
    elif "weights" in zfs:
        positions = zfs["weights"] 
    elif "holdings" in zfs:
        positions = zfs["holdings"]
    else:
        # If no position data available, skip transaction costs
        return alpha
    
    capital_ret = zfs["capital_ret"].copy()
    
    # Calculate position changes (turnover)
    position_changes = positions.diff().abs().sum(axis=1) if len(positions.shape) > 1 else positions.diff().abs()
    
    # Apply transaction costs only if we have volatility data
    if len(spy_volatility) > 0:
        # Ensure spy_volatility is a Series
        if isinstance(spy_volatility, pd.DataFrame):
            spy_volatility = spy_volatility.iloc[:, 0]  # Take first column
        
        for date in capital_ret.index:
            if date in spy_volatility.index:
                vol_value = spy_volatility.loc[date]
                if not pd.isna(vol_value):
                    if date in position_changes.index:
                        cost_multiplier = get_transaction_cost_multiplier(vol_value)
                        transaction_cost = position_changes.loc[date] * cost_multiplier
                        capital_ret.loc[date] = capital_ret.loc[date] - transaction_cost
    
    # Update the alpha object with adjusted returns
    alpha._zero_filtered_stats = zfs.copy()
    alpha._zero_filtered_stats["capital_ret"] = capital_ret
    
    return alpha

def apply_transaction_costs(returns, positions, spy_volatility):
    """Apply transaction costs based on position changes and volatility regime"""
    position_changes = positions.diff().abs()
    
    # Get transaction cost for each period
    transaction_costs = position_changes * 0
    for date in position_changes.index:
        if date in spy_volatility.index and not pd.isna(spy_volatility[date]):
            cost_multiplier = get_transaction_cost_multiplier(spy_volatility[date])
            transaction_costs[date] = position_changes[date] * cost_multiplier
    
    # Apply costs to returns
    net_returns = returns - transaction_costs
    return net_returns

def plot_combined_returns_with_benchmark(alphas, strategy_names, spy_returns):
    """Plot combined returns including S&P 500 benchmark"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    
    # Plot strategies
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        cumulative_returns = (1 + capital_ret).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values, 
                label=f'{name}', linewidth=2)
    
    # Plot S&P 500 benchmark (only if data is available)
    if len(spy_returns) > 0:
        spy_cumulative = (1 + spy_returns).cumprod()
        plt.plot(spy_cumulative.index, spy_cumulative.values, 
                label='S&P 500 (Buy & Hold)', linewidth=2, color='black', linestyle='--')
        title_suffix = " vs S&P 500"
    else:
        title_suffix = ""
    
    plt.title(f'Combined Cumulative Returns - All Strategies{title_suffix}', fontsize=14)
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 1, 2)
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        plt.plot(capital_ret.index, capital_ret.values, 
                label=f'{name}', linewidth=1, alpha=0.7)
    
    # Plot S&P 500 daily returns (only if data is available)
    if len(spy_returns) > 0:
        plt.plot(spy_returns.index, spy_returns.values, 
                label='S&P 500', linewidth=1, alpha=0.7, color='black', linestyle='--')
    
    plt.title(f'Daily Returns - All Strategies{title_suffix}', fontsize=14)
    plt.ylabel('Daily Return')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = './images/combined_returns_with_benchmark.png' if len(spy_returns) > 0 else './images/combined_returns.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_rolling_drawdown(alphas, strategy_names, window=63):
    """Create rolling drawdown plot"""
    plt.figure(figsize=(15, 8))
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        cumulative = (1 + capital_ret).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        plt.plot(drawdown.index, drawdown.values, label=f'{name}', linewidth=2)
    
    plt.title(f'Rolling Maximum Drawdown ({window}-day window)', fontsize=14)
    plt.ylabel('Drawdown')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/rolling_drawdown.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rolling_calmar(alphas, strategy_names, window=63):
    """Create rolling Calmar ratio plot"""
    plt.figure(figsize=(15, 8))
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        
        # Calculate rolling annual return
        rolling_return = capital_ret.rolling(window).mean() * 252
        
        # Calculate rolling max drawdown
        cumulative = (1 + capital_ret).cumprod()
        rolling_max = cumulative.rolling(window, min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(window).min().abs()
        
        # Calmar ratio = Annual Return / Max Drawdown
        calmar_ratio = rolling_return / max_drawdown.replace(0, np.nan)
        plt.plot(calmar_ratio.index, calmar_ratio.values, label=f'{name}', linewidth=2)
    
    plt.title(f'Rolling Calmar Ratio ({window}-day window)', fontsize=14)
    plt.ylabel('Calmar Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/rolling_calmar.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rolling_volatility(alphas, strategy_names, window=63):
    """Create rolling annualized volatility plot"""
    plt.figure(figsize=(15, 8))
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        rolling_vol = capital_ret.rolling(window).std() * np.sqrt(252)
        plt.plot(rolling_vol.index, rolling_vol.values, label=f'{name}', linewidth=2)
    
    plt.title(f'Rolling Annualized Volatility ({window}-day window)', fontsize=14)
    plt.ylabel('Annualized Volatility')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/rolling_volatility.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rolling_sharpe(alphas, strategy_names, window=63):
    """Create rolling Sharpe ratio plot"""
    plt.figure(figsize=(15, 8))
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        rolling_mean = capital_ret.rolling(window).mean() * 252
        rolling_std = capital_ret.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=f'{name}', linewidth=2)
    
    plt.title(f'Rolling Sharpe Ratio ({window}-day window)', fontsize=14)
    plt.ylabel('Sharpe Ratio')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./images/rolling_sharpe.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_return_distributions_and_qq(alphas, strategy_names):
    """Create return distribution and QQ plots - 2 subplots per strategy"""
    n_strategies = len(alphas)
    
    for i, (alpha, name) in enumerate(zip(alphas, strategy_names)):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"].dropna()
        
        # Distribution plot
        ax1 = axes[0]
        ax1.hist(capital_ret.values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = capital_ret.mean(), capital_ret.std()
        x = np.linspace(capital_ret.min(), capital_ret.max(), 100)
        normal_dist = stats.norm.pdf(x, mu, sigma)
        ax1.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
        
        ax1.set_title(f'{name}\nReturn Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Daily Returns')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # QQ plot
        ax2 = axes[1]
        stats.probplot(capital_ret.values, dist="norm", plot=ax2)
        ax2.set_title(f'{name}\nQ-Q Plot vs Normal Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        skew = capital_ret.skew()
        kurt = capital_ret.kurtosis()
        ax1.text(0.02, 0.98, f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'./images/return_distribution_qq_{i+1}.png', dpi=300, bbox_inches='tight')
        plt.show()

def plot_return_correlation_heatmap(alphas, strategy_names):
    """Create a correlation heatmap for the returns of multiple strategies"""
    returns_data = {}
    
    for i, alpha in enumerate(alphas):
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        returns_data[strategy_names[i]] = capital_ret
    
    returns_df = pd.DataFrame(returns_data)
    correlation_matrix = returns_df.corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": .8},
                vmin=-1, vmax=1,
                mask=mask)
    
    plt.title('Strategy Returns Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Strategies', fontsize=12, fontweight='bold')
    plt.ylabel('Strategies', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig('./images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return correlation_matrix

def plot_permuted_vs_actual_performance(actual_alphas, strategy_names, permuted_results_list):
    """Compare actual strategy performance vs performance on permuted data"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for strategy_idx in range(3):
        ax = axes[strategy_idx]
        
        actual_alpha = actual_alphas[strategy_idx]
        actual_capital_ret = actual_alpha.get_zero_filtered_stats()["capital_ret"]
        actual_cum_returns = (1 + actual_capital_ret).cumprod()
        
        ax.plot(actual_cum_returns.index, actual_cum_returns.values, 
                color='red', linewidth=3, label='Actual Data', alpha=0.9)
        
        for perm_idx, permuted_alphas in enumerate(permuted_results_list):
            perm_alpha = permuted_alphas[strategy_idx]
            perm_capital_ret = perm_alpha.get_zero_filtered_stats()["capital_ret"]
            perm_cum_returns = (1 + perm_capital_ret).cumprod()
            
            ax.plot(perm_cum_returns.index, perm_cum_returns.values, 
                    color='gray', linewidth=1, alpha=0.6, 
                    label='Permuted Data' if perm_idx == 0 else "")
        
        ax.set_title(f'{strategy_names[strategy_idx]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Returns', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Actual vs Permuted Data Performance Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('./images/actual_vs_permuted_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_results_table(alphas, strategy_names):
    """Create comprehensive results table with statistical tests"""
    all_results = []
    
    for alpha, name in zip(alphas, strategy_names):
        perf_stats = alpha.get_perf_stats(plot=False)
        hyp_tests = alpha.get_hypothesis_tests()
        
        zfs = alpha.get_zero_filtered_stats()
        capital_ret = zfs["capital_ret"]
        
        annual_return = capital_ret.mean() * 252
        annual_vol = capital_ret.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        
        cumulative = (1 + capital_ret).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        
        var_95 = np.percentile(capital_ret, 5)
        cvar_95 = capital_ret[capital_ret <= var_95].mean()
        
        skewness = capital_ret.skew()
        kurtosis = capital_ret.kurtosis()
        
        result_row = {
            'Strategy': name,
            'Annual Return': f"{annual_return:.4f}",
            'Annual Volatility': f"{annual_vol:.4f}",
            'Sharpe Ratio': f"{sharpe_ratio:.4f}",
            'Max Drawdown': f"{max_dd:.4f}",
            'Skewness': f"{skewness:.4f}",
            'Kurtosis': f"{kurtosis:.4f}",
            'VaR (95%)': f"{var_95:.6f}",
            'CVaR (95%)': f"{cvar_95:.6f}",
            'Sign Rank Test p-value': f"{hyp_tests['sign_rank']:.6f}",
            'Sign Test p-value': f"{hyp_tests['sign_test']:.6f}",
            'Asset Timing p-value': f"{hyp_tests['asset_timing']:.6f}",
            'Asset Picking p-value': f"{hyp_tests['asset_picking']:.6f}",
            'Skill Test 1 p-value': f"{hyp_tests['skill_1']:.6f}",
            'Skill Test 2 p-value': f"{hyp_tests['skill_2']:.6f}"
        }
        
        all_results.append(result_row)
    
    results_df = pd.DataFrame(all_results)
    return results_df

def main():
    
    # Create images directory
    Path("./images").mkdir(parents=True, exist_ok=True)
    
    # Set up time period
    period_start = datetime(2000, 1, 1, tzinfo=pytz.utc)
    period_end = datetime(2023, 1, 1, tzinfo=pytz.utc)
    
    # Load data
    print("Loading market data")
    tickers, ticker_dfs = get_ticker_dfs(start=period_start, end=period_end)
    tickers = tickers[:50]
    ticker_dfs = {ticker: ticker_dfs[ticker] for ticker in tickers}
    _, dataset = load_pickle("dataset.obj")
    for ticker in tickers:
        ticker_dfs.update({ticker+"_"+k: v for k, v in dataset[ticker].to_dict(orient="series").items()})
    
    # Get S&P 500 benchmark
    spy_returns = get_sp500_benchmark(period_start, period_end)
    spy_volatility = calculate_sp500_volatility_regime(spy_returns)
    
    print("Running strategy simulations")
    
    # Strategy 1: Volume-based Long-Short
    print("Running Gene 1: Volume-based Long-Short")
    g1 = Gene.str_to_gene("ls_25/75(neg(mean_12(cszscre(div(mult(volume,minus(minus(close,low),minus(high,close))),minus(high,low))))))")
    alpha1 = GeneticAlpha(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end, genome=g1)
    df1 = alpha1.run_simulation()
    
    # Apply transaction costs
    alpha1 = apply_transaction_costs_to_alpha(alpha1, spy_volatility)
    
    perf1 = alpha1.get_perf_stats(plot=False, gene_factor=1)
    print("Gene 1 Results:")
    print(df1)

    # Strategy 2: Open-Close Reversal
    print("Running Gene 2: Open-Close Reversal")
    g2 = Gene.str_to_gene("neg(mean_12(minus(const_1,div(open,close))))")
    alpha2 = GeneticAlpha(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end, genome=g2)
    df2 = alpha2.run_simulation()
    
    # Apply transaction costs
    alpha2 = apply_transaction_costs_to_alpha(alpha2, spy_volatility)
    
    perf2 = alpha2.get_perf_stats(plot=False, gene_factor=2)
    print("Gene 2 Results:")
    print(df2)

    # Strategy 3: Multi-Timeframe Momentum
    print("Running Gene 3: Multi-Timeframe Momentum")
    g3 = Gene.str_to_gene("plus(ite(gt(mean_10(close),mean_50(close)),const_1,const_0),ite(gt(mean_20(close),mean_100(close)),const_1,const_0),ite(gt(mean_50(close),mean_200(close)),const_1,const_0))")
    alpha3 = GeneticAlpha(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end, genome=g3)
    df3 = alpha3.run_simulation()
    
    # Apply transaction costs
    alpha3 = apply_transaction_costs_to_alpha(alpha3, spy_volatility)
    
    perf3 = alpha3.get_perf_stats(plot=False, gene_factor=3)
    print("Gene 3 Results:")
    print(df3)

    alphas = [alpha1, alpha2, alpha3]
    strategy_names = ['Gene 1: Volume-based Long-Short', 'Gene 2: Open-Close Reversal', 'Gene 3: Multi-Timeframe Momentum']
    
    print("Plotting combined returns with S&P 500 benchmark")
    plot_combined_returns_with_benchmark(alphas, strategy_names, spy_returns)
    
    print("Plotting return correlation heatmap")
    correlation_matrix = plot_return_correlation_heatmap(alphas, strategy_names)
    
    print("Plotting rolling drawdown")
    plot_rolling_drawdown(alphas, strategy_names)
    
    print("Plotting rolling Calmar ratio")
    plot_rolling_calmar(alphas, strategy_names)
    
    print("Plotting rolling volatility")
    plot_rolling_volatility(alphas, strategy_names)
    
    print("Plotting rolling Sharpe ratio")
    plot_rolling_sharpe(alphas, strategy_names)
    
    print("Plotting return distributions and QQ plots")
    plot_return_distributions_and_qq(alphas, strategy_names)
    
    ##
    try:
        permuted_datasets = load_pickle("permute.obj")
        
        selected_permuted = np.random.choice(len(permuted_datasets), 5, replace=False)
        permuted_results_list = []
        
        for perm_idx in selected_permuted:
            perm_ticker_dfs = permuted_datasets[perm_idx]
            
            for ticker in tickers:
                perm_ticker_dfs.update({ticker+"_"+k: v for k, v in perm_ticker_dfs[ticker].to_dict(orient="series").items()})
            
            perm_alpha1 = GeneticAlpha(insts=tickers, dfs=perm_ticker_dfs, start=period_start, end=period_end, genome=g1)
            perm_alpha2 = GeneticAlpha(insts=tickers, dfs=perm_ticker_dfs, start=period_start, end=period_end, genome=g2)
            perm_alpha3 = GeneticAlpha(insts=tickers, dfs=perm_ticker_dfs, start=period_start, end=period_end, genome=g3)
            
            permuted_results_list.append([perm_alpha1, perm_alpha2, perm_alpha3])
        
        print("   Plotting actual vs permuted performance")
        plot_permuted_vs_actual_performance(alphas, strategy_names, permuted_results_list)
        
    except FileNotFoundError:
        print("permute.obj not found.")
    
    print("Generating comprehensive results table")
    comprehensive_results = create_comprehensive_results_table(alphas, strategy_names)
    
    with pd.option_context('display.max_columns', None, 'display.width', None, 'display.max_colwidth', None):
        print(comprehensive_results.to_string(index=False))
    
    # comprehensive_results.to_csv('./results_with_statistical_tests.csv', index=False)
    
    # print("Generated files:")
    # print("- ./images/combined_returns_with_benchmark.png")
    # print("- ./images/correlation_heatmap.png") 
    # print("- ./images/rolling_drawdown.png")
    # print("- ./images/rolling_calmar.png")
    # print("- ./images/rolling_volatility.png")
    # print("- ./images/rolling_sharpe.png")
    # print("- ./images/return_distribution_qq_1.png")
    # print("- ./images/return_distribution_qq_2.png")
    # print("- ./images/return_distribution_qq_3.png")
    # if Path("permute.obj").exists():
    #     print("- ./images/actual_vs_permuted_performance.png")
    # print("- ./results_with_statistical_tests.csv")
    
    # print("\nStrategy Return Correlations:")
    # print(correlation_matrix)
    
    # print(f"\nTransaction Cost Regime Information:")
    # print(f"S&P 500 volatility mean: {spy_volatility.mean():.3f}")
    # print(f"S&P 500 volatility std: {spy_volatility.std():.3f}")
    # print("Transaction cost regimes applied:")
    # print("- Low vol (<0.15): 0.2% transaction cost")
    # print("- Normal (0.15-0.25): 0.35% transaction cost")
    # print("- Walking on ice (0.25-0.35): 0.4% transaction cost") 
    # print("- Crisis (>0.35): 0.5% transaction cost")
    
    # Calculate volatility regime distribution
    low_vol_pct = (spy_volatility < 0.15).sum() / len(spy_volatility) * 100
    normal_vol_pct = ((spy_volatility >= 0.15) & (spy_volatility < 0.25)).sum() / len(spy_volatility) * 100
    walking_ice_pct = ((spy_volatility >= 0.25) & (spy_volatility < 0.35)).sum() / len(spy_volatility) * 100
    crisis_pct = (spy_volatility >= 0.35).sum() / len(spy_volatility) * 100
    
    print(f"\nVolatility regime distribution over analysis period:")
    print(f"- Low vol periods: {low_vol_pct:.1f}%")
    print(f"- Normal vol periods: {normal_vol_pct:.1f}%") 
    print(f"- Walking on ice periods: {walking_ice_pct:.1f}%")
    print(f"- Crisis periods: {crisis_pct:.1f}%")
    
    return comprehensive_results

if __name__ == "__main__":
    results = main()

#### This is main