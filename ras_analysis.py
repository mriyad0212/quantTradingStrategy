import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from quantlab.utils import load_pickle
from quantlab.gene import Gene, GeneticAlpha
from pathlib import Path

def construct_matrix_from_series(rets_list, window=63):
    """
    Construct standardized returns matrix X[T,N] where each entry is a 'Sharpe unit'
    """
    z_scores = []
    for s in rets_list:
        vol = s.rolling(window).std()
        z = s / vol
        z_scores.append(z)
    
    X = pd.concat(z_scores, axis=1).dropna()
    X.columns = [f'Strategy_{i+1}' for i in range(len(rets_list))]
    return X

def estimate_rademacher_complexity(X, n_sim=1000):
    """
    Empirical Rademacher complexity: E[max_n (epsilon^T x_n) / T]
    Measures how well strategies align with random noise directions
    """
    T, N = X.shape
    X_vals = X.values
    R_vals = []
    
    for _ in range(n_sim):
        eps = np.random.choice([1, -1], size=T)
        max_proj = np.max(eps @ X_vals) / T
        R_vals.append(max_proj)
    
    return np.mean(R_vals)

def ras_bounds(X, R_hat, delta=0.05):
    """
    Apply RAS bounds with conservative penalties
    Returns lower bound on true Sharpe ratio with probability 1-delta
    """
    T, N = X.shape
    theta_hat = X.mean(axis=0).values
    
    # Conservative penalty terms
    term_statistical = 3 * np.sqrt(2 * np.log(2 / delta) / T)
    term_multiple = np.sqrt(2 * np.log(2 * N / delta) / T)
    
    theta_rademacher = theta_hat - 2 * R_hat
    theta_lower = theta_hat - 2 * R_hat - term_statistical - term_multiple
    
    return pd.DataFrame({
        'empirical_sharpe': theta_hat,
        'rademacher_adjusted': theta_rademacher,
        'conservative_lower_bound': theta_lower,
        'rademacher_penalty': R_hat,
        'statistical_penalty': term_statistical,
        'multiple_testing_penalty': term_multiple
    }, index=X.columns)

def plot_sharpe_impact(bounds_df, strategy_names):
    """
    Plot the impact of RAS adjustments on Sharpe ratios
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Sharpe ratio comparison
    x_pos = np.arange(len(strategy_names))
    width = 0.25
    
    ax1.bar(x_pos - width, bounds_df['empirical_sharpe'], width, 
            label='Empirical Sharpe', alpha=0.8, color='blue')
    ax1.bar(x_pos, bounds_df['rademacher_adjusted'], width,
            label='Rademacher Adjusted', alpha=0.8, color='orange')
    ax1.bar(x_pos + width, bounds_df['conservative_lower_bound'], width,
            label='Conservative Lower Bound', alpha=0.8, color='red')
    
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Annualized Sharpe Ratio')
    ax1.set_title('RAS Impact on Sharpe Ratios')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Gene {i+1}' for i in range(len(strategy_names))], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Penalty breakdown
    penalties = bounds_df[['rademacher_penalty', 'statistical_penalty', 'multiple_testing_penalty']]
    penalties.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Penalty Magnitude')
    ax2.set_title('RAS Penalty Breakdown')
    ax2.set_xticklabels([f'Gene {i+1}' for i in range(len(strategy_names))], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./images/ras_sharpe_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_strategy_orthogonality(X):
    """
    Analyze how orthogonal/correlated the strategies are
    Higher correlation = lower Rademacher complexity
    """
    corr_matrix = X.corr()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Strategy Return Correlation Matrix\n(Higher correlation → Lower Rademacher complexity)')
    
    # Add correlation values as text
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='black')
    
    plt.xticks(range(len(corr_matrix)), [f'Gene {i+1}' for i in range(len(corr_matrix))])
    plt.yticks(range(len(corr_matrix)), [f'Gene {i+1}' for i in range(len(corr_matrix))])
    plt.tight_layout()
    plt.savefig('./images/strategy_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()
    return avg_correlation

def main():
    Path("./images").mkdir(parents=True, exist_ok=True)
    
    period_start = datetime(2000, 1, 1, tzinfo=pytz.utc)
    period_end = datetime(2023, 1, 1, tzinfo=pytz.utc)
    
    tickers, ticker_dfs = load_pickle("dataset.obj")
    tickers = tickers[:50]
    ticker_dfs = {ticker: ticker_dfs[ticker] for ticker in tickers}
    
    _, dataset = load_pickle("dataset.obj")
    for ticker in tickers:
        ticker_dfs.update({ticker+"_"+k: v for k, v in dataset[ticker].to_dict(orient="series").items()})
    
    strategies = [
        ("ls_25/75(neg(mean_12(cszscre(div(mult(volume,minus(minus(close,low),minus(high,close))),minus(high,low))))))", "Gene 1: Volume-based Long-Short"),
        ("neg(mean_12(minus(const_1,div(open,close))))", "Gene 2: Open-Close Reversal"),
        ("plus(ite(gt(mean_10(close),mean_50(close)),const_1,const_0),ite(gt(mean_20(close),mean_100(close)),const_1,const_0),ite(gt(mean_50(close),mean_200(close)),const_1,const_0))", "Gene 3: Multi-Timeframe Momentum")
    ]
    
    alphas = []
    rets_list = []
    strategy_names = []
    
    for i, (gene_str, name) in enumerate(strategies):
        gene = Gene.str_to_gene(gene_str)
        alpha = GeneticAlpha(insts=tickers, dfs=ticker_dfs, start=period_start, end=period_end, genome=gene)
        alpha.run_simulation()
        
        capital_ret = alpha.get_zero_filtered_stats()["capital_ret"]
        alphas.append(alpha)
        rets_list.append(capital_ret)
        strategy_names.append(name)
    
    X = construct_matrix_from_series(rets_list, window=63)
    avg_correlation = analyze_strategy_orthogonality(X)
    R_hat = estimate_rademacher_complexity(X, n_sim=1000)
    bounds_df = ras_bounds(X, R_hat, delta=0.05)
    bounds_df_annual = bounds_df * np.sqrt(252)
    
    print("RAS RESULTS (Annualized)")
    print("="*40)
    print(bounds_df_annual.round(4))
    
    for i, strategy_name in enumerate(strategy_names):
        row = bounds_df_annual.iloc[i]
        empirical = row['empirical_sharpe']
        conservative = row['conservative_lower_bound']
        reduction = empirical - conservative
        
        print(f"\n{strategy_name}:")
        print(f"  Empirical: {empirical:.3f}, Conservative: {conservative:.3f}, Reduction: {reduction:.3f}")
        
        if conservative > 0:
            print(f"  Status: RAS POSITIVE")
        elif row['rademacher_adjusted'] > 0:
            print(f"  Status: Rademacher Positive")
        else:
            print(f"  Status: Not significant")
    
    plot_sharpe_impact(bounds_df_annual, strategy_names)

if __name__ == "__main__":
    main() 