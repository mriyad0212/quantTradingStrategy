import numpy as np
import pandas as pd

def performance_measures(r, plot=False, path="/images", annualization_factor=253, gene_factor=None):
    moment = lambda x, k: np.mean((x - np.mean(x)) ** k)
    stdmoment = lambda x, k: moment(x, k) / np.maximum(moment(x, 2) ** (k / 2), 1e-10)  # Avoid division by zero

    # Calculate cumulative returns and log returns
    cr = np.cumprod(1 + r)
    lr = np.log(cr)

    # Calculate drawdowns
    mdd = cr / cr.cummax() - 1
    rdd_fn = lambda cr, pr: cr / cr.rolling(pr).max() - 1
    rmdd_fn = lambda cr, pr: rdd_fn(cr, pr).rolling(pr).min()

    # Sortino ratio (return / downside volatility)
    downside_returns = r.values[r.values < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.nan
    srtno = np.mean(r.values) / np.maximum(downside_std, 1e-10) * np.sqrt(annualization_factor)

    # Sharpe ratio (return / volatility)
    shrpe = np.mean(r.values) / np.maximum(np.std(r.values), 1e-10) * np.sqrt(annualization_factor)

    # Basic return statistics
    mu1 = np.mean(r) * annualization_factor
    med = np.median(r) * annualization_factor
    stdev = np.std(r) * np.sqrt(annualization_factor)
    var_stat = stdev ** 2
    skw = stdmoment(r, 3)
    exkurt = stdmoment(r, 4) - 3

    # CAGR (Compound Annual Growth Rate)
    cagr_fn = lambda cr: (cr[-1] / cr[0]) ** (1 / len(cr)) - 1 if len(cr) > 0 and cr[0] > 0 else np.nan
    cagr_ann_fn = lambda cr: ((1 + cagr_fn(cr)) ** annualization_factor) - 1 if not np.isnan(cagr_fn(cr)) else np.nan
    cagr = cagr_ann_fn(cr)

    # Rolling CAGRg
    rcagr = cr.rolling(5 * annualization_factor).apply(cagr_ann_fn, raw=True)

    # Calmar ratio (return / max drawdown)
    roll_cagr = cr.rolling(3 * annualization_factor).apply(cagr_ann_fn, raw=True)
    roll_mdd = rmdd_fn(cr=cr, pr=3 * annualization_factor)
    calmar = roll_cagr / (roll_mdd * -1)

    # Value at Risk (VaR) - typically using the 5th percentile for losses
    var95 = np.percentile(r, 5)  # Changed from 95 to 5 for proper VaR
    cvar = r[r <= var95].mean() if len(r[r <= var95]) > 0 else np.nan  # Changed from < to <= and proper handling

    if plot:
        import os
        from pathlib import Path
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.gridspec import GridSpec

        # Create the output directory if it doesn't exist
        Path(os.path.abspath(os.getcwd() + path)).mkdir(parents=True, exist_ok=True)

        # Add gene factor to path if provided
        gene_suffix = f"_gene{gene_factor}" if gene_factor is not None else ""

        # Set the style for all plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]

        # 1. Cumulative Returns Plot
        plt.figure()
        plt.plot(cr, linewidth=2)
        plt.title(f'Cumulative Returns{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                  fontsize=14)
        plt.ylabel('Growth of $1')
        plt.grid(True)
        plt.savefig(f".{path}/cumulative_returns{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Return Distribution
        plt.figure()
        ax = sns.histplot(r, stat="probability", kde=True)
        ax.axvline(x=np.mean(r), linestyle="-", color='red', label=f"Mean: {np.mean(r):.4f}")
        ax.axvline(x=np.median(r), linestyle="dotted", color='blue', label=f"Median: {np.median(r):.4f}")
        ax.axvline(x=var95, linestyle="dashed", color='black', label=f"VaR (5%): {var95:.4f}")
        plt.title(f'Return Distribution{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                  fontsize=14)
        plt.legend()
        plt.savefig(f".{path}/return_distribution{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Log Returns
        plt.figure()
        plt.plot(lr, linewidth=1.5)
        plt.title(f'Cumulative Log Returns{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                  fontsize=14)
        plt.ylabel('Log Returns')
        plt.grid(True)
        plt.savefig(f".{path}/log_returns{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Drawdown Analysis
        plt.figure()
        plt.plot(mdd, linewidth=1.5, color='red')
        plt.fill_between(mdd.index, mdd, 0, color='red', alpha=0.3)
        plt.title(f'Drawdown Analysis{" for Gene " + str(gene_factor) if gene_factor is not None else ""}', fontsize=14)
        plt.ylabel('Drawdown (%)')
        plt.grid(True)
        plt.savefig(f".{path}/drawdowns{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Rolling Drawdown
        plt.figure()
        plt.plot(rdd_fn(cr, annualization_factor), label='Rolling Drawdown (1Y)', linewidth=1)
        plt.plot(rmdd_fn(cr, annualization_factor), label='Rolling Max Drawdown (1Y)', linewidth=2)
        plt.title(
            f'Rolling Drawdowns (1 Year Window){" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
            fontsize=14)
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f".{path}/rolling_drawdowns{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 6. Rolling Volatility
        plt.figure()
        rolling_vol = r.rolling(window=21).std() * np.sqrt(annualization_factor)
        plt.plot(rolling_vol, linewidth=1.5)
        plt.title(
            f'21-Day Rolling Annualized Volatility{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
            fontsize=14)
        plt.ylabel('Annualized Volatility')
        plt.grid(True)
        plt.savefig(f".{path}/rolling_volatility{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 7. Rolling CAGR
        plt.figure()
        plt.plot(rcagr, linewidth=1.5)
        plt.title(f'5-Year Rolling CAGR{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                  fontsize=14)
        plt.ylabel('CAGR (%)')
        plt.grid(True)
        plt.savefig(f".{path}/rolling_cagr{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 8. Rolling Sharpe Ratio
        plt.figure()
        rolling_mean = r.rolling(window=annualization_factor).mean() * annualization_factor
        rolling_std = r.rolling(window=annualization_factor).std() * np.sqrt(annualization_factor)
        rolling_sharpe = rolling_mean / rolling_std
        plt.plot(rolling_sharpe, linewidth=1.5)
        plt.title(f'1-Year Rolling Sharpe Ratio{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                  fontsize=14)
        plt.ylabel('Sharpe Ratio')
        plt.grid(True)
        plt.savefig(f".{path}/rolling_sharpe{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 9. Calmar Ratio
        plt.figure()
        plt.plot(calmar, linewidth=1.5)
        plt.title(f'3-Year Calmar Ratio{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                  fontsize=14)
        plt.ylabel('Calmar Ratio')
        plt.grid(True)
        plt.savefig(f".{path}/calmar_ratio{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 10. Monthly Returns Heatmap
        if len(r) > 30:  # Only if we have enough data
            plt.figure(figsize=(12, 8))
            returns_monthly = r.resample('M').apply(lambda x: (1 + x).prod() - 1)

            # Create dataframe with year and month as separate columns
            monthly_data = pd.DataFrame({
                'returns': returns_monthly.values,
                'year': returns_monthly.index.year,
                'month': returns_monthly.index.month
            })

            # Create pivot table for heatmap
            monthly_returns = monthly_data.pivot_table(
                index='month',
                columns='year',
                values='returns',
                aggfunc='first'
            )

            # Replace month numbers with month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_returns.index = [month_names[i - 1] for i in monthly_returns.index]

            # Create heatmap
            sns.heatmap(monthly_returns, annot=True, fmt=".2%", cmap="RdYlGn", center=0, linewidths=1, cbar=True)
            plt.title(f'Monthly Returns (%){" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                      fontsize=14)
            plt.savefig(f".{path}/monthly_returns{gene_suffix}.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 11. Return QQ Plot
        plt.figure()
        from scipy import stats
        fig = plt.figure(figsize=(10, 7))
        stats.probplot(r.dropna(), dist="norm", plot=plt)
        plt.title(
            f'Return QQ Plot (Normal Distribution){" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
            fontsize=14)
        plt.savefig(f".{path}/return_qqplot{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 12. Performance Dashboard - Combined plot with key metrics
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)

        # Cumulative returns subplot
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(cr, linewidth=2)
        ax1.set_title(f'Cumulative Returns{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                      fontsize=12)
        ax1.grid(True)

        # Return distribution subplot
        ax2 = fig.add_subplot(gs[1, 0])
        sns.histplot(r, stat="density", kde=True, ax=ax2)
        ax2.axvline(x=np.mean(r), linestyle="-", color='red')
        ax2.axvline(x=np.median(r), linestyle="dotted", color='blue')
        ax2.set_title(f'Return Distribution{" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
                      fontsize=12)

        # Drawdown subplot
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(mdd, linewidth=1.5, color='red')
        ax3.fill_between(mdd.index, mdd, 0, color='red', alpha=0.3)
        ax3.set_title(f'Drawdowns{" for Gene " + str(gene_factor) if gene_factor is not None else ""}', fontsize=12)
        ax3.grid(True)

        # Rolling metrics subplot
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(rolling_vol, linewidth=1.5, label='Rolling Volatility')
        ax4.set_title(
            f'Rolling Annualized Volatility (21d){" for Gene " + str(gene_factor) if gene_factor is not None else ""}',
            fontsize=12)
        ax4.grid(True)
        ax4.legend()

        # Key statistics text box
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        metrics_text = (
            f"Annualized Return: {mu1:.2%}\n"
            f"Annualized Volatility: {stdev:.2%}\n"
            f"Sharpe Ratio: {shrpe:.2f}\n"
            f"Sortino Ratio: {srtno:.2f}\n"
            f"CAGR: {cagr:.2%}\n"
            f"Max Drawdown: {np.min(mdd):.2%}\n"
            f"Calmar Ratio: {np.nanmean(calmar):.2f}\n"
            f"Skewness: {skw:.2f}\n"
            f"Excess Kurtosis: {exkurt:.2f}\n"
            f"VaR (5%): {var95:.2%}\n"
            f"CVaR (5%): {cvar:.2%}"
        )
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
                 verticalalignment='top', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f".{path}/performance_dashboard{gene_suffix}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    metrics_dict = {
        "cum_ret": cr,
        "log_ret": lr,
        "max_dd": mdd,
        "cagr": cagr,
        "srtno": srtno,
        "sharpe": shrpe,
        "mean_ret": mu1,
        "median_ret": med,
        "vol": stdev,
        "var_stat": var_stat,
        "skew": skw,
        "exkurt": exkurt,
        "rcagr": rcagr,
        "calmar": calmar,
        "var95": var95,
        "cvar": cvar
    }

    scalar_metrics = {}
    time_series_metrics = {}

    for key, value in metrics_dict.items():
        # Check if the value is a scalar or time series (pandas Series)
        if isinstance(value, (pd.Series, np.ndarray)) and hasattr(value, '__len__') and len(value) > 1:
            time_series_metrics[key] = value
        else:
            scalar_metrics[key] = value

    # Create a DataFrame for scalar metrics
    perf_df = pd.DataFrame(scalar_metrics, index=['Value']).T

    print(f"Performance metrics calculated for {len(r)} trading days.")
    print(perf_df)

    return metrics_dict