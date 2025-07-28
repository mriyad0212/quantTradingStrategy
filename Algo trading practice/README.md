# Algorithmic Trading Practice

A collection of quantitative trading strategies implemented in Python.

## Current Strategies

### Mean Reversion Strategy (`mean-reversion.py`)

A mean reversion trading strategy that trades GDX (VanEck Gold Miners ETF) based on price deviations from the 25-day Simple Moving Average.

**Strategy Logic:**
- **Long Signal**: Enter long position when price is >3.5 points below the 25-day SMA
- **Short Signal**: Enter short position when price is >3.5 points above the 25-day SMA  
- **Exit Signal**: Close position when price crosses back through the SMA (mean reversion)

**Performance (2013-2023):**
- Mean Reversion Strategy: +72.68% return
- Buy & Hold: -33.62% return
- Outperformed by: 106.3 percentage points

**Key Features:**
- Uses log returns for statistical stability
- Implements position sizing (-1 for short, 0 for neutral, 1 for long)
- Visualizes price distance from SMA and trading positions
- Compares strategy performance vs buy-and-hold

## Requirements

```bash
pip install yfinance pandas numpy matplotlib seaborn
```

## Usage

```bash
python mean-reversion.py
```

This will:
1. Download GDX data from Yahoo Finance
2. Calculate technical indicators (SMA, distance from SMA)
3. Generate trading signals based on mean reversion logic
4. Display performance plots and statistics
5. Show cumulative returns comparison

## Files

- `mean-reversion.py` - Main mean reversion strategy implementation
- `mean_reversion_fixed.py` - Version with enhanced output and saved plots
- Generated PNG files with strategy visualizations

## Strategy Explanation

The mean reversion strategy is based on the assumption that asset prices tend to revert to their historical average over time. When prices deviate significantly from their moving average, the strategy takes positions expecting the price to revert back to the mean.

This particular implementation:
1. Uses a 25-day Simple Moving Average as the "mean"
2. Sets thresholds at ±3.5 points from the SMA
3. Takes contrarian positions when thresholds are breached
4. Exits when prices revert back through the SMA

## Risk Disclaimer

This code is for educational and research purposes only. Past performance does not guarantee future results. Trading involves substantial risk of loss.
