# QF621 Quantitative Finance Project

## Overview
This project implements and analyzes three genetic programming-based trading strategies using comprehensive statistical testing and visualization.

## Features

### 1. Trading Strategies
- **Gene 1**: Volume-based Long-Short Strategy
- **Gene 2**: Open-Close Reversal Strategy  
- **Gene 3**: Multi-Timeframe Momentum Strategy

### 2. Analysis Components
- **Return Correlation Heatmap**: Visualizes correlation between strategy returns
- **Permuted Data Testing**: Tests strategy performance against 1000 randomly permuted datasets
- **Statistical Significance Tests**: Multiple tests including Wilcoxon signed-rank, asset timing/picking tests
- **Performance Comparison**: Actual vs permuted data performance visualization

### 3. Generated Outputs
- Strategy performance plots
- Correlation heatmap (`./images/correlation_heatmap.png`)
- Actual vs permuted performance comparison (`./images/actual_vs_permuted_performance.png`) 
- Comprehensive results table (`./results_with_statistical_tests.csv`)

## Quick Start

### Option 1: Run Complete Analysis
```bash
python run_analysis.py
```
This will automatically:
1. Check requirements
2. Generate permuted datasets (if needed)
3. Run all three strategies
4. Create correlation heatmap
5. Compare actual vs permuted performance
6. Generate statistical tests

### Option 2: Run Individual Components

#### Generate Permuted Datasets
```bash
python permute_data.py
```

#### Run Main Analysis
```bash
python main.py
```

## File Structure
- `main.py` - Main analysis script with all three strategies
- `permute_data.py` - Generates 1000 permuted datasets for statistical testing
- `run_analysis.py` - Comprehensive runner script
- `dataset.obj` - Historical stock data
- `quantlab/` - Quantitative analysis library
- `images/` - Generated visualization outputs

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- quantlab (included)

## Statistical Tests Included
- **Wilcoxon Signed-Rank Test**: Tests if median return > 0
- **Sign Test**: Tests if proportion of positive returns > 50%
- **Asset Timing Test**: Tests if market timing adds value
- **Asset Picking Test**: Tests if stock selection adds value  
- **Combined Skill Tests**: Tests overall strategy skill vs random chance

## Interpretation Guide
- **Lower p-values** in sign/signed-rank tests = more significant positive returns
- **Higher p-values** in timing/picking tests = no skill detected
- **Red line vs gray lines** in permuted plots shows actual vs random performance
- **Correlation heatmap** shows diversification benefits between strategies
