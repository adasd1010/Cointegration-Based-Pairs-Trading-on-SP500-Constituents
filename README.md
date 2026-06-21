# Cointegration-Based Pairs Trading on the S&P 500

**A full writeup of this project is available in the PDF in this repository.**

## Overview
A statistical pairs trading strategy on S&P 500 constituents. Pairs are selected by a correlation filter and a cointegration (ADF) test, sized with beta-weighted hedge ratios, and traded on the z-score of the cointegrated spread. Hyperparameters are tuned with Bayesian optimization (Optuna), and the strategy is evaluated out-of-sample across three market regimes with monthly walk-forward computation.

## Methodology
- **Pair Selection** — Used log-return correlation filter, Augmented Dickey-Fuller cointegration test on OLS regression residuals.
- **Signal & Sizing** — Entries on z-score exceeding predefined threshold, position size scales with the z-score.
- **Exits** — Four types: Exit (profit-taking), Burst (stop-loss), Time Limit, and Cointegration Lost
- **Rolling** — Correlation, cointegration, and hedge ratios recomputed monthly.

## Results
Out-of-sample, $1,000,000 initial capital:

| Metric | 2015–2018 | 2018–2021 | 2021–2024 |
| ------ | --------- | --------- | --------- |
| Net P&L (USD) | -84,779 | 626,921 | 896,732 |
| Maximum Drawdown (USD) | -505,115 | -272,114 | -990,206 |
| Return / Maximum Drawdown | -0.17 | 2.30 | 1.00 |
| Sharpe Ratio | -0.15 | 0.98 | 0.44 |
| Trades | 13,033 | 628 | 2,146 |
| Win Rate | 56.49% | 59.90% | 54.75% |

Profitable in two of three periods. The edge is heavily regime-dependent, concentrated in volatile markets. The 2015-2018 loss reflects an over-optimized parameter set bleeding out in transaction costs.

## Tech Stack
Python · `pandas` · `numpy` · `statsmodels` · `Optuna` · `matplotlib` · `yfinance`
