import pandas as pd
from data import read_data
from correlation import get_correlation_matrix, get_correlation_pairs
from cointegration import get_cointegrated_pairs
from zscore import compute_pairwise_zscores
from backtest import run_backtest_rolling

def run_rolling (test_date_start, test_date_end, price_data, params, lookback_window = 252, recompute_freq = 'MS'):
    trades = []
    
    # Pairs that remain cointegrated carry their open positions in this dict across all recompute iterations
    positions = {}

    correlation_threshold, p_threshold, entry_condition, exit_condition, burst_condition, max_holding_days, position_size_pct = params
    
    # 'MS' is pandas Month Start — recompute dates fall on the 1st calendar day of each month
    test_date_start = pd.to_datetime(test_date_start)
    test_date_end = pd.to_datetime(test_date_end)
    # 1. Build the list of monthly recompute dates spanning the test period
    recompute_dates = pd.date_range(test_date_start, test_date_end, freq = recompute_freq)

    for recompute_date in recompute_dates:
        # 2. Recompute correlation and cointegration on the trailing window as of this date
        correlation_matrix, price_data_lookback_window = get_correlation_matrix(price_data, recompute_date, lookback_window)
        correlation_pairs = get_correlation_pairs(correlation_matrix, price_data_lookback_window, threshold = correlation_threshold)
        cointegrated_pairs = get_cointegrated_pairs(price_data_lookback_window, correlation_pairs, p_threshold = p_threshold, end_date = recompute_date, lookback_window = lookback_window)
        
        # 3. Compute z-scores from this recompute date forward (used until the next recompute)
        pairwise_zscores = compute_pairwise_zscores(price_data, cointegrated_pairs, recompute_date)
       
        # 4. Force-close any open positions whose pair dropped out of the cointegration test this month
        trades = run_backtest_rolling(cointegrated_pairs = cointegrated_pairs,
                                      trades = trades,
                                      positions = positions,
                                      close_prices = price_data,
                                      pairwise_zscores = pairwise_zscores,
                                      commission = 0.0005,
                                      entry_condition = entry_condition,
                                      exit_condition = exit_condition,
                                      burst_condition = burst_condition,
                                      max_holding_days = max_holding_days,
                                      position_size_pct = position_size_pct,
                                      total_capital = 1000000)
    return trades
