import numpy as np
import pandas as pd
import datetime as dt
    
def compute_trade_result(close_prices: pd.DataFrame, date_exit: dt, position: tuple, position_info: list, trades: list, commission: float, exit_condition: str):
    i, j = position
    direction, date_entry, entry_zscore, i_amount, j_amount = position_info

    i_entry_price = close_prices.loc[date_entry, i]
    j_entry_price = close_prices.loc[date_entry, j]
    i_exit_price = close_prices.loc[date_exit, i]
    j_exit_price = close_prices.loc[date_exit, j]

    # direction = 1: go long i, short j / direction = -1: go short i, long j
    i_return = direction * (i_exit_price * i_amount - i_entry_price * i_amount)
    j_return = -direction * (j_exit_price * j_amount - j_entry_price * j_amount)

    slippage_entry = (i_amount * i_entry_price + j_amount * j_entry_price) * commission
    slippage_exit = (i_amount * i_exit_price + j_amount * j_exit_price) * commission
    total_slippage = slippage_entry + slippage_exit

    trades.append(((i, j), date_entry, date_exit, j_return+i_return, i_return, j_return, entry_zscore, total_slippage, exit_condition))
    return trades

def new_position(cointegrated_pairs: list, close_prices: pd.DataFrame, positions_dict: dict, date_entry: dt, pairs: tuple, zscore: float, capital_per_trade: float, budget: float):
    current_position_total = 0
    i, j = pairs
    beta = None
    for ci, cj, _, b, _ in cointegrated_pairs:
        if (ci, cj) == pairs:
            beta = b
    if beta is None:
        return positions_dict
    
    # Position size scaling with z-score magnitude
    scale = min(abs(zscore) / 2.0, 2.0)
    i_amount = int(scale * capital_per_trade / (close_prices.loc[date_entry, i] + beta * close_prices.loc[date_entry, j]))
    j_amount = int(scale * beta * capital_per_trade / (close_prices.loc[date_entry, i] + beta * close_prices.loc[date_entry, j]))

    new_position_value = close_prices.loc[date_entry, i] * i_amount + close_prices.loc[date_entry, j] * j_amount
    for position in positions_dict:
        i, j = position
        _, position_date_entry, _, position_i_amount, position_j_amount = positions_dict[position]
        current_position_total = current_position_total + position_i_amount * close_prices.loc[position_date_entry, i] + position_j_amount * close_prices.loc[position_date_entry, j]
    
    # prevents two simultaneous open positions for any single ticker
    if check_duplicates(pairs, positions_dict) != True and current_position_total + new_position_value <= budget:
        positions_dict[pairs] = [np.sign(zscore), date_entry, zscore, i_amount, j_amount]
    return positions_dict
        
def check_duplicates(pair: tuple, positions: dict):
    return any(bool(set(pair) & set(position)) for position in positions)

def run_backtest_rolling(cointegrated_pairs: list, trades: list, positions: dict, close_prices: pd.DataFrame, pairwise_zscores: dict, commission: float,
                         entry_condition: float, exit_condition: float, burst_condition: float, max_holding_days: int, position_size_pct: float, total_capital: float):
    # trades: list of (pair, entry_date, exit_date, total_return, i_return, j_return, entry_zscore, slippage, exit_type)
    # positions: {(i, j): [direction, entry_date, entry_zscore, i_amount, j_amount]}
    
    capital_per_trade = total_capital * position_size_pct
    test_dates = pairwise_zscores.index
    if len(test_dates) == 0:
        return trades
    
    # 1. Force-close positions for pairs that dropped out of the cointegrated set. 
    # Exit price is the first available close of the new month
    for position in list(positions):
        if position not in pairwise_zscores.columns:
            trades = compute_trade_result(close_prices, test_dates[0], position, positions[position], trades, commission, "cointegration_lost")
            del positions[position]

    for date in test_dates:

        # 2. Check exit for existing positions
        for position in list(positions):
            zscore = pairwise_zscores.at[date, position]
            direction = positions[position][0]
            date_entry = positions[position][1]
            holding_days = (date - date_entry).days
            
            if holding_days >= max_holding_days:
                trades = compute_trade_result(close_prices, date, position, positions[position], trades, commission, "time_limit")
                del positions[position]            
            elif (direction == 1 and zscore <= exit_condition or direction == -1 and zscore >= -exit_condition):
                trades = compute_trade_result(close_prices, date, position, positions[position], trades, commission, "exit")
                del positions[position]
            elif (direction == 1 and zscore >= burst_condition or direction == -1 and zscore <= -burst_condition):
                trades = compute_trade_result(close_prices, date, position, positions[position], trades, commission, "burst")
                del positions[position]
                
        # 3. Check for new entry signals (only when no ticker overlap)
        for pairs in pairwise_zscores:
            if abs(pairwise_zscores.at[date, pairs]) >= entry_condition:
                positions = new_position(cointegrated_pairs, close_prices, positions, date, pairs, pairwise_zscores.at[date, pairs], capital_per_trade, total_capital)
    return trades