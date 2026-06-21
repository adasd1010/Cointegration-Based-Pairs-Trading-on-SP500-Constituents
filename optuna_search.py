import optuna
import numpy as np
import pandas as pd
from rolling import run_rolling
from data import read_data

def objective(trial: optuna.Trial, price_data_train: pd.DataFrame, test_date_start, test_date_end):
    test_date_start = pd.to_datetime(test_date_start)
    test_date_end = pd.to_datetime(test_date_end)

    # Set bounds and optimize seven parameters
    correlation_threshold = trial.suggest_float("correlation_threshold", 0.75, 0.95)
    p_threshold = trial.suggest_float("p_threshold", 0.01, 0.05)
    entry = trial.suggest_float("entry_condition", 1.5, 3.5)

    # Bounds relative to entry, preventing degenerate combos (e.g., exit > entry)
    exit_ = trial.suggest_float("exit_condition", 0.1, entry - 0.5)
    burst = trial.suggest_float("burst_condition", entry + 0.5, entry + 1.5)
    max_holding_days = trial.suggest_int("max_holding_days", 15, 60)
    position_size_pct = trial.suggest_float("position_size_pct", 0.01, 0.20)

    params = [correlation_threshold, p_threshold, entry, exit_, burst, max_holding_days, position_size_pct]
    trades = run_rolling(test_date_start, test_date_end, price_data_train, params)
    trades_df = pd.DataFrame(trades, columns=['pair', 'entry_date', 'exit_date', 'total_return', 'i_return', 'j_return', 'entry_zscore', 'slippage', 'exit_type'])
    trades_df['net_return'] = trades_df['total_return'] - trades_df['slippage']
    daily_pnl = trades_df.groupby('exit_date')['net_return'].sum()
    
    # Zero-fill the training periods, including periods before the first trade.
    daily_pnl = daily_pnl.reindex(pd.date_range(start = test_date_start, end = test_date_end), fill_value = 0)
    
    sharpe = daily_pnl.mean()/daily_pnl.std() * np.sqrt(252)

    # Penalize degenerate trials: too few trades means no meaningful sample. std == 0 would cause division by zero
    if len(trades_df) <= 10 or daily_pnl.std() == 0:
        return -np.inf
    return sharpe

def get_optimized_parameters(filepath, test_date_start: pd.DatetimeIndex, test_date_end: pd.DatetimeIndex, n_trials: int, show_optuna_progress_bar: bool = True):
    price_data_train = read_data(filepath)
    study = optuna.create_study(direction="maximize", study_name = "Optimized Parameters")
    study.optimize(lambda trial: objective(trial, price_data_train, test_date_start, test_date_end), n_trials = n_trials, show_progress_bar = show_optuna_progress_bar, n_jobs = -1)

    best_parameters = study.best_params
    return best_parameters
