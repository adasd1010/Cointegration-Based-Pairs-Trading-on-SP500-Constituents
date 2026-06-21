import pandas as pd
import numpy as np
from optuna_search import get_optimized_parameters
from data import download_data, read_data
from rolling import run_rolling
from pathlib import Path
from plot import plot_results

dir = Path(__file__).resolve().parent

def main(train_date_start: pd.DatetimeIndex, skip_download: bool, train_years: int, test_years: int, optuna_trials: int, show_optuna_progress_bar: bool = True, lookback_window: int = 252, recompute_frequency = 'MS', skip_optuna = True):
    train_date_start = pd.to_datetime(train_date_start)
    
    if not skip_download:
        download_data(train_date_start, train_years, test_years)

    test_date_start = train_date_start + pd.DateOffset(years = train_years)
    test_date_end = test_date_start + pd.DateOffset(years = test_years)
    
    # Load parameters from previous Optuna run if Optuna was skipped. Else, optimize parameters by running Optuna
    if not skip_optuna:
        # Optuna only sees the data up to the day before the test window opens; prevents any lookahead bias
        parameters = get_optimized_parameters("SP500_Close_Train", train_date_start, test_date_start - pd.Timedelta(days = 1), optuna_trials, show_optuna_progress_bar)
        parameters = list(parameters.values())  # convert Optuna dict to ordered list matching run_rolling's unpacking order
        parameters_df = pd.DataFrame([parameters])
        parameters_df.to_csv(dir/"Optuna_Search_Result", index = False)
    else:
        parameters = list(pd.read_csv(dir/"Optuna_Search_Result").iloc[0])

    price_data_test = read_data("SP500_Close_Test")
    test_trades = run_rolling(test_date_start = test_date_start,
                              test_date_end = test_date_end, 
                              price_data = price_data_test, 
                              params = parameters,
                              lookback_window = lookback_window,
                              recompute_freq = recompute_frequency)

    trades_df = pd.DataFrame(test_trades)
    trades_df.to_csv(dir/"Trades_Result", index = False)
    plot_results(test_trades)
    

main(train_date_start = "2012",
     skip_download = True, 
     train_years = 3, 
     test_years = 3, 
     optuna_trials = 500, 
     skip_optuna = True
     )