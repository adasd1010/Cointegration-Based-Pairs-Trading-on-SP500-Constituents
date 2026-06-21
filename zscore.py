import pandas as pd
import datetime


def compute_pairwise_zscores(price_data:pd.DataFrame, cointegrated_pairs: list, test_dates_start: pd.DatetimeIndex):

    unique_cols = list(dict.fromkeys(ticker for i, j, alpha, beta, residual in cointegrated_pairs for ticker in (i, j)))
    price_data = price_data[unique_cols]

    pair_cols = [(i, j) for i, j, a, b, r in cointegrated_pairs]
    test_dates = price_data.index[price_data.index >= test_dates_start]

    # Pre-seeds the 30 trading days before the test period begins for the rolling window
    rolling_date_start = price_data.index[price_data.index < test_dates_start][-30:]
    rolling_dates = rolling_date_start.append(test_dates)

    results = {}
    # alpha and beta are fixed from the lookback-window OLS regression. Only the rolling mean/std adapt within this month
    for i, j, a, b, r in cointegrated_pairs:
        residual = price_data[j] - (a + b * price_data[i])
        residual_rolling = residual.loc[rolling_dates]
        roll = residual_rolling.rolling(30)
        zscore = (residual_rolling - roll.mean())/roll.std()
        results[(i,j)] = zscore
        
    pairwise_zscores = pd.DataFrame(results, index = price_data.index)
    pairwise_zscores.columns = pd.Index(pair_cols)
    return pairwise_zscores.loc[test_dates]