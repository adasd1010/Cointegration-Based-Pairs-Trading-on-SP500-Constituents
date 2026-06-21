import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def get_cointegrated_pairs(price_data, correlation_pairs, p_threshold, end_date, lookback_window = 252):
    end_date = pd.Timestamp(end_date)
    unique_cols = list(dict.fromkeys(ticker for pair in correlation_pairs for ticker in pair))
    data = price_data[unique_cols] 
    data = data.loc[:end_date].iloc[-lookback_window:]

    cointegrated_pairs = []

    for i, j in correlation_pairs:
        best = None
        best_pvalue = p_threshold
        
        # Both directions, Y ~ X and X ~ Y, is tested to find the orientation that produces a lower p-value
        for dependent, independent in [(j, i), (i, j)]:
            Xt = data[independent]
            Yt = data[dependent]
            regression = sm.OLS(Yt, sm.add_constant(Xt)).fit()

            alpha = regression.params["const"]
            beta = regression.params[independent]

            residual = Yt - (alpha + beta * Xt)
            p_value = adfuller(residual)[1]

            # abs(beta) > 0.1 is tested to guard against cases where estimated hedge ratio is near zero
            if p_value < best_pvalue and abs(beta) > 0.1:
                best_pvalue = p_value
                best = (independent, dependent, alpha, beta, residual)
        if best is not None:
            cointegrated_pairs.append(best)
    return cointegrated_pairs