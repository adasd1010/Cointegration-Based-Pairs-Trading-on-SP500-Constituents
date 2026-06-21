import numpy as np
import pandas as pd

def cut_price_data(price_data, end_date, lookback_window = 252):
    end_date = pd.Timestamp(end_date)
    
    # Delete lower-volume share classes of dual class pairs (GOOG/GOOGL, FOX/FOXA, NWS/NWSA)
    duplicates = {"GOOG", "FOX", "NWS"}
    price_data = price_data.drop(columns = duplicates, errors = 'ignore')
    price_data = price_data.loc[:end_date].iloc[-lookback_window:]
    return price_data

def get_correlation_matrix(price_data, end_date, lookback_window = 252):
    end_date = pd.Timestamp(end_date)
    data_close = cut_price_data(price_data, end_date, lookback_window)
    
    # Log returns for approximate stationarity and scale-invariance.
    data_log = np.log(data_close).diff().dropna()
    correlation_matrix = data_log.corr()
    return correlation_matrix, data_close

def get_correlation_pairs(correlation_matrix, data_close, threshold = 0.85):
    pairs = []
    for i in range(len(data_close.columns)):
        for j in range(i+1, len(data_close.columns)):
            if correlation_matrix.iloc[i,j] >= threshold and i != j:
                pairs.append((data_close.columns[i], data_close.columns[j]))
    return pairs
