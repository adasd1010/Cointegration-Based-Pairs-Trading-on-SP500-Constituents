import requests
import yfinance as yf
import pandas as pd
from pathlib import Path

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

dir = Path(__file__).resolve().parent

def download_data(start_date: pd.Timestamp, train_years: int, test_years: int):
    table = pd.read_html(requests.get(url, headers={'User-agent':'Mozila/5.0'}).text)[0]
    tickers = table['Symbol'].tolist()
    
    # yfinance uses hyphens for tickers like 122870.KQ -> 122870-KQ. Wikipedia lists them with dots.
    tickers = [ticker.replace(".", "-") for ticker in tickers]
 
    start_date = pd.Timestamp(start_date)
    data = yf.download(tickers = tickers, start = start_date, end = start_date + pd.DateOffset(years = train_years), auto_adjust= True)
    
    # Any tickers with missing price data, even a single day, is dropped
    data = data.dropna(axis = 1)
    data["Close"].to_csv(dir/"SP500_Close_Train", index = True)

    # The test data starts at start_date instead of test_date_start so that the rolling backtest has a full lookback window
    data = yf.download(tickers = tickers, start = start_date, end = start_date + pd.DateOffset(years = train_years) + pd.DateOffset(years = test_years), auto_adjust= True)
    data = data.dropna(axis = 1)
    data["Close"].to_csv(dir/"SP500_Close_Test", index = True)

def read_data(filepath):
    return pd.read_csv(dir/filepath, index_col = 0, parse_dates = True)