import yfinance as yf
import pandas as pd

def load_data(ticker, start_date, end_date=None, interval="1d"):
    """
    Load historical stock data from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Flatten column names if MultiIndex (remove ticker level)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    return data
