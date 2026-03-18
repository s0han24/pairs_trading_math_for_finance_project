import yfinance as yf
import pandas as pd


def download_prices(tickers, start="2018-01-01"):
    data = yf.download(tickers, start=start, auto_adjust=True)['Close']
    data = data.dropna(axis=1)

    return data
