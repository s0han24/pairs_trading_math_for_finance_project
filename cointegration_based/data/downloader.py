import yfinance as yf
import pandas as pd


def download_prices(tickers, start="2022-01-01", end="2024-01-01"):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    data = data.dropna(axis=1)
    
    ood_data = yf.download(tickers, start="2024-01-01", end="2026-01-01", auto_adjust=True)['Close']
    ood_data = ood_data.dropna(axis=1)
    
    return data, ood_data
