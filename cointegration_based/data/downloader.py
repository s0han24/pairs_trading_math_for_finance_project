import yfinance as yf
import pandas as pd


def download_prices(tickers, start_in="2022-01-01", end_in="2024-01-01", start_ood="2024-01-01", end_ood="2026-01-01"):
    data = yf.download(tickers, start=start_in, end=end_in, auto_adjust=True)['Close']
    data = data.dropna(axis=1)
    
    ood_data = yf.download(tickers, start=start_ood, end=end_ood, auto_adjust=True)['Close']
    ood_data = ood_data.dropna(axis=1)
    
    return data, ood_data
