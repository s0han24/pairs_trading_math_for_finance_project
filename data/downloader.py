import yfinance as yf
import pandas as pd


def download_price_data(tickers, start="2014-01-01", end="2026-01-01"):
    print("Downloading price data...")
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    prices = prices.dropna(axis=1)
    return prices
