import pandas as pd

NIFTY100_small = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "ITC.NS",
    "LT.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "HINDUNILVR.NS",
]

tickers = pd.read_csv("ind_nifty100list.csv")['Symbol'].tolist()

NIFTY50 = tickers

for i, ticker in enumerate(NIFTY50):
    if ticker.endswith('.NS'):
        continue
    else:
        NIFTY50[i] = ticker + '.NS'
