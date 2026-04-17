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

NIFTY100 = tickers

for i, ticker in enumerate(NIFTY100):
    if ticker.endswith('.NS'):
        continue
    else:
        NIFTY100[i] = ticker + '.NS'
