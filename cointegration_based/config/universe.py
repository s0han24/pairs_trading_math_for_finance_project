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

tickers = pd.read_html('https://ournifty.com/stock-list-in-nse-fo-futures-and-options.html#:~:text=NSE%20F%26O%20Stock%20List%3A%20%20%20%20SL,%20%201000%20%2052%20more%20rows%20')[0]

NIFTY50 = tickers.SYMBOL.to_list()

for i, ticker in enumerate(NIFTY50):
    if ticker.endswith('.NS'):
        continue
    else:
        NIFTY50[i] = ticker + '.NS'
