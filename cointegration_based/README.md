# Pairs Trading Strategy with Cointegration

This project implements a pairs trading strategy based on the concept of cointegration. The strategy identifies pairs of stocks whose prices have a long-term, stable relationship and then trades on the short-term deviations from this relationship.

## Project Structure

The project is organized into the following modules:

-   **main.py**: The entry point of the project. It orchestrates the entire workflow, from data downloading to backtesting.
-   **`data/`**: Contains the data downloading module.
-   **`config/`**: Contains configuration files, including the stock universe (universe.py) and strategy settings (`settings.py`).
-   **`pairs/`**: Contains modules for finding and filtering cointegrated pairs.
-   **`spread_models/`**: Contains modules for modeling the spread between pairs.
-   **`strategy/`**: Contains the trading signal generation module.
-   **`backtest/`**: Contains modules for backtesting the strategy and calculating performance metrics.

## Workflow

1.  **Configuration**: In `config/settings.py`, you can configure the cointegration test to be used.

2.  **Data Downloading**: The `data/downloader.py` module downloads historical stock prices from Yahoo Finance for a given list of tickers and a specified time period. The data is split into in-sample (for finding pairs) and out-of-sample (for backtesting) sets.

3.  **Pair Selection**:
    -   The `pairs/filtering.py` module first filters pairs of stocks based on their correlation. This is a quick way to identify pairs that are likely to be cointegrated.
    -   The `pairs/cointegration.py` module then uses the selected cointegration test (Engle-Granger or Johansen) to find pairs of stocks that have a statistically significant long-term relationship. This is configured in `config/settings.py`.

4.  **Spread Modeling**:
    -   The `spread_models/hedge_ratio.py` module estimates the hedge ratio (beta) between the two stocks in a cointegrated pair using Ordinary Least Squares (OLS).
    -   The `spread_models/spread.py` module computes the spread between the two stocks using the estimated hedge ratio. The spread is the difference between the price of one stock and the price of the other stock multiplied by the hedge ratio.
    -   The `spread_models/zscore.py` module calculates the z-score of the spread. The z-score measures how many standard deviations the current spread is from its moving average.

5.  **Signal Generation**: The `strategy/signals.py` module generates trading signals based on the z-score of the spread.
    -   When the z-score exceeds a certain positive threshold, it's a signal to short the spread (i.e., sell the first stock and buy the second).
    -   When the z-score falls below a certain negative threshold, it's a signal to long the spread (i.e., buy the first stock and sell the second).
    -   When the z-score reverts to zero, the position is closed.

6.  **Backtesting**:
    -   The `backtest/backtesting.py` module backtests the trading strategy on the out-of-sample data.
    -   The `backtest/metrics.py` module calculates performance metrics for the strategy, such as Sharpe ratio, maximum drawdown, and cumulative returns.

## How to Run

1.  **Configure the Strategy**: Open `cointegration_based/config/settings.py` and set your desired `COINT_TEST_METHOD`.
    ```python
    # Options: 'engle-granger', 'johansen'
    COINT_TEST_METHOD = 'engle-granger'
    ```

2.  **Execute the Backtest**: Run the main.py file:
    ```bash
    python main.py
    ```

The main.py file contains two test cases:

1.  **COVID Crash Period**: This test uses data from the COVID-19 crash period to check if the filtering mechanism can identify good pairs in a volatile market.
2.  **Pre-COVID Period**: This test uses data from a more stable pre-COVID period to see if the strategy can achieve better performance in a different market regime.

## Dependencies

The project requires the following Python libraries:

-   `yfinance`
-   `pandas`
-   `numpy`
-   `statsmodels`
-   `matplotlib`

You can install them using pip:

```bash
pip install yfinance pandas numpy statsmodels matplotlib
```
