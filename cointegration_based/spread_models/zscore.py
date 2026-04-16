import numpy as np
from sklearn.linear_model import LinearRegression

def zscore(series, window=60, method="simple"):

    if method == "simple":
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
    elif method == "ou":
        return ou_process_zscore(series, window=window)
    else:        
        raise ValueError(f"Unknown z-score method: {method}")

    return (series - mean) / std

def ou_process_zscore(series, window=60):
    """
    Compute the z-score of a series based on an Ornstein-Uhlenbeck process.
    This is a more sophisticated method that accounts for mean reversion.
    """
    # Estimate parameters of the OU process
    delta = series.diff().dropna()
    lagged = series.shift(1).dropna()
    
    # Fit a linear regression to estimate theta and mu
    model = LinearRegression().fit(lagged.values.reshape(-1, 1), delta.values)
    
    theta = -model.coef_[0]
    mu = model.intercept_ / (1 - theta)
    
    # Compute the z-score based on the OU process
    ou_mean = mu + (series - mu) * np.exp(-theta * window)
    ou_std = np.sqrt((1 - np.exp(-2 * theta * window)) / (2 * theta))
    
    return (series - ou_mean) / ou_std
