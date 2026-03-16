def zscore(series, window=60):

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()

    return (series - mean) / std
