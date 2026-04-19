## Results:
For k=5
| Strategy | Final Capital | Max Drawdown | Sharpe Ratio | Alpha | Beta | Annual Volatility |
| --- | --- | --- | --- | --- | --- | --- |
| EG-Pearson-staticOU-SR-Sorted-StopLoss | 315.47 | -26.72% | 0.65 | 0.1858 | -0.1059 | 24.74% |
| EG-Pearson-SR-Sorted | 307.17 | -36.22% | 0.86 | 0.1522 | -0.0479 | 16.45% |
| EG-Pearson-SR-Sorted-StopLoss | 307.17 | -36.22% | 0.86 | 0.1522 | -0.0479 | 16.45% |
| Johansen-Pearson-SR-Sorted | 204.66 | -66.45% | 0.43 | 0.1273 | -0.0397 | 27.17% |
| Johansen-Pearson-SR-Sorted-StopLoss | 204.66 | -66.45% | 0.43 | 0.1273 | -0.0397 | 27.17% |
| EG-Pearson-staticOU-SR-Sorted | 167.24 | -49.71% | 0.35 | 0.2465 | -0.4320 | 41.16% |
| Johansen-Pearson-staticOU-SR-Sorted-StopLoss | 161.83 | -32.39% | 0.35 | 0.0767 | 0.0289 | 24.08% |
| Johansen-Pearson-staticOU-SR-Sorted | 133.08 | -40.59% | 0.26 | 0.0776 | -0.0139 | 28.99% |

probability cutoff = 0.0
| Strategy | Final Capital | Max Drawdown | Sharpe Ratio | Alpha | Beta | Annual Volatility |
| --- | --- | --- | --- | --- | --- | --- |
| ML-XGB-k5 | 473.00 | -11.54% | 1.61 | 0.1735 | 0.0386 | 11.38% |
| ML-RAF-k5 | 284.10 | -18.99% | 1.16 | 0.1197 | 0.0188 | 10.70% |
| ML-Ensemble-k5 | 256.90 | -30.45% | 1.02 | 0.1067 | 0.0270 | 11.03% |
| ML-DNN-k5 | 188.54 | -33.31% | 0.73 | 0.0796 | -0.0086 | 10.64% |

probability cutoff = 0.5
| Strategy | Final Capital | Max Drawdown | Sharpe Ratio | Alpha | Beta | Annual Volatility |
| --- | --- | --- | --- | --- | --- | --- |
| ML-XGB-k5 | 399.90 | -12.71% | 1.43 | 0.1578 | 0.0253 | 11.48% |
| ML-Ensemble-k5 | 291.74 | -25.52% | 1.16 | 0.1182 | 0.0391 | 11.01% |
| ML-RAF-k5 | 243.29 | -18.83% | 0.99 | 0.1073 | -0.0025 | 10.82% |
| ML-DNN-k5 | 164.23 | -16.14% | 0.58 | 0.0605 | 0.0059 | 10.68% |

Baseline results:
| Strategy | Final Capital | Max Drawdown | Sharpe Ratio | Alpha | Beta | Annual Volatility |
| --- | --- | --- | --- | --- | --- | --- |
| Greedy | 128580.36 | -52.93% | 1.41 | 0.7207 | 1.1792 | 62.49% |
| EW_BuyHold | 1633.68 | -36.52% | 1.41 | 0.1744 | 0.9764 | 21.81% |
| EqualWeight | 1534.34 | -37.37% | 1.39 | 0.1656 | 0.9923 | 21.63% |
| Market | 440.72 | -38.44% | 0.97 | 0.0290 | 1.0024 | 17.15% |
| 40-60 | 344.56 | -22.71% | 1.26 | 0.0539 | 0.5789 | 10.57% |
