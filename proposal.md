# Pairs Trading Framework

### Problem Setup:
A pairs trading strategy identifies two co-moving assets that have temporarily diverged in price. The strategy is to go long on the underperformer and short on the overperformer, betting on the mean reverting property of the assets. Another component of the strategy is risk management to cap losses.

### Objectives:
We propose implementing a pairs trading framework which consists of 3 main components:
- Identifying co-moving pairs
- Deciding position sizes
- Risk management to prevent catastrophic losses


### Baselines

### Co-Movement Identification:

### Deciding Position Sizes

### Risk Management Mathods

### Backtesting


### TODO: add the following
1. Proposed Methodology 
- detecting co-moving assets
- strategies' details
- objectives: return optimization, sharpe ratio optimization, CVaR constrained minimisation(maybe?)
- Risk management strategies: basically how we decide to pull out if things don't go well
2. Evaluation & Baselines
 * Baseline 1
 * Baseline 2
 * Evaluation metrics: VaR, CVaR, returns, sharpe ratio, etc
3. Backtesting and other implementation details
- what data are we using?
- what consists of our world?
- how are we doing backtests, reasons why they cover am exhaustive set of scenarios

