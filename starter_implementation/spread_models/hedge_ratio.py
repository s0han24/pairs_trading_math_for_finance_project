from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def estimate_hedge_ratio(y, x):

    x = add_constant(x)
    model = OLS(y, x).fit()

    return model.params['LT.NS']
