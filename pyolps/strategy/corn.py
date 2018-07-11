import numpy as np
import tqdm
from .. import utils


def corn_expert(data, w, c):
    """
    Generates portfolio for a specified parameter setting.
    
    :param data: market sequence vectors
    :param w: window size
    :param c: correlation coefficient threshold
    """
    T, N = data.shape
    if T <= w:
        weight = np.ones(N) / N
        return weight

    if w == 0:
        histdata = data.copy()
    else:
        indices = []
        d2 = data[-w:].flatten()
        for i in range(w, T):
            d1 = data[i - w:i].flatten()
            if np.corrcoef(d1, d2)[0, 1] >= c:
                indices.append(i)
        histdata = data[indices]

    if len(histdata) == 0:
        weight = np.ones(N) / N
    else:
        weight = utils.optimize_weights(histdata)
    return weight / sum(weight)


def corn_run(data, tc, opts, w=5, c=0.1, verbose=False):
    """
    Correlation-drivven Nonparametric Learning strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param w: window size
    :param c: correlation coefficient threshold
    :param verbose: enable verbose output
    
    Returns:
    :cum_ret: cumulative wealth achived at the end of a period.
    :cumprod_ret: cumulative wealth achieved till the end each period.
    :daily_ret: daily return achieved by a strategy.
    :daily_portfolio: daily portfolio, achieved by the strategy
    :exp_ret: individual experts return
    """
    n, m = data.shape
    cum_ret = 1
    cumprod_ret = np.ones(n)
    daily_ret = np.ones(n)
    day_weight = np.ones(m) / m
    day_weight_o = np.zeros(m)
    daily_portfolio = np.zeros((n, m))

    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        
        if t >= 1:
            day_weight = corn_expert(data[:t], w, c)
            
        daily_portfolio[t] = day_weight
        
        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = data[t] @ day_weight * tc_coef
        
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))

    return cum_ret, cumprod_ret, daily_ret, daily_portfolio

