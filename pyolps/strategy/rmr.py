import numpy as np
import tqdm
from .. import utils


def compute_L1_median(X, maxiter=200, tol=1e-10):
    """ 
    Finds L1 median to historical prices
    
    :param X: prices
    :param maxiter: max number of iterations
    :param tol: toleration level
    """
    y = np.median(X, axis=0)
    y_last = None
    
    for i in range(maxiter):
        d = np.sqrt(((X - y)**2).sum(axis=1))
        if (d == 0).any():
            d = np.sqrt(((X - np.mean(X, axis=0))**2).sum(axis=1))

        y = (X / d[:,np.newaxis]).sum(0) / (1. / d).sum()
        
        if y_last is not None and \
            np.linalg.norm(y - y_last, ord=1) / np.linalg.norm(y_last, ord=1) <= tol:
            break
        
        y_last = y

    return y / X[-1]


def rmr_next_weight(data_close, data, t1, day_weight, w, epsilon):
    """
    Calculates the day_weight at the t + 1 day with robust mean reversion method
   
    :param data:
    :param t1: new day 
    :param day_weight: weight before the new day
    :param w: length of window
    :param epsilon: parameter to control the reversion threshold
    
    Returns:
    :day_weight: weight at the new day
    """
    
    if t1 + 1 < w + 2:
        x_t1 = data[t1 - 1]
    else:
        x_t1 = compute_L1_median(data_close[t1 - w: t1])

    denom = (np.linalg.norm(x_t1 - np.mean(x_t1)))**2 
    if denom == 0:
        alpha = 0
    else:
        alpha = min(0, (np.dot(x_t1, day_weight) - epsilon) / denom)

    alpha = min(100000, alpha)
    weight = day_weight - alpha * (x_t1 - np.mean(x_t1))
    weight = utils.simplex_projection(weight)
    return weight / sum(weight)


def rmr_run(data, tc, opts, epsilon=10, window=30, verbose=False):
    """
    Robust Mean Reversion strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param epsilon: reversion threshold
    :param window: window size for computing median
    :param verbose: enable verbose output
    
    Returns:
    :cum_ret: cumulative wealth achived at the end of a period.
    :cumprod_ret: cumulative wealth achieved till the end each period.
    :daily_ret: daily return achieved by a strategy.
    :daily_portfolio: daily portfolio, achieved by the strategy
    """
    n, m = data.shape
    cum_ret = 1
    cumprod_ret = np.ones(n)
    daily_ret = np.ones(n)
    day_weight = np.ones(m) / m
    day_weight_o = np.zeros(m)
    day_weight_n = np.zeros(m)

    daily_portfolio = np.zeros((n, m))
    data_phi = np.zeros(m)
    turno = 0
    # close prices according to ratios
    data_close = np.ones((n, m))
    for i in range(1, n):
        data_close[i] = data_close[i - 1] * data[i]

    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        
        daily_portfolio[t] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]
                               
        if t != n - 1:
            day_weight_n = rmr_next_weight(data_close, data, t + 1, 
                                           day_weight, window, epsilon)
       
            turno = turno + sum(abs(day_weight_n - day_weight))
            day_weight = day_weight_n

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio, turno

