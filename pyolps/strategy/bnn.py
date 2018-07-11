import numpy as np
import tqdm
from .. import utils


def bnn_expert(data, k, pl):
    """
    Generates portfolio for a specified parameter setting.
    
    :param data: market sequence vectors
    :param k: sequence length
    :param pl: parameter to control the no. of nearest neighbors
    """
    T, N = data.shape
    m = 0
    histdata = np.zeros((T, N)) 
    normid = np.zeros(T)

    if T <= k + 1:
        weight = np.ones(N) / N
        return weight

    if k == 0 and pl == 0:
        histdata = data[:T]
        m = T
    else:
        histdata = data[:T]
        normid[:k] = np.inf
        for i in range(k, T):
            data2 = data[i - k: i - 1] - data[T - k + 1: T]
            normid[i] = np.sqrt(np.trace(data2 @ data2.T))

        sortpos = np.argsort(normid)
        m = int(np.floor(pl * T))
        histdata = histdata[sortpos[:m]]

    if m == 0:
        weight = np.ones(N) / N
        return weight

    weight = utils.optimize_weights(histdata[:m])

    return weight / sum(weight)
    
    
def bnn_kernel(data, k, l, exp_ret, exp_w):
    """
    Generates portfolio the BNN strategy.
   
    :param data: market price ratios vectors
    :param k: sequence length
    :param l: number of nearest neighbors
    :exp_ret: experts return
    :exp_w: experts weights
    
    Returns:
    :weight: final portfolio, used for next rebalance
    :exp_w: today's individual expert's portfolio
    """
    exp_w[k * l] = bnn_expert(data, 0, 0)
    for k_index in range(k):
        for l_index in range(l):
            pl = 0.02 + 0.5 * l_index / (l - 1)
            exp_w[k_index * l + l_index] = bnn_expert(data, k_index + 1, pl)

    # Combine portfolios according to q(k, l) and previous expert return    
    q = 1 / (k * l + 1)
    numerator = q * exp_ret[0, l] * exp_w[k * l]
    denominator = q * exp_ret[0, l]
    for k_index in range(k):
        for l_index in range(l):
            numerator += q * exp_ret[k_index, l_index] * exp_w[k_index * l + l_index]
            denominator += q * exp_ret[k_index, l_index]

    weight = numerator / denominator    
    return weight / sum(weight), exp_w


def bnn_run(data, tc, opts, k=5, l=10, verbose=False):
    """
    BNN strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param k: sequence length
    :param l: number of nearest neighbors
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

    exp_ret = np.ones((k, l + 1))
    exp_w = np.ones((k * (l + 1), m)) / m

    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        
        if t >= 1:
            day_weight, exp_w = bnn_kernel(data[:t], k, l, exp_ret, exp_w)

        daily_portfolio[t] = day_weight
        
        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = np.dot(data[t], day_weight) * tc_coef
        
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]
        # experts return
        for k_index in range(k):
            for l_index in range(l):
                exp_ret[k_index, l_index] *= np.dot(data[t], exp_w[k_index * l + l_index])

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio, exp_ret

