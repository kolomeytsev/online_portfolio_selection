import numpy as np
import tqdm
from .. import utils


def olmar1_next_weight(data, weight_o, epsilon, window):
    """
    Generates portfolio for a specified parameter setting.
   
    :param x: last market price ratios vector
    :param weight_o: last portfolio, also can be last price relative adjusted.
    """
    
    T, N = data.shape

    if T < window + 1:
        data_phi = data[T - 1]
    else:
        data_phi = np.zeros(N)
        tmp_x = np.ones(N)
        for i in range(1, window + 1):
            data_phi = data_phi + 1. / tmp_x
            tmp_x = tmp_x * data[T - i]
        data_phi /= window

    # Step 3: Suffer loss
    ell = max([0, epsilon - data_phi @ weight_o])

    # Step 4: Set parameter
    x_bar = np.mean(data_phi)    
    denom_part = data_phi - x_bar
    denominator = np.dot(denom_part, denom_part)
    if denominator != 0:
        lmbd = ell / denominator
    else:
        #Zero volatility
        lmbd = 0

    # Step 5: Update portfolio
    weight = weight_o + lmbd * (data_phi - x_bar)

    # Step 6: Normalize portfolio
    weight = utils.simplex_projection(weight)
    
    return weight / sum(weight)


def olmar1_run(data, tc, opts, epsilon=10, window=5, verbose=False):
    """
    Online Newton Step variant 1 strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param epsilon: mean reversion threshold
    :param window: window size for moving average
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
    daily_portfolio = np.zeros((n, m))
    
    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        if t >= 2:
            day_weight = olmar1_next_weight(data[:t], day_weight, epsilon, window)
        
        daily_portfolio[t] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio


def olmar2_next_weight(x, data_phi, weight_o, epsilon, alpha):
    """
    Generates portfolio for a specified parameter setting.
   
    :param x: last market price ratios vector
    :param data_phi: last moving average
    :param weight_o: last portfolio.
    """
    
    N = x.shape[0]

    data_phi = alpha + (1 - alpha) * data_phi / x

    # Step 3: Suffer loss
    ell = max([0, epsilon - data_phi @ weight_o])

    # Step 4: Set parameter
    x_bar = np.mean(data_phi)
    denom_part = data_phi - x_bar
    denominator = np.dot(denom_part, denom_part)
    if denominator != 0:
        lmbd = ell / denominator
    else:
        #Zero volatility
        lmbd = 0

    # Step 5: Update portfolio
    weight = weight_o + lmbd * (data_phi - x_bar)

    # Step 6: Normalize portfolio
    weight = utils.simplex_projection(weight)
    
    return weight / sum(weight), data_phi


def olmar2_run(data, tc, opts, epsilon=10, alpha=0.5, verbose=False):
    """
    Online Newton Step variant 2 strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param epsilon: mean reversion threshold
    :param alpha: trade off parameter for calculating moving average
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
    daily_portfolio = np.zeros((n, m))
    data_phi = np.zeros(m)
    
    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        if t >= 1:
            day_weight, data_phi = olmar2_next_weight(data[t - 1], data_phi, 
                                                      day_weight, epsilon, alpha)
        
        daily_portfolio[t] = day_weight
        
        if (day_weight < -0.00001).any() or \
            (day_weight.sum() > 1.00001):
            print('mrpa_expert: t=%d, sum(day_weight)=%d, ending' % (t, day_weight.sum()))
            return None

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio


def olmar_run(data, tc, opts, variant=1, epsilon=10, 
                window=5, alpha=0.5, verbose=False):
    """
    Online Newton Step strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param variant: variant of algorithm: 1 or 2
    :param epsilon: mean reversion threshold
    :param window: window size for moving average
    :param alpha: trade off parameter for calculating moving average
    :param verbose: enable verbose output
    
    Returns:
    :cum_ret: cumulative wealth achived at the end of a period.
    :cumprod_ret: cumulative wealth achieved till the end each period.
    :daily_ret: daily return achieved by a strategy.
    :daily_portfolio: daily portfolio, achieved by the strategy
    """

    if variant == 1:
        res = olmar1_run(data, tc, opts, epsilon, window, verbose)
    elif variant == 2:
        res = olmar2_run(data, tc, opts, epsilon, alpha, verbose)
    else:
        print("Wrong variant parameter. Must be 1 or 2. Returning None.")
        res = None
    return res

