import numpy as np
import tqdm


def bah_run(data, tc, opts, day_weight=None, verbose=False):
    """
    Buy And Hold strategy.
    If day_weight is None then performs Uniform BAH.
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param day_weight: portfolio vector
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
    
    if day_weight is None:
        day_weight = np.ones(m) / m
    else:
        day_weight = np.array(day_weight)
    day_weight_o = np.zeros(m)
    daily_portfolio = np.zeros((n, m))
    
    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')
    
    for t in tqdm.trange(n):
        if t >= 1:
            day_weight = day_weight_o / sum(day_weight_o)

        daily_portfolio[t, :] = day_weight

        tc_coef = (1 - tc  * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio

