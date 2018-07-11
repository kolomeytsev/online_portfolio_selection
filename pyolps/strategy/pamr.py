import numpy as np
import tqdm
from .. import utils


def pamr_expert(x, weight_o, eta):
    """
    Generates portfolio for a specified parameter setting.
   
    :param x: last market price ratios vector
    :param weight_o: last portfolio
    :param eta: lagarange multiplier
    """
    weight = weight_o - eta * (x - np.mean(x))
    weight = utils.simplex_projection(weight)
    if (weight < -0.00001).any() or (weight > 1.00001).any():
        str_print = 'pamr_expert: t=%d, sum(weight)=%f, returning uniform weights'
        print(str_print % (t, weight.sum()))
        return np.ones(len(weight_o)) / len(weight_o)

    return weight / sum(weight)


def pamr_run(data, tc, opts, eps=0.5, C=500, variant=0, verbose=False):
    """
    Passive Aggressive Mean Reversion strategy.
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param eps: mean reversion threshold
    :param C: aggressive parameter for variant 1 and 2
    :param variant: variants of strategy 0, 1 or 2
    :param verbose: enable verbose output
    
    Returns:
    :cum_ret: cumulative wealth achived at the end of a period.
    :cumprod_ret: cumulative wealth achieved till the end each period.
    :daily_ret: daily return achieved by a strategy.
    :daily_portfolio: daily portfolio, achieved by the strategy
    :exp_ret: experts' returns in the first fold
    """
    n, m = data.shape
    cum_ret = 1
    cumprod_ret = np.ones(n)
    daily_ret = np.ones(n)
    day_weight = np.ones(m) / m
    day_weight_o = np.zeros(m)
    daily_portfolio = np.zeros((n, m))
    eta = np.inf
    
    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        if t >= 2:
            day_weight = pamr_expert(data[t - 1], day_weight, eta)
        
        daily_portfolio[t] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = np.dot(data[t], day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]
        
        # update lagrange multiplier
        denom_part = data[t] - np.mean(data[t])
        if variant == 0:
            denominator = np.dot(denom_part, denom_part)
        elif variant == 1:
            denominator = np.dot(denom_part, denom_part)
        elif variant == 2:
            denominator = np.dot(denom_part, denom_part)
        else:
            print("Wrong variant parameter: must be 0, 1 or 2. Exiting.")
            return None
        if denominator != 0:
            eta = (daily_ret[t] - eps) / denominator
        eta = max(0, eta)
        eta = min(1e10, eta)

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio

