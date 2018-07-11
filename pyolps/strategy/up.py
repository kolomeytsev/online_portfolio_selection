import numpy as np
import tqdm


def up_next_weight(data, weight_o):
    """
    Generates UP portfolio for a specified parameter setting.
   
    :param data: market price ratios vectors
    :param weight_o: last portfolio, also can be last price relative adjusted.
    """
    
    del0 = 4e-3 # minimum coordinate
    delta = 5e-3 # spacing of grid
    M = 10 # number of samples
    S = 5 # number of steps in the random walk

    N = data.shape[1]

    # Computing Universal Portfolio.
    r = np.ones(N) / N # Start each one at the uniform point
    b = np.ones(r.shape[0])
    
    allM = np.zeros((N, M)) # Take the average of m samples
    for m in range(M):
        b = r.copy()
        for i in range(S):
            bnew = b.copy()
            j = np.random.randint(N - 1)
            a = np.random.choice([-1, 1])
            bnew[j] = b[j] + (a * delta)
            bnew[N - 1] = b[N - 1] - (a * delta)
            if bnew[j] >= del0 and bnew[N - 1] >= del0:
                muliplier_x = min(1, np.exp((b[N - 1] - (2 * del0)) / (N * delta)))
                x = np.prod(data @ b) * muliplier_x
                muliplier_y = min(1, np.exp((bnew[N - 1] - (2 * del0)) / (N * delta)))
                y = np.prod(data @ bnew) * muliplier_y
                pr = min(y / x, 1) # or pr = min(x / y, 1)
                if np.random.rand() < pr:
                    b = bnew.copy()
        allM[:, m] = b

    weight = np.mean(allM, 1) # Taking the average of m samples.
    return weight / sum(weight)


def up_run(data, tc, opts, verbose=False):
    """
    Universal Portfolios strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
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
        if t >= 1:
            day_weight = up_next_weight(data[:t], day_weight)

        daily_portfolio[t, :] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))

    return cum_ret, cumprod_ret, daily_ret, daily_portfolio

