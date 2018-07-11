import numpy as np
import tqdm


def anticor_expert(data, weight_o, w):
    """
    Generates portfolio for a specified parameter setting.
   
    :param data: last market price ratios vector
    :param last_weight: last portfolio, also can be last price relative adjusted.
    """
    T, N = data.shape
    weight = weight_o.copy()

    if T < 2 * w:
        return weight

    LX_1 = np.log(data[T - 2 * w: T - w])
    LX_2 = np.log(data[T - w: T])
            
    mu_1 = np.mean(LX_1, axis=0)
    mu_2 = np.mean(LX_2, axis=0)
    M_cov = np.zeros((N, N))
    M_cor = np.zeros((N, N))
    n_LX_1 = LX_1 - np.repeat(mu_1.reshape(1, -1), w, axis=0)
    n_LX_2 = LX_2 - np.repeat(mu_2.reshape(1, -1), w, axis=0)
    
    Std_1 = np.diag(n_LX_1.T @ n_LX_1) / (w - 1)
    Std_2 = np.diag(n_LX_2.T @ n_LX_2) / (w - 1)
    Std_12 = Std_1.reshape(-1, 1) @ Std_2.reshape(1, -1)
    M_cov = n_LX_1.T @ n_LX_2 / (w - 1)
    
    M_cor[Std_12 == 0] = 0
    
    M_cor[Std_12 != 0] = M_cov[Std_12 != 0] / np.sqrt(Std_12[Std_12 != 0])
    
    claim = np.zeros((N, N))
    w_mu_2 = np.repeat(mu_2.reshape(-1, 1), N, axis=1)
    w_mu_1 = np.repeat(mu_2.reshape(1, -1), N, axis=0)
    
    s_12 = (w_mu_2 >= w_mu_1) & (M_cor > 0)
    claim[s_12] += M_cor[s_12]
    
    diag_M_cor = np.diag(M_cor)
    
    cor_1 = np.repeat(-diag_M_cor.reshape(-1, 1), N, axis=1)
    cor_2 = np.repeat(-diag_M_cor.reshape(1, -1), N, axis=0)
    cor_1 = np.maximum(0, cor_1)
    cor_2 = np.maximum(0, cor_2)
    claim[s_12] += cor_1[s_12] + cor_2[s_12]
    
    transfer = np.zeros((N, N))
    sum_claim = np.repeat(np.sum(claim, axis=1).reshape(-1, 1), N, axis=1)
    s_1 = np.abs(sum_claim) > 0
    w_weight_o = np.repeat(weight_o.reshape(-1, 1), N, axis=1)
    transfer[s_1] = w_weight_o[s_1] * claim[s_1] / sum_claim[s_1]
    
    transfer_ij = transfer.T - transfer
    weight -= np.sum(transfer_ij, axis=0)
        
    return weight


def anticor_kernel(data, window, exp_ret, exp_w):
    """
    kernel for BAH(Anticor) strategy.
    BAH(Anticor) has one folds of experts.
    
    :param data: market sequence vectors
    :param window: maximum window size, the number of experts (window-1)
    :param exp_ret: experts' return in the first fold
    :param exp_w: experts' weights in the first fold
    
    Returns:
    :weight: final portfolio, used for next rebalance
    :exp_w: experts weights in the first fold
    """
    for k in range(window - 1):
        exp_w[k] = anticor_expert(data, exp_w[k], k + 2)
    # combine portfolios
    numerator = 0.
    denominator = 0.
    for k in range(window - 1):
        numerator += exp_ret[k] * exp_w[k]
        denominator += exp_ret[k]
    weight = numerator / denominator
    return weight / sum(weight), exp_w


def anticor_run(data, tc, opts, window=30, verbose=False):
    """
    BAH(Anticor) strategy.
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param window: window size
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
    
    # Variables for expert
    exp_ret = np.ones(window - 1)
    exp_w = np.ones((window - 1, m)) / m
    
    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        if t >= 1:
            day_weight, exp_w = anticor_kernel(data[:t], window, exp_ret, exp_w)
        
        daily_portfolio[t] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret

        day_weight_o = day_weight * data[t] / daily_ret[t]
        
        for k in range(window - 1):
            exp_ret[k] *= data[t] @ exp_w[k]
        exp_ret = exp_ret / sum(exp_ret)

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio, exp_ret


def anticor_anticor_kernel(data, window, exp_ret, exp_w, 
                                data_day, exp_ret2, exp_w2):
    """
    Fernel for BAH(Anticor(Anticor)) strategy
    BAH(Anticor()) has two folds of experts
    
    :param data: market sequence vectors
    :param window: maximum window size, the number of experts (window-1)
    :param exp_ret: experts' return in the first fold
    :param exp_w: experts' weights in the first fold
    
    Returns:
    :weight: final portfolio, used for next rebalance
    :exp_w: experts weights in the first fold
    """
    for k in range(window - 1):
        exp_w[k] = anticor_expert(data, exp_w[k], k + 2)

    for k in range(window - 1):
        exp_w2[k] = anticor_expert(data_day, exp_w2[k], k + 2)
    # combine portfolios
    numerator = 0.
    denominator = 0.
    for k in range(window - 1):
        numerator += exp_ret2[k] * exp_w2[k]
        denominator += exp_ret2[k]
    weight1 = numerator / denominator
    weight = exp_w.T @ weight1
    return weight / sum(weight), exp_w, exp_w2


def anticor_anticor_run(data, tc, opts, window=30, verbose=False):
    """
    BAH(Anticor(Anticor)) strategy.
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param window: window size
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
    
    # Variables for expert
    exp_ret = np.ones(window - 1)
    exp_w = np.ones((window - 1, m)) / m
    
    exp_ret2 = np.ones(window - 1)
    exp_w2 = np.ones((window - 1, window - 1)) / (window - 1)
    
    data_day = np.zeros((n, window - 1))
    
    if verbose:
        print('Parameters [tc: %f].\n' % tc)
        print('day\t Daily Return\t Total return\n')

    for t in tqdm.trange(n):
        if t >= 1:
            day_weight, exp_w, exp_w2 = anticor_anticor_kernel(data[:t], window, 
                                    exp_ret, exp_w, data_day[:t], exp_ret2, exp_w2)
        
        daily_portfolio[t] = day_weight

        tc_coef = (1 - tc * sum(abs(day_weight - day_weight_o)))
        daily_ret[t] = (data[t] @ day_weight) * tc_coef
        cum_ret = cum_ret * daily_ret[t]
        cumprod_ret[t] = cum_ret
        if cum_ret < 0:
            print("something went wrong: cum_ret < 0")
            return None

        day_weight_o = day_weight * data[t] / daily_ret[t]

        data_day[t] = data[t].reshape(1, -1) @ exp_w.T
        
        for k in range(window - 1):
            exp_w[k] = exp_w[k] * data[t] / data_day[t, k]
            exp_ret2[k] *= data_day[t] @ exp_w2[k]
            exp_w2[k] *= data_day[t] / (data_day[t] @ exp_w2[k])

        if verbose:
            if (t + 1) % opts['display_interval'] == 0:
                print('%d\t%f\t%f\n' % (t + 1, daily_ret[t], cumprod_ret[t]))
    
    return cum_ret, cumprod_ret, daily_ret, daily_portfolio, exp_ret, exp_ret2

