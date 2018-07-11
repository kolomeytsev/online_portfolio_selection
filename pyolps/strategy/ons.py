import numpy as np
import tqdm
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False


def find_projection_to_simplex(x, M):
    n = M.shape[0]
    P = matrix(2 * M)
    q = matrix(-2 * M @ x)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros((n, 1)))
    A = matrix(np.ones((1, n)))
    b = matrix(1.)
    sol = solvers.qp(P, q, G, h, A, b)
    return np.squeeze(sol['x'])


def ons_next_weight(x, last_weight, A, b, eta, beta, delta):
    """
    Generates portfolio for a specified parameter setting.
   
    :param x: last market price ratios vector
    :param last_weight: last portfolio, also can be last price relative adjusted.
    """
    N = len(x)
    grad = x / (x @ last_weight)
    grad = grad.reshape(-1, 1)
    hessian = -1 * (grad @ grad.T)
    A += (-1) * hessian
    b += (1 + 1 / beta) * grad

    A_inv = np.linalg.inv(A)
    weight = find_projection_to_simplex(delta * A_inv @ b, A)
    weight = (1 - eta) * weight + eta * np.ones(N) / N
    return weight / sum(weight)


def ons_run(data, tc, opts, eta=0., beta=1., delta=0.125, verbose=False):
    """
    Online Newton Step strategy. 
    
    :param data: market price ratios vectors
    :param tc: transaction fee rate
    :param opts: option parameter for behvaioral control
    :param eta: mixture parameter
    :param beta: tradeoff parameter
    :param delta: heuristic tuning parameter
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
    
    A = np.eye(m)
    b = np.zeros((m, 1))
    for t in tqdm.trange(n):
        if t >= 1:
            day_weight = ons_next_weight(data[t - 1], day_weight, A, b, 
                                               eta, beta, delta)
        
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

