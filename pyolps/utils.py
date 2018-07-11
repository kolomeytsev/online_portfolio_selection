import numpy as np
import scipy.optimize as optimize


def optimize_weights(X, **kwargs):
    """Finds best constant rebalanced portfolio weights.
    :param X: Prices in ratios
    :params kwargs: additional parameters to scipy optimizer.
    """

    x_0 =  np.ones(X.shape[1]) / float(X.shape[1])
    objective = lambda b: -np.sum(np.log(np.maximum(np.dot(X - 1, b) + 1, 0.0001)))
    cons = ({'type': 'eq', 'fun': lambda b: 1 - sum(b)},)
    while True:
        res = optimize.minimize(objective, x_0, bounds=[(0., 1.)] * len(x_0), 
                                constraints=cons, method='slsqp', **kwargs)
        # result can be out-of-bounds -> try it again
        EPS = 1e-7
        if (res.x < 0. - EPS).any() or (res.x > 1. + EPS).any():
            X = X + np.random.randn(1)[0] * 1e-5
            print('Optimal weights not found, trying again...')
            continue
        elif res.success:
            break
        else:
            if np.isnan(res.x).any():
                print('Solution does not exist, use zero weights.')
                res.x = np.zeros(X.shape[1])
            else:
                print('Converged, but not successfully.')
            break

    return res.x


def simplex_projection(v, b=1):
    """
    Simplex_Projection Projects point onto simplex of specified radius.
    w = simplex_projection(v, b) returns the vector w which is the solution
    to the following constrained minimization problem:
        
        min   ||w - v||_2
        s.t.  sum(w) <= b, w >= 0.

    That is, performs Euclidean projection of v to the positive simplex of
    radius b.
    """
    if b < 0:
        print('Radius of simplex is negative: %2.3f\n', b)
        return None

    v = (v > 0) * v
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    z = (u > (sv - b) / range(1, len(u) + 1))
    non_neg_indices = np.argwhere(z != 0)
    if len(non_neg_indices):
        rho = non_neg_indices[-1, -1]   
    else:
        rho = 0
    theta = np.maximum(0, (sv[rho] - b) / (rho + 1))
    return np.maximum(v - theta, 0)


def make_ratios(data):
    """
    Converts prices to ratios.
    
    :param data: numpy.ndarray containing prices.
    """
    x0 = np.ones(data.shape[1])
    x = data[1:] / data[:-1]
    return np.vstack((x0, x))

