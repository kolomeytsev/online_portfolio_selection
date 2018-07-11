import os
from agentnet.display import Metrics
from sklearn.base import BaseEstimator

import lasagne
import theano
import theano.tensor as T

import numpy as np
import pickle
from itertools import chain
from functools import reduce
from operator import add

#number of "tradeable" days in the year
NB_DAYS_IN_YEAR = 260
# Span for ewma
VOL_WINDOW_SPAN = 10
# transaction cost
COST_COEFFICIENT = 1e-4

class PortfolioNN(BaseEstimator):

    def __init__(self, X_inputs, instrument_returns, portfolio_layers,
                 asset_classes = ["assets"], params = None, optimizer=lasagne.updates.adam,
                 training_costs=COST_COEFFICIENT, l2_penalty = None,
                 name="RRL", updates_callback=lambda:{}, **kwargs):
        super(PortfolioNN,self).__init__()
        self.name= name
        self.metrics = Metrics()
        self.epoch=0
        self.params =  params or lasagne.layers.get_all_params(portfolio_layers, trainable=True)
        self.portfolio_weights = lasagne.layers.get_output(portfolio_layers)
        self.portfolio_weights = dict(zip(asset_classes, self.portfolio_weights))
        instrument_returns = dict(zip(asset_classes, instrument_returns))
        self.returns = get_portfolio_returns_symb(instrument_returns, self.portfolio_weights,
                                                     cost_coefficient=training_costs)
        self.warmup_time = theano.shared(0,"n ticks without loss")        
        batch_sharpe = sharpe_ratio_symb(self.returns[:,self.warmup_time:])

        if self.name != "RRL_model":
            loss = -ewma(batch_sharpe).mean()
        else:
            loss = -batch_sharpe.mean()
            
        if l2_penalty is not None:
            loss += l2_penalty

        #optimizing:
        updates = optimizer(loss,self.params)
        updates += updates_callback()
        self.train_fun = theano.function(X_inputs+list(map(instrument_returns.get,asset_classes)),
                                         -loss, updates=updates)
        #predictions
        portfolio_weights_det = lasagne.layers.get_output(portfolio_layers,deterministic=True)
        portfolio_weights_det = dict(zip(asset_classes, portfolio_weights_det))
        self._inputs = X_inputs
        #compile predict function
        self._predict = theano.function(X_inputs,portfolio_weights_det,updates= updates_callback())

    def fit(self, X, y, nb_epochs=100, batch_size=10,seq_len=100, warmup_time=0, verbose=False):
        old_warmup_time = self.warmup_time.get_value()
        self.warmup_time.set_value(warmup_time)
        gen = generate_minibatches(X, y, batch_size, seq_len)
        for Xbatch, ybatch in iterate_over(gen, nb_epochs):
            X_per_class = [Xbatch]
            y_per_class = [ybatch]
            self.metrics["loss"][self.epoch] = self.train_fun(*X_per_class + y_per_class)
            if verbose:
                print("iteration %i: loss = %.5f" % (self.epoch, self.metrics["loss"][self.epoch]))
            self.epoch += 1
        self.warmup_time.set_value(old_warmup_time)
        return self
    
    def predict(self, X,**kwargs):
        tensor_ndims = [np.ndim(X)]
        assert np.ndim(X) in (3, 4)
        batch_flag = tensor_ndims[0] == 4
        if not batch_flag:
            # prepend batch to all tensors
            X = X[None, ...]

        predictions = self._predict(X)
        if not batch_flag:
            for asset_class in predictions:
                predictions[asset_class] = predictions[asset_class][0]
        return predictions
    
    def update_metrics(self, X, y, prefix="", verbose=True,
                       cost_coefficient=COST_COEFFICIENT):
        portfolio_weights = self.predict(X,y=y)
        model_returns = get_portfolio_returns_fin(y, portfolio_weights,
                                                        cost_coefficient=cost_coefficient)
        model_sharpe = sharpe_ratio_fin(model_returns)
        base_returns = get_base_returns_fin(y,)
        base_sharpe = sharpe_ratio_fin(base_returns)
        self.metrics[prefix + "base_sharpe"][self.epoch] = base_sharpe
        self.metrics[prefix + "model_sharpe"][self.epoch] = model_sharpe
        if verbose:
            print(prefix + 'sharpes = ', base_sharpe, model_sharpe)
        return model_returns

    def save(self,fname=None,allow_overwrite=False):
        param_values = [param.get_value() for param in self.params]
        with open(fname,'wb') as fdump:
            pickle.dump(param_values,fdump)

    def load(self,fname):
        with open(fname, 'rb') as fdump:
            param_values = pickle.load(fdump)

        for param,value in zip(self.params,param_values):
            param.set_value(value)


def ewma(series, axis=None, span=VOL_WINDOW_SPAN, adjust=True, initial=None):
    """
    Exponentially-weighted moving average
    """
    if axis is None:
        if series.ndim == 1:
            axis=0
        else:
            raise ValueError("Please specify which axis to compute ewma over (usually time axis)") 
    assert span >= 1
    alpha = 2. / (span + 1) 
    series = T.swapaxes(series, axis, 0)
    if adjust:
        assert initial is None
        initial = T.zeros_like(series[0])
    else:
        if initial is None:
            initial = series[0]
        initial /= alpha
 
    def ewma_numerator_step(a_i, prev_ewma):
        return a_i + (1. - alpha) * prev_ewma
 
    ewma_numerators, _ = theano.scan(ewma_numerator_step, series,
                                     outputs_info=initial, strict=True)
 
    if adjust:
        ewma_denominators = T.cumsum((1 - alpha) ** T.arange(ewma_numerators.shape[0]))
        series_ewma = ewma_numerators / ewma_denominators.reshape((-1,)+(1,)*(ewma_numerators.ndim-1))
    else:
        series_ewma = ewma_numerators * alpha 
    series_ewma = T.swapaxes(series_ewma, 0, axis)
    return series_ewma
   

def generate_minibatches(X ,y, batch_size, seq_length,subseq_borders = None):
    n_ticks = X.shape[0]
    if subseq_borders is None:
        subseq_borders = [(0,n_ticks)]
    valid_starts = []
    for seq_start,seq_end in subseq_borders:
        valid_starts.append( np.arange(seq_start, seq_end - seq_length))
    valid_starts = np.concatenate(valid_starts)
    while True:
        batch_starts = np.random.choice(valid_starts ,batch_size)
        batch_ix = batch_starts[: ,None] + np.arange(seq_length)[None ,:]
        X_batch = X[batch_ix]
        y_batch = y[batch_ix]
        yield X_batch,y_batch

def iterate_over(generator,n_iters):
    """
    Takes first n_iters minibatches from the generator
    """
    for i,batch in enumerate(generator):
        if i>n_iters:break
        yield batch

def get_norm_fin(portfolio_weights):
    """
    computes weights norm
    :param portfolio_weights: weights tensor [tick,instr] or [batch,tick,instr] or a dict of such
    :return: a vector of weight norms
    """
    if type(portfolio_weights) is dict:
        return reduce(add, list(map(get_norm_fin, portfolio_weights.values())))
    return np.abs(portfolio_weights).sum(axis=-1,keepdims=True)

def normalize_weights_fin(portfolio_weights):
    """
    Normalizes weights over instruments, making sure all abs(weights) sum to 1

    :param portfolio_weights: weights tensor [tick,instr] or [batch,tick,instr] or a dict of such
    :returns: normalized weights of the same shape as portfolio_weights
    """

    norm = get_norm_fin(portfolio_weights)
    norm[norm==0]=1 # fix nan

    if type(portfolio_weights) is dict:
        portfolio_weights = {asset:weights/norm for asset,weights in portfolio_weights.items()}
    else:
        portfolio_weights /= norm

    return portfolio_weights

from agentnet.utils.format import is_numpy_object
def get_reallocation_costs_fin(portfolio_weights,
                           initial_weights='same',
                           cost_coefficient=COST_COEFFICIENT):
    assert is_numpy_object(portfolio_weights)
    if initial_weights == 'same':
        initial_weights = portfolio_weights[0, :]
    elif initial_weights == 'zeros':
        initial_weights = np.zeros_like(portfolio_weights[0, :])
    assert is_numpy_object(initial_weights)

    #prepend initial_W to weights
    W_padded = np.concatenate([initial_weights[None, :], portfolio_weights], axis=0)
    W_diff = np.diff(W_padded,axis=0)

    #vector of costs charged at each time tick
    return np.abs(W_diff).sum(axis=1) * cost_coefficient



def get_portfolio_returns_fin(instrument_returns, portfolio_weights,
                          cost_coefficient=COST_COEFFICIENT,
                          initial_weights='same'):
    if isinstance(portfolio_weights, dict):
        assert isinstance(instrument_returns, dict)
        asset_classes = portfolio_weights.keys()
        instrument_returns = np.concatenate([instrument_returns[asset] for asset in asset_classes], axis=1)
        portfolio_weights = np.concatenate([portfolio_weights[asset] for asset in asset_classes], axis=1)
        if isinstance(initial_weights, dict):
            initial_weights = np.concatenate([initial_weights[asset] for asset in asset_classes], axis=0)

    raw_returns = (instrument_returns * portfolio_weights).sum(axis=-1)


    costs = get_reallocation_costs_fin(portfolio_weights,
                                   initial_weights=initial_weights,
                                   cost_coefficient=cost_coefficient)

    actual_returns = raw_returns - costs

    return actual_returns

def get_base_returns_fin(instrument_returns):
    if isinstance(instrument_returns, dict):
        instrument_returns = np.concatenate([instrument_returns[asset] for asset in instrument_returns.keys()], axis=1)

    return instrument_returns.mean(axis=1)


def sharpe_ratio_fin(R, na_rm = True, period=1. / NB_DAYS_IN_YEAR):
    if na_rm: R = R[np.isfinite(R)]
    return R.mean() / (R.std()*np.sqrt(period))



def normalize_weights_symb(portfolio_weights,relative=False):
    if relative:
        portfolio_weights -= portfolio_weights.mean(axis=-1, keepdims=True)
    norm = T.abs_(portfolio_weights).sum(axis=-1, keepdims=True)
    norm = T.switch(T.neq(norm,0), norm, 1)
    portfolio_weights /= norm

    return portfolio_weights

from agentnet.utils.format import is_theano_object,is_numpy_object
def get_reallocation_costs_symb(portfolio_weights,
                           initial_W='same',
                           cost_coefficient=COST_COEFFICIENT):
    if initial_W == 'same':
        initial_W = portfolio_weights[:, 0, :]
    elif initial_W == 'zeros':
        initial_W = T.zeros_like(portfolio_weights[:, 0, :])

    assert is_theano_object(portfolio_weights) or is_numpy_object(portfolio_weights)
    assert is_theano_object(initial_W) or is_numpy_object(initial_W)

    W_padded = T.concatenate([initial_W[:, None, :], portfolio_weights], axis=1)
    W_diff = W_padded[:, 1:, :] - W_padded[:, :-1, :]

    costs = T.abs_(W_diff).sum(axis=-1) * cost_coefficient
    return costs


def get_portfolio_returns_symb(instrument_returns, portfolio_weights,
                          cost_coefficient=COST_COEFFICIENT,
                          initial_W='same'):
    if isinstance(portfolio_weights, dict):
        assert isinstance(instrument_returns, dict)
        asset_classes = portfolio_weights.keys()
        instrument_returns = T.concatenate([instrument_returns[asset] for asset in asset_classes], axis=-1)
        portfolio_weights = T.concatenate([portfolio_weights[asset] for asset in asset_classes], axis=-1)
        if isinstance(initial_W, dict):
            initial_W = T.concatenate([initial_W[asset] for asset in asset_classes], axis=-1)

    raw_returns = (instrument_returns * portfolio_weights).sum(axis=-1)
    if cost_coefficient > 0:
        costs = get_reallocation_costs_symb(portfolio_weights,initial_W,cost_coefficient)
        actual_returns = raw_returns - costs
    else:
        actual_returns = raw_returns

    return actual_returns



def sharpe_ratio_symb(returns, period=1./NB_DAYS_IN_YEAR,axis=-1):
    return returns.mean(axis=axis) / (returns.std(axis=axis) * (period ** .5))