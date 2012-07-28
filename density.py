from __future__ import division
import numpy as np
na = np.newaxis
import operator

import parallel

def kde(sigma,locs,vals=None):
    '''
    returns a density function p defined by
        p(x) = \sum_{xbar \in locs} vals[xbar] * rbf(xbar,x; sigma)
    rbf(xbar,x; sigma) is the value of a normalized isotropic gaussian density
    with mean xbar and variance sigma evaluated at x
    by default, vals[i] = 1/len(locs), useful for the case when locs are samples
    '''
    # TODO should the computation here be done in log space with logaddexp?
    # yes, esp for higher dimensions...

    if vals is None:
        vals = 1./locs.shape[0] * np.ones(locs.shape[0])
    else:
        assert vals.ndim == 1 and np.allclose(vals.sum(),1)

    locs, vals = np.array(locs), np.array(vals)
    assert locs.ndim == 2 and locs.shape[0] == vals.shape[0]

    K = locs.shape[1] # dimensionality

    def p(x):
        assert x.ndim == 2
        # parallelizes over locs but not over x
        chunksize = 1000000 # max intermediate array size is chunksize doubles
        locchunks = np.array_split(locs,x.shape[0]*locs.shape[0]//chunksize+1)
        valchunks = np.array_split(vals,x.shape[0]*locs.shape[0]//chunksize+1)
        numchunks = len(locchunks)

        def f((ls,vs,x,K,sigma)):
            return np.dot(np.ones(len(ls)),vs[:,na]
                    / (2*np.pi)**(K/2) / np.sqrt(sigma**K)
                    * np.exp(-0.5*((ls[:,na,:] - x[na,:,:])**2).sum(2)/sigma))
        return reduce(operator.add, parallel.dv.map_sync(f, zip(locchunks,valchunks,[x]*numchunks,[K]*numchunks,[sigma]*numchunks)))

    return p

