from __future__ import division
import numpy as np
na = np.newaxis
from warnings import warn

from simplex import proj_to_2D, mesh
from dirichlet import log_censored_dirichlet_density
from sampling import density_from_samples_parallel
from parallel import dv
from density import kde

def chunk_indices(T,npoints):
    # TODO this is a pretty dumb implementation :D
    return map(lambda x: x[-1], np.array_split(np.arange(T),npoints))

def kldist(pvec,qvec):
    '''
    between two pmfs (or KDEs evaluated over same supports and normalized)
    sorta deprecated...
    '''
    return np.sum(np.where(np.logical_or(pvec==0,qvec==0),0.,pvec*(np.log(pvec) - np.log(qvec))))

def kldist_samples(samples,q):
    r'''
    from phat(x) = \frac{1}{|samples|} \sum_{xbar \in samples} \delta(x-xbar)
    to q

    samples is a list of points in X
    q is a function with domain X, like from a call to density.kde
    must have \int{x \in X} q(x) \approx 1
    '''
    warn('untested')
    N = samples.shape[0]
    return -1./N * np.log(N * q(samples)).sum()

def get_autocorr(chains,plotting=True):
    '''
    component-by-component
    '''
    warn('untested')
    chains = np.array(chains)
    results = np.zeros(chains.shape)
    for chainidx, chain in enumerate(chains):
        for idx in chain.shape[1]:
            temp = np.correlate(chain[:,idx],chain[:,idx],'full')
            results[chainidx,:,idx] = temp[temp.shape[0]//2:]
    if plotting:
        raise NotImplementedError
    return results

def get_statistic_convergence(chains,ncomputepoints,plotting=True):
    '''
    mean, var of components, and l2 distances to the truth
    '''
    warn('untested')
    chains = np.array(chains,ndmin=3)

    p = chains.shape[2]

    ### estimate true parameters using all samples
    truemean = chains.mean(0).mean(0) # vector of length p
    truevar = chains.reshape((-1,p)).var(0) # vector of length p

    ### compute statistics at the compute points
    # preallocate outputs
    means = np.empty((chains.shape[0],ncomputepoints,p))
    variances = np.empty(means.shape)

    # loop over chains
    for chainidx, chain in enumerate(chains):
        # loop over chunks
        for chunkidx, sampleidx in enumerate(chunk_indices(chain.shape[0], ncomputepoints)):
            # compute statistics up to the current time
            means[chainidx,chunkidx] = chain[sampleidx//2:sampleidx].mean(0)
            variances[chainidx,chunkidx] = chain[sampleidx//2:sampleidx].var(0)

    ### get distances to truth at the compute points
    mean_distances = np.sqrt(((means - truemean[na,na,:])**2).sum(-1))
    var_distances = np.sqrt(((variances - truevar[na,na,:])**2).sum(-1))

    ### plot
    if plotting:
        raise NotImplementedError

    return (means,variances), (truemean,truevar), (mean_distances, var_distances)

def get_Rhat(chains,ncomputepoints,plotting=True):
    '''
    see Monitoring Convergence of Iterative Simulations
    '''

    warn('untested')
    chains_all = np.array(chains)

    outs = np.empty(ncomputepoints)
    for chunkidx, sampleidx in enumerate(chunk_indices(chains.shape[1],ncomputepoints)):
        chains = chains_all[:,sampleidx//2:sampleidx,:]

        m,n,p = chains.shape
        # get means
        mu_all = chains.mean(0).mean(0)
        mu_each = chains.mean(1)
        # B/n is between-chain covariance
        temp = mu_each - mu_all
        B_over_n = 1/(m-1) * temp.T.dot(temp)
        # W is within-chain covaraince
        temp = chains - mu_each[:,na,:]
        W = 1/(m*(n-1)) * np.tensordot(temp,temp,axes=([0,1],[0,1]))

        Vhat = (n-1)/n * W + (1+1/m) * B_over_n
        Rhatp = np.linalg.eigvalsh(np.linalg.solve(W,Vhat)).max()

        outs[chunkidx] = Rhatp

    if plotting:
        raise NotImplementedError

    return outs

def get_kldivs(chains,ncomputepoints,meshsize=100,params={'alpha':2.,'beta':30.,'data':np.array([[0,2,0],[0,0,0],[0,0,0]])},plotting=True):
    alpha, beta, data = params['alpha'], params['beta'], params['data']
    p = chains.shape[2]
    assert p == 3

    ### construct a 'true' density object by discrete approximate integration
    # get density evaluated on a mesh, (mesh3D, dvals)
    mesh3D = mesh(meshsize)
    dvals = log_censored_dirichlet_density(mesh3D, alpha, data=data)
    dvals = np.exp(dvals - dvals.max())
    dvals /= dvals.sum()

    # interpolate into a density function
    true_density = kde(0.05,mesh3D,dvals)

    ### get kl divergence to truth at cmopute points
    # preallocate outputs
    dists = np.zeros((chains.shape[0],ncomputepoints))

    # loop over chains
    for chainidx, chain in enumerate(chains):
        # loop over chunks
        for chunkidx, sampleidx in enumerate(chunk_indices(chain.shape[0], ncomputepoints)):
            # get relevant samples
            samples = chain[sampleidx//2:sampleidx]
            # compute kldiv against true_density
            dists[chainidx,chunkidx] = kldist_samples(samples,true_density)

    ### plot
    if plotting:
        raise NotImplementedError

    return dists

# based on preliminary timeit tests, same number of samples is 10x faster with
# aux vars, and the pic might look better
# kldist is slightly better for same number of samples but not way. i guess
# improvement is cpu time? that means Python hurts
# also *easier to implement and *no tuning

