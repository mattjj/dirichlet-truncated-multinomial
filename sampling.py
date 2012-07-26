from __future__ import division
import numpy as np
na = np.newaxis
import time
from warnings import warn
import cPickle

from dirichlet import log_dirichlet_density, log_censored_dirichlet_density
from simplex import mesh, proj_to_2D
from parallel import dv

### SAMPLING

def generate_pi_samples_withauxvars(alpha,n_samples,data):
    K = data.shape[0]

    starttime = time.time()
    sample_counts = data.sum(1)

    # randomly initialize pi
    pi = np.random.dirichlet(alpha * np.ones(K))
    # set up aux count array
    counts = data.copy()

    samples = np.zeros((n_samples,K))
    for ii in range(n_samples):
        ### sample aux vars given pi
        for idx, sample_count in enumerate(sample_counts):
            counts[idx,idx] = np.random.geometric(1.-pi[idx],size=sample_count).sum() - sample_count # support is {1,2,...}, but we want {0,1,...}
        ### sample pi given aux vars
        pi = np.random.dirichlet(alpha * np.ones(K) + counts.sum(0))
        samples[ii] = pi

    print 'done drawing samples in %0.2f seconds' % (time.time() - starttime)

    return samples

def generate_pi_samples_mh(alpha,n_samples,data,beta):
    starttime = time.time()
    K = data.shape[0]

    # randomly initialize pi
    pi = np.random.dirichlet(alpha * np.ones(K))
    current_val = log_censored_dirichlet_density(pi,alpha=alpha,data=data)

    samples = []

    n_accepts = 0

    # for efficiency, gaussian proposals outside of loop
    # proposals = sigma * np.random.normal(size=(n_samples,K))
    # proposals -= np.dot(proposals,np.ones(K)/K)[:,na]

    # loop mh proposals
    for ii in range(n_samples):
        ### make a proposal
        pi_prime = np.random.dirichlet(beta * pi)
        ### get proposal probability and sample it
        new_val = log_censored_dirichlet_density(pi_prime,alpha=alpha,data=data)
        if new_val > -np.inf:
            a = min(1.,np.exp(new_val - current_val
                + log_dirichlet_density(pi,alpha=beta*pi_prime)
                  - log_dirichlet_density(pi_prime,alpha=beta*pi)))
            if np.random.rand() < a:
                n_accepts += 1
                pi = pi_prime
                current_val = new_val

            samples.append(pi)

    print 'done drawing samples in %0.2f seconds' % (time.time() - starttime)
    print '%d proposals, %d accepted, acceptance ratio %0.4f' % (n_samples , n_accepts, n_accepts / n_samples)

    return samples

def get_samples_parallel(nruns,nrawsamples,params={'alpha':2.,'beta':30.,'data':np.array([[0,2,0],[0,0,0],[0,0,0]])}):
    # data shape sets dimensionality
    alpha, beta, data = params['alpha'], params['beta'], params['data']

    mhsamples_list  = dv.map_sync(lambda tup: generate_pi_samples_mh(tup[0],tup[1],tup[2],tup[3]), [(alpha,nrawsamples,data,beta)]*nruns)
    auxsamples_list = dv.map_sync(lambda tup: generate_pi_samples_withauxvars(tup[0],tup[1],tup[2]), [(alpha,nrawsamples,data)]*nruns)

    dv.purge_results('all')

    return mhsamples_list, auxsamples_list

def run_and_save_samples(*args,**kwargs):
    filename = kwargs['filename'] if 'filename' in kwargs else 'samples'
    mhsamples_list, auxsamples_list = get_samples_parallel(*args,**kwargs)
    with open(filename,'w') as outfile:
        cPickle.dump((mhsamples_list, auxsamples_list), outfile, protocol=2)
    return mhsamples_list, auxsamples_list

def load_samples(filename='samples'):
    with open(filename,'r') as infile:
        samples = cPickle.load(infile)
    return samples

### TESTS

