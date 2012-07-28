from __future__ import division
import numpy as np
na = np.newaxis
import time
from warnings import warn
import cPickle, os

from dirichlet import log_dirichlet_density, log_censored_dirichlet_density
import parallel

### SAMPLING

def generate_pi_samples_withauxvars(alpha,n_samples,data):
    K = data.shape[0]

    starttime = time.time()
    sample_counts = data.sum(1)

    # only generate aux vars for the ones we need
    auxvar_indices = np.arange(len(sample_counts))[sample_counts > 0]
    sample_counts = sample_counts[sample_counts > 0]

    # randomly initialize pi
    pi = np.random.dirichlet(alpha * np.ones(K))
    # set up aux count array
    counts = data.copy()

    samples = np.zeros((n_samples,K))
    for ii in range(n_samples):
        ### sample aux vars given pi
        for idx, sample_count in zip(auxvar_indices, sample_counts):
            counts[idx,idx] = np.random.geometric(1.-pi[idx],size=sample_count).sum() - sample_count # support is {1,2,...}, but we want {0,1,...}
        ### sample pi given aux vars
        pi = np.random.dirichlet(alpha * np.ones(K) + counts.sum(0))
        samples[ii] = pi

    # print 'done drawing samples in %0.2f seconds' % (time.time() - starttime)

    return samples

def generate_pi_samples_mh(alpha,n_samples,data,beta):
    # starttime = time.time()
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
    n_total = 0
    while len(samples) < n_samples:
        ### make a proposal
        pi_prime = np.random.dirichlet(beta * pi)
        n_total += 1
        ### get proposal probability and sample it
        new_val = log_censored_dirichlet_density(pi_prime,alpha=alpha,data=data)
        if new_val > -np.inf: # in our tests, this is always true
            a = min(1.,np.exp(new_val - current_val
                + log_dirichlet_density(pi,alpha=beta*pi_prime)
                  - log_dirichlet_density(pi_prime,alpha=beta*pi)))
            if np.random.rand() < a:
                n_accepts += 1
                pi = pi_prime
                current_val = new_val

            samples.append(pi)

    # print 'done drawing samples in %0.2f seconds' % (time.time() - starttime)
    # print '%d total proposals, %d valid proposals, %d accepted, valid acceptance ratio %0.4f' % (n_total,n_samples, n_accepts, n_accepts / n_samples)

    return samples

def get_samples_parallel(sampler,nruns,*args):
    # the line below doesn't work because ipython parallel can't pickle closures :(
    # samples_list  = parallel.dv.map_sync(lambda tup: sampler(*tup), [args]*nruns)
    # so I wrote this weird thing instead...
    def applier(tup):
        return apply(tup[0],tup[1])
    samples_list = parallel.dv.map_sync(applier, zip([sampler]*nruns,[args]*nruns))

    parallel.dv.purge_results('all')
    return samples_list

def load_or_run_samples(nruns,nsamples,alpha,beta,data):
    filename = './samples/%d.%d.%d.samples' % (nruns,nsamples,data.shape[0])

    if os.path.isfile(filename):
        with open(filename,'r') as infile:
            alphaloaded, betaloaded, dataloaded, mhsamples_list, auxsamples_list = cPickle.load(infile)

        if alphaloaded == alpha and betaloaded == beta and (dataloaded == data).all():
            return mhsamples_list, auxsamples_list
        else:
            print 'warning: existing file %s has different parameters\nresampling and clobbering...' % filename

    mhsamples_list = get_samples_parallel(generate_pi_samples_mh,nruns,alpha,nsamples,data,beta)
    auxsamples_list = get_samples_parallel(generate_pi_samples_withauxvars,nruns,alpha,nsamples,data)

    with open(filename,'w') as outfile:
        cPickle.dump((alpha,beta,data,mhsamples_list,auxsamples_list), outfile, protocol=2)

    return mhsamples_list, auxsamples_list
