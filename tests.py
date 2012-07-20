from __future__ import division
import numpy as np
from warnings import warn

from simplex import proj_to_2D, mesh
from dirichlet import log_censored_dirichlet_density
from sampling import density_from_samples_parallel, dv

def kldist(pvec,qvec):
    '''between two pmfs (or KDEs evaluated over same supports)'''
    return np.sum(np.where(np.logical_or(pvec==0,qvec==0),0.,pvec*(np.log(pvec) - np.log(qvec))))

def kldist_samples(samples,q):
    r'''
    from phat(x) = \frac{1}{|samples|} \sum_{xbar \in samples} \delta(x-xbar)
    to q

    samples is a list of points in X
    q is a function with domain X
    must have \int{x \in X} q(x) \approx 1
    '''



def get_kldivs(mhsamples_list,auxsamples_list,ncomputepoints,params={'alpha':2.,'beta':30.,'data':np.array([[0,2,0],[0,0,0],[0,0,0]])},plotting=True):
    alpha, beta, data = params['alpha'], params['beta'], params['data']

    mesh3D = mesh(100)
    mesh2D = proj_to_2D(mesh3D)

    # TODO not a proper density!! fix to be a density!
    trued = log_censored_dirichlet_density(mesh3D,alpha,data=data)
    trued = np.exp(trued - trued.max())
    trued /= trued.sum()

    mhsamples_list, auxsamples_list = map(proj_to_2D,mhsamples_list), map(proj_to_2D,auxsamples_list)

    outs = []
    for mhsamples, auxsamples in zip(mhsamples_list,auxsamples_list):
        # TODO this inner loop could be factored into a function

        # preallocate stuff for the loop
        thetimes = [timechunk[-1] for timechunk in np.split(np.arange(len(auxsamples)),ncomputepoints)]
        approx_to_true_distances = np.zeros((len(thetimes),2))
        true_to_approx_distances = np.zeros((len(thetimes),2))
        auxd = np.zeros(len(mesh2D))
        mhd = np.zeros(len(mesh2D))

        # for every compute time
        for idx,(auxschunk,mhschunk) in enumerate(zip(
                np.split(auxsamples,ncomputepoints),
                np.split(mhsamples,ncomputepoints))):

            # build a density from each chunk
            # note: both are normed, so as long as sample chunks are all same size
            # (from using np.split) we can just sum and renormalize
            auxd += density_from_samples_parallel(auxschunk,mesh2D,sigma=0.01,normed=True)/max(idx,1) # 1, 1, 2, 3
            auxd /= auxd.sum()
            mhd += density_from_samples_parallel(mhschunk,mesh2D,sigma=0.01,normed=True)/max(idx,1)
            mhd /= mhd.sum()

            # save distance measurements
            approx_to_true_distances[idx] = (kldist(auxd,trued),kldist(mhd,trued))
            true_to_approx_distances[idx] = (kldist(trued,auxd),kldist(trued,mhd))

            dv.purge_results('all')

        outs.append((approx_to_true_distances, true_to_approx_distances))

    if plotting:
        from matlotlib import pyplot as plt
        import operator
        plt.figure()

        approx_to_true, true_to_approx = zip(*outs)
        for idx,(x,name) in enumerate(zip(zip(*outs),['KL(approx,true)','KL(true,approx)'])):
            plt.subplot(2,1,idx+1)
            plt.title(name)
            means = reduce(operator.add,x)/len(x)
            # TODO make 10th/90th instead of std
            stds = np.sqrt(reduce(operator.add,[(out-means)**2 for out in x])/len(x))
            plt.errorbar(np.arange(len(means[:,0])),means[:,0],yerr=stds[:,0],fmt='b-',label='aux')
            plt.errorbar(np.arange(len(means[:,1])),means[:,1],yerr=stds[:,1],fmt='r-',label='mh')
            plt.legend()

    return outs

def get_autocorr(mhsamples_list,auxsamples_list,plotting=True):
    # what is autocorr for a vector sequence? rho(corr matrix at lag i) for all
    # i>0?
    # maybe i'll just do it component-by-component...
    results = []
    for mhsamples, auxsamples in zip(mhsamples_list,auxsamples_list):
        # TODO this inner loop could be factored into a function
        # TODO just does zeroth component
        mhresult = np.correlate(mhsamples[:,0],mhsamples[:,0],'full')
        mhresult = mhresult[len(mhresult)//2:]

        auxresult = np.correlate(auxsamples[:,0],auxsamples[:,0],'full')
        auxresult = auxresult[len(auxresult)//2:]

        results.append((mhresult,auxresult))

    if plotting:
        raise NotImplementedError

    return results

def get_statistic_convergence(mhsamples_list,auxsamples_list,ncomputepoints,params={'alpha':2.,'beta':30.,'data':np.array([[0,2,0],[0,0,0],[0,0,0]])},plotting=True):
    # mean, var of components

    alpha, beta, data = params['alpha'], params['beta'], params['data']
    K = data.shape[0]

    # estimate true parameters using all samples from aux
    temp = np.concatenate(auxsamples_list,axis=0)
    truemean = temp.mean(0) # vector of length K
    truevar = temp.var(0) # vector of length K

    outs = []
    # loop over chains, loop over chunks
    for mhsamples, auxsamples in zip(mhsamples_list,auxsamples_list):
        # TODO this could be factored into a single-chain function

        means = np.zeros((ncomputepoints,2,K))
        varis = np.zeros((ncomputepoints,2,K))

        chunksize = len(auxsamples)//ncomputepoints
        for idx in range(ncomputepoints):
            t = chunksize * (1+idx)

            means[idx,0] = auxsamples[:t].mean(0)
            means[idx,1] = mhsamples[:t].mean(0)

            varis[idx,0] = auxsamples[:t].var(0)
            varis[idx,1] = mhsamples[:t].var(0)

        outs.append((means,varis))

    if plotting:
        from matplotlib import pyplot as plt
        raise NotImplementedError

    return outs, (truemean, truevar)

# TODO higher dim experiment

# TODO try higher dimensions (cant use kl measure, maybe how fast it finds a
# really strong mode? or... )
# two experiments: 2D and 30D
# in 2D case, do kl
# in 30D case, try:
#  - estimatino of expectation and cov matrix: get truth by running lots, then
#  tack convergence to that correct bidness!! ***
#  - time to hit neighborhood of correct thing, avg across many trials
#  - autocorrelation, avg across many trials

# based on preliminary timeit tests, same number of samples is 10x faster with
# aux vars, and the pic might look better
# kldist is slightly better for same number of samples but not way. i guess
# improvement is cpu time? that means Python hurts
# also *easier to implement and *no tuning



### OLD JUNKYARD BELOW HERE!! JUST SCRAP! USE FOR PARTS!

def test_speed(nsamples,burnin,useevery,numcomputepoints,plotting=True):
    warn('havent tried this in a while!!!')
    # TODO downweight old samples?? mh needs some burnin
    import sys
    alpha = 2.
    beta = 30.
    data = np.zeros((3,3))
    data[0,1] += 2

    mesh3D = mesh(100)
    mesh2D = proj_to_2D(mesh3D)

    # use direct riemann integration to get true density
    trued = log_censored_dirichlet_density(mesh3D,alpha,data=data)
    trued = np.exp(trued - trued.max())
    trued /= trued.sum()

    # get all the samples
    auxsamples = proj_to_2D(generate_pi_samples_withauxvars(alpha,nsamples,data)[burnin::useevery])
    mhsamples = proj_to_2D(generate_pi_samples_mh(alpha,nsamples,data,beta)[burnin::useevery])

    # preallocate stuff for the loop
    thetimes = [timechunk[-1]*useevery for timechunk in np.split(np.arange(len(auxsamples)),numcomputepoints)]
    approx_to_true_distances = np.zeros((len(thetimes),2))
    true_to_approx_distances = np.zeros((len(thetimes),2))
    auxd = np.zeros(len(mesh2D))
    mhd = np.zeros(len(mesh2D))

    # for every compute time
    print len(thetimes)
    for idx,(auxschunk,mhschunk) in enumerate(zip(
            np.split(auxsamples,numcomputepoints),
            np.split(mhsamples,numcomputepoints))):

        # build a density from each chunk
        # note: both are normed, so as long as sample chunks are all same size
        # (from using np.split) we can just sum and renormalize
        auxd += density_from_samples_parallel(auxschunk,mesh2D,sigma=0.1,normed=True)
        auxd /= auxd.sum()
        mhd += density_from_samples_parallel(mhschunk,mesh2D,sigma=0.1,normed=True)
        mhd /= mhd.sum()

        # save distance measurements
        approx_to_true_distances[idx] = (kldist(auxd,trued),kldist(mhd,trued))
        true_to_approx_distances[idx] = (kldist(trued,auxd),kldist(trued,mhd))

        sys.stdout.write('.'); sys.stdout.flush()

    if plotting:
        from matplotlib import pyplot as plt
        plt.figure()

        plt.subplot(2,1,1)
        plt.title('KL(approx,true)')
        plt.plot(thetimes,approx_to_true_distances[:,0],'b-',label='aux')
        plt.plot(thetimes,approx_to_true_distances[:,1],'r-',label='mh')
        plt.legend()

        plt.subplot(2,1,2)
        plt.title('KL(true,approx)')
        plt.plot(thetimes,true_to_approx_distances[:,0],'b-',label='aux')
        plt.plot(thetimes,true_to_approx_distances[:,1],'r-',label='mh')
        plt.legend()

    return approx_to_true_distances, true_to_approx_distances


