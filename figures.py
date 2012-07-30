from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
na = np.newaxis
from scipy.interpolate import griddata

import simplex, dirichlet, sampling, tests, density, timing

allfigfuncs = []
SAVING = True
# plt.interactive(False)

#################################
#  Figure-Generating Functions  #
#################################

def prior_posterior_2D(meshsize=250,alpha=2.,data=np.array([[0,2,0],[0,0,0],[0,0,0]])):
    assert data.shape == (3,3)

    mesh3D = simplex.mesh(meshsize)
    mesh2D = simplex.proj_to_2D(mesh3D) # use specialized b/c it plays nicer with triangulation algorithm

    priorvals = np.exp(dirichlet.log_dirichlet_density(mesh3D,alpha))

    posteriorvals_uncensored = np.exp(dirichlet.log_dirichlet_density(mesh3D,alpha,data=data.sum(0)))

    temp = dirichlet.log_censored_dirichlet_density(mesh3D,alpha,data=data)
    temp = np.exp(temp - temp.max())
    posteriorvals_censored = temp/temp.sum() # direct discretized integration!

    # used for grid interpolation
    xi = np.linspace(mesh2D[:,0].min(), mesh2D[:,0].max(), 2000, endpoint=True)
    yi = np.linspace(mesh2D[:,1].min(), mesh2D[:,1].max(), 2000, endpoint=True)

    plt.figure(figsize=(8,8))
    # use exactly one of the next two code lines!
    # this one performs interpolation to get a rectangular-pixel grid, but
    # produces a blurred image
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),priorvals,(xi[na,:],yi[:,na]),method='cubic'))
    # this one exactly represents the data by performing a DeLaunay
    # triangulation, but it must draw each triangular pixel individually,
    # resulting in large files and slow draw times
    # plt.tripcolor(mesh2D[:,0],mesh2D[:,1],priorvals) # exact triangles, no blurring
    plt.axis('off')
    save('../writeup/figures/dirichlet_prior_2D.pdf')

    plt.figure(figsize=(8,8))
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),posteriorvals_uncensored,(xi[na,:],yi[:,na]),method='cubic'))
    # plt.tripcolor(mesh2D[:,0],mesh2D[:,1],posteriorvals_uncensored)
    plt.axis('off')
    save('../writeup/figures/dirichlet_uncensored_posterior_2D.pdf')

    plt.figure(figsize=(8,8))
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),posteriorvals_censored,(xi[na,:],yi[:,na]),method='cubic'))
    # plt.tripcolor(mesh2D[:,0],mesh2D[:,1],posteriorvals_censored)
    plt.axis('off')
    save('../writeup/figures/dirichlet_censored_posterior_2D.pdf')

allfigfuncs.append(prior_posterior_2D)

def aux_posterior_2D(meshsize=250,alpha=2.,data=np.array([[0,2,0],[0,0,0],[0,0,0]])):
    assert data.shape == (3,3)

    mesh3D = simplex.mesh(meshsize)
    mesh2D = simplex.proj_to_2D(mesh3D) # use specialized b/c it plays nicer with triangulation algorithm

    # get samples
    auxsamples = sampling.generate_pi_samples_withauxvars(alpha,10000,data)

    # evaluate a kde based on the samples
    aux_kde = density.kde(0.005,auxsamples[len(auxsamples)//20:])
    aux_kde_vals = aux_kde(mesh3D)

    ### plot

    # used for grid interpolation
    xi = np.linspace(mesh2D[:,0].min(), mesh2D[:,0].max(), 2000, endpoint=True)
    yi = np.linspace(mesh2D[:,1].min(), mesh2D[:,1].max(), 2000, endpoint=True)

    plt.figure(figsize=(8,8))
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),aux_kde_vals,(xi[na,:],yi[:,na]),method='cubic'))
    plt.axis('off')

    save('../writeup/figures/dirichlet_censored_auxvar_posterior_2D.pdf')

allfigfuncs.append(aux_posterior_2D)

def Rhatp(nsamples=1000,ncomputepoints=25,nruns=50,ndims=10):
    # get samples
    data = np.zeros((ndims,ndims))
    data[np.roll(np.arange(ndims//2),1),np.arange(ndims//2)] = 10 # fill half the dims with data
    alpha = 2. # Dirichlet prior hyperparameter
    beta = 160. # MH proposal distribution parameter, set so acceptance rate is about 0.24 with ndims=10
    mhsamples, auxsamples = map(np.array,
            sampling.load_or_run_samples(nruns,nsamples,alpha,beta,data))

    # get Rhatps
    aux_R = tests.get_Rhat(auxsamples,ncomputepoints=ncomputepoints)
    mh_R = tests.get_Rhat(mhsamples,ncomputepoints=ncomputepoints)

    ### plot without time scaling
    plt.figure()

    # plt.subplot(2,1,1)
    plt.plot(tests.chunk_indices(nsamples,ncomputepoints),aux_R,'bx-',label='Aux. Var. Sampler')
    plt.plot(tests.chunk_indices(nsamples,ncomputepoints),mh_R,'gx-',label='MH Sampler')
    plt.ylim(0,1.1*mh_R.max())
    plt.xlim(0,1000)
    plt.xlabel('sample index')
    plt.legend()
    plt.title('MH and Aux. Var. Samplers MSPRF vs Sample Indices')

    # plt.subplot(2,1,2)
    # plt.plot(tests.chunk_indices(nsamples,ncomputepoints),aux_R,'bx-')
    # plt.ylim(0,1.1*aux_R.max())
    # plt.xlim(0,closeindex)
    # plt.xlabel('sample index')
    # plt.title('Aux. Var. Sampler MSPRF vs Sample Indices')

    save('../writeup/figures/MSPRF_sampleindexscaling_%dD.pdf' % ndims)

    ### plot with time scaling
    plt.figure()

    # compute time per sample
    aux_timing = timing.get_auxvar_timing(data=data,alpha=alpha)
    mh_timing = timing.get_mh_timing(data=data,beta=beta,alpha=alpha)

    plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints))*aux_timing,
            aux_R,'bx-',label='Aux. Var. Sampler')
    plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints))*mh_timing,
            mh_R,'gx-',label='MH Sampler')
    plt.ylim(0,1.1*mh_R.max())
    plt.xlim(0,mh_timing*nsamples)
    plt.xlabel('seconds')
    plt.legend()
    plt.title('MH and Aux. Var. Sampler MSPRF vs Computation Time')

    save('../writeup/figures/MSPRF_timescaling_%dD.pdf' % ndims)

allfigfuncs.append(Rhatp)

def autocorrelation(nsamples=1000,nruns=50,ndims=10):
    # get samples
    data = np.zeros((ndims,ndims))
    data[np.roll(np.arange(ndims//2),1),np.arange(ndims//2)] = 10 # fill half the dims with data
    alpha = 2. # Dirichlet prior hyperparameter
    beta = 160. # MH proposal distribution parameter, set so acceptance rate is about 0.24 with ndims=10
    mhsamples, auxsamples = map(np.array,
            sampling.load_or_run_samples(nruns,nsamples,alpha,beta,data))

    # compute autocorrelations
    aux_corrs = tests.get_autocorr(auxsamples)
    mh_corrs = tests.get_autocorr(mhsamples)

    # plot
    for component, ordinalname in zip([0,1],['first','second']):
        plt.figure()

        for corrs, samplername, color in zip([aux_corrs, mh_corrs],['Aux. Var.','MH'],['b','g']):
            plt.plot(corrs.mean(0)[:,component],color+'-',label='%s Sampler' % samplername)
            plt.plot(scoreatpercentile(corrs[...,component],per=10,axis=0),color+'--')
            plt.plot(scoreatpercentile(corrs[...,component],per=90,axis=0),color+'--')

        plt.legend()
        plt.xlabel('lag')
        plt.xlim(0,np.where(mh_corrs.mean(0)[:,component] < 0.01)[0][0])
        plt.title('%s Component Autocorrelations' % ordinalname.capitalize())

        save('../writeup/figures/autocorrelations_%dD_%s.pdf' % (ndims,ordinalname))

allfigfuncs.append(autocorrelation)

def statistic_convergence(nsamples=5000,ncomputepoints=50,nruns=50,ndims=10):
    # get samples
    data = np.zeros((ndims,ndims))
    data[np.roll(np.arange(ndims//2),1),np.arange(ndims//2)] = 10 # fill half the dims with data
    alpha = 2. # Dirichlet prior hyperparameter
    beta = 160. # MH proposal distribution parameter, set so acceptance rate is about 0.24 with ndims=10
    mhsamples, auxsamples = map(np.array,
            sampling.load_or_run_samples(nruns,nsamples,alpha,beta,data))

    # compute statistics
    (mhmeans, mhvars), (mh_truemean, mh_truevar), (mh_mean_ds, mh_var_ds) = \
            tests.get_statistic_convergence(mhsamples,ncomputepoints)
    (auxmeans, auxvars), (aux_truemean, aux_truevar), (aux_mean_ds, aux_var_ds) = \
            tests.get_statistic_convergence(auxsamples,ncomputepoints)

    # check that the estimated "true" statistics agree
    assert ((mh_truemean - aux_truemean)**2).sum() < 1e-5 \
            and ((mh_truevar - aux_truevar)**2).sum() < 1e-5

    # get time scaling
    aux_timing = timing.get_auxvar_timing(data=data,alpha=alpha)
    mh_timing = timing.get_mh_timing(data=data,beta=beta,alpha=alpha)

    # plot
    for samplerds, statisticname in zip(((aux_mean_ds,mh_mean_ds),(aux_var_ds,mh_var_ds)),('mean','variance')):
        # sample index scaling
        plt.figure()

        for ds, samplername, color in zip(samplerds, ['Aux. Var.','MH'],['b','g']):
            plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints)),
                    ds.mean(0),color+'-',label='%s Sampler' % samplername)
            plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints)),
                    scoreatpercentile(ds,per=10,axis=0),color+'--')
            plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints)),
                    scoreatpercentile(ds,per=90,axis=0),color+'--')

        plt.legend()
        plt.xlabel('sample index')
        plt.title('%s Convergence' % statisticname.capitalize())

        save('../writeup/figures/statisticconvergence_%dD_%s.pdf' % (ndims,statisticname))


        # time scaling
        plt.figure()

        for ds, samplername, color, timescaling in zip(samplerds, ['Aux. Var.','MH'],['b','g'],
                (aux_timing,mh_timing)):
            plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints))*timescaling,
                    ds.mean(0),color+'-',label='%s Sampler' % samplername)
            plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints))*timescaling,
                    scoreatpercentile(ds,per=10,axis=0),color+'--')
            plt.plot(np.array(tests.chunk_indices(nsamples,ncomputepoints))*timescaling,
                    scoreatpercentile(ds,per=90,axis=0),color+'--')

        plt.legend()
        plt.xlabel('seconds')
        plt.title('%s Convergence' % statisticname.capitalize())

        save('../writeup/figures/statisticconvergence_timescaling_%dD_%s.pdf' % (ndims,statisticname))

allfigfuncs.append(statistic_convergence)

###############
#  Utilities  #
###############

import os
def save(pathstr):
    filepath = os.path.abspath(pathstr)
    if SAVING:
        if (not os.path.isfile(pathstr)) or raw_input('save over %s? [y/N] ' % filepath).lower() == 'y':
            plt.savefig(filepath)
            print 'saved %s' % filepath
            return
    print 'not saved'

def scoreatpercentile(data,per,axis):
    '''
    like the function in scipy.stats but with an axis argument, and works on
    arrays.
    '''
    a = np.sort(data,axis=axis)
    idx = per/100. * (data.shape[axis]-1)

    if (idx % 1 == 0):
        return a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]]
    else:
        lowerweight = 1-(idx % 1)
        upperweight = (idx % 1)
        idx = int(np.floor(idx))
        return lowerweight * a[[slice(None) if ii != axis else idx for ii in range(a.ndim)]] \
                + upperweight * a[[slice(None) if ii != axis else idx+1 for ii in range(a.ndim)]]

##########################
#  Generate All Figures  #
##########################

def main():
    for f in allfigfuncs:
        f()
    plt.show()

if __name__ == '__main__':
    main()
