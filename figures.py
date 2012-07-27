from __future__ import division
import os
from matplotlib import pyplot as plt
import numpy as np
na = np.newaxis

import simplex, dirichlet, sampling, tests

allfigfuncs = []
SAVING = True
# plt.interactive(False)

#################################
#  Figure-Generating Functions  #
#################################

def prior_posterior_2D(meshsize=250,alpha=2.,data=np.array([[0,2,0],[0,0,0],[0,0,0]])):
    from scipy.interpolate import griddata
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
    save('./figures/dirichlet_prior_2D.pdf')

    plt.figure(figsize=(8,8))
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),posteriorvals_uncensored,(xi[na,:],yi[:,na]),method='cubic'))
    # plt.tripcolor(mesh2D[:,0],mesh2D[:,1],posteriorvals_uncensored)
    plt.axis('off')
    save('./figures/dirichlet_uncensored_posterior_2D.pdf')

    plt.figure(figsize=(8,8))
    plt.imshow(griddata((mesh2D[:,0],mesh2D[:,1]),posteriorvals_censored,(xi[na,:],yi[:,na]),method='cubic'))
    # plt.tripcolor(mesh2D[:,0],mesh2D[:,1],posteriorvals_censored)
    plt.axis('off')
    save('./figures/dirichlet_censored_posterior_2D.pdf')

allfigfuncs.append(prior_posterior_2D)

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

    save('./figures/MSPRF_sampleindexscaling_%dD.pdf' % ndims)

    ### plot with time scaling
    plt.figure()

    # compute time per sample
    import timing
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

    save('./figures/MSPRF_timescaling_%dD.pdf' % ndims)

allfigfuncs.append(Rhatp)

def autocorrelation():
    raise NotImplementedError

allfigfuncs.append(autocorrelation)

def statistic_convergence():
    raise NotImplementedError

allfigfuncs.append(statistic_convergence)

###############
#  Utilities  #
###############

def save(pathstr):
    filepath = os.path.abspath(pathstr)
    if SAVING:
        if (not os.path.isfile(pathstr)) or raw_input('save over %s? [y/N] ' % filepath).lower() == 'y':
            plt.savefig(filepath)
            print 'saved %s' % filepath
            return
    print 'not saved'

##########################
#  Generate All Figures  #
##########################

def main():
    for f in allfigfuncs:
        f()
    plt.show()

if __name__ == '__main__':
    main()
