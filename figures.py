from __future__ import division
import os
from matplotlib import pyplot as plt
import numpy as np
na = np.newaxis

import simplex, dirichlet

allfigfuncs = []
SAVING = True
plt.interactive(False)

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
