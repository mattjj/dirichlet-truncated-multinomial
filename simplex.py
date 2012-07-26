from __future__ import division
import numpy as np

def mesh(N,edges=True):
    if not edges:
        N = N-1

    tups = []
    for count, x1 in enumerate(np.linspace(0.,1.,num=N,endpoint=edges)):
        for x2 in np.linspace(0.,1.-x1,num=N-count,endpoint=edges):
            x3 = 1-x1-x2
            tups.append((x1,x2,x3))
    tups = np.array(tups)
    if not edges:
        tups = tups[np.logical_not((tups == 0).any(axis=1))]
    return tups

def proj_to_2D(points):
    # TODO special case of proj_vec, should be removed!
    return np.dot(points,np.array([[0,1],[1,-0.5],[-1,-0.5]]))

def _get_projector(n):
    foo = np.ones((n,n-1))
    foo[np.arange(n-1),np.arange(n-1)] = -(n-1)
    Q,R = np.linalg.qr(foo)
    return Q

def proj_vec(v):
    v = np.array(v,ndmin=2)
    Q = _get_projector(v.shape[1])
    return np.dot(v,Q)

def proj_matrix(mat):
    mat = np.array(mat,ndmin=2)
    Q = _get_projector(mat.shape[1])
    return Q.T.dot(mat).dot(Q)

def test():
    from matplotlib import pyplot as plt

    mesh3D = mesh(25,edges=True)
    mesh2D = proj_to_2D(mesh3D)
    plt.figure(); plt.plot(mesh2D[:,0],mesh2D[:,1],'bo',label='incl. edges')
    mesh3D_noedges = mesh(25,edges=False)
    mesh2D_noedges = proj_to_2D(mesh3D_noedges)
    plt.plot(mesh2D_noedges[:,0],mesh2D_noedges[:,1],'rx',label='excl. edges')
    plt.legend()

    from mpl_toolkits.mplot3d import Axes3D # this import is needed, really!
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(mesh3D[:,0],mesh3D[:,1],mesh3D[:,2],c='b')

    plt.show()

