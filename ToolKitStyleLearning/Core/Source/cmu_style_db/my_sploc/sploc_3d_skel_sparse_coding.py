#My SPLOC implementation for 3D skeleton data
from __future__ import division, print_function, absolute_import
from __future__ import unicode_literals
import argparse
import numpy as np
from scipy.linalg import svd, norm, cho_factor, cho_solve
import h5py
import time

import matlab_wrapper

from geodesic import GeodesicDistanceComputation

def project_weight(x):
    x = np.maximum(0., x)
    max_x = x.max()
    if max_x == 0:
        return x
    else:
        return x / max_x

def prox_l1l2(Lambda, x, beta):
    xlen = np.sqrt((x**2).sum(axis=-1))
    with np.errstate(divide='ignore'):
        shrinkage = np.maximum(0.0, 1 - beta * Lambda / xlen)
    return (x * shrinkage[...,np.newaxis])

#Tinh fuzzy support map cho mot component(1 diem, 1 idx)
#Ban chat day la mot matran he so cho cac diem xung quanh 1 component(1 diem)
def compute_support_map(idx, geodesics, min_dist, max_dist):
    #Phi la khoang cach tu tat cac ca vertexes den diem 'idx'
    phi = geodesics(idx)
    #Dung [min, max] distances de tinh fuzzy map tu reality distance map.
    return (np.clip(phi, min_dist, max_dist) - min_dist) / (max_dist - min_dist)

def my_smap(idx, K):
    #Define a matrix of my support map
    M = np.ones(shape=(22, 22))
    #M = np.ones(shape=(31, 31))
    '''
    M[4, [3,4,5]] = 0    
    M[8, [7, 8, 9]] = 0    
    M[20, [19, 20,21]] = 0
    M[16, [15, 16, 17]] = 0
    M[1, [1, 2, 6]] = 0

    M[9, [9,8]] = 0
    M[10, [10,11,1]] = 0
    M[14, [13,14,15]] = 0
    M[18, [12,18,19]] = 0
    M[5, [5,4]] = 0
    '''
    #M[idx, idx] = 0
    return M[idx, :]

def my_l1_pca_sparse_coding(skel_3d, K, my_percent):
    rest_shape = "first" # which frame to use as rest-shape ("first" or "average")
    smooth_min_dist = 0.1 # minimum geodesic distance for support map, d_min_in paper
    smooth_max_dist = 0.7 # maximum geodesic distance for support map, d_max in paper
    num_iters_max = 5 #10 # number of iterations to run
    sparsity_lambda = 0.0#20. # sparsity parameter, lambda in the paper

    rho = 10.0 # penalty parameter for ADMM
    num_admm_iterations = 10 # number of ADMM iterations

    # preprocessing: (external script)
        # rigidly align sequence
        # normalize into -0.5 ... 0.5 box

    verts = skel_3d
    F, N, _ = verts.shape

    if rest_shape == "first":
        Xmean = verts[0]
    elif rest_shape == "average":
        Xmean = np.mean(verts, axis=0)

    # prepare geodesic distance computation on the restpose mesh
#    compute_geodesic_distance = GeodesicDistanceComputation(Xmean, tris)

    # form animation matrix, subtract mean and normalize
    # notice that in contrast to the paper, X is an array of shape (F, N, 3) here
    X = verts - Xmean[np.newaxis] # Make X has zero mean
    pre_scale_factor =  1# / np.std(X) 
    X *= pre_scale_factor # Make X has std of 1
    R = X.copy() # residual
    R[np.isnan(R)] = 0
    R[np.isinf(R)] = 0


    # find initial components explaining the residual
    C = []
    W = []
#    idxes = np.asarray(np.arange(31))
    idxes = np.asarray([4, 8, 20, 16, 1, 9 ,10, 14, 18, 5])#np.zeros(K, dtype=int)
#    idxes = np.asarray([3,4,5,7,8,9,15,16,17,19,20, 21, 18, 1, 2, 6,10,11,12,13,14])#np.zeros(K, dtype=int)
    for k in xrange(K):
        # find the vertex explaining the most variance across the residual animation
        magnitude = (R**2).sum(axis=2)
        idx = np.argmax(magnitude.sum(axis=0))
#        idx = idxes[k]
#        idxes[k]  = idx
        print (idx)
        # find linear component explaining the motion of this vertex
        U, s, Vt = svd(R[:,idx,:].reshape(R.shape[0], -1).T, full_matrices=False)
        wk = s[0] * Vt[0,:] # weights
        # invert weight according to their projection onto the constraint set 
        # this fixes problems with negative weights and non-negativity constraints
#        '''
        wk_proj = project_weight(wk)
        wk_proj_negative = project_weight(-wk)
        wk = wk_proj \
                if norm(wk_proj) > norm(wk_proj_negative) \
                else wk_proj_negative
#        '''
#        s = 1 - compute_support_map(idx, compute_geodesic_distance, smooth_min_dist, smooth_max_dist)
        s = np.zeros(N)
        s[idx] = 1
#        s = 1 - my_smap(idx, K)
        print ('init', idx)
        # solve for optimal component inside support map
        ck = (np.tensordot(wk, R, (0, 0)) * s[:,np.newaxis])\
                / np.inner(wk, wk)
#        print 's.shape = ', s.shape
#        print 's[:,np.newaxis].shape = ', s[:,np.newaxis].shape
#        print 'np.tensordot(wk, R, (0, 0)).shape = ', np.tensordot(wk, R, (0, 0)).shape
#        print 'np.inner(wk, wk).shape = ', np.inner(wk,wk).shape
#        print ck.shape
        C.append(ck)
        W.append(wk)
        # update residual
        R -= np.outer(wk, ck).reshape(R.shape)
    C = np.array(C)
    W = np.array(W).T

    '''
    # undo scaling
    C /= pre_scale_factor
    e = 1
    return W, C, Xmean, e,idxes
    '''
    
    # prepare auxiluary variables
    Lambda = np.empty((K, N))
    U = np.zeros((K, N, 3))

    # main global optimization
    print ('START GLOBAL optimization')
    for it in xrange(num_iters_max):
        print ('GLOBAL optimization, ', it)
        # update weights
        #Rflat = [F1J1xyz,F1J2xyz...F1J21xyz; ...; FnJ1xyz, FnJ2xyz, FnJ21xyz]
        Rflat = R.reshape(F, N*3) # flattened residual
        for k in xrange(C.shape[0]): # for each component
            Ck = C[k].ravel()#Return a contiguous flattened array.
            #print 'Ck.shape = ', C[k].shape
            #print 'Ck.ravel = ', Ck.shape
            Ck_norm = np.inner(Ck, Ck)#dot product la Tich vo huong (scalar)
            if Ck_norm <= 1.e-8:
                # the component seems to be zero everywhere, so set it's activation to 0 also
                W[:,k] = 0
                continue # prevent divide by zero
            # block coordinate descent update
            Rflat += np.outer(W[:,k], Ck)
            opt = np.dot(Rflat, Ck) / Ck_norm
            W[:,k] = project_weight(opt)
            Rflat -= np.outer(W[:,k], Ck)
        # update spatially varying regularization strength
        for k in xrange(K):
            ck = C[k]
            # find vertex with biggest displacement in component and compute support map around it
            idx = (ck**2).sum(axis=1).argmax()
            idx = idxes[k]
            #idxes[k] = idx
#            idx = 2*k-1
            print ('Xet component ', k, ' idx = ', idx)
            support_map = np.ones(N)
#            support_map = my_smap(idx, K)
#            support_map[idx] = 0
#            support_map = compute_support_map(idx, compute_geodesic_distance, 
#                                              smooth_min_dist, smooth_max_dist)
            # update L1 regularization strength according to this support map
            Lambda[k] = sparsity_lambda * support_map

        # update components
        #USING MATLAB CODE HERE
        # Motion X -> Label
        # Weight W -> C_data
        # Output C -> 
        print ('W.shape = ', W.shape)
        print ('R.shape = ', R.shape)
        print ('Rflat.shape = ', Rflat.shape)
        print ('X.shape = ', X.shape)

        matlab = matlab_wrapper.MatlabSession(matlab_root="/Applications/MATLAB_R2014a.app")
        Label = X.reshape(X.shape[0],-1)
        print ('Label.shape = ', Label.shape)
        matlab.put('Label', Label)
        matlab.put('C_data', W)
        matlab.put('my_percent', my_percent)

        matlab.eval('dr_yu_sparsecoding')

        C = matlab.get('C_sparse')
        err = matlab.get('err')
        print ('C.shape = ', C.shape)
        print ('err = ', err)
        #Reshape C size k*66 -> k*22*3
        C = C.reshape(K, 22,3)
        print('done reshape')
        print('none_zero percent = ', 100*np.count_nonzero(C)/(K*22*3), '%')

        '''
        Z = C.copy() # dual variable
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        c = np.dot(W.T, X.reshape(X.shape[0], -1))
        solve_prefactored = cho_factor(G + rho * np.eye(G.shape[0]))
        # ADMM iterations
        for admm_it in xrange(num_admm_iterations):
            C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
            Z = prox_l1l2(Lambda, C + U, 1. / rho)
            U = U + C - Z
        # set updated components to dual Z, 
        # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
        C = Z
        '''
        # evaluate objective function
        print (W.shape,' ', C.shape)
        R = X - np.tensordot(W, C, (1, 0)) # residual
        sparsity = np.sum(Lambda * np.sqrt((C**2).sum(axis=2)))
        print ('sparsity err= ', sparsity)
        e = (R**2).sum() + sparsity
        # TODO convergence check
        print ("iteration %03d, E=%f" % (it, e))
    
    # undo scaling
    C /= pre_scale_factor
    print ('pre_scale_factor = ', pre_scale_factor)
    return W, C, Xmean, e,idxes
def my_l1_pca_no_support(skel_3d, K):
    rest_shape = "first" # which frame to use as rest-shape ("first" or "average")
    smooth_min_dist = 0.1 # minimum geodesic distance for support map, d_min_in paper
    smooth_max_dist = 0.7 # maximum geodesic distance for support map, d_max in paper
    num_iters_max = 100 # number of iterations to run
    sparsity_lambda = 2. # sparsity parameter, lambda in the paper

    rho = 10.0 # penalty parameter for ADMM
    num_admm_iterations = 100 # number of ADMM iterations

    # preprocessing: (external script)
        # rigidly align sequence
        # normalize into -0.5 ... 0.5 box

    verts = skel_3d
    F, N, _ = verts.shape

    if rest_shape == "first":
        Xmean = verts[0]
    elif rest_shape == "average":
        Xmean = np.mean(verts, axis=0)

    # prepare geodesic distance computation on the restpose mesh
#    compute_geodesic_distance = GeodesicDistanceComputation(Xmean, tris)

    # form animation matrix, subtract mean and normalize
    # notice that in contrast to the paper, X is an array of shape (F, N, 3) here
    X = verts - Xmean[np.newaxis] # Make X has zero mean
    pre_scale_factor = 1 / np.std(X) 
    X *= pre_scale_factor # Make X has std of 1
    R = X.copy() # residual

    # find initial components explaining the residual
    C = []
    W = []
    idxes = np.asarray([4, 8, 20, 16, 1, 9 ,10, 14, 18, 5])#np.zeros(K, dtype=int)
    for k in xrange(K):
        # find the vertex explaining the most variance across the residual animation
        magnitude = (R**2).sum(axis=2)
        idx = np.argmax(magnitude.sum(axis=0))
        idx = idxes[k]
        print (idx)
        # find linear component explaining the motion of this vertex
        U, s, Vt = svd(R[:,idx,:].reshape(R.shape[0], -1).T, full_matrices=False)
        wk = s[0] * Vt[0,:] # weights
        # invert weight according to their projection onto the constraint set 
        # this fixes problems with negative weights and non-negativity constraints
        wk_proj = project_weight(wk)
        wk_proj_negative = project_weight(-wk)
        wk = wk_proj \
                if norm(wk_proj) > norm(wk_proj_negative) \
                else wk_proj_negative
        s = np.ones(N)
        print ('init', idx)
        # solve for optimal component inside support map
        ck = (np.tensordot(wk, R, (0, 0)) * s[:,np.newaxis])\
                / np.inner(wk, wk)
#        print 's.shape = ', s.shape
#        print 's[:,np.newaxis].shape = ', s[:,np.newaxis].shape
#        print 'np.tensordot(wk, R, (0, 0)).shape = ', np.tensordot(wk, R, (0, 0)).shape
#        print 'np.inner(wk, wk).shape = ', np.inner(wk,wk).shape
#        print ck.shape
        C.append(ck)
        W.append(wk)
        # update residual
        R -= np.outer(wk, ck).reshape(R.shape)
    C = np.array(C)
    W = np.array(W).T

    '''
    # undo scaling
    C /= pre_scale_factor
    return W, C, Xmean, idxes
    '''

    # prepare auxiluary variables
    Lambda = np.empty((K, N))
    U = np.zeros((K, N, 3))

    # main global optimization
    print ('START GLOBAL optimization')
    for it in xrange(num_iters_max):
        print ('GLOBAL optimization, ', it)
        # update weights
        #Rflat = [F1J1xyz,F1J2xyz...F1J21xyz; ...; FnJ1xyz, FnJ2xyz, FnJ21xyz]
        Rflat = R.reshape(F, N*3) # flattened residual
        for k in xrange(C.shape[0]): # for each component
            Ck = C[k].ravel()#Return a contiguous flattened array.
            #print 'Ck.shape = ', C[k].shape
            #print 'Ck.ravel = ', Ck.shape
            Ck_norm = np.inner(Ck, Ck)#dot product la Tich vo huong (scalar)
            if Ck_norm <= 1.e-8:
                # the component seems to be zero everywhere, so set it's activation to 0 also
                W[:,k] = 0
                continue # prevent divide by zero
            # block coordinate descent update
            Rflat += np.outer(W[:,k], Ck)
            opt = np.dot(Rflat, Ck) / Ck_norm
            W[:,k] = project_weight(opt)
            Rflat -= np.outer(W[:,k], Ck)
        # update spatially varying regularization strength
        for k in xrange(K):
            ck = C[k]
            # find vertex with biggest displacement in component and compute support map around it
            idx = (ck**2).sum(axis=1).argmax()
            idx = idxes[k]
            idxes[k] = idx
            print ('Xet component ', k, ' idx = ', idx)
            support_map = np.ones(N)
            # update L1 regularization strength according to this support map
            Lambda[k] = sparsity_lambda * support_map

        # update components using SPARSE CODING
        #We assign W to C_data size (,)
        print ('W.shape = ', W.shape)
        #Assign motion R to Label size()        
        print ('R.shape = ', R.shape)

        Z = C.copy() # dual variable
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        c = np.dot(W.T, X.reshape(X.shape[0], -1))
        solve_prefactored = cho_factor(G + rho * np.eye(G.shape[0]))
        # ADMM iterations
        for admm_it in xrange(num_admm_iterations):
            C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
            Z = prox_l1l2(Lambda, C + U, 1. / rho)
            U = U + C - Z
        # set updated components to dual Z, 
        # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
        C = Z
        print('C.shape = ', C.shape) #k *22*3
        print('none_zero percent = ', 100*np.count_nonzero(C)/(K*22*3), '%')
        # evaluate objective function
        R = X - np.tensordot(W, C, (1, 0)) # residual
        sparsity = np.sum(Lambda * np.sqrt((C**2).sum(axis=2)))
        e = (R**2).sum() + sparsity
        # TODO convergence check
        print ("iteration %03d, E=%f" % (it, e))
    
    # undo scaling
    C /= pre_scale_factor
    return W, C, Xmean, idxes
if __name__ == '__main__':
    #Generate a random 3d matrix
    skel_3d = 100*np.random.rand(200, 60, 3)
    w, c, Xmean = my_l1_pca(skel_3d, 50)
    print (skel_3d.shape, w.shape, c.shape, Xmean.shape)
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    print (w.shape, c.shape  )  
    print (rec.shape)

