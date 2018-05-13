import os
import sys, scipy
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

sys.path.append('motion')
sys.path.append('my_sploc')
import BVH as BVH
import Animation as Animation
import matplotlib.animation as animation
from Quaternions import Quaternions
from Pivots import Pivots

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sploc_3d_skel_sparse_coding import my_l1_pca_sparse_coding, my_l1_pca_no_support
from sploc_3d_skel import my_l1_pca

from ReadDataForL1PCA import process_file, new_plot

def sparse_it(myX, k, Nbasic, my_percent):
    basic = []
    for i in xrange(Nbasic):
        w, c, Xmean, e, idxes = my_l1_pca_sparse_coding(myX, k, my_percent)
        basic.append([w,c, Xmean, e])
        print Xmean.shape, np.mean(Xmean)
        print Xmean
        #update residual
        rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
        myX = myX - rec
        #err = (myX**2).sum() # checked = returned e
        #print('err after unscaling = ', err)
    return basic

#Case 1: Replace xth basic of X1 by xth of X2, then save to X3
def sparse_yu_1(basic1, basic2, Nbasic, k, x):#(C1, C2, x, y, k):
    basic3 = basic1[:] #X3 = deep copy basic1
    C1 = basic1[x][1] #Cxth of basic1
    C2 = basic2[x][1] #Cxth of baiic2
    print C1.shape, C2.shape    
    #C1 size k *22*3
    #Ci size 22*3 -svd-> Ki(22*3) * Vi(3*3)
    #KK(all Ki) size k *22*3; VV(all Vi) size k *3*3 
    new_C = np.zeros_like(C1) #size k *22*3
    for i in range(k):
        print 'svd_i = ', i
        k1i, v1i = svd_ci(C1[i,:,:])
        k2i, v2i = svd_ci(C2[i,:,:])
        print k1i.shape, v1i.shape
        new_C[i,:,:] = np.dot(k2i, v1i)
    basic3[x][1] = new_C
    return basic3

#This function edit joint with abnormal limb lengh, start form hips
    
def plot_fully_rec(basic, Nbasic, label, my_limb = {}, save_mp4=False):
    fully_rec = 0
    for i in xrange(Nbasic):
        xmeani = basic[i][2]
        fully_rec = fully_rec   + np.tensordot(basic[i][0], basic[i][1], (1, 0)) + xmeani[np.newaxis]
    new_plot(fully_rec, np.arange(0), 'result2/'+label, False, True, my_limb = my_limb, save_mp4=save_mp4)    

def plot_err(basic, Nbasic, label = ''):
    err = []
    for i in xrange(Nbasic):
        err.append(basic[i][3])
    plt.plot(err)
    plt.title(label)
    plt.show()

#position size N * fr_no*3
def cal_limb(positions):
    #get first frame
    frame = positions[0,:,:]
    fr_no = frame.shape[0]
    pairs = [[1,2], [1,6], [1, 10],
    [2,3], [3,4], [4,5],
    [6,7], [7,8], [8,9],
    [10, 11], [ 11, 12],
    [12, 13], [12,14], [12, 18],
    [14,15], [15,16], [16, 17],
    [18,19], [19,20], [20,21]]
    limb = {}
    for pair in pairs:
        limb[str(pair[0])+';'+str(pair[1])] = np.linalg.norm(frame[pair[0]] - frame[pair[1]])
    return limb

def time_wraping(x, y):
    '''
    x = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]])
    y = np.array([[2,2], [3,3], [4,4]])
    distance, path = fastdtw(x, y, dist=euclidean)
    path = np.array(path)
    print(distance)
    '''
    foot_l = 4
    foot_r = 8
    x_l = x[:, foot_l, :]
    y_l = y[:, foot_l, :]
    x_l_0 = x[:, foot_l, 0]
    y_l_0 = y[:, foot_l, 0]
    distance, path = fastdtw(x_l, y_l, dist=euclidean)
    path = np.array(path)
    print x_l.shape, y_l.shape, path.shape

    plt.subplot(3, 1, 1)
    plt.plot(x_l_0)
    plt.plot(y_l_0)

    plt.subplot(3, 1, 2)
    plt.plot(x_l_0)
    plt.plot(y_l_0 + 10)
    for i in xrange(path.shape[0]):
        if i % 10 != 0:
            continue
        plt.plot([path[i,0], path[i,1]],[x_l_0[path[i,0]], y_l_0[path[i,1]] + 10],'r-')
    
    plt.subplot(3, 1, 3)
    plt.plot(x_l_0[path[:,0]])
    plt.plot(y_l_0[path[:,1]] + 10)
    for i in xrange(path.shape[0]):
        if i % 10 != 0:
            continue
        plt.plot([i, i],[x_l_0[path[i,0]], y_l_0[path[i,1]] + 10],'r-')

    plt.show()

#    input('dtwing...')
    x = x[path[:,0],:,:]
    y = y[path[:,1],:,:]
    return x[::2], y[::2]

def my_test_no_opt(): 
    # 12 paintful one leg
    # 08 neutral
    # 09 joy
    # 02 03 clumsy
    # 07 elderly
    # 11 military manner
    # 15 sad

#    positions, root_pos = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 270, 700, convert_60 = False)
#    positions2, root_pos2 = process_file('two_cmu_retargeted/142_15.bvh', 66, 30 , 300, 900, convert_60 = False)

    positions, root_pos = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 270, 700, convert_60 = False)
    positions2, root_pos2 = process_file('two_cmu_retargeted/142_12.bvh', 66, 30 , 270, 750, convert_60 = False)
        
    positions, positions2 =  time_wraping(positions, positions2)

    my_limb = cal_limb(positions)
    print positions.shape
    #Show readed motion
    #new_plot(positions, np.arange(0), 'Motion read from file')
    #new_plot(compensate_translation(positions, root_pos), np.arange(0), 'Compensated motion')

    #START NEW IDEA OF SPARSE CODING
    #1. Decompose motion into basic motions
    k =10
    my_percent = 0.1
    Nbasic = 3
    myX1 = positions[:]
    myX2 = positions2[:]
    basic1 = sparse_it(myX1, k, Nbasic, my_percent)
    #plot_err(basic1, Nbasic, 'Err of M1')
    basic2 = sparse_it(myX2, k, Nbasic, my_percent)
    #plot_err(basic2, Nbasic, 'Err of M2')
    plot_fully_rec(basic1, Nbasic, 'test fully_rec M1 basic 1 2 3', my_limb, save_mp4=True)
    plot_fully_rec(basic1, 2, 'test fully_rec M1 basic 1 2', my_limb, save_mp4=True)
    plot_fully_rec(basic1, 1, 'test fully_rec M1 basic 1', my_limb, save_mp4=True)
    plot_fully_rec(basic2, Nbasic, 'test fully_rec M2 basic 1 2 3', my_limb, save_mp4=True)

    x = 1
    basic3 = sparse_yu_1(basic1, basic2, Nbasic, k, x)
    plot_fully_rec(basic3, Nbasic, 'test fully_rec of M3, exchanging basic ' + str(x+1), my_limb, save_mp4=True)

    x = 2
    basic3 = sparse_yu_1(basic1, basic2, Nbasic, k, x)
    plot_fully_rec(basic3, Nbasic, 'test fully_rec of M3, exchanging basic ' + str(x+1), my_limb, save_mp4=True)


    '''
    myX = np.copy(positions)
    basic = sparse_it(myX, k, Nbasic, my_percent)

    #2. Fully reconstructed motion
    fully_rec = np.zeros_like(rec)
    for i in xrange(Nbasic):
        xmeani = basic[i][2]
        fully_rec = fully_rec   + np.tensordot(basic[i][0], basic[i][1], (1, 0)) + xmeani[np.newaxis]
    new_plot(fully_rec, idxes, 'Fully reconstructed motion')    
    '''
    #3. Synthesize


    #4. Retargeting


    input('sth')
    '''
    #Plot C components.
    Cc = np.copy(c)
    for i in range(k):
        Ci = Cc[i, :, :]
        x,y = scipy.where(Ci>0)
        Ci[x,y] = 1
        x,y = scipy.where(Ci<0)
        Ci[x,y] = 1
        plt.imshow(Ci, cmap='gray')
        plt.colorbar()
        plt.show()
    '''
    w, c, Xmean, e, idxes = my_l1_pca_sparse_coding(myX, k, my_percent)
    #Show reconstructed motion to verify decomposition process
    print idxes
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    ori_rec= np.copy(rec)

    new_plot(rec, idxes, 'reconstructed motion, after l1_pca')
    #new_plot(compensate_translation(rec, root_pos), idxes, 'Added translation: reconstructed motion, after l1_pca')

    '''
    #TESTING PLOT COMPONENTs saparately
    for i in xrange(k):        
        print c.shape, positions.shape
        frame = 300
        c1 = np.zeros((frame, positions.shape[1], positions.shape[2]))
        for j in xrange(frame):
            alpha = np.sin(j/30. * np.pi*2)
            alpha = np.abs(alpha)
            c1[j, :, :] = Xmean + alpha* c[i, :, :] #+ alpha* c[i+1, :, :]
#        c1 = Xmean + c[i, :, :].reshape(1, positions.shape[1], positions.shape[2])
        print c1.shape
        a = np.arange(1)
        a[0] = idxes[i]
        new_plot(c1, a, 'sin weigh * C'+ str(i))
    '''
    #DEBUGGING show ONLY some components [sta to end].
    for i in xrange(k-1):
        my_c = np.zeros_like(c)
        sta = i
        end = i+1
        my_c[sta:end, :, :] = c[sta:end, :, :]
        print my_c.shape
        print w.shape
        rec = np.tensordot(w, my_c, (1, 0)) + Xmean[np.newaxis]
        a = np.arange(1)
        a[0] = idxes[i]
        new_plot(rec , a, 'Show some components [sta to end]' + str(i))
        #new_plot(compensate_translation(rec, root_pos) , a, 'Show some components [sta to end]' + str(i))
        
    #Double some components
    new_c = np.zeros_like(c)
    new_c[:, :, :] = 3*c[:, :, :] 
    new_c[5:6, :, :] = c[5:6, :, :] 
    rec = np.tensordot(w, new_c, (1, 0)) + Xmean[np.newaxis]
    new_plot(rec, idxes, 'Double C5 C6')

    #Double some weight
    new_w = np.zeros_like(w)
    new_w = 3*w
    rec = np.tensordot(new_w, c, (1, 0)) + Xmean[np.newaxis]
    new_plot(rec, idxes, 'Triple w')

def compensate_translation(positions, root_pos):
    X = np.copy(positions)
    X[:, :, 0] = X[:, :, 0] + root_pos[:, :, 0]
    X[:, :, 2] = X[:, :, 2] + root_pos[:, :, 2]
    return X

def my_svd(Cs, Cc, alpha): #Size kcomponents * no. of joints, example (10 x 21)
    U1, S1, V1 = np.linalg.svd(Cs, full_matrices=True)
    print U1.shape, S1.shape, V1.shape
    S11 = np.zeros(Cs.shape, dtype=complex)
    k = min(Cs.shape) #The smaller dimension
    S11[:k, :k] = np.diag(S1)
    U1 = np.dot(U1, S11)
    #print U1.shape, S11.shape, V1.shape
    #print np.allclose(Cs, np.dot(U1, V1))
    
    U2, S2, V2 = np.linalg.svd(Cc, full_matrices=True)
    #print U2.shape, S2.shape, V2.shape
    S22 = np.zeros(Cc.shape, dtype=complex)
    k = min(Cc.shape) #The smaller dimension
    S22[:k, :k] = np.diag(S2)
    U2 = np.dot(U2, S22)
    #print U1.shape, S11.shape, V1.shape
    #print np.allclose(Cs, np.dot(U1, V1))

    # Do linear combination on (U Sigma) of style and content using weight alpha in [0,1]
    # newU = (1-alpha)(I Us Sigmas) + alpha(I Uc Sigmac)
    newU = (1-alpha)*U1 + alpha*U2

    #4. Reconstruct sO = Wc(newU)VcT
    newC = np.dot(newU, V2)
    return newC

def my_svd_2(Cs, Cc, alpha): #Size kcomponents * no. of joints, example (10 x 21)
#    U2, s2, V2 = np.linalg.svd(verts_2, full_matrices=False, compute_uv = True)
#    U_second = U2*s2             
#    temp = np.dot((alpha*U_first + (1-alpha) *U_second),V2);

    U1, S1, V1 = np.linalg.svd(Cs, full_matrices=False, compute_uv = True)
    print U1.shape, S1.shape, V1.shape
    Ua = U1*S1
    '''
    print U1.shape, S1.shape, V1.shape
    print np.allclose(Cs, np.dot(U1, V1))
    tU1, tS1, tV1 = np.linalg.svd(Cs, full_matrices=True)
    tS11 = np.zeros(Cs.shape, dtype=complex)
    k = Cs.shape[0] #The smaller dimension
    tS11[:k, :k] = np.diag(tS1)
    tU1 = np.dot(tU1, tS11)
    print tU1.shape, tS1.shape, tV1.shape
    print np.allclose(Cs, np.dot(tU1, tV1))
    print(np.allclose(tU1, U1))
    input('stop')
    '''
    U2, S2, V2 = np.linalg.svd(Cc, full_matrices=False, compute_uv= True)
    S22 = np.diag(S2)
    print(np.allclose(U2* S2, np.dot(U2, S22)))

    Ub = U2*S2

    # Do linear combination on (U Sigma) of style and content using weight alpha in [0,1]
    # newU = (1-alpha)(I Us Sigmas) + alpha(I Uc Sigmac)
    newU = (1-alpha)*Ua + alpha*Ub

    #4. Reconstruct sO = Wc(newU)VcT
    newC = np.dot(newU, V2)
    return newC

def svd_ci(C1): #Size no. joints * 3
    U1, S1, V1 = np.linalg.svd(C1, full_matrices=False, compute_uv = True)
    print U1.shape, S1.shape, V1.shape
    #S11 = np.diag(S1)
    Ka = U1*S1
    #C1 = Ka dot V1
    #print np.allclose(C1, np.dot(U1, np.dot(S11, V1)))
    #print np.allclose(C1, np.dot(np.dot(U1, S11), V1))
    #print np.allclose(U1*S1, np.dot(U1, S11))
    #print np.allclose(C1, np.dot(Ka, V1))
    return Ka, V1
#Case 1: Choose xth svdC of first and yth svdC of second motion, others are zero, then synthesize
def svd_yu_1(C1, C2, x, y, k):
    #C1 size k *22*3
    #Ci size 22*3 -svd-> Ki(22*3) * Vi(3*3)
    #KK(all Ki) size k *22*3; VV(all Vi) size k *3*3 
    new_C = np.zeros_like(C1) #size k *22*3
    for i in range(k):
        print 'svd_i = ', i
        k1i, v1i = svd_ci(C1[i,:,:])
        k2i, v2i = svd_ci(C2[i,:,:])
        print k1i.shape, v1i.shape
        if i == x: #Use C1
            new_C[i,:,:] = np.dot(k1i, v1i)
        if i == y: #Use C2
            new_C[i,:,:] = np.dot(k2i, v1i)
    return new_C
#Case 2: combine xth component with alpha, other component <- zero
def svd_yu_2(C1, C2, x, alpha, k):
    #C1 size k *22*3
    #Ci size 22*3 -svd-> Ki(22*3) * Vi(3*3)
    #KK(all Ki) size k *22*3; VV(all Vi) size k *3*3 
    new_C = np.zeros_like(C1) #size k *22*3
    for i in range(k):
        print 'svd_i = ', i
        if i == x: #Use C1
            k1i, v1i = svd_ci(C1[i,:,:])
            k2i, v2i = svd_ci(C2[i,:,:])
            new_ki = alpha*k1i + (1-alpha)*k2i
            new_C[i,:,:] = np.dot(new_ki, v1i)
    return new_C
#Case 4: combine xth component with alpha, keep other component like C1
def svd_yu_4(C1, C2, x, alpha, k):
    #C1 size k *22*3
    #Ci size 22*3 -svd-> Ki(22*3) * Vi(3*3)
    #KK(all Ki) size k *22*3; VV(all Vi) size k *3*3 
    #new_C = np.zeros_like(C1) #size k *22*3
    new_C = np.copy(C1)
    for i in range(k):
        print 'svd_i = ', i
        if i == x: #Use C1
            k1i, v1i = svd_ci(C1[i,:,:])
            k2i, v2i = svd_ci(C2[i,:,:])
            new_ki = alpha*k1i + (1-alpha)*k2i
            new_C[i,:,:] = np.dot(new_ki, v1i)
    return new_C
#Case 3: Retarget xth of C1 and yth of C2 to C3
def svd_yu_3(C1, C2, C3, x, y, k):
    #C1 size k *22*3
    #Ci size 22*3 -svd-> Ki(22*3) * Vi(3*3)
    #KK(all Ki) size k *22*3; VV(all Vi) size k *3*3 
    new_C = np.zeros_like(C1) #size k *22*3
    for i in range(k):
        print 'svd_i = ', i
        if i == x: #Use C1
            k1i, v1i = svd_ci(C1[i,:,:])
            k2i, v2i = svd_ci(C2[i,:,:])
            k3i, v3i = svd_ci(C3[i,:,:])
            new_C[i,:,:] = np.dot(k1i, v3i)
        if i == y: #Use C2
            k1i, v1i = svd_ci(C1[i,:,:])
            k2i, v2i = svd_ci(C2[i,:,:])
            k3i, v3i = svd_ci(C3[i,:,:])
            new_C[i,:,:] = np.dot(k2i, v3i)
    return new_C

#This funnction tests mixing component (0 0 ck 0 0)   
def my_mix_c():
    # 12 paintful one leg
    # 08 neutral
    # 02 clumsy
    # 07 elderly
    # 11 military manner
    #No. of components
    positions, root_pos = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 270, 700, convert_60 = False)
    positions2, root_pos2 = process_file('two_cmu_retargeted/142_12.bvh', 66, 30 , 270, 750, convert_60 = False)    
    sC, sS =  time_wraping(positions, positions2)
    #Cal limb length of original motion
    k = 10
    #Read content motion
#    sC, Ctran = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    my_limb = cal_limb(sC)
    #Read style motion
#    sS, Stran = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 340, 640)
    #Read 3rd motion
#    sT, Ttran = process_file('two_cmu_retargeted/142_02.bvh', 66, 30 , 340, 640)
    #Visulize
    print sC.shape, sS.shape
    new_plot(sC, np.arange(0), 'result2/'+'dtw M1 motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    new_plot(sS, np.arange(0), 'result2/'+'dtw M2 motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    input('sth')
    #new_plot(sT, np.arange(0), 'result2/'+' M3 motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)

    #new_plot(compensate_translation(sC, Ctran), np.arange(0), 'result2/'+ 'Compensated M1 motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    #new_plot(compensate_translation(sS, Stran), np.arange(0), 'result2/'+ 'Compensated M2 motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    #new_plot(compensate_translation(sT, Ttran), np.arange(0), 'result2/'+ 'Compensated M3 motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)

    #L1PCA decomposition
    Wc, Cc, cmean, cidxes = my_l1_pca(sC, k)
    Ws, Cs, smean, sidxes = my_l1_pca(sS, k)
#    Wt, Ct, tmean, tidxes = my_l1_pca(sT, k)

    '''
    #Visualize xth and yth component
    for x in xrange(0, 10):
        new_C = np.zeros_like(Cc)
        new_C[x, :,:] = Cc[x, :,:]
        c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
        #new_plot(c_rec, cidxes,'result2/'+'M1: component '+  str(x) + 'th',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)

    y = 0
    new_C = np.zeros_like(Cc)
    new_C[y,:,:] = Cs[y,:,:]
    rec = np.tensordot(Ws, new_C, (1, 0)) + smean[np.newaxis]
    #new_plot(compensate_translation(rec, Stran) , sidxes,'Ss component '+ str(y) )
    #new_plot(rec , sidxes,'result2/'+'M2 component '+ str(y) +'th',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    '''
    '''
    #YU_1 svd
    new_C = svd_yu_1(Cc, Cs, x,y, k)
    c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
    new_plot(c_rec, cidxes,'result2/'+'DRYU 1: Combine '+ str(x) + 'th of M1 with ' +str(y) + 'th of M2',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    '''
    '''
    #YU_2 svd for synthesizing one component, other <- zero
    x = 0
    for alphaaa in xrange(0,11):
        alpha = alphaaa/10.0
        new_C = svd_yu_2(Cc, Cs, x, alpha, k)
        c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
        new_plot(c_rec, cidxes,'result2/'+'DRYU 2: Combine '+ str(x) + 'th component of M1 M2 with alpha = ' +str(alpha),fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    '''
    #YU_4 svd for synthesizing one component, other <- Cc
    x = 0
    for alphaaa in xrange(0,11):
        alpha = alphaaa/10.0
        new_C = svd_yu_4(Cc, Cs, x, alpha, k)
        c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
        new_plot(c_rec, cidxes,'result2/'+'dtw DRYU 4: Combine '+ str(x) + 'th component of M1 M2 with alpha = ' +str(alpha) + ' keep remaining C = Cc',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    
    '''
    #YU_3 svd
    x = 1
    y = 2
    new_C = svd_yu_3(Cc, Cs, Ct, x, y, k)
    c_rec = np.tensordot(Wt, new_C, (1, 0)) + tmean[np.newaxis]
    new_plot(compensate_translation(c_rec,Ttran), tidxes,'Compensated DRYU 3: Retarget '+ str(x) + 'th component of M1 with ' +str(y) + 'th comp of M2 to M3', True)
    new_plot(c_rec, tidxes,'result2/'+'DRYU 3: Retarget '+ str(x) + 'th component of M1 with ' +str(y) + 'th comp of M2 to M3',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    '''

if __name__ == "__main__":
#    my_test_no_opt()
    my_mix_c()

