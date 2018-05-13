import os
import sys, scipy
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

from load_mat_rot import load_a_mat, save_a_mat

sys.path.append('motion')
sys.path.append('my_sploc')
import BVH as BVH
import Animation as Animation
import matplotlib.animation as animation
from Quaternions import Quaternions
from Pivots import Pivots
from Animation import Animation as AnimationClass

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sploc_3d_skel import my_l1_pca, my_l1_pca_no_support

from ReadDataForL1PCA import process_file, new_plot, process_file_rotation, softmin, softmax
def load_mat_e_n(filename):
    datadata = io.loadmat(filename)
    clumsy = datadata['clumsy']
    elderly = datadata['elderly']
    neutral = datadata['neutral']
    return elderly, neutral

def load_mat_c_n(filename):
    datadata = io.loadmat(filename)
    clumsy = datadata['clumsy']
    elderly = datadata['elderly']
    neutral = datadata['neutral']
    return clumsy, neutral

def save_mat(filename, S, C, O):
    io.savemat(filename, {'S':S, 'C':C, 'O': O})
def play_euler_rotation(rotations, positions, orients, offsets, parents, names, frametime, order, world,title):
    rotations = Quaternions.from_euler(np.radians(rotations), order, world)

    anim = AnimationClass(rotations, positions, orients, offsets, parents)

    """ Convert to 60 fps """
    anim = anim[::2]
    """ Do FK """
    global_positions = Animation.positions_global(anim)
    global_rotations = Animation.rotations_parents_global(anim)
    rotations = global_rotations[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    positions = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    fid_l, fid_r = np.array([4,5]), np.array([8,9])
    foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    
    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    #???
    trajectory_filterwidth = 3
    reference = positions[:,0] * np.array([1,0,1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    print(positions.shape)    
    
    """ Get Foot Contacts """
    #When left/right feet touch ground
    velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
    
    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    
    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    
    """ Get Root Velocity """
    #Velocity of hips, index 0
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()

    """ Remove Translation """
    #Subtract joint's coordinate by root coordinate (1st and 3rd)
    root_pos = np.copy(positions[:, 0:1, :])
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]

    new_plot(positions, np.arange(0), title)

def my_test_no_opt(): 
    # 12 paintful one leg
    # 08 neutral
    # 02 03 clumsy
    # 07 elderly
    # 11 military manner

    rotations, positions, orients, offsets, parents, names, frametime, order, world = process_file_rotation('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 700)

    k = 10
    w, c, Xmean, idxes = my_l1_pca(rotations, k)
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]

    play_euler_rotation(rec, positions, orients, offsets, parents, names, frametime, order, world, 'Reconstruced euler rotations motion')

    print positions.shape
    input('sth')
    #---------------------------------------------------------------------------------------------------
    #Show readed motion
    new_plot(positions, np.arange(0), 'Motion read from file')
    #new_plot(compensate_translation(positions, root_pos), np.arange(0), 'Compensated motion')
    k =21
    w, c, Xmean, idxes = my_l1_pca(positions, k)
    
    #Show reconstructed motion to verify decomposition process
    print idxes
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    ori_rec= np.copy(rec)

    new_plot(rec, idxes, 'reconstructed motion, after l1_pca')
    #new_plot(compensate_translation(rec, root_pos), idxes, 'Added translation: reconstructed motion, after l1_pca')

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
        new_plot(compensate_translation(rec, root_pos) , a, 'Show some components [sta to end]' + str(i))
        
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
    '''

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
#Case 2: combine xth component with alpha
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

def my_plot(w, title = 'my_plot'):
    print(title + ', shape = ' + str(w.shape))
    plt.plot(w)
    #plt.plot(w[0:w.shape[0]:10])
    plt.show()

def my_mix_c_rot():
    # 12 paintful one leg
    # 08 neutral
    # 02 clumsy
    # 07 elderly
    # 11 military manner
    '''
    rotations, positions, orients, offsets, parents, names, frametime, order, world = process_file_rotation('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 700)
    k = 10
    w, c, Xmean, idxes = my_l1_pca(rotations, k)
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    play_euler_rotation(rec, positions, orients, offsets, parents, names, frametime, order, world, 'Reconstruced euler rotations motion')
    '''
    #No. of components
    k = 10
    #Read content motion
    #sC, Ctran = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    sC, Cpositions, Corients, Coffsets, Cparents, Cnames, Cframetime, Corder, Cworld = process_file_rotation('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    #Read style motion
    #sS, Stran = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 340, 640)
    sS, Spositions, Sorients, Soffsets, Sparents, Snames, Sframetime, Sorder, Sworld = process_file_rotation('two_cmu_retargeted/142_11.bvh', 66, 30 , 340, 640)
    #Read 3rd motion
    #sT, Ttran = process_file('two_cmu_retargeted/142_02.bvh', 66, 30 , 340, 640)
    sT, Tpositions, Torients, Toffsets, Tparents, Tnames, Tframetime, Torder, Tworld = process_file_rotation('two_cmu_retargeted/142_02.bvh', 66, 30 , 340, 640)

    #L1PCA decomposition
    Wc, Cc, cmean, cidxes = my_l1_pca(sC, k)
    Ws, Cs, smean, sidxes = my_l1_pca(sS, k)
    Wt, Ct, tmean, tidxes = my_l1_pca(sT, k)

    #Visualize reconstructed M1
    c_rec = np.tensordot(Wc, Cc, (1, 0)) + cmean[np.newaxis]
    play_euler_rotation(c_rec, Cpositions, Corients, Coffsets, Cparents, Cnames, Cframetime, Corder, Cworld, 'Reconstruced M1 ')    

    #Visualize xth and yth component
    x = 0
    y = 0
    '''
    for x in xrange(k):   
        new_C = np.zeros_like(Cc)
        new_C[x, :,:] = Cc[x, :,:]
        c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
        play_euler_rotation(c_rec, Cpositions, Corients, Coffsets, Cparents, Cnames, Cframetime, Corder, Cworld, 'M1 component: i = ' + str(x))
    input('sth')
    new_C = np.zeros_like(Cs)
    new_C[y,:,:] = Cs[y,:,:]
    s_rec = np.tensordot(Ws, new_C, (1, 0)) + smean[np.newaxis]
    play_euler_rotation(s_rec, Spositions, Sorients, Soffsets, Sparents, Snames, Sframetime, Sorder, Sworld, 'M2 component: i = ' +str(y))    
    '''
    y=1
    new_C = np.zeros_like(Cs)
    new_C[y,:,:] = Cs[y,:,:]
    s_rec = np.tensordot(Ws, new_C, (1, 0)) + smean[np.newaxis]
    play_euler_rotation(s_rec, Spositions, Sorients, Soffsets, Sparents, Snames, Sframetime, Sorder, Sworld, 'M2 component: i = ' +str(y))    

    x = 0
    y = 1
    #YU_1 svd
    new_C = svd_yu_1(Cc, Cs, x,y, k)
    c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
    play_euler_rotation(c_rec, Cpositions, Corients, Coffsets, Cparents, Cnames, Cframetime, Corder, Cworld, 'DRYU 1: Combine '+ str(x) + 'th of M1 with ' +str(y) + 'th of M2')
    
    #------------STOP HERE-------- Mar 13
    # - Hardcode #joint = 31 in smap func in sploc_3d_skel
    # - Create func process_file_rotation that read a bvh, return all raw infomation
    # - Create func play_euler_rotation that reconstruct motion from rotation matrix + auxiliary info
    # ...

    input('stop')

    #YU_2 svd
    alpha = 0.5
    new_C = svd_yu_1(Cc, Cs, x,alpha, k)
    c_rec = np.tensordot(Wc, new_C, (1, 0)) + cmean[np.newaxis]
    track = new_plot(c_rec, cidxes,'DRYU 2: Combine '+ str(x) + 'th component of M1 M2 with alpha = ' +str(alpha), True)
    plt.plot(track)
    plt.ylabel('DR YU 2: Tracking of fistance from J3 to J4')
    plt.show()


    #YU_3 svd
    new_C = svd_yu_3(Cc, Cs, Ct, x, y, k)
    c_rec = np.tensordot(Wt, new_C, (1, 0)) + tmean[np.newaxis]
    track = new_plot(compensate_translation(c_rec,Ttran), tidxes,'Compensated DRYU 3: Retarget '+ str(x) + 'th component of M1 with ' +str(y) + 'th comp of M2 to M3', True)
    plt.plot(track)
    plt.ylabel('DR YU 3: Tracking of fistance from J3 to J4')
    plt.show()
    new_plot(c_rec, tidxes,'DRYU 3: Retarget '+ str(x) + 'th component of M1 with ' +str(y) + 'th comp of M2 to M3')

def my_test_17Mar():
    #No. of components
    k = 10
    #Read content motion
    tran, rot = load_a_mat('rotation_db.mat', 'clumsy')
    print tran.shape, rot.shape
    #Decompose
    w, c, Xmean, idxes = my_l1_pca(rot, k)
    #Reconstruct the original
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    print rec.shape
    '''
    #Reconstruct one component
    i = 1
    my_c = np.zeros_like(c)
    sta = i
    end = i+1
    my_c[sta:end, :, :] = c[sta:end, :, :]
    print my_c.shape
    print w.shape
    rec = np.tensordot(w, my_c, (1, 0)) + Xmean[np.newaxis]
    '''
    save_a_mat('ci.mat',tran,rec)
    

    '''
    #Reconstruct each components
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
    '''
    save_a_mat('my_output.mat',tran,rec)

if __name__ == "__main__":
#    my_test_no_opt()
#    my_mix_c()
    my_mix_c_rot()
#    my_test_17Mar()

