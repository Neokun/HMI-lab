import os
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

sys.path.append('motion')
sys.path.append('my_sploc')
import BVH as BVH
import Animation as Animation
from Animation import Animation as AnimationClass
import matplotlib.animation as animation
from Quaternions import Quaternions
from Pivots import Pivots

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sploc_3d_skel import my_l1_pca

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)
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

def new_plot(clips, idxes, label='new_plot', track = False, fix_limb = False, my_limb = {}, save_mp4=False, pltshow = False):
    limb = []
    labels = ['   {0}'.format(i) for i in range(clips.shape[1])]
    print 'start new_plot'
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(label, fontsize=14, fontweight='bold')
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=-90)
    minz = np.min(clips[:,:,2])
    maxz = np.max(clips[:,:,2])
    minx = np.min(clips[:,:,0])
    maxx = np.max(clips[:,:,0])
    miny = np.min(clips[:,:,1])
    maxy = np.max(clips[:,:,1])
    print minz, maxz
    ax.set_xlim3d(minx, maxx)
    ax.set_zlim3d(minz, maxz)
    ax.set_ylim3d(miny, maxy)
    fr_no = clips.shape[0]
    jo_no = clips.shape[1]
    pairs = [[1,2], [1,6], [1, 10],
    [2,3], [3,4], [4,5],
    [6,7], [7,8], [8,9],
    [10, 11], [ 11, 12],
    [12, 13], [12,14], [12, 18],
    [14,15], [15,16], [16, 17],
    [18,19], [19,20], [20,21]]

    #Gen neighbor matrix
    Nei = {}
    for pair in pairs:
        if Nei.has_key(pair[0]):           
            Nei[pair[0]].append(pair[1])
        else:
            Nei[pair[0]] = [pair[1]]
    #print Nei
    #Child matrix
    Chil = {}
    Chil[1] = [2,3,4,5,6,7,8,9]
    Chil[2] = [3,4,5]
    Chil[3] = [4,5]
    Chil[4] = [5]
    Chil[5] = []
    Chil[6] = [7,8,9]
    Chil[7] = [8,9]
    Chil[8] = [9]
    Chil[9] = []
    Chil[10] = [11,12,13,14,15,16,17,18,19,20,21]
    Chil[11] = [12,13,14,15,16,17,18,19,20,21]
    Chil[12] = [13,14,15,16,17,18,19,20,21]
    Chil[13] = []
    Chil[14] = [15,16,17]
    Chil[15] = [16,17]
    Chil[16] = [17]
    Chil[17] = []
    Chil[18] = [19,20,21]
    Chil[19] = [20,21]
    Chil[20] = [21]
    Chil[21] = []
    #print Chil

    if fix_limb:
        #Lopp over all frames
        for f in xrange(fr_no):
            frame = clips[f,:,:]
            #Loop each joint from hips 1
            for i in xrange(1, jo_no-1):
                Jparent = frame[i,:] #x,y,z
                print i,' ', Jparent
                if not Nei.has_key(i):
                    print 'no neighbor, continue'
                    continue
                neighbor = Nei[i]
                for nei in neighbor:
                    #1. Cal limb(Jchil, Jparent)
                    Jchil = frame[nei,:]
                    l = np.linalg.norm(Jchil- Jparent)
                    print nei, ' nei ', Jchil
                    #2. Cal J'chil satisifes limb length constraint
                    l_constraint = my_limb[str(i)+';'+str(nei)]
                    print l_constraint
                    Jchilnew = Jparent + (l_constraint/l)*(Jchil-Jparent)
                    #3. Cal vector J'chil - Jchil
                    Voldnew = Jchilnew - Jchil
                    frame[nei, :] = Jchilnew
                    print Voldnew
                    #4. Apply translation over all childs of Jchil    
                    all_child = Chil[nei]
                    for achild in all_child:
                        #Translate by Voldnew vector
                        frame[achild,:] = frame[achild, :] + Voldnew 

    def animate(i):
        #print 'animate ', i
        fr = clips[i,:,:]
        ax.cla()
        #draw joint index
        for x, y, z, l in zip(fr[:,0], fr[:,1], fr[:,2], labels): 
            ax.text(x,y,z,l, color='black', fontsize = 8)
        #draw lines
        for pair in pairs:
            ax.plot(fr[pair,0], fr[pair,1], fr[pair,2], '-b',linewidth = 5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(minx, maxx)
        ax.set_zlim3d(minz, maxz)
        ax.set_ylim3d(miny, maxy)
        #draw joints
        ax.plot(fr[:,0], fr[:,1], fr[:,2], 'ro')
        #j34 = np.linalg.norm(fr[3,:] - fr[4,:])
        #print 'distance j3rd to j4th = ', j34
        if track:
            limb.append(j34)
        if idxes.shape[0] > 0:
#            ax.plot(fr[0:20:2,0], fr[0:20:2,1], fr[0:20:2,2], 'ro') # odd index highlight
            ax.plot(fr[idxes,0], fr[idxes,1], fr[idxes,2], 'ro')
#            ax.plot([fr[4,0]], [fr[4,1]], [fr[4,2]], 'g*')
            pass
#        ax.plot([fr[5,0]], [fr[5,1]], [fr[5,2]], 'ro')
    ani = animation.FuncAnimation(fig, animate, np.arange(fr_no), interval=1)
    if save_mp4:
        ani.save(label + '.mp4', fps=60, dpi=160)
    if pltshow:
        plt.show()
    print 'end new_plot'
    return limb

def plotframe(clips):
    print 'plotframe func, ', clips.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=10., azim=100)
    clips = clips[::10]
    fr_no = clips.shape[0]
    print 'frame_no, ', fr_no
    for i in range(fr_no):
        if i%10 == 0:
            print (i)
            plt.pause(1)
        #print (clips[i, :, :])
        fr = clips[i,:,:]
        if i == 1:
            plt.pause(20)
        #ax.scatter(fr[:,0], fr[:,1], fr[:,2], c='b', marker='-')
        ax.plot(fr[:,0], fr[:,1], fr[:,2], '-o')
        plt.pause(.001)
    plt.show()
    print 'end of plotframe'

def process_file_rotation(filename, window, window_step, c_start = 0, c_end = -1):
    rotations, positions, orients, offsets, parents, names, frametime, order, world = BVH.load(filename, euler = True)
    rotations = rotations[c_start:c_end, :, :]
    positions = positions[c_start:c_end, :, :]
    #rotations[:, 7, :] = 2*rotations[:, 7 ,:] 
    #rotations = alpha*rotations #TEST DOUBLE THE ANGLE
    return rotations, positions, orients, offsets, parents, names, frametime, order, world
    rotations = Quaternions.from_euler(np.radians(rotations), order, world)

    anim = AnimationClass(rotations, positions, orients, offsets, parents)

    '''Remove some redundency at the begining and the end of motion'''
    print(c_start, c_end)
    print(anim.shape)
    anim = anim[c_start:c_end:1, :]
    print(anim.shape)
    #anim = anim[]
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
    print 'before return, ', positions.shape
    return positions, root_pos


def process_file(filename, window, window_step, c_start = 0, c_end = -1, convert_60 = True):
    anim, names, frametime = BVH.load(filename)

    '''Remove some redundency at the begining and the end of motion'''
    print(c_start, c_end)
    print(anim.shape)
    anim = anim[c_start:c_end:1, :]
    print(anim.shape)

    #anim = anim[]
    """ Convert to 60 fps """
    if convert_60:
        anim = anim[::2]
    
    """ Do FK """
    global_positions = Animation.positions_global(anim)
    global_rotations = Animation.rotations_parents_global(anim)
    print (global_positions.shape)
    print global_rotations.shape
    """ Remove Uneeded Joints """
    positions = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    rotations = global_rotations[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    """ Put on Floor """
    print (positions.shape)
    #new_plot(positions)
#    return positions
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
    print 'before return, ', positions.shape
    return positions, root_pos
#    new_plot(positions)
#    input('sth')
    print(positions.shape)
    
    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]    
    positions = rotation * positions
    print(positions.shape)
 
    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
    
#    plotframe(positions)
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    print (positions.shape)

    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    print (positions.shape)
    '''
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    print (positions.shape)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    print (positions.shape)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    print (positions.shape)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    print (positions.shape)
    '''
    
    """ Slide over windows """
    windows = []
    windows_classes = []
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            left[:,-7:-4] = 0.0
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            right[:,-7:-4] = 0.0
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        
        """ Find Class """
        cls = -1
        if filename.startswith('hdm05'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
            cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
        if filename.startswith('styletransfer'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0]
            cls = np.array([
                styletransfer_motions.index('_'.join(cls_name.split('_')[1:-1])),
                styletransfer_styles.index(cls_name.split('_')[0])])
        windows_classes.append(cls)
        
    return windows, windows_classes

    
def get_files(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh']

def zerosmean(X):
    X = np.swapaxes(X, 1, 2).astype(float)
    feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])
    Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
    Xmean[:,-7:-4] = 0.0
    Xmean[:,-4:]   = 0.5
    Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
    Xstd[:,feet]  = 0.9 * Xstd[:,feet]
    Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
    Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
    Xstd[:,-4:]   = 0.5

    X = (X - Xmean) / Xstd
    X = np.swapaxes(X, 1, 2)
    return X

#Limited classes
def read_two_cmu_style():
    cmu_files = get_files('two_cmu_retargeted')
    cmu_clips = []
    styles = []
    s_classes = [0, 0, 1, 2, 3]
    s_skips = [200, 1000, 150, 300, 250]
    e_skips = [-1, -1, -1, -1, -1]
    for i, item in enumerate(cmu_files):
        print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
        clips, _ = process_file(item, 66, 30 , s_skips[i], e_skips[i])        
        cmu_clips += clips
        s = [s_classes[i]] * len(clips)
        styles += s
    data_clips = np.array(cmu_clips)
    data_labels = np.array(styles)
    print (data_clips.shape, data_labels.shape)
    #np.savez_compressed('cmu_styles_73_30', clips=data_clips)

    #Permute dataset randomly
    idx = np.random.permutation(data_labels.shape[0])
    data_clips = data_clips[idx, :, :]
    print (data_clips.shape)
    print (data_clips.min(), data_clips.max())
    data_labels = data_labels[idx]
    data_clips = zerosmean(data_clips)
    print (data_clips.min(), data_clips.max())
    #X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    print (data_clips.shape)
    #Try to swap x and y
    data_clips = np.swapaxes(data_clips, 1,2)
    print (data_clips.shape)
    return data_clips, data_labels
#Read sub_cmu_retarget, skip some redundency
def read_sub_cmu_style():
    cmu_files = get_files('sub_cmu_retargeted')
    cmu_clips = []
    styles = []
    s_classes = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 11, 12]
    s_skips = [200, 1000, 200, 260, 150, 200, 300, 250, 200, 250, 250, 350, 200, 250, 250]
    e_skips = [-1, -1, -1, 2500, -1, 2300, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for i, item in enumerate(cmu_files):
        print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
        clips, _ = process_file(item, 66, 30 , s_skips[i], e_skips[i])        
        cmu_clips += clips
        s = [s_classes[i]] * len(clips)
        styles += s
    data_clips = np.array(cmu_clips)
    data_labels = np.array(styles)
    print (data_clips.shape, data_labels.shape)
    #np.savez_compressed('cmu_styles_73_30', clips=data_clips)

    #Permute dataset randomly
    idx = np.random.permutation(data_labels.shape[0])
    data_clips = data_clips[idx, :, :]
    print (data_clips.shape)
    print (data_clips.min(), data_clips.max())
    data_labels = data_labels[idx]
    data_clips = zerosmean(data_clips)
    print (data_clips.min(), data_clips.max())
    #X = np.swapaxes(X, 1, 2).astype(theano.config.floatX)
    print (data_clips.shape)
    #Try to swap x and y
    data_clips = np.swapaxes(data_clips, 1,2)
    print (data_clips.shape)
    return data_clips, data_labels

#Read motion from file, return: Train, validation, test sets
def read_cmu_style():
    cmu_files = get_files('cmu_retargeted')
    cmu_clips = []
    styles = []
    s_classes = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 17, 18]
    for i, item in enumerate(cmu_files):
        print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
        clips, _ = process_file(item, 73*2, 73)        
        cmu_clips += clips
        s = [s_classes[i]] * len(clips)
        styles += s
    data_clips = np.array(cmu_clips)
    data_labels = np.array(styles)
    print (data_clips.shape, data_labels.shape)
    #np.savez_compressed('cmu_styles_73_30', clips=data_clips)

    #Permute dataset randomly
    idx = np.random.permutation(data_labels.shape[0])
    data_clips = data_clips[idx, :, :]
    print (data_clips.shape)
    print (data_clips.min(), data_clips.max())
    data_labels = data_labels[idx]
    data_clips = zerosmean(data_clips)
    print (data_clips.min(), data_clips.max())
    input('sth')
    return data_clips, data_labels

def my_test(): 
    # 12 paintful one leg
    # 08 neutral
    # 02 03 clumsy
    # 07 elderly
    # 11 military manner
    positions = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 300, 700)

    #TEST: NORMALIZE TO ZERO MEAN AND STD OF ONE
    print positions.shape

    pmean = np.mean(positions, axis=0)
    pstd = np.std(positions, axis=0)
    pstd[0, :] = 1
    new_positions = (positions-pmean)/pstd
    positions = new_positions

    print positions.shape
    #Show readed motion
    new_plot(positions, np.arange(0), 'Motion read from file')
    k =10
    w, c, Xmean, idxes = my_l1_pca(positions, k)
    #Show reconstructed motion to verify decomposition process
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    ori_rec= rec

    rec = rec*pstd + pmean

    new_plot(rec, idxes, 'reconstructed motion, after l1_pca')
    print np.allclose(rec, positions)

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

    #Activate some components: Let say I choose some component, then active them and see the effect.
    frame = 300
    acts = [0, 1,2,3,4, 5,6,7,8,9]
    A = np.zeros((frame, positions.shape[1], positions.shape[2]))
    for i in xrange(frame):
        A[i, :, :] = Xmean
        alpha = np.sin(i/30. * np.pi*2)
        alpha = np.abs(alpha)        
        for j in acts:
            A[i, :, :] = A[i, :, :] + alpha * c[j, :, :]
    new_plot(A, idxes, '300 frames of Xmean + some activating components')

    #Boost some components in a motion: = Original motion + some extra SIN activating components
    acts = [6]
    A = ori_rec
    frame = A.shape[0]
    for i in xrange(frame):
        alpha = np.sin(i/30. * np.pi*2)
        alpha = 3*np.abs(alpha)        
        for j in acts:
            A[i, :, :] = A[i, :, :] + alpha * c[j, :, :]
    rec = A
    rec = rec*pstd + pmean
    A = rec
    new_plot(A, idxes, 'Activate some components with sin funcs')

#    '''
    #DEBUGGING show ONLY some components [sta to end].
    for i in xrange(k-1):
        my_c = np.zeros_like(c)
        sta = i
        end = i+1
        my_c[sta:end, :, :] = c[sta:end, :, :]
        rec = np.tensordot(w, my_c, (1, 0)) + Xmean[np.newaxis]
        a = np.arange(1)
        a[0] = idxes[i]
        rec = rec*pstd + pmean
        new_plot(rec, a, 'Show some components [sta to end]' + str(i))
#    '''

    #Double some components
    new_c = np.zeros_like(c)
    new_c[:, :, :] = 3*c[:, :, :] 
    new_c[5:6, :, :] = c[5:6, :, :] 
    rec = np.tensordot(w, new_c, (1, 0)) + Xmean[np.newaxis]
    rec = rec*pstd + pmean
    new_plot(rec, idxes, 'Double C5 C6')

    #Double some weight
    new_w = np.zeros_like(w)
    new_w = 3*w
    rec = np.tensordot(new_w, c, (1, 0)) + Xmean[np.newaxis]
    new_plot(rec, idxes, 'Triple w')
    '''
    #Visualize weights
    for i in xrange(k):
        wi = w[:, i]
        plt.plot(wi)
        plt.ylabel('w' + str(i))
        plt.show()
    '''
def compensate_translation(positions, root_pos):
    X = np.copy(positions)
    X[:, :, 0] = X[:, :, 0] + root_pos[:, :, 0]
    X[:, :, 2] = X[:, :, 2] + root_pos[:, :, 2]
    return X

def my_test_no_opt(): 
    # 12 paintful one leg
    # 08 neutral
    # 02 03 clumsy
    # 07 elderly
    # 11 military manner
    positions, root_pos = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 300, 700)
    print positions.shape
    #Show readed motion
    new_plot(positions, np.arange(0), 'Motion read from file')
    new_plot(compensate_translation(positions, root_pos), np.arange(0), 'Compensated motion')
    k =10
    w, c, Xmean, idxes = my_l1_pca(positions, k)
    #Show reconstructed motion to verify decomposition process
    print idxes
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    ori_rec= np.copy(rec)

    new_plot(rec, idxes, 'reconstructed motion, after l1_pca')
    new_plot(compensate_translation(rec, root_pos), idxes, 'Added translation: reconstructed motion, after l1_pca')

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

    #Activate some components: Let say I choose some component, then active them and see the effect.
    frame = 300
    acts = [0, 1,2,3,4, 5,6,7,8,9]
    A = np.zeros((frame, positions.shape[1], positions.shape[2]))
    for i in xrange(frame):
        A[i, :, :] = Xmean
        alpha = np.sin(i/30. * np.pi*2)
        alpha = np.abs(alpha)        
        for j in acts:
            A[i, :, :] = A[i, :, :] + alpha * c[j, :, :]
    new_plot(A, idxes, '300 frames of Xmean + some activating components')

    #Boost some components in a motion: = Original motion + some extra SIN activating components
    acts = [6]
    A = ori_rec
    frame = A.shape[0]
    for i in xrange(frame):
        alpha = np.sin(i/30. * np.pi*2)
        alpha = 3*np.abs(alpha)        
        for j in acts:
            A[i, :, :] = A[i, :, :] + alpha * c[j, :, :]
    new_plot(A, idxes, 'Activate some components with sin funcs')

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
        new_plot(rec, a, 'Show some components [sta to end]' + str(i))

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
    #Visualize weights
    for i in xrange(k):
        wi = w[:, i]
        plt.plot(wi)
        plt.ylabel('w' + str(i))
        plt.show()
    '''
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
def my_transfer():
    # 12 paintful one leg
    # 08 neutral
    # 02 clumsy
    # 07 elderly
    # 11 military manner

    #No. of components
    k = 10
    #Read content motion
    sC, tran = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    #Read style motion
    sS, tran = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 340, 640)
    #Visulize
    print sC.shape, sS.shape
    new_plot(sC, np.arange(0), ' Content motion')
    new_plot(sS, np.arange(0), ' Style motion')

    #L1PCA decomposition
    Wc, Cc, cmean, cidxes = my_l1_pca(sC, k)
    Ws, Cs, smean, sidxes = my_l1_pca(sS, k)

    #MY M-SVD
    alpha = 1.0
    Cnew = np.zeros_like(Cc)
    print Cnew.shape
    for i in xrange(k):
        print 'SVDing ', i
        Cnew[i,:,:] = my_svd_2(Cs[i,:,:], Cc[i,:,:], alpha)
    new_rec = np.tensordot(Wc, Cnew, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec,np.arange(0), 'TEST: svd with alpha = ' + str(alpha))

    alpha = 0.5
    Cnew = np.zeros_like(Cc)
    print Cnew.shape
    for i in xrange(k):
        print 'SVDing ', i
        Cnew[i,:,:] = my_svd_2(Cs[i,:,:], Cc[i,:,:], alpha)
    new_rec = np.tensordot(Wc, Cnew, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec,np.arange(0), 'TEST: svd with alpha = ' + str(alpha))

    #Combination
    alpha = 0.5
    Csc = alpha * Cc + (1-alpha) * Cs
    new_rec = np.tensordot(Wc, Csc, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec, np.arange(0), 'DR. YU: Linear C combination with alpha = ' + str(alpha))

    #Non-linear swap components
    acts = [1, 2, 4, 5, 7, 9]
    for i in acts:
        Csc = Cc
        Csc[0:i,: ,: ] = Cs[0:i, :, :]
        new_rec = np.tensordot(Wc, Csc, (1, 0)) + cmean[np.newaxis]
        new_plot(new_rec, np.arange(0), 'Swapping components :' + str(i))


    '''
    #Component combination via SVD

    #Reconstruction
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    print w.shape, c.shape    
    print rec.shape
    new_plot(rec)
    '''

#This funnction tests mixing component (0 0 ck 0 0)   
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
def my_mix_c():
    # 12 paintful one leg
    # 08 neutral
    # 02 clumsy
    # 07 elderly
    # 11 military manner

    #No. of components
    k = 10
    #Read content motion
#    sC, tran = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    #Read style motion
#    sS, tran = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 340, 640)

    positions, root_pos = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 270, 700, convert_60 = False)
    positions2, root_pos2 = process_file('two_cmu_retargeted/142_12.bvh', 66, 30 , 270, 750, convert_60 = False)
    
    sC, sS =  time_wraping(positions, positions2)

    #Cal limb length of original motion
    my_limb = cal_limb(sC)

    #Visulize
    print sC.shape, sS.shape
    
    new_plot(sC, np.arange(0), 'dtw Mix_c Content motion',fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)
    new_plot(sS, np.arange(0), 'dtw Mix_c Style motion', my_limb = my_limb,fix_limb = True, save_mp4=True, pltshow = False)

    #L1PCA decomposition
    Wc, Cc, cmean, cidxes = my_l1_pca(sC, k)
    Ws, Cs, smean, sidxes = my_l1_pca(sS, k)

    #Show reconstructed motions
    c_rec = np.tensordot(Wc, Cc, (1, 0)) + cmean[np.newaxis]
    s_rec = np.tensordot(Ws, Cs, (1, 0)) + smean[np.newaxis]
    new_plot(c_rec, cidxes,'Mix_c Content Reconstruction', fix_limb = True,my_limb = my_limb, save_mp4=True, pltshow = False)
    new_plot(s_rec, sidxes, 'Mix_c Style Reconstruction', fix_limb = True,my_limb= my_limb, save_mp4=True, pltshow = False)

    #SVD and mix one component
    alpha = 0.1
    Cnew = np.zeros_like(Cc)
    print Cnew.shape
    for i in [0,1]:#Two legs
        print 'SVDing ', i
        Cnew[i,:,:] = my_svd_2(Cs[i,:,:], Cc[i,:,:], alpha)
    new_rec = np.tensordot(Wc, Cnew, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec, cidxes, 'Mix_c Mixing first and second ck ' + str(alpha), fix_limb = True,my_limb = my_limb, save_mp4=True, pltshow = False)

    #MY M-SVD
    alpha = 0.9
    Cnew = np.zeros_like(Cc)
    print Cnew.shape
    for i in xrange(k):
        print 'SVDing ', i
        Cnew[i,:,:] = my_svd_2(Cs[i,:,:], Cc[i,:,:], alpha)
    new_rec = np.tensordot(Wc, Cnew, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec,np.arange(0), 'Mix_c TEST: svd with alpha = ' + str(alpha),fix_limb = True, my_limb = my_limb, save_mp4=True, pltshow = False)

    alpha = 0.5
    Cnew = np.zeros_like(Cc)
    print Cnew.shape
    for i in xrange(k):
        print 'SVDing ', i
        Cnew[i,:,:] = my_svd_2(Cs[i,:,:], Cc[i,:,:], alpha)
    new_rec = np.tensordot(Wc, Cnew, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec,np.arange(0), 'Mix_c TEST: svd with alpha = ' + str(alpha), fix_limb = True,my_limb = my_limb, save_mp4=True, pltshow = False)

    '''
    #Combination
    alpha = 0.0
    Csc = alpha * Cc + (1-alpha) * Cs
    new_rec = np.tensordot(Wc, Csc, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec, np.arange(0), 'no dtw Mix_c DR. YU: Linear C combination with alpha = ' + str(alpha), save_mp4=True, pltshow = False)

    #Non-linear swap components
    acts = [1, 2, 4, 5, 7, 9]
    for i in acts:
        Csc = Cc
        Csc[0:i,: ,: ] = Cs[0:i, :, :]
        new_rec = np.tensordot(Wc, Csc, (1, 0)) + cmean[np.newaxis]
        new_plot(new_rec, np.arange(0), 'Mix_c Swapping components :' + str(i), save_mp4=True, pltshow = False)
    '''
    '''
    #Component combination via SVD

    #Reconstruction
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    print w.shape, c.shape    
    print rec.shape
    new_plot(rec)
    '''

def my_transfer_2():
    # 12 paintful one leg
    # 08 neutral
    # 02 clumsy
    # 07 elderly
    # 11 military manner

    #No. of components
    k = 10
    #Read content motion
    sC = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    #Read style motion
    sS = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 500, 1000)
    #Visulize
    print sC.shape, sS.shape
    new_plot(sC, np.arange(0))
    new_plot(sS, np.arange(0))

    #L1PCA decomposition
    Wc, Cc, cmean, cidxes = my_l1_pca(sC, k)
    Ws, Cs, smean, sidxes = my_l1_pca(sS, k)

    #MY M-SVD
    alpha = 0.5
    Cnew = np.zeros_like(Cc)
    print Cnew.shape
    for i in xrange(k):
        print 'SVDing ', i, ' ', Cnew.shape
        Cnew[i,:,:] = my_svd_2(Cs[i,:,:], Cc[i,:,:], alpha)
    new_rec = np.tensordot(Wc, Cnew, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec,np.arange(0), 'svd with alpha = ' + str(alpha))

    #Combination
    alpha = 0.5
    Csc = alpha * Cc + (1-alpha) * Cs
    new_rec = np.tensordot(Wc, Csc, (1, 0)) + cmean[np.newaxis]
    print new_rec.shape
    new_plot(new_rec, np.arange(0), 'Linear C combination with alpha = ' + str(alpha))
    input('sth')

    '''
    #Component combination via SVD

    #Reconstruction
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    print w.shape, c.shape    
    print rec.shape
    new_plot(rec)
    '''

def my_test_ab(): 
    # 12 paintful one leg
    # 08 neutral
    # 02 03 clumsy
    # 07 elderly
    # 11 military manner
    p1, residual = process_file('two_cmu_retargeted/142_08.bvh', 66, 30 , 300, 600)
    p2, residual = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 340, 640)
#    new_plot(p1, np.arange(0), 'p1')
#    new_plot(p2, np.arange(0), 'p2')

    print p1.shape, p2.shape
    positions = np.concatenate([p1, p2], axis = 0)
    print positions.shape
    new_plot(positions, np.arange(0), 'p1 + p2')

#    input('test_ab')

#    positions = process_file('two_cmu_retargeted/142_11.bvh', 66, 30 , 300, 700)

    #TEST: NORMALIZE TO ZERO MEAN AND STD OF ONE
    print positions.shape
    pmean = np.mean(positions, axis=0)
    pstd = np.std(positions, axis=0)
    pstd[0, :] = 1
    new_positions = (positions-pmean)/pstd
    positions = new_positions
    print positions.shape
    #Show readed motion
    new_plot(positions, np.arange(0), 'Motion read from file')
    k =5
    w, c, Xmean, idxes = my_l1_pca(positions, k)
    #Show reconstructed motion to verify decomposition process
    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    ori_rec= rec

    rec = rec*pstd + pmean

    new_plot(rec, idxes, 'reconstructed motion, after l1_pca')
    print np.allclose(rec, positions)

    #Concatenate motions, then swap components
    print c.shape # k *22*3 => k *
    print w.shape # 300 * k

    s_id = 0
    swap_w = np.zeros_like(w)
    swap_w[0:150, s_id] = w[150:300, s_id]
    swap_w[150:300, s_id] = w[0:150, s_id]
    w[:] = swap_w

    '''
    swap_c = np.zeros_like(c)
    swap_c[:,0:11, :] = c[:, 11:22, :]
    swap_c[:,11:22, :] = c[:, 0:11, :]
    c = swap_c
    '''

    rec = np.tensordot(w, c, (1, 0)) + Xmean[np.newaxis]
    rec = rec*pstd + pmean
    new_plot(rec, idxes, 'reconstructed motion, after swap_w')

#    input('swaping')

    #Boost some components in a motion: = Original motion + some extra SIN activating components
    acts = [6]
    A = ori_rec
    frame = A.shape[0]
    for i in xrange(frame):
        alpha = np.sin(i/30. * np.pi*2)
        alpha = 3*np.abs(alpha)        
        for j in acts:
            A[i, :, :] = A[i, :, :] + alpha * c[j, :, :]
    rec = A
    rec = rec*pstd + pmean
    A = rec
    new_plot(A, idxes, 'Activate some components with sin funcs ' )

#    '''
    #DEBUGGING show ONLY some components [sta to end].
    for i in xrange(k-1):
        my_c = np.zeros_like(c)
        sta = i
        end = i+1
        my_c[sta:end, :, :] = c[sta:end, :, :]
        rec = np.tensordot(w, my_c, (1, 0)) + Xmean[np.newaxis]
        a = np.arange(1)
        a[0] = idxes[i]
        rec = rec*pstd + pmean
        new_plot(rec, a, 'Show some components [sta to end]' + str(i))
#    '''

    #Double some components
    new_c = np.zeros_like(c)
    new_c[:, :, :] = 3*c[:, :, :] 
    new_c[5:6, :, :] = c[5:6, :, :] 
    rec = np.tensordot(w, new_c, (1, 0)) + Xmean[np.newaxis]
    rec = rec*pstd + pmean
    new_plot(rec, idxes, 'Double C5 C6')

    #Double some weight
    new_w = np.zeros_like(w)
    new_w = 3*w
    rec = np.tensordot(new_w, c, (1, 0)) + Xmean[np.newaxis]
    new_plot(rec, idxes, 'Triple w')
    '''
    #Visualize weights
    for i in xrange(k):
        wi = w[:, i]
        plt.plot(wi)
        plt.ylabel('w' + str(i))
        plt.show()
    '''
#TESTING
#read_cmu_style()
#read_sub_cmu_style()
if __name__ == "__main__":
#    my_test_no_opt()
    my_mix_c()
#    my_test_ab()


#    my_test()
#    input('end of my_test()')

#    my_transfer()

#    my_transfer_2()

