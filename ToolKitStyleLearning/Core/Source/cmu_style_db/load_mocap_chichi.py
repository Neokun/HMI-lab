import os
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

sys.path.append('motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
walk_id = 0
run_id = 0

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)
    
def plotframe(clips):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(elev=10., azim=100)
    print (clips.shape)
    clips = clips[::10]
    fr_no = clips.shape[0]
    for i in range(200, fr_no):
        if i%10 == 0:
            print (i)
        #print (clips[i, :, :])
        fr = clips[i,:,:]
        #ax.scatter(fr[:,0], fr[:,1], fr[:,2], c='b', marker='-')
        ax.plot(fr[:,0], fr[:,1], fr[:,2], '-o')
        plt.pause(.0001)
    plt.show()

def process_file(filename, window, window_step, c_start = 0, c_end = -1):
    anim, names, frametime = BVH.load(filename)

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
    print (global_positions.shape)
    print (global_rotations.shape)

    """ Remove Uneeded Joints """
    positionsOri = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    positions = global_positions[:,np.array(np.arange(31))]
    """ Put on Floor """
    print (positions.shape)
    '''
    print (rotations.shape)
    fid_l, fid_r = np.array([4,5]), np.array([8,9])
    foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    
    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    #???
    trajectory_filterwidth = 3
    reference = positions[:,0] * np.array([1,0,1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    print(positions.shape)    
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
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
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
#    plotframe(positions)
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
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    print (positions.shape)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    print (positions.shape)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    print (positions.shape)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
    '''
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    print (positions.shape)
    """ Slide over windows """
    windows = []
    windows_classes = []
    out_dir_train = 'chichi_train_out/'
    out_dir_test = 'chichi_test_out/'
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
        c = filename[12]
        if c == '1':
            global walk_id
            np.savetxt(out_dir_train + c + '('+ str(walk_id)+').txt', slice, delimiter=',')
            walk_id = walk_id + 1;
        if c == '2':
            global run_id
            np.savetxt(out_dir_train + c + '('+ str(run_id)+').txt', slice, delimiter=',')
            run_id = run_id + 1;
        
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

#Read motion from file, return: Train, validation, test sets
def read_chichi():
    cmu_files = get_files('chichi_rain')
    cmu_clips = []
    for i, item in enumerate(cmu_files):
        print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
        clips, _ = process_file(item, 93, 48)        
        cmu_clips += clips

#TESTING
#read_cmu_style()
#read_sub_cmu_style()
read_chichi()

