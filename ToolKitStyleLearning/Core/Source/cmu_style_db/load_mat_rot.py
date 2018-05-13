import sys
import numpy as np
import scipy.io as io

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
def load_all(filename):
    datadata = io.loadmat(filename)
    clumsy = datadata['clumsy']
    elderly = datadata['elderly']
    neutral = datadata['neutral']
    return clumsy, elderly, neutral 
def load_a_mat(filename, key='clumsy'):
    datadata = io.loadmat(filename)
    sS = datadata[key]
    rot = sS[:, 3:96]
    tran = sS[:, 0:3]

    F = rot.shape[0]
    N = rot.shape[1]

    x = rot[:,0:N:3]
    y = rot[:,1:N:3]
    z = rot[:,2:N:3]

    print x.shape, y.shape, z.shape
    
    xyz = np.zeros(shape=(F,N/3,3)) 
    xyz[:,:,0] = x
    xyz[:,:,1] = y
    xyz[:,:,2] = z
    print rot.shape ,xyz.shape
#    xyz = rot.reshape(300, 31,3)
#    input('checking reshape rot')
       
    return tran, xyz, 

def save_a_mat(filename,tran, rot, key ='clumsy'):
    print rot.shape
    xyz = rot.reshape(rot.shape[0], rot.shape[1]*rot.shape[2])
    print xyz.shape
    out = np.concatenate((tran, xyz), axis=1)
    print xyz.shape, rot.shape, out.shape
    io.savemat(filename, {key:out})
    return out

def save_mat(filename, S, C, O):
	io.savemat(filename, {'S':S, 'C':C, 'O': O})

#-2. Read input from file
#sS, sC = load_mat_e_n('rotation_db.mat')
if __name__ == "__main__":
    datadata = io.loadmat('rotation_db.mat')
    sS = datadata['clumsy']

    tran, rot = load_a_mat('rotation_db.mat', 'clumsy')
    print tran.shape, rot.shape
    out = save_a_mat('test_output.mat',tran,rot)
    print sS.shape, out.shape
    print sS[0,0:10:1]
    print out[0,0:10:1]
    print np.allclose(sS, out)
