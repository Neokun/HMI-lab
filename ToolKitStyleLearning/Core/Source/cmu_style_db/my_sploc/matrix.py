import numpy as np
import random as rd
def generate_matrix(row,col): # generate matrix which specific column and row
    X  = []
    
    R  = [] # for row
    C  = [] # for col
    TR = []
    TC = []
    E  = [] 
    for k in xrange(row):
        for j in xrange(col):
            E.append(rd.random())
            C.append(E)
            E = []
        #R.append(C)
        X.append(C)
        C = []
    X = np.asarray(X)
    return X

X = generate_matrix(3,4)
#print('X',X[0])
#print('X size', X.size)
#print('X shape', X.shape)

    
    
    
