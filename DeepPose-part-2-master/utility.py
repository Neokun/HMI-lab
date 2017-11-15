# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:58:04 2017

@author: tungnb
"""
import sys
import time
import numpy as np
import cv2

def readVid(namevid):
    #objective:get frames from video
    #input:path of video
    #output:frames of video,frames per second,width of video,height of video
    vid = cv2.VideoCapture(namevid)
    num_f = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    obj = []
    t = time.time() ###
    print ""  ###
    sys.stdout.write("reading video ... 0%") ###
    for i in range(num_f):
        ret, frame = vid.read()
        if ret == False:
            break
        obj.append(frame)
        if time.time()-t>1: ###
            a=np.ceil(i*100*1.0/(num_f-1)) ###
            b = ("reading video ... " + "%d"%a+"%") ###
            sys.stdout.write('\r'+b) ###
            t=time.time() ###
    sys.stdout.write('\r'+"reading video ... 100%") ### 
    print "" ###
    time.sleep(1) ###
    vid.release()
    return obj, fps, width, height
#Read skeleton file
def readSkeleton(filename):
	ske = []
	f = open(filename, 'r')
	lines = f.readlines()    
	f.close()
	if len(lines) < 1:
		return ske
	for line in lines:
		one_ske = []
		for l in line.split(';'):
			x, y = [int(i) for i in l.split()]
			one_ske.append((x,y))
		#Maps Deepcut skeleton to hetpin's skeleton
		tmp = []
		tmp.append(one_ske[13])
		tmp.append(one_ske[12])
		tmp.append(one_ske[8])
		tmp.append(one_ske[9])
		tmp.append(one_ske[2])
		tmp.append(one_ske[3])
		tmp.append(one_ske[7])
		tmp.append(one_ske[6])
		tmp.append(one_ske[10])
		tmp.append(one_ske[11])
		tmp.append(one_ske[1])
		tmp.append(one_ske[0])
		tmp.append(one_ske[4])
		tmp.append(one_ske[5])
		ske.append(tmp)
	return ske

# Save skeleton to file
def saveSkeleton(ske, filename):
    outFile = open(filename,'w')
    for x in range(0, len(ske)):
        outFile.write(str(ske[x][11][0])+' '+str(ske[x][11][1])+';')
        outFile.write(str(ske[x][10][0])+' '+str(ske[x][10][1])+';')
        outFile.write(str(ske[x][4][0])+' '+str(ske[x][4][1])+';')
        outFile.write(str(ske[x][5][0])+' '+str(ske[x][5][1])+';')
        outFile.write(str(ske[x][12][0])+' '+str(ske[x][12][1])+';')
        outFile.write(str(ske[x][13][0])+' '+str(ske[x][13][1])+';')
        outFile.write(str(ske[x][7][0])+' '+str(ske[x][7][1])+';')
        outFile.write(str(ske[x][6][0])+' '+str(ske[x][6][1])+';')
        outFile.write(str(ske[x][2][0])+' '+str(ske[x][2][1])+';')
        outFile.write(str(ske[x][3][0])+' '+str(ske[x][3][1])+';')
        outFile.write(str(ske[x][8][0])+' '+str(ske[x][8][1])+';')
        outFile.write(str(ske[x][9][0])+' '+str(ske[x][9][1])+';')
        outFile.write(str(ske[x][1][0])+' '+str(ske[x][1][1])+';')
        outFile.write(str(ske[x][0][0])+' '+str(ske[x][0][1])+'\n')        
    outFile.close()   
    
#Read lookup table
def readLookupTable(filename):
	table = []
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	for line in lines:
		x, y = [int(i) for i in line.split(':')]
		#minus one, since python count from one for array index
		x = x -1
		y = y -1
		table.append((x,y))
	return table

def checkNearby(m, one_ske):# m, r, p stands for mouse, radius, point (joint)
	#print "Checking ", m
	r = 20
	k = 0 #for index of joint
	for p in one_ske:
		#print p , m
		if (abs(p[1]-m[1])<r) and (abs(p[0]-m[0])<r):
			print "Ok " , k
			return k
		k = k + 1
	return -1