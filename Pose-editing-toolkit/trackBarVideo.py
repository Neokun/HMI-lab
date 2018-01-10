# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
from utility import readLookupTable, readSkeleton
#------------------GolbalVariable----
resizedFrame = []
ske = []
table = []
img = None
pos = 0
cap = 0

#Draw skeleton
def draw_skeleton(inputframe, one_ske, table):
	draw_frame = inputframe.copy()
	i = 0	
	# compute the middle of the hip
	centerHipY = (one_ske[4][0]+one_ske[5][0])/2
	centerHipX = (one_ske[4][1]+one_ske[5][1])/2
	centerHip = (centerHipY, centerHipX)
	cv2.line(draw_frame, one_ske[1], centerHip, blue, thickness)

	for xy in table:
		cv2.line(draw_frame, one_ske[xy[0]], one_ske[xy[1]], colors[i], thickness)        
		#draw joints
		cv2.circle(draw_frame, one_ske[xy[0]], joint_r, green, 2)
		cv2.circle(draw_frame, one_ske[xy[1]], joint_r, green, 2)
		i = i + 1
	cv2.imshow('mywindow',draw_frame)	
	return draw_frame

def onChange(trackbarValue):
    global pos, cap, img
    pos = trackbarValue
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
    err,img = cap.read()    
    try:
        draw_skeleton(img, ske[trackbarValue], table)
    except Exception, e:
        print e
    pass

def checkNearby(m, one_ske):# m, r, p stands for mouse, radius, point (joint)
	#print "Checking ", m
	r = 10
	k = 0 #for index of joint
	for p in one_ske:
		#print p , m
		if (abs(p[1]-m[1])<r) and (abs(p[0]-m[0])<r):
			print "Ok " , k
			return k
		k = k + 1
	return -1
#Drag function
moving = False
def click_and_drag(event, x, y, flags, param):
	global moving, pos, img
	if event == cv2.EVENT_LBUTTONDOWN:
		#Check click nearby joint with radius of r pixcels
		if checkNearby((x,y), ske[pos]) >= 0:
			moving = True
		else:
			moving = False
	if event == cv2.EVENT_MOUSEMOVE:
		if moving:
			#update moving joint
			k = checkNearby((x,y), ske[pos])
			if k >= 0:
				ske[pos][k] = (x,y)
			else:
				moving = False			
			#draw
			draw_skeleton(img, ske[pos], table)
	if event == cv2.EVENT_LBUTTONUP:
		moving = False
        
#Gen 14 different colors for 14 edges
def gen_colors():
	delta = 256/4;
	colors = []
	for i in range(0,4):
		for j in range(0,4):
			for k in range(0,4):
				colors.append((delta*i, delta*j, delta *k))
	return colors

#------------------Skeleton-style---------------
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
thickness = 10
joint_r = 5
colors = gen_colors()
#------------------EndOfSkeletonStyle-----------


def run(filename):
    global table, ske, img, cap
    table = readLookupTable("lookup.skeleton")
    ske = readSkeleton(filename.split('.')[0] + ".skeleton")# Read the stored skeleton
    
    cap = cv2.VideoCapture(filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cv2.namedWindow('mywindow')
    cv2.setMouseCallback('mywindow',click_and_drag)
    cv2.createTrackbar( 'start', 'mywindow', 0, length, onChange )
    
    onChange(0)
    cv2.waitKey()
    
    start = cv2.getTrackbarPos('start','mywindow')
    
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    #while cap.isOpened():
    #    err,img = cap.read()    
    #    cv2.imshow("mywindow", img)
    #    k = cv2.waitKey(10) & 0xff
    #    if k==27:
    #        break
    cap.release()
    cv2.destroyAllWindows()
    return ske