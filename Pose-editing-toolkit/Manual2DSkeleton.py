import numpy as np
import cv2
#import cv
from random import randint
import random
import sys

#Read skeleton file
def readSkeleton(filename):
	ske = []
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
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

#Draw skeleton
def draw_skeleton(inputframe, one_ske, table):
	draw_frame = inputframe.copy()
	i = 0
	for xy in table:
		cv2.line(draw_frame, one_ske[xy[0]], one_ske[xy[1]], colors[i], thickness)
		#draw joints
		cv2.circle(draw_frame, one_ske[xy[0]], joint_r, green, 2)
		cv2.circle(draw_frame, one_ske[xy[1]], joint_r, green, 2)
		i = i + 1
	cv2.imshow('frame',draw_frame)	
	return draw_frame

#Normalize input to 400*400 - Just to reduce to complexity
def resizewh(input):
	r = 400.0 / frame.shape[1]
	dim = (400, int(frame.shape[0] * r))
	return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
#Gen 14 different colors for 14 edges
def gen_colors():
	delta = 256/4;
	colors = []
	for i in range(0,4):
		for j in range(0,4):
			for k in range(0,4):
				colors.append((delta*i, delta*j, delta *k))
	return colors

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

#Drag function
moving = False
def click_and_drag(event, x, y, flags, param):
	global moving
	if event == cv2.EVENT_LBUTTONDOWN:
		#Check click nearby joint with radius of r pixcels
		if checkNearby((x,y), ske[i]) >= 0:
			moving = True
		else:
			moving = False
	if event == cv2.EVENT_MOUSEMOVE:
		if moving:
			#update moving joint
			k = checkNearby((x,y), ske[i])
			if k >= 0:
				ske[i][k] = (x,y)
			else:
				moving = False
			#draw
			draw_skeleton(frame, ske[i], table)
	if event == cv2.EVENT_LBUTTONUP:
		moving = False
#Read 2D skeleton
ske = readSkeleton("muagiday.skeleton")
#Define skeleton lookup table
table = readLookupTable("lookup.skeleton")
#Read video file
cap = cv2.VideoCapture('muagiday.mov')
if( not cap.isOpened() ):
	print('Input video not found, please check your input file')
	sys.exit()
ret, frame = cap.read()
frame = resizewh(frame)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
thickness = 10
joint_r = 5
colors = gen_colors()
#Init video writer
height , width , layers =  frame.shape
# Define the codec and create VideoWriter object
#fourcc = cv2.FOURCC('M', 'P', 'E', 'G')
#out = cv2.VideoWriter('SemiFtuSkeleton.avi', fourcc, 2, (width, height), True)
#MAIN LOOP
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',click_and_drag)

i = 0;
draw_skeleton(frame, ske[i], table)

while(cap.isOpened()):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if cv2.waitKey(1) & 0xFF == ord('n'):
		i = i + 1
		ret, frame = cap.read()
		frame = resizewh(frame)
		drawn = draw_skeleton(frame, ske[i], table)
#		out.write(drawn)
	#Temporary: Since limited number of extracted skeleton
	if i == 300:
		break
cap.release()
#out.release()
cv2.destroyAllWindows()







