# -*- coding: utf-8 -*- 
## -> display purposes
### -> to look good
 
import cv2
import numpy as np
import time
import sys

keyFrameFile = open('savedKeyFrame.txt', 'w') 
cap = [] # list to store frames
drawing = False # true if mouse is pressed
img = None
ix,iy = -1,-1


def readvid(namevid):
    #objective:get frames from video
    #input:path of video
    #output:frames of video,frames per second,width of video,height of video
    global cap

    vid = cv2.VideoCapture(namevid)
    num_f = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    t=time.time() ###
    print ""  ###
    sys.stdout.write("reading video ... 0%") ###
    for i in range(num_f):
        ret, frame = vid.read()
        if ret == False:
            break
        cap.append(frame)

        #show %
        if time.time()-t>1: ###
            a=np.ceil(i*100*1.0/(num_f-1)) ###
            b = ("reading video ... " + "%d"%a+"%") ###
            sys.stdout.write('\r'+b) ###
            t=time.time() ###
            
    sys.stdout.write('\r'+"reading video ... 100%") ### 
    print "" ###
    time.sleep(1) ###
    vid.release()
    return fps, width, height

def draw_rectangle(event,x,y,flags,param):
    global ix,iy, drawing, cap, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            i = cv2.getTrackbarPos('frame','Choose frame')
            img = cap[i].copy()
            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0))
            # cv2.imshow('Choose frame',img)
       
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def chooseframe(fps,height):
    #objective:GUI letting user choose frame
    #input:frames of video,frames per second of video, indices of keyframes,height of video,bool true:show legend//false:don't show legend
    #output:frame chosen
    global cap, img

    def onChange(trackbarValue):
        global cap, img
        img = cap[trackbarValue].copy()
             
    #initilization
    n=len(cap)
    cv2.namedWindow('Choose frame',cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Choose frame',20,20)
    cv2.resizeWindow('Choose frame', 600,600)
    cv2.setMouseCallback('Choose frame', draw_rectangle)
 
    cv2.createTrackbar('frame','Choose frame',0,n-1,onChange)
    onChange(0)
     
    font = cv2.FONT_HERSHEY_COMPLEX

    #type of current keyFrame to save (start frame or end frame)
    typeFrame = "Start keyFrame"

    i=0   
    #run process
    while(1):
        cv2.imshow('Choose frame',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('p'):
            while i<n:
                k = cv2.waitKey(1000/fps) & 0xFF
                if k == ord('p'):
                    break
                img=np.copy(cap[i])
                cv2.putText(img,"Press on 'p' to Play/Pause",(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
                cv2.imshow('Choose frame',img)
                i=i+1
                cv2.setTrackbarPos('frame','Choose frame',i)

        # get current positions of four trackbars
        i = cv2.getTrackbarPos('frame','Choose frame')
        if k == ord('s'):
            if typeFrame == "Start keyFrame":
                typeFrame = "End keyFrame"
                keyFrameFile.write(str(i) + ' ')
            else:
                typeFrame = "Start keyFrame"
                keyFrameFile.write(str(i) + '\n')

        cv2.putText(img,"Press 's' to save " + typeFrame,(5,25), font, 0.45,(255,255,255),1,cv2.LINE_4)
        cv2.putText(img,"Press on 'p' to Play/Pause",(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
        cv2.putText(img,"Press Esc to exit",(5,height-7), font, 0.45,(255,255,255),1,cv2.LINE_4)
     
    cv2.destroyAllWindows()
    return i
 
#run everything
def runMotionTrack(filename):
    fps,width,height=readvid(filename)
    n=len(cap)
    if n>0:
        i=chooseframe(fps,height)

runMotionTrack("input.mp4")