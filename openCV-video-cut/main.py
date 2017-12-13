# -*- coding: utf-8 -*- 
## -> display purposes
### -> to look good
 
import cv2
import numpy as np
import time
import sys

frameFile = open('savedFrame.txt', 'w') 
cap = [] # list to store frames
drawing = False # true if mouse is pressed
img = None #current frame
x0,y0 = -1,-1 #current top-left of the rectangle
x1,y1 = -1,-1 #current bot-right of the rectangle


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
    #objectice: function to draw a reactangle by mouse
    #input: mouse event, mouse position
    #output: frame was drawed
    global x0, y0, x1, y1, drawing, cap, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x0,y0 = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            i = cv2.getTrackbarPos('frame','Choose frame')
            img = cap[i].copy()
            cv2.rectangle(img,(x0,y0),(x,y),(0,255,0))
            x1,y1 = x,y
       
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def chooseframe(fps,height):
    #objective:GUI letting user choose frame
    #input:frames of video,frames per second of video, height of video
    #output:save frame chosen and rectangle chosen
    global cap, img, x0, y0, x1, y1

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
            frameFile.write('{},{},{},{},{}\n'.format(i,x0,y0,x1,y1))
            print('saved frame {}, rectangle:{},{},{},{}'.format(i,x0,y0,x1,y1))

        cv2.putText(img,"Press 's' to save frame and rectangle",(5,25), font, 0.45,(255,255,255),1,cv2.LINE_4)
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