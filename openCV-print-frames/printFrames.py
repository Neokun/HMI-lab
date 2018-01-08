# -*- coding: utf-8 -*-
## -> display purposes
### -> to look good

import cv2
import numpy as np
import time
import sys

frameFile = open('savedFrame.txt', 'r')
cap = []            # list to store frames
idKFrames = []      # list of index of key frame which is stored in savedFrame.txt

def readvid(namevid):
    #objective:get frames from video
    #input:path of video
    #output:frames of video,frames per second,width of video,height of video
    global cap

    vid = cv2.VideoCapture(namevid)
    num_f = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))  # Number of frames
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    t = time.time()
    print ""
    sys.stdout.write("reading video ... 0%")
    for i in range(num_f):
        ret, frame = vid.read()
        if ret == False:
            break
        cap.append(frame)

        #show %
        if time.time() - t > 1:
            a = np.ceil(i * 100 * 1.0 / (num_f - 1))
            b = ("reading video ... " + "%d" % a + "%")
            sys.stdout.write('\r' + b)
            t = time.time()

    sys.stdout.write('\r' + "reading video ... 100%")
    print ""
    time.sleep(1)
    vid.release()
    return fps, width, height

def readFramefile(filename):
    for line in frameFile:
        a = ""
        for i in range(len(line)):
            if line[i] == ",":
                break
            a = a + line[i]
        idKFrames.append(int(a))

    sys.stdout.write("List key frames: " + str(idKFrames))
    return 1

def chooseframe():
    #objective:GUI letting user choose frame
    #input:frames of video,frames per second of video, height of video
    #output:save frame chosen and rectangle chosen
    n = len(idKFrames)
    for i in range(n):
        img = cap[idKFrames[i]]
        sys.stdout.write('\n' + "save key frame: %d" %idKFrames[i])
        strname = str(idKFrames[i])
        addzero = 4 - len(strname)
        for j in range(addzero):
            strname = "0" + strname
        cv2.imwrite("key-frames/" + strname + ".jpg", img)

    sys.stdout.write("\nSave key frames successfully!!!\n")
    return 1

#run everything


def runMotionTrack(vidName, KFFName):   #KFFName stands for key frames file name.
    fps, width, height = readvid(vidName)
    readFramefile(KFFName)
    chooseframe()
    


runMotionTrack("input.mp4", "savedFrame.txt")
