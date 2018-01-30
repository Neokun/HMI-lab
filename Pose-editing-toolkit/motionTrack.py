# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 09:45:35 2017
 
@author: PhucMinh
"""
 
## -> display purposes
### -> to look good
 
import cv2
import numpy as np
import time
import sys
 
 
color = np.random.randint(0,255,(12,3)) ##
color[0]=[0,0,255] ##
color[1]=[0,255,0] ##
color[2]=[255,0,0] ##
color[3]=[0,255,255] ##
color[4]=[255,0,255] ##
color[5]=[255,255,0] ##
color[6]=[0,127,255] ##
color[7]=[0,255,127] ##
color[8]=[127,0,255] ##
color[9]=[255,0,127] ##
color[10]=[127,255,0] ##
color[11]=[255,127,0] ##
legendTEXT=[] ##
for i in range(1,13): ##
    legendTEXT.append("point %d"%i) ##
 
def readvid(namevid):
    #objective:get frames from video
    #input:path of video
    #output:frames of video,frames per second,width of video,height of video
    vid = cv2.VideoCapture(namevid)
    num_f = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) #Number of frames
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    obj = []
    t=time.time() ###
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
 
def indicekeyframe(obj,num_key):
    #objective:get indices of keyframes from frames
    #input:frames of video, number of keyframes desired
    #output:indices of keyframes
    def absdif(I,J):
        k = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        l = cv2.cvtColor(J, cv2.COLOR_BGR2GRAY)
        m = cv2.calcHist([k], [0], None, [256], [0, 256])
        n = cv2.calcHist([l], [0], None, [256], [0, 256])
        dif = np.abs(m - n)
        res = np.sum(dif)
        return res
     
    X=[]
    for k in range(len(obj)-1):
        K = obj[k] #read kth frame
        Kplus = obj[k+1] #read k+1th frame
        #Using Sum histogram diff as Ranking, others are: Minkowski, helliger
        #distances...
        sss = absdif(K,Kplus) #calculate sum of different histogram between K, Kplus
        X.append(sss); #Save to array X
        k=k+1
     
    I=np.argsort(X)+1
    J = I[len(I) - num_key : len(I)]
    J = sorted(J)
    return J
 
def chooseframe(obj,fps,J,height,legend):
    #objective:GUI letting user choose frame
    #input:frames of video,frames per second of video, indices of keyframes,height of video,bool true:show legend//false:don't show legend
    #output:frame chosen
    global legendTEXT ##
    def nothing(x):
        #not important
        pass
    def keyframeonright(i,J):
        #objective:get previous keyframe from current frame
        #input:indice of current frame, indices of keyframes
        #output:indice of keyframe
        n=len(J)
        if n==0:
            return 0
        j=0
        while j<n-1 and J[j]<=i:
            j=j+1
        if J[j]<=i:
            return i
        return J[j]
    def keyframeonleft(i,J):
        #objective:get next keyframe from current frame
        #input:indice of current frame, indices of keyframes
        #output:indice of keyframe
        n=len(J)
        if n==0:
            return 0
        j=0
        while j<n and J[j]<i:
            j=j+1
        if j!=0:
            j=j-1
        if J[j]<=i:
            return J[j]
        else:
            return i
             
    #initilization
    n=len(obj)
    img = np.copy(obj[0])
    cv2.namedWindow('Choose frame',cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Choose frame',20,20)
    cv2.resizeWindow('Choose frame', 600,600)
 
    cv2.createTrackbar('frame','Choose frame',0,len(obj)-1,nothing)
     
    font = cv2.FONT_HERSHEY_COMPLEX
    if legend==True:
        cv2.namedWindow('Legend',cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('Legend',620,20)
        cv2.resizeWindow('Legend',200,600)
        legendimg=np.ones((600,200,3),dtype=np.uint8)*255
        for i in range(12):
            cv2.circle(legendimg,(20,42+i*42),10,color[i].tolist(),-1)
            cv2.putText(legendimg,legendTEXT[i],(40,45+42*i), font, 0.45,(0,0,0),1,cv2.LINE_4)
     
     
        cv2.imshow('Legend',legendimg)
         
     
    cv2.imshow('Choose frame',img)
 
    i=0   
    #run process
    while(1):
        cv2.imshow('Choose frame',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        if k == 112:
            while i<n:
                k = cv2.waitKey(1000/fps) & 0xFF
                if k == 112:
                    break
                img=np.copy(obj[i])
                cv2.putText(img,"Press on 'p' to Play/Pause",(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
                cv2.imshow('Choose frame',img)
                i=i+1
                cv2.setTrackbarPos('frame','Choose frame',i)
        # get current positions of four trackbars
        i = cv2.getTrackbarPos('frame','Choose frame')
        if k == ord('g'):
            cv2.setTrackbarPos('frame','Choose frame',keyframeonleft(i,J))
        if k == ord('h'):
            cv2.setTrackbarPos('frame','Choose frame',keyframeonright(i,J))
        img=np.copy(obj[i])
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img,'Choose frame and press Enter',(5,25), font, 0.45,(255,255,255),1,cv2.LINE_4)
        cv2.putText(img,"Press on 'p' to Play/Pause",(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
        cv2.putText(img,"Press on 'g'/'h' to jump to keyframe",(5,height-7), font, 0.45,(255,255,255),1,cv2.LINE_4)
         
     
 
    cv2.destroyAllWindows()
    return i
     
drawing = False # true if mouse is pressed
ix,iy = -1,-1
def choosepoints(donttouchthisimg,badpoints,height,skippable):
    #objective:GUI letting user choose points on frame
    #input:frame,points automatically chosen,height of frame,bool true:user can decide to skip step//false:user has to proceed
    #output:position of chosen points
     
    #define drawing
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,color,legendTEXT ##
        if event == cv2.EVENT_LBUTTONDOWN and len(POS)<len(color):
            drawing = True
            ix,iy = x,y
            BACKUP.append(np.copy(img))
            cv2.imshow('Choose points',img)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True and len(POS)<len(color):
                imgcopy=np.copy(img)
                i=len(POS)
                cv2.circle(imgcopy,(x,y),2,color[i].tolist(),-1)
                cv2.imshow('Choose points',imgcopy)
 
        elif event == cv2.EVENT_LBUTTONUP:
            if len(POS)<=len(color) and len(POS)<len(color):
                drawing = False
                i=len(POS)
                cv2.circle(img,(x,y),2,color[i].tolist(),-1)
                cv2.imshow('Choose points',img)
                POS.append([x,y])
                 
                 
    #initialization        
    POS=[]            
    img = np.copy(donttouchthisimg)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img,'Draw points and press Enter',(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
    cv2.putText(img,"Press on 'u' to Undo",(5,25), font, 0.45,(255,255,255),1,cv2.LINE_4)
    if skippable ==True:
        cv2.putText(img,"Press on 's' to Skip",(5,height-7), font, 0.45,(255,255,255),1,cv2.LINE_4)
    BACKUP = [np.copy(img)]
    cv2.namedWindow('Choose points',cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Choose points',20,20)
    cv2.resizeWindow('Choose points', 600,600)
    cv2.setMouseCallback('Choose points',draw_circle)
     
    cv2.namedWindow('Legend',cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Legend',620,20)
    cv2.resizeWindow('Legend',200,600)
    legendimg=np.ones((600,200,3),dtype=np.uint8)*255
    for i in range(12):
        cv2.circle(legendimg,(20,42+i*42),10,color[i].tolist(),-1)
        cv2.putText(legendimg,legendTEXT[i],(40,45+42*i), font, 0.45,(0,0,0),1,cv2.LINE_4)
     
     
    cv2.imshow('Legend',legendimg)
    cv2.imshow('Choose points',img)
     
    #run process
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('u') and len(BACKUP)>0:
            img = BACKUP[-1]
            if len(POS)>0:
                POS.pop()
            BACKUP.pop()
            cv2.imshow('Choose points',img)
        elif k == 13 and POS!=[]:
            break
        elif k == ord('s') and skippable==True:
            POS=[]
            for j,new in enumerate(badpoints):
                pos = new.ravel()
                POS.append(pos)
            cv2.destroyAllWindows()
            return POS
    cv2.destroyAllWindows()
    return POS
 
 
 
 
def trackpointsfull(POS,obj,width,height,fps):
    #objective:GUI letting user to track points in video
    #input:points in first frame, frames of video, width of video, height of video, frames per second of video
    #ouput:position of points in video
     
    global realn ##
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
     
    realn=len(obj)
    framepos=0
    trackedframes=[]
    POSITION=[]
     
    print ""
    sys.stdout.write("tracking points ... 0%") ###
 
    def trackpoints(POSI,partobj,width,height,fps,obj,framepos,trackedframes,POSITION):
        #objective: let user to interact with GUI
        global color,legendTEXT ##
         
        realn=len(obj)
        n=len(partobj)
         
        def nothing(x):
            pass
        cv2.namedWindow('Tracking',cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('Tracking',350,20)
        cv2.resizeWindow('Tracking', 600,600)
        cv2.createTrackbar('frame','Tracking',0,realn-1,nothing)
        cv2.setTrackbarPos('frame','Tracking',framepos)
        cv2.createTrackbar('error%','Tracking',0,100,nothing)
        cv2.setTrackbarPos('error%','Tracking',10)
         
        cv2.namedWindow('Legend',cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow('Legend',950,20)
        cv2.resizeWindow('Legend',200,600)
        legendimg=np.ones((600,200,3),dtype=np.uint8)*255
        font=cv2.FONT_HERSHEY_COMPLEX
        for i in range(12):
            cv2.circle(legendimg,(20,42+i*42),10,color[i].tolist(),-1)
            cv2.putText(legendimg,legendTEXT[i],(40,45+42*i), font, 0.45,(0,0,0),1,cv2.LINE_4)
     
     
        cv2.imshow('Legend',legendimg)
         
         
         
        old_frame = partobj[0]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0=np.array(POSI,dtype=np.float32)
        oframe = np.copy(old_frame)
        POSITION.append(p0)
         
         
        for j in range(len(p0)):
            a,b = p0[j]
            oframe = cv2.circle(oframe,(a,b),2,color[j].tolist(),-1)
        trackedframes.append(oframe)
        showoframe=np.copy(oframe)
        cv2.putText(showoframe,"Press on 'p' to Resume/Pause tracking",(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
        cv2.putText(showoframe,"Press on 'Esc' to leave",(5,height-7), font, 0.45,(255,255,255),1,cv2.LINE_4)
        cv2.imshow('Tracking',showoframe)
        exitbool = False
        while(1):
            resumebutton = cv2.waitKey(1) & 0xff
            if resumebutton == ord('p'):
                break
            if resumebutton == 27:
                exitbool = True
                break
        t=time.time()
        for i in range(n-1):
            frame = np.copy(partobj[i+1])
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            stbool=True
            errbool=True
            for j in range(len(st)):
                if st[j]!=1 or err[j]>min(height,width)*cv2.getTrackbarPos('error%','Tracking')/100:
                    if st[j]!=1:
                        stbool=False
                        print ""
                        print "tracking lost"
                    if err[j]>min(height,width)*cv2.getTrackbarPos('error%','Tracking')/100:
                        errbool=False
                        print ""
                        print "tracking not accurate"
            if stbool==True and errbool==True:
                framepos=framepos+1
                cv2.setTrackbarPos('frame','Tracking',framepos)
                for j,new in enumerate(p1):
                    a,b = new.ravel()
                    frame = cv2.circle(frame,(a,b),2,color[j].tolist(),-1)
                trackedframes.append(frame)
                showframe=np.copy(frame)
                cv2.putText(showframe,"Press on 'p' to Resume/Pause tracking",(5,12), font, 0.45,(255,255,255),1,cv2.LINE_4)
                cv2.putText(showframe,"Press on 'Esc' to leave",(5,height-7), font, 0.45,(255,255,255),1,cv2.LINE_4)
                if exitbool==False:
                    cv2.putText(showframe,"Press on 'r' to Reset tracking",(5,25), font, 0.45,(255,255,255),1,cv2.LINE_4)
                POSITION.append(p1)
                cv2.imshow('Tracking',showframe)
                if i==(n-2):
                    while(1):
                        cv2.setTrackbarPos('frame','Tracking',framepos)
                        leavebutton=cv2.waitKey(1) & 0xff
                        if leavebutton == ord('r'):
                            J=indicekeyframe(trackedframes,20)
                            posframe=chooseframe(trackedframes, fps,J,height,True)
                            partobj=np.copy(obj)
                            POSI=choosepoints(partobj[posframe],np.array([[0,0]]),height,False)
                            trackedframes=trackedframes[0:posframe]
                            POSITION=POSITION[0:posframe]
                            framepos=posframe
                            return trackpoints(POSI,partobj[posframe:realn],width,height,fps,obj,framepos,trackedframes,POSITION)
                        elif leavebutton == 27:
                            exitbool = True
                            break
                k = cv2.waitKey(1000/fps) & 0xff
                if k == ord('p'):
                    while(1):
                        cv2.setTrackbarPos('frame','Tracking',framepos)
                        unpause=cv2.waitKey(1) & 0xff
                        if unpause == ord('p'):
                            break
                        elif unpause == ord('r'):
                            J=indicekeyframe(trackedframes,20)
                            posframe=chooseframe(trackedframes, fps,J,height,True)
                            partobj=np.copy(obj)
                            POSI=choosepoints(partobj[posframe],np.array([[0,0]]),height,False)
                            trackedframes=trackedframes[0:posframe]
                            POSITION=POSITION[0:posframe]
                            framepos=posframe
                            return trackpoints(POSI,partobj[posframe:realn],width,height,fps,obj,framepos,trackedframes,POSITION)
                        elif unpause == 27:
                            exitbool = True
                            break
                if k == 27 or exitbool == True:
                    break
                if k == ord('r'):
                    cv2.destroyAllWindows()
                    J=indicekeyframe(trackedframes,20)
                    posframe=chooseframe(trackedframes, fps,J,height,True)
                    partobj=np.copy(obj)
                    POSI=choosepoints(partobj[posframe],np.array([[0,0]]),height,False)
                    trackedframes=trackedframes[0:posframe]
                    POSITION=POSITION[0:posframe]
                    framepos=posframe
                    return trackpoints(POSI,partobj[posframe:realn],width,height,fps,obj,framepos,trackedframes,POSITION)
                # Now update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = p1.reshape(-1,1,2)
                if time.time()-t>1: ###
                    aa=np.ceil(framepos*100*1.0/(realn-1)) ###
                    bb = ("tracking points ... " + "%d"%aa+"%") ###
                    sys.stdout.write('\r'+bb) ###
                    t=time.time() ###
                 
            else:
                cv2.destroyAllWindows()
                POSI=choosepoints(partobj[i+1],p0,height,True)
                return trackpoints(POSI,partobj[i+1:n],width,height,fps,obj,framepos,trackedframes,POSITION)
 
        sys.stdout.write('\r'+"traking points ... 100%") ###
        print"" ###
        time.sleep(1) ###
         
        cv2.destroyAllWindows()
        return POSITION
         
    return trackpoints(POS,obj,width,height,fps,obj,framepos,trackedframes,POSITION)
         
 
 
 
 
def writevidtrack(obj,points,fps,width,height):
    #objective:write video with marked points
    #input:frames of video, position of points in video, frames per second of video, width of video, height of video
    #output:video
    global color
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    obj_sum = cv2.VideoWriter('dancewithtracking.avi', fourcc, fps, (width, height))
    t=time.time()
    print ""
    sys.stdout.write("writing tracked video ... 0%")
    for i in range(len(points)):
        frame=np.copy(obj[i])
        for j,new in enumerate(points[i]):
            a,b = new.ravel()
            frame = cv2.circle(frame,(a,b),2,color[j].tolist(),-1)
        obj_sum.write(frame)
        if time.time()-t>1:
            aa=np.ceil(i*100*1.0/(len(points)-1))
            bb = ("writing video ... " + "%d"%aa+"%")
            sys.stdout.write('\r'+bb)
            t=time.time()
    sys.stdout.write('\r'+"writing tracked video ... 100%") 
    print ""
    time.sleep(1)
    obj_sum.release()
 
 
def writevidskeleton(points,fps,width,height):
    #objective:write video of skeleton
    #input:frames of video, position of points in video, frames per second of video, width of video, height of video
    #output:video
    global color
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    obj_sum = cv2.VideoWriter('skeleton.avi', fourcc, fps, (width, height))
    t=time.time()
    print ""
    sys.stdout.write("writing skeleton video ... 0%")
    for i in range(len(points)):
        frame=np.ones((height,width,3),dtype=np.uint8)*255
        for j,new in enumerate(points[i]):
            a,b = new.ravel()
            frame = cv2.circle(frame,(a,b),2,color[j].tolist(),-1)
        obj_sum.write(frame)
        if time.time()-t>1:
            aa=np.ceil(i*100*1.0/(len(points)-1))
            bb = ("writing video ... " + "%d"%aa+"%")
            sys.stdout.write('\r'+bb)
            t=time.time()
    sys.stdout.write('\r'+"writing skeleton video ... 100%") 
    print ""
    print ""
    time.sleep(1)
    print "Done"
    time.sleep(1)
    obj_sum.release()  
     
     
 
#run everything
def runMotionTrack(filename):
    obj,fps,width,height=readvid(filename)
    n=len(obj)
    if n>0:
        J=indicekeyframe(obj,20)
        i=chooseframe(obj,fps,J,height,False)
        POS=choosepoints(obj[i],np.array([[0,0]]),height,False)
        points=trackpointsfull(POS, obj[i:n],width,height,fps)
        writevidtrack(obj[i:n],points,fps,width,height)
        writevidskeleton(points,fps,width,height)