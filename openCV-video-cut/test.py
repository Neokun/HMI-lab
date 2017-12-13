import cv2
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_rectangle(event,x,y,flags,param):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            draw_frame = img.copy()
            cv2.rectangle(draw_frame,(ix,iy),(x,y),(0,255,0))
            cv2.imshow('image', draw_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_rectangle)
 
cv2.imshow('image', img)
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()