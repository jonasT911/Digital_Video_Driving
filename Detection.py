import cv2
import numpy as np
from math import sqrt
import os
from features import *

# Test: 
# (2) Largest Area  -- VERRY Effective (use additional masking with lanes ?)
# (3) Grab Centroid and Use Detect getRoadColor ... create mask
# (4) With a proper road mask we can dilate and try to use it to extract lanes as well

def getLane(img, colors = ["white", "yellow", "red"]):

    masks =[]
    for color in colors:
        masks.append(max_channel(cv2.threshold(color_mask(img, color),200,255, cv2.THRESH_TRUNC)[1]))

    #masks.append(max_channel(gabor_mask(img).astype(np.uint8))*255)  # try this on reduced image
    lane = np.zeros_like(masks[0])
    for mask in masks:
        print(mask)
        lane = cv2.bitwise_or(lane, mask)

    return cv2.threshold(lane, 150,255,cv2.THRESH_BINARY)[1]


def satRoadMask(img, color = [0,0,0], getMetrics = True, printImages = True):
    img = np.array(img, dtype=np.uint8)
    img = scale(img, .5)
    img = img[0:img.shape[0]-180,:,:]   # crop car
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row,col, channel = img.shape
    mid = col//2
    broadSatMask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1], 0, 30) 
    
    # combine with color mask and lane ?

    cut = sky_detector(img, thresh = .97)      # Remove sky , or cropping
    cut = row//2
    for i in range(0, cut): 
        gray[i,:] = 0
 
    road   = cv2.threshold(cv2.bitwise_and(gray, broadSatMask),100,255, cv2.THRESH_BINARY)[1]
    #lane  # Try to combine with lane

    try: # try largest contour
        contours, hierarchy = cv2.findContours(road, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contourSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    except:
        print("Contour Failed")
        roadArea = 0
    
    distx, xavg = margin(road, dim = 1)
    
    if (printImages):
        cv2.imwrite("DebuggingImages/Masks/test_"+str(i)+".png",road) 

    if getMetrics == True:
       
        cnt_disty, yavg = margin(road, dim = 0)
        ytop,ymid1,ymid2,ylow = cdf(cnt_disty,low=0.01,high =.51, mid = 0.25, mid2 =.5,bounds =False)
    
    else:
        return road, (xavg-mid)

