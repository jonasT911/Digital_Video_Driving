import cv2
import numpy as np
from math import sqrt
import os
from features import *

# KMEANS To Define A region of Interest
'''def laneMethod1(img):
    road=img 
    gab = gabor_mask(road)
    blue = max_channel(cv2.threshold(color_mask(road, 'blue'), 200,255, cv2.THRESH_TRUNC)[1])  # OR cv2.THRESH_BINARY
    white  = max_channel(cv2.threshold(color_mask(road, 'white'),200,255, cv2.THRESH_TRUNC)[1])
    red = 0
    adds = white +gab + red
    shoes = np.zeros([20,col])
    hat = np.zeros([0, col]) # cut
    block = np.concatenate((hat,adds), axis=0)
    final = np.concatenate((block,shoes), axis=0)'''

# Needs Work
def getLaneMask(img, colors = ["white", "yellow", "red"]):
    masks =[]
    for color in colors:
        masks.append(max_channel(cv2.threshold(color_mask(img, color),200,255, cv2.THRESH_TRUNC)[1]))
    lane = np.zeros_like(masks[0])
    for mask in masks:
        lane = lane+mask
    cv2.imshow("here",cv2.threshold(lane, 150,255,cv2.THRESH_BINARY)[1])
    cv2.waitKey(0)
    return cv2.threshold(lane, 150,255,cv2.THRESH_BINARY)[1]

# Don't use
def satRoadMask(img, gray, getMetrics = True, printImages = True):
    broadSatMask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1], 0, 30) 
    #lowSatMask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1], 0, 15)
    road  = cv2.threshold(cv2.bitwise_and(gray, broadSatMask),50,255, cv2.THRESH_BINARY)[1]
    #lane  # Try to combine with lane
    try: # try largest contour
        contours, hierarchy = cv2.findContours(road, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contourSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = contourSorted[-1]
        cv2.drawContours(road, [cnt], 0, (0,255,0), 3)
        cv2.drawContours(gray, [cnt], 0, (255,255,255), -1)
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


def region_of_interest(img, pts):
    mask = np.zeros_like(img)
    trapezoid_cnt = np.array( pts )
    cv2.drawContours(mask, [trapezoid_cnt], 0, (255,255,255), -1)
    masked_image = cv2.bitwise_and(img, mask)
    # cv2.imshow("test",mask)
    return max_channel(masked_image) # send max_channel

def shadowHsvMask(hsv):
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 0, 0])
    upper = np.array([110, 40, 255])
    shadow = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(hsv, hsv, mask=shadow)

def roadHsvMask(hsv):
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    lower = np.array([85, 10, 90])
    upper = np.array([100, 30, 255])
    roadhsv = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(hsv, hsv, mask=roadhsv)\

def gravelMask(hsv):
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 10, 90])
    upper = np.array([60, 30, 255])
    roadhsv = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(hsv, hsv, mask=roadhsv)

def roadLabMask(lab):
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)
    # lower = np.array([10, 122, 120])
    # upper = np.array([255, 128, 132])
    lower = np.array([5, 122, 120])
    upper = np.array([255, 128, 130])
    roadlab = cv2.inRange(lab, lower, upper)
    return cv2.bitwise_and(lab, lab, mask=roadlab)

def treeLaneMask():
    pass

## VERY Hand-Tuned But should Capture Something
    ## Peripheral Slices: touch them in the middle ?? 
def crop(img,view = "first", peripherals = True ):
    row, col, channel = img.shape
    if view == 'first':
        vertices = np.array([[2*col//11,5*row//7], [3*col//8 , row//2], [5*col//8,row//2], [9*col//11,5*row//7]])
        full = region_of_interest(img, vertices)
        if peripherals == True:
            vertices = np.array([[0,row], [0,4*row//7], [3*col//8 , row//2], [5*col//8,row//2], [col,4*row//7], [col,row]])
            full = region_of_interest(img, vertices)
        return full
    elif view == 'third':

        ##Forward Slice, make future more "restrictive" -- esepcially for tree map
        vertices = np.array([[col//4,row//2], [7*col//16 , 2*row//7], [9*col//16,2*row//7], [3*col//4,row//2]])
        front = region_of_interest(img, vertices)

        if peripherals == True:
            vertices = np.array([[0 , 5*row//7], [col//4,row//2], [col//3,row], [0,row]]) # tune row//2 or row//3 to be more/less inclusive
            left = region_of_interest(img, vertices)    

            vertices = np.array([[col,5*row//7], [3*col//4,row//2], [2*col//3,row], [col,row]])
            right = region_of_interest(img, vertices)

            return cv2.bitwise_or(cv2.bitwise_or(left, right), front)
    
        else:
            return front
    else:
        return 


def getRoad(img, playerView = "first", sides = False, dispTarget = True):

        img = scale(img, .2)    # print(img.shape)
        row,col,channel= img.shape
        mid = col//2
        lab = max_channel(roadLabMask(img)) 
        #lab = max_channel(roadHsvMask(img)) 
        region = crop(img, view = playerView, peripherals = sides)
        cropped = (cv2.bitwise_and(region, lab)>0)*1
        tgt, ey = getCenter(cropped)
        error = tgt - mid
        if dispTarget == True:
            cv2.circle(img,(tgt,ey),radius = 4, color=(0,255,0), thickness=-1)
            cv2.circle(img,(mid,ey),radius = 4, color=(0,0,255), thickness=-1)
            cv2.line(img, (mid,row), (mid,ey),color=(0,0,255), thickness=2)
            cv2.line(img, (mid,row),(tgt,ey),color=(0,255,0), thickness=2)
            return img, cropped, error
        return  error
        



def calcError(mask,mid):
    return getCenter(mask)[0] - mid



if __name__ == '__main__':

    folder = 'more-test-imgs/'
    direct = os.listdir(folder)
    print(direct[0])
    for file in direct:
            
        #img = cv2.imread(folder+file)
        img =cv2.imread('all-road-tune-filter.jpg')
        img = scale(img, .5)    # print(img.shape)
        row,col,channel= img.shape
        mid = col//2
        x,y,z =getRoad(img)