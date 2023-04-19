import numpy as np
import cv2, os
import math, matplotlib.pyplot as plt

 


def create_gaborfilter(lambd = 10.0):     #  https://www.freedomvc.com/index.php/2021/10/16/gabor-filter-in-edge-detection/###
    # Produces a set of gabor filters
     
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges 
    lambd = lambd
    gamma = 0.5
    thetas = [0, 10, 20, 30, 40, 50, 60,70,80,90]  # flat, vertical and diagonals
    psi = 0  # Offset value - lower generates cleaner results

    for theta in thetas:  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)

    return filters, thetas


def margin(img, dim = 1, disp = False):   # dim 1 is columns, dim 0 is rows
    dist,x = [], []
    for j in range(0, img.shape[dim]):
        x.append(j)
        if dim == 1:
            dist.append(int(np.sum(img[:,j])))
        if dim == 0:
            dist.append(int(np.sum(img[j,:])))
    normed = dist/np.sum(dist)
    EX = sum([i*normed[i] for i in range(0,len(normed))])
    if disp == True:
        plotDist(normed)

    return normed, EX   # dist, int(EX)

def cdf(dist, low = .02, high = .98 , mid = .4, mid2 = .6, bounds= False):  # for y "low" is actually "higher" in the frame
    cdf, left, right, mid, summed, last = [], 0, 0,0,0, 0
    for i in range(0, len(dist)):
        summed = dist[i] + summed
        cdf.append(summed)
        if summed > mid and last <= mid:
            mid1 = i
        if summed >mid2 and last <= mid2:
            mid2 = i   
        if summed > low and last <= low:
            left = i
        if summed > high  and last <= high:
            right = i
        last = summed

    if bounds == True:
        return left, right
    else:
        return left, mid1, mid2, right

def max_channel(img):
    fin = np.zeros([img.shape[0], img.shape[1]])
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            fin[i,j] = max(img[i,j,:])
    return fin

def gabor_lane(img, lam = 10):
    filters, thetas = create_gaborfilter(lam)
    bank = orient_select(img,filters, thetas)
    #orientations = []  # flat, Right, Vertical, Left

    if len(bank.shape) > 2:
        bank = max_channel(bank) # drop channels

    return bank

def gabor_mask(img):

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    # hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)   # hsl and hsv basically look the same
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

    # Normal Frequencies
    gab_hsv = gabor_lane(hsv)
    #gab_hsl = gabor_lane(hsl)
    #gab_lab =  gabor_lane(lab)
    #gab_yuv = gabor_lane(yuv)    # Weak Indicator
    gab_img = gabor_lane(img)
#   gab_hue = gabor_lane(hue)
    # gab_sat = gabor_lane(sat)
    #gab_gray = gabor_lane(gray)  # Not so necessary

    gabors = [gab_hsv, gab_img, ]   # lower complexity --> 2 color spaces 
    #weights = [  1   ,   1     ,   1     ,   1   ]

    # votes = voter(gabors, weights) # not necessary
    #votes = cv2.bitwise_and(img, img, mask=votes)
    
    gabs = sum(gabors)
    #cv2.imshow("gab_mask", gabs)

    return gabs

def perspective_map(img):
    pts1 = np.float32([[0, 260], [640, 260],
                       [0, 400], [640, 400]])
    pts2 = np.float32([[0, 0], [400, 0],
                       [0, 640], [400, 640]])
     
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (500, 600))

    cv2.imshow(result)
    return result

def scale(img, a = .30):
    scale_percent = a # percent of original size
    width, height = int(img.shape[1] * scale_percent), int(img.shape[0] * scale_percent)
    dim = (width, height)  
    # resize image
    img1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)    
    return img1


def sky_detector(img, thresh =.95):  
    blue_mask = color_mask(img, color = "blue")
    dist, esky = margin(blue_mask, dim = 0)
    sum,i = 0, 0
    while sum < thresh:            
        sum = sum + dist[i]
        i +=1
    cut = i

    return cut


# supports green, yellow, white, green,  red and blue
def color_mask(img, color = 'red'):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    result = img.copy()


    if color == 'yellow':
        lower = np.array([22, 93, 0])
        upper = np.array([45, 255, 255])
        full_mask = cv2.inRange(hsv, lower, upper)
    
    elif color == 'red':
        # lower boundary RED color range values; Hue (0 - 10)
        lower1 = np.array([0, 100, 20])
        upper1 = np.array([10, 255, 255])
 
        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])

        lower_mask = cv2.inRange(hsv, lower1, upper1)
        upper_mask = cv2.inRange(hsv, lower2, upper2)

        full_red = lower_mask + upper_mask
        return cv2.bitwise_and(hsv, result, mask=full_red)
    
    elif color == 'green':  # look in low saturation range ?
        lower = np.array([30, 50, 100])
        upper = np.array([70, 255, 255])
        green = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_and(hsv, hsv, mask=green)

    elif color == 'bright green':  
        lower = np.array([40, 0, 0])
        upper = np.array([70, 255, 255])
        gr = cv2.inRange(hsv, lower, upper)
        return cv2.bitwise_and(hsv, hsv, mask=gr)
   
    elif color == 'blue':
        # Sky Detection during day, select for 100 to 130 hue at all saturation levels
        blue = cv2.inRange(hsv, np.array([100,0,100]) ,  np.array([130,255,255]))
        return cv2.bitwise_and(hsv, hsv, mask=blue)
    else:
        # default mode will be a white lane detector 
        # Try HLS mask which transforms white into broader range
        lower_white = np.array([0,150,0])
        upper_white = np.array([255,255,50])  # WHITE: Try turning down saturation but expanding white allowance for lane markings
        hls = cv2.cvtColor(result, cv2.COLOR_BGR2HLS)
        white = cv2.inRange(hls, lower_white, upper_white)

        return cv2.bitwise_and(hls, hls , mask=white)

    return None

def multidim_intersect(arr1, arr2):
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def norm(x):
    return x[0] + x[1]

def kmeans(bgr_img, K = 10, channels = 3):
    
    Z = bgr_img.reshape((-1,channels))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]  
    res2 = res.reshape((bgr_img.shape))

    return res2, label


def getCenter(Map):
    dist, ex = margin(Map, dim = 1)
    dist, ey = margin(Map, dim = 0)
    return (int(ex), int(ey))


def fillContour(Map, all = False):
    contours, hierarchy = cv2.findContours(Map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    new = np.zeros_like(Map)
    if all == True:
        for cnt in contours:
            new = new + cv2.drawContours(new, [cnt], 0, (255,255,255), -1)
    else:
        cnt = contourSorted[1]
        new = cv2.drawContours(Map, [cnt], 0, (0,255,0), 3)  # DEBUG
    return new


def darkChannel(img):
    dark = np.zeros([img.shape[0], img.shape[1]])
    red = img[:,:,2]
    blue = img[:,:,0]
    green = img[:,:,1]
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            r, b, g= red[i,j], blue[i,j], green[i,j]
            d = int(min(r, b, g))
            dark[i,j] = d 
    dark = np.array(dark, dtype = np.uint8)
    return dark

def plotDist(dist):
    x = [i for i in range(0,len(dist))]
    plt.plot(x,dist) #axvline/hline for showing stdev
    plt.title('Marignal')
    plt.show() 
    return 

def orient_select(img,filters, thetas):
    final = np.zeros_like(img)
    all = {} # stores each fitlered copy (and thresholded for strongest responses)
    for i in range(0, len(filters)):
        theta = thetas[i]
        kern = filters[i]
        img_filter = cv2.filter2D(img, -1, kern)# normalize each response
        img_filter = cv2.threshold(img_filter,200,255, cv2.THRESH_BINARY)[1]
        all[theta] = img_filter
    
    # ===== Adder block ======#
    for theta in thetas:  #[20,30,80,90]:  
        if theta == 0:  # always skip 0 but try other orientations
            continue 
        bit = all[theta]//10
        #edge = cv2.Canny(all[theta], 100, 250)
        #mask = cv2.bitwise_and(bit, bit, mask=edge)
        final = final + bit
        # cv2.imshow("build", final)
        # cv2.waitKey(0)
    i = 0
    while i < len(thetas)-1:     # subtract away 0 as many times as potentially added above
        bit = all[0]  # could also try 10
        #edge = cv2.Canny(all[0], 100, 250)//25
        # cv2.imshow("each1", bit)
        # mask = cv2.bitwise_and(bit, bit, mask=edge)
        final = final - bit # put a floor at 0
        final = final.astype(np.float16)
        # cv2.imshow("build", final)
        # cv2.waitKey(0)
        i += 1

    final = final.clip(min=0).astype(np.uint16)

    final = final*(255/np.amax(final))
    #print(np.amax(final))
    return final


