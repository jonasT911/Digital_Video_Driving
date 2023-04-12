import cv2
import cv
import numpy as np
from math import sqrt
import os
from skimage import measure

TESTING=False


def convertToYUV(color):

    image= np.copy(color)


    image=np.float16(image)
    v1 = [0.299, 0.587, 0.114]
    v2 = [-0.147, - 0.289,  0.436]
    v3 = [0.615, - 0.515, - 0.1]    
    mat1 = np.array([v1, v2, v3])
    mat1 = mat1.T
    image = np.matmul(image, mat1)
    output = np.float16(image.reshape(color.shape))

    
    
    return output
    

def findRoad( roadColor, picture):
   
    diffColor=np.array(picture, dtype=np.uint8)
    combine=np.array(picture, dtype=np.uint8)
    diffSmall=diffColor-roadColor
    diffFromRoad=np.array(diffSmall, dtype=np.uint32)
    R=diffFromRoad[:,:,0]**2
    G=diffFromRoad[:,:,1]**2
    B=diffFromRoad[:,:,2]**2
    totalRBG=R+B+G
     
    diffPic=convertToYUV(diffColor)
    target=convertToYUV(roadColor)
    
    diffFromRoad=diffPic-target
    
    
    total=(diffFromRoad[:,:,1])**2+(diffFromRoad[:,:,2])**2 
    
    for j in range(total.shape[0]):
        differences = [(sqrt(i)<5 and (j>int(total.shape[0]/2))) for i in total[j]]
        differencesColor = [((sqrt(i)<170 )and (j>int(total.shape[0]/2))) for i in totalRBG[j]] ##Later I will change this to use Y values instead.
        
        diffPic[j]=np.reshape(differences, (diffPic.shape[1], 1))
        diffColor[j]=np.reshape(differencesColor, (diffPic.shape[1], 1))
        combine[j] =((diffPic[j]+diffColor[j])>1)*255
    
   
    if(TESTING):
        cv2.imwrite("UVDiff.png",np.array(diffPic*255, dtype=np.uint8) ) 
        temp=255*diffColor
        cv2.imwrite("RGBDiff.png", np.array(temp, dtype=np.uint8)) 
    return np.array(combine, dtype=np.uint8)
    
def rejectColor(oldColor,newColor):
    if(sum(oldColor)==0):
        return newColor
    oldYUV=convertToYUV(np.array(oldColor))
    newYUV=convertToYUV(np.array(newColor))
    diff=abs(oldYUV-newYUV)
    if(diff[0]>40 or diff[1]>3 or diff[2]>3):
        #I may change this to allow leakage
        print("rejected new color")
        return oldColor 
    else:
        return newColor
    
def getRoadColor(picture):
    array=np.array(picture, dtype=np.uint8)
    width=array.shape[1]
    height=array.shape[0]
    cropped=array[(height-200):height-50 ,int(3*width/8):int( 5*width/8), :]
    divided=cropped/(cropped.shape[0]*cropped.shape[1])
    #cv2.imwrite("crop.png", cropped) 
    avg=sum(sum(divided))
    output=np.array(avg, dtype=np.uint8)

    return output
    
def compressOnXDirection(image):
    imageCopy = np.array(image, dtype=np.uint32)
    imageCopy=imageCopy/255
   
   
    x = np.sum(imageCopy[:,:,0], axis=0)
    #print("X is " +str(x))
    
    #Delete this
    t = int(x.shape[0]/2)
   
    #End delete
    
    return x
    
    
        
if __name__== '__main__':
    #For testing purposes

    TESTING=True
    
    img = cv2.imread("FlatTrackDrive/test_6.png", cv2.IMREAD_ANYCOLOR)
    
    color = getRoadColor(img)
    print("color is " +str(color))

    print("Begin YUV Comparison")
    photoTwo = findRoad( color, img)

    cv2.imwrite("RoadLocation.png", photoTwo) 

    compressOnXDirection(photoTwo)
#TODO: Add system to remove non-contiguouos pixels.
    for filename in os.listdir('FlatTrackDrive'):
       print(filename)
       img = cv2.imread("FlatTrackDrive/"+str(filename), cv2.IMREAD_ANYCOLOR)
       color = getRoadColor(img)
       print(color)
       photoTwo = findRoad( color, img)
       cv2.imwrite(str(filename)+".png", photoTwo) 