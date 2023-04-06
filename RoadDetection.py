import cv2
import numpy as np
from math import sqrt

    
def luminenceDifference(colorOne,colorTwo):
    pass
    
def findRoad( roadColor, picture):
    outPic=np.array(picture, dtype=np.uint8)
    diffFromRoad=picture-roadColor
    R=diffFromRoad[:,:,0]**2
    G=diffFromRoad[:,:,1]**2
    B=diffFromRoad[:,:,2]**2
    total=R+B+G #I need to check if there are overflow errors
    for j in range(total.shape[0]):
        differences = [sqrt(i) for i in total[j]]
        outPic[j]=np.reshape(differences, (outPic.shape[1], 1))
    print(total)
    
def updateRoadColor(picture):
    width=picture.shape[1]
    height=picture.shape[0]
    cropped=picture[(height-100):height-50 ,int(width/4):int( 3*width/4), :]
    divided=cropped/(cropped.shape[0]*cropped.shape[1])
    #cv2.imwrite("crop.png", cropped) 
    avg=sum(sum(divided))
    output=np.array(avg, dtype=np.uint8)

    return output
    
def importPhoto(pictureName):#For testing purposes
    img = cv2.imread(pictureName, cv2.IMREAD_ANYCOLOR)
    
        
if __name__== '__main__':
    #Run testing Script

    img = cv2.imread("FlatTrackDrive/test_1.png", cv2.IMREAD_ANYCOLOR)
    
    color = updateRoadColor(img)
    print("color is " +str(color))
    
    findRoad( color, img)