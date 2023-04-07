import cv2
import numpy as np
from math import sqrt

    
def luminenceDifference(colorOne,colorTwo):
    pass
        
def convertToYUV(color):
    pass
    
def findRoad( roadColor, picture):
    outPic=np.array(picture, dtype=np.uint8)
    diffPic=np.array(picture, dtype=np.uint8)
    diffSmall=picture-roadColor
    diffFromRoad=np.array(diffSmall, dtype=np.uint32)
    R=diffFromRoad[:,:,0]**2
    G=diffFromRoad[:,:,1]**2
    B=diffFromRoad[:,:,2]**2
    print("Initial " + str(diffFromRoad[0][0][0]))
    print("Squared " + str(R[0][0]))
    print("Squared " + str(G[0][0]))
    print("Squared " + str(B[0][0]))
    total=R+B+G #I need to check if there are overflow errors
    for j in range(total.shape[0]):
        differences = [(sqrt(i)<100)*255 for i in total[j]]
        diffPic[j]=np.reshape(differences, (outPic.shape[1], 1))
        
    
    cv2.imwrite("diffPic.png", diffPic) 
    return diffPic
    
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
    
    photo = findRoad( color, img)