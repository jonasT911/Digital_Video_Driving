import cv2
import numpy as np

def colorDifferenceRGB(colorOne,colorTwo):
    pass
    
def luminenceDifference(colorOne,colorTwo):
    pass
    
def findRoad( roadColor, picture):
    
    pass
    
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
    