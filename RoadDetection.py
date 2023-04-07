import cv2
import numpy as np
from math import sqrt

    
def luminenceDifference(colorOne,colorTwo):
    pass
        
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
    
def findRoadRGB( roadColor, picture):
   
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
        diffPic[j]=np.reshape(differences, (diffPic.shape[1], 1))
    print("done with for")
    return diffPic
    
def findRoadYUV( roadColor, picture):

    diffColor=np.array(picture, dtype=np.uint8)
    combine=np.array(picture, dtype=np.uint8)
    diffSmall=picture-roadColor
    diffFromRoad=np.array(diffSmall, dtype=np.uint32)
    R=diffFromRoad[:,:,0]**2
    G=diffFromRoad[:,:,1]**2
    B=diffFromRoad[:,:,2]**2
    totalRBG=R+B+G
     
    diffPic=convertToYUV(picture)
    target=convertToYUV(roadColor)
    print("Conversions Done")
    diffFromRoad=diffPic-target
    
    
    total=(diffFromRoad[:,:,1])**2+(diffFromRoad[:,:,2])**2 
    
    for j in range(total.shape[0]):
        differences = [(sqrt(i)<5)*255 for i in total[j]]
        differencesColor = [(sqrt(i)<100) for i in totalRBG[j]]
        
        diffPic[j]=np.reshape(differences, (diffPic.shape[1], 1))
        diffColor[j]=np.reshape(differencesColor, (diffPic.shape[1], 1))
        combine[j] =diffPic[j]*diffColor[j]
    
    print("Done with for loop")
    
    return np.array(combine, dtype=np.uint8)
    
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
    #For testing purposes

    img = cv2.imread("FlatTrackDrive/test_1.png", cv2.IMREAD_ANYCOLOR)
    
    color = updateRoadColor(img)
    print("color is " +str(color))

    print("Begin YUV Comparison")
    photoTwo = findRoadYUV( color, img)
    photo = findRoadRGB( color, img)
    cv2.imwrite("diffPic.png", photo) 
   
    cv2.imwrite("ZdiffPic.png", photoTwo) 