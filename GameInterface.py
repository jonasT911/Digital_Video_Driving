import PIL.ImageGrab
import win32com.client as comclt
from time import sleep
import time, pyautogui
from pynput.keyboard import Key, Controller
import cv2 
import numpy as np
    
import RoadDetection

CameraRecord = []
print("Digital Video Project: begin")
keyboard = Controller()#Move this Maybe?

#Debugging Controls
PrintImages=True
ShowRoadDetectionImages=True

class gameInterface:
    iteration=0
    roadColor=[0,0,0]
    driveKey="w"
    leftKey="a"
    rightKey="d"
    
    def __init__(self):
        iteration=0
        #wsh.AppActivate("notepad") # select another application
        wsh= comclt.Dispatch("WScript.Shell")
        wsh.AppActivate("RaceRoom") 
        #wsh.AppActivate("Warhammer: Vermintide 2") 
        
    def takePicture(self):
        im = PIL.ImageGrab.grab()
        CameraRecord.append(im)
        return im
        
    def drive(self,dutyCycle): #This needs to be done with multithreading
    
        sleep((1-dutyCycle)*1)
        keyboard.release(self.driveKey)
        sleep(dutyCycle*1)
        keyboard.press(self.driveKey)
        

    def turnCar(self,direction, dutyCycle=1):
        #direction will influence the duty cycle of turning left or right later
        #>0 means the road is to the right of the car. <0 means its to the left
        if(direction>-10000000): #This needs to be weighted to the amount of white on screen
            keyboard.press(self.rightKey)
        else:
            keyboard.release(self.rightKey)
        if(direction<10000000):
            keyboard.press(self.leftKey)
        else:
            keyboard.release(self.leftKey)
      
    def chooseDirection(self,picture):
    
        self.roadColor=RoadDetection.updateRoadColor(picture)#Later I might give the ability to reject dramatic changes
        RoadLocation=RoadDetection.findRoad( self.roadColor, picture)
        distribution=RoadDetection. compressOnXDirection(RoadLocation)
        total=0;
        for i in range(distribution.shape[0]):
            total=total + (i-distribution.shape[0]/2)*distribution[i]
       
        if(ShowRoadDetectionImages):
            print("iteration is " + str(self.iteration))
            print(total)
            print(self.roadColor)
           
            cv2.imwrite(str( self.iteration)+".png", RoadLocation) 
            self.iteration+=1
        
        self.turnCar(total)
        
        
if __name__== '__main__':
    driver =gameInterface()
    
    x=0
    while x<15:
        img=driver.takePicture()
        driver.chooseDirection(img)
        driver.drive(.7)
        #print("Picture taken")
        x+=1
      
    i=0
    if(PrintImages):
        for im in CameraRecord:
          #  CameraRecord[i].show()#Showing is very slow
            i=i+1
            numpydata = np.array(im.convert('RGB'))
            destRGB = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
            cv2.imwrite("test_"+str(i)+".png", destRGB) 
    print("end")

    