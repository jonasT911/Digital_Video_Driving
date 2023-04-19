import PIL.ImageGrab
import win32com.client as comclt
from time import sleep
import time, pyautogui
from pynput.keyboard import Key, Controller
import cv2 
import numpy as np
    
import RoadDetection

import threading

import datetime

from Detection import *

from Detection import crop
CameraRecord = []
detectionImages = []
print("Digital Video Project: begin")
keyboard = Controller()#Move this Maybe?

#Debugging Controls
PrintImages=True
ShowRoadDetectionImages=True

MAX_SPEED=.41

class gameInterface:
    iteration=0
    roadColor=[0,0,0]
    driveKey="w"
    leftKey="a"
    rightKey="d"
    speed=MAX_SPEED
    
    previousError=0
    
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
        
    def pressAKey(self,dutyCycle,key): #This needs to be done with multithreading
        
        keyboard.press(key)
        sleep((dutyCycle)*.3)
        keyboard.release(key)
        sleep((1-dutyCycle)*.3)
       
    def driveThread(self): #This needs to be done with multithreading
        while driver.speed>-1:
          
            driver.pressAKey(self.speed,self.driveKey)

       

    def turnCar(self,direction):
        threshold=10
        diff=direction-self.previousError
        #direction will influence the duty cycle of turning left or right later
        #>0 means the road is to the right of the car. <0 means its to the left
        if(direction<-threshold):#Turn Left
            keyboard.release(self.rightKey)
            keyboard.press(self.leftKey)
            #print("Turning Left")
        elif(direction>threshold):      #Turn Right  
            keyboard.release(self.leftKey)
            keyboard.press(self.rightKey)
            #print("Turning Right")
        elif(abs(direction)<threshold):#Drive Straight
           
            keyboard.release(self.leftKey)
            keyboard.release(self.rightKey)
        
        #Catch large swerves.
        if(abs(diff)>5):
            if(diff<0):
                keyboard.release(self.rightKey)
                keyboard.press(self.leftKey)
            else:
                keyboard.release(self.leftKey)
                keyboard.press(self.rightKey)  
         
        self.previousError=direction
      
    def chooseDirection(self,picture):
      
        newColor=RoadDetection.getRoadColor(picture)#Later I might give the ability to reject dramatic changes

        self.roadColor=RoadDetection.rejectColor(self.roadColor,newColor)
        c=datetime.datetime.now()
        
        RoadLocation=RoadDetection.findRoad( self.roadColor, picture) #Too slow
        d=datetime.datetime.now()
        
        distribution=RoadDetection.compressOnXDirection(RoadLocation)
        e=datetime.datetime.now()
        
        total=sum(distribution);
        avgXLocation=0;
        for i in range(distribution.shape[0]):
            avgXLocation += (i-distribution.shape[0]/2)*distribution[i]
        
    
        avgXLocation=avgXLocation/total
        if(ShowRoadDetectionImages):
   
            detectionImages.append(RoadLocation)
           
            self.iteration+=1
        # print("iteration is " + str(self.iteration))
        print(avgXLocation)
        #print(self.roadColor) 
        self.turnCar(avgXLocation)
     
        
        print("Total Time")
   
        print(str(d-c))
        print(str(e-d))
        print("Total Time END")
        
if __name__== '__main__':
    driver =gameInterface()


    drive_thread = threading.Thread(target=driver.driveThread, args=())

    drive_thread.start()


    try:    
        x=0
        while True:
            img=driver.takePicture()
            c=datetime.datetime.now()
            # driver.chooseDirection(img)
            # region = crop(img, view = playerView, peripherals = sides)
            convertedImg= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            
            try:
                LineDemo,maskImg,error=getRoad(convertedImg,dispTarget=True)
                maskImg*=255
               
                cv2.imwrite("DebuggingImages/Mask_"+str(x)+".png", maskImg)
            except:
                print("No Road found")
            print("Error value is " +str(error))
            driver.turnCar(error)
            d=datetime.datetime.now()
            print("Time value is" +str(d-c))
            
            # slowDown=abs(error)/4000
            # slowDown=min(slowDown,.1)
            # driver.speed=MAX_SPEED-slowDown
            
           
            x+=1
            
      
    finally:
        driver.speed=-1
        drive_thread.join()  
        
    i=0
    if(PrintImages):
        for im in CameraRecord:
          #  CameraRecord[i].show()#Showing is very slow
            i=i+1
            numpydata = np.array(im.convert('RGB'))
            destRGB = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
            cv2.imwrite("DebuggingImages/test_"+str(i)+".png", destRGB) 
    if(ShowRoadDetectionImages):
        i=0
        for im in detectionImages:
          #  CameraRecord[i].show()#Showing is very slow
            i=i+1
            numpydata = np.array(im)
            destRGB = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)

            cv2.imwrite("DebuggingImages/RoadLoc_"+str( i)+".png",destRGB ) 
    print("end")

    