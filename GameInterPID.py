import PIL.ImageGrab
import win32com.client as comclt
import win32gui
from time import sleep
import time, pyautogui
from pynput.keyboard import Key, Controller
import cv2 
import numpy as np
import RoadDetection
#import RoadDetection2
from simplePID import PID
import threading
#from features import *
import multiprocessing
from multiprocessing import Process

#from RoadDetection2 import simpleRoad
import datetime


print("Call Again?")
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
    brakeKey = "s"   # also will begin to reverse
    speed= .43     #0.4
    error = 0
    go = True
    brake = False
    buffer = list(np.zeros(4))
    
    def __init__(self):
        iteration=0
        #wsh.AppActivate("notepad") # select another application
        wsh= comclt.Dispatch("WScript.Shell")
        #wsh.AppActivate("RaceRoom") 
        wsh.AppActivate("RaceRoom Racing Experience")
        #wsh.AppActivate("Warhammer: Vermintide 2") 
        
    def takePicture(self):
        bbox = self.getWindow()
        im = PIL.ImageGrab.grab(bbox)
        CameraRecord.append(im)
        return im
        
    def pressAKey(self,dutyCycle,key): #This needs to be done with multithreading
        
        keyboard.press(key)
        sleep((dutyCycle)*.3)
        keyboard.release(key)
        sleep((1-dutyCycle)*.3)
       
    def driveThread(self, hold): #self? or driver?
        while self.speed>-1:
            if self.go == True:
                self.pressAKey(hold,self.driveKey)
                #print("Drive!")
                #print(self.error)
                
            elif self.brake == True:
                self.pressAKey(hold,self.brakeKey)
                #print("Brake!")
            else:
                continue

    def accelerate(self, fade):
        keyboard.press(self.driveKey)
        sleep(.5*fade)
        keyboard.release(self.driveKey)
        sleep(.1)
    
    
    def turnCar(self,direction, hold=1):
        #direction will influence the duty cycle of turning left or right later
        #>0 means the road is to the right of the car. <0 means its to the left
        if(direction<-.001):  #Turn Left
            keyboard.release(self.rightKey)
            keyboard.press(self.leftKey)
            sleep(hold/24)                        # Tune Hold Time with PID
            keyboard.release(self.leftKey) 
            driver.accelerate(fade = 1)
            print("Turning Left")
        elif(direction>.001):  #Turn Right  
            keyboard.release(self.leftKey)
            keyboard.press(self.rightKey)
            sleep(hold/24)
            keyboard.release(self.rightKey)
            driver.accelerate(fade = 1)
            print("Turning Right")
        elif(abs(direction)<.001):#Drive Straight , ignore control signal
            keyboard.release(self.leftKey)
            keyboard.release(self.rightKey)
    
    # picks out raceroom window only
    def getWindow(self):
        toplist, winlist = [], []
        def enum_cb(hwnd, results):
            winlist.append((hwnd, win32gui.GetWindowText(hwnd)))
        win32gui.EnumWindows(enum_cb, toplist)
        raceroom = [(hwnd, title) for hwnd, title in winlist if 'raceroom' in title.lower()]
        # just grab the hwnd for first window matching firefox
        rr = raceroom[0]
        hwnd = rr[0]
        win32gui.SetForegroundWindow(hwnd)
        bbox = win32gui.GetWindowRect(hwnd)
        return bbox



    def chooseDirection(self,picture):
        img = np.array(picture, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        newColor=RoadDetection.getRoadColor(picture)#Later I might give the ability to reject dramatic changes
        self.roadColor=RoadDetection.rejectColor(self.roadColor,newColor)
        RoadLocation=RoadDetection.findRoad( self.roadColor, picture)
        distribution=RoadDetection. compressOnXDirection(RoadLocation)

        total=sum(distribution)
        avgXLocation=0
        for i in range(distribution.shape[0]):
            avgXLocation=avgXLocation + (i*distribution[i])
        
        avgXLocation= (avgXLocation/total)
        error = avgXLocation - distribution.shape[0]//2
        
        if(ShowRoadDetectionImages):
            print("iteration is " + str(self.iteration))
            print("Error: " , error)   
            #print(self.roadColor)
           
            cv2.imwrite("DebuggingImages/"+str( self.iteration)+".png", RoadLocation) 
            self.iteration+=1

        return error
        
    # Performance is not good here yet
    def measureTarget(self,picture):  

        mask, lane, orig, mid, nearTgt, farTgt, area = RoadDetection2.getRoadMask(picture,getMetrics = True)
        if area == 0:
            self.go = False
            self.brake = True
        else:
            self.go = True
        # (2) Calculate Error:
        self.error = farTgt[0] - mid

        # smooth the error, or the control
        self.buffer.pop(0)
        self.buffer.append(self.error)
        self.error = np.mean(self.buffer)
        
        if(ShowRoadDetectionImages):
            print("iteration is " + str(self.iteration))
            print("Error: " , self.error)   
            print(self.roadColor)
           
            cv2.imwrite("DebuggingImages/Masks/"+str( self.iteration)+".png", mask)
            cv2.imwrite("DebuggingImages/Lanes/"+str( self.iteration)+".png", lane)  
            cv2.imwrite("DebuggingImages/Points/"+str( self.iteration)+".png", orig) 
            self.iteration+=1
        
        return self.error, area, farTgt  # come back to area 
    
    def steer(self,error):

        # Not hitting ?
        #print("Steer!")
        control = abs(pid(error))  # Try to correct from last measurement
        #print("Control Correction: " , control)   
        self.turnCar(error, hold = control) 
        

if __name__== '__main__':
    driver =gameInterface()
    
    wts  = [ 1.8 ,  .2  , .2 ]                  # Tune Control Parameters Here, 1e-3 scale factor
    
    wts = [ 1e-2 *w for w in wts]
    pid = PID(wts[0],wts[1],wts[2], setpoint = 0) # try to drive error to 0
    #spid = PID(wts[0],wts[1],wts[2], setpoint = .3) # try to control speed better 

    drive_thread = threading.Thread(target=driver.driveThread, args=(driver.speed,))
    #steer_thread = Process(target=driver.steerThread, args=(pid,driver.error,))

    drive_thread.start()
    
    try:    
        x=0
        while x<20:
            # Initial Steps, give some acceletation/boost

            
            img=driver.takePicture()
            # img=cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_BGR2RGB)
            # error, area, target = driver.measureTarget(img)
            a=datetime.datetime.now()
            error = driver.chooseDirection(img)
            #mask, error = simpleRoad(img, getMetrics=False)
            print("Error:", error)
            b=datetime.datetime.now()
            driver.steer(error)
            c=datetime.datetime.now()

            print("Time data")
            print(b-a)
            print(c-b)
            print("End Time data")

            if (ShowRoadDetectionImages):
                pass
                #cv2.imwrite("DebuggingImages/test_"+str(x)+".png",mask) 

            x+=1
    finally:
        driver.speed=-1
        drive_thread.join()
        #steer_thread.join()
        
    i=0
    if(PrintImages):
        for im in CameraRecord:
          #  CameraRecord[i].show()#Showing is very slow
            i=i+1
            numpydata = np.array(im.convert('RGB'))
            destRGB = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
            cv2.imwrite("DebuggingImages/Masks/test_"+str(i)+".png",img) 
    print("end")



    
