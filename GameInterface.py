import PIL.ImageGrab
import win32com.client as comclt
from time import sleep
import time, pyautogui
from pynput.keyboard import Key, Controller
import cv2 
import numpy as np
    
CameraRecord = []
print("Digital Video Project: begin")
keyboard = Controller()#Move this Maybe?
class gameInterface:

 
    def __init__(self):
        pass
        #wsh.AppActivate("notepad") # select another application
        wsh= comclt.Dispatch("WScript.Shell")
        wsh.AppActivate("RaceRoom") 
        #wsh.AppActivate("Warhammer: Vermintide 2") 
        
    def takePicture(self):
        im = PIL.ImageGrab.grab()
        CameraRecord.append(im)
       
        

    def hold_w (self,hold_time):
        start = time.time()
        while time.time() - start < hold_time:
            pyautogui.press('w')
               
    def drive(self): #This needs to be done with multithreading
        
        sleep(.2)
        keyboard.press(key)
        sleep(.5)
        keyboard.release(key)
      

if __name__== '__main__':
    driver =gameInterface()
    key="w"
    x=0
    while x<25:
        driver.takePicture()
        driver.drive()
        #print("Picture taken")
        x+=1
      
    i=0
    for im in CameraRecord:
      #  CameraRecord[i].show()#Showing is very slow
        i=i+1
        numpydata = np.array(im.convert('RGB'))
        destRGB = cv2.cvtColor(numpydata, cv2.COLOR_BGR2RGB)
        cv2.imwrite("test_"+str(i)+".png", destRGB) 
    print("end")

    