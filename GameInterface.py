import PIL.ImageGrab
import win32com.client as comclt
from time import sleep
import time, pyautogui
from pynput.keyboard import Key, Controller
    
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
        
        sleep(.5)
        keyboard.press(key)
        sleep(1)
        keyboard.release(key)
      

if __name__== '__main__':
    driver =gameInterface()
    key="w"
    x=0
    while x<10:
        driver.takePicture()
        driver.drive()
        #print("Picture taken")
        x+=1
      

    [im.show() for im in CameraRecord]
        #CameraRecord[i].show()#Showing is very slow
    print("end")

    