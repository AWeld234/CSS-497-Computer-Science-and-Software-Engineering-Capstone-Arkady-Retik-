import time
import sys
from socket import socket, AF_INET, SOCK_DGRAM
import cv2 as cv
import numpy as np
import os
from time import time
from time import sleep
import mouse

import pyautogui
import win32api
import win32con
import pyautogui
import pydirectinput

from windowcapture import WindowCapture
from vision import Vision
from hsvfilter import HsvFilter
from pynput.mouse import Button, Controller

# Change the working directory to the folder this script is in.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Server connection to secondary machine is opened
SERVER_IP = '10.0.0.91'
PORT_NUMBER = 5000
SIZE = 1024
print("Test client sending packets to IP {0}, via port {1}\n".format(SERVER_IP, PORT_NUMBER))
mySocket = socket(AF_INET, SOCK_DGRAM)


# initialize the WindowCapture class
wincap = WindowCapture()
# initialize the Vision class
visionTarget = Vision('targets.JPG')

visionTarget.init_control_gui()

'''
# https://www.crazygames.com/game/guns-and-bottle
wincap = WindowCapture()
vision_gunsnbottle = Vision('gunsnbottle.jpg')
'''
hsvfilter = HsvFilter(164, 140, 0, 179, 255, 255, 0, 0, 0, 0)

pynputMouse = Controller()
loop_time = time()
while (True):

    # get an updated image of the game
    screenshot = wincap.get_screenshot()

    outputImage = visionTarget.apply_hsv_filter(screenshot, hsvfilter)

    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    # https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html

    # runs object detection
    rectangles = visionTarget.find(screenshot, 0.5)

    # draw results of the detections, Either drawRectangles or drawCrosshairs
    # can be chosen depending on the shape you want drawn
    rectangles = visionTarget.get_click_points(rectangles)
    # print("rectanlges: ", rectangles)

    if len(rectangles) != 0:
        counter = 0
        for i in rectangles:
            cord1 = rectangles[counter][0]
            cord2 = rectangles[counter][1]
            cord1string = str(cord1)
            cord2string = str(cord2)
            clickPointsCords = cord1string + ', ' + cord2string
            mySocket.sendto(clickPointsCords.encode('utf-8'), (SERVER_IP, PORT_NUMBER))
            counter = counter + 1
    outputImage = visionTarget.draw_crosshairs(screenshot, rectangles)

    print("Slept, not continuing")

    # display the processed image
    cv.imshow('Matches', outputImage)
    # debug the loop rate
    print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

print('Done.')
sys.exit()
