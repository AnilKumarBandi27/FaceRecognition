# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 16:56:50 2018

@author: Team 16
"""

# importing OpenCV library
import cv2

# importing matplotlib library
#import matplotlib.pyplot as plt

import os

# Function for converting a RGB image into Gray Scale

def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#image_dir = os.path.join(BASE_DIR, "images")

# Read image / images
read_image = cv2.imread('images/a.jpg')

#load cascade classifier training file for haarcascade 
haar_face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')


# Function for detecting the faces

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
 
 #just making a copy of image passed, so that passed image is not changed 
 img_copy = colored_img.copy()          
 
 
 #convert the test image to gray image as opencv face detector expects gray images
 gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
 
 #let's detect multiscale (some images may be closer to camera than others) images
 faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);      
 
 #Go through the list of faces and draw them as rectangles on original colored image
 for (x, y, w, h) in faces:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)              
 
 return img_copy

#call our function to detect faces 
faces_detected_img = detect_faces(haar_face_cascade, read_image)

#convert image to RGB and show image 
#plt.imshow(convertToRGB(faces_detected_img))


# Show Image Using thw OpenCV
cv2.imshow('Show Image', faces_detected_img)
cv2.waitKey(0)
cv2.destroyAllWindows()