import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math

def corner_kernel(n): #Returns kernel for the desired corner
    kernel = np.zeros((5,5),np.uint8)
    if n == 1:
        kernel[2,2],kernel[2,3],kernel[3,2] = 1,1,1
        return kernel
    elif n == 2:
        kernel[2,2],kernel[2,1],kernel[3,2] = 1,1,1
        return kernel
    elif n == 3:
        kernel[2,2],kernel[2,1],kernel[1,2] = 1,1,1
        return kernel
    else:
        kernel[2,2],kernel[2,3],kernel[1,2] = 1,1,1
        return kernel

def square_edge_detector(filename):
    #Loading and Thresholding
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    (thresh, BWimg) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    threshold = 0
    BWimg[BWimg>threshold]=1
    #corner kernels
    corner_1 = corner_kernel(1)
    corner_2 = corner_kernel(2)
    corner_3 = corner_kernel(3)
    corner_4 = corner_kernel(4)
    #erosion
    c1,c2,c3,c4 = cv2.erode(BWimg,corner_1),cv2.erode(BWimg,corner_2),cv2.erode(BWimg,corner_3),cv2.erode(BWimg,corner_4)
    #OR
    blank_img = np.zeros(BWimg.shape,np.uint8)
    blank_img = np.logical_or(blank_img,c1)
    blank_img = np.logical_or(blank_img,c2)
    blank_img = np.logical_or(blank_img,c3)
    blank_img = np.logical_or(blank_img,c4)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap="gray")
    plt.xlabel("Original")
    plt.subplot(1,2,2)
    plt.imshow(blank_img,cmap="gray")
    plt.xlabel("Square edges detected")
    plt.show()

def detect_plus(filename):
    #Loading and Thresholding
    img = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    (thresh, BWimg) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    threshold = 0
    BWimg[BWimg>threshold]=1
    #Plus kernel
    plus = cv2.imread("./images/plus.png",cv2.IMREAD_GRAYSCALE)
    plus[plus>threshold] = 1
    #Erosion
    blank = img.copy()
    blank = cv2.erode(img,plus)
    res = np.zeros(img.shape,np.uint8)
    for i in range(blank.shape[1]):
        for j in range(blank.shape[0]):
            if blank[i,j] >0:
                block = img[i-2:i+3,j-2:j+3]
                res[i-2:i+3,j-2:j+3] = np.logical_and(block,plus)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap="gray")
    plt.xlabel("Original")
    plt.subplot(1,2,2)
    plt.imshow(res,cmap="gray")
    plt.xlabel("Plus detected")
    plt.show()

#This procedure will execute detection for test_text.png and letter_e as the detected letter
#You can change the text image and letter kernel by modifying the next two lines of code
#Loading and Thresholding
txt = cv2.imread("./images/test_text.png",cv2.IMREAD_GRAYSCALE)
letter = cv2.imread("./images/letter_e.png",cv2.IMREAD_GRAYSCALE)
(thresh, txt) = cv2.threshold(txt, 127, 255, cv2.THRESH_BINARY)
threshold = 0
txt[txt>threshold]=1
txt = 1-txt
(thresh, letter) = cv2.threshold(letter, 127, 255, cv2.THRESH_BINARY)
letter[letter>threshold]=1
letter = 1-letter
#Dilation
txt = cv2.dilate(txt,np.ones((2,2),np.uint8))
#Erosion
blank = cv2.erode(txt,letter)
#Location + Logical AND
res = np.zeros(txt.shape,np.uint8)
h,w = letter.shape
for i in range(blank.shape[0]):
    for j in range(blank.shape[1]):
        if blank[i,j] >0:
            block = txt[i-h//2:i+h//2+1,j-w//2:j+w//2+1]
            res[i-h//2:i+h//2+1,j-w//2:j+w//2+1] = np.logical_and(block,letter)
plt.subplot(1,2,1)
plt.imshow(txt,cmap="gray")
plt.xlabel("Original text")
plt.subplot(1,2,2)
plt.imshow(res,cmap="gray")
plt.xlabel("Detected letter e")
plt.show()
