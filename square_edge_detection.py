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
square_edge_detector("./images/empty_square.png")
