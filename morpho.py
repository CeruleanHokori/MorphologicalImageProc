import cv2 
import numpy as np
from matplotlib import pyplot as plt
import math

#! Remark: Erosion depends on the thickness of the digit
#! If the digit is too thin, erosion will delete its borders
def number_of_blobs(filename, plot_result=False):
    #Loading and Thresholding
    eight = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    (thresh, BWeight) = cv2.threshold(eight, 127, 255, cv2.THRESH_BINARY)
    threshold = 0
    BWeight[BWeight>threshold]=1
    #Erosion
    kernel = np.ones((2,2),np.uint8)
    eroded_eight = cv2.erode(BWeight,kernel,iterations=1)
    #Flood filling to find holes
    flooded = eroded_eight.copy()
    mask = np.zeros((flooded.shape[0]+2,flooded.shape[1]+2),np.uint8)
    _,flooded,_,_ = cv2.floodFill(flooded,mask,(0,0),1)
    #Dilation
    dilated_img = cv2.dilate(flooded,kernel=np.ones((6,6),np.uint8))
    #Finding blobs by contouring them
    contour_blobs,_ = cv2.findContours(dilated_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(eight.shape,np.uint8)
    if plot_result:
        cv2.drawContours(blank,contour_blobs,-1,(255,255,255),3)
        plt.subplot(1,3,1)
        
        plt.imshow(BWeight,cmap='gray')
        plt.xlabel('Original')
        plt.subplot(1,3,2)
        plt.imshow(eroded_eight,cmap='gray')
        plt.xlabel('Eroded')
        plt.subplot(1,3,3)
        plt.imshow(blank,cmap='gray')
        plt.xlabel('Contouring blobs')
        #plt.imshow(dilated_img,cmap='gray')
        plt.show()
    return len(contour_blobs) -1 #We substract 1 because findContours also detects the background which is white after floodFill


