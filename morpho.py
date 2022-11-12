from ssl import VerifyFlags
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

def line(filename,plot_result=False):
    #This function analyzes the image and returns a list of [theta,r] lines found in the image
    #We then process that list to find the number of (near) horizontal lines and number of (near) vertical lines 
    test = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    (thresh, Bw) = cv2.threshold(test, 127, 255, cv2.THRESH_BINARY)
    threshold = 0
    Bw[Bw>threshold]=1
    kernel = np.ones((3,3),np.uint8)
    eroded = cv2.erode(Bw,kernel,iterations=1)
    lines = cv2.HoughLines(eroded,1,np.pi/18,50,None,0,0)
    dest = np.zeros(eroded.shape,np.uint8)
    if plot_result:
        plt.imshow(eroded,cmap="gray")
        plt.show()
        for i in range(0,len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(dest,pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
        plt.imshow(dest,cmap='gray')
        plt.xlabel(f"{len(lines)} line found")
        plt.show()
    if lines is not None:
        return lines
    else:
        return []

def h_or_v(theta): #Tells if the angle theta is vertical or horizontal
    #! Houghlines gives the angle from the vertical and not horiontal (polar)
    new_theta = theta*180/np.pi
    if theta > 360:
        new_theta = theta - math.floor(theta/360)*360
    if 0<=new_theta<=20 or 160<=new_theta<=180:
        return 0 #vertical
    if 70<=new_theta<=110:
        return 1 #horizontal
    else:
        return 2 #Inclined

def line_type(lines): #Takes the lines returned by line function and returns the number of lines found
    #and their orientation
    h_and_v = [0,0,0]
    for line in lines:
        theta = line[0][1]
        h_and_v[h_or_v(theta)] += 1
    return h_and_v #This list contains the number of vertical,horizontal and inclined lines


def decide(filename):
    nb_blobs = number_of_blobs(filename)
    if nb_blobs == 1: #[0,6,9]
        lines = line_type(line(filename))
        return [0,6,9]
    if nb_blobs == 2:
        print(8)
        return 8
    else: #[1,3,4,5,7]
        lines = line_type(line(filename))
        return [1,3,4,5,7]


#Uncomment the following line and execute one of the desired operations
# Filename = ""

#Execute the decision tree
#print(decide(Filename))

#Execute the following line to count the number of blobs
#number_of_blobs(Filename,plot_result=True)
