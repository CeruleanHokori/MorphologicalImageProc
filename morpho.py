import cv2 
import numpy as np
from matplotlib import pyplot as plt

eight = cv2.imread("./images/8.png",cv2.IMREAD_GRAYSCALE)
(thresh, BWeight) = cv2.threshold(eight, 127, 255, cv2.THRESH_BINARY)
threshold = 0
eight[eight>threshold]=1
kernel = np.ones((8,8),np.uint8)
eroded_eight = cv2.erode(eight,kernel,iterations=1)
# Contours using findContours
# contours, hierarchy = cv2.findContours(eroded_eight,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
copy = eroded_eight.copy()
# cv2.drawContours(copy,contours,-1,(255,0,0), 1)
mask = np.zeros((copy.shape[0]+2,copy.shape[1]+2),np.uint8)
_,img,_,_ = cv2.floodFill(copy,mask,(0,0),255)
dilated_img = cv2.dilate(img,kernel=np.ones((11,11),np.uint8))
plt.subplot(1,3,1)
plt.imshow(BWeight,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(eroded_eight,cmap='gray')
plt.subplot(1,3,3)
# plt.imshow(copy)
plt.imshow(dilated_img,cmap='gray')
plt.show()
s = []
for elt in eight:
    for pixel in elt:
        if pixel not in s:
            s.append(pixel)
print(s)