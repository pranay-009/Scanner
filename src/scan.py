import cv2 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

path=r"/content/WhatsApp Image 2022-06-15 at 5.37.12 PM.jpeg"
img=cv2.imread(path)
img2 = cv2.GaussianBlur(img,(7,7),0) 
gray=cv2.bitwise_not(img2,mask=None)
gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
kernel2=np.ones((1,1),np.uint8)
kernel = np.ones((1,1),np.uint8)
gray = cv2.erode(gray, kernel2,cv2.BORDER_REFLECT) 
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
gray=cv2.fastNlMeansDenoising(gray,2,2,7,21)
#gray = cv2.erode(gray, kernel2) 
#gausBlur = cv2.GaussianBlur(img, (5,5),0) 
retval2,threshold2 = cv2.threshold(gray,11,225,cv2.THRESH_OTSU)
th = cv2.adaptiveThreshold(gray,250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13,2)

plt.figure(figsize=(25,25))
plt.subplot(1,3,1)
plt.title("adaptive gaussian blurr")
plt.imshow(th,cmap="gray")
plt.subplot(1,3,2)
plt.title("original image")
plt.imshow(gray,cmap="gray")
plt.subplot(1,3,3)
plt.title("otsu image")
plt.imshow(threshold2,cmap="gray")
