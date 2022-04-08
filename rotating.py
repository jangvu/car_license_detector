#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:08:31 2022

@author: jang
"""
import pytesseract
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')

    img = cv2.medianBlur(src_img, 3)

    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size

    #print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1

    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))

img = cv2.imread('cropped_img.jpg')
#IMG2GRAY
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)

#tresh
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.imshow(thresh)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(thresh,(5,5),0)
th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

plt.imshow(th3)
plt.show()




corrected_img = deskew(img)
cv2.imwrite('corrected_img.jpg', corrected_img)
cv2.imshow('Img', corrected_img)


img = cv2.imread('corrected_img.jpg')
#IMG2GRAY
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)

#tresh
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.imshow(thresh)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(thresh,(5,5),0)
th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

plt.imshow(th3)
plt.show()



img = 'corrected_img.jpg'
predicted_license_plates = []
custom_config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
predicted_result = pytesseract.image_to_string(img, lang ='eng',config = custom_config)
filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
predicted_license_plates.append(filter_predicted_result)
print(predicted_license_plates)
