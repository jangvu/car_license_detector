
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
from rotating import deskew
from liscense_detector import LicensePlateDetector


lpd = LicensePlateDetector(
    pth_weights='yolov3_training_last.weights', 
    pth_cfg='yolov3_testing.cfg', 
    pth_classes='classes.txt'
)

# Detect license plate
lpd.detect('/Users/jang/Desktop/Hadoop_practice/car_liscense_plate_detector/test/img2.jpg')
#lpd.crop_plate
image = Image.fromarray(lpd.fig_image.astype(np.uint8))
image.show()

# Plot original image with rectangle around the plate


# Crop plate and show cropped plate
lpd.crop_plate()
plt.figure(figsize=(10, 4))
plt.imshow(cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB))


#Tessract Engine
img = 'cropped_img.jpg'
predicted_license_plates = []
custom_config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
predicted_result = pytesseract.image_to_string(img, lang ='eng',config = custom_config)
filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
print("Before rotating")
print(filter_predicted_result)

img = cv2.imread('cropped_img.jpg')
corrected_img = deskew(img)
cv2.imwrite('corrected_img.jpg', corrected_img)

img = 'corrected_img.jpg'
custom_config ='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
predicted_result = pytesseract.image_to_string(img, lang ='eng',config = custom_config)
filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
print('After rotating')
print(filter_predicted_result)
