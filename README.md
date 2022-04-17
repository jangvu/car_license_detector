Car License Detector
A small computer vision project based on kaggle dataset (https://www.kaggle.com/datasets/elysian01/car-number-plate-detection?resource=download)

In this project, I used labelIMG to label the images, yolov3 to train the model, and pytesseract to recognize the car license number.

Step 1: Using labelIMG to label the license plate.




Step 2: Training a custom object detectection by using yolov3.




Step 4: Creating a detector using trained model and extracting the license plate part from the image.
![detected_img](https://user-images.githubusercontent.com/50269219/162429525-b2a08636-2b74-4e49-9546-088ac59513d9.jpg)
![cropped_img](https://user-images.githubusercontent.com/50269219/162429540-0e7113ac-8223-40b8-84e8-4c35ddaf2a1f.jpg)




Step 5: Rotating the image in order to get a better result.
![corrected_img](https://user-images.githubusercontent.com/50269219/162429763-8403d1c6-6b7e-47f5-b7fe-b6eb093f9f13.jpg)

Before rotating



<img width="328" alt="Screen Shot 2022-04-08 at 13 52 46" src="https://user-images.githubusercontent.com/50269219/162430582-10d9a20c-7776-4490-83fb-32db688b80b3.png">


After rotating




<img width="329" alt="Screen Shot 2022-04-08 at 13 52 54" src="https://user-images.githubusercontent.com/50269219/162430596-724be6f0-ebc8-40b9-9e9c-fb3e81026be1.png">


Step 6: Using pytesseract to extract the license number from the image.

