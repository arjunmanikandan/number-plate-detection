#Install and Import dependencies
#imutils is used for image resizing,reshaping,etc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr
import os,json

def read_json(config_path):
    with open(config_path,"r") as file:
        config = json.load(file)
    return config

def construct_gray_scale_image(config):
    car_image = cv2.imread(config["car_image_path"])
    gray_scale_image = cv2.cvtColor(car_image,cv2.COLOR_BGR2GRAY)
    return car_image,gray_scale_image

def display_gray_scale_img(gray_scale_image):
    plt.imshow(gray_scale_image)
    plt.show() 

#bilateralFilter parameters d(Diameter),SigmaColor(Blending two similar colors),SigmaSpace(Smoothen images)
def apply_filter(gray_scale_image): 
    bfilter = cv2.bilateralFilter(gray_scale_image,11,17,17) #Noise reduction & preserve imp features
    img_with_edges = cv2.Canny(bfilter,30,200) #Detect edges
    return img_with_edges

#Detect shapes within the image(Find Rectangle)
# cv2.findContours() detect boundaries of the shapes 
# RETR_TREE gives nested shapes,CHAIN_APPROX_SIMPLE removes unwanted edges
def identify_contours(filtered_image):
    keypoints = cv2.findContours(filtered_image.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]
    return contours

def identify_number_plate(image_contours):
    location = None
    for contour in image_contours:
        approx = cv2.approxPolyDP(contour,10,True) # 10 signifies accuracy number, True indicates that shape is closed
        if len(approx) == 4:
            location = approx
            break
    return location

def apply_mask(car_image,gray_scale_image,number_plate_location):
    mask = np.zeros(gray_scale_image.shape,np.uint8) #Array of zeros based on gray scale img size
    license_plate = cv2.drawContours(mask,[number_plate_location],0,255,thickness=-1) #Draw boundary on the spec loc numberplate in white rest all black
    license_plate = cv2.bitwise_and(car_image,car_image,mask=mask) #img comparison pixel by pixel, only white areas to be shown
    return license_plate,mask

def display_number_plate(license_plate,mask):
    x,y = np.where(mask==255) #row indices and column indices where pixel values are 255
    x1,y1 = np.min(x),np.min(y)
    x2,y2 = np.max(x),np.max(y)
    cropped_image = license_plate[x1:x2+1,y1:y2+1]
    plt.imshow(cropped_image)
    plt.show()

def destroy_windows():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    config = read_json(os.getenv("CONFIG_PATH"))
    car_image,gray_scale_image = construct_gray_scale_image(config)
    filtered_image = apply_filter(gray_scale_image)
    image_contours = identify_contours(filtered_image)
    number_plate_location = identify_number_plate(image_contours)
    license_plate,mask = apply_mask(car_image,gray_scale_image,number_plate_location)
    display_gray_scale_img(filtered_image)
    display_number_plate(license_plate,mask)
    destroy_windows()

main()