import cv2
import numpy as np
from ultralytics import YOLO

#load yolov8 model
model = YOLO("yolov8n.pt")

#load video
video_path = "highway_traffic.mp4"
cap = cv2.VideoCapture(video_path)
result = True

#result gives true or false
#Total frames = fps x duration(secs)
while True: 
    result,frame = cap.read()
    cv2.imshow("Frame",frame)
    classes = model.track(frame,persist=True) #To retain all the objects detected(detect and track) car,truck,flight..
    frame_ = classes[0].plot() # draws bounding boxes on the detected objects
    cv2.imshow("frame",frame_) #frame_ gives unique id for tracking,class_labels and confidence scores
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break


    