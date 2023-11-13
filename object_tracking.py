import cv2
import numpy as np
from object_detection import ObjectDetection

# Initialize ObjectDetection Detection
od = ObjectDetection()

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('video.mp4')

while True:
    _, frame = cap.read()

    # Detect objects
    (classs_ids, score, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    for class_id in classs_ids:
        print(class_id)
    # Display the resulting frame
        

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


