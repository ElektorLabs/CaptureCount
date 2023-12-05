# This code is working but one object is getting detected multiple times in one second. 

import cv2
import pandas as pd
import numpy as np
import subprocess
import os
from datetime import datetime

# Load pre-trained model and configuration
model = './yolo/yolov3.weights'  # Update this path to your model's path
config = './yolo/yolov3.cfg'     # Update this path to your config file's path
net = cv2.dnn.readNetFromDarknet(config, model)

# Load class names
classes = []
with open("./yolo/coco.names", "r") as f:  # Update to the path of your coco.names file
    classes = [line.strip() for line in f.readlines()]

# Function to get output layers
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Function to draw bounding box
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = np.random.uniform(0, 255, size=(3,))
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Function to capture image using libcamera-still
def capture_image(image_path):
    subprocess.run(["libcamera-still", "-o", image_path, "--width", "1280", "--height", "720", "-n", "-t", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Initialize a DataFrame to store counts
data_frame = pd.DataFrame(columns=['Timestamp', 'Type', 'Count'])

# Main loop
image_path = "temp.jpg"
csv_file_path = "object_counts.csv"  # Path to save CSV file

try:
    while True:
        capture_image(image_path)

        # Check if image is captured correctly
        if not os.path.exists(image_path):
            print("Image capture failed, image file not found.")
            continue

        frame = cv2.imread(image_path)
        if frame is None:
            print("Failed to read the image file.")
            continue

        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392

        # Create a blob and pass it through the model
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        # Process the outputs
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            data_frame = pd.concat([data_frame, pd.DataFrame([{'Timestamp': datetime.now(), 'Type': classes[class_ids[i]], 'Count': 1}])], ignore_index=True)

        # Display the frame
        cv2.imshow("Object Detection", frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    if os.path.exists(image_path):
        os.remove(image_path)

    # Save counts to CSV
    data_frame.to_csv(csv_file_path, index=False)
