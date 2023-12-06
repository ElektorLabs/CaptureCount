# Wildlife Tracker on Raspberry Pi

## Note
- There is an alternative script named `wildlifetracker.py` in the repository, designed to work on Ubuntu 23.10 on Raspberry Pi 5.
- Despite the name, the script is capable of detecting a wide range of objects as specified in the YOLO `coco.names` file.

## Overview
`wildlifetracker-pi.py` is a Python script designed for real-time object detection on Raspberry Pi 5, using the YOLO (You Only Look Once) model developed by Joseph Redmon. This powerful tool can identify and track various objects as defined in the `coco.names` file from the YOLO model. It's optimized for Pi OS on Raspberry Pi 5.

## Features
- Real-time object detection using YOLOv3.
- Saves full-frame images upon object detection, rather than cropped images of detected objects.
- Utilizes a pre-trained YOLO model for efficient and accurate detection.
- Keeps track of detected objects and their counts.
- Outputs results in various formats including images, CSV files, and logs.

## Prerequisites
- Raspberry Pi 5 with Pi OS installed.
- Python environment with necessary libraries (`cv2`, `pandas`, `numpy`, `subprocess`, `os`).
- YOLOv3 model files (`yolov3.weights`, `yolov3.cfg`) and `coco.names`.

## Installation
1. Clone the repository to your Raspberry Pi.
2. Ensure Python and the required libraries are installed.
3. Download YOLOv3 model files and `coco.names`, and place them in the `./yolo` directory within the script's folder.
4. Run `wildlifetracker-pi.py`.

## Usage
- Execute the script: `python wildlifetracker-pi.py`
- The script continuously captures images and performs object detection.
- Detected objects are framed with bounding boxes in real-time display.
- Full-frame images are saved in the `output` folder when an object is detected.
- Object count per category is saved in `object_counts.csv` and `total_object_counts.csv`.
- A text log of total counts is also maintained in `total_counts_log.txt`.

## Acknowledgements
Credits to Joseph Redmon [@pjreddie](https://github.com/pjreddie) for developing the YOLO: Real-Time Object Detection model, which was crucial in the creation of this project.


