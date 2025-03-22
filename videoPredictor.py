# Install required libraries (if not already installed)
!pip install ultralytics opencv-python-headless matplotlib

import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from math import ceil, sqrt
from ultralytics import YOLO

# Load the YOLOv8x model (one of the most accurate in the YOLO family)
model = YOLO('yolov8x.pt')

# Define the device: use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define the video path (upload the file in Colab or provide the correct path)
video_path = 'videotest.mp4'  # Replace with the name or path of your video
try:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video!")
except Exception as e:
    print(f"Error: {e}")
    exit()

print("Processing video...")

# Define processing parameters
frame_skip = 5          # Process 1 out of every 5 frames to optimize time
conf_threshold = 0.3    # Minimum confidence threshold to consider a detection

frame_count = 0
detection_crops = []    # List to store cropped detections, labels, and confidence scores

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process only 1 frame every 'frame_skip'
    if frame_count % frame_skip == 0:
        try:
            results = model(frame)
        except Exception as e:
            print(f"Error during detection in frame {frame_count}: {e}")
            continue

        # Iterate over detected objects
        for box in results[0].boxes:
            # Extract bounding box coordinates
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy.tolist())

            # Ensure coordinates are within frame limits
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

            # Check if the bounding box has valid dimensions
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract label and confidence score
            cls_id = int(box.cls)
            label = results[0].names[cls_id]
            conf = float(box.conf)

            # Apply confidence filter
            if conf < conf_threshold:
                continue

            # Crop detected object and add to list
            crop = frame[y1:y2, x1:x2]
            detection_crops.append((crop, label, conf))
    frame_count += 1

cap.release()

# Display detected objects in a grid
if detection_crops:
    print("Displaying cropped detected objects:")
    num_detections = len(detection_crops)
    cols = int(sqrt(num_detections))
    rows = ceil(num_detections / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)  # Flatten array for easier iteration

    for i, (crop, label, conf) in enumerate(detection_crops):
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        axes[i].imshow(crop_rgb)
        axes[i].set_title(f"{label}: {conf:.2f}")
        axes[i].axis('off')

    # Disable axes of unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("No objects detected.")
