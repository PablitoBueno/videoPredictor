# YOLOv8 Object Detection and Cropping from Video

## Description
This application processes a video to perform object detection using the YOLOv8x model from Ultralytics. It extracts detected objects from selected frames (skipping a number of frames to optimize performance), crops them based on bounding box coordinates, and displays them in a grid. The script also applies a confidence filter to ensure only reliable detections are shown.

## Features
- **YOLOv8x Model**: Uses one of the most accurate models in the YOLO family for object detection.
- **GPU Support**: Automatically uses GPU if available, with a fallback to CPU.
- **Video Processing**: Reads a video file and processes one out of every few frames (configurable) to speed up detection.
- **Detection Filtering**: Applies a confidence threshold to filter out low-confidence detections.
- **Object Cropping**: Extracts and crops detected objects from video frames.
- **Visualization**: Displays the cropped objects in a grid using Matplotlib.

## Technologies Used
- Python
- Ultralytics YOLO
- OpenCV (opencv-python-headless)
- Matplotlib
- NumPy
- PyTorch

## Installation and Setup

### Requirements
- Python 3.8 or later.

### Installing Dependencies
Install the required libraries using pip:
```sh
pip install ultralytics opencv-python-headless matplotlib
```

## How to Run
1. Ensure the video file (e.g., `videotest.mp4`) is available in the specified path or adjust the `video_path` variable accordingly.
2. Run the script:
   ```sh
   python your_script_name.py
   ```
   The script will process the video, extract object detections, and display the cropped objects in a grid.

## Configuration Parameters
- **frame_skip**: Process 1 out of every `n` frames to reduce computation time (default: 5).
- **conf_threshold**: Minimum confidence threshold for a detection to be considered (default: 0.3).
