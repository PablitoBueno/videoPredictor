# Install necessary libraries (run once)
!pip install ultralytics opencv-python-headless matplotlib tqdm

# Upload your .mp4 video in Colab
from google.colab import files
uploaded = files.upload()

# Automatically pick the first uploaded file as video_path
import os
video_path = next(iter(uploaded))
print(f"Video uploaded: {video_path}")

import os
import cv2
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """
    Convert a frame index to a "MM:SS.ss" timestamp.
    """
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def process_segment(args):
    """
    Run YOLOv8 detection on [start_frame, end_frame] of the video.
    Returns blocks_by_class dict and avg inference time.
    """
    video_path, start_frame, end_frame, target_classes, frame_skip, conf_threshold, model_weights = args

    # Load model inside each worker
    model = YOLO(model_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    blocks_by_class = {cls: [] for cls in target_classes}
    in_block = {cls: False for cls in target_classes}
    block_start = {cls: None for cls in target_classes}
    first_crop = {cls: None for cls in target_classes}
    last_crop = {cls: None for cls in target_classes}

    frame_idx = start_frame
    infer_times = []

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_skip == 0:
            t0 = time.time()
            with torch.no_grad():
                results = model(frame)
            t1 = time.time()
            infer_times.append(t1 - t0)

            # collect per-class detections this frame
            dets = {cls: [] for cls in target_classes}
            for box in results[0].boxes:
                conf = float(box.conf)
                label = results[0].names[int(box.cls)]
                if conf < conf_threshold or label not in target_classes:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                x1,y1 = max(x1,0), max(y1,0)
                x2,y2 = min(x2,frame.shape[1]), min(y2,frame.shape[0])
                if x2<=x1 or y2<=y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                dets[label].append((crop, conf))

            # update blocks
            for cls in target_classes:
                if dets[cls]:
                    # best crop this frame
                    best_crop, _ = max(dets[cls], key=lambda x: x[1])
                    if not in_block[cls]:
                        in_block[cls] = True
                        block_start[cls] = frame_idx
                        first_crop[cls] = best_crop
                    last_crop[cls] = best_crop
                else:
                    if in_block[cls]:
                        blocks_by_class[cls].append({
                            "start_frame": block_start[cls],
                            "end_frame": frame_idx - 1,
                            "first_crop": first_crop[cls],
                            "last_crop": last_crop[cls],
                        })
                        in_block[cls] = False

        frame_idx += 1

    cap.release()
    # close any open blocks
    for cls in target_classes:
        if in_block[cls]:
            blocks_by_class[cls].append({
                "start_frame": block_start[cls],
                "end_frame": min(end_frame, frame_idx-1),
                "first_crop": first_crop[cls],
                "last_crop": last_crop[cls],
            })
    avg_inf = sum(infer_times)/len(infer_times) if infer_times else 0.0
    return {"blocks_by_class": blocks_by_class, "avg_inference": avg_inf}


def adaptive_parallel_process(video_path: str,
                              target_classes: list,
                              frame_skip: int = 5,
                              conf_threshold: float = 0.3,
                              model_weights: str = "yolov8x.pt"):
    # read metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {video_path} | FPS: {fps:.2f} | Frames: {total_frames}")

    # split into segments = CPU cores
    num_workers = os.cpu_count() or 1
    frames_per = math.ceil(total_frames / num_workers)
    segments = []
    for i in range(num_workers):
        start_f = i * frames_per
        end_f = min((i+1)*frames_per - 1, total_frames - 1)
        if start_f <= end_f:
            segments.append((start_f, end_f))
    print(f"Using {len(segments)} workers, ~{frames_per} frames each")

    # prepare arguments
    args_list = [
        (video_path, start, end, target_classes, frame_skip, conf_threshold, model_weights)
        for start, end in segments
    ]

    # parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=len(segments)) as exe:
        futures = [exe.submit(process_segment, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Segments done"):
            results.append(f.result())

    # merge & sort
    merged = {cls: [] for cls in target_classes}
    inf_times = []
    for r in results:
        inf_times.append(r["avg_inference"])
        for cls in target_classes:
            merged[cls].extend(r["blocks_by_class"][cls])
    for cls in target_classes:
        merged[cls].sort(key=lambda b: b["start_frame"])

    # merge adjacent blocks across segments
    final = {cls: [] for cls in target_classes}
    for cls in target_classes:
        for blk in merged[cls]:
            if not final[cls]:
                final[cls].append(blk)
            else:
                prev = final[cls][-1]
                if blk["start_frame"] <= prev["end_frame"] + frame_skip:
                    # merge
                    prev["end_frame"] = max(prev["end_frame"], blk["end_frame"])
                    prev["last_crop"] = blk["last_crop"]
                else:
                    final[cls].append(blk)

    overall_inf = sum(inf_times)/len(inf_times) if inf_times else 0.0
    print(f"Overall avg inference: {overall_inf:.3f} s/frame")
    return final, fps


# Lista de classes disponíveis no COCO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Solicita ao usuário as classes desejadas
user_input = input("Enter COCO classes to detect (comma-separated, e.g., dog, person): ")

# Processa a entrada, remove espaços e mantém apenas classes válidas
raw_classes = [cls.strip().lower() for cls in user_input.split(',') if cls.strip()]
target_classes = [cls for cls in raw_classes if cls in COCO_CLASSES]

# Identifica classes inválidas
invalid_classes = [cls for cls in raw_classes if cls not in COCO_CLASSES]

# Alerta o usuário sobre entradas inválidas
if invalid_classes:
    print(f"Warning: The following classes are not valid COCO classes and will be ignored: {', '.join(invalid_classes)}")

# Se nenhuma classe válida foi inserida, encerra
if not target_classes:
    print("No valid COCO classes were entered. Exiting.")
    exit()

# Tweak if you like
frame_skip = 5
conf_threshold = 0.3

# Run
final_blocks, fps = adaptive_parallel_process(
    video_path=video_path,
    target_classes=target_classes,
    frame_skip=frame_skip,
    conf_threshold=conf_threshold,
    model_weights="yolov8x.pt"
)


for cls, blocks in final_blocks.items():
    print(f"\n=== Class: {cls} ===")
    if not blocks:
        print(" No blocks detected.")
    for idx, blk in enumerate(blocks):
        start_ts = frame_to_timestamp(blk["start_frame"], fps)
        end_ts   = frame_to_timestamp(blk["end_frame"], fps)
        print(f" Block {idx+1}: {start_ts} → {end_ts} (frames {blk['start_frame']}–{blk['end_frame']})")

        fig, axes = plt.subplots(1, 2, figsize=(6,3))
        axes[0].imshow(cv2.cvtColor(blk["first_crop"], cv2.COLOR_BGR2RGB))
        axes[0].set_title("First Appearance")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(blk["last_crop"], cv2.COLOR_BGR2RGB))
        axes[1].set_title("Last Appearance")
        axes[1].axis("off")

        plt.suptitle(f"{cls} • Block {idx+1}: {start_ts}→{end_ts}")
        plt.tight_layout()
        plt.show()


# Install necessary libraries (run once)
!pip install ultralytics opencv-python-headless matplotlib tqdm


# Upload your .mp4 video in Colab
from google.colab import files
uploaded = files.upload()

# Automatically pick the first uploaded file as video_path
import os
video_path = next(iter(uploaded))
print(f"Video uploaded: {video_path}")


import os
import cv2
import time
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def frame_to_timestamp(frame_idx: int, fps: float) -> str:
    """
    Convert a frame index to a "MM:SS.ss" timestamp.
    """
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def process_segment(args):
    """
    Run YOLOv8 detection on [start_frame, end_frame] of the video.
    Returns blocks_by_class dict and avg inference time.
    """
    video_path, start_frame, end_frame, target_classes, frame_skip, conf_threshold, model_weights = args

    # Load model inside each worker
    model = YOLO(model_weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    blocks_by_class = {cls: [] for cls in target_classes}
    in_block = {cls: False for cls in target_classes}
    block_start = {cls: None for cls in target_classes}
    first_crop = {cls: None for cls in target_classes}
    last_crop = {cls: None for cls in target_classes}

    frame_idx = start_frame
    infer_times = []

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_skip == 0:
            t0 = time.time()
            with torch.no_grad():
                results = model(frame)
            t1 = time.time()
            infer_times.append(t1 - t0)

            # collect per-class detections this frame
            dets = {cls: [] for cls in target_classes}
            for box in results[0].boxes:
                conf = float(box.conf)
                label = results[0].names[int(box.cls)]
                if conf < conf_threshold or label not in target_classes:
                    continue
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                x1,y1 = max(x1,0), max(y1,0)
                x2,y2 = min(x2,frame.shape[1]), min(y2,frame.shape[0])
                if x2<=x1 or y2<=y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                dets[label].append((crop, conf))

            # update blocks
            for cls in target_classes:
                if dets[cls]:
                    # best crop this frame
                    best_crop, _ = max(dets[cls], key=lambda x: x[1])
                    if not in_block[cls]:
                        in_block[cls] = True
                        block_start[cls] = frame_idx
                        first_crop[cls] = best_crop
                    last_crop[cls] = best_crop
                else:
                    if in_block[cls]:
                        blocks_by_class[cls].append({
                            "start_frame": block_start[cls],
                            "end_frame": frame_idx - 1,
                            "first_crop": first_crop[cls],
                            "last_crop": last_crop[cls],
                        })
                        in_block[cls] = False

        frame_idx += 1

    cap.release()
    # close any open blocks
    for cls in target_classes:
        if in_block[cls]:
            blocks_by_class[cls].append({
                "start_frame": block_start[cls],
                "end_frame": min(end_frame, frame_idx-1),
                "first_crop": first_crop[cls],
                "last_crop": last_crop[cls],
            })
    avg_inf = sum(infer_times)/len(infer_times) if infer_times else 0.0
    return {"blocks_by_class": blocks_by_class, "avg_inference": avg_inf}


def adaptive_parallel_process(video_path: str,
                              target_classes: list,
                              frame_skip: int = 5,
                              conf_threshold: float = 0.3,
                              model_weights: str = "yolov8x.pt"):
    # read metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {video_path} | FPS: {fps:.2f} | Frames: {total_frames}")

    # split into segments = CPU cores
    num_workers = os.cpu_count() or 1
    frames_per = math.ceil(total_frames / num_workers)
    segments = []
    for i in range(num_workers):
        start_f = i * frames_per
        end_f = min((i+1)*frames_per - 1, total_frames - 1)
        if start_f <= end_f:
            segments.append((start_f, end_f))
    print(f"Using {len(segments)} workers, ~{frames_per} frames each")

    # prepare arguments
    args_list = [
        (video_path, start, end, target_classes, frame_skip, conf_threshold, model_weights)
        for start, end in segments
    ]

    # parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=len(segments)) as exe:
        futures = [exe.submit(process_segment, args) for args in args_list]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Segments done"):
            results.append(f.result())

    # merge & sort
    merged = {cls: [] for cls in target_classes}
    inf_times = []
    for r in results:
        inf_times.append(r["avg_inference"])
        for cls in target_classes:
            merged[cls].extend(r["blocks_by_class"][cls])
    for cls in target_classes:
        merged[cls].sort(key=lambda b: b["start_frame"])

    # merge adjacent blocks across segments
    final = {cls: [] for cls in target_classes}
    for cls in target_classes:
        for blk in merged[cls]:
            if not final[cls]:
                final[cls].append(blk)
            else:
                prev = final[cls][-1]
                if blk["start_frame"] <= prev["end_frame"] + frame_skip:
                    # merge
                    prev["end_frame"] = max(prev["end_frame"], blk["end_frame"])
                    prev["last_crop"] = blk["last_crop"]
                else:
                    final[cls].append(blk)

    overall_inf = sum(inf_times)/len(inf_times) if inf_times else 0.0
    print(f"Overall avg inference: {overall_inf:.3f} s/frame")

    # Performance Metrics
    fps_metric = total_frames / (time.time() - sum(inf_times)) if inf_times else 0.0
    print(f"Frames Per Second (FPS): {fps_metric:.2f}")
    print(f"Average Inference Time: {overall_inf:.3f} seconds per frame")

    return final, fps


# Lista de classes disponíveis no COCO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Solicita ao usuário as classes desejadas
user_input = input("Enter COCO classes to detect (comma-separated, e.g., dog, person): ")

# Processa a entrada, remove espaços e mantém apenas classes válidas
raw_classes = [cls.strip().lower() for cls in user_input.split(',') if cls.strip()]
target_classes = [cls for cls in raw_classes if cls in COCO_CLASSES]

# Identifica classes inválidas
invalid_classes = [cls for cls in raw_classes if cls not in COCO_CLASSES]

# Alerta o usuário sobre entradas inválidas
if invalid_classes:
    print(f"Warning: The following classes are not valid COCO classes and will be ignored: {', '.join(invalid_classes)}")

# Se nenhuma classe válida foi inserida, encerra
if not target_classes:
    print("No valid COCO classes were entered. Exiting.")
    exit()

# Tweak if you like
frame_skip = 5
conf_threshold = 0.3

# Run
final_blocks, fps = adaptive_parallel_process(
    video_path=video_path,
    target_classes=target_classes,
    frame_skip=frame_skip,
    conf_threshold=conf_threshold,
    model_weights="yolov8x.pt"
)


for cls, blocks in final_blocks.items():
    print(f"\n=== Class: {cls} ===")
    if not blocks:
        print(" No blocks detected.")
    for idx, blk in enumerate(blocks):
        start_ts = frame_to_timestamp(blk["start_frame"], fps)
        end_ts   = frame_to_timestamp(blk["end_frame"], fps)
        print(f" Block {idx+1}: {start_ts} → {end_ts} (frames {blk['start_frame']}–{blk['end_frame']})")

        fig, axes = plt.subplots(1, 2, figsize=(6,3))
        axes[0].imshow(cv2.cvtColor(blk["first_crop"], cv2.COLOR_BGR2RGB))
        axes[0].set_title("First Appearance")
        axes[0].axis("off")

        axes[1].imshow(cv2.cvtColor(blk["last_crop"], cv2.COLOR_BGR2RGB))
        axes[1].set_title("Last Appearance")
        axes[1].axis("off")

        plt.suptitle(f"{cls} • Block {idx+1}: {start_ts}→{end_ts}")
        plt.tight_layout()
        plt.show()
