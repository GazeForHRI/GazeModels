from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

from ultralytics import YOLO  # YOLOv8 model import

weights = "./yolo_head/models/YOLOv8/yolov8n-face.pt"  # Updated to YOLOv8 model
imgsz = 640
device = torch.device("cuda:0")
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
augment = False

def detect(source):
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load YOLOv8 model
    model = YOLO(weights).to(device)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device))  # run once

    source = source.half() if half else source.float()  # uint8 to fp16/32
    source /= 255.0  # Normalize to [0,1]
    if source.ndimension() == 3:
        source = source.unsqueeze(0)

    # Inference
    results = model(source, conf=conf_thres, iou=iou_thres)
    pred = results[0].boxes.data  # Extract detection results

    # Apply class filtering (if necessary)
    if classes is not None:
        pred = pred[torch.isin(pred[:, 5], torch.tensor(classes, device=pred.device))]  # Keep only specified classes

    det = pred  # Maintain original structure
    return det


### yolov5 (crowdhumanyolov5m.pt) model:

# import argparse
# import time
# from pathlib import Path

# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from numpy import random

# import sys
# sys.path.append("yolo_head")

# from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages
# from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
#     scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
# from utils.torch_utils import select_device, load_classifier, time_synchronized

# weights = "crowdhuman_yolov5m.pt"
# imgsz = 640
# device = torch.device("cuda:0")
# conf_thres = 0.25
# iou_thres = 0.45
# classes = None
# agnostic_nms = False
# augment = False

# def detect(source):
#     #source, weights, imgsz = source, weights, img_size
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Load model
#     model = attempt_load(weights, map_location=device)  # load FP32 model
#     if half:
#         model.half()  # to FP16

#     # Second-stage classifier
#     classify = False
#     if classify:
#         modelc = load_classifier(name='resnet101', n=2)  # initialize
#         modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#     #source = torch.from_numpy(source).to(device)
#     source = source.half() if half else source.float()  # uint8 to fp16/32
#     source /= 255.0  # 0 - 255 to 0.0 - 1.0
#     if source.ndimension() == 3:
#         source = source.unsqueeze(0)

#     # Inference
#     #torch.Size([1, 3, 288, 640])
#     pred = model(source, augment=augment)[0]

#     # Apply NMS
#     pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    
#     # Apply Classifier
#     #if classify:
#     #    pred = apply_classifier(pred, modelc, img, im0s)
#     det = pred[0]
#     return det

### yolov8 (yolov8l-face.pt) model:
# import argparse
# import time
# from pathlib import Path

# import cv2
# import torch
# import torch.backends.cudnn as cudnn
# from numpy import random

# import os
# from ultralytics import YOLO  # YOLOv8 model import

# weights = "./yolo_head/models/YOLOv8/yolov8l-face.pt"  # Updated to YOLOv8 model
# imgsz = 640
# device = torch.device("cuda:0")
# conf_thres = 0.25
# iou_thres = 0.45
# classes = None
# agnostic_nms = False
# augment = False

# def detect(source):
#     half = device.type != 'cpu'  # half precision only supported on CUDA

#     # Load YOLOv8 model
#     model = YOLO(weights).to(device)

#     # Run inference
#     if device.type != 'cpu':
#         model(torch.zeros(1, 3, imgsz, imgsz).to(device))  # run once

#     source = source.half() if half else source.float()  # uint8 to fp16/32
#     source /= 255.0  # Normalize to [0,1]
#     if source.ndimension() == 3:
#         source = source.unsqueeze(0)

#     # Inference
#     results = model(source, conf=conf_thres, iou=iou_thres)
#     pred = results[0].boxes.data  # Extract detection results

#     # Apply class filtering (if necessary)
#     if classes is not None:
#         pred = pred[torch.isin(pred[:, 5], torch.tensor(classes, device=pred.device))]  # Keep only specified classes

#     det = pred  # Maintain original structure
#     return det
