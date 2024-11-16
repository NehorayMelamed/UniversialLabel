import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential  # For applying augmentation sequences to images
from kornia import augmentation as aug
from tqdm import tqdm
from ultralytics import YOLO, YOLOWorld
from PIL import Image
from torchvision.ops import nms

from sahi.detection_tta_sahi import (
    apply_augmentations, yolo_inference_callback,
    inverse_transformations, merge_keypoints_to_bbox, draw_bounding_boxes_on_image
)
from sahi.sahi import SahiSlider
from sahi.sahi_utils import adjust_bboxes, calculate_slice_bboxes, crop_tensor, scale_frame
from UL.ul_detection import ULDetection
from UL.ul_segmentation import ULSegmentation
from common.model_name_registry import ModelNameRegistryDetection, ModelNameRegistrySegmentation


if __name__ == "__main__":
    # Set device to GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the input image
    image_path = "/home/nehoray/PycharmProjects/UniversaLabeler/data/images/images/al_qurnah_14.tif"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]

    # Convert image to tensor format
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device).float()

    # --------------------------- ULDetection Example ---------------------------
    detectors = [ModelNameRegistryDetection.WALDO, ModelNameRegistryDetection.YOLO_ALFRED]
    detection_priorities = {ModelNameRegistryDetection.YOLO_ALFRED.name: 1, ModelNameRegistryDetection.WALDO.name: 2}
    ul_detection = ULDetection(detectors, detection_priorities)

    # Process image using ULDetection and apply NMS to combine results
    detection_results = ul_detection.process_image(image_tensor)
    print("Detection Results:", detection_results)

    # Apply Non-Maximum Suppression (NMS) on detection boxes (using torchvision's `nms` function)
    if len(detection_results) > 0:
        boxes = torch.tensor(detection_results[:, :4], dtype=torch.float32).to(device)
        scores = torch.tensor(detection_results[:, 5], dtype=torch.float32).to(device)
        keep_indices = nms(boxes, scores, iou_threshold=0.5)
        nms_results = detection_results[keep_indices.cpu().numpy()]
        print("Results after NMS:", nms_results)

    # --------------------------- ULSegmentation Example ---------------------------
    segmentors = [ModelNameRegistrySegmentation.OPEN_EARTH_MAP, ModelNameRegistrySegmentation.SAM2]
    segmentation_priorities = {}
    ul_segmentation = ULSegmentation(segmentors, segmentation_priorities)

    # Process image using ULSegmentation
    segmentation_result = ul_segmentation.process_image(image_tensor)
    print("Segmentation Result:", segmentation_result)

    # --------------------------- SahiSlider Example ---------------------------
    yolo_model = YOLOWorld()  # Load YOLO model using YOLOWorld

    # Instantiate SahiSlider for sliding window detection
    sahi_slider = SahiSlider(
        model=yolo_model,
        model_input_dimensions=(640, 640),
        slice_dimensions=(256, 256),
        detection_conf_threshold=0.5,
        transforms=[aug.RandomHorizontalFlip(p=0.5)]
    )

    # Set detection classes for SahiSlider
    sahi_slider.set_prompt(["car", "bus"])

    # Get bounding boxes using SahiSlider
    bbox_result = sahi_slider.get_bboxes(image)
    print("Bounding Boxes from SahiSlider:", bbox_result)

    # Save the detection image with bounding boxes to disk
    sahi_slider.save_detection_image_to_disk(image, save_path="sahi_slider_detection_result.jpg")
    print(f"Detection result saved to 'sahi_slider_detection_result.jpg'")




